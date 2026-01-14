#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
vLLM Inference Stage - Conversation-Batched Version

This version keeps all turns of a conversation together as a single AudioBatch,
enabling parallel processing and merging in streaming mode.

Key difference from vllm_inference.py:
- Returns list[AudioBatch] where each AudioBatch contains ALL turns of ONE conversation
- Enables MergeConversationStage to start merging as soon as one conversation completes
"""

import hashlib
import json
import os
import re
from string import Template
from typing import Any

import yaml
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks.audio_batch import AudioBatch


class vLLMInference(ProcessingStage[AudioBatch, AudioBatch]):
    """
    vLLM inference stage that keeps conversation turns together.
    
    Input: AudioBatch with multiple topics
    Output: list[AudioBatch] where each AudioBatch = 1 conversation with all its turns
    """
    
    name = "vLLMInference"
    
    def __init__(
        self,
        prompt: str = None,
        prompt_field: str = None,
        prompt_file: str = None,
        generation_field: str = 'generation',
        model: dict = {},
        inference: dict = {},
        apply_chat_template: dict = {},
        **kwargs
    ):
        super().__init__()
        
        if sum([prompt is not None, prompt_field is not None, prompt_file is not None]) != 1:
            raise ValueError("Exactly one of prompt, prompt_field, or prompt_file must be specified")
        
        self.prompt = prompt
        self.prompt_field = prompt_field
        self.prompt_file = prompt_file
        self.generation_field = generation_field
        self.model_params = model
        self.inference_params = inference
        self.chat_template_params = apply_chat_template
        
        if self.prompt_file:
            with open(self.prompt_file, 'r') as f:
                self.prompt_data = yaml.safe_load(f)
        else:
            self.prompt_data = None
        
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None

    def generate_conversation_id(self, turns: list[dict]) -> str:
        """Generate deterministic conversation ID from turns."""
        conversation_text = ''.join([
            f"{turn['speaker']}:{turn['utterance']}"
            for turn in turns
        ])
        hash_object = hashlib.sha256(conversation_text.encode())
        return hash_object.hexdigest()[:16]

    def validate_json_output(self, text: str) -> dict | None:
        """Validate and parse JSON output from LLM."""
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                return None
            
            parsed = json.loads(json_match.group())
            
            if 'turns' not in parsed or not isinstance(parsed['turns'], list):
                return None
            
            for turn in parsed['turns']:
                if not all(k in turn for k in ['speaker', 'utterance']):
                    return None
                # Validate non-empty speaker and utterance
                if not turn.get('speaker', '').strip():
                    return None
                if not turn.get('utterance', '').strip():
                    return None
                if not isinstance(turn.get('overlap', 0.0), (int, float)):
                    return None
            
            return parsed
        except (json.JSONDecodeError, AttributeError):
            return None

    def generate_batch_with_retry(self, llm, prompts: list[str], max_retry_rounds: int = 5):
        """Generate with retry logic for failed outputs."""
        from vllm import SamplingParams
        
        current_prompts = prompts.copy()
        current_indices = list(range(len(prompts)))
        validated_outputs = [None] * len(prompts)
        
        for retry_round in range(max_retry_rounds):
            if not current_prompts:
                break
            
            sampling_params = SamplingParams(**self.inference_params)
            outputs = llm.generate(current_prompts, sampling_params)
            
            next_prompts = []
            next_indices = []
            successful_count = 0
            
            for output, original_idx in zip(outputs, current_indices):
                generated_text = output.outputs[0].text
                validated = self.validate_json_output(generated_text)
                
                if validated:
                    validated_outputs[original_idx] = validated
                    successful_count += 1
                else:
                    next_prompts.append(current_prompts[current_indices.index(original_idx)])
                    next_indices.append(original_idx)
            
            if next_prompts:
                current_prompts = next_prompts
                current_indices = next_indices
            else:
                break
        
        return validated_outputs

    def get_entry_prompt(self, entry: dict) -> str:
        """Build prompt for a single entry."""
        if self.prompt:
            prompt = {role: content for role, content in [self.prompt.split(':', 1)]}
        elif self.prompt_field:
            if self.prompt_field not in entry:
                raise ValueError(f"Prompt field '{self.prompt_field}' not found in entry")
            prompt_text = entry[self.prompt_field]
            prompt = {role: content for role, content in [prompt_text.split(':', 1)]}
        elif self.prompt_file:
            if not self.prompt_data:
                raise ValueError("Prompt file was not loaded correctly")
            topic = entry.get('topic', '')
            prompt = {
                role: Template(content).safe_substitute(topic=topic)
                for role, content in self.prompt_data.items()
            }
        else:
            raise ValueError("No prompt source specified")
        
        entry_chat = []
        for role in ['system', 'user', 'assistant']:
            if role not in prompt:
                continue
            entry_chat.append(dict(
                role=role,
                content=prompt[role]
            ))

        entry_prompt = self.tokenizer.apply_chat_template(
            entry_chat,
            **self.chat_template_params
        )
        
        if isinstance(entry_prompt, list):
            entry_prompt = self.tokenizer.decode(entry_prompt, skip_special_tokens=False)

        return entry_prompt

    def setup(self, worker_metadata = None):
        """Initialize the LLM model once when the worker starts."""
        from vllm import LLM
        from transformers import AutoTokenizer
        
        self.llm = LLM(**self.model_params)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_params['model'],
            trust_remote_code=True
        )

    def process(self, entries: AudioBatch) -> list[AudioBatch]:
        """
        Process batch: generate conversations and return one AudioBatch per conversation.
        
        Input: AudioBatch with N topics
        Output: list[AudioBatch] where each AudioBatch contains all turns of one conversation
        """
        
        if not hasattr(self, 'llm') or self.llm is None:
            from vllm import LLM
            from transformers import AutoTokenizer
            self.llm = LLM(**self.model_params)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_params['model'],
                trust_remote_code=True
            )
        
        entry_prompts = []
        for entry in entries.data:
            entry_prompt = self.get_entry_prompt(entry)
            entry_prompts.append(entry_prompt)
        
        validated_outputs = self.generate_batch_with_retry(self.llm, entry_prompts, max_retry_rounds=5)
        
        output_batches = []
        successful_conversations = 0
        total_turns = 0
        
        for i, (data_entry, output_generation) in enumerate(zip(entries.data, validated_outputs)):
            if output_generation is not None:
                try:
                    conversation_id = self.generate_conversation_id(output_generation["turns"])
                    
                    conversation_turns = []
                    for turn in output_generation['turns']:
                        turn_entry = {
                            'conversation_id': conversation_id,
                            'speaker': turn['speaker'],
                            'utterance': turn['utterance'],
                            'overlap': turn.get('overlap', 0.0),
                            'topic': data_entry.get('topic', 'unknown'),
                        }
                        conversation_turns.append(turn_entry)
                        total_turns += 1
                    
                    conversation_batch = AudioBatch(
                        data=conversation_turns,
                        task_id=f"{entries.task_id}_conv_{conversation_id}",
                        dataset_name=entries.dataset_name
                    )
                    output_batches.append(conversation_batch)
                    successful_conversations += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process validated output {i+1}: {e}")
            else:
                logger.warning(f"Skipping failed generation {i+1}")

        return output_batches


class DocumentToAudioStage(ProcessingStage):
    """Stage to convert DocumentBatch to AudioBatch"""
    
    def process(self, task) -> list[AudioBatch]:
        from nemo_curator.tasks import DocumentBatch
        if isinstance(task, DocumentBatch):
            data = task.data.to_dict(orient='records')
        else:
            data = task.data if isinstance(task.data, list) else [task.data]
        
        return [
            AudioBatch(
                data=data,
                task_id=task.task_id,
                dataset_name=task.dataset_name,
            )
        ]

