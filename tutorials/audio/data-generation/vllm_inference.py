# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import yaml
import json
import re
import hashlib
import jsonschema
from loguru import logger
from nemo_curator.stages.base import ProcessingStage

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioBatch, DocumentBatch

class vLLMInference(ProcessingStage[AudioBatch, AudioBatch]):
    """
    A processor that performs inference using a vLLM model on entries from an input manifest.

    This class supports three prompt configuration modes:
    - a static prompt template (`prompt`)
    - a field in each entry containing the prompt (`prompt_field`)
    - a YAML file containing the prompt structure (`prompt_file`)

    The prompts are converted into chat-style input using a tokenizer chat template,
    passed to the vLLM engine for generation, and the results are written to an output manifest.

    Args:
        prompt (str, optional): A fixed prompt used for all entries.
        prompt_field (str, optional): The key in each entry that holds the prompt template.
        prompt_file (str, optional): Path to a YAML file containing the prompt structure.
        generation_field (str): Name of the output field to store generated text. Default is 'generation'.
        model (dict): Parameters to initialize the vLLM model.
        inference (dict): Sampling parameters passed to vLLM.SamplingParams.
        apply_chat_template (dict): Arguments passed to the tokenizer's `apply_chat_template` method.
        **kwargs: Passed to the BaseProcessor (includes `input_manifest_file` and `output_manifest_file`).

    Raises:
        ValueError: If zero or more than one prompt configuration methods are used simultaneously.

    Returns:
        A line-delimited JSON manifest where each entry includes the original fields
        plus a field with the generated output.

    .. note::
        For detailed parameter options, refer to the following documentation:

        - model: https://docs.vllm.ai/en/latest/api/vllm/index.html#vllm.LLM
        - inference: https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html
        - apply_chat_template: https://huggingface.co/docs/transformers/main/en/chat_templating#applychattemplate

        Make sure to install `optree>=0.13.0` and `vllm` before using this processor:
            pip install "optree>=0.13.0" vllm

    """
    num_workers = None
    batch_size = 1
    name = "vLLMInference"  # Stage identifier for pipeline configuration
    
    CONVERSATION_SCHEMA = {
        "type": "object",
        "properties": {
            "turns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string"},
                        "overlap": {"type": "number"},
                        "utterance": {"type": "string"}
                    },
                    "required": ["speaker", "overlap", "utterance"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["turns"],
        "additionalProperties": False
    }

    def __init__(self,
                 prompt: str = None,
                 prompt_field: str = None,
                 prompt_file: str = None,
                 generation_field: str = 'generation',
                 model: dict = {},
                 inference: dict = {},
                 apply_chat_template: dict = {},
                 **kwargs):

        from vllm import SamplingParams
        from transformers import AutoTokenizer

        super().__init__(**kwargs)
    
        self.prompt = prompt
        self.prompt_field = prompt_field
        self.generation_field = generation_field

        # Ensure that exactly one prompt method is used
        prompt_args_counter = sum([prompt is not None, prompt_field is not None, prompt_file is not None])
        if prompt_args_counter < 1:
            raise ValueError(f'One of `prompt`, `prompt_field` or `prompt_file` should be provided.')
        elif prompt_args_counter > 1:
            err = []
            if prompt:
                err.append(f'`prompt` ({prompt})')
            if prompt_field:
                err.append(f'`prompt_field` ({prompt_field})')
            if prompt_file:
                err.append(f'`prompt_file` ({prompt_file})')
            raise ValueError(f'Found more than one prompt values: {", ".join(err)}.')

        self.prompt_file = prompt_file

        self.model_params = model
        self.sampling_params = SamplingParams(**inference)
        self.chat_template_params = apply_chat_template

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_params['model'])

    def _read_prompt_file(self, topic):
        with open(self.prompt_file, 'r') as f:
            text=f.read()
        filled = text.replace('<CATEGORY>', topic)
        return yaml.safe_load(filled)

    def generate_conversation_id(self, turns):
        """Generate a conversation id based on the turns by hashing the result"""
        return hashlib.sha256(json.dumps(turns).encode()).hexdigest()
    
    def validate_and_parse_json(self, text_output):
        """
        Validate JSON output against the conversation schema.
        Handles LLM outputs that wrap JSON in markdown code blocks or add extra text.
        """
        # Clean up the output
        text = text_output.strip()
        
        # Try to extract JSON from markdown code blocks
        markdown_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        markdown_match = re.search(markdown_pattern, text, re.DOTALL)
        
        if markdown_match:
            text = markdown_match.group(1).strip()
        
        # Try to extract JSON object/array from text with extra content
        json_pattern = r'(\{.*\}|\[.*\])'
        json_match = re.search(json_pattern, text, re.DOTALL)
        
        if json_match and not markdown_match:
            text = json_match.group(1).strip()
        
        try:
            # Try to parse JSON
            parsed_json = json.loads(text)
            
            # Validate against schema
            jsonschema.validate(parsed_json, self.CONVERSATION_SCHEMA)
            
            return parsed_json
            
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decode error: {e}\nRaw output (first 500 chars): {text_output[:500]}")
                
        except jsonschema.ValidationError as e:
            raise jsonschema.ValidationError(f"Schema validation error: {e.message}")
    
    def validate_batch_outputs(self, outputs):
        """Validate a batch of outputs and identify successful and failed ones."""
        successful_results = []
        failed_indices = []
        
        for i, output in enumerate(outputs):
            try:
                text_output = output.outputs[0].text
                validated_json = self.validate_and_parse_json(text_output)
                successful_results.append(validated_json)
                logger.debug(f"✓ Output {i+1}: Validation successful")
                
            except (ValueError, jsonschema.ValidationError) as e:
                successful_results.append(None)
                failed_indices.append(i)
                logger.warning(f"✗ Output {i+1}: Validation failed - {str(e)[:100]}")
                logger.debug(f"  Raw output: {text_output[:300]}...")
        
        return successful_results, failed_indices
    
    def generate_batch_with_retry(self, llm, prompts, max_retry_rounds=5):
        """Generate outputs for all prompts with batch retry logic."""
        logger.info(f"Starting batch generation for {len(prompts)} prompts...")
        
        pending_prompts = prompts.copy()
        pending_indices = list(range(len(prompts)))
        final_results = [None] * len(prompts)
        
        for round_num in range(max_retry_rounds):
            if not pending_prompts:
                break
                
            logger.info(f"=== Retry Round {round_num + 1}/{max_retry_rounds} ===")
            logger.info(f"Processing {len(pending_prompts)} prompts...")
            
            # Generate batch outputs
            outputs = llm.generate(pending_prompts, self.sampling_params)
            
            # Validate outputs and identify failures
            results, failed_local_indices = self.validate_batch_outputs(outputs)
            
            # Update final results
            new_pending_prompts = []
            new_pending_indices = []
            
            for local_idx, original_idx in enumerate(pending_indices):
                if local_idx not in failed_local_indices:
                    final_results[original_idx] = results[local_idx]
                else:
                    new_pending_prompts.append(pending_prompts[local_idx])
                    new_pending_indices.append(original_idx)
            
            pending_prompts = new_pending_prompts
            pending_indices = new_pending_indices
            
            success_count = len(prompts) - len(pending_prompts)
            logger.info(f"Round {round_num + 1} complete: {success_count}/{len(prompts)} successful")
            
            if not pending_prompts:
                logger.info("🎉 All prompts successfully validated!")
                break
        
        if pending_prompts:
            logger.warning(f"⚠️  {len(pending_prompts)} prompts failed after {max_retry_rounds} rounds")
        
        return final_results

    def get_entry_prompt(self, data_entry):
        """Format the prompt for a single data entry using the chat template."""
        entry_chat = []
        if self.prompt_field:
            prompt = data_entry[self.prompt_field]
        elif self.prompt_file:
            topic = data_entry.get('topic', 'general conversation')
            prompt = self._read_prompt_file(topic)
        else:
            prompt = self.prompt

        for role in prompt:
            entry_chat.append(dict(
                role=role,
                content=prompt[role]
            ))

        entry_prompt = self.tokenizer.apply_chat_template(
            entry_chat,
            **self.chat_template_params
        )
        
        # If tokenizer returns token IDs (list), convert to string
        if isinstance(entry_prompt, list):
            entry_prompt = self.tokenizer.decode(entry_prompt, skip_special_tokens=False)

        return entry_prompt

    def setup(self, worker_metadata = None):
        """Initialize the LLM model once when the worker starts.
        worker_metadata is not used in this stage, but is required by the base class.
        """
        from vllm import LLM
        logger.info("Initializing vLLM model...")
        self.llm = LLM(**self.model_params)
        logger.info("vLLM model initialized and ready!")

    def process(self, entries: AudioBatch) -> AudioBatch:
        """Main processing function: reads entries, builds prompts, runs generation, writes results."""
        
        # Use the pre-initialized LLM instance
        if not hasattr(self, 'llm'):
            # Fallback: initialize if not already done (shouldn't happen in normal flow)
            from vllm import LLM
            logger.warning("LLM not initialized in setup(), initializing now...")
            self.llm = LLM(**self.model_params)
        
        # Generate prompts for each entry (each entry has a topic)
        entry_prompts = []
        for entry in entries.data:
            entry_prompt = self.get_entry_prompt(entry)
            entry_prompts.append(entry_prompt)
        
        # Batch generate with validation and retry
        logger.info(f"Generating {len(entry_prompts)} conversations with batch retry logic...")
        validated_outputs = self.generate_batch_with_retry(self.llm, entry_prompts, max_retry_rounds=5)
        
        # Process successful generations - EXPAND EACH CONVERSATION INTO INDIVIDUAL TURNS
        # This matches the SDP approach where each turn is a separate manifest entry
        data_entries = []
        successful_conversations = 0
        total_turns = 0
        
        for i, (data_entry, output_generation) in enumerate(zip(entries.data, validated_outputs)):
            if output_generation is not None:
                try:
                    # Generate conversation ID from turns (like SDP does)
                    conversation_id = self.generate_conversation_id(output_generation["turns"])
                    
                    # Create one manifest entry per turn (matching SDP structure)
                    for turn in output_generation['turns']:
                        turn_entry = {
                            'conversation_id': conversation_id,
                            'speaker': turn['speaker'],
                            'utterance': turn['utterance'],
                            'overlap': turn.get('overlap', 0.0),
                            'topic': data_entry.get('topic', 'unknown'),
                        }
                        data_entries.append(turn_entry)
                        total_turns += 1
                    
                    successful_conversations += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process validated output {i+1}: {e}")
            else:
                logger.warning(f"Skipping failed generation {i+1}")
        
        logger.info(f"📊 Final Results:")
        logger.info(f"   Conversations: {successful_conversations}/{len(entries.data)} generated successfully")
        logger.info(f"   Total turns: {total_turns}")
        logger.info(f"   Avg turns/conversation: {total_turns/successful_conversations if successful_conversations > 0 else 0:.1f}")

        return AudioBatch(data=data_entries,
                task_id=entries.task_id,
                dataset_name=entries.dataset_name)



class DocumentToAudioStage(ProcessingStage[DocumentBatch, AudioBatch]):
    """
    Stage to conver DocumentBatch to AudioBatch

    """

    def process(self, task: DocumentBatch) -> list[AudioBatch]:
        return [
            AudioBatch(
                data=task.data.to_dict(orient='records'),
                task_id=task.task_id,
                dataset_name=task.dataset_name,
            )
        ]
