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
Audio Data Generation Workflow

End-to-end workflow for generating audio conversations from topics using:
1. Topic Expansion
2. vLLM Conversation Generation (conversation-batched)
3. TTS Audio Generation  
4. MFA Alignment
5. Conversation Merging

This workflow pattern enables proper RayClient integration by creating executors
internally, allowing Ray to be started externally via RayClient without conflicts.
"""

import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from nemo_curator.backends.base import BaseExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage

# Import custom stages (they must be in the same directory or in sys.path)
from vllm_inference_conversation_batched import vLLMInference, DocumentToAudioStage
from tts_generation import ChatterboxTTSGeneration
from mfa_rttm_generation import MFAToRTTMGeneration
from merge_conversation_stage_batched import MergeConversationStage
from topic_expander import TopicExpander


@dataclass
class AudioDataGenerationWorkflow:
    """
    End-to-end workflow for audio data generation from topics.
    
    This workflow combines all stages into a single pipeline that can be run
    with RayClient for explicit resource control, or standalone using environment
    variables for Ray configuration.
    
    Stages:
    1. Topic Expansion: Expand topics into N conversation prompts
    2. vLLM Conversation Generation: Generate conversations (conversation-batched)
    3. TTS Audio Generation: Convert text to speech
    4. MFA Alignment: Generate RTTM timing information
    5. Conversation Merging: Merge turns into complete conversations
    """
    
    # Input/Output paths
    input_manifest: str
    output_dir: str
    reference_voices: str
    
    # Generation parameters
    num_conversations: int = 100
    batch_size: int = 10
    random_seed: int | None = None
    
    # LLM parameters
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct-1M"
    max_model_len: int = 8192
    max_tokens: int = 4096
    llm_temperature: float = 0.8
    prompt_file: str | None = None
    
    # TTS parameters
    tts_device: str = "cuda"
    cfg_weight: float = 0.5
    exaggeration: float = 0.5
    tts_temperature: float = 0.8
    normalize_audio: bool = True
    normalize_level: float = -20.0
    min_pause: float = 0.3
    max_pause: float = 1.5
    
    # MFA parameters
    mfa_command: str = "mfa"
    mfa_acoustic_model: str = "english_us_arpa"
    mfa_dictionary: str = "english_us_arpa"
    use_phone_intervals: bool = False
    max_gap_for_merge: float = 0.3
    
    # Merge parameters
    max_pause_duration: float = 2.0
    randomize_pauses: bool = False
    
    # Stage control flags
    skip_llm: bool = False
    skip_tts: bool = False
    skip_mfa: bool = False
    skip_merge: bool = False
    
    # Execution parameters
    verbose: bool = False
    
    # Internal paths (set in __post_init__)
    conversations_dir: str = field(init=False)
    audio_dir: str = field(init=False)
    audio_manifest_dir: str = field(init=False)
    rttm_dir: str = field(init=False)
    final_manifest_dir: str = field(init=False)
    merged_audio_dir: str = field(init=False)
    merged_rttm_dir: str = field(init=False)
    merged_manifest_dir: str = field(init=False)
    
    def __post_init__(self):
        """Setup output directories."""
        self.conversations_dir = os.path.join(self.output_dir, "conversations")
        self.audio_dir = os.path.join(self.output_dir, "audio")
        self.audio_manifest_dir = os.path.join(self.output_dir, "audio_manifest")
        self.rttm_dir = os.path.join(self.output_dir, "rttms")
        self.final_manifest_dir = os.path.join(self.output_dir, "manifest")
        self.merged_audio_dir = os.path.join(self.output_dir, "merged_audio")
        self.merged_rttm_dir = os.path.join(self.output_dir, "merged_rttms")
        self.merged_manifest_dir = os.path.join(self.output_dir, "merged_manifest")
        
        os.makedirs(self.conversations_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.audio_manifest_dir, exist_ok=True)
        os.makedirs(self.rttm_dir, exist_ok=True)
        os.makedirs(self.final_manifest_dir, exist_ok=True)
        os.makedirs(self.merged_audio_dir, exist_ok=True)
        os.makedirs(self.merged_rttm_dir, exist_ok=True)
        os.makedirs(self.merged_manifest_dir, exist_ok=True)
        
        if not os.path.exists(self.input_manifest):
            raise FileNotFoundError(f"Input manifest not found: {self.input_manifest}")
        
        if not self.skip_tts and not os.path.exists(self.reference_voices):
            raise FileNotFoundError(f"Reference voices dataset not found: {self.reference_voices}")
    
    def _log_configuration(self):
        """Log workflow configuration."""
        logger.info("="*80)
        logger.info("AUDIO DATA GENERATION WORKFLOW")
        logger.info("="*80)
        logger.info(f"Input: {self.input_manifest}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Conversations: {self.num_conversations}")
        logger.info("="*80)
    
    def _log_summary(self):
        """Log final summary of workflow execution."""
        logger.info("="*80)
        logger.info("✅ WORKFLOW COMPLETE")
        logger.info("="*80)
        
        if not self.skip_llm:
            total_turns = 0
            conversation_ids = set()
            output_files = glob.glob(os.path.join(self.conversations_dir, "*.jsonl"))
            for output_file in output_files:
                with open(output_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            total_turns += 1
                            try:
                                entry = json.loads(line)
                                conversation_ids.add(entry.get('conversation_id', ''))
                            except:
                                pass
            
            logger.info(f"Conversations: {len(conversation_ids)}, Total turns: {total_turns}")
        
        if not self.skip_merge:
            merged_files = list(Path(self.merged_audio_dir).glob("*.wav"))
            logger.info(f"Merged Conversations: {len(merged_files)}")
        
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*80)
    
    def run(self, executor: BaseExecutor | None = None):
        """
        Run the complete workflow.
        
        Args:
            executor: Executor to use. If None, creates XennaExecutor.
        
        Returns:
            list: Final output tasks from the pipeline
        """
        if executor is None:
            from nemo_curator.backends.xenna import XennaExecutor
            executor = XennaExecutor(config={
                "execution_mode": "streaming",
                "autoscale_interval_s": 60,
            })
        
        if self.verbose:
            self._log_configuration()
        
        try:
            pipeline = Pipeline(name="audio_data_generation")
            
            if not self.skip_llm:
                pipeline.add_stage(JsonlReader(file_paths=self.input_manifest))
                pipeline.add_stage(DocumentToAudioStage())
                pipeline.add_stage(TopicExpander(
                    num_conversations=self.num_conversations,
                    conversations_per_batch=self.batch_size,
                    seed=self.random_seed,
                ))
                
                pipeline.add_stage(
                    vLLMInference(
                        generation_field="src_text",
                        prompt_file=self.prompt_file,
                        model={
                            "model": self.llm_model,
                            "tensor_parallel_size": 1,
                            "max_model_len": self.max_model_len,
                            "enable_chunked_prefill": False,
                            "enforce_eager": True,
                            "dtype": "float16",
                            "gpu_memory_utilization": 0.8,
                            "max_num_seqs": 8,
                        },
                        inference={
                            "max_tokens": self.max_tokens,
                            "temperature": self.llm_temperature,
                            "top_p": 0.95,
                        },
                        apply_chat_template={
                            "tokenize": False,
                            "add_generation_prompt": True
                        },
                    ).with_(
                        batch_size=self.batch_size,
                        resources=Resources(gpus=1)
                    )
                )
            
            if not self.skip_tts:
                pipeline.add_stage(
                    ChatterboxTTSGeneration(
                        output_audio_dir=self.audio_dir,
                        reference_voices_dataset=self.reference_voices,
                        device=self.tts_device,
                        cfg_weight=self.cfg_weight,
                        exaggeration=self.exaggeration,
                        temperature=self.tts_temperature,
                        normalize_audio=self.normalize_audio,
                        normalize_level=self.normalize_level,
                    ).with_(
                        batch_size=1,
                        resources=Resources(gpus=1)
                    )
                )
            elif not self.skip_llm:
                raise ValueError("Cannot skip TTS if LLM is enabled. Use --skip-llm to start from existing conversations.")
            
            if not self.skip_mfa:
                pipeline.add_stage(
                    MFAToRTTMGeneration(
                        rttm_output_dir=self.rttm_dir,
                        mfa_command=self.mfa_command,
                        acoustic_model=self.mfa_acoustic_model,
                        dictionary=self.mfa_dictionary,
                        use_phone_intervals=self.use_phone_intervals,
                        max_gap_for_merge=self.max_gap_for_merge,
                        text_field="utterance",
                    )
                )
            elif not self.skip_tts:
                raise ValueError("Cannot skip MFA if TTS is enabled. Use --skip-tts to start from existing audio.")
            
            if not self.skip_merge:
                pipeline.add_stage(
                    MergeConversationStage(
                        output_audio_dir=self.merged_audio_dir,
                        output_rttm_dir=self.merged_rttm_dir,
                        max_pause_duration=self.max_pause_duration,
                        randomize_pauses=self.randomize_pauses,
                    ).with_(
                        batch_size=1,
                    )
                )
                pipeline.add_stage(AudioToDocumentStage().with_(batch_size=1))
                pipeline.add_stage(JsonlWriter(path=self.merged_manifest_dir, write_kwargs={"force_ascii": False}))
            
            results = pipeline.run(executor)
            
            if self.verbose:
                self._log_summary()
            
            return results
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise

