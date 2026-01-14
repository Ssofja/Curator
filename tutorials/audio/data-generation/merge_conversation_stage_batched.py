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
Merge Conversation Stage - Conversation-Batched Version

This version processes conversation-batched AudioBatch objects where each
AudioBatch already contains all turns of a single conversation, enabling
parallel merging in streaming mode.

Key difference from merge_conversation_stage.py:
- Expects each AudioBatch to contain turns from ONE conversation only
- No grouping by conversation_id needed
- Can run in parallel (no need for num_workers=1)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks.audio_batch import AudioBatch


class MergeConversationStage(ProcessingStage):
    """
    Merge conversation turns into complete conversations with synchronized RTTMs.
    
    In conversation-batched mode:
    - Input: AudioBatch containing all turns of ONE conversation
    - Output: AudioBatch with ONE entry (the merged conversation)
    - Can run in parallel since each batch = one independent conversation
    """
    
    def __init__(
        self,
        output_audio_dir: str,
        output_rttm_dir: str,
        max_pause_duration: float = 2.0,
        randomize_pauses: bool = False,
    ):
        super().__init__()
        self.output_audio_dir = Path(output_audio_dir)
        self.output_rttm_dir = Path(output_rttm_dir)
        self.max_pause_duration = max_pause_duration
        self.randomize_pauses = randomize_pauses
        
        self.output_audio_dir.mkdir(parents=True, exist_ok=True)
        self.output_rttm_dir.mkdir(parents=True, exist_ok=True)
    
    def process(self, batch: AudioBatch) -> AudioBatch:
        """
        Process ONE conversation batch: merge all its turns.
        
        Input: AudioBatch with turns from one conversation
        Output: AudioBatch with one merged entry
        """
        entries = batch.data
        
        if not entries:
            return AudioBatch(data=[], task_id=batch.task_id, dataset_name=batch.dataset_name)
        
        conversation_id = entries[0].get('conversation_id', 'unknown')
        
        for entry in entries:
            if entry.get('conversation_id') != conversation_id:
                logger.warning(f"Mixed conversation IDs in batch! Expected {conversation_id}, got {entry.get('conversation_id')}")
        
        sorted_turns = sorted(entries, key=lambda x: entries.index(x))
        
        merged_entry = self._merge_conversation_turns(conversation_id, sorted_turns)
        
        if merged_entry:
            return AudioBatch(
                data=[merged_entry],
                task_id=batch.task_id,
                dataset_name=batch.dataset_name
            )
        else:
            logger.error(f"Failed to merge conversation {conversation_id[:8]}...")
            return AudioBatch(data=[], task_id=batch.task_id, dataset_name=batch.dataset_name)
    
    def _merge_conversation_turns(self, conversation_id: str, turns: List[Dict]) -> Dict | None:
        """
        Merge all turns of a conversation into a single audio+RTTM file.
        """
        try:
            merged_audio_segments = []
            merged_rttm_lines = []
            current_time = 0.0
            
            for turn_idx, turn in enumerate(turns):
                audio_filepath = turn.get('audio_filepath')
                rttm_filepath = turn.get('rttm_filepath')
                speaker = turn.get('speaker', 'unknown')
                overlap = float(turn.get('overlap', 0.0))
                
                if not audio_filepath or not os.path.exists(audio_filepath):
                    continue
                
                turn_audio, sample_rate = sf.read(audio_filepath)
                turn_duration = len(turn_audio) / sample_rate
                
                if overlap > 0:
                    overlap_seconds = min(overlap, current_time)
                    turn_start_time = current_time - overlap_seconds
                else:
                    turn_start_time = current_time
                
                if rttm_filepath and os.path.exists(rttm_filepath):
                    with open(rttm_filepath, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue
                            
                            parts = line.split()
                            if len(parts) >= 5 and parts[0] == 'SPEAKER':
                                rttm_start = float(parts[3])
                                rttm_duration = float(parts[4])
                                
                                adjusted_start = turn_start_time + rttm_start
                                
                                adjusted_line = (
                                    f"SPEAKER {conversation_id} 1 {adjusted_start:.3f} {rttm_duration:.3f} "
                                    f"<NA> <NA> {speaker} <NA> <NA>"
                                )
                                merged_rttm_lines.append(adjusted_line)
                else:
                    rttm_line = (
                        f"SPEAKER {conversation_id} 1 {turn_start_time:.3f} {turn_duration:.3f} "
                        f"<NA> <NA> {speaker} <NA> <NA>"
                    )
                    merged_rttm_lines.append(rttm_line)
                
                if overlap > 0 and merged_audio_segments:
                    overlap_samples = int(overlap_seconds * sample_rate)
                    
                    prev_audio = merged_audio_segments[-1]
                    
                    if overlap_samples < len(prev_audio):
                        non_overlap_part = prev_audio[:-overlap_samples]
                        overlap_part = prev_audio[-overlap_samples:]
                        
                        mixed_overlap = overlap_part[:len(turn_audio)] + turn_audio[:len(overlap_part)]
                        mixed_overlap = mixed_overlap * 0.7
                        
                        remaining_turn = turn_audio[len(overlap_part):]
                        merged_audio_segments[-1] = non_overlap_part
                        merged_audio_segments.append(mixed_overlap)
                        if len(remaining_turn) > 0:
                            merged_audio_segments.append(remaining_turn)
                    else:
                        merged_audio_segments.append(turn_audio)
                else:
                    merged_audio_segments.append(turn_audio)
                
                if turn_idx < len(turns) - 1:
                    if self.randomize_pauses:
                        pause_duration = np.random.uniform(0.3, self.max_pause_duration)
                    else:
                        pause_duration = self.max_pause_duration * 0.5
                    
                    next_overlap = float(turns[turn_idx + 1].get('overlap', 0.0))
                    if next_overlap <= 0:
                        pause_samples = int(pause_duration * sample_rate)
                        silence = np.zeros(pause_samples)
                        merged_audio_segments.append(silence)
                        current_time += pause_duration
                
                current_time = turn_start_time + turn_duration
            
            if not merged_audio_segments:
                logger.error(f"No audio segments to merge for conversation {conversation_id[:8]}...")
                return None
            
            merged_audio = np.concatenate(merged_audio_segments)
            merged_duration = len(merged_audio) / sample_rate
            
            output_audio_path = self.output_audio_dir / f"{conversation_id}.wav"
            sf.write(output_audio_path, merged_audio, sample_rate)
            
            output_rttm_path = self.output_rttm_dir / f"{conversation_id}.rttm"
            with open(output_rttm_path, 'w') as f:
                f.write('\n'.join(merged_rttm_lines) + '\n')
            
            merged_entry = {
                'audio_filepath': str(output_audio_path),
                'rttm_filepath': str(output_rttm_path),
                'duration': merged_duration,
                'conversation_id': conversation_id,
                'num_turns': len(turns),
                'speakers': list(set(turn.get('speaker', 'unknown') for turn in turns)),
                'topic': turns[0].get('topic', 'unknown'),
            }
            
            return merged_entry
            
        except Exception as e:
            logger.error(f"Error merging conversation {conversation_id[:8]}...: {e}")
            import traceback
            traceback.print_exc()
            return None

