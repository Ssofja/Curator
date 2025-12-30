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
Merge Conversation Stage for Curator

This stage merges individual turn audio files into complete conversations,
with synchronized RTTM files. It handles overlaps and pauses correctly.

Based on NeMo-SDP's merge_conversation_fixed_rttm.py
"""

import json
import os
from collections import defaultdict
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
    
    This stage:
    1. Groups entries by conversation_id
    2. Merges audio files (handling overlaps/pauses)
    3. Merges RTTM files (synchronized with audio)
    4. Outputs one entry per conversation
    
    Must run sequentially (1 worker) to maintain conversation grouping.
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
        
        logger.info(f"MergeConversationStage initialized")
        logger.info(f"  Output audio: {self.output_audio_dir}")
        logger.info(f"  Output RTTM: {self.output_rttm_dir}")
        logger.info(f"  Max pause: {self.max_pause_duration}s")
        logger.info(f"  Randomize pauses: {self.randomize_pauses}")
    
    def xenna_stage_spec(self) -> dict[str, Any]:
        """Configure for sequential execution (1 worker)."""
        return {
            "num_workers": 1,
        }
    
    def process(self, batch: AudioBatch) -> AudioBatch:
        """Process batch: group by conversation and merge."""
        entries = []
        
        # Check if batch.data contains manifest filepaths or direct entries
        for item in batch.data:
            if isinstance(item, str) and os.path.isfile(item):
                logger.info(f"Reading manifest file: {item}")
                with open(item, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            manifest_entry = json.loads(line)
                            entries.append(manifest_entry)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON line in {item}: {e}")
                            continue
            else:
                entries.append(item)
        
        logger.info(f"Processing batch of {len(entries)} entries")
        
        # Group by conversation_id
        conversations = self._group_by_conversation(entries)
        
        logger.info(f"Found {len(conversations)} conversations to merge")
        
        # Merge each conversation
        merged_entries = []
        for conv_id, turns in conversations.items():
            try:
                merged_entry = self._merge_conversation(conv_id, turns)
                if merged_entry:
                    merged_entries.append(merged_entry)
            except Exception as e:
                logger.error(f"Error merging conversation {conv_id[:8]}: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"✅ Merged {len(merged_entries)} conversations")
        
        return AudioBatch(data=merged_entries)
    
    def _group_by_conversation(self, entries: List[Dict]) -> Dict[str, List[Dict]]:
        """Group entries by conversation_id."""
        conversations = defaultdict(list)
        
        for entry in entries:
            conv_id = entry.get('conversation_id', 'unknown')
            conversations[conv_id].append(entry)
        
        return dict(conversations)
    
    def _merge_conversation(self, conv_id: str, turns: List[Dict]) -> Dict:
        """Merge all turns for a conversation."""
        logger.info(f"📦 Merging conversation {conv_id[:8]}... ({len(turns)} turns)")
        
        # Sort by order (or just use as-is if already ordered)
        # turns = sorted(turns, key=lambda x: x.get('turn_index', 0))
        
        # Collect metadata
        speakers = set()
        merged_utterance = []
        topic = turns[0].get('topic', '') if turns else ''
        
        for turn in turns:
            speaker = turn.get('speaker', '')
            utterance = turn.get('utterance', turn.get('text', ''))
            
            if speaker:
                speakers.add(speaker)
            if utterance:
                merged_utterance.append(utterance)
        
        # Create output paths
        conv_id_short = conv_id[:12] if len(conv_id) > 12 else conv_id
        merged_audio_path = self.output_audio_dir / f"{conv_id_short}_conversation.wav"
        merged_rttm_path = self.output_rttm_dir / f"{conv_id_short}_conversation.rttm"
        
        # Merge audio files (returns actual overlaps used)
        actual_overlaps, sample_rate = self._merge_audio_files(turns, merged_audio_path)
        
        if actual_overlaps is None:
            logger.error(f"  ❌ Failed to merge audio for conversation {conv_id[:8]}")
            return None
        
        # Merge RTTM files with synchronized timing
        self._merge_rttm_files(turns, merged_rttm_path, actual_overlaps, conv_id_short)
        
        # Calculate total duration
        if os.path.exists(merged_audio_path):
            audio_info = sf.info(merged_audio_path)
            total_duration = audio_info.duration
        else:
            total_duration = sum(turn.get('duration', 0) for turn in turns)
        
        # Create merged entry
        merged_entry = {
            "conversation_id": conv_id,
            "audio_filepath": str(merged_audio_path),
            "rttm_filepath": str(merged_rttm_path),
            "duration": total_duration,
            "num_speakers": len(speakers),
            "num_turns": len(turns),
            "topic": topic,
            "utterance": " ".join(merged_utterance),
        }
        
        logger.info(f"  ✅ Merged: {merged_audio_path.name} ({total_duration:.2f}s)")
        
        return merged_entry
    
    def _extract_speaking_segments(self, audio_filepath: str, rttm_filepath: str):
        """Extract speaking segments from audio based on RTTM, keeping pauses <= 1s."""
        timestamps = self._get_rttm_timestamps(rttm_filepath)
        
        if not timestamps:
            logger.warning(f"No segments in RTTM: {rttm_filepath}")
            return None, None
        
        try:
            audio_data, sr = sf.read(audio_filepath)
        except Exception as e:
            logger.error(f"Error reading audio {audio_filepath}: {e}")
            return None, None
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        timestamps = sorted(timestamps, key=lambda x: x[0])
        
        speaking_segments = []
        max_pause_duration = 1.0  # Keep pauses <= 1 second
        
        for i, (start_time, duration) in enumerate(timestamps):
            start_idx = int(start_time * sr)
            end_idx = int((start_time + duration) * sr)
            
            start_idx = max(0, start_idx)
            end_idx = min(len(audio_data), end_idx)
            
            if start_idx < end_idx:
                segment = audio_data[start_idx:end_idx]
                speaking_segments.append(segment)
                
                # Check pause to next segment
                if i < len(timestamps) - 1:
                    next_start = timestamps[i + 1][0]
                    current_end = start_time + duration
                    pause_duration = next_start - current_end
                    
                    # Include pauses <= 1 second
                    if 0 < pause_duration <= max_pause_duration:
                        pause_start_idx = end_idx
                        pause_end_idx = int(next_start * sr)
                        pause_end_idx = min(len(audio_data), pause_end_idx)
                        
                        if pause_start_idx < pause_end_idx:
                            pause_segment = audio_data[pause_start_idx:pause_end_idx]
                            speaking_segments.append(pause_segment)
        
        if not speaking_segments:
            return None, None
        
        concatenated_audio = np.concatenate(speaking_segments)
        
        return concatenated_audio, sr
    
    def _get_rttm_timestamps(self, rttm_filepath: str) -> List[tuple]:
        """Extract (start, duration) timestamps from RTTM file."""
        timestamps = []
        
        if not os.path.exists(rttm_filepath):
            return timestamps
        
        try:
            with open(rttm_filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith('SPEAKER'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            start = float(parts[3])
                            duration = float(parts[4])
                            timestamps.append((start, duration))
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            logger.error(f"Error reading RTTM {rttm_filepath}: {e}")
        
        return timestamps
    
    def _merge_audio_files(self, turns: List[Dict], output_path: Path) -> tuple:
        """Merge audio files with overlaps/pauses. Returns (actual_overlaps, sample_rate)."""
        if not turns:
            return None, None
        
        merged_audio = None
        current_position = 0
        sample_rate = None
        actual_overlaps = []
        
        for i, turn in enumerate(turns):
            try:
                audio_file = turn.get('audio_filepath')
                rttm_file = turn.get('rttm_filepath', '')
                
                if not audio_file or not os.path.exists(audio_file):
                    logger.warning(f"Audio file not found: {audio_file}")
                    actual_overlaps.append(turn.get('overlap', 0))
                    continue
                
                # Extract speaking segments based on RTTM
                if rttm_file and os.path.exists(rttm_file):
                    audio_data, sr = self._extract_speaking_segments(audio_file, rttm_file)
                    if audio_data is None:
                        audio_data, sr = sf.read(audio_file)
                        if len(audio_data.shape) > 1:
                            audio_data = audio_data.mean(axis=1)
                else:
                    audio_data, sr = sf.read(audio_file)
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.mean(axis=1)
                
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    logger.warning(f"Sample rate mismatch: {sr} vs {sample_rate}")
                
                overlap = turn.get('overlap', 0)
                
                if merged_audio is None:
                    # First turn
                    merged_audio = audio_data.copy()
                    current_position = len(audio_data)
                    actual_overlaps.append(0.0)
                else:
                    # Handle overlap/pause
                    if overlap < 0:
                        # Negative = pause
                        pause_duration = abs(overlap)
                        
                        # Cap pause at maximum
                        if pause_duration > self.max_pause_duration:
                            pause_duration = self.max_pause_duration
                        
                        # Optionally randomize
                        if self.randomize_pauses and pause_duration > 0.5:
                            import random
                            pause_duration = random.uniform(0.3, min(pause_duration, self.max_pause_duration))
                        
                        pause_samples = int(pause_duration * sample_rate)
                        actual_overlaps.append(-pause_duration)
                        
                        # Add silence
                        start_position = current_position + pause_samples
                        end_position = start_position + len(audio_data)
                        
                        if end_position > len(merged_audio):
                            extended = np.zeros(end_position, dtype=merged_audio.dtype)
                            extended[:len(merged_audio)] = merged_audio
                            merged_audio = extended
                        
                        merged_audio[start_position:end_position] = audio_data
                        current_position = end_position
                        
                    elif overlap > 0:
                        # Positive = overlap/interruption
                        overlap_samples = int(overlap * sample_rate)
                        start_position = max(0, current_position - overlap_samples)
                        end_position = start_position + len(audio_data)
                        
                        actual_overlaps.append(overlap)
                        
                        if end_position > len(merged_audio):
                            extended = np.zeros(end_position, dtype=merged_audio.dtype)
                            extended[:len(merged_audio)] = merged_audio
                            merged_audio = extended
                        
                        # Mix overlapping portion
                        if overlap_samples > 0 and start_position < len(merged_audio):
                            overlap_end = min(start_position + overlap_samples, len(merged_audio), end_position)
                            audio_overlap = audio_data[:overlap_end - start_position]
                            merged_audio[start_position:overlap_end] = (
                                merged_audio[start_position:overlap_end] * 0.5 + 
                                audio_overlap * 0.5
                            )
                            
                            # Add non-overlapping portion
                            if overlap_end < end_position:
                                remaining = audio_data[overlap_end - start_position:]
                                merged_audio[overlap_end:end_position] = remaining
                        
                        current_position = end_position
                    else:
                        # Zero = immediate continuation
                        actual_overlaps.append(0.0)
                        
                        start_position = current_position
                        end_position = start_position + len(audio_data)
                        
                        if end_position > len(merged_audio):
                            extended = np.zeros(end_position, dtype=merged_audio.dtype)
                            extended[:len(merged_audio)] = merged_audio
                            merged_audio = extended
                        
                        merged_audio[start_position:end_position] = audio_data
                        current_position = end_position
            
            except Exception as e:
                logger.error(f"Error processing audio file: {e}")
                actual_overlaps.append(turn.get('overlap', 0))
                continue
        
        # Save merged audio
        if merged_audio is not None and sample_rate is not None:
            sf.write(str(output_path), merged_audio, sample_rate)
            logger.debug(f"  💾 Saved merged audio: {output_path}")
        else:
            logger.error("No valid audio data to save")
            return None, None
        
        return actual_overlaps, sample_rate
    
    def _merge_rttm_files(self, turns: List[Dict], output_path: Path, actual_overlaps: List[float], conv_id_short: str):
        """Merge RTTM files with synchronized timing."""
        if not turns:
            return
        
        merged_rttm_lines = []
        current_time_offset = 0.0
        
        logger.debug(f"  📝 Merging {len(turns)} RTTM files...")
        
        for i, turn in enumerate(turns):
            try:
                rttm_file = turn.get('rttm_filepath')
                if not rttm_file or not os.path.exists(rttm_file):
                    logger.warning(f"  ⚠️  RTTM not found: {rttm_file}")
                    continue
                
                overlap = actual_overlaps[i] if i < len(actual_overlaps) else turn.get('overlap', 0)
                speaker = turn.get('speaker', f'speaker_{i}')
                
                # Apply overlap/pause before placing segments (except first turn)
                if i > 0:
                    if overlap < 0:
                        # Pause
                        pause_duration = abs(overlap)
                        current_time_offset += pause_duration
                    elif overlap > 0:
                        # Overlap - back up
                        current_time_offset = max(0, current_time_offset - overlap)
                
                # Read RTTM segments
                with open(rttm_file, 'r') as f:
                    rttm_lines = f.readlines()
                
                # Parse segments
                segments = []
                for line in rttm_lines:
                    line = line.strip()
                    if not line or not line.startswith('SPEAKER'):
                        continue
                    
                    parts = line.split()
                    if len(parts) < 8:
                        continue
                    
                    try:
                        seg_start = float(parts[3])
                        seg_duration = float(parts[4])
                        segments.append((seg_start, seg_duration))
                    except (ValueError, IndexError):
                        continue
                
                if not segments:
                    continue
                
                # Sort segments
                segments.sort(key=lambda x: x[0])
                
                # Place segments, keeping pauses <= 1 second
                local_offset = 0.0
                max_pause = 1.0
                
                for seg_idx, (seg_start, seg_duration) in enumerate(segments):
                    adjusted_start = current_time_offset + local_offset
                    
                    rttm_parts = [
                        "SPEAKER",
                        conv_id_short,
                        "1",
                        f"{adjusted_start:.3f}",
                        f"{seg_duration:.3f}",
                        "<NA>",
                        "<NA>",
                        speaker,
                        "<NA>",
                        "<NA>"
                    ]
                    
                    merged_rttm_lines.append(" ".join(rttm_parts))
                    
                    local_offset += seg_duration
                    
                    # Check pause to next segment
                    if seg_idx < len(segments) - 1:
                        next_seg_start = segments[seg_idx + 1][0]
                        current_seg_end = seg_start + seg_duration
                        pause_duration = next_seg_start - current_seg_end
                        
                        if 0 < pause_duration <= max_pause:
                            local_offset += pause_duration
                
                # Update offset
                current_time_offset += local_offset
                
            except Exception as e:
                logger.error(f"  ❌ Error processing RTTM: {e}")
                continue
        
        # Save merged RTTM
        if merged_rttm_lines:
            with open(str(output_path), 'w') as f:
                for line in merged_rttm_lines:
                    f.write(line + '\n')
            logger.debug(f"  ✅ Saved {len(merged_rttm_lines)} segments to RTTM")
        else:
            logger.error("  ❌ No valid RTTM data to save")
