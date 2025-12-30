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
MFA-based RTTM Generation Stage for Curator

This stage uses Montreal Forced Aligner (MFA) to generate accurate RTTM files
from audio files and their corresponding text transcriptions.

The flow is:
1. Audio generation creates individual files per speaker turn with text
2. MFA runs forced alignment on each file → generates TextGrid files (word/phone level)
3. This processor converts TextGrids to RTTMs for each file
4. MergeGeneratedConversation uses these RTTMs to merge conversations accurately
"""

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks.audio_batch import AudioBatch

try:
    import textgrid
except ImportError:
    logger.warning("textgrid library not found. Installing praatio for TextGrid parsing...")
    try:
        from praatio import textgrid
    except ImportError:
        logger.error("Neither textgrid nor praatio found. Please install one: pip install textgrid or pip install praatio")
        textgrid = None


class MFAToRTTMGeneration(ProcessingStage):
    """
    Use Montreal Forced Aligner to generate RTTM files from audio and text.
    
    This stage processes each entry individually using `mfa align_one`, enabling
    parallel processing across multiple workers.
    
    Each worker:
    1. Takes an audio file and its text
    2. Runs MFA align_one on that single file
    3. Converts TextGrid output to RTTM
    4. Returns updated entry with RTTM file path
    """
    
    def __init__(
        self,
        rttm_output_dir: str,                   # Directory to save RTTM files
        mfa_command: str = "mfa",               # MFA command (adjust if using micromamba)
        acoustic_model: str = "english_us_arpa",  # Acoustic model to use
        dictionary: str = "english_us_arpa",      # Pronunciation dictionary
        use_phone_intervals: bool = False,      # Use phone-level (True) or word-level (False) intervals
        max_gap_for_merge: float = 0.3,        # Merge intervals if gap < this (seconds)
        cleanup_temp_files: bool = True,        # Clean up temporary MFA files after processing
        text_field: str = "utterance",          # Field in manifest containing text
    ):
        super().__init__()
        self.rttm_output_dir = Path(rttm_output_dir)
        self.mfa_command = mfa_command
        self.acoustic_model = acoustic_model
        self.dictionary = dictionary
        self.use_phone_intervals = use_phone_intervals
        self.max_gap_for_merge = max_gap_for_merge
        self.cleanup_temp_files = cleanup_temp_files
        self.text_field = text_field
        
        # Create output directory
        self.rttm_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"MFAToRTTMGeneration initialized (using align_one for parallel processing)")
        logger.info(f"  RTTM output directory: {self.rttm_output_dir}")
        logger.info(f"  MFA command: {self.mfa_command}")
        logger.info(f"  Acoustic model: {self.acoustic_model}")
        logger.info(f"  Dictionary: {self.dictionary}")
        logger.info(f"  Using {'phone' if self.use_phone_intervals else 'word'}-level intervals")
        logger.info(f"  Max gap for merge: {self.max_gap_for_merge}s")
    
    def xenna_stage_spec(self) -> dict[str, Any]:
        """Allow parallel processing - Curator will determine num_workers automatically."""
        return {}  # Let Curator auto-determine workers for parallel processing
    
    def process(self, batch: AudioBatch) -> AudioBatch:
        """
        Process each entry in the batch individually using MFA align_one.
        
        This enables parallel processing as each entry is independent.
        """
        updated_entries = []
        
        for entry in batch.data:
            try:
                updated_entry = self._process_single_entry(entry)
                updated_entries.append(updated_entry)
            except Exception as e:
                logger.error(f"Error processing entry: {e}")
                # Keep original entry if processing fails
                updated_entries.append(entry)
        
        logger.info(f"✅ MFA processing complete: {len(updated_entries)} entries processed")
        return AudioBatch(data=updated_entries)
    
    def _process_single_entry(self, entry: Dict) -> Dict:
        """Process a single entry with MFA align_one."""
        audio_filepath = entry.get('audio_filepath')
        text = entry.get(self.text_field, '').strip()
        
        if not audio_filepath or not text:
            logger.warning(f"Skipping entry without audio or text")
            return entry
        
        if not os.path.exists(audio_filepath):
            logger.warning(f"Audio file not found: {audio_filepath}")
            return entry
        
        # Create temporary directory for this entry
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Prepare input for MFA
            audio_path = Path(audio_filepath)
            file_stem = audio_path.stem
            
            # Create text file
            text_file = temp_path / f"{file_stem}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Output directory for TextGrid
            output_dir = temp_path / "output"
            output_dir.mkdir()
            
            # Run MFA align_one
            success = self._run_mfa_align_one(
                audio_filepath=audio_filepath,
                text_filepath=str(text_file),
                output_dir=output_dir,
                file_stem=file_stem
            )
            
            if not success:
                logger.warning(f"MFA alignment failed for {file_stem}")
                return entry
            
            # Convert TextGrid to RTTM
            textgrid_path = output_dir / f"{file_stem}.TextGrid"
            if not textgrid_path.exists():
                logger.warning(f"TextGrid not found: {textgrid_path}")
                return entry
            
            rttm_filepath = self._textgrid_to_rttm(
                textgrid_path,
                file_stem,
                entry
            )
            
            # Update entry
            updated_entry = entry.copy()
            updated_entry['rttm_filepath'] = str(rttm_filepath)
            updated_entry['textgrid_filepath'] = str(textgrid_path)
            
            return updated_entry
    
    def _run_mfa_align_one(
        self,
        audio_filepath: str,
        text_filepath: str,
        output_dir: Path,
        file_stem: str
    ) -> bool:
        """Run MFA align_one command on a single audio file."""
        
        try:
            # Split mfa_command if it contains multiple parts (e.g., "micromamba run -n env mfa")
            mfa_cmd_parts = shlex.split(self.mfa_command)
            
            cmd = mfa_cmd_parts + [
                "align_one",
                str(audio_filepath),
                str(text_filepath),
                self.dictionary,
                self.acoustic_model,
                str(output_dir / f"{file_stem}.TextGrid"),
                "--single_speaker",
                "--output_format", "long_textgrid"
            ]
            
            logger.debug(f"Running MFA command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"MFA align_one failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return False
            
            logger.debug(f"MFA alignment completed for {file_stem}")
            return True
            
        except Exception as e:
            logger.error(f"Error running MFA align_one: {e}")
            return False
    
    def _textgrid_to_rttm(
        self,
        textgrid_path: Path,
        file_stem: str,
        entry: Dict
    ) -> Path:
        """
        Convert a TextGrid file to RTTM format.
        
        Args:
            textgrid_path: Path to TextGrid file
            file_stem: File stem for RTTM naming
            entry: Original manifest entry
            
        Returns:
            Path to generated RTTM file
        """
        if textgrid is None:
            raise ImportError("TextGrid parsing library not available")
        
        # Parse TextGrid
        try:
            # Try praatio first
            if hasattr(textgrid, 'openTextgrid'):
                tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)
                # Get words or phones tier
                tier_name = "phones" if self.use_phone_intervals else "words"
                tier = tg.getTier(tier_name) if tier_name in tg.tierNames else tg.getTier(tg.tierNames[0])
                intervals = [(entry_item.start, entry_item.end, entry_item.label) for entry_item in tier.entries]
            else:
                # Try textgrid library
                tg = textgrid.TextGrid.fromFile(str(textgrid_path))
                tier_name = "phones" if self.use_phone_intervals else "words"
                tier = tg.getFirst(tier_name) if tier_name else tg.tiers[0]
                intervals = [(interval.minTime, interval.maxTime, interval.mark) for interval in tier]
        except Exception as e:
            logger.error(f"Error parsing TextGrid {textgrid_path}: {e}")
            intervals = []
        
        # Filter out empty intervals and silence markers
        silence_markers = {'', 'sp', 'sil', 'spn', '<eps>'}
        speech_intervals = []
        
        for start, end, label in intervals:
            if label.strip() and label.strip() not in silence_markers:
                speech_intervals.append({
                    'start': start,
                    'duration': end - start
                })
        
        # Merge close intervals
        merged_intervals = self._merge_intervals(speech_intervals)
        
        # Get speaker from entry
        speaker = entry.get('speaker', 'speaker_0')
        
        # Generate RTTM file
        rttm_filepath = self.rttm_output_dir / f"{file_stem}.rttm"
        
        with open(rttm_filepath, 'w', encoding='utf-8') as f:
            for interval in merged_intervals:
                # RTTM format: SPEAKER file_id channel start_time duration <NA> <NA> speaker_id <NA> <NA>
                line = (
                    f"SPEAKER {file_stem} 1 "
                    f"{interval['start']:.3f} {interval['duration']:.3f} "
                    f"<NA> <NA> {speaker} <NA> <NA>\n"
                )
                f.write(line)
        
        logger.debug(f"TextGrid->RTTM: {file_stem} | {len(speech_intervals)} intervals -> {len(merged_intervals)} segments")
        
        return rttm_filepath
    
    def _merge_intervals(self, intervals: List[Dict]) -> List[Dict]:
        """
        Merge consecutive intervals into continuous speech segments.
        
        Args:
            intervals: List of intervals with 'start' and 'duration'
        
        Returns:
            List of merged speech segments
        """
        if not intervals:
            return []
        
        # Sort by start time
        sorted_intervals = sorted(intervals, key=lambda x: x['start'])
        
        merged = []
        current_start = sorted_intervals[0]['start']
        current_end = current_start + sorted_intervals[0]['duration']
        
        for i in range(1, len(sorted_intervals)):
            interval = sorted_intervals[i]
            interval_start = interval['start']
            interval_end = interval_start + interval['duration']
            
            # Calculate gap between current segment end and next interval start
            gap = interval_start - current_end
            
            if gap <= self.max_gap_for_merge:
                # Small gap or overlap - merge
                current_end = max(current_end, interval_end)
            else:
                # Large gap - finalize current segment and start new one
                merged.append({
                    'start': current_start,
                    'duration': current_end - current_start
                })
                current_start = interval_start
                current_end = interval_end
        
        # Add final segment
        merged.append({
            'start': current_start,
            'duration': current_end - current_start
        })
        
        return merged
