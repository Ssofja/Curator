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

import atexit
import glob
import json
import os
import random
import shutil
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio as ta
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks.audio_batch import AudioBatch


class ChatterboxTTSGeneration(ProcessingStage):
    """
    Processor that generates audio using ChatterboxTTS model for individual utterances.
    
    This processor takes a manifest containing individual utterances with speaker
    and text information, generates audio for each utterance using Chatterbox TTS
    with voice cloning, and outputs the same manifest with added audio file paths
    and durations.
    
    The processor supports:
    - Speaker-conditioned audio generation using reference voices from david_ai_filtered
    - Consistent voice assignment per speaker across turns
    - Reference speaker selection with RTTM-based pause removal
    - Multi-speaker conversations with voice cloning
    - Automatic audio normalization for proper volume levels
    - Audio file management and manifest updates
    
    Args:
        output_audio_dir (str): Directory where generated audio files will be saved.
        reference_voices_dataset (str): Path to david_ai_filtered dataset containing wavs and rttms.
        device (str): Device to use for inference (cuda/cpu). Defaults to "cuda".
        max_reference_duration (float): Maximum duration in seconds for reference audio. Defaults to 60.0.
        sample_rate (int): Sample rate for generated audio (Chatterbox uses 24000). Defaults to 24000.
        cfg_weight (float): Classifier-free guidance weight (0.0-1.0). Defaults to 0.5.
        exaggeration (float): Emotion exaggeration level (0.0-1.0). Defaults to 0.5.
        temperature (float): Sampling temperature. Defaults to 0.8.
        repetition_penalty (float): Repetition penalty. Defaults to 1.2.
        min_p (float): Min-p sampling parameter. Defaults to 0.05.
        top_p (float): Top-p sampling parameter. Defaults to 1.0.
        normalize_audio (bool): Whether to normalize generated audio volume. Defaults to True.
        normalize_level (float): Target loudness in dB for normalization. Defaults to -20.0.
        preserve_original_fields (bool): Whether to preserve original manifest fields. Defaults to True.
    """
    
    def __init__(
        self,
        output_audio_dir: str,
        reference_voices_dataset: str,
        device: str = "cuda",
        max_reference_duration: float = 60.0,
        sample_rate: int = 24000,
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        normalize_audio: bool = True,
        normalize_level: float = -20.0,
        preserve_original_fields: bool = True,
    ):
        super().__init__()
        
        
        if not os.path.exists(reference_voices_dataset):
            raise ValueError(f"Reference voices dataset path not found: {reference_voices_dataset}")
            
        self.output_audio_dir = output_audio_dir
        self.reference_voices_dataset = reference_voices_dataset
        self.device = device
        self.max_reference_duration = max_reference_duration
        self.sample_rate = sample_rate
        self.cfg_weight = cfg_weight
        self.exaggeration = float(exaggeration)
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.min_p = min_p
        self.top_p = top_p

        self.normalize_audio = normalize_audio
        self.normalize_level = normalize_level
        self.preserve_original_fields = preserve_original_fields
        
        os.makedirs(self.output_audio_dir, exist_ok=True)
        
        self.model = None
        self.reference_wavs_list = None
        self.speaker_to_reference = {}
        self.current_conversation_id = None
        
        self.temp_dir = tempfile.mkdtemp(prefix="chatterbox_processed_audio_")
        logger.info(f"Created temporary directory for processed audio: {self.temp_dir}")
        
        atexit.register(self._cleanup_temp_dir)
    
    def setup(self, worker_metadata=None):
        """Initialize the TTS model once when the worker starts."""
        logger.info("Setting up ChatterboxTTSGeneration worker...")
        self.prepare()  # Load model and reference voices
        logger.info("ChatterboxTTSGeneration worker ready!")
    
    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {self.temp_dir}: {e}")
    
    def __del__(self):
        self._cleanup_temp_dir()
        
    def _load_model(self):
        """Load ChatterboxTTS model."""
        try:
            from chatterbox.tts import ChatterboxTTS
            
            logger.info("Loading ChatterboxTTS model...")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            logger.info("ChatterboxTTS model loaded successfully")
            
        except ImportError as e:
            logger.error("ChatterboxTTS not installed. Please install: pip install chatterbox-tts")
            raise
        except Exception as e:
            logger.error(f"Error loading ChatterboxTTS model: {e}")
            raise
    
    def _load_reference_audio_files(self):
        """Load reference WAV files from david_ai_filtered."""
        try:
            wav_pattern = os.path.join(self.reference_voices_dataset, "wavs", "*", "*.wav")
            self.reference_wavs_list = glob.glob(wav_pattern)
            
            if not self.reference_wavs_list:
                raise ValueError(f"No reference audio files found in {self.reference_voices_dataset}/wavs/")
            
            logger.info(f"Found {len(self.reference_wavs_list)} reference audio files")
            
        except Exception as e:
            logger.error(f"Error loading reference audio files: {e}")
            raise
    
    def _get_audio_duration(self, file_path: str) -> float:
        """Get audio duration using soundfile."""
        with sf.SoundFile(file_path) as audio_file:
            return len(audio_file) / audio_file.samplerate
    
    def _process_audio_with_rttm(self, audio_filepath: str, rttm_filepath: str) -> str:
        """
        Process audio file using RTTM to remove pauses and extract only speech segments.
        This is crucial for multi-channel podcast data with many pauses.
        
        Args:
            audio_filepath (str): Path to the input audio file
            rttm_filepath (str): Path to the RTTM file containing speech segments
            
        Returns:
            str: Path to the processed audio file with pauses removed
        """
        try:
            audio_data, sample_rate = sf.read(audio_filepath)
            
            speech_segments = []
            with open(rttm_filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5 and parts[0] == 'SPEAKER':
                        start_time = float(parts[3])
                        duration = float(parts[4])
                        end_time = start_time + duration
                        speech_segments.append((start_time, end_time))
            
            if not speech_segments:
                logger.warning(f"No speech segments found in RTTM file: {rttm_filepath}")
                return audio_filepath
            
            speech_segments.sort(key=lambda x: x[0])
            
            concatenated_segments = []
            total_extracted_duration = 0.0
            
            for start_time, end_time in speech_segments:
                segment_duration = end_time - start_time
                
                if total_extracted_duration + segment_duration > self.max_reference_duration:
                    remaining_duration = self.max_reference_duration - total_extracted_duration
                    if remaining_duration > 0.1:
                        end_time = start_time + remaining_duration
                        segment_duration = remaining_duration
                    else:
                        break
                
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_data), end_sample)
                
                if start_sample < end_sample:
                    segment = audio_data[start_sample:end_sample]
                    concatenated_segments.append(segment)
                    total_extracted_duration += segment_duration
                
                if total_extracted_duration >= self.max_reference_duration:
                    break
            
            if not concatenated_segments:
                logger.warning(f"No valid speech segments extracted from: {audio_filepath}")
                return audio_filepath
            
            processed_audio = np.concatenate(concatenated_segments, axis=0)
            
            base_name = os.path.splitext(os.path.basename(audio_filepath))[0]
            dialog_id = audio_filepath.split(os.sep)[-2]
            output_filepath = os.path.join(
                self.temp_dir, 
                dialog_id, 
                f"{base_name}_no_pauses_{self.max_reference_duration}s.wav"
            )
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            
            sf.write(output_filepath, processed_audio, sample_rate)
            
            logger.info(f"Processed reference audio (removed pauses): {output_filepath}")
            logger.info(f"  Original duration: {len(audio_data) / sample_rate:.2f}s, "
                       f"Extracted speech: {len(processed_audio) / sample_rate:.2f}s")
            
            return output_filepath
            
        except Exception as e:
            logger.error(f"Error processing audio with RTTM {rttm_filepath}: {e}")
            return audio_filepath
    
    def _get_reference_audio_for_speaker(self, speaker: str, already_assigned_references: set) -> str:
        """
        Get or assign reference audio file for a speaker.
        Maintains consistent voice across turns for the same speaker.
        
        Args:
            speaker (str): Speaker identifier
            already_assigned_references (set): Set of already assigned reference audio paths
            
        Returns:
            str: Path to processed reference audio file (with pauses removed)
        """
        if speaker in self.speaker_to_reference:
            logger.info(f"Using cached reference for speaker '{speaker}': {self.speaker_to_reference[speaker]}")
            return self.speaker_to_reference[speaker]
        
        available_wavs = [
            wav for wav in self.reference_wavs_list 
            if wav not in already_assigned_references
        ]
        
        if not available_wavs:
            logger.warning("All reference voices assigned, allowing reuse")
            available_wavs = self.reference_wavs_list
        
        selected_wav = random.choice(available_wavs)
        logger.info(f"🎯 Random selection: chose from {len(available_wavs)} available voices")
        
        wav_parts = selected_wav.split(os.sep)
        dialog_id = wav_parts[-2]
        speaker_file = wav_parts[-1]
        speaker_id = os.path.splitext(speaker_file)[0]
        
        logger.info(f"  📂 Selected: {dialog_id}/{speaker_id}")
        
        rttm_path = os.path.join(self.reference_voices_dataset, "rttms", dialog_id, f"{speaker_id}.rttm")
        
        if os.path.exists(rttm_path):
            processed_audio = self._process_audio_with_rttm(selected_wav, rttm_path)
        else:
            logger.warning(f"RTTM not found for {selected_wav}, using original audio")
            processed_audio = selected_wav
        
        self.speaker_to_reference[speaker] = processed_audio
        logger.info(f"Assigned new reference for speaker '{speaker}': {processed_audio}")
        
        return processed_audio
    
    def _normalize_audio(self, wav: torch.Tensor, target_level: float = -20.0) -> torch.Tensor:
        """
        Normalize audio to target dB level to fix low volume issues in TTS output.
        
        Args:
            wav: Audio tensor
            target_level: Target loudness in dB (default -20 dB is good for speech)
        
        Returns:
            Normalized audio tensor
        """
        wav_np = wav.numpy() if isinstance(wav, torch.Tensor) else wav
        
        rms = np.sqrt(np.mean(wav_np**2))
        
        if rms < 1e-10:
            logger.warning("Audio is silent or near-silent, skipping normalization")
            return wav
        
        current_level_db = 20 * np.log10(rms)
        
        gain_db = target_level - current_level_db
        gain_linear = 10 ** (gain_db / 20)
        
        normalized_wav = wav_np * gain_linear
        
        normalized_wav = np.tanh(normalized_wav * 0.9) / 0.9
        normalized_wav = np.clip(normalized_wav, -0.99, 0.99)
        
        logger.debug(f"Volume normalization: {gain_db:+.2f} dB (before: {current_level_db:.2f} dB)")
        
        return torch.from_numpy(normalized_wav).reshape(wav.shape)
    
    def _generate_audio_for_turn(self, text: str, speaker: str, reference_wav: str, conversation_id: str = "unknown") -> np.ndarray:
        """
        Generate audio for a single conversation turn using ChatterboxTTS.
        
        Args:
            text (str): Text to synthesize
            speaker (str): Speaker identifier
            reference_wav (str): Path to reference audio for voice cloning
            conversation_id (str): Conversation identifier for consistent exaggeration
            
        Returns:
            np.ndarray: Generated audio as numpy array
        """
        try:
            logger.info(f"Generating audio for speaker '{speaker}' (exag={self.exaggeration:.3f}): '{text[:50]}...'")
            
            with torch.inference_mode():
                wav = self.model.generate(
                    text,
                    audio_prompt_path=reference_wav,
                    cfg_weight=self.cfg_weight,
                    exaggeration=self.exaggeration,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                    min_p=self.min_p,
                    top_p=self.top_p,
                )
            
            if self.normalize_audio:
                wav = self._normalize_audio(wav, target_level=self.normalize_level)
            
            audio = wav.squeeze(0).numpy()
            
            return audio
            
        except Exception as e:
            logger.error(f"Error generating audio for turn: {e}")
            return np.zeros(self.sample_rate * 2)

    def prepare(self):
        """Initialize model and data once before processing entries."""
        if self.model is None:
            self._load_model()
            self._load_reference_audio_files()
    
    def process_dataset_entry(self, data_entry: Dict) -> Dict:
        """
        Generate audio for a single data entry (utterance).
        Maintains speaker-to-voice consistency within each conversation.
        Resets voices for new conversations to ensure variety across different dialogues.
        """
        if self.model is None:
            self.prepare()
        
        if 'utterance' in data_entry:
            text = data_entry.get('utterance', '').strip()
        elif 'text' in data_entry:
            text = data_entry.get('text', '').strip()
        else:
            logger.warning("No text or utterance found in data entry, skipping")
            raise ValueError(f"No text or utterance found in data entry '{data_entry}'")    

        speaker = data_entry.get('speaker', 'unknown')
        conversation_id = data_entry.get('conversation_id', 'unknown')
        
        if self.current_conversation_id != conversation_id:
            if self.current_conversation_id is not None:
                logger.info(f"🔄 New conversation detected ({conversation_id[:8]}...), resetting voice assignments")
                logger.info(f"   Previous conversation: {self.current_conversation_id[:8]}...")
                logger.info(f"   Speakers in previous conversation: {list(self.speaker_to_reference.keys())}")
            
            self.speaker_to_reference = {}
            self.current_conversation_id = conversation_id
            
            logger.info(f"🎤 Starting new conversation: {conversation_id[:8]}...")
        
        if not text:
            raise ValueError("Empty text found for speaker '{speaker}'")
        
        try:
            already_assigned = set(self.speaker_to_reference.values())
            
            reference_audio = self._get_reference_audio_for_speaker(speaker, already_assigned)
            
            start_time = time.time()
            
            turn_audio = self._generate_audio_for_turn(
                text=text,
                speaker=speaker,
                reference_wav=reference_audio,
                conversation_id=conversation_id
            )
            
            conv_id_short = conversation_id[:12] if len(conversation_id) > 12 else conversation_id
            text_hash = hash(text) % 1000000
            output_filename = f"{conv_id_short}_{speaker}_{text_hash}.wav"
            output_audio_path = os.path.join(self.output_audio_dir, output_filename)
            
            sf.write(output_audio_path, turn_audio, self.sample_rate)
            
            updated_entry = data_entry.copy()
            updated_entry['audio_filepath'] = output_audio_path
            updated_entry['duration'] = len(turn_audio) / self.sample_rate
            
            if self.preserve_original_fields:
                for key, value in data_entry.items():
                    if key not in updated_entry:
                        updated_entry[key] = value
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Generated audio in {elapsed:.2f}s: {output_audio_path} "
                       f"(duration: {updated_entry['duration']:.2f}s)")
            
            return updated_entry
            
        except Exception as e:
            logger.error(f"Error processing entry for speaker '{speaker}': {e}")
            import traceback
            traceback.print_exc()
            return data_entry
    
    def process(self, batch: AudioBatch) -> AudioBatch:
        """Process batch of audio data entries."""
        if self.model is None:
            self.prepare()
        
        processed_data = []
        for entry in batch.data:
            if isinstance(entry, str) and os.path.isfile(entry):
                logger.info(f"Reading manifest file: {entry}")
                with open(entry, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            manifest_entry = json.loads(line)
                            updated_entry = self.process_dataset_entry(manifest_entry)
                            processed_data.append(updated_entry)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON line in {entry}: {e}")
                            continue
            else:
                updated_entry = self.process_dataset_entry(entry)
                processed_data.append(updated_entry)
        
        return AudioBatch(data=processed_data)
