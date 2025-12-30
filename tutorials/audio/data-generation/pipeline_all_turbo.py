import argparse
import os
import sys

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.resources import Resources

sys.path.insert(0, os.path.dirname(__file__))
from vllm_inference import vLLMInference, DocumentToAudioStage
from tts_generation_turbo import ChatterboxTurboTTSGeneration
from mfa_rttm_generation import MFAToRTTMGeneration
from merge_conversation_stage import MergeConversationStage


def parse_args():
    parser = argparse.ArgumentParser(description="Chatterbox-Turbo TTS Pipeline (350M params, optimized for low-latency)")
    parser.add_argument("--raw-data-dir", type=str, default=None)
    parser.add_argument("--input-manifest", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--reference-voices", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    
    parser.add_argument("--num-generations", type=int, default=10)
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--llm-temperature", type=float, default=0.8)
    
    # Chatterbox-Turbo parameters (simplified - no cfg_weight or exaggeration)
    parser.add_argument("--tts-device", type=str, default="cuda")
    parser.add_argument("--tts-temperature", type=float, default=0.8)
    parser.add_argument("--normalize-audio", action="store_true", default=True)
    parser.add_argument("--normalize-level", type=float, default=-20.0)
    
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-mfa", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    
    parser.add_argument("--mfa-command", type=str, default="mfa")
    parser.add_argument("--mfa-acoustic-model", type=str, default="english_us_arpa")
    parser.add_argument("--mfa-dictionary", type=str, default="english_us_arpa")
    parser.add_argument("--mfa-num-jobs", type=int, default=4)
    parser.add_argument("--use-phone-intervals", action="store_true")
    parser.add_argument("--max-gap-for-merge", type=float, default=0.3)
    
    parser.add_argument("--max-pause-duration", type=float, default=2.0)
    parser.add_argument("--randomize-pauses", action="store_true")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.raw_data_dir is None:
        args.raw_data_dir = os.path.join(script_dir, "in")
    
    if args.input_manifest is None:
        args.input_manifest = os.path.join(args.raw_data_dir, "in.jsonl")
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.raw_data_dir, "result")
    
    if args.prompt_file is None:
        args.prompt_file = os.path.join(args.raw_data_dir, "prompt.yaml")
    
    if not os.path.exists(args.input_manifest):
        raise FileNotFoundError(f"Input manifest not found: {args.input_manifest}")
    
    if not args.skip_tts and not os.path.exists(args.reference_voices):
        raise FileNotFoundError(f"Reference voices dataset not found: {args.reference_voices}")
    
    conversations_dir = os.path.join(args.output_dir, "conversations")
    audio_dir = os.path.join(args.output_dir, "audio")
    audio_manifest_dir = os.path.join(args.output_dir, "audio_manifest")
    rttm_dir = os.path.join(args.output_dir, "rttms")
    final_manifest_dir = os.path.join(args.output_dir, "manifest")
    merged_audio_dir = os.path.join(args.output_dir, "merged_audio")
    merged_rttm_dir = os.path.join(args.output_dir, "merged_rttms")
    merged_manifest_dir = os.path.join(args.output_dir, "merged_manifest")
    
    os.makedirs(conversations_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(audio_manifest_dir, exist_ok=True)
    os.makedirs(rttm_dir, exist_ok=True)
    os.makedirs(final_manifest_dir, exist_ok=True)
    os.makedirs(merged_audio_dir, exist_ok=True)
    os.makedirs(merged_rttm_dir, exist_ok=True)
    os.makedirs(merged_manifest_dir, exist_ok=True)
    
    if args.verbose:
        logger.info("="*80)
        logger.info("End-to-End Pipeline: LLM + Chatterbox-Turbo TTS + MFA + Merge")
        logger.info("Chatterbox-Turbo: 350M params, optimized for low-latency voice agents")
        logger.info("="*80)
        logger.info(f"Input manifest: {args.input_manifest}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Conversations dir: {conversations_dir}")
        logger.info(f"Reference voices: {args.reference_voices}")
        logger.info(f"Num generations: {args.num_generations}")
        logger.info("="*80)
    
    if not args.skip_llm:
        if args.verbose:
            logger.info("\nStage 1: LLM Conversation Generation")
            logger.info(f"Model: {args.llm_model}")
            logger.info(f"Max tokens: {args.max_tokens}")
            logger.info(f"Temperature: {args.llm_temperature}")
        llm_pipeline = Pipeline(name="llm_generation")
        
        llm_pipeline.add_stage(JsonlReader(file_paths=args.input_manifest))
        llm_pipeline.add_stage(DocumentToAudioStage())
        llm_pipeline.add_stage(
            vLLMInference(
                generation_field="src_text",
                prompt_file=args.prompt_file,
                num_generations=args.num_generations,
                model={
                    "model": args.llm_model,
                    "tensor_parallel_size": 1,
                    "max_model_len": args.max_model_len,
                    "enable_chunked_prefill": False,
                    "enforce_eager": True,
                    "dtype": "float16",
                    "gpu_memory_utilization": 0.8,
                    "max_num_seqs": 8,
                },
                inference={
                    "max_tokens": args.max_tokens,
                    "temperature": args.llm_temperature,
                    "top_p": 0.95,
                },
                apply_chat_template={
                    "tokenize": False,
                    "add_generation_prompt": True
                }
            ).with_(batch_size=args.batch_size, resources=Resources(gpus=1))
        )
        
        from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
        llm_pipeline.add_stage(AudioToDocumentStage().with_(batch_size=1))
        llm_pipeline.add_stage(JsonlWriter(path=conversations_dir, write_kwargs={"force_ascii": False}))
        
        executor = XennaExecutor()
        llm_pipeline.run(executor)
        
        if args.verbose:
            import json
            import glob
            total_turns = 0
            conversation_ids = set()
            output_files = glob.glob(os.path.join(conversations_dir, "*.jsonl"))
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
            
            logger.info(f"\nStage 1 Complete:")
            logger.info(f"  Output dir: {conversations_dir}")
            logger.info(f"  Total turns: {total_turns}")
            logger.info(f"  Conversations: {len(conversation_ids)}")
            logger.info(f"  Avg turns/conv: {total_turns/len(conversation_ids) if conversation_ids else 0:.1f}")
    else:
        if args.verbose:
            logger.info("\nStage 1: Skipped (using existing conversations)")
            logger.info(f"  Using: {conversations_dir}")
    
    if not args.skip_tts:
        if args.verbose:
            logger.info("\nStage 2: Chatterbox-Turbo TTS Audio Generation")
            logger.info(f"Model: Chatterbox-Turbo (350M params, low-latency)")
            logger.info(f"Device: {args.tts_device}")
            logger.info(f"Temperature: {args.tts_temperature}")
            logger.info(f"Output audio: {audio_dir}")
            logger.info(f"Output manifest: {audio_manifest_dir}")
        
        import glob
        conversation_files = glob.glob(os.path.join(conversations_dir, "*.jsonl"))
        if not conversation_files:
            raise FileNotFoundError(f"No conversation files found in {conversations_dir}")
        
        tts_pipeline = Pipeline(name="tts_turbo_generation")
        
        tts_pipeline.add_stage(JsonlReader(file_paths=conversation_files))
        tts_pipeline.add_stage(DocumentToAudioStage())
        tts_pipeline.add_stage(
            ChatterboxTurboTTSGeneration(
                output_audio_dir=audio_dir,
                reference_voices_dataset=args.reference_voices,
                device=args.tts_device,
                temperature=args.tts_temperature,
                normalize_audio=args.normalize_audio,
                normalize_level=args.normalize_level,
            ).with_(batch_size=args.batch_size, resources=Resources(gpus=1))
        )
        
        from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
        tts_pipeline.add_stage(AudioToDocumentStage().with_(batch_size=1))
        tts_pipeline.add_stage(JsonlWriter(path=audio_manifest_dir, write_kwargs={"force_ascii": False}))
        
        executor = XennaExecutor()
        tts_pipeline.run(executor)
        
        if args.verbose:
            logger.info(f"\nStage 2 Complete:")
            logger.info(f"  Audio files: {audio_dir}")
            logger.info(f"  Manifest: {audio_manifest_dir}")
    else:
        if args.verbose:
            logger.info("\nStage 2: Skipped (conversations only)")
    
    if not args.skip_mfa:
        if args.verbose:
            logger.info("\nStage 3: MFA RTTM Generation")
            logger.info(f"MFA command: {args.mfa_command}")
            logger.info(f"Acoustic model: {args.mfa_acoustic_model}")
            logger.info(f"Output RTTMs: {rttm_dir}")
        
        import glob
        audio_manifest_files = glob.glob(os.path.join(audio_manifest_dir, "*.jsonl"))
        if not audio_manifest_files:
            raise FileNotFoundError(f"No audio manifest files found in {audio_manifest_dir}")
        
        mfa_pipeline = Pipeline(name="mfa_rttm_generation")
        
        mfa_pipeline.add_stage(JsonlReader(file_paths=audio_manifest_files))
        mfa_pipeline.add_stage(DocumentToAudioStage())
        mfa_pipeline.add_stage(
            MFAToRTTMGeneration(
                rttm_output_dir=rttm_dir,
                mfa_command=args.mfa_command,
                acoustic_model=args.mfa_acoustic_model,
                dictionary=args.mfa_dictionary,
                use_phone_intervals=args.use_phone_intervals,
                max_gap_for_merge=args.max_gap_for_merge,
                num_jobs=args.mfa_num_jobs,
                text_field="utterance",
            )
        )
        
        from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
        mfa_pipeline.add_stage(AudioToDocumentStage().with_(batch_size=1))
        mfa_pipeline.add_stage(JsonlWriter(path=final_manifest_dir, write_kwargs={"force_ascii": False}))
        
        executor = XennaExecutor()
        mfa_pipeline.run(executor)
        
        if args.verbose:
            logger.info(f"\nStage 3 Complete:")
            logger.info(f"  RTTM files: {rttm_dir}")
            logger.info(f"  Final manifest: {final_manifest_dir}")
    else:
        if args.verbose:
            logger.info("\nStage 3: Skipped (no MFA processing)")
    
    if not args.skip_merge:
        if args.verbose:
            logger.info("\nStage 4: Merge Conversations")
            logger.info(f"Max pause: {args.max_pause_duration}s")
            logger.info(f"Output audio: {merged_audio_dir}")
            logger.info(f"Output RTTMs: {merged_rttm_dir}")
        
        import glob
        final_manifest_files = glob.glob(os.path.join(final_manifest_dir, "*.jsonl"))
        if not final_manifest_files:
            raise FileNotFoundError(f"No final manifest files found in {final_manifest_dir}")
        
        merge_pipeline = Pipeline(name="merge_conversations")
        
        merge_pipeline.add_stage(JsonlReader(file_paths=final_manifest_files))
        merge_pipeline.add_stage(DocumentToAudioStage())
        merge_pipeline.add_stage(
            MergeConversationStage(
                output_audio_dir=merged_audio_dir,
                output_rttm_dir=merged_rttm_dir,
                max_pause_duration=args.max_pause_duration,
                randomize_pauses=args.randomize_pauses,
            )
        )
        
        from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
        merge_pipeline.add_stage(AudioToDocumentStage().with_(batch_size=1))
        merge_pipeline.add_stage(JsonlWriter(path=merged_manifest_dir, write_kwargs={"force_ascii": False}))
        
        executor = XennaExecutor()
        merge_pipeline.run(executor)
        
        if args.verbose:
            logger.info(f"\nStage 4 Complete:")
            logger.info(f"  Merged audio: {merged_audio_dir}")
            logger.info(f"  Merged RTTMs: {merged_rttm_dir}")
            logger.info(f"  Merged manifest: {merged_manifest_dir}")
    else:
        if args.verbose:
            logger.info("\nStage 4: Skipped (no conversation merging)")
    
    if args.verbose:
        logger.info("\n" + "="*80)
        logger.info("Pipeline Complete!")
        logger.info("="*80)
        logger.info(f"Output structure:")
        logger.info(f"  {args.output_dir}/")
        logger.info(f"  ├── conversations/      # LLM output (Stage 1)")
        logger.info(f"  ├── audio/             # Turn audio files (Stage 2 - Turbo TTS)")
        logger.info(f"  ├── audio_manifest/    # Manifests with audio + text (Stage 2)")
        logger.info(f"  ├── rttms/             # Turn RTTMs (Stage 3)")
        logger.info(f"  ├── manifest/          # Turn manifests with RTTMs (Stage 3)")
        logger.info(f"  ├── merged_audio/      # Conversation audio (Stage 4)")
        logger.info(f"  ├── merged_rttms/      # Conversation RTTMs (Stage 4)")
        logger.info(f"  └── merged_manifest/   # Final conversation manifests (Stage 4)")
        logger.info("="*80)


if __name__ == "__main__":
    main()
