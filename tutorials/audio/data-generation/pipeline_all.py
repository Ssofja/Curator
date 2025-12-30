import argparse
import glob
import json
import os
import sys

from loguru import logger

from nemo_curator.backends.xenna.executor import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage

sys.path.insert(0, os.path.dirname(__file__))
from vllm_inference import vLLMInference, DocumentToAudioStage
from tts_generation import ChatterboxTTSGeneration
from mfa_rttm_generation import MFAToRTTMGeneration
from merge_conversation_stage import MergeConversationStage
from topic_expander import TopicExpander


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", type=str, default=None)
    parser.add_argument("--input-manifest", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--reference-voices", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    
    parser.add_argument("--num-conversations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=None)
    
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--llm-temperature", type=float, default=0.8)
    
    parser.add_argument("--tts-device", type=str, default="cuda")
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--tts-temperature", type=float, default=0.8)
    parser.add_argument("--normalize-audio", action="store_true", default=True)
    parser.add_argument("--normalize-level", type=float, default=-20.0)
    parser.add_argument("--min-pause", type=float, default=0.3)
    parser.add_argument("--max-pause", type=float, default=1.5)
    
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-mfa", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    
    parser.add_argument("--mfa-command", type=str, default="mfa")
    parser.add_argument("--mfa-acoustic-model", type=str, default="english_us_arpa")
    parser.add_argument("--mfa-dictionary", type=str, default="english_us_arpa")
    parser.add_argument("--use-phone-intervals", action="store_true")
    parser.add_argument("--max-gap-for-merge", type=float, default=0.3)
    
    parser.add_argument("--max-pause-duration", type=float, default=2.0)
    parser.add_argument("--randomize-pauses", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    
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
        logger.info(f"End-to-End Pipeline | Input: {args.input_manifest} | Output: {args.output_dir} | Conversations: {args.num_conversations}")
    
    pipeline = Pipeline(name="end_to_end_pipeline")
    executor = XennaExecutor(config={
        "execution_mode": "batch",
    })
    
    if not args.skip_llm:
        if args.verbose:
            logger.info(f"Stage 0: Topic Expansion | Expanding to {args.num_conversations} conversations")
        
        pipeline.add_stage(JsonlReader(file_paths=args.input_manifest))
        pipeline.add_stage(DocumentToAudioStage())
        pipeline.add_stage(TopicExpander(
            num_conversations=args.num_conversations,
            conversations_per_batch=args.batch_size,
            seed=args.random_seed,
        ))
        
        if args.verbose:
            logger.info(f"Stage 1: LLM | Model: {args.llm_model} | Max tokens: {args.max_tokens} | Temp: {args.llm_temperature}")
        
        pipeline.add_stage(
            vLLMInference(
                generation_field="src_text",
                prompt_file=args.prompt_file,
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
            ).with_(
                batch_size=args.batch_size, 
                resources=Resources(gpus=1)
            )
        )
    
    if not args.skip_tts:
        if args.verbose:
            logger.info(f"Stage 2: TTS | Device: {args.tts_device} | Exaggeration: {args.exaggeration} | Output: {audio_dir}")
        
        pipeline.add_stage(
            ChatterboxTTSGeneration(
                output_audio_dir=audio_dir,
                reference_voices_dataset=args.reference_voices,
                device=args.tts_device,
                cfg_weight=args.cfg_weight,
                exaggeration=args.exaggeration,
                temperature=args.tts_temperature,
                normalize_audio=args.normalize_audio,
                normalize_level=args.normalize_level,
            ).with_(
                batch_size=args.batch_size, 
                resources=Resources(gpus=1)
            )
        )
    else:
        if args.verbose:
            logger.info("Stage 2: Skipped")
        if not args.skip_llm:
            raise ValueError("Cannot skip TTS if LLM is enabled. Use --skip-llm to start from existing conversations.")
    
    if not args.skip_mfa:
        if args.verbose:
            logger.info(f"Stage 3: MFA | Command: {args.mfa_command} | Model: {args.mfa_acoustic_model} | Output: {rttm_dir}")
                
        pipeline.add_stage(
            MFAToRTTMGeneration(
                rttm_output_dir=rttm_dir,
                mfa_command=args.mfa_command,
                acoustic_model=args.mfa_acoustic_model,
                dictionary=args.mfa_dictionary,
                use_phone_intervals=args.use_phone_intervals,
                max_gap_for_merge=args.max_gap_for_merge,
                text_field="utterance",
            )
        )
    else:
        if args.verbose:
            logger.info("Stage 3: Skipped")
        if not args.skip_tts:
            raise ValueError("Cannot skip MFA if TTS is enabled. Use --skip-tts to start from existing audio.")
    
    if not args.skip_merge:
        if args.verbose:
            logger.info(f"Stage 4: Merge | Max pause: {args.max_pause_duration}s | Audio: {merged_audio_dir} | RTTMs: {merged_rttm_dir}")
        
        pipeline.add_stage(
            MergeConversationStage(
                output_audio_dir=merged_audio_dir,
                output_rttm_dir=merged_rttm_dir,
                max_pause_duration=args.max_pause_duration,
                randomize_pauses=args.randomize_pauses,
            )
        )
        pipeline.add_stage(AudioToDocumentStage().with_(batch_size=1))
        pipeline.add_stage(JsonlWriter(path=merged_manifest_dir, write_kwargs={"force_ascii": False}))
    else:
        if args.verbose:
            logger.info("Stage 4: Skipped")
    
    if args.verbose:
        logger.info("Running unified pipeline...")
    
    pipeline.run(executor)
    
    if args.verbose:
        if not args.skip_llm:
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
            
            logger.info(f"Stage 1 ✓ | Turns: {total_turns} | Conversations: {len(conversation_ids)} | Avg: {total_turns/len(conversation_ids) if conversation_ids else 0:.1f}")
        
        if not args.skip_tts:
            logger.info(f"Stage 2 ✓ | Audio: {audio_dir} | Manifest: {audio_manifest_dir}")
        
        if not args.skip_mfa:
            logger.info(f"Stage 3 ✓ | RTTMs: {rttm_dir} | Manifest: {final_manifest_dir}")
        
        if not args.skip_merge:
            logger.info(f"Stage 4 ✓ | Audio: {merged_audio_dir} | RTTMs: {merged_rttm_dir} | Manifest: {merged_manifest_dir}")
    
    if args.verbose:        
        logger.info("Pipeline complete!")
        logger.info(f"Output: {args.output_dir}/ | conversations/ audio/ audio_manifest/ rttms/ manifest/ merged_audio/ merged_rttms/ merged_manifest/")


if __name__ == "__main__":
    main()

