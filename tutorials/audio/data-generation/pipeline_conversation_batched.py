"""
Pipeline with Conversation-Batched Processing for Parallel Streaming

This version keeps all turns of each conversation together through the pipeline,
enabling true parallel processing in streaming mode with explicit Ray configuration.

Key Features:
1. TopicExpander: Creates N conversation prompts
2. vLLMInference: Returns list[AudioBatch], each with ALL turns of ONE conversation
3. TTS/MFA: Process each conversation's turns together
4. MergeConversationStage: Merges immediately (no waiting for all conversations)
5. RayClient: Explicit CPU/GPU resource configuration

Benefits:
-  Streaming mode works: conversations flow through pipeline independently
-  Parallel merging: MergeConversationStage can process multiple conversations at once
-  Lower latency: First conversation completes while others are still being generated
-  Better resource utilization: All stages can be active simultaneously
-  Explicit Ray control: Configure CPUs/GPUs via command line args

Usage:
    python pipeline_conversation_batched.py \\
        --input-manifest=/path/to/topics.jsonl \\
        --output-dir=/path/to/output \\
        --reference-voices=/path/to/voices \\
        --num-conversations=80 \\
        --batch-size=10 \\
        --num-cpus=8 \\
        --num-gpus=8 \\
        --verbose

Single-Node Slurm Usage:
    srun --gpus-per-node=8 \\
         --container-image=/path/to/image.sqsh \\
         --container-mounts=/path:/artifacts \\
         bash -c "python pipeline_conversation_batched.py \\
           --num-cpus=8 --num-gpus=8 \\
           --num-conversations=80 --batch-size=10 \\
           --verbose"

Multi-Node Slurm Usage:
    # Use the run_multinode.sh script for true multi-node execution:
    sbatch run_multinode.sh
    
    # Or connect to an existing Ray cluster:
    python pipeline_conversation_batched.py \\
        --ray-address=HEAD_NODE_IP:6379 \\
        --num-cpus=448 --num-gpus=56 \\
        --num-conversations=300 --batch-size=1 \\
        --verbose
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import ray
from loguru import logger

from nemo_curator.core.client import RayClient

sys.path.insert(0, os.path.dirname(__file__))
from audio_data_workflow import AudioDataGenerationWorkflow


def parse_args():
    parser = argparse.ArgumentParser(description="Conversation-batched audio data generation pipeline")
    parser.add_argument("--raw-data-dir", type=str, default=None)
    parser.add_argument("--input-manifest", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--reference-voices", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    
    parser.add_argument("--num-conversations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=None)
    
    # Ray resource configuration
    parser.add_argument("--num-cpus", type=int, default=8, help="Number of CPUs for Ray to use")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs for Ray to use")
    parser.add_argument("--ray-address", type=str, default=None, 
                        help="Address of existing Ray cluster to connect to (e.g., '192.168.1.1:6379'). "
                             "If not provided, a new local Ray cluster will be started.")
    
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
    
    parser.add_argument("--batch-size", type=int, default=10, 
                       help="Number of conversations per batch for vLLM generation")
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    pipeline_start_time = time.time()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.raw_data_dir is None:
        args.raw_data_dir = os.path.join(script_dir, "in")
    
    if args.input_manifest is None:
        args.input_manifest = os.path.join(args.raw_data_dir, "in.jsonl")
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.raw_data_dir, "result")
    
    if args.prompt_file is None:
        args.prompt_file = os.path.join(args.raw_data_dir, "prompt.yaml")
    
    client = None
    
    if args.ray_address:
        ray.init(address=args.ray_address, ignore_reinit_error=True)
        
        cluster_resources = ray.cluster_resources()
    else:
        client = RayClient(
            num_cpus=args.num_cpus,
            num_gpus=args.num_gpus,
            include_dashboard=True,
        )
        client.start()
    
    try:
        workflow = AudioDataGenerationWorkflow(
            input_manifest=args.input_manifest,
            output_dir=args.output_dir,
            reference_voices=args.reference_voices,
            num_conversations=args.num_conversations,
            batch_size=args.batch_size,
            random_seed=args.random_seed,
            llm_model=args.llm_model,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            llm_temperature=args.llm_temperature,
            prompt_file=args.prompt_file,
            tts_device=args.tts_device,
            cfg_weight=args.cfg_weight,
            exaggeration=args.exaggeration,
            tts_temperature=args.tts_temperature,
            normalize_audio=args.normalize_audio,
            normalize_level=args.normalize_level,
            min_pause=args.min_pause,
            max_pause=args.max_pause,
            mfa_command=args.mfa_command,
            mfa_acoustic_model=args.mfa_acoustic_model,
            mfa_dictionary=args.mfa_dictionary,
            use_phone_intervals=args.use_phone_intervals,
            max_gap_for_merge=args.max_gap_for_merge,
            max_pause_duration=args.max_pause_duration,
            randomize_pauses=args.randomize_pauses,
            skip_llm=args.skip_llm,
            skip_tts=args.skip_tts,
            skip_mfa=args.skip_mfa,
            skip_merge=args.skip_merge,
            verbose=args.verbose,
        )
        
        workflow.run()
        
        pipeline_end_time = time.time()
        total_duration = pipeline_end_time - pipeline_start_time
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = total_duration % 60
        
        logger.info("=" * 80)
        logger.info(" Pipeline completed successfully!")
        logger.info(f"Total execution time: {hours:02d}:{minutes:02d}:{seconds:06.3f}")
        logger.info("=" * 80)
        
    except Exception as e:
        pipeline_end_time = time.time()
        total_duration = pipeline_end_time - pipeline_start_time
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = total_duration % 60
        
        logger.error("=" * 80)
        logger.error(f"Pipeline failed after {hours:02d}:{minutes:02d}:{seconds:06.3f}")
        logger.error(f"Error: {e}")
        logger.error("=" * 80)
        raise
        
    finally:
        if client is not None:
            client.stop()
        elif args.ray_address:
            ray.shutdown()


if __name__ == "__main__":
    main()

