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
Topic Expander Stage for Curator

This stage takes a list of topics and expands them into N conversation entries,
each with a randomly selected topic. This allows NeMo Curator to parallelize
conversation generation across multiple workers.
"""

import random
from typing import Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioBatch


class TopicExpander(ProcessingStage):
    """
    Expand a list of topics into N conversation generation tasks.
    
    This stage:
    1. Reads all topics from the input batch
    2. Creates N entries, each with a randomly selected topic
    3. Outputs N entries for parallel processing by downstream stages
    
    Args:
        num_conversations (int): Number of conversations to generate
        seed (int, optional): Random seed for reproducibility
    """
    
    def __init__(
        self,
        num_conversations: int,
        conversations_per_batch: int = 10,
        seed: int = None,
    ):
        super().__init__()
        self.num_conversations = num_conversations
        self.conversations_per_batch = conversations_per_batch
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
        
        logger.info(f"TopicExpander initialized: {num_conversations} conversations, {conversations_per_batch} per batch")
    
    def xenna_stage_spec(self) -> dict[str, Any]:
        """Configure for sequential execution (processes all topics at once)."""
        return {
            "num_workers": 1,  # Process all topics in one worker to expand them
        }
    
    def process(self, batch: AudioBatch) -> list[AudioBatch]:
        """
        Expand topics into batches of conversations.
        Creates larger batches to enable efficient parallel processing.
        """
        # Collect all topics from the input batch
        topics = []
        for entry in batch.data:
            if isinstance(entry, dict) and 'topic' in entry:
                topics.append(entry['topic'])
            elif isinstance(entry, str):
                topics.append(entry)
        
        if not topics:
            logger.error("No topics found in input batch!")
            return []
        
        logger.info(f"Found {len(topics)} topics, expanding to {self.num_conversations} conversations in batches of {self.conversations_per_batch}")
        
        # Create batches with multiple conversations each
        output_batches = []
        current_batch_entries = []
        
        for i in range(self.num_conversations):
            topic = random.choice(topics)
            entry = {
                'topic': topic,
                'conversation_index': i,
            }
            current_batch_entries.append(entry)
            
            # When we have enough conversations, create a batch
            if len(current_batch_entries) >= self.conversations_per_batch:
                output_batches.append(AudioBatch(
                    data=current_batch_entries,
                    task_id=f"conversations_{len(output_batches)}",
                    dataset_name=batch.dataset_name,
                ))
                current_batch_entries = []
        
        # Add remaining conversations as final batch
        if current_batch_entries:
            output_batches.append(AudioBatch(
                data=current_batch_entries,
                task_id=f"conversations_{len(output_batches)}",
                dataset_name=batch.dataset_name,
            ))
        
        logger.info(f"✅ Created {len(output_batches)} batches for parallel processing")
        
        return output_batches
