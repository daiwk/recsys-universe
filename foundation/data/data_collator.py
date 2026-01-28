"""
Data collator for RL training.
Handles padding and batching of variable-length sequences.
"""

import torch
from typing import List, Dict, Any


class RLDataCollator:
    """
    Data collator for reinforcement learning.
    
    Handles padding of variable-length sequences for batch training.
    """
    
    def __init__(self, pad_token_id: int = 0):
        """
        Initialize data collator.
        
        Args:
            pad_token_id: Token ID to use for padding
        """
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of data.
        
        Args:
            batch: List of data dictionaries
            
        Returns:
            Dictionary of batched tensors
        """
        # Find max length in batch
        max_length = max(item['input_ids'].shape[0] for item in batch)
        
        # Initialize batched tensors
        batched = {}
        
        # Process each key
        keys = batch[0].keys()
        
        for key in keys:
            if key == 'input_ids':
                # Pad input IDs
                padded = torch.full(
                    (len(batch), max_length),
                    self.pad_token_id,
                    dtype=torch.long
                )
                for i, item in enumerate(batch):
                    length = item['input_ids'].shape[0]
                    padded[i, :length] = item['input_ids']
                batched[key] = padded
                
            elif key == 'attention_mask':
                # Pad attention mask
                padded = torch.zeros(len(batch), max_length, dtype=torch.long)
                for i, item in enumerate(batch):
                    length = item['attention_mask'].shape[0]
                    padded[i, :length] = item['attention_mask']
                batched[key] = padded
                
            elif key == 'labels':
                # Pad labels (use -100 for ignored positions)
                padded = torch.full(
                    (len(batch), max_length),
                    -100,
                    dtype=torch.long
                )
                for i, item in enumerate(batch):
                    length = item['labels'].shape[0]
                    padded[i, :length] = item['labels']
                batched[key] = padded
                
            else:
                # Stack other tensors
                try:
                    batched[key] = torch.stack([item[key] for item in batch])
                except:
                    # Skip if cannot stack
                    pass
                    
        return batched
