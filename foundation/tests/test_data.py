"""
Unit tests for data components.
"""

import pytest
import torch
from unittest.mock import Mock

from foundation.data.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from foundation.data.rollout_generator import RolloutGenerator
from foundation.data.data_collator import RLDataCollator


class TestReplayBuffer:
    """Tests for ReplayBuffer."""
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        buffer = ReplayBuffer(capacity=100)
        assert buffer.capacity == 100
        assert len(buffer) == 0
        
    def test_add_and_sample(self):
        """Test adding and sampling from buffer."""
        buffer = ReplayBuffer(capacity=100)
        
        # Add items
        for i in range(10):
            buffer.add({
                'state': torch.randn(10),
                'action': torch.randint(0, 10, (1,)),
                'reward': torch.tensor([float(i)])
            })
        
        assert len(buffer) == 10
        
        # Sample
        batch = buffer.sample(5)
        assert len(batch) == 5
        
    def test_buffer_overflow(self):
        """Test buffer overflow handling."""
        buffer = ReplayBuffer(capacity=5)
        
        # Add more items than capacity
        for i in range(10):
            buffer.add({'idx': i})
        
        assert len(buffer) == 5
        
    def test_clear(self):
        """Test clearing the buffer."""
        buffer = ReplayBuffer(capacity=100)
        buffer.add({'data': 1})
        buffer.add({'data': 2})
        
        buffer.clear()
        assert len(buffer) == 0


class TestPrioritizedReplayBuffer:
    """Tests for PrioritizedReplayBuffer."""
    
    def test_buffer_initialization(self):
        """Test prioritized buffer initialization."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
        assert buffer.capacity == 100
        assert buffer.alpha == 0.6
        assert len(buffer) == 0
        
    def test_add_with_priority(self):
        """Test adding items with priority."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        buffer.add({
            'state': torch.randn(10),
            'reward': torch.tensor([1.0])
        }, priority=1.0)
        
        buffer.add({
            'state': torch.randn(10),
            'reward': torch.tensor([2.0])
        }, priority=2.0)
        
        assert len(buffer) == 2
        
    def test_update_priorities(self):
        """Test updating priorities."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        buffer.add({'data': 1}, priority=1.0)
        buffer.add({'data': 2}, priority=1.0)
        
        # Update priority
        buffer.update_priorities([0], [5.0])
        
        # Higher priority item should be sampled more often
        samples = buffer.sample(100)
        count_item_1 = sum(1 for s in samples if s['data'] == 1)
        # With high priority, item 1 should be sampled more
        assert count_item_1 > 30  # Rough check


class TestRolloutGenerator:
    """Tests for RolloutGenerator."""
    
    def test_generator_initialization(self):
        """Test rollout generator initialization."""
        mock_policy = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_policy.tokenizer = mock_tokenizer
        
        generator = RolloutGenerator(
            policy_model=mock_policy,
            max_new_tokens=100
        )
        
        assert generator.policy_model == mock_policy
        assert generator.max_new_tokens == 100
        
    def test_generate_rollouts(self):
        """Test generating rollouts."""
        mock_policy = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_policy.tokenizer = mock_tokenizer
        
        # Mock generate method
        mock_output = Mock()
        mock_output.sequences = torch.randint(0, 1000, (1, 20))
        mock_output.log_probs = torch.randn(1, 19)
        mock_policy.generate.return_value = mock_output
        
        generator = RolloutGenerator(
            policy_model=mock_policy,
            max_new_tokens=50
        )
        
        prompts = ["prompt 1", "prompt 2"]
        rollouts = generator.generate(prompts, num_samples=1)
        
        assert len(rollouts) == 2
        assert mock_policy.generate.called


class TestRLDataCollator:
    """Tests for RLDataCollator."""
    
    def test_collator_initialization(self):
        """Test data collator initialization."""
        collator = RLDataCollator(pad_token_id=0)
        assert collator.pad_token_id == 0
        
    def test_collate_sequences(self):
        """Test collating sequences of different lengths."""
        collator = RLDataCollator(pad_token_id=0)
        
        batch = [
            {'input_ids': torch.tensor([1, 2, 3]), 'attention_mask': torch.tensor([1, 1, 1])},
            {'input_ids': torch.tensor([4, 5]), 'attention_mask': torch.tensor([1, 1])},
        ]
        
        collated = collator(batch)
        
        assert 'input_ids' in collated
        assert 'attention_mask' in collated
        # Should be padded to max length
        assert collated['input_ids'].shape[1] == 3
        
    def test_collate_with_labels(self):
        """Test collating with labels."""
        collator = RLDataCollator(pad_token_id=0)
        
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1])
            },
            {
                'input_ids': torch.tensor([4, 5]),
                'labels': torch.tensor([4, 5]),
                'attention_mask': torch.tensor([1, 1])
            },
        ]
        
        collated = collator(batch)
        
        assert 'labels' in collated
        assert collated['labels'].shape == collated['input_ids'].shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
