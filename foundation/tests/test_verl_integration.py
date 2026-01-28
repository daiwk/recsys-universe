"""
Unit tests for verl integration.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from foundation.rl.verl_integration import (
    VerlTrainerConfig,
    VerlDataProtoAdapter,
    VerlWorker,
    VerlWorkerGroup,
    is_verl_available,
    get_verl_version,
)


class TestVerlAvailability:
    """Tests for verl availability checking."""
    
    def test_is_verl_available(self):
        """Test checking if verl is available."""
        # This will return False if verl is not installed
        available = is_verl_available()
        assert isinstance(available, bool)
        
    def test_get_verl_version(self):
        """Test getting verl version."""
        version = get_verl_version()
        # Should return None if verl is not available
        assert version is None or isinstance(version, str)


class TestVerlTrainerConfig:
    """Tests for VerlTrainerConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = VerlTrainerConfig()
        
        assert config.model_name == "Qwen/Qwen3-7B"
        assert config.learning_rate == 1e-6
        assert config.batch_size == 4
        assert config.clip_ratio == 0.2
        assert config.entropy_coeff == 0.01
        assert config.num_gpus == 1
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = VerlTrainerConfig(
            model_name="Qwen/Qwen3-1.8B",
            learning_rate=5e-6,
            batch_size=8,
            num_gpus=4
        )
        
        assert config.model_name == "Qwen/Qwen3-1.8B"
        assert config.learning_rate == 5e-6
        assert config.batch_size == 8
        assert config.num_gpus == 4


class TestVerlDataProtoAdapter:
    """Tests for VerlDataProtoAdapter."""
    
    @pytest.fixture
    def adapter(self):
        """Create a DataProto adapter."""
        return VerlDataProtoAdapter()
    
    def test_create_batch(self, adapter):
        """Test creating a batch."""
        input_ids = torch.randint(0, 1000, (4, 20))
        attention_mask = torch.ones(4, 20)
        
        batch = adapter.create_batch(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert torch.equal(batch["input_ids"], input_ids)
        assert torch.equal(batch["attention_mask"], attention_mask)
        
    def test_create_batch_with_labels(self, adapter):
        """Test creating a batch with labels."""
        input_ids = torch.randint(0, 1000, (4, 20))
        attention_mask = torch.ones(4, 20)
        labels = torch.randint(0, 1000, (4, 20))
        
        batch = adapter.create_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        assert "labels" in batch
        assert torch.equal(batch["labels"], labels)
        
    def test_create_batch_with_metadata(self, adapter):
        """Test creating a batch with metadata."""
        input_ids = torch.randint(0, 1000, (4, 20))
        attention_mask = torch.ones(4, 20)
        metadata = {"task": "test", "batch_id": 1}
        
        batch = adapter.create_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            metadata=metadata
        )
        
        assert "metadata" in batch
        assert batch["metadata"] == metadata
        
    def test_split_batch(self, adapter):
        """Test splitting a batch."""
        batch = {
            "input_ids": torch.randint(0, 1000, (8, 20)),
            "attention_mask": torch.ones(8, 20),
        }
        
        splits = adapter.split_batch(batch, num_splits=4)
        
        assert len(splits) == 4
        # Each split should have batch size 2 (8 // 4)
        for split in splits:
            assert split["input_ids"].shape[0] == 2
            
    def test_split_batch_uneven(self, adapter):
        """Test splitting a batch with uneven sizes."""
        batch = {
            "input_ids": torch.randint(0, 1000, (7, 20)),
            "attention_mask": torch.ones(7, 20),
        }
        
        splits = adapter.split_batch(batch, num_splits=3)
        
        assert len(splits) == 3
        # Last split should have the remaining samples
        total_samples = sum(split["input_ids"].shape[0] for split in splits)
        assert total_samples == 7


class TestVerlWorker:
    """Tests for VerlWorker."""
    
    @pytest.fixture
    def worker(self):
        """Create a VerlWorker."""
        return VerlWorker(worker_id=0)
    
    def test_worker_initialization(self, worker):
        """Test worker initialization."""
        assert worker.worker_id == 0
        
    def test_compute_log_probs(self, worker):
        """Test computing log probabilities."""
        batch = {
            "input_ids": torch.randint(0, 1000, (4, 20)),
            "attention_mask": torch.ones(4, 20),
        }
        
        log_probs = worker.compute_log_probs(batch)
        
        # Should return tensor of shape (batch_size, seq_len - 1)
        assert log_probs.shape[0] == 4
        assert log_probs.shape[1] == 19  # 20 - 1
        
    def test_generate(self, worker):
        """Test generation."""
        batch = {
            "input_ids": torch.randint(0, 1000, (4, 10)),
            "attention_mask": torch.ones(4, 10),
        }
        
        generated = worker.generate(batch, max_new_tokens=20)
        
        # Should return tensor of shape (batch_size, max_new_tokens)
        assert generated.shape[0] == 4
        assert generated.shape[1] == 20


class TestVerlWorkerGroup:
    """Tests for VerlWorkerGroup."""
    
    @pytest.fixture
    def worker_group(self):
        """Create a VerlWorkerGroup."""
        return VerlWorkerGroup(num_workers=4)
    
    def test_worker_group_initialization(self, worker_group):
        """Test worker group initialization."""
        assert worker_group.num_workers == 4
        assert len(worker_group.workers) == 0
        
    def test_initialize_workers(self, worker_group):
        """Test initializing workers."""
        worker_group.initialize()
        
        assert len(worker_group.workers) == 4
        for i, worker in enumerate(worker_group.workers):
            assert worker.worker_id == i
            
    def test_execute_parallel(self, worker_group):
        """Test parallel execution."""
        worker_group.initialize()
        
        # Create batches for each worker
        batches = [
            {"input_ids": torch.randint(0, 1000, (2, 10))}
            for _ in range(4)
        ]
        
        results = worker_group.execute_parallel("compute_log_probs", batches)
        
        assert len(results) == 4
        
    def test_shutdown(self, worker_group):
        """Test shutting down worker group."""
        worker_group.initialize()
        worker_group.shutdown()
        
        assert len(worker_group.workers) == 0


class TestVerlTrainerWrapper:
    """Tests for VerlTrainerWrapper."""
    
    def test_trainer_requires_verl(self):
        """Test that trainer requires verl to be installed."""
        from foundation.rl.verl_integration import VerlTrainerWrapper, VerlTrainerConfig
        
        # If verl is not available, should raise ImportError
        if not is_verl_available():
            with pytest.raises(ImportError):
                config = VerlTrainerConfig()
                VerlTrainerWrapper(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
