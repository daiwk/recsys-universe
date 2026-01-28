"""
Unit tests for Foundation RL models.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch

from foundation.models.model_utils import ModelConfig, get_model_config
from foundation.models.qwen3_policy import Qwen3PolicyModel, PolicyOutput
from foundation.models.qwen3_critic import Qwen3CriticModel, CriticOutput


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ModelConfig()
        assert config.model_name == "Qwen/Qwen3-7B"
        assert config.device_map == "auto"
        assert config.torch_dtype == "bfloat16"
        assert config.use_lora is True
        assert config.lora_r == 64
        assert config.lora_alpha == 16
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            model_name="Qwen/Qwen3-1.8B",
            lora_r=32,
            load_in_4bit=True
        )
        assert config.model_name == "Qwen/Qwen3-1.8B"
        assert config.lora_r == 32
        assert config.load_in_4bit is True
        
    def test_lora_target_modules_default(self):
        """Test default LoRA target modules."""
        config = ModelConfig()
        expected_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        assert config.lora_target_modules == expected_modules


class TestGetModelConfig:
    """Tests for get_model_config function."""
    
    def test_get_default_config(self):
        """Test getting default config."""
        config = get_model_config()
        assert isinstance(config, ModelConfig)
        
    def test_get_config_from_dict(self):
        """Test getting config from dictionary."""
        config_dict = {
            "model_name": "Qwen/Qwen3-1.8B",
            "lora_r": 32
        }
        config = get_model_config(config_dict)
        assert config.model_name == "Qwen/Qwen3-1.8B"
        assert config.lora_r == 32


class TestPolicyOutput:
    """Tests for PolicyOutput dataclass."""
    
    def test_policy_output_creation(self):
        """Test creating PolicyOutput."""
        sequences = torch.randint(0, 1000, (2, 10))
        log_probs = torch.randn(2, 9)
        
        output = PolicyOutput(
            sequences=sequences,
            log_probs=log_probs
        )
        
        assert torch.equal(output.sequences, sequences)
        assert torch.equal(output.log_probs, log_probs)
        assert output.logits is None
        assert output.values is None


class TestCriticOutput:
    """Tests for CriticOutput dataclass."""
    
    def test_critic_output_creation(self):
        """Test creating CriticOutput."""
        values = torch.randn(2, 10)
        
        output = CriticOutput(values=values)
        
        assert torch.equal(output.values, values)
        assert output.last_hidden_state is None


class TestQwen3PolicyModelMock:
    """Tests for Qwen3PolicyModel using mocks."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock policy model."""
        with patch('foundation.models.model_utils.load_model_with_lora') as mock_load:
            # Mock base model
            mock_base = Mock()
            mock_base.config.hidden_size = 4096
            mock_base.generate.return_value = torch.randint(0, 1000, (1, 20))
            
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer.eos_token_id = 1
            
            mock_load.return_value = (mock_base, mock_tokenizer)
            
            config = ModelConfig(model_name="test-model")
            model = Qwen3PolicyModel(config=config)
            yield model
    
    def test_model_initialization(self, mock_model):
        """Test model initialization."""
        assert mock_model.model_name == "test-model"
        assert mock_model.config is not None
        
    def test_forward_pass(self, mock_model):
        """Test forward pass."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        # Mock the base model forward
        mock_output = Mock()
        mock_output.logits = torch.randn(2, 10, 1000)
        mock_model.base_model.return_value = mock_output
        
        output = mock_model.forward(input_ids, attention_mask)
        assert output is not None


class TestQwen3CriticModelMock:
    """Tests for Qwen3CriticModel using mocks."""
    
    @pytest.fixture
    def mock_critic(self):
        """Create a mock critic model."""
        with patch('foundation.models.model_utils.load_model_with_lora') as mock_load:
            # Mock base model
            mock_base = Mock()
            mock_base.config.hidden_size = 4096
            
            # Create mock hidden states output
            mock_output = Mock()
            mock_output.hidden_states = [torch.randn(2, 10, 4096) for _ in range(3)]
            mock_base.return_value = mock_output
            
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token_id = 0
            
            mock_load.return_value = (mock_base, mock_tokenizer)
            
            config = ModelConfig(model_name="test-model")
            critic = Qwen3CriticModel(config=config)
            yield critic
    
    def test_critic_initialization(self, mock_critic):
        """Test critic initialization."""
        assert mock_critic.model_name == "test-model"
        assert mock_critic.hidden_size == 4096
        assert mock_critic.value_head is not None
        
    def test_forward_pass(self, mock_critic):
        """Test forward pass."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        output = mock_critic.forward(input_ids, attention_mask)
        
        assert isinstance(output, CriticOutput)
        assert output.values.shape[0] == 2  # batch size


class TestModelUtils:
    """Tests for model utilities."""
    
    def test_get_generation_config(self):
        """Test generation config creation."""
        from foundation.models.model_utils import get_generation_config
        
        config = ModelConfig(
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.95
        )
        
        gen_config = get_generation_config(config)
        assert gen_config.max_new_tokens == 256
        assert gen_config.temperature == 0.8
        assert gen_config.top_p == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
