"""
Unit tests for RL trainers (PPO and GRPO).
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch

from foundation.rl.ppo_trainer import PPOTrainer, PPOConfig
from foundation.rl.grpo_trainer import GRPOTrainer, GRPOConfig
from foundation.rl.reward_model import RewardModel, RewardConfig, MathRewardModel
from foundation.rl.verl_adapter import VerlAdapter, VerlTrajectory, RLAlgorithm


class TestPPOConfig:
    """Tests for PPOConfig."""
    
    def test_default_config(self):
        """Test default PPO configuration."""
        config = PPOConfig()
        assert config.policy_lr == 1e-6
        assert config.critic_lr == 1e-5
        assert config.clip_epsilon == 0.2
        assert config.kl_coef == 0.01
        assert config.vf_coef == 0.5
        assert config.entropy_coef == 0.01
        
    def test_custom_config(self):
        """Test custom PPO configuration."""
        config = PPOConfig(
            policy_lr=5e-6,
            clip_epsilon=0.1,
            num_epochs=2
        )
        assert config.policy_lr == 5e-6
        assert config.clip_epsilon == 0.1
        assert config.num_epochs == 2


class TestGRPOConfig:
    """Tests for GRPOConfig."""
    
    def test_default_config(self):
        """Test default GRPO configuration."""
        config = GRPOConfig()
        assert config.policy_lr == 1e-6
        assert config.group_size == 8
        assert config.clip_epsilon == 0.2
        
    def test_grpo_specific_config(self):
        """Test GRPO-specific configuration."""
        config = GRPOConfig(group_size=16)
        assert config.group_size == 16


class TestRewardModel:
    """Tests for RewardModel."""
    
    def test_default_reward_config(self):
        """Test default reward configuration."""
        config = RewardConfig()
        assert config.format_reward_weight == 0.1
        assert config.accuracy_reward_weight == 1.0
        assert config.require_thinking_tags is True
        
    def test_format_reward(self):
        """Test format reward computation."""
        config = RewardConfig(require_thinking_tags=True, require_answer_tags=True)
        reward_model = RewardModel(config)
        
        # Test with proper tags
        response_with_tags = "<think>Thinking...</think><answer>42</answer>"
        format_reward = reward_model._compute_format_reward(response_with_tags)
        assert format_reward == 1.0
        
        # Test without tags
        response_without_tags = "Just a response"
        format_reward = reward_model._compute_format_reward(response_without_tags)
        assert format_reward == 0.0
        
    def test_length_penalty(self):
        """Test length penalty computation."""
        config = RewardConfig(min_length=10, max_length=100)
        reward_model = RewardModel(config)
        
        # Test normal length
        penalty = reward_model._compute_length_penalty("This is a normal length response")
        assert penalty == 0.0
        
        # Test too short
        penalty = reward_model._compute_length_penalty("Short")
        assert penalty > 0


class TestMathRewardModel:
    """Tests for MathRewardModel."""
    
    def test_numeric_answer_extraction(self):
        """Test extracting numeric answers."""
        reward_model = MathRewardModel()
        
        # Test exact match
        response = "<answer>42</answer>"
        reward = reward_model._compute_accuracy_reward(response, "42")
        assert reward == 1.0
        
        # Test numeric comparison
        response = "<answer>3.14159</answer>"
        reward = reward_model._compute_accuracy_reward(response, "3.14159")
        assert reward == 1.0
        
        # Test wrong answer
        response = "<answer>100</answer>"
        reward = reward_model._compute_accuracy_reward(response, "42")
        assert reward == 0.0


class TestVerlAdapter:
    """Tests for VerlAdapter."""
    
    @pytest.fixture
    def mock_adapter(self):
        """Create a mock verl adapter."""
        mock_policy = Mock()
        mock_policy.tokenizer = Mock()
        mock_policy.tokenizer.pad_token_id = 0
        
        adapter = VerlAdapter(
            policy_model=mock_policy,
            algorithm=RLAlgorithm.PPO
        )
        return adapter
    
    def test_adapter_initialization(self, mock_adapter):
        """Test adapter initialization."""
        assert mock_adapter.algorithm == RLAlgorithm.PPO
        assert mock_adapter.global_step == 0
        assert mock_adapter.config['algorithm'] == 'ppo'
        
    def test_group_relative_advantages(self, mock_adapter):
        """Test group-relative advantage computation."""
        # Create mock trajectories
        trajectories = []
        for i in range(8):  # group_size = 8
            traj = Mock(spec=VerlTrajectory)
            traj.rewards = torch.tensor([float(i)])
            traj.advantages = None
            traj.returns = None
            trajectories.append(traj)
        
        # Mock the method to avoid actual computation
        result = mock_adapter._compute_group_relative_advantages(trajectories, 8)
        
        # Should return trajectories
        assert len(result) == 8


class TestVerlTrajectory:
    """Tests for VerlTrajectory."""
    
    def test_trajectory_creation(self):
        """Test creating a trajectory."""
        traj = VerlTrajectory(
            input_ids=torch.randint(0, 1000, (1, 20)),
            attention_mask=torch.ones(1, 20),
            action_mask=torch.zeros(1, 20),
            old_log_probs=torch.randn(1, 19),
            rewards=torch.tensor([1.0])
        )
        
        assert traj.batch_size == 1
        assert traj.seq_length == 20
        assert traj.advantages is None
        
    def test_trajectory_to_device(self):
        """Test moving trajectory to device."""
        traj = VerlTrajectory(
            input_ids=torch.randint(0, 1000, (1, 20)),
            attention_mask=torch.ones(1, 20),
            action_mask=torch.zeros(1, 20),
            old_log_probs=torch.randn(1, 19),
            rewards=torch.tensor([1.0]),
            advantages=torch.tensor([0.5])
        )
        
        # Test moving to CPU (should work even if CUDA not available)
        traj_cpu = traj.to('cpu')
        assert traj_cpu.input_ids.device.type == 'cpu'


class TestPPOTrainerMock:
    """Tests for PPOTrainer using mocks."""
    
    @pytest.fixture
    def mock_ppo_trainer(self):
        """Create a mock PPO trainer."""
        mock_policy = Mock()
        mock_critic = Mock()
        
        config = PPOConfig(num_epochs=1, mini_batch_size=1)
        
        trainer = PPOTrainer(
            policy_model=mock_policy,
            critic_model=mock_critic,
            config=config
        )
        return trainer
    
    def test_trainer_initialization(self, mock_ppo_trainer):
        """Test trainer initialization."""
        assert mock_ppo_trainer.config.num_epochs == 1
        assert mock_ppo_trainer.global_step == 0
        assert 'policy_loss' in mock_ppo_trainer.metrics
        
    def test_compute_value_loss(self, mock_ppo_trainer):
        """Test value loss computation."""
        # Mock critic
        mock_ppo_trainer.critic_model.get_values.return_value = torch.tensor([0.5, 0.6])
        
        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'returns': torch.tensor([1.0, 0.5])
        }
        
        value_loss = mock_ppo_trainer._compute_value_loss(batch)
        assert isinstance(value_loss, torch.Tensor)


class TestGRPOTrainerMock:
    """Tests for GRPOTrainer using mocks."""
    
    @pytest.fixture
    def mock_grpo_trainer(self):
        """Create a mock GRPO trainer."""
        mock_policy = Mock()
        
        config = GRPOConfig(num_epochs=1, group_size=4)
        
        trainer = GRPOTrainer(
            policy_model=mock_policy,
            config=config
        )
        return trainer
    
    def test_trainer_initialization(self, mock_grpo_trainer):
        """Test GRPO trainer initialization."""
        assert mock_grpo_trainer.config.group_size == 4
        assert mock_grpo_trainer.config.num_epochs == 1
        assert 'group_relative_advantage' in mock_grpo_trainer.metrics
        
    def test_group_relative_advantages(self, mock_grpo_trainer):
        """Test group-relative advantage computation."""
        # Create mock trajectories
        trajectories = []
        for i in range(8):
            traj = Mock(spec=VerlTrajectory)
            traj.rewards = torch.tensor([float(i)])
            traj.advantages = None
            traj.returns = None
            trajectories.append(traj)
        
        # Group into 2 groups of 4
        grouped = [trajectories[:4], trajectories[4:]]
        
        result = mock_grpo_trainer._compute_group_relative_advantages(
            trajectories, grouped
        )
        
        assert len(result) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
