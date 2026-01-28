"""
Qwen3-7b Critic Model for RL training.
Implements value function estimation for PPO/GRPO.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from foundation.models.model_utils import ModelConfig, load_model_with_lora


@dataclass
class CriticOutput:
    """Output from critic model."""
    values: torch.Tensor
    last_hidden_state: Optional[torch.Tensor] = None


class Qwen3CriticModel(nn.Module):
    """
    Qwen3-7b Critic Model for value estimation.
    
    This model estimates state values for advantage computation in RL.
    Uses a separate value head on top of the base model.
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: str = "Qwen/Qwen3-7B",
        value_head_dim: int = 1,
        use_separate_model: bool = True,
    ):
        super().__init__()
        self.config = config or ModelConfig(model_name=model_name)
        self.model_name = model_name
        self.value_head_dim = value_head_dim
        self.use_separate_model = use_separate_model
        
        # Load base model
        self.base_model, self.tokenizer = load_model_with_lora(
            None,
            self.config,
            is_trainable=True
        )
        
        # Get hidden size from config
        self.hidden_size = self.base_model.config.hidden_size
        
        # Value head - maps from hidden states to scalar values
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, value_head_dim),
        )
        
        # Initialize value head
        self._init_value_head()
        
    def _init_value_head(self):
        """Initialize value head with small weights."""
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_last_hidden: bool = False,
    ) -> CriticOutput:
        """
        Forward pass to compute state values.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_last_hidden: Whether to return last hidden states
            
        Returns:
            CriticOutput containing values
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Use last hidden state
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden]
        
        # Compute values for each position
        values = self.value_head(hidden_states).squeeze(-1)  # [batch, seq_len]
        
        # Apply attention mask
        if attention_mask is not None:
            values = values * attention_mask.float()
        
        return CriticOutput(
            values=values,
            last_hidden_state=hidden_states if return_last_hidden else None
        )
    
    def get_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_last_token: bool = True,
    ) -> torch.Tensor:
        """
        Get state values for the input sequences.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            use_last_token: If True, return value at last token position
                          If False, return values for all positions
                          
        Returns:
            Values [batch_size] if use_last_token else [batch_size, seq_len]
        """
        output = self.forward(input_ids, attention_mask)
        values = output.values
        
        if use_last_token:
            # Get value at the last non-padded token
            if attention_mask is not None:
                # Find last valid token for each sequence
                seq_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
                batch_size = values.shape[0]
                values = values[torch.arange(batch_size), seq_lengths]
            else:
                values = values[:, -1]
        
        return values
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards [batch_size, seq_len]
            values: Current state values [batch_size, seq_len]
            next_values: Next state values [batch_size, seq_len]
            dones: Done flags [batch_size, seq_len]
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            advantages: Advantage estimates [batch_size, seq_len]
            returns: Return estimates [batch_size, seq_len]
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        # Compute advantages backwards
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = next_values[:, t]
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values[:, t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[:, t]) * last_gae
            advantages[:, t] = last_gae
        
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def save_pretrained(self, save_path: str):
        """Save model to path."""
        self.base_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save value head separately
        value_head_path = f"{save_path}/value_head.pt"
        torch.save(self.value_head.state_dict(), value_head_path)
    
    def load_pretrained(self, load_path: str):
        """Load model from path."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            load_path,
            device_map=self.config.device_map,
            torch_dtype=self.base_model.dtype,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Load value head
        value_head_path = f"{load_path}/value_head.pt"
        try:
            state_dict = torch.load(value_head_path, map_location="cpu")
            self.value_head.load_state_dict(state_dict)
        except FileNotFoundError:
            print(f"Warning: Value head not found at {value_head_path}, using initialized weights")
