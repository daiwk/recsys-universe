"""
Qwen3-7b Policy Model for RL training.
Implements actor-critic style policy with verl framework compatibility.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

from foundation.models.model_utils import (
    ModelConfig, 
    load_model_with_lora, 
    get_generation_config
)


@dataclass
class PolicyOutput:
    """Output from policy model."""
    sequences: torch.Tensor
    log_probs: torch.Tensor
    logits: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    
    
class Qwen3PolicyModel(nn.Module):
    """
    Qwen3-7b Policy Model for reinforcement learning.
    
    This model serves as the policy (actor) in RL training,
    generating actions (text) given observations (prompts).
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: str = "Qwen/Qwen3-7B",
    ):
        super().__init__()
        self.config = config or ModelConfig(model_name=model_name)
        self.model_name = model_name
        
        # Load model and tokenizer
        self.base_model, self.tokenizer = load_model_with_lora(
            None,  # Will load AutoModelForCausalLM internally
            self.config,
            is_trainable=True
        )
        
        # Generation config
        self.generation_config = get_generation_config(self.config)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the policy model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for computing loss [batch_size, seq_len]
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary containing logits, loss (if labels provided), etc.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
        )
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> PolicyOutput:
        """
        Generate sequences using the policy.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate per input
            
        Returns:
            PolicyOutput containing generated sequences and log probs
        """
        # Update generation config
        gen_config = self.generation_config
        if max_new_tokens is not None:
            gen_config.max_new_tokens = max_new_tokens
        if temperature is not None:
            gen_config.temperature = temperature
        if top_p is not None:
            gen_config.top_p = top_p
        gen_config.do_sample = do_sample
        gen_config.num_return_sequences = num_return_sequences
        
        # Generate
        with torch.no_grad():
            output_sequences = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Compute log probabilities
        log_probs = self._compute_log_probs(
            output_sequences,
            input_ids.shape[1]  # prompt length
        )
        
        return PolicyOutput(
            sequences=output_sequences,
            log_probs=log_probs,
            attention_mask=(output_sequences != self.tokenizer.pad_token_id).long()
        )
    
    def _compute_log_probs(
        self,
        sequences: torch.Tensor,
        prompt_length: int
    ) -> torch.Tensor:
        """
        Compute log probabilities for generated sequences.
        
        Args:
            sequences: Generated sequences [batch_size, total_seq_len]
            prompt_length: Length of the prompt
            
        Returns:
            Log probabilities [batch_size, gen_seq_len]
        """
        outputs = self.base_model(
            input_ids=sequences,
            return_dict=True,
        )
        
        logits = outputs.logits[:, :-1, :]  # [batch, seq-1, vocab]
        targets = sequences[:, 1:]  # [batch, seq-1]
        
        # Compute log probs
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=targets.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask prompt tokens (only keep generation tokens)
        mask = torch.zeros_like(token_log_probs)
        mask[:, prompt_length-1:] = 1
        token_log_probs = token_log_probs * mask
        
        return token_log_probs
    
    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log probabilities for given sequences.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Log probabilities [batch_size, seq_len-1]
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=targets.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs
    
    def save_pretrained(self, save_path: str):
        """Save model to path."""
        self.base_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
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
