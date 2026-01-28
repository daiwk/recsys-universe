"""
Reward model for RL training.
Supports multiple reward computation strategies.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Callable, Union
from dataclasses import dataclass
import re


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    # Reward weights
    format_reward_weight: float = 0.1
    accuracy_reward_weight: float = 1.0
    length_penalty_weight: float = 0.01
    repetition_penalty_weight: float = 0.1
    
    # Format checking
    require_thinking_tags: bool = True
    require_answer_tags: bool = True
    
    # Length constraints
    min_length: int = 10
    max_length: int = 2048
    
    # Repetition detection
    repetition_ngram_size: int = 3
    repetition_threshold: float = 0.3


class RewardModel:
    """
    Reward model for computing rewards from generated text.
    
    Supports:
    1. Rule-based rewards (format, length, repetition)
    2. Model-based rewards (learned reward model)
    3. Custom reward functions
    """
    
    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        model_based_reward: Optional[nn.Module] = None,
        custom_reward_fn: Optional[Callable] = None,
    ):
        self.config = config or RewardConfig()
        self.model_based_reward = model_based_reward
        self.custom_reward_fn = custom_reward_fn
        
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        ground_truth: Optional[List[str]] = None,
        tokenizer=None,
    ) -> torch.Tensor:
        """
        Compute rewards for generated responses.
        
        Args:
            prompts: List of prompt strings
            responses: List of response strings
            ground_truth: Optional ground truth answers
            tokenizer: Tokenizer for model-based rewards
            
        Returns:
            Rewards tensor [batch_size]
        """
        batch_size = len(responses)
        rewards = torch.zeros(batch_size)
        
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            reward = self._compute_single_reward(
                prompt, response, 
                ground_truth[i] if ground_truth else None
            )
            rewards[i] = reward
            
        # Add model-based rewards if available
        if self.model_based_reward is not None and tokenizer is not None:
            model_rewards = self._compute_model_rewards(
                prompts, responses, tokenizer
            )
            rewards = rewards + model_rewards
            
        # Add custom rewards if available
        if self.custom_reward_fn is not None:
            custom_rewards = self.custom_reward_fn(prompts, responses, ground_truth)
            rewards = rewards + custom_rewards
            
        return rewards
    
    def _compute_single_reward(
        self,
        prompt: str,
        response: str,
        ground_truth: Optional[str] = None,
    ) -> float:
        """Compute reward for a single response."""
        reward = 0.0
        
        # Format reward
        reward += self.config.format_reward_weight * self._compute_format_reward(response)
        
        # Length penalty
        reward -= self.config.length_penalty_weight * self._compute_length_penalty(response)
        
        # Repetition penalty
        reward -= self.config.repetition_penalty_weight * self._compute_repetition_penalty(response)
        
        # Accuracy reward (if ground truth available)
        if ground_truth is not None:
            reward += self.config.accuracy_reward_weight * self._compute_accuracy_reward(
                response, ground_truth
            )
            
        return reward
    
    def _compute_format_reward(self, response: str) -> float:
        """Check if response follows required format."""
        reward = 0.0
        
        if self.config.require_thinking_tags:
            # Check for <think>...</think> tags
            if re.search(r'<think>.*?</think>', response, re.DOTALL):
                reward += 0.5
                
        if self.config.require_answer_tags:
            # Check for <answer>...</answer> tags
            if re.search(r'<answer>.*?</answer>', response, re.DOTALL):
                reward += 0.5
                
        return reward
    
    def _compute_length_penalty(self, response: str) -> float:
        """Compute penalty for responses that are too short or too long."""
        length = len(response.split())
        
        if length < self.config.min_length:
            return (self.config.min_length - length) / self.config.min_length
        elif length > self.config.max_length:
            return (length - self.config.max_length) / self.config.max_length
        else:
            return 0.0
    
    def _compute_repetition_penalty(self, response: str) -> float:
        """Detect and penalize repetitive text."""
        words = response.split()
        if len(words) < self.config.repetition_ngram_size:
            return 0.0
            
        # Count n-grams
        ngrams = {}
        total_ngrams = 0
        for i in range(len(words) - self.config.repetition_ngram_size + 1):
            ngram = tuple(words[i:i + self.config.repetition_ngram_size])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
            total_ngrams += 1
            
        # Calculate repetition ratio
        if total_ngrams == 0:
            return 0.0
            
        repeated_ngrams = sum(1 for count in ngrams.values() if count > 1)
        repetition_ratio = repeated_ngrams / len(ngrams) if ngrams else 0
        
        if repetition_ratio > self.config.repetition_threshold:
            return repetition_ratio
        return 0.0
    
    def _compute_accuracy_reward(self, response: str, ground_truth: str) -> float:
        """
        Compute accuracy reward by comparing response to ground truth.
        
        This is a simple implementation. For more complex tasks,
        consider using a learned reward model or task-specific metrics.
        """
        # Extract answer from response (if using tags)
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            extracted_answer = answer_match.group(1).strip()
        else:
            extracted_answer = response.strip()
            
        ground_truth = ground_truth.strip()
        
        # Exact match
        if extracted_answer.lower() == ground_truth.lower():
            return 1.0
            
        # Partial match (contains)
        if ground_truth.lower() in extracted_answer.lower():
            return 0.5
            
        # Numeric comparison (for math problems)
        try:
            pred_num = float(extracted_answer)
            true_num = float(ground_truth)
            if abs(pred_num - true_num) < 1e-3:
                return 1.0
        except (ValueError, TypeError):
            pass
            
        return 0.0
    
    def _compute_model_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        tokenizer,
    ) -> torch.Tensor:
        """Compute rewards using a learned reward model."""
        # Concatenate prompts and responses
        full_texts = [f"{p}{r}" for p, r in zip(prompts, responses)]
        
        # Tokenize
        inputs = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Get reward scores
        with torch.no_grad():
            outputs = self.model_based_reward(**inputs)
            # Assume reward model outputs a scalar score
            if hasattr(outputs, 'logits'):
                rewards = outputs.logits.squeeze(-1)
            else:
                rewards = outputs.squeeze(-1)
                
        return rewards.cpu()


class MathRewardModel(RewardModel):
    """Specialized reward model for math problems."""
    
    def _compute_accuracy_reward(self, response: str, ground_truth: str) -> float:
        """Compute accuracy reward for math problems."""
        # Extract answer from response
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            extracted_answer = answer_match.group(1).strip()
        else:
            # Try to find the last number in the response
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if numbers:
                extracted_answer = numbers[-1]
            else:
                extracted_answer = response.strip()
        
        ground_truth = ground_truth.strip()
        
        # Try numeric comparison
        try:
            pred_num = float(extracted_answer.replace(',', ''))
            true_num = float(ground_truth.replace(',', ''))
            if abs(pred_num - true_num) < 1e-3:
                return 1.0
        except (ValueError, TypeError):
            pass
            
        # String match fallback
        if extracted_answer.lower() == ground_truth.lower():
            return 1.0
            
        return 0.0


class CodeRewardModel(RewardModel):
    """Specialized reward model for code generation tasks."""
    
    def __init__(self, config: Optional[RewardConfig] = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.config.require_thinking_tags = False
        self.config.require_answer_tags = False
        
    def _compute_format_reward(self, response: str) -> float:
        """Check code formatting."""
        reward = 0.0
        
        # Check for code blocks
        if '```' in response:
            reward += 0.3
            
        # Check for function/class definitions
        if 'def ' in response or 'class ' in response:
            reward += 0.3
            
        # Check for proper indentation
        lines = response.split('\n')
        has_indentation = any(line.startswith('    ') or line.startswith('\t') for line in lines)
        if has_indentation:
            reward += 0.4
            
        return reward
    
    def _compute_accuracy_reward(self, response: str, ground_truth: str) -> float:
        """Compute accuracy reward for code."""
        # Extract code from markdown blocks
        code_match = re.search(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1).strip()
        else:
            extracted_code = response.strip()
            
        ground_truth_code = ground_truth.strip()
        
        # Normalize whitespace
        extracted_code = '\n'.join(line.strip() for line in extracted_code.split('\n'))
        ground_truth_code = '\n'.join(line.strip() for line in ground_truth_code.split('\n'))
        
        # Exact match
        if extracted_code == ground_truth_code:
            return 1.0
            
        # AST-based comparison could be added here for more robust matching
        
        return 0.0
