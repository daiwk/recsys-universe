"""
Configuration management for Foundation RL framework.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os


@dataclass
class RLConfig:
    """Main configuration for RL training."""
    
    # Model configuration
    model_name: str = "Qwen/Qwen3-7B"
    model_revision: Optional[str] = None
    trust_remote_code: bool = True
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None
    
    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Training configuration
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Generation configuration
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    # PPO/GRPO configuration
    ppo_clip_epsilon: float = 0.2
    value_clip: float = 0.2
    kl_coef: float = 0.01
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # GRPO specific
    group_size: int = 8
    
    # Distributed training
    use_distributed: bool = False
    num_actors: int = 4
    num_learners: int = 1
    actor_steps: int = 10
    learner_steps: int = 1
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # Checkpoint
    resume_from_checkpoint: Optional[str] = None
    save_total_limit: int = 3
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RLConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


def get_rl_config(config_path: Optional[str] = None) -> RLConfig:
    """
    Get RL configuration.
    
    Args:
        config_path: Path to config file (JSON or YAML)
        
    Returns:
        RLConfig instance
    """
    config = RLConfig()
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        if config_path.endswith('.json'):
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = RLConfig.from_dict(config_dict)
        elif config_path.endswith(('.yaml', '.yml')):
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                config = RLConfig.from_dict(config_dict)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files")
    
    # Override with environment variables
    if os.environ.get('RL_MODEL_NAME'):
        config.model_name = os.environ['RL_MODEL_NAME']
    if os.environ.get('RL_OUTPUT_DIR'):
        config.output_dir = os.environ['RL_OUTPUT_DIR']
    if os.environ.get('RL_LEARNING_RATE'):
        config.learning_rate = float(os.environ['RL_LEARNING_RATE'])
    if os.environ.get('RL_USE_DISTRIBUTED'):
        config.use_distributed = os.environ['RL_USE_DISTRIBUTED'].lower() == 'true'
    
    return config
