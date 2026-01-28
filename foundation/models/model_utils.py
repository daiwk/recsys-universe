"""
Model utilities for loading and configuring Qwen3 models.
"""

from typing import Optional, Dict, Any
import torch
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for Qwen3 models."""
    model_name: str = "Qwen/Qwen3-7B"
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: Optional[list] = None
    
    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Generation config
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


def get_model_config(config_dict: Optional[Dict[str, Any]] = None) -> ModelConfig:
    """Create ModelConfig from dictionary."""
    if config_dict is None:
        return ModelConfig()
    return ModelConfig(**config_dict)


def load_model_with_lora(
    model_class,
    config: ModelConfig,
    is_trainable: bool = True
):
    """
    Load a model with optional LoRA configuration.
    
    Args:
        model_class: The model class to instantiate
        config: ModelConfig instance
        is_trainable: Whether the model needs gradients
        
    Returns:
        Loaded model (potentially with LoRA adapters)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        raise ImportError(
            "Please install transformers and peft: "
            "pip install transformers peft"
        ) from e
    
    # Parse torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)
    
    # Setup quantization config if needed
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif config.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=config.device_map if not quantization_config else None,
        torch_dtype=torch_dtype,
        trust_remote_code=config.trust_remote_code,
        quantization_config=quantization_config,
    )
    
    # Apply LoRA if enabled
    if config.use_lora and is_trainable and not (config.load_in_4bit or config.load_in_8bit):
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_generation_config(config: ModelConfig):
    """Get generation configuration from ModelConfig."""
    from transformers import GenerationConfig
    
    return GenerationConfig(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        do_sample=config.do_sample,
        pad_token_id=None,
        eos_token_id=None,
    )


def merge_lora_weights(model):
    """Merge LoRA weights into base model for inference."""
    if hasattr(model, 'merge_and_unload'):
        return model.merge_and_unload()
    return model
