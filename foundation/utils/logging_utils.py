"""
Logging utilities for Foundation RL framework.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup a logger with console and optional file handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        format_string: Optional custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name. Creates a basic logger if not exists.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Setup basic handler if none exists
    if not logger.handlers:
        setup_logger(name)
    
    return logger


class TensorBoardLogger:
    """Simple TensorBoard logger wrapper."""
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = log_dir
        self.writer = None
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            print("Warning: tensorboard not available, logging to console only")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
        else:
            print(f"Step {step}: {tag} = {value}")
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars."""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        else:
            print(f"Step {step}: {main_tag} = {tag_scalar_dict}")
    
    def close(self):
        """Close the writer."""
        if self.writer:
            self.writer.close()


class WandbLogger:
    """Weights & Biases logger wrapper."""
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            name: Run name
            config: Configuration dictionary
        """
        self.project = project
        self.name = name
        
        try:
            import wandb
            self.wandb = wandb
            self.wandb.init(project=project, name=name, config=config)
            self.enabled = True
        except ImportError:
            print("Warning: wandb not available")
            self.enabled = False
    
    def log(self, data: dict, step: Optional[int] = None):
        """Log data to W&B."""
        if self.enabled:
            self.wandb.log(data, step=step)
    
    def finish(self):
        """Finish the run."""
        if self.enabled:
            self.wandb.finish()
