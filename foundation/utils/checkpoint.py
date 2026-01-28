"""
Checkpoint management utilities.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch


class CheckpointManager:
    """Manager for saving and loading checkpoints."""
    
    def __init__(
        self,
        output_dir: str,
        save_total_limit: int = 3,
        save_safetensors: bool = True,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory to save checkpoints
            save_total_limit: Maximum number of checkpoints to keep
            save_safetensors: Whether to use safetensors format
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit
        self.save_safetensors = save_safetensors
        
        # Track saved checkpoints
        self.checkpoints: List[Path] = []
        self._load_existing_checkpoints()
    
    def _load_existing_checkpoints(self):
        """Load list of existing checkpoints."""
        if not self.output_dir.exists():
            return
            
        for checkpoint_dir in sorted(self.output_dir.iterdir()):
            if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith('checkpoint-'):
                self.checkpoints.append(checkpoint_dir)
    
    def save_checkpoint(
        self,
        step: int,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        scheduler_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            step: Training step
            model_state: Model state dictionary
            optimizer_state: Optional optimizer state
            scheduler_state: Optional scheduler state
            metadata: Optional metadata
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        if self.save_safetensors:
            try:
                from safetensors.torch import save_file
                save_file(model_state, checkpoint_dir / "model.safetensors")
            except ImportError:
                torch.save(model_state, checkpoint_dir / "pytorch_model.bin")
        else:
            torch.save(model_state, checkpoint_dir / "pytorch_model.bin")
        
        # Save optimizer state
        if optimizer_state is not None:
            torch.save(optimizer_state, checkpoint_dir / "optimizer.pt")
        
        # Save scheduler state
        if scheduler_state is not None:
            torch.save(scheduler_state, checkpoint_dir / "scheduler.pt")
        
        # Save metadata
        if metadata is not None:
            with open(checkpoint_dir / "trainer_state.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Track checkpoint
        self.checkpoints.append(checkpoint_dir)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_dir
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (uses latest if None)
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            
        Returns:
            Dictionary containing loaded states
        """
        if checkpoint_path is None:
            # Use latest checkpoint
            if not self.checkpoints:
                raise ValueError("No checkpoints found")
            checkpoint_dir = self.checkpoints[-1]
        else:
            checkpoint_dir = Path(checkpoint_path)
        
        result = {}
        
        # Load model state
        safetensors_path = checkpoint_dir / "model.safetensors"
        pytorch_path = checkpoint_dir / "pytorch_model.bin"
        
        if safetensors_path.exists():
            try:
                from safetensors.torch import load_file
                result['model_state'] = load_file(safetensors_path)
            except ImportError:
                result['model_state'] = torch.load(pytorch_path, map_location='cpu')
        elif pytorch_path.exists():
            result['model_state'] = torch.load(pytorch_path, map_location='cpu')
        else:
            raise FileNotFoundError(f"No model file found in {checkpoint_dir}")
        
        # Load optimizer state
        if load_optimizer:
            optimizer_path = checkpoint_dir / "optimizer.pt"
            if optimizer_path.exists():
                result['optimizer_state'] = torch.load(optimizer_path, map_location='cpu')
        
        # Load scheduler state
        if load_scheduler:
            scheduler_path = checkpoint_dir / "scheduler.pt"
            if scheduler_path.exists():
                result['scheduler_state'] = torch.load(scheduler_path, map_location='cpu')
        
        # Load metadata
        metadata_path = checkpoint_dir / "trainer_state.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                result['metadata'] = json.load(f)
        
        return result
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if exceeding limit."""
        while len(self.checkpoints) > self.save_total_limit:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                shutil.rmtree(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the latest checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None
    
    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints."""
        return self.checkpoints.copy()
