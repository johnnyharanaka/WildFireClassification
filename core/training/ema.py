"""Exponential Moving Average (EMA) model for Teacher-Student learning.

This module implements EMA for creating a stable teacher model that generates
pseudo-labels (attention-based bounding boxes) for student training.

Based on techniques from:
- Mean Teacher (Tarvainen & Valpola, 2017)
- MoCo (He et al., 2020)
- BYOL (Grill et al., 2020)
"""

import copy
import torch
import torch.nn as nn


class EMAModel:
    """Exponential Moving Average model wrapper for teacher-student learning.

    Creates a shadow copy of the student model that is updated using exponential
    moving average of the student's parameters. The teacher model produces more
    stable outputs (e.g., attention maps, bounding boxes) compared to the
    actively training student.

    Usage:
        # Create EMA teacher from student
        ema = EMAModel(student_model, momentum=0.999)

        # Training loop
        for batch in dataloader:
            # Use teacher for pseudo-labels
            teacher_output = ema.get_model()(batch)

            # Train student
            student_output = student_model(batch)
            loss.backward()
            optimizer.step()

            # Update teacher
            ema.update(student_model)
    """

    def __init__(self, model: nn.Module, momentum: float = 0.999):
        """Initialize EMA model.

        Args:
            model: Student model to create shadow copy from
            momentum: EMA momentum factor (default: 0.999)
                     Higher values = slower updates, more stable teacher
                     ema_param = momentum * ema_param + (1 - momentum) * student_param

        Note:
            - Creates a deep copy of the model
            - Sets teacher to eval mode permanently
            - Disables gradients for all teacher parameters
        """
        self.momentum = momentum

        # Create deep copy of student model
        self.ema_model = copy.deepcopy(model)

        # Set to eval mode (no dropout, batchnorm in eval mode)
        self.ema_model.eval()

        # Disable gradients for EMA model (teacher doesn't train)
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters using student model.

        Formula: ema_param = momentum * ema_param + (1 - momentum) * student_param

        Args:
            model: Student model with updated parameters

        Note:
            - Called after each optimizer.step()
            - Uses torch.no_grad() to avoid tracking gradients
            - Updates all parameters including buffers (e.g., batchnorm running stats)
        """
        # Update model parameters
        for ema_param, student_param in zip(
            self.ema_model.parameters(),
            model.parameters()
        ):
            ema_param.data.mul_(self.momentum).add_(
                student_param.data,
                alpha=1 - self.momentum
            )

        # Update buffers (e.g., running_mean and running_var in BatchNorm)
        for ema_buffer, student_buffer in zip(
            self.ema_model.buffers(),
            model.buffers()
        ):
            ema_buffer.data.mul_(self.momentum).add_(
                student_buffer.data,
                alpha=1 - self.momentum
            )

    def get_model(self) -> nn.Module:
        """Get the EMA model for inference.

        Returns:
            EMA model (teacher) in eval mode

        Note:
            - Model is always in eval mode
            - Use this for generating pseudo-labels, attention maps, etc.
            - Do NOT use for training (gradients disabled)
        """
        return self.ema_model

    def state_dict(self):
        """Get EMA model state dict for checkpointing.

        Returns:
            State dict of EMA model
        """
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """Load EMA model from checkpoint.

        Args:
            state_dict: State dict to load
        """
        self.ema_model.load_state_dict(state_dict)

    def to(self, device):
        """Move EMA model to device.

        Args:
            device: torch device (cuda/mps/cpu)

        Returns:
            self for chaining
        """
        self.ema_model.to(device)
        return self
