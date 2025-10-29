"""
Model architecture wrapper for Flow Matching.
Provides consistent interface for flow-based models.
"""
import torch
import torch.nn as nn
from models.unet import SimpleUNet


class FlowUNet(SimpleUNet):
    """
    U-Net adapted for Flow Matching.
    
    Key differences from standard diffusion U-Net:
    - Time embedding represents continuous time t ∈ [0, 1]
    - Model predicts vector field v(x_t, t) instead of noise ε
    - No noise schedule required
    """
    
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        cond_dim=6,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_dim=256,
    ):
        """
        Initialize FlowUNet.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels (usually same as input)
            cond_dim: Conditioning dimension
            base_channels: Base channel count
            channel_mults: Channel multipliers per layer
            time_dim: Time embedding dimension
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_dim=cond_dim,
            base_channels=base_channels,
            channel_mults=channel_mults,
            time_dim=time_dim
        )
        
        # Initialize output layer with smaller weights for stability
        self._init_output_layer()
    
    def _init_output_layer(self):
        """Initialize the final output layer with small weights."""
        # final_conv is a Sequential with [Conv, ReLU, Conv]
        # Initialize the last Conv2d layer
        final_layer = self.final_conv[-1]
        nn.init.normal_(final_layer.weight, std=0.01)
        if final_layer.bias is not None:
            nn.init.zeros_(final_layer.bias)
    
    def forward(self, x, t, cond=None):
        """
        Forward pass for Flow Matching.
        
        Args:
            x: Input tensor x_t at time t [batch, channels, height, width]
            t: Time values t ∈ [0, 1] [batch]
            cond: Conditioning [batch, cond_dim] or None
            
        Returns:
            Predicted vector field v(x_t, t) [batch, channels, height, width]
        """
        # Handle unconditional case (for CFG)
        if cond is None:
            batch_size = x.shape[0]
            cond = torch.zeros(batch_size, self.cond_dim, device=x.device)
        
        # Use parent forward method
        # The time embedding in SimpleUNet handles t values correctly
        return super().forward(x, t, cond)
    
    def predict_velocity(self, x_t, t, cond=None):
        """
        Predict the velocity field (vector field) at time t.
        
        This is the main method for Flow Matching.
        
        Args:
            x_t: Current state at time t
            t: Current time
            cond: Conditioning
            
        Returns:
            Velocity field v(x_t, t)
        """
        return self.forward(x_t, t, cond)


def create_flow_model(config):
    """
    Create a Flow Matching model from config.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        FlowUNet model
    """
    return FlowUNet(
        in_channels=config.get('in_channels', 1),
        out_channels=config.get('out_channels', 1),
        cond_dim=config.get('cond_dim', 6),
        base_channels=config.get('base_channels', 64),
        channel_mults=tuple(config.get('channel_mults', [1, 2, 4, 8])),
        time_dim=config.get('time_dim', 256)
    )





