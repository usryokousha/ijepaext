import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, MemEffAttention
from typing import Optional

class MultiAxisAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qk_scale: Optional[float] = None,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_class: nn.Module = Attention,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.grid_attn = attn_class(
            dim // 2, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop,
            proj_bias=proj_bias
        )

        self.patch_attn = attn_class(
            dim // 2, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop,
            proj_bias=proj_bias
        )
        self._patch_num = None
        self._grid_num = None

    def _patch_attn(self, x: torch.Tensor):
        B, G, P, C = x.shape
        x = x.reshape(B * G, P, C)
        x = self.patch_attn(x)
        return x.reshape(B, G, P, C)
    
    def _grid_attn(self, x: torch.Tensor):
        B, G, P, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * P, G, C)
        x = self.grid_attn(x)
        return x.reshape(B, P, G, C).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor):
        """
        Splits the input tensor into grid and patch tensors, 
        applies attention to each, and concatenates the results.
        """
        x_grid, x_patch = torch.chunk(x, 2, dim=-1)
        x_grid = self._grid_attn(x_grid)
        x_patch = self._patch_attn(x_patch)
        return torch.cat([x_grid, x_patch], dim=-1)

def block_images(inputs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Converts the image to blocked patches."""
    # inputs: (batch_size, height, width, channels)
    _, height, width, channel_dim = inputs.shape
    patch_length = patch_size**2

    outputs = inputs.permute(0, 3, 1, 2)
    outputs = outputs.reshape(-1, channel_dim, height // patch_size, patch_size, width // patch_size, patch_size)
    outputs = outputs.permute(0, 2, 4, 1, 3, 5)
    outputs = outputs.reshape(-1, height * width // patch_length, patch_length, channel_dim)
    # outputs: (batch_size, grid_h * grid_w, patch_h * patch_w, channels)
    return outputs


def unblock_images(inputs: torch.Tensor, grid_size: int, patch_size: int) -> torch.Tensor:
    """Converts blocked patches to the image."""
    # inputs: (batch_size, grid_h * grid_w, patch_h * patch_w, channels)
    grid_width = grid_size
    grid_height = inputs.shape[1] // grid_width
    channel_dim = inputs.shape[3]

    outputs = inputs.reshape(-1, grid_height, grid_width, patch_size, patch_size, channel_dim)
    outputs = outputs.permute(0, 1, 3, 2, 4, 5)
    outputs = outputs.reshape(-1, grid_height * patch_size, grid_width * patch_size, channel_dim)
    # outputs: (batch_size, height, width, channels)
    return outputs