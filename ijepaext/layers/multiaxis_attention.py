import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, CrossAttention, MemEffAttention, MemEffCrossAttention
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

    def _separate_patch(self, x):
        """Input tensor of shape [batch, grid, patch, dim]"""
        B, G, P, C = x.shape
        self._grid_num = G
        x = x.reshape(B * G, P, C)
        return x
    
    def _separate_grid(self, x):
        """Input tensor of shape [batch, grid, patch, dim]"""
        B, G, P, C = x.shape
        self._patch_num = P
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * P, G, C)
        return x
    
    def _merge_patch(self, x):
        """Input tensor of shape [batch * grid, patch, dim]"""
        BG, P, C = x.shape
        x = x.reshape(BG // self._grid_num, self._grid_num, P, C)
        return x
    
    def _merge_grid(self, x):
        """Input tensor of shape [batch * patch, grid, dim]"""
        BP, G, C = x.shape
        x = x.reshape(BP // self._patch_num, self._patch_num, G, C)
        x = x.permute(0, 2, 1, 3)
        return x
    
    def _grid_attn(self, x: torch.Tensor):
        """Input tensor of shape [batch, grid, patch, dim]"""
        x = self._separate_patch(x)
        x = self.grid_attn(x)
        x = self._merge_patch(x)
        return x

    def _patch_attn(self, x: torch.Tensor):
        """Input tensor of shape [batch, grid, patch, dim]"""
        x = self._separate_grid(x)
        x = self.patch_attn(x)
        x = self._merge_grid(x)
        return x

    def forward(self, x: torch.Tensor):
        """Input tensor of shape [batch, grid, patch, dim]"""
        x_grid, x_patch = torch.split(x, x.shape[-1] // 2, dim=-1)
        x_grid = self._grid_attn(x_grid)
        x_patch = self._patch_attn(x_patch)
        return torch.cat([x_grid, x_patch], dim=-1) 
