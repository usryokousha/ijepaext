import torch
from torch import nn

from typing import Optional

def generate_lookup_tensor(
        length: int,
        max_relative_position: Optional[int] = None,
):
    """Generate a one_hot lookup tensor to reindex embeddings along one dimension.

    Args:
        length: the length to reindex to.
        max_relative_position: the maximum relative position to consider.
            Relative position embeddings for distances above this threshold
            are zeroed out.
    Returns:
        a lookup Tensor of size [length, length, vocab_size] that satisfies
            ret[n,m,v] = 1{m - n + max_relative_position = v}.
    """
    if max_relative_position is None:
        max_relative_position = length - 1
    # Return the cached lookup tensor, otherwise compute it and cache it.
    vocab_size = 2 * max_relative_position + 1
    ret = torch.zeros(length, length, vocab_size)
    for i in range(length):
        for x in range(length):
            v = x - i + max_relative_position
            if abs(x - i) > max_relative_position:
                continue
            ret[i, x, v] = 1
    return ret

def reindex_2d_einsum_lookup(
        relative_position_tensor,
        height: int,
        width: int,
        height_lookup: torch.Tensor,
        width_lookup: torch.Tensor,
) -> torch.Tensor:
    """Reindex 2d relative position bias with 2 independent einsum lookups.

    Adapted from:
     https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/attention_utils.py

    Args:
        relative_position_tensor: tensor of shape
            [..., vocab_height, vocab_width, ...].
        height: height to reindex to.
        width: width to reindex to.
        height_lookup: one-hot height lookup
        width_lookup: one-hot width lookup
    Returns:
        reindexed_tensor: a Tensor of shape
            [..., height * width, height * width, ...]
    """
    reindexed_tensor = torch.einsum('nhw,ixh->nixw', relative_position_tensor, height_lookup)
    reindexed_tensor = torch.einsum('nixw,jyw->nijxy', reindexed_tensor, width_lookup)
    area = height * width
    return reindexed_tensor.reshape(relative_position_tensor.shape[0], area, area)


class RelativePositionBias(nn.Module):
    """ Relative Position Bias Impl (Compatible with Tensorflow MaxViT models)
    Adapted from:
     https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/attention_utils.py
    """
    def __init__(self, window_size, num_heads, prefix_tokens=0):
        super().__init__()
        assert prefix_tokens <= 1
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.num_heads = num_heads

        vocab_height = 2 * window_size[0] - 1
        vocab_width = 2 * window_size[1] - 1
        self.bias_shape = (self.num_heads, vocab_height, vocab_width)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.bias_shape))
        self.register_buffer('height_lookup', generate_lookup_tensor(window_size[0]), persistent=False)
        self.register_buffer('width_lookup', generate_lookup_tensor(window_size[1]), persistent=False)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.relative_position_bias_table, std=.02)

    def get_bias(self) -> torch.Tensor:
        # FIXME change to not use one-hot/einsum?
        return reindex_2d_einsum_lookup(
            self.relative_position_bias_table,
            self.window_size[0],
            self.window_size[1],
            self.height_lookup,
            self.width_lookup
        )

    def forward(self, attn, shared_attn_bias: Optional[torch.Tensor] = None):
        return attn + self.get_bias()