# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from ijepaext.layers import (Mlp, 
                             SwiGLUFFNFused, 
                             DecoderBlock, 
                             get_2d_pos_embed, 
                             apply_masks, 
                             repeat_interleave_batch)


logger = logging.getLogger("ijepaext")

from .vision_transformer import named_apply, BlockChunk, init_weights_vit_timm

class CrossPredictorVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        predictor_embed_dim=768,
        latent_embed_dim=768,
        latent_num_patches = 196,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        act_layer=nn.GELU,
        block_fn=DecoderBlock,
        ffn_layer="mlp",
        block_chunks=1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.latent_embed = nn.Linear(latent_embed_dim, predictor_embed_dim)
        self.pos_embed = get_2d_pos_embed(embed_dim, img_size // patch_size, cls_token=False)
        self.latent_pos_embed = nn.Parameter(embed_dim, latent_num_patches)

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.predictor_norm = norm_layer(embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        named_apply(init_weights_vit_timm, self)

    def prepare_tokens_with_masks(self, x, latent, masks_context, masks_predict):
        assert (masks_predict is not None) and (masks_context is not None), \
            'Cannot run predictor without mask indices'

        if not isinstance(masks_context, list):
            masks_context = [masks_context]

        if not isinstance(masks_predict, list):
            masks_predict = [masks_predict]

        # batch size
        B = x.shape[0] // len(masks_context)

        # map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)
        latent = self.latent_embed(latent)

        # add positional embeddings to x tokens and latent tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_context)
        latent += self.latent_pos_embed.repeat(B, 1, 1)
        num_context = x.shape[1]

        # concat mask tokens to x
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks_predict)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_context))
        
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        
        pred_tokens += pos_embs
        x = x.repeat(len(masks_predict), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        return x, latent, num_context
    
    def forward_features(self, x, masks_context, masks_predict):
        x, latent, num_context = self.prepare_tokens_with_masks(x, masks_context, masks_predict)
        for blk in self.blocks:
            x = blk(x, latent)

        x_norm = self.predictor_norm(x)
        x_norm = x_norm[:, :num_context, :]
        x_norm = self.predictor_proj(x_norm)
        return x_norm
    
    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        return tuple(outputs)

def vit_predictor_small(patch_size=16, **kwargs):
    model = CrossPredictorVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4,
        **kwargs,
    )
    return model

def vit_predictor_base(patch_size=16, **kwargs):
    model = CrossPredictorVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4,
        **kwargs,
    )
    return model

def vit_predictor_large(patch_size=16, **kwargs):
    model = CrossPredictorVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=16,
        mlp_ratio=4,
        **kwargs,
    )
    return model

def vit_predictor_huge(patch_size=16, **kwargs):
    model = CrossPredictorVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=32,
        mlp_ratio=4,
        **kwargs,
    )
    return model

def vit_predictor_giant(patch_size=16, **kwargs):
    model = CrossPredictorVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=32,
        mlp_ratio=4,
        **kwargs,
    )
    return model
