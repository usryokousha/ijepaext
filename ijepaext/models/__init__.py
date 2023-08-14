# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits

logger = logging.getLogger("ijepaext")

def build_model(args, only_target_encoder=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
        )
        target_encoder = vits.__dict__[args.arch](**vit_kwargs)
        if only_target_encoder:
            return target_encoder, target_encoder.embed_dim
        
        context_encoder = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = context_encoder.embed_dim
    return context_encoder, target_encoder, embed_dim


def build_model_from_cfg(cfg, only_target_encoder=False):
    return build_model(cfg.context_encoder, only_target_encoder=only_target_encoder, img_size=cfg.crops.global_crops_size)
