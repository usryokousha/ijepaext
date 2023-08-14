# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits
from . import predictor as pvits
from . import cross_predictor as cpvits


logger = logging.getLogger("ijepaext")


def build_model(encoder_args, predictor_args, only_teacher=False, img_size=224):
    encoder_args.arch = encoder_args.arch.removesuffix("_memeff")
    if "vit" in encoder_args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=encoder_args.patch_size,
            init_values=encoder_args.layerscale,
            ffn_layer=encoder_args.ffn_layer,
            block_chunks=encoder_args.block_chunks,
            qkv_bias=encoder_args.qkv_bias,
            proj_bias=encoder_args.proj_bias,
            ffn_bias=encoder_args.ffn_bias,
        )
        teacher_encoder = vits.__dict__[encoder_args.arch](**vit_kwargs)
        if only_teacher:
            return teacher_encoder, teacher_encoder.embed_dim
        
        student_encoder = vits.__dict__[encoder_args.arch](
            **vit_kwargs,
            drop_path_rate=encoder_args.drop_path_rate,
            drop_path_uniform=encoder_args.drop_path_uniform,
        )
    
        predictor = pvits.__dict__[predictor_args.arch](
            **vit_kwargs,
            drop_path_rate=predictor_args.drop_path_rate,
            drop_path_uniform=predictor_args.drop_path_uniform,
        )
        embed_dim = student_encoder.embed_dim
    return student_encoder, teacher_encoder, predictor, embed_dim

def build_cross_model(encoder_args, predictor_args, latent_args, only_teacher=False, img_size=224, latent_img_size=224):
    encoder_args.arch = encoder_args.arch.removesuffix("_memeff")
    if "vit" in encoder_args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=encoder_args.patch_size,
            init_values=encoder_args.layerscale,
            ffn_layer=encoder_args.ffn_layer,
            block_chunks=encoder_args.block_chunks,
            qkv_bias=encoder_args.qkv_bias,
            proj_bias=encoder_args.proj_bias,
            ffn_bias=encoder_args.ffn_bias,
        )
        teacher_encoder = vits.__dict__[encoder_args.arch](**vit_kwargs)
        if only_teacher:
            return teacher_encoder, teacher_encoder.embed_dim
        
        student_encoder = vits.__dict__[encoder_args.arch](
            **vit_kwargs,
            drop_path_rate=encoder_args.drop_path_rate,
            drop_path_uniform=encoder_args.drop_path_uniform,
        )
    
        cross_predictor = cpvits.__dict__[predictor_args.arch](
            **vit_kwargs,
            drop_path_rate=predictor_args.drop_path_rate,
            drop_path_uniform=predictor_args.drop_path_uniform,
        )
    
    if "vit" in latent_args.arch:
        vit_latent_kwargs = dict(
            img_size=latent_img_size,
            patch_size=latent_args.patch_size,
            init_values=latent_args.layerscale,
            ffn_layer=latent_args.ffn_layer,
            block_chunks=latent_args.block_chunks,
            qkv_bias=latent_args.qkv_bias,
            proj_bias=latent_args.proj_bias,
            ffn_bias=latent_args.ffn_bias,
        )
        latent_encoder = vits.__dict__[latent_args.arch](
            **vit_latent_kwargs,
            drop_path_rate=latent_args.drop_path_rate,
            drop_path_uniform=latent_args.drop_path_uniform,
        )

        embed_dim = student_encoder.embed_dim
    return student_encoder, teacher_encoder, latent_encoder, cross_predictor, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student.encoder, 
                       cfg.student.predictor, 
                       only_teacher=only_teacher, 
                       img_size=cfg.crops.global_crops_size)

def build_cross_model_from_cfg(cfg, only_teacher=False):
    return build_cross_model(cfg.student.encoder, 
                             cfg.student.predictor, 
                             cfg.student.latent, 
                             only_teacher=only_teacher, 
                             img_size=cfg.crops.crops_size, 
                             latent_img_size=cfg.crops.latent_crops_size)
