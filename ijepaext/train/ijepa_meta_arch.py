# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torch import nn

from ijepaext.models import build_model_from_cfg
from ijepaext.layers import apply_masks, repeat_interleave_batch
from ijepaext.utils.utils import has_batchnorms
from ijepaext.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from ijepaext.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from ijepaext.models.vision_transformer import BlockChunk

try:
    from xformers.ops import fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
assert XFORMERS_AVAILABLE, "xFormers is required for ijepaext training"


logger = logging.getLogger("ijepaext")


class IJEPAMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_encoder, teacher_encoder, predictor = build_model_from_cfg(cfg)
        student_model_dict["encoder"] = student_encoder
        student_model_dict["predictor"] = predictor
        teacher_model_dict["encoder"] = teacher_encoder
        logger.info(f"OPTIONS -- context encoder : embed_dim: {student_encoder.embed_dim}")
        logger.info(f"OPTIONS -- predictor : embed_dim: {predictor.embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            student_encoder.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = student_encoder.embed_dim

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, mask_context, mask_predictor):
        images = images.cuda(non_blocking=True)
        mask_context = [m.cuda(non_blocking=True) for m in mask_context]
        mask_predictor = [m.cuda(non_blocking=True) for m in mask_predictor]
        
        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            targets = self.teacher.encoder(images)["x_norm"]
            targets = nn.functional.layer_norm(targets, [targets.shape[-1]])
            targets = apply_masks(targets, mask_predictor)
            targets = repeat_interleave_batch(targets, self.cfg.data.batch_size)
            return targets

        h = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        loss_dict = {}
        loss_accumulator = 0  # for backprop
        
        # student output
        z = self.student.encoder(images, mask_context)["x_norm"]
        z = self.student.predictor(z, mask_context, mask_predictor)

        l1_loss = nn.functional.smooth_l1_loss(z, h)
        loss_dict["loss"] = l1_loss
        loss_accumulator += l1_loss

        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.target_encoder._streams = self.teacher.context_encoder._streams
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for ms, mt in zip(get_fsdp_modules(self.student.target_encoder), get_fsdp_modules(self.teacher.context_encoder)):
                student_param_list += ms.params
                teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        self.teacher.encoder.load_state_dict(self.student.encoder.state_dict())
        student_model_cfg = self.cfg.compute_precision.student.encoder
        self.student.encoder = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student.encoder)
        self.student.predictor = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student.predictor)
        teacher_model_cfg = self.cfg.compute_precision.teacher.encoder
        self.teacher.encoder = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher.encoder)
