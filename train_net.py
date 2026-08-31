# ==========================================
# Modified from DiffusionDet training code
# ==========================================
# Copyright (c) Facebook, Inc. and its affiliates.
"""
TextDiffDet training entry.

This public version contains only the basic training pipeline.
"""

import itertools
from typing import Any, Dict, List, Set

import torch

import dataset
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.solver.build import maybe_add_gradient_clipping

from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config


class Trainer(DefaultTrainer):
    """Minimal trainer for TextDiffDet."""

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        for name, value in model.named_parameters(recurse=True):
            if not value.requires_grad or value in memo:
                continue

            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY

            if "backbone" in name:
                lr *= cfg.SOLVER.BACKBONE_MULTIPLIER

            params.append(
                {
                    "params": [value],
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enabled = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[group["params"] for group in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enabled else optim

        if cfg.SOLVER.OPTIMIZER == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
            )
        elif cfg.SOLVER.OPTIMIZER == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params,
                cfg.SOLVER.BASE_LR,
            )
        else:
            raise NotImplementedError(
                f"Unsupported optimizer: {cfg.SOLVER.OPTIMIZER}"
            )

        if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)

        return optimizer


def setup(args):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
