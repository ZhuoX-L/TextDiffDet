# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import logging
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

__all__ = ["DiffusionDetDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    根据 config 构建一组图像增强 TransformGen
    Returns:
        list[TransformGen]
    """
    if is_train:
        # 训练时的最短边范围、多尺度方式
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        # 测试阶段固定尺寸
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"

    # 若使用 range 方式，则必须给定两个 min size
    if sample_style == "range":
        assert len(min_size) == 2, "range 模式下 min_size 长度必须为 2."

    logger = logging.getLogger(__name__)
    tfm_gens = []

    if is_train:
        # 训练阶段加入随机翻转增强
        tfm_gens.append(T.RandomFlip())

    # 无论训练/测试都会使用最短边缩放
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    if is_train:
        logger.info("训练阶段使用的 TransformGens: " + str(tfm_gens))

    return tfm_gens


class DiffusionDetDatasetMapper:
    """
    DiffusionDet 的数据预处理 Mapper。

    输入 detectron2 格式 dict，输出模型可直接使用的 tensor 格式。
    包含步骤：
    1. 读取图像
    2. 应用几何变换（包括增强）
    3. 随机裁剪（如开启）
    4. 将图片和标注转为 tensor
    """

    def __init__(self, cfg, is_train=True):

        # 若训练且 crop 开启，则加入裁剪相关 transform
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                # 随机 resize，三种尺寸中随机选一
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                # 随机裁剪
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        # 基础 transform
        self.tfm_gens = build_transform_gen(cfg, is_train)

        logging.getLogger(__name__).info(
            "训练阶段的所有 TransformGens: {}, crop 配置: {}".format(
                str(self.tfm_gens), str(self.crop_gen)
            )
        )

        self.img_format = cfg.INPUT.FORMAT  # 图片格式：RGB/BGR
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        dataset_dict: Detectron2 Dataset format

        返回：
            detectron2 模型能直接接受的格式
        """
        # deepcopy 防止修改原始数据
        dataset_dict = copy.deepcopy(dataset_dict)

        # 按指定格式读取图像（BGR 或 RGB）
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # 是否使用 crop
        if self.crop_gen is None:
            # 无裁剪：只应用基本变换
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            # 有裁剪：50% 概率使用裁剪策略
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                # 基本增强的最后一个 resize 之前插入 crop 操作
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w

        # 转为 torch tensor，通道变为 C,H,W，同时保证连续内存
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            # 测试阶段通常不需要 annotations
            dataset_dict.pop("annotations", None)
            return dataset_dict

        # ========== 训练阶段处理标注 ==========
        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                # 移除 segmentation 和 keypoints（DiffusionDet 不需要）
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # 将标注应用几何变换，并移除 crowd 标注
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            # 转为 detectron2 的 Instances 格式
            instances = utils.annotations_to_instances(annos, image_shape)
            # 移除无效实例，如面积为 0 的 bbox
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
