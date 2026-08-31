# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Rufeng Zhang, Peize Sun
# Contact: {sunpeize, cxrfzhang}@foxmail.com
# 
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
from itertools import count          # 提供 count() 这样的无限自增迭代器
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel  # 分布式并行封装

from detectron2.modeling import GeneralizedRCNNWithTTA, DatasetMapperTTA
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference_single_image
from detectron2.structures import Instances, Boxes


class DiffusionDetWithTTA(GeneralizedRCNNWithTTA):
    """
        针对 DiffusionDet 的测试时增强（Test-Time Augmentation, TTA）封装。
        继承自 Detectron2 的 GeneralizedRCNNWithTTA。

        使用方式：和原来模型 forward 接口一样，只是在推理时自动对图像做多尺度/翻转等增强，
        再把多次预测结果合并。
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
        """
            Args:
                cfg (CfgNode): 配置对象
                model (DiffusionDet): 原始 DiffusionDet 模型
                tta_mapper (callable): 输入一个 dataset dict，返回带增强版本的若干 dict。
                                       默认使用 DatasetMapperTTA(cfg)
                batch_size (int): 对增强后的多张图做推理时的 batch 大小
        """
        # 先调用 nn.Module.__init__，避免 “cannot assign module before Module.__init__()” 错误
        nn.Module.__init__(self)
        # 如果传进来的是 DDP 封装的模型，需要取出里面真正的 model
        if isinstance(model, DistributedDataParallel):
            model = model.module

        # 把 cfg 克隆一份，避免外部被修改
        self.cfg = cfg.clone()
        self.model = model

        # 如果没有传自定义的 tta_mapper，就用 Detectron2 默认的 DatasetMapperTTA
        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

        # 下面这些是 cvpods 风格的 TTA 配置
        self.enable_cvpods_tta = cfg.TEST.AUG.CVPODS_TTA          # 是否启用 cvpods TTA 逻辑
        self.enable_scale_filter = cfg.TEST.AUG.SCALE_FILTER      # 是否按目标尺寸做过滤
        self.scale_ranges = cfg.TEST.AUG.SCALE_RANGES             # 各个尺度的目标面积范围
        self.max_detection = cfg.MODEL.DiffusionDet.NUM_PROPOSALS # 最多的检测数量（和 proposals 数一致）

    def _batch_inference(self, batched_inputs, detected_instances=None):
        """
        以 batch 形式做推理，而不是一次性把 batched_inputs 全丢进模型。
        相比父类的实现，这里支持 cvpods 的 TTA 逻辑。

        Args:
            batched_inputs (list[dict]): 与 DiffusionDet.forward 相同的输入格式
            detected_instances: 这里没用，兼容接口而已

        Returns:
            outputs (list[Instances]): 与 DiffusionDet.forward 相同的输出列表
        """
        if detected_instances is None:
            detected_instances = [None] * len(batched_inputs)

        # 如果有水平翻转增强，那么同一个尺度会出现两次（原图+翻转），所以乘以 2
        factors = 2 if self.tta_mapper.flip else 1
        if self.enable_scale_filter:
            # 如果启用 scale filter，增强后的图像数量应该等于 尺度数 * flip因子
            assert len(batched_inputs) == len(self.scale_ranges) * factors

        outputs = []
        inputs, instances = [], []
        # 用 count() 给每个输入自动编号 idx
        for idx, input, instance in zip(count(), batched_inputs, detected_instances):
            inputs.append(input)
            instances.append(instance)

            if self.enable_cvpods_tta:
                # cvpods 模式下，直接对当前 inputs 做一次前向（这里 inputs 实际上长度就是 1）
                # do_postprocess=False 表示不做 Detectron2 默认后处理
                output = self.model.forward(inputs, do_postprocess=False)[0]

                if self.enable_scale_filter:
                    # 取出预测框
                    pred_boxes = output.get("pred_boxes")
                    # 按配置的 scale_ranges 过滤掉太大/太小的框
                    keep = self.filter_boxes(
                        pred_boxes.tensor,
                        *self.scale_ranges[idx // factors]  # idx//factors 对应当前尺度
                    )
                    # 根据 keep 重新构造一个新的 Instances 结果
                    output = Instances(
                        image_size=output.image_size,
                        pred_boxes=Boxes(pred_boxes.tensor[keep]),
                        pred_classes=output.pred_classes[keep],
                        scores=output.scores[keep])
                # 把每个增强版本的结果都放进 outputs
                outputs.extend([output])
            else:
                # 非 cvpods_tta 模式下，按 batch_size 聚合后一起前向
                if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
                    outputs.extend(
                        self.model.forward(
                            inputs,
                            do_postprocess=False,
                        )
                    )
            # 每轮完成后清空 inputs、instances 缓冲
            inputs, instances = [], []
        return outputs

    @staticmethod
    def filter_boxes(boxes, min_scale, max_scale):
        """
        根据面积过滤 boxes。

        Args:
            boxes: 形状为 (N, 4) 的张量，格式为 xyxy
            min_scale, max_scale: 面积范围阈值

        Returns:
            keep (BoolTensor): 表示保留哪些框
        """
        # 计算宽高
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        # w*h 在 [min_scale^2, max_scale^2] 中的框保留
        keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
        return keep

    def _inference_one_image(self, input):
        """
        对一张图片做 TTA 推理并合并结果。

        Args:
            input (dict): 包含 "image" (CHW tensor)、"height"、"width" 等字段

        Returns:
            dict: 包含 "instances" 的输出字典
        """
        # 原图尺寸 (H, W)
        orig_shape = (input["height"], input["width"])
        # 得到增强后的多份 input 以及对应的空间变换 tfms
        augmented_inputs, tfms = self._get_augmented_inputs(input)
        # 对所有增强版本做推理，获取所有 boxes/score/class，并反变换回原图坐标
        all_boxes, all_scores, all_classes = self._get_augmented_boxes(augmented_inputs, tfms)

        # 根据是否启用 cvpods_tta 选择不同的合并策略
        if self.enable_cvpods_tta:
            merged_instances = self._merge_detections_cvpods_tta(
                all_boxes, all_scores, all_classes, orig_shape
            )
        else:
            merged_instances = self._merge_detections(
                all_boxes, all_scores, all_classes, orig_shape
            )

        return {"instances": merged_instances}

    def _merge_detections(self, all_boxes, all_scores, all_classes, shape_hw):
        """
        普通 TTA 情况下的结果合并，调用 Detectron2 自带的 fast_rcnn_inference_single_image。
        """
        # 检测总数
        num_boxes = len(all_boxes)
        # DiffusionDet 的类别数
        num_classes = self.cfg.MODEL.DiffusionDet.NUM_CLASSES
        # fast_rcnn_inference_single_image 期望的 scores 形状为 [N, num_classes+1]（+1 是背景）
        all_scores_2d = torch.zeros(num_boxes, num_classes + 1, device=all_boxes.device)
        # 把每个 box 的 score 填到对应的类别位置上
        for idx, cls, score in zip(count(), all_classes, all_scores):
            all_scores_2d[idx, cls] = score

        # 调用 Detectron2 的 NMS & top-k 合并逻辑
        merged_instances, _ = fast_rcnn_inference_single_image(
            all_boxes,
            all_scores_2d,
            shape_hw,
            1e-8,  # score_thresh，这里给一个极小的阈值，相当于不过滤
            self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,  # NMS 阈值
            self.cfg.TEST.DETECTIONS_PER_IMAGE,        # 每张图最多输出多少个框
        )

        return merged_instances

    def _merge_detections_cvpods_tta(self, all_boxes, all_scores, all_classes, shape_hw):
        """
        cvpods 风格的 TTA 合并：使用 soft-vote NMS 对多尺度结果进行融合。
        """
        # 先把 scores、classes 转为 tensor，并放到与 all_boxes 相同设备
        all_scores = torch.tensor(all_scores).to(all_boxes.device)
        all_classes = torch.tensor(all_classes).to(all_boxes.device)

        # 进行多尺度结果融合：soft_vote / vote NMS
        all_boxes, all_scores, all_classes = self.merge_result_from_multi_scales(
            all_boxes, all_scores, all_classes,
            nms_type="soft_vote",     # NMS 类型（软投票）
            vote_thresh=0.65,         # IOU 大于该阈值的框会被合并
            max_detection=self.max_detection
        )

        # 转成 Boxes 结构，并裁剪到图像边界内
        all_boxes = Boxes(all_boxes)
        all_boxes.clip(shape_hw)

        # 构造一个 Instances 作为最终结果
        result = Instances(shape_hw)
        result.pred_boxes = all_boxes
        result.scores = all_scores
        result.pred_classes = all_classes.long()
        return result

    def merge_result_from_multi_scales(
            self, boxes, scores, labels, nms_type="soft-vote", vote_thresh=0.65, max_detection=100
    ):
        """
        对多尺度预测结果进行融合，核心调用 batched_vote_nms。
        """
        boxes, scores, labels = self.batched_vote_nms(
            boxes, scores, labels, nms_type, vote_thresh
        )

        number_of_detections = boxes.shape[0]
        # 限制总检测数不超过 max_detection
        if number_of_detections > max_detection > 0:
            boxes = boxes[:max_detection]
            scores = scores[:max_detection]
            labels = labels[:max_detection]

        return boxes, scores, labels

    def batched_vote_nms(self, boxes, scores, labels, vote_type, vote_thresh=0.65):
        """
        按类别做 NMS：通过给不同类别的框加偏移，把所有框拼在一起再统一 NMS。
        """
        # 先把 labels 转成 float，方便后续参与运算
        labels = labels.float()
        # 找到所有 boxes 坐标中的最大值 + 1，作为 offset 基数
        max_coordinates = boxes.max() + 1
        # 为每一类生成不同的偏移量（label * max_coordinates）
        offsets = labels.reshape(-1, 1) * max_coordinates
        # 对每个 box 加上各自的 offset，这样不同类别的 boxes 在坐标上就完全不重叠
        boxes = boxes + offsets

        # 进行投票式的 NMS，最终 boxes / scores / labels 中的 labels 已经和偏移过的一致
        boxes, scores, labels = self.bbox_vote(boxes, scores, labels, vote_thresh, vote_type)
        # 把之前加上的偏移再减掉，恢复原始坐标
        boxes -= labels.reshape(-1, 1) * max_coordinates

        return boxes, scores, labels

    def bbox_vote(self, boxes, scores, labels, vote_thresh, vote_type="softvote"):
        """
        核心的 bbox 投票与合并逻辑。

        Args:
            boxes: (N, 4)
            scores: (N,)
            labels: (N,)
            vote_thresh: IOU 合并阈值
            vote_type: "soft_vote" 或 "vote"

        Returns:
            经过投票合并后的 boxes, scores, labels
        """
        assert boxes.shape[0] == scores.shape[0] == labels.shape[0]
        # 把 boxes、scores、labels 合并成一个 (N, 6) 的张量 [x1,x2,y1,y2,score,label]
        det = torch.cat((boxes, scores.reshape(-1, 1), labels.reshape(-1, 1)), dim=1)

        # 初始化一个空的 (0, 6) 张量，用来存储最终 vote 结果
        vote_results = torch.zeros(0, 6, device=det.device)
        if det.numel() == 0:
            # 没有框则直接返回空结果
            return vote_results[:, :4], vote_results[:, 4], vote_results[:, 5]

        # 按分数从大到小排序
        order = scores.argsort(descending=True)
        det = det[order]

        # 逐个处理
        while det.shape[0] > 0:
            # 计算 det[0] 与剩余所有框的 IOU
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            xx1 = torch.max(det[0, 0], det[:, 0])
            yy1 = torch.max(det[0, 1], det[:, 1])
            xx2 = torch.min(det[0, 2], det[:, 2])
            yy2 = torch.min(det[0, 3], det[:, 3])
            w = torch.clamp(xx2 - xx1, min=0.)
            h = torch.clamp(yy2 - yy1, min=0.)
            inter = w * h
            iou = inter / (area[0] + area[:] - inter)

            # 找出 IOU >= 阈值 的那些框的索引
            merge_index = torch.where(iou >= vote_thresh)[0]
            # 这些框用于合并
            vote_det = det[merge_index, :]
            # 剩余的框中，IOU < 阈值 的保留下来继续后续循环
            det = det[iou < vote_thresh]

            if merge_index.shape[0] <= 1:
                # 如果只有一个框（没别的可合并），直接把它加入结果
                vote_results = torch.cat((vote_results, vote_det), dim=0)
            else:
                # 若有多个框，按 vote_type 不同做不同的合并方式
                if vote_type == "soft_vote":
                    # soft_vote：使用 IOU 作为权重进行软投票
                    vote_det_iou = iou[merge_index]
                    det_accu_sum = self.get_soft_dets_sum(vote_det, vote_det_iou)
                elif vote_type == "vote":
                    # vote：普通投票加权
                    det_accu_sum = self.get_dets_sum(vote_det)
                # 把合并后的结果拼接到 vote_results
                vote_results = torch.cat((vote_results, det_accu_sum), dim=0)

        # 最终再按 score 排序一次
        order = vote_results[:, 4].argsort(descending=True)
        vote_results = vote_results[order, :]

        # 返回 boxes、scores、labels
        return vote_results[:, :4], vote_results[:, 4], vote_results[:, 5]

    @staticmethod
    def get_dets_sum(vote_det):
        """
        普通投票合并方式：
        - 用 score 作为权重对 box 坐标加权平均
        - score 取集合里最高的那个
        - label 取第一个框的 label
        """
        # 用分数作为权重乘到坐标上（每个维度都乘同一个 score）
        vote_det[:, :4] *= vote_det[:, 4:5].repeat(1, 4)
        # 取这些框中最大的 score
        max_score = vote_det[:, 4].max()
        # 初始化一个 (1, 6) 的结果张量
        det_accu_sum = torch.zeros((1, 6), device=vote_det.device)
        # 坐标 = 加权坐标和 / 分数和
        det_accu_sum[:, :4] = torch.sum(vote_det[:, :4], dim=0) / torch.sum(vote_det[:, 4])
        # 分数取最大值
        det_accu_sum[:, 4] = max_score
        # label 直接用第一个框的
        det_accu_sum[:, 5] = vote_det[0, 5]
        return det_accu_sum

    @staticmethod
    def get_soft_dets_sum(vote_det, vote_det_iou):
        """
        软投票合并方式：
        - 原始 vote_det 用分数加权求一个“主框”
        - soft_vote_det 额外保留一部分较高分框（分数乘 (1 - IOU) 做惩罚）
        """
        # 复制一份 vote_det，避免原地修改
        soft_vote_det = vote_det.detach().clone()
        # 分数乘以 (1 - IOU)，IOU 大的惩罚更多
        soft_vote_det[:, 4] *= (1 - vote_det_iou)

        INFERENCE_TH = 0.05
        # 只保留 soft 之后分数仍大于阈值的框
        soft_index = torch.where(soft_vote_det[:, 4] >= INFERENCE_TH)[0]
        soft_vote_det = soft_vote_det[soft_index, :]

        # 主框部分还是普通的 score 加权平均
        vote_det[:, :4] *= vote_det[:, 4:5].repeat(1, 4)
        max_score = vote_det[:, 4].max()
        det_accu_sum = torch.zeros((1, 6), device=vote_det.device)
        det_accu_sum[:, :4] = torch.sum(vote_det[:, :4], dim=0) / torch.sum(vote_det[:, 4])
        det_accu_sum[:, 4] = max_score
        det_accu_sum[:, 5] = vote_det[0, 5]

        # 如果 soft_vote_det 中还有框，就一起拼到结果里
        if soft_vote_det.shape[0] > 0:
            det_accu_sum = torch.cat((det_accu_sum, soft_vote_det), dim=0)
        return det_accu_sum
