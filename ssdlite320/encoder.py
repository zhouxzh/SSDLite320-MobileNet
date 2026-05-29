import itertools
from math import sqrt
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


BoxOrder = Literal["ltrb", "xywh"]


def calc_iou_tensor(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two XYXY box tensors."""
    num_boxes_a = boxes_a.size(0)
    num_boxes_b = boxes_b.size(0)

    expanded_boxes_a = boxes_a.unsqueeze(1).expand(-1, num_boxes_b, -1)
    expanded_boxes_b = boxes_b.unsqueeze(0).expand(num_boxes_a, -1, -1)

    left_top = torch.max(expanded_boxes_a[:, :, :2], expanded_boxes_b[:, :, :2])
    right_bottom = torch.min(expanded_boxes_a[:, :, 2:], expanded_boxes_b[:, :, 2:])

    overlap = (right_bottom - left_top).clamp(min=0)
    intersection = overlap[:, :, 0] * overlap[:, :, 1]

    area_a = (expanded_boxes_a[:, :, 2] - expanded_boxes_a[:, :, 0]) * (
        expanded_boxes_a[:, :, 3] - expanded_boxes_a[:, :, 1]
    )
    area_b = (expanded_boxes_b[:, :, 2] - expanded_boxes_b[:, :, 0]) * (
        expanded_boxes_b[:, :, 3] - expanded_boxes_b[:, :, 1]
    )
    return intersection / (area_a + area_b - intersection)


class Encoder:
    """Encode ground-truth boxes and decode SSD head outputs with NMS."""

    def __init__(self, dboxes: "DefaultBoxes"):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(
        self,
        bboxes_in: torch.Tensor,
        labels_in: torch.Tensor,
        criteria: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match ground-truth boxes to default boxes and convert matches to XYWH."""
        ious = calc_iou_tensor(bboxes_in, self.dboxes)
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        _, best_bbox_idx = ious.max(dim=1)

        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)
        matched_bbox_indices = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[matched_bbox_indices]] = matched_bbox_indices

        positive_mask = best_dbox_ious > criteria
        labels_out = torch.zeros(self.nboxes, dtype=torch.long)
        labels_out[positive_mask] = labels_in[best_dbox_idx[positive_mask]]

        bboxes_out = self.dboxes.clone()
        bboxes_out[positive_mask, :] = bboxes_in[best_dbox_idx[positive_mask], :]

        center_x = 0.5 * (bboxes_out[:, 0] + bboxes_out[:, 2])
        center_y = 0.5 * (bboxes_out[:, 1] + bboxes_out[:, 3])
        width = -bboxes_out[:, 0] + bboxes_out[:, 2]
        height = -bboxes_out[:, 1] + bboxes_out[:, 3]
        bboxes_out[:, 0] = center_x
        bboxes_out[:, 1] = center_y
        bboxes_out[:, 2] = width
        bboxes_out[:, 3] = height
        return bboxes_out, labels_out

    def scale_back_batch(
        self,
        bboxes_in: torch.Tensor,
        scores_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert SSD XYWH deltas back to XYXY boxes and class probabilities."""
        dboxes_xywh = self.dboxes_xywh.to(device=bboxes_in.device, dtype=bboxes_in.dtype)

        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * dboxes_xywh[:, :, 2:] + dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * dboxes_xywh[:, :, 2:]

        left = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        top = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        right = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        bottom = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = left
        bboxes_in[:, :, 1] = top
        bboxes_in[:, :, 2] = right
        bboxes_in[:, :, 3] = bottom

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(
        self,
        bboxes_in: torch.Tensor | np.ndarray,
        scores_in: torch.Tensor | np.ndarray,
        criteria: float = 0.45,
        max_output: int = 200,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Decode one batch of SSD outputs into post-NMS predictions."""
        if not torch.is_tensor(bboxes_in):
            bboxes_in = torch.from_numpy(bboxes_in)
        if not torch.is_tensor(scores_in):
            scores_in = torch.from_numpy(scores_in)

        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        output = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            output.append(self.decode_single(bbox.squeeze(0), prob.squeeze(0), criteria, max_output))
        return output

    def decode_single(
        self,
        bboxes_in: torch.Tensor,
        scores_in: torch.Tensor,
        criteria: float,
        max_output: int,
        max_num: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode a single image prediction tensor with class-wise NMS."""
        bboxes_out: list[torch.Tensor] = []
        scores_out: list[torch.Tensor] = []
        labels_out: list[int] = []

        for class_index, class_scores in enumerate(scores_in.split(1, 1)):
            if class_index == 0:
                continue

            class_scores = class_scores.squeeze(1)
            keep_mask = class_scores > 0.05
            candidate_boxes = bboxes_in[keep_mask, :]
            candidate_scores = class_scores[keep_mask]
            if candidate_scores.numel() == 0:
                continue

            _, sorted_indices = candidate_scores.sort(dim=0)
            sorted_indices = sorted_indices[-max_num:]
            selected_indices: list[int] = []

            while sorted_indices.numel() > 0:
                current_index = sorted_indices[-1].item()
                sorted_boxes = candidate_boxes[sorted_indices, :]
                current_box = candidate_boxes[current_index, :].unsqueeze(dim=0)
                iou_sorted = calc_iou_tensor(sorted_boxes, current_box).squeeze()
                sorted_indices = sorted_indices[iou_sorted < criteria]
                selected_indices.append(current_index)

            bboxes_out.append(candidate_boxes[selected_indices, :])
            scores_out.append(candidate_scores[selected_indices])
            labels_out.extend([class_index] * len(selected_indices))

        if not bboxes_out:
            return (
                torch.empty((0, 4), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
                torch.empty((0,), dtype=torch.float32),
            )

        merged_boxes = torch.cat(bboxes_out, dim=0)
        merged_labels = torch.tensor(labels_out, dtype=torch.long)
        merged_scores = torch.cat(scores_out, dim=0)

        _, max_ids = merged_scores.sort(dim=0)
        max_ids = max_ids[-max_output:].to("cpu")
        return merged_boxes[max_ids, :], merged_labels[max_ids], merged_scores[max_ids]


class DefaultBoxes:
    """Generate SSD default boxes for each feature map location."""

    def __init__(
        self,
        fig_size: int,
        feat_size: list[int],
        steps: list[float],
        scales: list[float],
        aspect_ratios: list[list[int]],
        scale_xy: float = 0.1,
        scale_wh: float = 0.2,
    ):
        self.feat_size = feat_size
        self.fig_size = fig_size
        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh
        self.steps = steps
        self.scales = scales
        self.aspect_ratios = aspect_ratios

        feature_scales = fig_size / np.array(steps)
        default_boxes: list[tuple[float, float, float, float]] = []

        for level_index, feature_size in enumerate(self.feat_size):
            scale_small = scales[level_index] / fig_size
            scale_large = scales[level_index + 1] / fig_size
            scale_mid = sqrt(scale_small * scale_large)
            all_sizes = [(scale_small, scale_small), (scale_mid, scale_mid)]

            for alpha in aspect_ratios[level_index]:
                width, height = scale_small * sqrt(alpha), scale_small / sqrt(alpha)
                all_sizes.append((width, height))
                all_sizes.append((height, width))

            for width, height in all_sizes:
                for row_index, col_index in itertools.product(range(feature_size), repeat=2):
                    center_x = (col_index + 0.5) / feature_scales[level_index]
                    center_y = (row_index + 0.5) / feature_scales[level_index]
                    default_boxes.append((center_x, center_y, width, height))

        self.dboxes = torch.tensor(default_boxes, dtype=torch.float32).clamp_(min=0, max=1)
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    @property
    def scale_xy(self) -> float:
        return self.scale_xy_

    @property
    def scale_wh(self) -> float:
        return self.scale_wh_

    def __call__(self, order: BoxOrder = "ltrb") -> torch.Tensor:
        if order == "ltrb":
            return self.dboxes_ltrb
        if order == "xywh":
            return self.dboxes
        raise ValueError(f"Unsupported default-box order: {order}")


def dboxes320_coco(min_ratio: float = 0.1, max_ratio: float = 0.9) -> DefaultBoxes:
    """Build the 320x320 COCO default-box configuration used by this project."""
    figsize = 320
    feat_size = [20, 10, 5, 3, 2, 1]
    steps = [figsize / feature_size for feature_size in feat_size]

    num_layers = len(feat_size)
    scales_norm = [
        min_ratio + (max_ratio - min_ratio) * layer_index / (num_layers - 1)
        for layer_index in range(num_layers)
    ]
    scales_norm.append(1.0)
    scales = [scale * figsize for scale in scales_norm]

    aspect_ratios = [[2, 3] for _ in range(num_layers)]
    return DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)