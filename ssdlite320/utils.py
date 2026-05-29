from __future__ import annotations

from typing import Any, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch


def extract_category_names(
    dataset: Any,
    split: str | None = "train",
    include_background: bool = False,
) -> list[str] | None:
    try:
        data_source = dataset if hasattr(dataset, "features") else dataset[split]
        features = data_source.features
        objects_feature = features.get("objects") if isinstance(features, dict) else features["objects"]
        category_feature = getattr(objects_feature, "feature", objects_feature)["category"]
        names = getattr(category_feature, "feature", category_feature).names
        if not isinstance(names, list):
            return None
        return ["BACKGROUND", *names] if include_background else names
    except Exception as error:
        print(f"Error extracting category names: {error}")
        return None


def build_validation_metrics(stats: np.ndarray) -> dict[str, float]:
    return {
        "mAP": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
    }


def validate_default_box_ratios(min_ratio: float, max_ratio: float) -> None:
    if not (0.0 < min_ratio < max_ratio < 1.0):
        raise ValueError("--dbox-min-ratio 与 --dbox-max-ratio 必须满足 0 < min < max < 1。")


def visualize_sample(
    img_tensor: torch.Tensor,
    gt_boxes: torch.Tensor | np.ndarray,
    gt_labels: torch.Tensor,
    p_boxes: torch.Tensor,
    p_labels: torch.Tensor,
    p_scores: torch.Tensor,
    category_names: Sequence[str] | None,
    save_path: str,
    score_threshold: float = 0.4,
) -> None:
    device = img_tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    image = img_tensor.clone()
    image.mul_(std).add_(mean)
    image = image.permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1)
    image_height, image_width = image.shape[:2]

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()

    for box, label in zip(gt_boxes, gt_labels):
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        ax.add_patch(patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor="lime", facecolor="none"))

        label_name = str(label.item())
        if category_names and label.item() < len(category_names):
            label_name = category_names[label.item()]
        ax.text(xmin, ymin, f"GT: {label_name}", color="lime", fontsize=9, backgroundcolor="black", alpha=0.6)

    if p_boxes.numel() > 0:
        pred_boxes = p_boxes.cpu().numpy().copy()
        pred_boxes[:, [0, 2]] *= image_width
        pred_boxes[:, [1, 3]] *= image_height
        pred_labels = p_labels.cpu().numpy()
        pred_scores = p_scores.cpu().numpy()
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if score < score_threshold:
                continue
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            ax.add_patch(patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor="red", facecolor="none"))

            label_name = str(label)
            if category_names and label < len(category_names):
                label_name = category_names[label]
            ax.text(xmin, ymax, f"Pred: {label_name} {score:.2f}", color="white", fontsize=9, backgroundcolor="red", alpha=0.7)

    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)