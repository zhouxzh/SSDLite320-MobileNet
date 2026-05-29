from argparse import Namespace
from pathlib import Path
from typing import Any
import json

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision.transforms import v2
from torchvision import tv_tensors
from .encoder import dboxes320_coco, Encoder


DEFAULT_IMAGE_SIZE = 320
COCO_DATASET_NAME = "detection-datasets/coco"
COCO_CACHE_DIR = Path("data")
COCO_GT_FILENAME = "coco_gt.json"
COCO_GT_CACHE_PATH = COCO_CACHE_DIR / COCO_GT_FILENAME
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# Dataset download and transform builders
# -----------------------------------------------------------------------------


def download_and_load_coco():
    """Load the COCO dataset from Hugging Face into the local cache directory."""
    dataset_name = COCO_DATASET_NAME
    print(f"Loading dataset: {dataset_name} ...")
    try:
        dataset = load_dataset(dataset_name, cache_dir=str(COCO_CACHE_DIR))
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def collate_detection_eval_batch(batch):
    """Stack images while keeping variable-length targets as per-image tuples."""
    images, boxes, labels, img_ids = zip(*batch)
    return torch.stack(images, 0), boxes, labels, img_ids


def load_coco_ground_truth_api(gt_file: str):
    from pycocotools.coco import COCO

    try:
        return COCO(gt_file)
    except Exception as error:
        print(f"读取 COCO Ground Truth 文件失败: {gt_file}")
        print(f"错误信息: {error}")
        print("如果这个文件已经损坏，请手动删除 data/coco_gt.json 后重新运行。")
        raise


def build_normalize_transform() -> v2.Normalize:
    """Keep ImageNet normalization aligned with timm MobileNet pretraining."""
    return v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def build_resize_and_normalize_steps(img_size: int) -> list[Any]:
    return [
        v2.Resize((img_size, img_size)),
        v2.SanitizeBoundingBoxes(),
        build_normalize_transform(),
    ]


def build_train_transform(img_size: int, use_augmentation: bool) -> v2.Compose:
    base_steps: list[Any] = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]

    if use_augmentation:
        augmentation_steps: list[Any] = [
            v2.RandomZoomOut(fill=IMAGENET_MEAN, p=0.5),
            v2.RandomIoUCrop(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomPhotometricDistort(p=0.4),
        ]
        return v2.Compose(base_steps + augmentation_steps + build_resize_and_normalize_steps(img_size))

    return v2.Compose(base_steps + build_resize_and_normalize_steps(img_size))


def build_eval_transform(img_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((img_size, img_size)),
            build_normalize_transform(),
        ]
    )


# -----------------------------------------------------------------------------
# Annotation and box helpers
# -----------------------------------------------------------------------------


def parse_coco_annotations(objects: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert HF COCO annotations into XYXY boxes and 1-based class labels."""
    boxes: list[list[float]] = []
    labels: list[int] = []

    for bbox, category in zip(objects.get('bbox', []), objects.get('category', [])):
        xmin, ymin, xmax, ymax = bbox
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(category + 1)

    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def build_bounding_boxes_tensor(boxes_xyxy: torch.Tensor, image_size: tuple[int, int]) -> tv_tensors.BoundingBoxes:
    height, width = image_size
    if boxes_xyxy.numel() == 0:
        boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
    return tv_tensors.BoundingBoxes(boxes_xyxy, format="XYXY", canvas_size=(height, width))


def normalize_boxes_to_unit_interval(boxes_xyxy: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy

    height, width = image_size
    scale_tensor = torch.tensor([width, height, width, height], dtype=torch.float32, device=boxes_xyxy.device)
    return (boxes_xyxy / scale_tensor).clamp(0, 1)


def resize_eval_boxes(boxes_xyxy: torch.Tensor, original_size: tuple[int, int], image_size: int) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy

    original_width, original_height = original_size
    resized_boxes = boxes_xyxy.clone()
    resized_boxes[:, [0, 2]] /= original_width
    resized_boxes[:, [1, 3]] /= original_height
    resized_boxes[:, [0, 2]] *= image_size
    resized_boxes[:, [1, 3]] *= image_size
    return resized_boxes


def encode_training_targets(
    box_coder: Encoder,
    boxes_xyxy: torch.Tensor,
    labels: torch.Tensor,
    output_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    boxes_norm = normalize_boxes_to_unit_interval(boxes_xyxy, output_size)

    if boxes_norm.size(0) > 0:
        return box_coder.encode(boxes_norm, labels)

    encoded_locs = box_coder.dboxes_xywh.squeeze(0).clone()
    encoded_labels = torch.zeros(box_coder.nboxes, dtype=torch.long)
    return encoded_locs, encoded_labels


# -----------------------------------------------------------------------------
# Dataset adapter
# -----------------------------------------------------------------------------


class COCOSSDDataset(Dataset):
    """COCO dataset adapter that emits SSD training targets or eval targets."""

    def __init__(
        self,
        hf_dataset,
        img_size: int = DEFAULT_IMAGE_SIZE,
        is_train: bool = True,
        args: Namespace | None = None,
    ):
        self.dataset = hf_dataset
        self.img_size = img_size
        self.is_train = is_train
        self.args = args
        self.use_augmentation = bool(getattr(args, 'augment', False))
        self.dbox_min_ratio = getattr(args, 'dbox_min_ratio', 0.1)
        self.dbox_max_ratio = getattr(args, 'dbox_max_ratio', 0.9)

        self.dboxes = dboxes320_coco(min_ratio=self.dbox_min_ratio, max_ratio=self.dbox_max_ratio)
        self.box_coder = Encoder(self.dboxes)
        self.train_transform = build_train_transform(img_size, use_augmentation=self.use_augmentation)
        self.eval_transform = build_eval_transform(img_size)

        if self.is_train and self.use_augmentation:
            print("Data Augmentation enabled (SSD-style + timm-pretrain-friendly normalize).")
        else:
            print("Data Augmentation disabled (resize + normalize).")

    def __len__(self):
        return len(self.dataset)

    def _load_sample(self, idx: int):
        item = self.dataset[idx]
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return item, image

    def _get_train_item(self, item, image):
        boxes_t, labels_t = parse_coco_annotations(item['objects'])
        width, height = image.size
        boxes_tv = build_bounding_boxes_tensor(boxes_t, (height, width))
        labels_in = labels_t if labels_t.numel() > 0 else torch.zeros((0,), dtype=torch.long)

        transformed = self.train_transform({"image": image, "boxes": boxes_tv, "labels": labels_in})
        img_tensor = transformed["image"]
        out_boxes = transformed["boxes"].as_subclass(torch.Tensor)
        labels_aug = transformed["labels"]

        encoded_locs, encoded_labels = encode_training_targets(
            self.box_coder,
            out_boxes,
            labels_aug,
            img_tensor.shape[-2:],
        )
        return img_tensor, encoded_locs, encoded_labels

    def _get_eval_item(self, item, image, idx: int):
        orig_w, orig_h = image.size
        boxes_t, labels_t = parse_coco_annotations(item['objects'])
        img_tensor = self.eval_transform(image)
        gt_boxes = resize_eval_boxes(boxes_t, (orig_w, orig_h), self.img_size)
        image_id = item.get('image_id', idx)
        return img_tensor, gt_boxes, labels_t, image_id

    def __getitem__(self, idx):
        item, image = self._load_sample(idx)

        if self.is_train:
            return self._get_train_item(item, image)

        return self._get_eval_item(item, image, idx)


# -----------------------------------------------------------------------------
# Dataloader builders and COCO export
# -----------------------------------------------------------------------------


def build_coco_ground_truth_dict(val_ds_hf, image_size: int = DEFAULT_IMAGE_SIZE) -> dict[str, Any]:
    coco_gt_dict = {"images": [], "annotations": [], "categories": []}
    category_ids: set[int] = set()

    for index in range(len(val_ds_hf)):
        item = val_ds_hf[index]
        image_id = item.get('image_id', index)
        image_width, image_height = item['image'].size
        coco_gt_dict["images"].append({"id": image_id, "width": image_size, "height": image_size})

        objects = item.get('objects', {})
        for bbox, category_id in zip(objects.get('bbox', []), objects.get('category', [])):
            xmin, ymin, xmax, ymax = bbox
            bx = xmin * image_size / image_width
            by = ymin * image_size / image_height
            bw = (xmax - xmin) * image_size / image_width
            bh = (ymax - ymin) * image_size / image_height

            coco_gt_dict["annotations"].append(
                {
                    "id": len(coco_gt_dict["annotations"]),
                    "image_id": image_id,
                    "category_id": category_id + 1,
                    "bbox": [bx, by, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            category_ids.add(category_id + 1)

    for category_id in sorted(category_ids):
        coco_gt_dict["categories"].append({"id": category_id, "name": str(category_id)})

    return coco_gt_dict


def get_train_loader(full_dataset, batch_size: int, num_workers: int = 4, args: Namespace | None = None) -> DataLoader:
    if full_dataset is None:
        raise RuntimeError("Failed to load dataset")
    train_ds_hf = full_dataset['train']
    ssd_dataset_train = COCOSSDDataset(train_ds_hf, is_train=True, args=args)
    train_loader = DataLoader(
        ssd_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader


def get_val_dataloader(full_dataset, batch_size: int, num_workers: int = 4, args: Namespace | None = None) -> DataLoader:
    val_ds_hf = full_dataset['val']
    ssd_dataset_val = COCOSSDDataset(val_ds_hf, is_train=False, args=args)
    val_loader = DataLoader(
        ssd_dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_detection_eval_batch,
    )
    return val_loader


def get_coco_ground_truth(val_ds_hf) -> str:
    """Build and cache a COCO-format ground-truth JSON aligned to 320x320 eval inputs."""
    COCO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if COCO_GT_CACHE_PATH.exists():
        print(f"Using cached COCO Ground Truth: {COCO_GT_CACHE_PATH}")
        return str(COCO_GT_CACHE_PATH)

    print(f"Preparing COCO Ground Truth and caching to: {COCO_GT_CACHE_PATH}")
    coco_gt_dict = build_coco_ground_truth_dict(val_ds_hf, image_size=DEFAULT_IMAGE_SIZE)
    with COCO_GT_CACHE_PATH.open("w", encoding="utf-8") as file:
        json.dump(coco_gt_dict, file)

    return str(COCO_GT_CACHE_PATH)
