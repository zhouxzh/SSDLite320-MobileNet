from __future__ import annotations

import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import onnxruntime as ort
import torch
import torchvision.transforms.functional as tvf
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import DEFAULT_DATASET_NAME, DEFAULT_MODEL_REPO_ID
from .data_hf import IMAGENET_MEAN, IMAGENET_STD, get_coco_ground_truth, load_coco_ground_truth_api
from .encoder import Encoder, dboxes320_coco
from .utils import build_validation_metrics, extract_category_names, visualize_sample


class ONNXEvalDataset(Dataset):
    """Preprocess HF COCO validation images for fixed-batch ONNX evaluation."""

    def __init__(self, val_dataset, img_size: int):
        self.val_dataset = val_dataset
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.val_dataset)

    def __getitem__(self, index: int):
        item = self.val_dataset[index]
        image_tensor, _ = preprocess_onnx_image(item["image"], self.img_size)
        return image_tensor, int(item["image_id"])


# -----------------------------------------------------------------------------
# Shared metric logging helpers
# -----------------------------------------------------------------------------


def write_validation_metrics(writer: SummaryWriter | None, epoch: int, metrics: dict[str, float]) -> None:
    if writer is None:
        return

    writer.add_scalar("Val/mAP", metrics["mAP"], epoch)
    writer.add_scalar("Val/AP50", metrics["mAP_50"], epoch)
    writer.add_scalar("Val/AP75", metrics["mAP_75"], epoch)
    writer.add_scalar("Val/mAP_small", metrics["mAP_small"], epoch)
    writer.add_scalar("Val/mAP_medium", metrics["mAP_medium"], epoch)
    writer.add_scalar("Val/mAP_large", metrics["mAP_large"], epoch)


def print_validation_metrics(metrics: dict[str, float], epoch: int | None = None, total_epochs: int | None = None) -> None:
    prefix = ""
    if epoch is not None:
        current_epoch = epoch + 1
        prefix = f"Epoch [{current_epoch}/{total_epochs}] " if total_epochs is not None else f"Epoch {current_epoch} "
    print(
        f"{prefix}"
        f"mAP: {metrics['mAP']:.4f}, "
        f"AP50: {metrics['mAP_50']:.4f}, "
        f"AP75: {metrics['mAP_75']:.4f}, "
        f"small: {metrics['mAP_small']:.4f}, "
        f"medium: {metrics['mAP_medium']:.4f}, "
        f"large: {metrics['mAP_large']:.4f}"
    )


def format_onnx_shape(shape) -> str:
    if not shape:
        return "unknown"
    return "x".join("?" if dim is None else str(dim) for dim in shape)


def print_onnx_evaluation_summary(args, onnx_path: str, session: ort.InferenceSession, result_path: Path) -> None:
    input_meta = session.get_inputs()[0]
    output_summaries = [f"{output.name}({format_onnx_shape(output.shape)})" for output in session.get_outputs()]
    print("开始 ONNX 验证:")
    print(f"- backbone: {args.backbone}")
    print(f"- onnx_path: {onnx_path}")
    print(f"- providers: {session.get_providers()}")
    print(f"- dataset: {DEFAULT_DATASET_NAME} / val")
    print(f"- img_size: {args.img_size}")
    print(f"- dbox_ratio: min={args.dbox_min_ratio}, max={args.dbox_max_ratio}")
    print(f"- input: {input_meta.name}({format_onnx_shape(input_meta.shape)})")
    print(f"- outputs: {', '.join(output_summaries)}")
    print(f"- num_visualizations: {args.num_visualizations}")
    print(f"- result_file: {result_path}")
    print(f"- csv_file: {args.csv_file}")


def build_coco_predictions(decoded, image_id: int, image_width: int, image_height: int) -> list[dict[str, int | float | list[float]]]:
    p_boxes, p_labels, p_scores = decoded
    if p_boxes.numel() == 0:
        return []

    scaled_boxes = p_boxes.clone()
    scaled_boxes[:, [0, 2]] *= image_width
    scaled_boxes[:, [1, 3]] *= image_height
    scaled_boxes[:, 2] -= scaled_boxes[:, 0]
    scaled_boxes[:, 3] -= scaled_boxes[:, 1]

    return [
        {
            "image_id": image_id,
            "category_id": label,
            "bbox": [round(x, 3) for x in box],
            "score": round(score, 5),
        }
        for box, label, score in zip(scaled_boxes.tolist(), p_labels.tolist(), p_scores.tolist())
    ]


def run_coco_evaluation(coco_gt, predictions, image_ids: Sequence[int] | None = None) -> dict[str, float] | None:
    if not predictions:
        return None

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    if image_ids is not None:
        coco_eval.params.imgIds = sorted(image_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return build_validation_metrics(coco_eval.stats)


# -----------------------------------------------------------------------------
# Training-time validation helpers
# -----------------------------------------------------------------------------


def build_torch_validation_visualization_dir(args, epoch: int) -> Path | None:
    num_visualizations = max(0, int(getattr(args, "num_visualizations", 0)))
    if num_visualizations <= 0:
        return None

    viz_dir = Path("viz_results") / f"ssd320_{args.backbone}" / f"epoch_{epoch + 1}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    return viz_dir


def collect_torch_predictions(
    eval_model: torch.nn.Module,
    val_loader,
    eval_encoder: Encoder,
    category_names: Sequence[str] | None,
    device: torch.device,
    viz_dir: Path | None,
    num_visualizations: int,
) -> list[dict[str, int | float | list[float]]]:
    predictions: list[dict[str, int | float | list[float]]] = []
    viz_count = 0

    with torch.no_grad():
        for v_images, v_boxes_list, v_labels_list, v_img_ids in val_loader:
            v_images = v_images.to(device)
            locs, confs = eval_model(v_images)
            decoded_batch = eval_encoder.decode_batch(locs, confs)

            if viz_count < num_visualizations and viz_dir is not None:
                for batch_index, decoded in enumerate(decoded_batch):
                    if viz_count >= num_visualizations:
                        break
                    p_boxes, p_labels, p_scores = decoded
                    visualize_sample(
                        v_images[batch_index],
                        v_boxes_list[batch_index],
                        v_labels_list[batch_index],
                        p_boxes,
                        p_labels,
                        p_scores,
                        category_names,
                        str(viz_dir / f"val_{viz_count}.jpg"),
                    )
                    viz_count += 1

            for batch_index, decoded in enumerate(decoded_batch):
                img_id = v_img_ids[batch_index]
                if torch.is_tensor(img_id):
                    img_id = img_id.item()
                predictions.extend(
                    build_coco_predictions(
                        decoded,
                        image_id=img_id,
                        image_width=v_images.shape[-1],
                        image_height=v_images.shape[-2],
                    )
                )

    return predictions


def evaluate_torch_model(
    ssd_model: torch.nn.Module,
    epoch: int,
    val_loader,
    eval_encoder: Encoder,
    coco_gt,
    category_names: Sequence[str] | None,
    device: torch.device,
    writer: SummaryWriter | None,
    args,
) -> dict[str, float] | None:
    print(f"Epoch {epoch + 1} 结束, 开始评估验证集 mAP 并进行可视化...")
    eval_model = ssd_model.module if hasattr(ssd_model, "module") else ssd_model
    eval_model.eval()

    num_visualizations = max(0, int(getattr(args, "num_visualizations", 0)))
    viz_dir = build_torch_validation_visualization_dir(args, epoch)

    predictions = collect_torch_predictions(
        eval_model=eval_model,
        val_loader=val_loader,
        eval_encoder=eval_encoder,
        category_names=category_names,
        device=device,
        viz_dir=viz_dir,
        num_visualizations=num_visualizations,
    )

    if not predictions:
        print(f"Epoch {epoch + 1}: 未检测到任何目标，results_coco 为空。")
        return None

    print(f"收集到 {len(predictions)} 条预测结果，正在计算 mAP...")
    metrics = run_coco_evaluation(coco_gt, predictions)
    if metrics is None:
        return None
    print_validation_metrics(metrics, epoch=epoch, total_epochs=args.epochs)
    write_validation_metrics(writer, epoch, metrics)
    return metrics


def run_torch_validation_if_needed(
    ssd_model: torch.nn.Module,
    epoch: int,
    val_loader,
    eval_encoder: Encoder,
    coco_gt,
    category_names: Sequence[str] | None,
    device: torch.device,
    writer: SummaryWriter | None,
    args,
    main_process: bool,
) -> dict[str, float] | None:
    if not main_process or (epoch + 1) % args.eval_interval != 0:
        return None

    return evaluate_torch_model(
        ssd_model,
        epoch,
        val_loader,
        eval_encoder,
        coco_gt,
        category_names,
        device,
        writer,
        args,
    )


# -----------------------------------------------------------------------------
# ONNX evaluation entry points
# -----------------------------------------------------------------------------


def resolve_onnx_result_path(args, onnx_path: str) -> Path:
    return Path(args.result_file or f"val_results/{Path(onnx_path).stem}_predictions.json")


def evaluate_exported_onnx(args) -> dict[str, float | str]:
    val_dataset = load_dataset(DEFAULT_DATASET_NAME, split="val", cache_dir=args.cache_dir)
    onnx_path = resolve_onnx_model_path(args)
    session = build_onnx_session(onnx_path, args.provider)
    encoder = Encoder(dboxes320_coco(min_ratio=args.dbox_min_ratio, max_ratio=args.dbox_max_ratio))
    result_path = resolve_onnx_result_path(args, onnx_path)

    print_onnx_evaluation_summary(args, onnx_path, session, result_path)

    if args.num_visualizations > 0:
        save_onnx_visualizations(args, session, encoder, val_dataset)

    gt_file = get_coco_ground_truth(val_dataset)
    metrics = evaluate_onnx_dataset(args, val_dataset, session, encoder, gt_file, result_path)
    export_metrics_to_csv(args, metrics, onnx_path, result_path)
    return metrics


def resolve_onnx_model_path(args) -> str:
    if args.onnx_path:
        onnx_path = Path(args.onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX 模型文件不存在: {onnx_path}")
        return str(onnx_path)

    local_path = Path("weights") / f"ssd320_{args.backbone}.onnx"
    if local_path.exists():
        return str(local_path)

    print(f"未找到本地 ONNX 模型: {local_path}，尝试自动下载...")
    downloaded_path = hf_hub_download(repo_id=args.model_repo_id, filename=local_path.name)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded_path, local_path)
    print(f"已自动下载 ONNX 模型到: {local_path}")
    return str(local_path)


def build_onnx_session(onnx_model_path: str, provider: str) -> ort.InferenceSession:
    available_providers = set(ort.get_available_providers())
    if provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif provider == "cuda":
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError("当前 onnxruntime 环境不支持 CUDAExecutionProvider。")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    print(f"使用 ONNX Runtime providers: {providers}")
    return ort.InferenceSession(onnx_model_path, ort.SessionOptions(), providers=providers)


# -----------------------------------------------------------------------------
# ONNX inference and visualization helpers
# -----------------------------------------------------------------------------


def preprocess_onnx_image(image, img_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if image.mode != "RGB":
        image = image.convert("RGB")

    resized_image = image.resize((img_size, img_size))
    image_tensor = tvf.to_tensor(resized_image)
    image_tensor = tvf.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return image_tensor, image_tensor.unsqueeze(0).numpy()


def run_onnx_inference(image, session: ort.InferenceSession, encoder: Encoder, args):
    image_tensor, image_batch = preprocess_onnx_image(image, args.img_size)
    decoded = decode_onnx_predictions(session, encoder, image_batch, args)
    return image_tensor, decoded


def extract_ground_truth(item, img_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    objects = item["objects"]
    if len(objects["bbox"]) == 0:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

    image_width, image_height = item["image"].size
    boxes = torch.tensor(objects["bbox"], dtype=torch.float32)
    labels = torch.tensor(objects["category"], dtype=torch.long) + 1
    boxes[:, [0, 2]] *= img_size / image_width
    boxes[:, [1, 3]] *= img_size / image_height
    return boxes, labels


def save_onnx_visualizations(args, session: ort.InferenceSession, encoder: Encoder, val_dataset) -> None:
    category_names = extract_category_names(val_dataset, split=None, include_background=True)
    save_dir = Path("viz_results") / "onnx_val" / args.backbone
    save_dir.mkdir(parents=True, exist_ok=True)

    for index in range(min(args.num_visualizations, len(val_dataset))):
        item = val_dataset[index]
        image_tensor, decoded = run_onnx_inference(item["image"], session, encoder, args)
        gt_boxes, gt_labels = extract_ground_truth(item, args.img_size)
        p_boxes, p_labels, p_scores = decoded
        visualize_sample(
            image_tensor,
            gt_boxes,
            gt_labels,
            p_boxes,
            p_labels,
            p_scores,
            category_names,
            str(save_dir / f"vis_{index}.jpg"),
        )


# -----------------------------------------------------------------------------
# ONNX prediction collection and COCO evaluation
# -----------------------------------------------------------------------------


def collect_onnx_predictions(args, val_dataset, session: ort.InferenceSession, encoder: Encoder):
    predictions: list[dict[str, int | float | list[float]]] = []
    image_ids: list[int] = []
    inference_time = 0.0
    input_name = session.get_inputs()[0].name
    num_workers = max(0, int(getattr(args, "num_workers", 0)))
    preprocess_batch_size = max(1, int(getattr(args, "preprocess_batch_size", 1)))

    if num_workers > 0:
        eval_loader = DataLoader(
            ONNXEvalDataset(val_dataset, args.img_size),
            batch_size=preprocess_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=getattr(args, "prefetch_factor", 2),
        )

        with tqdm(total=len(val_dataset), desc="Evaluating ONNX") as progress:
            for image_batch, batch_image_ids in eval_loader:
                for image_tensor, image_id in zip(image_batch, batch_image_ids):
                    step_start = time.time()
                    locs, confs = session.run(["boxes", "scores"], {input_name: image_tensor.unsqueeze(0).numpy()})
                    decoded = encoder.decode_batch(
                        torch.from_numpy(locs),
                        torch.from_numpy(confs),
                        criteria=args.decode_iou_threshold,
                        max_output=args.max_output,
                    )[0]
                    inference_time += time.time() - step_start
                    image_id = int(image_id)
                    image_ids.append(image_id)
                    predictions.extend(build_coco_predictions(decoded, image_id, args.img_size, args.img_size))
                    progress.update(1)
        return predictions, image_ids, inference_time

    for item in tqdm(val_dataset, desc="Evaluating ONNX"):
        step_start = time.time()
        _, decoded = run_onnx_inference(item["image"], session, encoder, args)
        inference_time += time.time() - step_start
        image_id = item["image_id"]
        image_ids.append(image_id)
        predictions.extend(build_coco_predictions(decoded, image_id, args.img_size, args.img_size))

    return predictions, image_ids, inference_time


def save_predictions_json(result_path: Path, predictions: list[dict[str, int | float | list[float]]]) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(predictions, file)


def evaluate_onnx_dataset(args, val_dataset, session: ort.InferenceSession, encoder: Encoder, gt_file: str, result_path: Path) -> dict[str, float | str]:
    start_time = time.time()
    predictions, image_ids, inference_time = collect_onnx_predictions(args, val_dataset, session, encoder)

    save_predictions_json(result_path, predictions)

    coco_gt = load_coco_ground_truth_api(gt_file)
    metrics = run_coco_evaluation(coco_gt, predictions, image_ids=image_ids)
    if metrics is None:
        raise RuntimeError("ONNX 验证未产生任何预测结果，无法计算 COCO 指标。")

    total_time = time.time() - start_time
    count = len(val_dataset)
    metrics.update(
        {
            "result_file": str(result_path),
            "fps_total": count / total_time if total_time > 0 else 0.0,
            "fps_inference": count / inference_time if inference_time > 0 else 0.0,
        }
    )
    return metrics


def decode_onnx_predictions(session: ort.InferenceSession, encoder: Encoder, image_batch, args):
    input_name = session.get_inputs()[0].name
    locs, confs = session.run(["boxes", "scores"], {input_name: image_batch})
    return encoder.decode_batch(
        torch.from_numpy(locs),
        torch.from_numpy(confs),
        criteria=args.decode_iou_threshold,
        max_output=args.max_output,
    )[0]


# -----------------------------------------------------------------------------
# Result export helpers
# -----------------------------------------------------------------------------


def export_metrics_to_csv(args, metrics: dict[str, float | str], onnx_path: str, result_path: Path) -> None:
    csv_path = Path(args.csv_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "timestamp",
        "backbone",
        "onnx_path",
        "provider",
        "img_size",
        "dbox_min_ratio",
        "dbox_max_ratio",
        "mAP",
        "AP50",
        "AP75",
        "mAP_small",
        "mAP_medium",
        "mAP_large",
        "fps_total",
        "fps_inference",
        "result_file",
    ]

    row = [
        datetime.now().isoformat(timespec="seconds"),
        args.backbone,
        onnx_path,
        args.provider,
        args.img_size,
        args.dbox_min_ratio,
        args.dbox_max_ratio,
        metrics["mAP"],
        metrics["mAP_50"],
        metrics["mAP_75"],
        metrics["mAP_small"],
        metrics["mAP_medium"],
        metrics["mAP_large"],
        metrics["fps_total"],
        metrics["fps_inference"],
        str(result_path),
    ]
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)
    print(f"验证指标已写入 CSV: {csv_path}")
