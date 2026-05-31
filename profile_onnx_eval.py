from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import onnxruntime as ort
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssdlite320.config import DEFAULT_DATASET_NAME, DEFAULT_IMAGE_SIZE
from ssdlite320.encoder import Encoder, dboxes320_coco
from ssdlite320.eval import ONNXEvalDataset, build_coco_predictions, preprocess_onnx_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the ONNX COCO eval pipeline without running full COCO metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx-path", default=None, help="ONNX model to profile. Defaults to the first weights/*.onnx file.")
    parser.add_argument("--provider", choices=["cuda", "cpu", "auto"], default="cuda", help="ONNX Runtime provider")
    parser.add_argument("--cache-dir", default="./data", help="Dataset cache directory")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Inference image size")
    parser.add_argument("--limit", type=int, default=200, help="Number of validation images to profile; use 0 for all images")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup ONNX runs before profiling")
    parser.add_argument("--decode-iou-threshold", type=float, default=0.5, help="IoU threshold used during decode NMS")
    parser.add_argument("--max-output", type=int, default=200, help="Maximum predictions kept per image")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers used for parallel dataset item loading and preprocessing")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Dataloader prefetch factor when --num-workers > 0")
    parser.add_argument("--preprocess-batch-size", type=int, default=16, help="Dataloader batch size for preprocessed images")
    parser.add_argument("--dbox-min-ratio", type=float, default=0.1, help="Minimum default-box ratio used for decoding")
    parser.add_argument("--dbox-max-ratio", type=float, default=0.9, help="Maximum default-box ratio used for decoding")
    return parser.parse_args()


def resolve_onnx_path(path_arg: str | None) -> Path:
    if path_arg:
        onnx_path = Path(path_arg)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        return onnx_path

    models = sorted(Path("weights").glob("*.onnx"))
    if not models:
        raise FileNotFoundError("No ONNX models found under weights/")
    return models[0]


def build_session(onnx_path: Path, provider: str) -> ort.InferenceSession:
    available_providers = set(ort.get_available_providers())
    if provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif provider == "cuda":
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError("CUDAExecutionProvider is not available.")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    if provider == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDAExecutionProvider failed to initialize; refusing CPU fallback.")
    return session


def run_warmup(session: ort.InferenceSession, image_batch, warmup: int) -> float:
    if warmup <= 0:
        return 0.0

    input_name = session.get_inputs()[0].name
    start = time.perf_counter()
    for _ in range(warmup):
        session.run(["boxes", "scores"], {input_name: image_batch})
    return time.perf_counter() - start


def print_summary(timers: dict[str, float], counts: dict[str, int], num_images: int) -> None:
    measured_total = timers["profile_total"]
    stage_names = [
        "dataset_item",
        "preprocess",
        "loader_wait",
        "onnx_run",
        "tensor_wrap",
        "decode_nms",
        "coco_predictions",
    ]

    print("\nProfile summary")
    print(f"- images: {num_images}")
    print(f"- measured total: {measured_total:.4f}s")
    print(f"- throughput: {num_images / measured_total:.2f} images/s" if measured_total > 0 else "- throughput: n/a")
    print()
    print(f"{'stage':<18} {'total_s':>10} {'ms/img':>10} {'percent':>9} {'count':>8}")
    print("-" * 59)
    for name in stage_names:
        total = timers[name]
        percent = 100.0 * total / measured_total if measured_total > 0 else 0.0
        per_image = 1000.0 * total / num_images if num_images > 0 else 0.0
        print(f"{name:<18} {total:10.4f} {per_image:10.3f} {percent:8.1f}% {counts[name]:8d}")

    stage_total = sum(timers[name] for name in stage_names)
    overhead = measured_total - stage_total
    print("-" * 59)
    print(f"{'unattributed':<18} {overhead:10.4f} {1000.0 * overhead / num_images:10.3f} "
          f"{100.0 * overhead / measured_total if measured_total > 0 else 0.0:8.1f}%")


def main() -> None:
    args = parse_args()
    onnx_path = resolve_onnx_path(args.onnx_path)

    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"ONNX path: {onnx_path}")

    session_start = time.perf_counter()
    session = build_session(onnx_path, args.provider)
    session_time = time.perf_counter() - session_start
    print(f"Active providers: {session.get_providers()}")
    print(f"Session init: {session_time:.4f}s")

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()
    print(f"Input: {input_meta.name} {input_meta.shape} {input_meta.type}")
    print(f"Outputs: {[(output.name, output.shape, output.type) for output in output_meta]}")

    dataset_start = time.perf_counter()
    val_dataset = load_dataset(DEFAULT_DATASET_NAME, split="val", cache_dir=args.cache_dir)
    dataset_load_time = time.perf_counter() - dataset_start
    num_images = len(val_dataset) if args.limit == 0 else min(args.limit, len(val_dataset))
    if num_images <= 0:
        raise ValueError("--limit must be greater than 0, or use 0 for the full dataset.")
    print(f"Dataset: {DEFAULT_DATASET_NAME} / val")
    print(f"Dataset load: {dataset_load_time:.4f}s")
    print(f"Profile images: {num_images}")

    encoder = Encoder(dboxes320_coco(min_ratio=args.dbox_min_ratio, max_ratio=args.dbox_max_ratio))
    input_name = session.get_inputs()[0].name
    timers: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    predictions_count = 0

    first_item = val_dataset[0]
    _, first_batch = preprocess_onnx_image(first_item["image"], args.img_size)
    warmup_time = run_warmup(session, first_batch, args.warmup)
    print(f"Warmup: {args.warmup} runs, {warmup_time:.4f}s")

    profile_start = time.perf_counter()
    if args.num_workers > 0:
        eval_dataset = ONNXEvalDataset(val_dataset.select(range(num_images)), args.img_size)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.preprocess_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
        loader_iter = iter(eval_loader)
        progress = tqdm(total=num_images, desc="Profiling ONNX eval")
        while progress.n < num_images:
            step_start = time.perf_counter()
            image_batch, batch_image_ids = next(loader_iter)
            timers["loader_wait"] += time.perf_counter() - step_start
            counts["loader_wait"] += len(batch_image_ids)

            for image_tensor, image_id in zip(image_batch, batch_image_ids):
                step_start = time.perf_counter()
                locs, confs = session.run(["boxes", "scores"], {input_name: image_tensor.unsqueeze(0).numpy()})
                timers["onnx_run"] += time.perf_counter() - step_start
                counts["onnx_run"] += 1

                step_start = time.perf_counter()
                locs_t = torch.from_numpy(locs)
                confs_t = torch.from_numpy(confs)
                timers["tensor_wrap"] += time.perf_counter() - step_start
                counts["tensor_wrap"] += 1

                step_start = time.perf_counter()
                decoded = encoder.decode_batch(
                    locs_t,
                    confs_t,
                    criteria=args.decode_iou_threshold,
                    max_output=args.max_output,
                )[0]
                timers["decode_nms"] += time.perf_counter() - step_start
                counts["decode_nms"] += 1

                step_start = time.perf_counter()
                predictions = build_coco_predictions(decoded, int(image_id), args.img_size, args.img_size)
                predictions_count += len(predictions)
                timers["coco_predictions"] += time.perf_counter() - step_start
                counts["coco_predictions"] += 1
                progress.update(1)
        progress.close()
    else:
        for index in tqdm(range(num_images), desc="Profiling ONNX eval"):
            step_start = time.perf_counter()
            item = val_dataset[index]
            timers["dataset_item"] += time.perf_counter() - step_start
            counts["dataset_item"] += 1

            step_start = time.perf_counter()
            _, image_batch = preprocess_onnx_image(item["image"], args.img_size)
            timers["preprocess"] += time.perf_counter() - step_start
            counts["preprocess"] += 1

            step_start = time.perf_counter()
            locs, confs = session.run(["boxes", "scores"], {input_name: image_batch})
            timers["onnx_run"] += time.perf_counter() - step_start
            counts["onnx_run"] += 1

            step_start = time.perf_counter()
            locs_t = torch.from_numpy(locs)
            confs_t = torch.from_numpy(confs)
            timers["tensor_wrap"] += time.perf_counter() - step_start
            counts["tensor_wrap"] += 1

            step_start = time.perf_counter()
            decoded = encoder.decode_batch(
                locs_t,
                confs_t,
                criteria=args.decode_iou_threshold,
                max_output=args.max_output,
            )[0]
            timers["decode_nms"] += time.perf_counter() - step_start
            counts["decode_nms"] += 1

            step_start = time.perf_counter()
            predictions = build_coco_predictions(decoded, item["image_id"], args.img_size, args.img_size)
            predictions_count += len(predictions)
            timers["coco_predictions"] += time.perf_counter() - step_start
            counts["coco_predictions"] += 1

    timers["profile_total"] = time.perf_counter() - profile_start
    print_summary(timers, counts, num_images)
    print(f"\nPredictions built: {predictions_count}")


if __name__ == "__main__":
    main()
