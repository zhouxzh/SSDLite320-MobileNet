from __future__ import annotations

import argparse
import sys

from ssdlite320.config import DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_REPO_ID
from ssdlite320.runtime import (
    build_validation_resources,
    export_onnx_model,
    find_best_checkpoint,
    find_latest_checkpoint,
    initialize_distributed_runtime,
    load_training_data,
    shutdown_distributed_runtime,
)


# -----------------------------------------------------------------------------
# Command-line parsing helpers
# -----------------------------------------------------------------------------


LEGACY_ARG_MAPPINGS = {
    "--export-best-onnx": "--export-onnx-from-best-checkpoint",
    "--no-export-best-onnx": "--no-export-onnx-from-best-checkpoint",
}


def validate_default_box_ratios(min_ratio: float, max_ratio: float) -> None:
    if not (0.0 < min_ratio < max_ratio < 1.0):
        raise ValueError("--dbox-min-ratio 与 --dbox-max-ratio 必须满足 0 < min < max < 1。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or validate SSDLite320 models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    build_train_parser(
        subparsers.add_parser(
            "train",
            help="Train SSDLite320",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    )
    build_val_parser(
        subparsers.add_parser(
            "val",
            help="Evaluate an exported ONNX model on COCO val",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    )
    return parser


def build_train_parser(parser: argparse.ArgumentParser) -> None:
    core_group = parser.add_argument_group("core training")
    core_group.add_argument("--backbone", type=str, default="mobilenetv4_conv_small", help="Model backbone")
    core_group.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True, help="Use timm pretrained backbone weights")
    core_group.add_argument("--epochs", type=int, default=400, help="Total training epochs")
    core_group.add_argument("--batch-size", type=int, default=64, help="Batch size per process; effective LR is scaled by batch_size * world_size")
    core_group.add_argument("--lr", type=float, default=0.003, help="Base learning rate before linear scaling")
    core_group.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    core_group.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")

    schedule_group = parser.add_argument_group("schedule and validation")
    schedule_group.add_argument("--freeze-backbone-epochs", type=int, default=5, help="Freeze backbone for the first N epochs")
    schedule_group.add_argument("--freeze-warmup-epochs", type=int, default=1, help="Linear warmup epochs used only during the freeze-backbone phase")
    schedule_group.add_argument("--warmup-epochs", type=int, default=3, help="Linear warmup epochs for full-model training before cosine annealing")
    schedule_group.add_argument("--hold-ratio", type=float, default=0, help="Constant-LR hold ratio applied to the remaining full-model epochs after warmup and before cosine annealing")
    schedule_group.add_argument(
        "--cosine-min-lr-ratio",
        type=float,
        default=0.05,
        help="Minimum cosine LR expressed as a ratio of the effective base LR",
    )
    schedule_group.add_argument("--eval-interval", type=int, default=10, help="Run validation every N epochs")
    schedule_group.add_argument("--num-visualizations", type=int, default=0, help="How many validation images to save per evaluation round")
    schedule_group.add_argument("--patience", type=int, default=20, help="Early stopping patience in eval rounds")
    schedule_group.add_argument("--min-delta", type=float, default=1e-4, help="Minimum mAP improvement to reset patience")

    model_group = parser.add_argument_group("model details")
    model_group.add_argument("--num-classes", type=int, default=81, help="Number of classes including background")
    model_group.add_argument("--dbox-min-ratio", type=float, default=0.1, help="Minimum default-box ratio")
    model_group.add_argument("--dbox-max-ratio", type=float, default=0.9, help="Maximum default-box ratio")

    data_group = parser.add_argument_group("data loading")
    data_group.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading per process")
    data_group.add_argument("--prefetch-factor", type=int, default=1, help="Dataloader prefetch factor")
    data_group.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True, help="Pin dataloader memory for faster host-to-device transfer")
    data_group.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="Enable training data augmentation")

    runtime_group = parser.add_argument_group("runtime and export")
    runtime_group.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    runtime_group.add_argument("--ddp", action=argparse.BooleanOptionalAction, default=True, help="Enable distributed data parallel")
    runtime_group.add_argument("--restart", action="store_true", help="Resume training from the last checkpoint")
    runtime_group.add_argument(
        "--export-onnx-from-best-checkpoint",
        dest="export_onnx_from_best_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Training exports ONNX by default; enable this to export from the best checkpoint instead of the final in-memory model",
    )


def build_val_parser(parser: argparse.ArgumentParser) -> None:
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--backbone", default="mobilenetv4_conv_small", help="Backbone name used in the exported ONNX filename")
    model_group.add_argument("--onnx-path", default=None, help="Explicit ONNX model path; overrides --backbone lookup")
    model_group.add_argument("--model-repo-id", default=DEFAULT_MODEL_REPO_ID, help="Hugging Face repo used when the ONNX file must be downloaded")

    eval_group = parser.add_argument_group("evaluation")
    eval_group.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto", help="ONNX Runtime execution provider")
    eval_group.add_argument("--cache-dir", default="./data", help="Dataset cache directory")
    eval_group.add_argument("--img-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Inference image size")
    eval_group.add_argument("--num-visualizations", type=int, default=0, help="How many validation images to save for visualization")
    eval_group.add_argument("--decode-iou-threshold", type=float, default=0.5, help="IoU threshold used during decode NMS")
    eval_group.add_argument("--max-output", type=int, default=200, help="Maximum predictions kept per image")

    default_box_group = parser.add_argument_group("default boxes")
    default_box_group.add_argument("--dbox-min-ratio", type=float, default=0.1, help="Minimum default-box ratio used for decoding")
    default_box_group.add_argument("--dbox-max-ratio", type=float, default=0.9, help="Maximum default-box ratio used for decoding")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--result-file", default=None, help="Path to save COCO-format predictions")
    output_group.add_argument("--csv-file", default="reports/onnx_validation_metrics.csv", help="CSV file used to accumulate validation metrics")


def normalize_command_argv(argv: list[str] | None) -> list[str]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    raw_args = [LEGACY_ARG_MAPPINGS.get(arg, arg) for arg in raw_args]
    if not raw_args or raw_args[0] not in {"train", "val"}:
        return ["train", *raw_args]
    return raw_args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    normalized_argv = normalize_command_argv(argv)
    return parser.parse_args(normalized_argv)


# -----------------------------------------------------------------------------
# Argument validation
# -----------------------------------------------------------------------------


def validate_train_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs 必须大于 0。")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0。")
    if args.lr <= 0.0:
        raise ValueError("--lr 必须大于 0。")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay 不能为负数。")
    if not (0.0 <= args.momentum < 1.0):
        raise ValueError("--momentum 必须满足 0 <= momentum < 1。")
    if args.freeze_backbone_epochs < 0:
        raise ValueError("--freeze-backbone-epochs 不能为负数。")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs 不能为负数。")
    if args.freeze_warmup_epochs < 0:
        raise ValueError("--freeze-warmup-epochs 不能为负数。")
    if not (0.0 <= args.hold_ratio < 1.0):
        raise ValueError("--hold-ratio 必须满足 0 <= ratio < 1。")
    if args.patience < 0:
        raise ValueError("--patience 不能为负数。")
    if args.num_workers < 0:
        raise ValueError("--num-workers 不能为负数。")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor 必须大于 0。")
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval 必须大于 0。")
    if args.num_visualizations < 0:
        raise ValueError("--num-visualizations 不能为负数。")
    if args.freeze_warmup_epochs > args.freeze_backbone_epochs:
        raise ValueError("--freeze-warmup-epochs 不能大于 --freeze-backbone-epochs。")
    if args.epochs < args.freeze_backbone_epochs:
        raise ValueError("--freeze-backbone-epochs 不能大于 --epochs。")
    validate_default_box_ratios(args.dbox_min_ratio, args.dbox_max_ratio)
    if not (0.0 <= args.cosine_min_lr_ratio < 1.0):
        raise ValueError("--cosine-min-lr-ratio 必须满足 0 <= ratio < 1。")


def validate_val_args(args: argparse.Namespace) -> None:
    if args.img_size <= 0:
        raise ValueError("--img-size 必须大于 0。")
    if args.num_visualizations < 0:
        raise ValueError("--num-visualizations 不能为负数。")
    if not (0.0 <= args.decode_iou_threshold <= 1.0):
        raise ValueError("--decode-iou-threshold 必须满足 0 <= threshold <= 1。")
    if args.max_output <= 0:
        raise ValueError("--max-output 必须大于 0。")
    validate_default_box_ratios(args.dbox_min_ratio, args.dbox_max_ratio)


# -----------------------------------------------------------------------------
# Command handlers
# -----------------------------------------------------------------------------


def run_train_command(args: argparse.Namespace) -> None:
    """Top-level CLI entry for training.

    阅读顺序建议：
    1. 先看这里，了解命令入口做了哪些准备。
    2. 再看 ssdlite320.train.run_training_plan，了解训练阶段如何串起来。
    """
    from ssdlite320.train import run_training_plan, setup_training

    validate_train_args(args)
    args = initialize_distributed_runtime(args)

    is_main_process = (not args.distributed) or args.rank == 0
    if args.distributed:
        print(f"[Rank {args.rank}] DDP 初始化完成: local_rank={args.local_rank}, world_size={args.world_size}")

    if args.export_onnx_from_best_checkpoint:
        best_checkpoint = find_best_checkpoint(args.backbone)
        if best_checkpoint is None:
            raise FileNotFoundError(f"未找到 best checkpoint: checkpoints/ssd320_{args.backbone}_best.pth")
        train_state = setup_training(args, resume_checkpoint=best_checkpoint)
        if train_state.writer is not None:
            train_state.writer.close()
        if is_main_process:
            export_onnx_model(args, train_state, checkpoint_path=best_checkpoint)
        shutdown_distributed_runtime()
        return

    full_dataset, train_loader, val_loader = load_training_data(args, is_main_process)
    coco_gt, category_names = build_validation_resources(full_dataset)
    resume_checkpoint, start_epoch = find_latest_checkpoint(args.backbone) if args.restart else (None, 0)
    train_state = setup_training(args, resume_checkpoint=resume_checkpoint)
    run_training_plan(args, train_loader, val_loader, coco_gt, category_names, train_state, start_epoch)

    if train_state.writer is not None:
        train_state.writer.close()
    if is_main_process:
        export_onnx_model(args, train_state)
    shutdown_distributed_runtime()


def run_val_command(args: argparse.Namespace) -> None:
    from ssdlite320.eval import evaluate_exported_onnx

    validate_val_args(args)
    metrics = evaluate_exported_onnx(args)
    print(
        "验证完成: "
        f"mAP={metrics['mAP']:.4f}, AP50={metrics['mAP_50']:.4f}, AP75={metrics['mAP_75']:.4f}, "
        f"small={metrics['mAP_small']:.4f}, medium={metrics['mAP_medium']:.4f}, large={metrics['mAP_large']:.4f}"
    )


def dispatch_command(args: argparse.Namespace) -> None:
    if args.command == "val":
        run_val_command(args)
    else:
        run_train_command(args)


# -----------------------------------------------------------------------------
# Program entry
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dispatch_command(args)


if __name__ == "__main__":
    main()
