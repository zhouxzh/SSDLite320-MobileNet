from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from .config import DEFAULT_ONNX_OPSET_VERSION, HYBRID_MOBILENETV4_ONNX_OPSET_VERSION

if TYPE_CHECKING:
    import argparse
    from torch.utils.data import DataLoader


# -----------------------------------------------------------------------------
# Distributed runtime helpers
# -----------------------------------------------------------------------------


def initialize_distributed_runtime(args: "argparse.Namespace") -> "argparse.Namespace":
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    args.world_size = world_size
    args.rank = rank
    args.local_rank = local_rank
    args.distributed = False

    if args.ddp:
        if not str(args.device).startswith("cuda"):
            raise RuntimeError("启用 DDP 时 --device 必须是 cuda。")
        if not torch.cuda.is_available():
            raise RuntimeError("已启用 CUDA/DDP，但当前环境未检测到 CUDA，程序退出。")
        if world_size <= 1:
            raise RuntimeError("已启用 DDP，但 WORLD_SIZE<=1。请使用 torchrun --nproc_per_node=2（或更多）启动。")
        torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                device_id=torch.device("cuda", local_rank),
            )
        except TypeError:
            dist.init_process_group(backend="nccl", init_method="env://")
        args.distributed = True
        args.device = "cuda"
    else:
        if world_size > 1:
            raise RuntimeError("检测到 WORLD_SIZE>1 但 --ddp 被禁用。请启用 --ddp 或改用单进程启动。")
        if str(args.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("已指定 --device cuda，但当前环境未检测到 CUDA，程序退出。")

    return args


def shutdown_distributed_runtime() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def sync_early_stop_flag(should_stop: bool, distributed: bool, device) -> bool:
    if not distributed:
        return should_stop

    import torch
    import torch.distributed as dist

    stop_tensor = torch.tensor(1 if should_stop else 0, device=device, dtype=torch.int)
    dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
    return stop_tensor.item() > 0


def attach_distributed_sampler(train_loader: "DataLoader", args: "argparse.Namespace") -> "DataLoader":
    if not args.distributed:
        return train_loader

    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    train_sampler = DistributedSampler(
        train_loader.dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        drop_last=False,
    )
    return DataLoader(
        train_loader.dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=train_loader.collate_fn,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )


# -----------------------------------------------------------------------------
# Dataset and validation resource helpers
# -----------------------------------------------------------------------------


def load_training_data(args: "argparse.Namespace", is_main_process: bool):
    from .data_hf import download_and_load_coco, get_train_loader, get_val_dataloader

    if is_main_process:
        print("Loading training data...")
    full_dataset = download_and_load_coco()
    train_loader = get_train_loader(full_dataset, args.batch_size, num_workers=args.num_workers, args=args)
    train_loader = attach_distributed_sampler(train_loader, args)
    if is_main_process:
        print("Loading validation data...")
    val_loader = get_val_dataloader(full_dataset, args.batch_size, num_workers=args.num_workers, args=args)
    return full_dataset, train_loader, val_loader


def build_validation_resources(full_dataset):
    from .data_hf import get_coco_ground_truth, load_coco_ground_truth_api
    from .utils import extract_category_names

    gt_file = get_coco_ground_truth(full_dataset["val"])
    coco_gt = load_coco_ground_truth_api(gt_file)
    category_names = extract_category_names(full_dataset, include_background=True)
    return coco_gt, category_names


# -----------------------------------------------------------------------------
# Checkpoint and export helpers
# -----------------------------------------------------------------------------


def resolve_onnx_opset_version(backbone: str) -> int:
    if backbone.startswith("mobilenetv4_hybrid"):
        return HYBRID_MOBILENETV4_ONNX_OPSET_VERSION
    return DEFAULT_ONNX_OPSET_VERSION


def unwrap_checkpoint_state(checkpoint_state):
    if isinstance(checkpoint_state, dict) and "model_state_dict" in checkpoint_state:
        return checkpoint_state["model_state_dict"]
    return checkpoint_state


def find_latest_checkpoint(backbone: str) -> tuple[str | None, int]:
    import torch

    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        return None, 0

    last_checkpoint = os.path.join(checkpoints_dir, f"ssd320_{backbone}_last.pth")
    if os.path.exists(last_checkpoint):
        checkpoint_state = torch.load(last_checkpoint, map_location="cpu")
        epoch = checkpoint_state.get("epoch") if isinstance(checkpoint_state, dict) else None
        start_epoch = int(epoch) + 1 if epoch is not None else 0
        print(f"Found last checkpoint: {last_checkpoint} (next epoch {start_epoch})")
        return last_checkpoint, start_epoch

    pattern = re.compile(rf"ssd320_{re.escape(backbone)}_(\d+)\.pth")
    latest_epoch = -1
    latest_path = None
    for filename in os.listdir(checkpoints_dir):
        match = pattern.match(filename)
        if match and int(match.group(1)) > latest_epoch:
            latest_epoch = int(match.group(1))
            latest_path = os.path.join(checkpoints_dir, filename)

    if latest_path is None:
        print(f"No checkpoint found for backbone {backbone}.")
        return None, 0

    print(f"Found legacy latest checkpoint: {latest_path} (Epoch {latest_epoch})")
    return latest_path, latest_epoch + 1


def find_best_checkpoint(backbone: str) -> str | None:
    best_checkpoint = os.path.join("checkpoints", f"ssd320_{backbone}_best.pth")
    if os.path.exists(best_checkpoint):
        print(f"Found best checkpoint: {best_checkpoint}")
        return best_checkpoint

    print(f"Best checkpoint not found for backbone {backbone}: {best_checkpoint}")
    return None


def save_model_checkpoint(base_model, backbone: str, epoch: int, *, is_best: bool = False, metrics: dict[str, float] | None = None) -> str:
    import torch

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/ssd320_{backbone}_{'best' if is_best else 'last'}.pth"
    checkpoint_state = {
        "epoch": epoch,
        "backbone": backbone,
        "model_state_dict": base_model.state_dict(),
    }
    if metrics is not None:
        checkpoint_state["metrics"] = metrics
    torch.save(checkpoint_state, checkpoint_path)
    return checkpoint_path


def export_onnx_model(args, train_state, checkpoint_path: str | None = None) -> None:
    import torch

    export_model = train_state.base_model if checkpoint_path else (train_state.ssd_model or train_state.base_model)
    device = train_state.device
    os.makedirs("weights", exist_ok=True)
    onnx_path = f"weights/ssd320_{args.backbone}.onnx"
    onnx_opset_version = resolve_onnx_opset_version(args.backbone)

    if checkpoint_path is not None:
        checkpoint_state = torch.load(checkpoint_path, map_location=device)
        export_model.load_state_dict(unwrap_checkpoint_state(checkpoint_state))
        print(f"Loaded checkpoint for ONNX export: {checkpoint_path}")

    print(f"正在导出 ONNX 模型至 {onnx_path} (opset {onnx_opset_version})...")
    export_model = export_model.module if hasattr(export_model, "module") else export_model
    export_model.eval()
    dummy_input = torch.randn(1, 3, 320, 320).to(device)
    try:
        torch.onnx.export(
            export_model,
            dummy_input,
            onnx_path,
            verbose=False,
            input_names=["input"],
            output_names=["boxes", "scores"],
            opset_version=onnx_opset_version,
            dynamo=False,
        )

        import onnx

        exported_model = onnx.load(onnx_path)
        actual_opset = next((op.version for op in exported_model.opset_import if op.domain in {"", "ai.onnx"}), None)
        if actual_opset != onnx_opset_version:
            raise RuntimeError(
                f"ONNX 导出结果 opset={actual_opset}，不符合要求的 opset {onnx_opset_version}。"
            )
        print(f"ONNX 模型已导出至: {onnx_path}")
    except Exception as error:
        print(f"导出 ONNX 失败: {error}")
