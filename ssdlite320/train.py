"""SSDLite training pipeline.

推荐阅读顺序：
1. run_training_plan: 训练总入口，先看整体流程。
2. train_freeze_backbone_phase / train_full_model_phase: 看两个训练阶段如何衔接。
3. run_phase_epoch / train_one_epoch: 看每个 epoch 和每个 batch 做了什么。
4. setup_training / build_phase_runtime / build_phase_scheduler: 看模型、优化器、AMP、LR 调度如何构建。

函数分类：
- 顶层入口: run_training_plan
- 阶段入口: train_freeze_backbone_phase, train_full_model_phase
- 训练执行: run_phase_epoch, train_one_epoch, update_early_stopping
- 初始化与运行时: setup_training, build_phase_runtime
- 阶段配置与调度: build_freeze_phase_config, build_full_phase_config, build_phase_scheduler
- 日志与辅助: build_tensorboard_log_dir, write_training_config, log_phase_schedule
"""

import math
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.tensorboard import SummaryWriter

from .encoder import Encoder, dboxes320_coco
from .eval import run_torch_validation_if_needed
from .model import Loss, MobileNet, SSD320
from .runtime import save_model_checkpoint, sync_early_stop_flag, unwrap_checkpoint_state


# -----------------------------------------------------------------------------
# Shared training state
# -----------------------------------------------------------------------------


@dataclass
class TrainingContext:
    """Shared objects needed across the optional freeze and full-training phases."""

    distributed: bool
    main_process: bool
    writer: SummaryWriter | None
    lr: float
    device: torch.device
    eval_encoder: Encoder
    base_model: SSD320
    criterion: Loss
    best_map: float = -1.0
    no_improve: int = 0
    ssd_model: torch.nn.Module | None = None


@dataclass(frozen=True)
class PhaseExecutionConfig:
    """Describe one training phase in terms of epoch span and LR schedule span."""

    name: str
    start_epoch: int
    end_epoch: int
    warmup_epochs: int
    hold_epochs: int
    schedule_anchor_epoch: int


@dataclass
class PhaseRuntime:
    """Runtime objects that are recreated when a new training phase starts."""

    ssd_model: torch.nn.Module
    optimizer: optim.Optimizer
    scaler: torch.amp.GradScaler
    scheduler: LRScheduler | None


# -----------------------------------------------------------------------------
# Top-level training entry
# -----------------------------------------------------------------------------


def run_training_plan(
    args: Namespace,
    train_loader,
    val_loader,
    coco_gt,
    category_names: Sequence[str] | None,
    train_state: TrainingContext,
    start_epoch: int,
) -> None:
    """Top-level training entry inside ssdlite320.train.

    这层只负责决定当前应该从哪个阶段开始，以及阶段之间如何衔接。
    """
    freeze_phase_config = build_freeze_phase_config(args, start_epoch)
    if freeze_phase_config is not None:
        train_freeze_backbone_phase(
            args,
            train_loader,
            val_loader,
            coco_gt,
            category_names,
            train_state,
            start_epoch=freeze_phase_config.start_epoch,
        )

    full_start_epoch = max(start_epoch, min(args.freeze_backbone_epochs, args.epochs))
    train_full_model_phase(args, train_loader, val_loader, coco_gt, category_names, train_state, full_start_epoch)


# -----------------------------------------------------------------------------
# Phase entry points
# -----------------------------------------------------------------------------


def train_freeze_backbone_phase(
    args: Namespace,
    train_loader,
    val_loader,
    coco_gt,
    category_names: Sequence[str] | None,
    train_state: TrainingContext,
    start_epoch: int = 0,
) -> None:
    """Optional first phase: freeze backbone parameters and only train the detection head.

    LR policy for this phase:
    - First do batch-wise warmup.
    - After warmup ends, keep LR constant until the freeze phase ends.
    """
    phase_config = build_freeze_phase_config(args, start_epoch)
    if phase_config is None:
        return

    set_backbone_requires_grad(train_state.base_model, requires_grad=False)
    phase_runtime = build_phase_runtime(
        base_model=train_state.base_model,
        args=args,
        distributed=train_state.distributed,
        device=train_state.device,
        base_lr=train_state.lr,
        phase_config=phase_config,
        steps_per_epoch=len(train_loader),
    )

    print_phase_header(train_state.main_process, train_state.distributed, phase_config)
    if train_state.main_process:
        print("冻结 backbone，仅训练检测头。")
    log_phase_schedule(train_state.main_process, phase_config, args, train_state.lr, len(train_loader))
    write_phase_tensorboard_metadata(train_state.writer, phase_config, args, train_state.lr, len(train_loader))

    global_step = phase_config.start_epoch * len(train_loader)
    for epoch in range(phase_config.start_epoch, phase_config.end_epoch):
        global_step, _ = run_phase_epoch(
            phase_runtime=phase_runtime,
            epoch=epoch,
            criterion=train_state.criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            eval_encoder=train_state.eval_encoder,
            coco_gt=coco_gt,
            category_names=category_names,
            device=train_state.device,
            writer=train_state.writer,
            args=args,
            main_process=train_state.main_process,
            base_model=train_state.base_model,
            global_step=global_step,
        )

    train_state.ssd_model = phase_runtime.ssd_model


def train_full_model_phase(
    args: Namespace,
    train_loader,
    val_loader,
    coco_gt,
    category_names: Sequence[str] | None,
    train_state: TrainingContext,
    start_epoch: int,
) -> bool:
    """Main phase: unfreeze the backbone and continue full-model fine-tuning.

    LR policy for this phase:
    - First do batch-wise warmup.
    - Then keep LR flat for the configured hold span.
    - Finally switch to cosine decay until the end of training.
    """
    phase_config = build_full_phase_config(args, start_epoch)
    if phase_config is None:
        return False

    set_backbone_requires_grad(train_state.base_model, requires_grad=True)
    phase_runtime = build_phase_runtime(
        base_model=train_state.base_model,
        args=args,
        distributed=train_state.distributed,
        device=train_state.device,
        base_lr=train_state.lr,
        phase_config=phase_config,
        steps_per_epoch=len(train_loader),
    )

    print_phase_header(train_state.main_process, train_state.distributed, phase_config)
    if train_state.main_process:
        print("解冻 backbone，进行全量训练。")
        print(f"early stopping: patience={args.patience}, start_epoch={resolve_early_stop_start_epoch(args)}")
    log_phase_schedule(train_state.main_process, phase_config, args, train_state.lr, len(train_loader))
    write_phase_tensorboard_metadata(train_state.writer, phase_config, args, train_state.lr, len(train_loader))

    best_map = train_state.best_map
    no_improve = train_state.no_improve
    should_stop = False
    global_step = phase_config.start_epoch * len(train_loader)
    early_stop_start_epoch = resolve_early_stop_start_epoch(args)

    for epoch in range(phase_config.start_epoch, phase_config.end_epoch):
        global_step, val_metrics = run_phase_epoch(
            phase_runtime=phase_runtime,
            epoch=epoch,
            criterion=train_state.criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            eval_encoder=train_state.eval_encoder,
            coco_gt=coco_gt,
            category_names=category_names,
            device=train_state.device,
            writer=train_state.writer,
            args=args,
            main_process=train_state.main_process,
            base_model=train_state.base_model,
            global_step=global_step,
        )

        if train_state.main_process:
            best_map, no_improve, should_stop = update_early_stopping(
                epoch=epoch,
                val_metrics=val_metrics,
                best_map=best_map,
                no_improve=no_improve,
                min_delta=args.min_delta,
                early_stop_start_epoch=early_stop_start_epoch,
                patience=args.patience,
                base_model=train_state.base_model,
                backbone=args.backbone,
            )

        should_stop = sync_early_stop_flag(should_stop, train_state.distributed, train_state.device)
        if should_stop:
            break

    train_state.best_map = best_map
    train_state.no_improve = no_improve
    train_state.ssd_model = phase_runtime.ssd_model
    return should_stop


# -----------------------------------------------------------------------------
# Epoch / batch execution
# -----------------------------------------------------------------------------


def run_phase_epoch(
    phase_runtime: PhaseRuntime,
    epoch: int,
    criterion: Loss,
    train_loader,
    val_loader,
    eval_encoder: Encoder,
    coco_gt,
    category_names: Sequence[str] | None,
    device: torch.device,
    writer: SummaryWriter | None,
    args: Namespace,
    main_process: bool,
    base_model: SSD320,
    global_step: int,
) -> tuple[int, dict[str, float] | None]:
    """Execute one epoch inside a phase: train, validate, checkpoint, then return metrics."""
    if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)

    global_step, average_loss = train_one_epoch(
        ssd_model=phase_runtime.ssd_model,
        optimizer=phase_runtime.optimizer,
        scaler=phase_runtime.scaler,
        scheduler=phase_runtime.scheduler,
        criterion=criterion,
        train_loader=train_loader,
        device=device,
        writer=writer,
        global_step=global_step,
    )

    if main_process:
        print(f"Epoch {epoch + 1}: average train loss={average_loss:.4f}")
        if writer is not None:
            writer.add_scalar('Train/EpochLoss', average_loss, epoch + 1)

    val_metrics = run_torch_validation_if_needed(
        phase_runtime.ssd_model,
        epoch,
        val_loader,
        eval_encoder,
        coco_gt,
        category_names,
        device,
        writer,
        args,
        main_process,
    )

    if main_process:
        save_model_checkpoint(base_model, args.backbone, epoch)

    return global_step, val_metrics


def train_one_epoch(
    ssd_model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: LRScheduler | None,
    criterion: Loss,
    train_loader,
    device: torch.device,
    writer: SummaryWriter | None,
    global_step: int,
) -> tuple[int, float]:
    """Run one full training epoch and return the updated step count and mean loss."""
    amp_enabled = device.type == 'cuda'
    ssd_model.train()
    loss_sum = 0.0
    batch_count = 0

    for batch_idx, (images, plocs, plabels) in enumerate(train_loader):
        images = images.to(device)
        plocs = plocs.to(device)
        plabels = plabels.to(device)
        current_lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            loc_preds, conf_preds = ssd_model(images)
            loc_preds = loc_preds.float()
            conf_preds = conf_preds.float()

            gloc = plocs.transpose(1, 2).contiguous()
            loss = criterion(loc_preds, conf_preds, gloc, plabels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(ssd_model.parameters(), max_norm=2.0)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_was_skipped = scaler.get_scale() < scale_before_step
        if scheduler is not None and not optimizer_step_was_skipped:
            scheduler.step()

        global_step += 1
        batch_count += 1
        loss_sum += float(loss.item())
        if batch_idx % 10 == 0 and writer is not None:
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/LR', current_lr, global_step)

    if dist.is_available() and dist.is_initialized():
        loss_stats = torch.tensor([loss_sum, float(batch_count)], device=device, dtype=torch.float64)
        dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
        loss_sum = loss_stats[0].item()
        batch_count = int(loss_stats[1].item())

    average_loss = loss_sum / batch_count if batch_count > 0 else 0.0
    return global_step, average_loss


def update_early_stopping(
    epoch: int,
    val_metrics: dict[str, float] | None,
    best_map: float,
    no_improve: int,
    min_delta: float,
    early_stop_start_epoch: int,
    patience: int,
    base_model: SSD320,
    backbone: str,
) -> tuple[float, int, bool]:
    if val_metrics is None:
        return best_map, no_improve, False

    val_map = val_metrics['mAP']
    if val_map > best_map + min_delta:
        best_map = val_map
        no_improve = 0
        best_path = save_model_checkpoint(base_model, backbone, epoch, is_best=True, metrics=val_metrics)
        print(f"保存最佳模型: {best_path}, mAP={best_map:.4f}")
        return best_map, no_improve, False

    current_epoch_number = epoch + 1
    if current_epoch_number < early_stop_start_epoch:
        print(
            f"mAP 暂无提升，但当前 epoch={current_epoch_number} 仍早于 early stopping "
            f"起始 epoch={early_stop_start_epoch}，继续观察余弦退火后半段。"
        )
        return best_map, no_improve, False

    no_improve += 1
    print(f"mAP 无提升计数: {no_improve}/{patience}")
    should_stop = no_improve >= patience
    if should_stop:
        print("触发 Early Stopping，提前结束训练。")
    return best_map, no_improve, should_stop


# -----------------------------------------------------------------------------
# Training setup
# -----------------------------------------------------------------------------


def setup_training(args: Namespace, resume_checkpoint: str | None = None) -> TrainingContext:
    """Create model, loss, writer, device, and checkpoint state shared by both phases."""
    distributed = dist.is_available() and dist.is_initialized()
    main_process = (not distributed) or dist.get_rank() == 0

    batch_size = args.batch_size
    world_size = dist.get_world_size() if distributed else 1
    global_batch_size = batch_size * world_size
    effective_lr = args.lr * (batch_size * world_size) / 32
    writer = SummaryWriter(log_dir=build_tensorboard_log_dir(args, effective_lr, global_batch_size)) if main_process else None

    if main_process:
        print(f"Batch Size(per proc): {batch_size}, World Size: {world_size}, Calculated Learning Rate: {effective_lr}")
        print(f"Default boxes: min_ratio={args.dbox_min_ratio}, max_ratio={args.dbox_max_ratio}")
    write_training_config(writer, args, effective_lr, batch_size, world_size, global_batch_size)

    dboxes = dboxes320_coco(min_ratio=args.dbox_min_ratio, max_ratio=args.dbox_max_ratio)
    eval_encoder = Encoder(dboxes)

    local_rank = getattr(args, 'local_rank', 0)
    if args.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    if main_process:
        print(f"使用设备: {device}")

    backbone_weights = None if resume_checkpoint else ('IMAGENET1K_V1' if args.pretrained_backbone else None)
    if main_process:
        if resume_checkpoint:
            print("检测到 resume checkpoint，跳过 pretrained backbone 初始化。")
        elif backbone_weights is not None:
            print(f"使用预训练 backbone 初始化: weights={backbone_weights}")
        else:
            print("不使用 pretrained backbone，采用随机初始化。")

    base_model = SSD320(
        backbone=MobileNet(backbone=args.backbone, weights=backbone_weights),
        num_classes=args.num_classes,
    )

    if resume_checkpoint:
        if main_process:
            print(f"Resuming training from checkpoint: {resume_checkpoint}")
        if Path(resume_checkpoint).exists():
            checkpoint_state = torch.load(resume_checkpoint, map_location=device)
            base_model.load_state_dict(unwrap_checkpoint_state(checkpoint_state))
            if main_process:
                print("Checkpoint loaded successfully.")
        elif main_process:
            print(f"Checkpoint file not found: {resume_checkpoint}")

    base_model.to(device)
    criterion = Loss(dboxes).to(device)
    return TrainingContext(
        distributed=distributed,
        main_process=main_process,
        writer=writer,
        lr=effective_lr,
        device=device,
        eval_encoder=eval_encoder,
        base_model=base_model,
        criterion=criterion,
    )


# -----------------------------------------------------------------------------
# Phase configuration and runtime helpers
# -----------------------------------------------------------------------------


def build_freeze_phase_config(args: Namespace, start_epoch: int = 0) -> PhaseExecutionConfig | None:
    """Build the optional head-only training phase."""
    freeze_end_epoch = min(args.freeze_backbone_epochs, args.epochs)
    if freeze_end_epoch <= 0 or start_epoch >= freeze_end_epoch:
        return None

    return PhaseExecutionConfig(
        name='freeze_backbone',
        start_epoch=start_epoch,
        end_epoch=freeze_end_epoch,
        warmup_epochs=args.freeze_warmup_epochs,
        hold_epochs=max(0, freeze_end_epoch - start_epoch - args.freeze_warmup_epochs),
        schedule_anchor_epoch=0,
    )


def resolve_full_phase_schedule_epochs(args: Namespace) -> tuple[int, int, int]:
    """Resolve total, warmup, and hold epochs for the full-model phase."""
    full_phase_anchor_epoch = min(args.freeze_backbone_epochs, args.epochs)
    total_full_phase_epochs = max(0, args.epochs - full_phase_anchor_epoch)
    warmup_epochs = min(args.warmup_epochs, total_full_phase_epochs)
    remaining_epochs_after_warmup = max(0, total_full_phase_epochs - warmup_epochs)
    hold_epochs = 0
    if args.hold_ratio > 0.0 and remaining_epochs_after_warmup > 0:
        hold_epochs = math.ceil(remaining_epochs_after_warmup * args.hold_ratio)
    hold_epochs = min(hold_epochs, remaining_epochs_after_warmup)
    return total_full_phase_epochs, warmup_epochs, hold_epochs


def build_full_phase_config(args: Namespace, start_epoch: int) -> PhaseExecutionConfig | None:
    """Build the full fine-tuning phase starting after the freeze stage."""
    epoch_end = args.epochs
    full_phase_anchor_epoch = min(args.freeze_backbone_epochs, epoch_end)
    phase_start_epoch = max(start_epoch, full_phase_anchor_epoch)
    if phase_start_epoch >= epoch_end:
        return None

    _, warmup_epochs, hold_epochs = resolve_full_phase_schedule_epochs(args)

    return PhaseExecutionConfig(
        name='full_model',
        start_epoch=phase_start_epoch,
        end_epoch=epoch_end,
        warmup_epochs=warmup_epochs,
        hold_epochs=hold_epochs,
        schedule_anchor_epoch=full_phase_anchor_epoch,
    )


def build_phase_runtime(
    base_model: SSD320,
    args: Namespace,
    distributed: bool,
    device: torch.device,
    base_lr: float,
    phase_config: PhaseExecutionConfig,
    steps_per_epoch: int,
) -> PhaseRuntime:
    """Create the per-phase runtime objects: DDP wrapper, optimizer, scaler, scheduler."""
    local_rank = getattr(args, 'local_rank', 0)
    ssd_model: torch.nn.Module = base_model
    if distributed:
        ssd_model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    trainable_params = [param for param in base_model.parameters() if param.requires_grad]
    optimizer = optim.SGD(
        trainable_params,
        lr=base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    completed_steps = max(0, phase_config.start_epoch - phase_config.schedule_anchor_epoch) * steps_per_epoch
    scheduler = build_phase_scheduler(optimizer, args, base_lr, phase_config, steps_per_epoch, completed_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    return PhaseRuntime(ssd_model=ssd_model, optimizer=optimizer, scaler=scaler, scheduler=scheduler)


def build_phase_scheduler(
    optimizer: optim.Optimizer,
    args: Namespace,
    base_lr: float,
    phase_config: PhaseExecutionConfig,
    steps_per_epoch: int,
    completed_steps: int,
) -> LRScheduler | None:
    """Build the LR scheduler for one phase.

    freeze_backbone:
    - warmup ends before freeze ends
    - then LR stays constant until freeze finishes

    full_model:
    - warmup ends before the hold span starts
    - then LR stays constant during the hold span
    - then LR follows cosine decay to the configured minimum ratio
    """
    total_phase_epochs = phase_config.end_epoch - phase_config.schedule_anchor_epoch
    total_phase_steps = total_phase_epochs * steps_per_epoch
    if total_phase_steps <= 0:
        return None

    warmup_steps = min(max(0, phase_config.warmup_epochs * steps_per_epoch), max(0, total_phase_steps - 1))
    remaining_steps_after_warmup = max(0, total_phase_steps - warmup_steps)
    hold_steps = min(max(0, phase_config.hold_epochs * steps_per_epoch), remaining_steps_after_warmup)
    min_lr_ratio = args.cosine_min_lr_ratio

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return 0.1 + 0.9 * (step / warmup_steps)

        if phase_config.name == 'freeze_backbone' or (hold_steps > 0 and step < warmup_steps + hold_steps):
            return 1.0

        cosine_span_steps = total_phase_steps - warmup_steps - hold_steps
        if cosine_span_steps <= 0:
            return 1.0

        cosine_steps = max(1, cosine_span_steps)
        progress = min(max(step - warmup_steps - hold_steps, 0), cosine_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress / cosine_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    last_epoch = completed_steps - 1
    if last_epoch >= 0:
        for param_group in optimizer.param_groups:
            param_group.setdefault('initial_lr', param_group['lr'])
    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------


def build_tensorboard_log_dir(args: Namespace, effective_lr: float, global_batch_size: int) -> str:
    """Build a log directory whose name records key tuning knobs for comparison."""
    run_stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    effective_lr_tag = f"{effective_lr:.6g}".replace('.', 'p')
    weight_decay_tag = f"{args.weight_decay:.6g}".replace('.', 'p')
    cosine_min_lr_tag = f"{args.cosine_min_lr_ratio:.6g}".replace('.', 'p')
    _, _, full_phase_hold_epochs = resolve_full_phase_schedule_epochs(args)
    hold_ratio_tag = f"{args.hold_ratio:.4f}".replace('.', 'p')
    run_tag = "_".join(
        [
            f"ep{args.epochs}",
            f"gbs{global_batch_size}",
            f"elr{effective_lr_tag}",
            f"wd{weight_decay_tag}",
            f"frz{args.freeze_backbone_epochs}",
            f"fw{args.freeze_warmup_epochs}",
            f"wu{args.warmup_epochs}",
            f"holdr{hold_ratio_tag}",
            f"holde{full_phase_hold_epochs}",
            f"cmin{cosine_min_lr_tag}",
            f"box{args.dbox_min_ratio:.2f}-{args.dbox_max_ratio:.2f}".replace('.', 'p'),
            f"aug{int(args.augment)}",
            f"pre{int(args.pretrained_backbone)}",
        ]
    )
    return str(Path("logs") / args.backbone / run_tag / run_stamp)


def write_training_config(
    writer: SummaryWriter | None,
    args: Namespace,
    effective_lr: float,
    batch_size: int,
    world_size: int,
    global_batch_size: int,
) -> None:
    if writer is None:
        return

    freeze_phase_warmup_epochs = min(args.freeze_warmup_epochs, args.freeze_backbone_epochs)
    total_full_phase_epochs, full_phase_warmup_epochs, full_phase_hold_epochs = resolve_full_phase_schedule_epochs(args)
    freeze_summary = resolve_schedule_summary('freeze_backbone', args, effective_lr, freeze_phase_warmup_epochs, 0)
    full_summary = resolve_schedule_summary('full_model', args, effective_lr, full_phase_warmup_epochs, full_phase_hold_epochs)
    early_stop_start_epoch = resolve_early_stop_start_epoch(args)
    scalar_config = {
        'Epochs': args.epochs,
        'BatchSizePerProc': batch_size,
        'WorldSize': world_size,
        'GlobalBatchSize': global_batch_size,
        'BaseLR': args.lr,
        'EffectiveLR': effective_lr,
        'Momentum': args.momentum,
        'WeightDecay': args.weight_decay,
        'FreezeBackboneEpochs': args.freeze_backbone_epochs,
        'FullPhaseEpochs': total_full_phase_epochs,
        'FreezeWarmupEpochs': freeze_phase_warmup_epochs,
        'FullWarmupEpochs': full_phase_warmup_epochs,
        'FullHoldRatio': args.hold_ratio,
        'FullHoldEpochs': full_phase_hold_epochs,
        'CosineMinLRRatio': args.cosine_min_lr_ratio,
        'EvalInterval': args.eval_interval,
        'Patience': args.patience,
        'MinDelta': args.min_delta,
        'NumClasses': args.num_classes,
        'DefaultBoxMinRatio': args.dbox_min_ratio,
        'DefaultBoxMaxRatio': args.dbox_max_ratio,
        'NumWorkers': args.num_workers,
        'PrefetchFactor': args.prefetch_factor,
        'NumVisualizations': args.num_visualizations,
        'EarlyStopStartEpoch': early_stop_start_epoch,
        'AugmentEnabled': int(args.augment),
        'PinMemoryEnabled': int(args.pin_memory),
        'PretrainedBackboneEnabled': int(args.pretrained_backbone),
        'DDPEnabled': int(args.ddp),
        'RestartEnabled': int(args.restart),
        'ExportBestOnnxEnabled': int(args.export_onnx_from_best_checkpoint),
    }
    text_config = {
        'Backbone': args.backbone,
        'Device': args.device,
        'LRSchedulerType': 'LambdaLR(freeze:warmup+hold, full:warmup+hold+cosine)',
        'FreezeLRSchedule': freeze_summary,
        'FullLRSchedule': full_summary,
    }
    args_summary = '\n'.join(f"- {key}: {value}" for key, value in sorted(vars(args).items()))

    writer.add_text(
        'Config/Summary',
        (
            f"backbone={args.backbone}, "
            f"dbox_min_ratio={args.dbox_min_ratio}, dbox_max_ratio={args.dbox_max_ratio}, "
            f"effective_lr={effective_lr}, freeze_schedule=({freeze_summary}), "
            f"full_schedule=({full_summary}), early_stop_start_epoch={early_stop_start_epoch}"
        ),
        0,
    )
    writer.add_text('Config/AllArgs', args_summary, 0)
    for name, value in text_config.items():
        writer.add_text(f'Config/{name}', str(value), 0)
    for name, value in scalar_config.items():
        writer.add_scalar(f'Config/{name}', value, 0)


def resolve_schedule_summary(phase_name: str, args: Namespace, base_lr: float, warmup_epochs: int, hold_epochs: int) -> str:
    if phase_name == 'freeze_backbone':
        return f"batch_warmup+hold peak_lr={base_lr:.6g}, warmup_epochs={warmup_epochs}"

    min_lr = base_lr * args.cosine_min_lr_ratio
    return (
        f"batch_warmup+hold+cosine hold_ratio={args.hold_ratio:.4f}, hold_epochs={hold_epochs}, "
        f"eta_min_ratio={args.cosine_min_lr_ratio:.4f}, eta_min={min_lr:.6g}, "
        f"warmup_epochs={warmup_epochs}"
    )


def write_phase_tensorboard_metadata(
    writer: SummaryWriter | None,
    phase_config: PhaseExecutionConfig,
    args: Namespace,
    base_lr: float,
    steps_per_epoch: int,
) -> None:
    """Write human-readable phase metadata to TensorBoard at the phase boundary."""
    if writer is None:
        return

    global_step = phase_config.start_epoch * steps_per_epoch
    phase_label = f"{phase_config.name} (epoch {phase_config.start_epoch + 1}-{phase_config.end_epoch})"
    schedule_summary = resolve_schedule_summary(
        phase_config.name,
        args,
        base_lr,
        phase_config.warmup_epochs,
        phase_config.hold_epochs,
    )
    writer.add_text('Phase/CurrentName', phase_label, global_step)
    writer.add_text('Phase/CurrentLRSchedule', schedule_summary, global_step)


def resolve_early_stop_start_epoch(args: Namespace) -> int:
    full_phase_start_epoch = min(args.freeze_backbone_epochs, args.epochs)
    _, _, full_phase_hold_epochs = resolve_full_phase_schedule_epochs(args)
    cosine_ready_epoch = full_phase_start_epoch + max(0, args.warmup_epochs) + full_phase_hold_epochs + 1
    return max(cosine_ready_epoch, round(args.epochs * 0.8))


def set_backbone_requires_grad(base_model: SSD320, requires_grad: bool) -> None:
    for param in base_model.feature_extractor.feature_extractor.parameters():
        param.requires_grad_(requires_grad)


def log_phase_schedule(
    main_process: bool,
    phase_config: PhaseExecutionConfig,
    args: Namespace,
    base_lr: float,
    steps_per_epoch: int,
) -> None:
    """Print the LR schedule summary for the current phase."""
    if not main_process:
        return

    total_epochs = phase_config.end_epoch - phase_config.schedule_anchor_epoch
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = min(max(0, phase_config.warmup_epochs * steps_per_epoch), max(0, total_steps - 1))
    remaining_steps_after_warmup = max(0, total_steps - warmup_steps)
    hold_steps = min(max(0, phase_config.hold_epochs * steps_per_epoch), remaining_steps_after_warmup)
    schedule_summary = resolve_schedule_summary(
        phase_config.name,
        args,
        base_lr,
        phase_config.warmup_epochs,
        phase_config.hold_epochs,
    )
    print(
        f"{phase_config.name} scheduler: total_epochs={total_epochs}, total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}, hold_steps={hold_steps}, {schedule_summary}"
    )


def print_phase_header(
    main_process: bool,
    distributed: bool,
    phase_config: PhaseExecutionConfig,
) -> None:
    if not main_process:
        return

    start_epoch = phase_config.start_epoch + 1
    end_epoch = phase_config.end_epoch
    if distributed:
        print(f"阶段 {phase_config.name}: 启用 DDP, epoch=[{start_epoch}, {end_epoch}]")
    else:
        print(f"阶段 {phase_config.name}: epoch=[{start_epoch}, {end_epoch}]")
