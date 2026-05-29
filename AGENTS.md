# MobileNet-SSD Repo Guide

## Overview

This repository implements a PyTorch `SSDLite320` detector with `timm` MobileNet-family backbones and a CLI entrypoint in `main.py`.

The main workflows are:
- Train a model with `python main.py train ...`
- Evaluate an exported ONNX model with `python main.py val ...`
- Export ONNX as part of training or from the best checkpoint

## Source Of Truth

Prefer the Python code over the README when they disagree.

Important current example: `README.md` documents some older defaults, while `main.py` currently defines the active CLI defaults for epochs, learning rate, eval interval, and cosine minimum LR ratio.

For behavior and defaults, check:
- `main.py`
- `ssdlite320/train.py`
- `ssdlite320/runtime.py`
- `ssdlite320/eval.py`

## Repo Map

- `main.py`: CLI parsing, argument validation, `train`/`val` command dispatch
- `ssdlite320/model.py`: backbone integration, SSD heads, loss
- `ssdlite320/encoder.py`: default boxes, encode/decode logic
- `ssdlite320/train.py`: optimizer setup, scheduling, epoch loop, early stopping, checkpoint saves
- `ssdlite320/eval.py`: PyTorch/ONNX evaluation, visualization, metrics export
- `ssdlite320/runtime.py`: DDP setup, data loading, checkpoint lookup, ONNX export helpers
- `ssdlite320/data_hf.py`: dataset loading and transforms
- `test/check_shapes.py`: lightweight shape sanity script
- `test/check_torchvision_ssdlite_anchor.py`: reference anchor inspection against torchvision

## Editing Guidance

- Keep changes aligned across training, evaluation, and ONNX export paths when model outputs, box encoding, or decode behavior change.
- Prefer small, localized edits in `ssdlite320/` and `test/`.
- Preserve the existing CLI shape unless the task explicitly asks for interface changes.
- Keep user-facing help text and validation errors consistent with the surrounding code style. Some existing messages are in Chinese; do not rewrite unrelated messages just for consistency.
- If a change affects defaults or CLI flags, update both the implementation and any relevant README examples.

## Do Not Modify By Default

Do not edit these paths unless the task explicitly requires it:
- `checkpoints/`
- `weights/`
- `logs/`
- `viz_results/`
- `val_results/`
- `reports/` produced by evaluation runs
- cached datasets under `data/`
- `__pycache__/`

Treat those directories as generated artifacts or local runtime state.

## Common Commands

- Train with default command normalization:
  `python main.py`
- Explicit train:
  `python main.py train --device cuda`
- Resume from last checkpoint:
  `python main.py train --device cuda --restart`
- Validate exported ONNX:
  `python main.py val --backbone mobilenetv4_conv_small --provider auto`
- Multi-GPU training:
  `torchrun --nproc_per_node=2 main.py train --device cuda`

## Validation Expectations

Use the lightest check that covers the change:

- CLI or argument parsing changes:
  `python main.py train --help`
  `python main.py val --help`
- Model shape, backbone, or feature-map changes:
  `python test/check_shapes.py`
- Anchor/default-box comparisons:
  `python test/check_torchvision_ssdlite_anchor.py`
- ONNX export or ONNX eval changes:
  run the relevant `python main.py val ...` command if the required model artifacts are available

If you cannot run a meaningful validation step because weights, datasets, or GPUs are unavailable, say so explicitly in the final report.
