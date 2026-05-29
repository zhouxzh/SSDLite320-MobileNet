#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"
LOG_DIR="${LOG_DIR:-logs/run_all_mobilenet}"

mkdir -p "$LOG_DIR"

DEFAULT_BACKBONES=(
	# "mobilenetv1_100"
	# "mobilenetv1_100h"
	# "mobilenetv1_125"
	# "mobilenetv2_050"
	# "mobilenetv2_100"
	# "mobilenetv2_140"
	# "mobilenetv3_small_100"
	# "mobilenetv3_large_100"
	# "mobilenetv3_large_150d"
	# "mobilenetv4_conv_small"
	# "mobilenetv4_conv_medium"
	# "mobilenetv4_conv_large"
	"mobilenetv4_hybrid_medium"
	"mobilenetv4_hybrid_large"
)

if [[ $# -gt 0 ]]; then
	BACKBONES=("$@")
else
	BACKBONES=("${DEFAULT_BACKBONES[@]}")
fi

echo "Workspace: $ROOT_DIR"
echo "Log directory: $LOG_DIR"
echo "nproc_per_node: $NPROC_PER_NODE"
echo "Training ${#BACKBONES[@]} MobileNet backbones with default train arguments"

for index in "${!BACKBONES[@]}"; do
	backbone="${BACKBONES[$index]}"
	master_port="$((MASTER_PORT_BASE + index))"
	log_file="$LOG_DIR/${backbone}.log"

	echo
	echo "[$((index + 1))/${#BACKBONES[@]}] Start training: $backbone"
	echo "Log file: $log_file"

	torchrun \
		--nproc_per_node="$NPROC_PER_NODE" \
		--master_port="$master_port" \
		main.py train \
		--backbone "$backbone" \
		--device cuda \
		--restart \
		--ddp 2>&1 | tee "$log_file"

	echo "[$((index + 1))/${#BACKBONES[@]}] Finished: $backbone"
done

echo
echo "All trainings completed successfully."
