#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WEIGHTS_DIR="${WEIGHTS_DIR:-weights}"
PROVIDER="cuda"
CACHE_DIR="${CACHE_DIR:-./data}"
IMG_SIZE="${IMG_SIZE:-320}"
NUM_VISUALIZATIONS="${NUM_VISUALIZATIONS:-0}"
DECODE_IOU_THRESHOLD="${DECODE_IOU_THRESHOLD:-0.5}"
MAX_OUTPUT="${MAX_OUTPUT:-200}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PREPROCESS_BATCH_SIZE="${PREPROCESS_BATCH_SIZE:-16}"
DBOX_MIN_RATIO="${DBOX_MIN_RATIO:-0.1}"
DBOX_MAX_RATIO="${DBOX_MAX_RATIO:-0.9}"
CSV_FILE="${CSV_FILE:-reports/onnx_validation_metrics_all_${RUN_ID}.csv}"
RESULT_DIR="${RESULT_DIR:-val_results/onnx_eval_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-logs/eval_all_onnx_${RUN_ID}}"

mkdir -p "$(dirname "$CSV_FILE")" "$RESULT_DIR" "$LOG_DIR"

if [[ $# -gt 0 ]]; then
	ONNX_MODELS=("$@")
else
	mapfile -t ONNX_MODELS < <(find "$WEIGHTS_DIR" -maxdepth 1 -type f -name '*.onnx' | sort)
fi

if [[ ${#ONNX_MODELS[@]} -eq 0 ]]; then
	echo "No ONNX models found in: $WEIGHTS_DIR" >&2
	exit 1
fi

echo "Checking ONNX Runtime CUDA provider..."
python - "${ONNX_MODELS[0]}" <<'PY'
import sys
import onnxruntime as ort

onnx_path = sys.argv[1]
available_providers = ort.get_available_providers()
print(f"ONNX Runtime version: {ort.__version__}")
print(f"Available providers: {available_providers}")

if "CUDAExecutionProvider" not in available_providers:
    raise SystemExit("CUDAExecutionProvider is not available in this Python environment.")

session = ort.InferenceSession(
    onnx_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
active_providers = session.get_providers()
print(f"CUDA preflight model: {onnx_path}")
print(f"Active providers: {active_providers}")

if "CUDAExecutionProvider" not in active_providers:
    raise SystemExit("CUDAExecutionProvider failed to initialize; refusing to run CPU fallback.")
PY

echo "Workspace: $ROOT_DIR"
echo "ONNX models: ${#ONNX_MODELS[@]}"
echo "Provider: $PROVIDER"
echo "CSV file: $CSV_FILE"
echo "Result directory: $RESULT_DIR"
echo "Log directory: $LOG_DIR"

for index in "${!ONNX_MODELS[@]}"; do
	onnx_path="${ONNX_MODELS[$index]}"
	model_name="$(basename "$onnx_path" .onnx)"
	backbone="${model_name#ssd320_}"
	result_file="$RESULT_DIR/${model_name}_predictions.json"
	log_file="$LOG_DIR/${model_name}.log"

	echo
	echo "[$((index + 1))/${#ONNX_MODELS[@]}] Evaluating: $onnx_path"
	echo "Backbone: $backbone"
	echo "Log file: $log_file"

	python main.py val \
		--backbone "$backbone" \
		--onnx-path "$onnx_path" \
		--provider "$PROVIDER" \
		--cache-dir "$CACHE_DIR" \
		--img-size "$IMG_SIZE" \
		--num-visualizations "$NUM_VISUALIZATIONS" \
		--decode-iou-threshold "$DECODE_IOU_THRESHOLD" \
		--max-output "$MAX_OUTPUT" \
		--num-workers "$NUM_WORKERS" \
		--prefetch-factor "$PREFETCH_FACTOR" \
		--preprocess-batch-size "$PREPROCESS_BATCH_SIZE" \
		--dbox-min-ratio "$DBOX_MIN_RATIO" \
		--dbox-max-ratio "$DBOX_MAX_RATIO" \
		--result-file "$result_file" \
		--csv-file "$CSV_FILE" 2>&1 | tee "$log_file"
done

echo
echo "All ONNX evaluations completed."
echo "CSV file: $CSV_FILE"
