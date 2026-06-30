#!/usr/bin/env bash
# Apply the trained SPARE-AD model to new subjects.
#
# Run from the repo root:
#   bash examples/SPARE_AD/testing/run_testing.sh
#
# Prep is applied automatically using parameters stored in the model.
# Outputs written to:
#   examples/SPARE_AD/testing/output/prepped.csv
#   examples/SPARE_AD/testing/output/predictions.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${SCRIPT_DIR}/../training/output/SPARE_AD_v1.0/model/SPARE_AD_v1.0.joblib"
INPUT="${SCRIPT_DIR}/input/raw_data.csv"
OUTPUT="${SCRIPT_DIR}/output"

if [ ! -f "${MODEL}" ]; then
    echo "Model not found: ${MODEL}"
    echo "Run training first:  bash examples/SPARE_AD/training/run_training.sh"
    exit 1
fi

echo "=== SPARE-AD Testing ==="
echo "Model  : ${MODEL}"
echo "Input  : ${INPUT}"
echo "Output : ${OUTPUT}"
echo ""

NiChart_SPARE test -m "${MODEL}" -i "${INPUT}" -o "${OUTPUT}"
