#!/usr/bin/env bash
# Train a SPARE-AD model from the example data.
#
# Run this script from the repo root:
#   bash examples/SPARE_AD/run_training.sh
#
# The trained model and experiment metadata are written to:
#   examples/SPARE_AD/output/<run_tag>/   (run_tag is set in train_config.json)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/input/train_config.json"

echo "=== SPARE-AD Training ==="
echo "Config : ${CONFIG}"
echo "Output : ${SCRIPT_DIR}/output/"
echo ""

NiChart_SPARE train -c "${CONFIG}"
