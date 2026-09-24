#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
YAML_DIR="$SCRIPT_DIR/../YAMLs"

export TENSILE_DISABLE_HELPER_CACHE=1

cd "$TENSILE_DIR"
rm -rf /tmp/fused-harness
./Tensile/bin/Tensile "$YAML_DIR/benchmark_k1_bf16.yaml" /tmp/fused-harness
