#!/usr/bin/env bash
# Run TensileLite param-group sweeps on a gfx1250 host with ROCm.
#
# Usage:
#   ./run_all.sh smoke              # 1 size/OOB category, faster
#   ./run_all.sh full               # all 60 Exact sizes
#   ./run_all.sh smoke 01_tdm_loads # one group only
#
# Env overrides:
#   TENSILE=.../tensilelite/bin/Tensile
#   ROOT=.../param_testing/runs

set -euo pipefail

MODE="${1:-smoke}"
ONLY="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$SCRIPT_DIR}"
TENSILE="${TENSILE:-$(cd "$SCRIPT_DIR/../../tensilelite/bin" && pwd)/Tensile}"

if [[ ! -x "$TENSILE" && ! -f "$TENSILE" ]]; then
  echo "Tensile not found at $TENSILE" >&2
  exit 1
fi

case "$MODE" in
  smoke) CFG=config_smoke.yaml; OUT=results_smoke ;;
  full)  CFG=config.yaml;       OUT=results ;;
  *) echo "mode must be smoke|full" >&2; exit 1 ;;
esac

GROUPS=(
  01_tdm_loads
  02_workgroup_cluster
  03_lds_segment
  04_memory_hints
  05_prefetch
  06_persistent_reuse
  07_epilogue
  08_codegen_ra
)

if [[ -n "$ONLY" ]]; then
  GROUPS=("$ONLY")
fi

export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"

for g in "${GROUPS[@]}"; do
  dir="$ROOT/$g"
  cfg="$dir/$CFG"
  out="$dir/$OUT"
  mkdir -p "$out"
  echo "======== $g ($MODE) ========"
  echo "config: $cfg"
  echo "output: $out"
  python3 "$TENSILE" "$cfg" "$out" 2>&1 | tee "$out/tensile_run.log"
  python3 "$SCRIPT_DIR/collect_results.py" "$out" --group "$g" --mode "$MODE"
done

echo "All done. Summaries under each group folder and $ROOT/summary_${MODE}.csv"
python3 "$SCRIPT_DIR/collect_results.py" --merge "$ROOT" --mode "$MODE"
