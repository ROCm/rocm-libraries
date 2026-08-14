#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Run the subtile bf16 tuning sweep for PartialRMS and/or PartialRMSQuant on
# an 8192x8192x8192 problem on gfx950.
#
# Usage:
#   run_subtile_bf16_tune.sh [--variant prms|prmsq|both] [--chip CHIP]
#                            [--venv PATH] [--client PATH] [--out-dir DIR]
#
# Arguments:
#   --variant prms|prmsq|both  Which YAML(s) to run (default: both).
#   --chip CHIP                GPU architecture string (default: gfx950).
#   --venv PATH                Path to the venv root (default: ~/.tensile-epilogues).
#   --client PATH              tensilelite-client binary (default: auto-detect from build_tmp/).
#   --out-dir DIR              Output root dir (default: epilogues/out/).
#
# Prerequisites:
#   - gfx950 GPU accessible.
#   - venv activated or --venv pointing at one that has Tensile installed.
#   - tensilelite-client built (invoke build-client, or pass --client explicitly).
#
# Results land in:
#   <out-dir>/tune_subtile_bf16_prms_8192/   (PartialRMS)
#   <out-dir>/tune_subtile_bf16_prmsq_8192/  (PartialRMSQuant)
#
# The winner for each run is printed at the end.

set -euo pipefail

TENSILELITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
YAML_DIR="${TENSILELITE_ROOT}/epilogues/yaml"

# ── Defaults ──────────────────────────────────────────────────────────────────
VARIANT="both"
CHIP="gfx950"
VENV="${HOME}/.tensile-epilogues"
CLIENT=""
OUT_DIR="${TENSILELITE_ROOT}/epilogues/out"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)  VARIANT="$2";  shift 2 ;;
        --chip)     CHIP="$2";     shift 2 ;;
        --venv)     VENV="$2";     shift 2 ;;
        --client)   CLIENT="$2";   shift 2 ;;
        --out-dir)  OUT_DIR="$2";  shift 2 ;;
        *) echo "error: unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Resolve Python ─────────────────────────────────────────────────────────────
if [[ -x "${VENV}/bin/python" ]]; then
    PYTHON="${VENV}/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

# ── Resolve client ─────────────────────────────────────────────────────────────
if [[ -z "$CLIENT" ]]; then
    _DEFAULT_CLIENT="${TENSILELITE_ROOT}/build_tmp/tensilelite/client/tensilelite-client"
    if [[ -x "$_DEFAULT_CLIENT" ]]; then
        CLIENT="$_DEFAULT_CLIENT"
    else
        echo "error: tensilelite-client not found at ${_DEFAULT_CLIENT}" >&2
        echo "       Build it with 'invoke build-client' or pass --client PATH." >&2
        exit 1
    fi
fi

# ── Validate variant ───────────────────────────────────────────────────────────
case "$VARIANT" in
    prms|prmsq|both) ;;
    *) echo "error: --variant must be prms, prmsq, or both" >&2; exit 1 ;;
esac

mkdir -p "$OUT_DIR"
export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# ── Helper: run one YAML and print the winner ──────────────────────────────────
_run_yaml() {
    local yaml="$1"
    local stem
    stem="$(basename "${yaml%.yaml}")"
    local out="${OUT_DIR}/${stem}"

    echo "========================================================================"
    echo "  Running: ${yaml##*/}"
    echo "  Output:  ${out}"
    echo "========================================================================"

    rm -rf "$out"
    mkdir -p "$out"

    "$PYTHON" "${TENSILELITE_ROOT}/Tensile/bin/Tensile" \
        "$yaml" "$out" \
        --prebuilt-client="$CLIENT"

    echo ""
    echo "── Results: ${yaml##*/} ──────────────────────────────────────────────"

    local csv="${out}/1_BenchmarkProblems/Cijk_Alik_Bljk_BBS_BH_PRMS_UserArgs_00/Data/00_Final.csv"
    if [[ ! -f "$csv" ]]; then
        echo "  No benchmark CSV found — all kernels may have been rejected."
        return
    fi

    "$PYTHON" - "$csv" <<'PYEOF'
import sys, csv, re

path = sys.argv[1]
with open(path) as f:
    header = f.readline().strip()
    data   = f.readline().strip()

cols = [c.strip() for c in header.split(',')]
vals = [v.strip() for v in data.split(',')]

results = []
for c, v in zip(cols[10:], vals[10:]):
    if not c.startswith('Cijk'):
        continue
    try:
        gf = float(v)
    except ValueError:
        continue
    if gf <= 0:
        continue
    mt   = re.search(r'MT(\d+x\d+x\d+)', c)
    sk   = re.search(r'_SK(\d+)_', c)
    pgr  = re.search(r'_PGR(\d+)_', c)
    eps  = re.search(r'_EPS(\d+)_', c)
    miwt = re.search(r'_MIWT(\d+_\d+)_', c)
    du   = re.search(r'_SUS(\d+)_', c)
    results.append((gf,
        mt.group(1) if mt else '?',
        int(sk.group(1)) if sk else 0,
        int(pgr.group(1)) if pgr else 0,
        int(eps.group(1)) if eps else 0,
        miwt.group(1).replace('_','x') if miwt else '?',
        int(du.group(1)) if du else 0,
    ))

if not results:
    print('  No valid results (all kernels returned -1 or were not benchmarked).')
    sys.exit(0)

results.sort(reverse=True)
print(f'  Total kernels: {len(results)}')
print(f'  {"Rank":>4}  {"GFlops":>12}  {"MT (M×N×K)":>15}  {"MIWT":>6}  {"DU":>4}  {"SK":>3}  {"PGR":>3}  {"EPS":>3}')
print('  ' + '-' * 72)
for rank, (gf, mt, sk, pgr, eps, miwt, du) in enumerate(results[:10], 1):
    print(f'  {rank:>4}  {gf:>12,.0f}  {mt:>15}  {miwt:>6}  {du:>4}  {sk:>3}  {pgr:>3}  {eps:>3}')
if len(results) > 10:
    print(f'  ... ({len(results) - 10} more)')
gf0, mt0, sk0, pgr0, eps0, miwt0, du0 = results[0]
print()
print(f'  Winner: {gf0:,.0f} GFlops  MT={mt0}  MIWT={miwt0}  DU={du0}  SK={sk0}  PGR={pgr0}')
PYEOF
    echo ""
}

# ── Main ───────────────────────────────────────────────────────────────────────
if [[ "$VARIANT" == "prms" || "$VARIANT" == "both" ]]; then
    _run_yaml "${YAML_DIR}/tune_subtile_bf16_prms_8192.yaml"
fi

if [[ "$VARIANT" == "prmsq" || "$VARIANT" == "both" ]]; then
    _run_yaml "${YAML_DIR}/tune_subtile_bf16_prmsq_8192.yaml"
fi

echo "Done. Results in: ${OUT_DIR}"
