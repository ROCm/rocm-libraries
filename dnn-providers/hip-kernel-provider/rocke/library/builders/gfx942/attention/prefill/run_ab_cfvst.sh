#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

LIB="$ROCM_REPO/dnn-providers/hip-kernel-provider/rocke/library"
SOURCE="$LIB/kernels/gfx942/attention_dense.py"

BENCH="$SCRIPT_DIR/bench_rocke_once.py"
OUT="$SCRIPT_DIR/results"

# Canonical rocKE development environment created by the repo build.
PYTHON="${ROCKE_PYTHON:-$ROCM_REPO/build/rocke-pyenv/bin/python}"

ROUNDS="${1:-5}"

CSV="$OUT/results.csv"
LOG="$OUT/all_runs.log"
VALIDATION_LOG="$OUT/validation.txt"


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: rocKE Python environment not found:"
    echo "  $PYTHON"
    echo
    echo "Create it from the repository root with:"
    echo
    echo "  cmake -S dnn-providers/hip-kernel-provider -B build \\"
    echo "      -DHIPKERNELPROVIDER_ENABLE_ROCKE=ON"
    echo "  cmake --build build --target rocke-pyenv"
    exit 1
fi

# PyTorch from the ROCm wheel stack and rocKE must use the same
# HIP/COMGR runtime. Mixing the wheel-provided ROCm libraries with
# /opt/rocm can load two LLVM stacks into one process and fail with:
#
#   Option 'spirv-expand-step' registered more than once
#
# Resolve the ROCm SDK bundled with the Python environment and pin
# rocKE's runtime libraries to that exact installation.
CORE="$("$PYTHON" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("_rocm_sdk_core")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit(
        "ERROR: _rocm_sdk_core not found in the rocKE Python environment"
    )

print(Path(next(iter(spec.submodule_search_locations))).resolve())
PY
)"

CORE_LIB="$CORE/lib"

export ROCKE_BACKEND=python
export ROCKE_LLVM_FLAVOR=llvm23
export ROCKE_COMGR_LIB="$CORE_LIB/libamd_comgr.so.3"
export ROCKE_HIP_LIB="$CORE_LIB/libamdhip64.so.7"

# Keep the wheel ROCm libraries first and avoid a preloaded system LLVM/HIP
# library from contaminating the process.
export LD_LIBRARY_PATH="$CORE_LIB"
unset LD_PRELOAD

if [[ ! -f "$ROCKE_COMGR_LIB" ]]; then
    echo "ERROR: COMGR library not found:"
    echo "  $ROCKE_COMGR_LIB"
    exit 1
fi

if [[ ! -f "$ROCKE_HIP_LIB" ]]; then
    echo "ERROR: HIP library not found:"
    echo "  $ROCKE_HIP_LIB"
    exit 1
fi

# Fail early if this Python environment cannot import ROCm PyTorch.
"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda.is_available() is False")

print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Python: $PYTHON"
echo "ROCm SDK core: $CORE"
echo "COMGR: $ROCKE_COMGR_LIB"
echo "HIP:   $ROCKE_HIP_LIB"

mkdir -p "$OUT"

echo "B,S,HQ,HKV,D,round,order,variant,cfvst,ms" > "$CSV"
: > "$LOG"
: > "$VALIDATION_LOG"


# ------------------------------------------------------------
# Preserve the original source file.
#
# This guarantees that even if validation or benchmarking fails,
# attention_dense.py is restored to its original state.
# ------------------------------------------------------------

SOURCE_BACKUP="$(mktemp)"
cp "$SOURCE" "$SOURCE_BACKUP"

cleanup() {
    cp "$SOURCE_BACKUP" "$SOURCE"
    rm -f "$SOURCE_BACKUP"
}

trap cleanup EXIT


# ------------------------------------------------------------
# CFVST policy variants
# ------------------------------------------------------------

set_off() {
    "$PYTHON" - "$SOURCE" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()

off = 'return _rows_per_instr(head_size) == 1 and dtype == "fp16"'
on  = 'return _rows_per_instr(head_size) == 1'

if off in s:
    pass
elif on in s:
    s = s.replace(on, off, 1)
    p.write_text(s)
else:
    raise SystemExit(
        "ERROR: could not find CFVST policy return line"
    )
PY
}


set_on() {
    "$PYTHON" - "$SOURCE" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()

off = 'return _rows_per_instr(head_size) == 1 and dtype == "fp16"'
on  = 'return _rows_per_instr(head_size) == 1'

if off in s:
    s = s.replace(off, on, 1)
    p.write_text(s)
elif on in s:
    pass
else:
    raise SystemExit(
        "ERROR: could not find CFVST policy return line"
    )
PY
}


# ------------------------------------------------------------
# Numerical validation for one variant / shape
# ------------------------------------------------------------

validate_one() {
    local B="$1"
    local S="$2"
    local HQ="$3"
    local HKV="$4"
    local D="$5"
    local variant="$6"
    local expected="$7"

    local shape="b${B}_s${S}_hq${HQ}_hkv${HKV}_d${D}"
    local tmp="$OUT/${shape}_${variant}_validation.txt"

    echo
    echo "------------------------------------------------"
    echo "VALIDATE: $variant"
    echo "B=$B S=$S HQ=$HQ HKV=$HKV D=$D"
    echo "------------------------------------------------"

    B="$B" \
    S="$S" \
    HQ="$HQ" \
    HKV="$HKV" \
    D="$D" \
    VALIDATE=1 \
        "$PYTHON" "$BENCH" \
        | tee "$tmp" \
        | tee -a "$VALIDATION_LOG" \
        | tee -a "$LOG"

    local state

    state="$(awk '/^cfvst:/ {print $2}' "$tmp" | tail -1)"

    if [[ "$state" != "$expected" ]]; then
        echo "ERROR: expected cfvst=$expected, got $state"
        exit 1
    fi

    if ! grep -q '^VALIDATION=PASS' "$tmp"; then
        echo "ERROR: numerical validation did not pass"
        exit 1
    fi
}


# ------------------------------------------------------------
# Performance measurement for one variant / shape
# ------------------------------------------------------------

run_one() {
    local B="$1"
    local S="$2"
    local HQ="$3"
    local HKV="$4"
    local D="$5"
    local round="$6"
    local order="$7"
    local variant="$8"
    local expected="$9"

    local shape="b${B}_s${S}_hq${HQ}_hkv${HKV}_d${D}"
    local tmp="$OUT/${shape}_${variant}_${round}.txt"

    echo
    echo "================================================"
    echo "B=$B S=$S HQ=$HQ HKV=$HKV D=$D"
    echo "ROUND $round ($order) : $variant"
    echo "================================================"

    B="$B" \
    S="$S" \
    HQ="$HQ" \
    HKV="$HKV" \
    D="$D" \
        "$PYTHON" "$BENCH" \
        | tee "$tmp" \
        | tee -a "$LOG"

    local state
    local ms

    state="$(awk '/^cfvst:/ {print $2}' "$tmp" | tail -1)"
    ms="$(awk -F= '/^RESULT_MS=/ {print $2}' "$tmp" | tail -1)"

    if [[ "$state" != "$expected" ]]; then
        echo "ERROR: expected cfvst=$expected, got $state"
        exit 1
    fi

    if [[ -z "$ms" ]]; then
        echo "ERROR: RESULT_MS missing"
        exit 1
    fi

    echo \
"$B,$S,$HQ,$HKV,$D,$round,$order,$variant,$state,$ms" \
        >> "$CSV"
}


# ------------------------------------------------------------
# BF16 D128 test shapes
#
# Format:
#
# B S HQ HKV D
# ------------------------------------------------------------

SHAPES=(
    "1 4096 32 8 128"
    "1 4096 32 16 128"

    "1 8192 32 8 128"
    "1 8192 32 16 128"

    "1 16384 32 8 128"

    "16 4096 32 8 128"
    "16 4096 32 16 128"

    "16 8192 32 8 128"
)


# ------------------------------------------------------------
# Run every shape
# ------------------------------------------------------------

for shape in "${SHAPES[@]}"; do

    read -r B S HQ HKV D <<< "$shape"

    echo
    echo
    echo "################################################"
    echo "SHAPE"
    echo "B=$B S=$S HQ=$HQ HKV=$HKV D=$D"
    echo "################################################"


    # --------------------------------------------------------
    # Numerical correctness
    # --------------------------------------------------------

    echo
    echo "NUMERICAL VALIDATION"

    set_off
    validate_one \
        "$B" "$S" "$HQ" "$HKV" "$D" \
        "A_without_cfvst" \
        "False"

    set_on
    validate_one \
        "$B" "$S" "$HQ" "$HKV" "$D" \
        "B_with_cfvst" \
        "True"

    echo
    echo "Both variants passed numerical validation."


    # --------------------------------------------------------
    # Performance
    #
    # Odd rounds  : A -> B
    # Even rounds : B -> A
    #
    # This prevents a systematic A-first ordering bias.
    # --------------------------------------------------------

    for ((i=1; i<=ROUNDS; i++)); do

        if (( i % 2 == 1 )); then

            order="AB"

            set_off
            run_one \
                "$B" "$S" "$HQ" "$HKV" "$D" \
                "$i" "$order" \
                "A_without_cfvst" \
                "False"

            set_on
            run_one \
                "$B" "$S" "$HQ" "$HKV" "$D" \
                "$i" "$order" \
                "B_with_cfvst" \
                "True"

        else

            order="BA"

            set_on
            run_one \
                "$B" "$S" "$HQ" "$HKV" "$D" \
                "$i" "$order" \
                "B_with_cfvst" \
                "True"

            set_off
            run_one \
                "$B" "$S" "$HQ" "$HKV" "$D" \
                "$i" "$order" \
                "A_without_cfvst" \
                "False"

        fi

    done

done


# ------------------------------------------------------------
# Raw results
# ------------------------------------------------------------

echo
echo
echo "================================================"
echo "RESULTS"
echo "================================================"

column -s, -t "$CSV" 2>/dev/null || cat "$CSV"


# ------------------------------------------------------------
# Per-shape summary
# ------------------------------------------------------------

echo
echo "================================================"
echo "SUMMARY"
echo "================================================"

"$PYTHON" - "$CSV" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict


# Keep measurements paired by round.  The performance requirement is based on
# same-session A/B comparisons, so reduction/speedup must be computed for each
# A/B pair first and only then summarized with a median.
data = defaultdict(
    lambda: {
        "A_without_cfvst": {},
        "B_with_cfvst": {},
    }
)


with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):

        key = (
            int(row["B"]),
            int(row["S"]),
            int(row["HQ"]),
            int(row["HKV"]),
            int(row["D"]),
        )

        round_no = int(row["round"])

        data[key][row["variant"]][round_no] = float(row["ms"])


print(
    f"{'B':>4} "
    f"{'S':>7} "
    f"{'HQ':>4} "
    f"{'HKV':>4} "
    f"{'D':>4} "
    f"{'A median ms':>13} "
    f"{'B median ms':>13} "
    f"{'Median reduction':>17} "
    f"{'Median speedup':>15}"
)

print("-" * 96)


for key in sorted(data):

    B, S, HQ, HKV, D = key

    a_by_round = data[key]["A_without_cfvst"]
    b_by_round = data[key]["B_with_cfvst"]

    a_rounds = set(a_by_round)
    b_rounds = set(b_by_round)

    if a_rounds != b_rounds:
        missing_a = sorted(b_rounds - a_rounds)
        missing_b = sorted(a_rounds - b_rounds)

        print(
            f"ERROR: unpaired A/B data for "
            f"B={B} S={S} HQ={HQ} HKV={HKV} D={D}; "
            f"missing A rounds={missing_a}, missing B rounds={missing_b}"
        )
        continue

    rounds = sorted(a_rounds)

    if not rounds:
        print(
            f"ERROR: no A/B data for "
            f"B={B} S={S} HQ={HQ} HKV={HKV} D={D}"
        )
        continue

    a = [a_by_round[r] for r in rounds]
    b = [b_by_round[r] for r in rounds]

    # These are useful descriptive latency medians.
    ma = statistics.median(a)
    mb = statistics.median(b)

    # The actual A/B summary is paired per round first, then medianed.
    reductions = [
        (a_by_round[r] - b_by_round[r]) / a_by_round[r] * 100.0
        for r in rounds
    ]
    speedups = [
        a_by_round[r] / b_by_round[r]
        for r in rounds
    ]

    median_reduction = statistics.median(reductions)
    median_speedup = statistics.median(speedups)

    print(
        f"{B:4d} "
        f"{S:7d} "
        f"{HQ:4d} "
        f"{HKV:4d} "
        f"{D:4d} "
        f"{ma:13.6f} "
        f"{mb:13.6f} "
        f"{median_reduction:16.2f}% "
        f"{median_speedup:14.4f}x"
    )


print()
print("Detailed paired measurements:")

for key in sorted(data):

    B, S, HQ, HKV, D = key

    a_by_round = data[key]["A_without_cfvst"]
    b_by_round = data[key]["B_with_cfvst"]

    common_rounds = sorted(set(a_by_round) & set(b_by_round))

    print()
    print(
        f"B={B} S={S} HQ={HQ} HKV={HKV} D={D}"
    )

    for r in common_rounds:
        a = a_by_round[r]
        b = b_by_round[r]
        reduction = (a - b) / a * 100.0
        speedup = a / b

        print(
            f"  round {r}: "
            f"A={a:.6f} ms, "
            f"B={b:.6f} ms, "
            f"reduction={reduction:.2f}%, "
            f"speedup={speedup:.4f}x"
        )
PY


echo
echo "CSV:        $CSV"
echo "Log:        $LOG"
echo "Validation: $VALIDATION_LOG"
echo
echo "Original attention_dense.py restored."