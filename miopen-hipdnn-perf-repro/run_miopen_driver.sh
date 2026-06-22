#!/usr/bin/env bash
# Native-MIOpen comparison for the hipDNN reproducer.
#
# Runs MIOpenDriver on the SAME conv as hipdnn_conv_cache_repro.cpp, as two
# separate processes, so you can reproduce the native MIOpen speedup on the
# second run (the persistent on-disk compiled-kernel cache in ~/.cache/miopen).
#
# Compare its second-run wall time against the hipDNN reproducer's per-process
# ~2.4 s plan build (see ./README.md).
#
# Portable: discovers the rocm-libraries checkout, MIOpenDriver and libMIOpen by
# name (no hard-coded build-dir names). See common.sh for overrides.
#
# Usage:
#   ./run_miopen_driver.sh           # two native runs (run 2 = warm disk cache, should be much faster)
#   ./run_miopen_driver.sh --cold    # wipe ~/.cache/miopen before run 1 (true cold start)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

ROCM="${ROCM_PATH:-/opt/rocm}"
REPO="$(repro_repo_root "$HERE")"

# Same conv as the C++ reproducer:
#   N16 C16 H16 W16  K16 R3 S3  pad1 stride1 dil1   forward-only, 100 iters, timed.
CONV_ARGS=(conv -n 16 -c 16 -H 16 -W 16 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1 -V 0 -i 100 -t 1)

DRIVER="$(repro_resolve "$REPO" MIOpenDriver MIOPEN_DRIVER)"

# Prefer the libMIOpen.so that ships next to the driver (same build); otherwise
# discover the newest one anywhere, or take the MIOPEN_LIB override.
MIOPEN_LIB="${MIOPEN_LIB:-}"
if [[ -z "$MIOPEN_LIB" ]]; then
    cand="$(cd "$(dirname "$DRIVER")/../lib" 2>/dev/null && pwd || true)/libMIOpen.so"
    if [[ -e "$cand" ]]; then
        MIOPEN_LIB="$cand"
    else
        MIOPEN_LIB="$(repro_resolve "$REPO" libMIOpen.so MIOPEN_LIB)"
    fi
fi
MIOPEN_LIBDIR="$(cd "$(dirname "$MIOPEN_LIB")" && pwd)"

export LD_LIBRARY_PATH="$MIOPEN_LIBDIR:$ROCM/lib:${LD_LIBRARY_PATH:-}"
# Some MIOpen builds also link the hipDNN backend lib; add its dir if present
# (harmless otherwise).
if BACKEND_LIB="$(repro_find_newest "$REPO" libhipdnn_backend.so)"; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(cd "$(dirname "$BACKEND_LIB")" && pwd)"
fi

COLD=0
for arg in "$@"; do
    case "$arg" in
        --cold) COLD=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [[ "$COLD" == "1" ]]; then
    # A true cold start must clear BOTH MIOpen user databases:
    #   - the compiled-kernel cache   (~/.cache/miopen)
    #   - the user find-db / perf-db  (~/.config/miopen)  <-- dominates the cold cost
    # When the home directory is on a network drive (NFS, etc.) MIOpen falls back
    # to local /tmp copies (/tmp/.cache/miopen, /tmp/.config/miopen) plus
    # /tmp/miopen-lockfiles. Wipe every location so the cold start is genuinely
    # cold regardless of where the databases actually live.
    echo "### wiping MIOpen kernel cache + find-db (home and /tmp fallbacks) for a true cold start"
    rm -rf "${HOME}/.cache/miopen" "${HOME}/.config/miopen" \
           /tmp/.cache/miopen /tmp/.config/miopen /tmp/miopen-lockfiles || true
fi

echo "driver            = $DRIVER"
echo "MIOpen libdir     = $MIOPEN_LIBDIR"
echo "conv              = ${CONV_ARGS[*]}"
echo
echo "==================== RUN 1 ===================="
time "$DRIVER" "${CONV_ARGS[@]}" >/dev/null
echo
echo "==================== RUN 2 (fresh process) ===================="
time "$DRIVER" "${CONV_ARGS[@]}" >/dev/null
