#!/usr/bin/env bash
# Run the hipDNN backend-API conv reproducer as TWO separate processes to expose
# cross-process (non-)caching of the plan build.
#
# Portable: discovers the rocm-libraries checkout and the runtime libraries it
# needs by name (no hard-coded build-dir names). See common.sh for overrides.
#
# Usage:
#   ./run_repro.sh              # two back-to-back processes (process 2 = "warm" disk caches)
#   ./run_repro.sh --cold       # wipe ~/.cache/miopen before process 1 (true cold start)
#
# What to look for:
#   If "conv plan build" in process 2 is just as large as process 1, the hipDNN
#   backend plan-build path is NOT benefiting from any cross-process cache.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

ROCM="${ROCM_PATH:-/opt/rocm}"
REPO="$(repro_repo_root "$HERE")"

BIN="$HERE/hipdnn_conv_cache_repro"
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: $BIN not built. Run ./build_repro.sh first." >&2
    exit 1
fi

# hipDNN must load the miopen engine plugin (libmiopen_plugin.so), which in turn
# is dynamically linked against MIOpen (libMIOpen.so) and the hipDNN backend lib.
# Discover all three and put their dirs on the loader path.
PLUGIN_LIB="$(repro_resolve "$REPO" libmiopen_plugin.so HIPDNN_PLUGIN_LIB)"
MIOPEN_LIB="$(repro_resolve "$REPO" libMIOpen.so MIOPEN_LIB)"
BACKEND_LIB="$(repro_resolve "$REPO" libhipdnn_backend.so HIPDNN_BACKEND_LIB)"

export HIPDNN_PLUGIN_DIR="$(cd "$(dirname "$PLUGIN_LIB")" && pwd)"
MIOPEN_LIBDIR="$(cd "$(dirname "$MIOPEN_LIB")" && pwd)"
BACKEND_LIBDIR="$(cd "$(dirname "$BACKEND_LIB")" && pwd)"

export LD_LIBRARY_PATH="$BACKEND_LIBDIR:$MIOPEN_LIBDIR:$ROCM/lib:${LD_LIBRARY_PATH:-}"

if [[ "${1:-}" == "--cold" ]]; then
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

echo "HIPDNN_PLUGIN_DIR = $HIPDNN_PLUGIN_DIR"
echo "MIOpen libdir     = $MIOPEN_LIBDIR"
echo "backend libdir    = $BACKEND_LIBDIR"
echo
echo "==================== PROCESS 1 ===================="
"$BIN"
echo
echo "==================== PROCESS 2 (fresh process) ===================="
"$BIN"
