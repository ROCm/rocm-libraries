#!/usr/bin/env bash
# Build the standalone hipDNN backend-API conv reproducer (hipDNN backend API only).
#
# Portable: discovers the rocm-libraries checkout and the build artifacts it
# needs by name, so it does not depend on where this folder lives or on any
# particular build-directory name. See common.sh for environment overrides.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

ROCM="${ROCM_PATH:-/opt/rocm}"
HIPCC="${HIPCC:-$ROCM/bin/hipcc}"

REPO="$(repro_repo_root "$HERE")"

# hipDNN public headers (in-tree source location; overridable).
BACKEND_INC="${HIPDNN_BACKEND_INCLUDE:-$REPO/projects/hipdnn/backend/include}"
# Generated export header + backend shared library (override or auto-discover).
EXPORT_HDR="$(repro_resolve "$REPO" hipdnn_backend_export.h HIPDNN_EXPORT_HEADER)"
BACKEND_LIB="$(repro_resolve "$REPO" libhipdnn_backend.so HIPDNN_BACKEND_LIB)"

EXPORT_INC="$(cd "$(dirname "$EXPORT_HDR")" && pwd)"
LIBDIR="$(cd "$(dirname "$BACKEND_LIB")" && pwd)"

OUT="$HERE/hipdnn_conv_cache_repro"

echo "repo root : $REPO"
echo "hipcc     : $HIPCC"
echo "backend   : $BACKEND_INC"
echo "export hdr: $EXPORT_INC"
echo "libdir    : $LIBDIR"

if [[ ! -e "$BACKEND_INC/hipdnn_backend.h" ]]; then
    echo "ERROR: hipdnn_backend.h not found in $BACKEND_INC (set HIPDNN_BACKEND_INCLUDE)" >&2
    exit 1
fi
if [[ ! -x "$HIPCC" ]]; then
    echo "ERROR: hipcc not found/executable at $HIPCC (set HIPCC or ROCM_PATH)" >&2
    exit 1
fi

"$HIPCC" -std=c++17 -O2 -g \
    -I"$BACKEND_INC" -I"$EXPORT_INC" -I"$ROCM/include" \
    "$HERE/hipdnn_conv_cache_repro.cpp" \
    -L"$LIBDIR" -lhipdnn_backend \
    -L"$ROCM/lib" -lamdhip64 \
    -Wl,-rpath,"$LIBDIR" -Wl,-rpath,"$ROCM/lib" \
    -o "$OUT"

echo "built: $OUT"
