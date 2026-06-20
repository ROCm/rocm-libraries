#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The engine definition-of-done in one command: build the C++ engine fresh and
# prove its LLVM-IR emission is byte-identical to the Python engine across every
# kernel family. A green run means the dual-backend contract still holds.
#
# Usage:
#   ck_dsl_c/tools/check_byte_identity.sh [--ir] [--only SUBSTR] [--build-root DIR]
#                                         [--ref-pyroot DIR] [--ref-shim DIR]
#
#   --ir            also run the IR-level canonical diff (diagnostic; reports the
#                   one known benign agpr_alloc drift). Default: LLVM-IR gate only.
#   --only SUBSTR   restrict to families whose name contains SUBSTR (repeatable
#                   via comma, e.g. --only gemm,attention).
#   --build-root D  where to build the engine archive (default: $TMPDIR/ckc_verify).
#                   Use local disk; never NFS.
#   --ref-pyroot D  compare the C++ engine against the Python engine in another
#                   tree D instead of this one (also: env CKDSL_REF_PYROOT). See
#                   the transitional note below.
#   --ref-shim D    dir prepended to the reference PYTHONPATH to stub modules the
#                   reference tree lacks (also: env CKDSL_REF_SHIM).
#
# Transitional note: the default gate compares the C++ engine against THIS tree's
# Python engine. While engine reconciliation work is in flight, the C++ side may
# intentionally match a not-yet-merged reference (e.g. a deliberately suppressed
# scheduler-hint sequence), which shows up here as GEMM-family `sched.group.barrier`
# drift. That is expected during the window, not a regression. To check against the
# reconciliation reference instead, point --ref-pyroot/--ref-shim at that tree.
# Once the reference is merged into this tree, the default gate is authoritative
# again. See dsl_docs/development/troubleshooting.md.
#
# Exit status: 0 only if the engine builds AND no family shows a real .ll
# mismatch and none fails to compile. Non-zero otherwise.
set -euo pipefail

ONLY=""
RUN_IR=0
BUILD_ROOT="${TMPDIR:-/tmp}/ckc_verify"
REF_PYROOT="${CKDSL_REF_PYROOT:-}"
REF_SHIM="${CKDSL_REF_SHIM:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --ir)         RUN_IR=1; shift ;;
    --only)       ONLY="$2"; shift 2 ;;
    --build-root) BUILD_ROOT="$2"; shift 2 ;;
    --ref-pyroot) REF_PYROOT="$2"; shift 2 ;;
    --ref-shim)   REF_SHIM="$2"; shift 2 ;;
    -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# tools/ -> ck_dsl_c -> python  (the dir that goes on PYTHONPATH).
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
CKC="$(cd "$SELF_DIR/.." && pwd)"
PY="$(cd "$CKC/.." && pwd)"
ARCHIVE="$BUILD_ROOT/libckc_core.a"

only_arg=()
[ -n "$ONLY" ] && only_arg=(--only "$ONLY")
ref_arg=()
[ -n "$REF_PYROOT" ] && ref_arg+=(--pyroot "$REF_PYROOT")
[ -n "$REF_SHIM" ] && ref_arg+=(--shim "$REF_SHIM")
[ -n "$REF_PYROOT" ] && echo "   ref    : $REF_PYROOT (comparing against another tree's Python)"

echo "== building engine archive (fresh) =="
echo "   source : $CKC"
echo "   build  : $BUILD_ROOT"
cmake -S "$CKC" -B "$BUILD_ROOT" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_ROOT" -j"$(nproc)" >/dev/null
[ -f "$ARCHIVE" ] || { echo "FATAL: archive not produced: $ARCHIVE" >&2; exit 1; }
echo "   archive: $ARCHIVE"

# Freshness/provenance: print the build-id stamped into the archive we just
# built. This is informational + a sanity print so a stale archive is obvious
# in the log. Link a tiny probe that calls ckc_build_id(); fall back to strings
# if the probe cannot be built. The build_id TU is off the emission path, so
# this does not affect the .ll contract.
probe_src="$BUILD_ROOT/_build_id_probe.cpp"
probe_bin="$BUILD_ROOT/_build_id_probe"
cat > "$probe_src" <<'PROBE'
extern "C" const char* ckc_build_id(void);
extern "C" const char* ckc_engine_version(void);
#include <cstdio>
int main(){ printf("%s %s\n", ckc_build_id(), ckc_engine_version()); return 0; }
PROBE
if c++ -std=c++20 "$probe_src" "$ARCHIVE" -lm -o "$probe_bin" 2>/dev/null; then
  echo "   build-id: $("$probe_bin")"
else
  echo "   build-id: $(strings "$ARCHIVE" | grep -E '^[0-9a-f]{16}$' | head -1 || echo unknown) (via strings)"
fi

run_gate() {  # mode-label, extra run_diff args...
  local label="$1"; shift
  local out rc
  echo
  echo "== differential gate: $label =="
  set +e
  out="$(PYTHONPATH="$PY" python3 -m ck_dsl_c.tests.differential.run_diff \
          --archive "$ARCHIVE" "${only_arg[@]}" "${ref_arg[@]}" "$@" 2>&1)"
  rc=$?
  set -e
  echo "$out"
  # run_diff treats COMPILE_FAIL as non-parity, so it can exit 0 with a broken
  # family. Fail the gate explicitly if any family failed to compile.
  if echo "$out" | grep -q "COMPILE_FAIL"; then
    echo "GATE FAIL ($label): a family failed to compile." >&2
    return 1
  fi
  return $rc
}

status=0
run_gate "LLVM-IR (the contract)" --mode ll || status=1
if [ "$RUN_IR" = 1 ]; then
  # Informational: the IR canonical diff has one known benign drift
  # (agpr_alloc bare-int-list attr). Reported, not gating.
  run_gate "IR canonical (diagnostic)" --mode ir --canonical || true
fi

echo
if [ "$status" = 0 ]; then
  echo "RESULT: GREEN — engine builds and .ll emission is byte-identical to Python."
else
  echo "RESULT: RED — see the mismatching families above. The two engines disagree."
  echo "Localize with: run_diff --mode ir --canonical --only <family>  (see dsl_docs/development/troubleshooting.md)"
  if [ -z "$REF_PYROOT" ]; then
    echo "Transitional check: if the drift is GEMM-family 'sched.group.barrier' only,"
    echo "  it is the expected pre-merge reconciliation state, not a regression —"
    echo "  re-run against the reference tree: --ref-pyroot <tree> --ref-shim <stubs>."
  fi
fi
exit "$status"
