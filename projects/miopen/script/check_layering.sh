#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# MIOpen layering guard.
#
# Enforces the layered architecture described in
# docs/plans/MIOpenLayeringRefactorPlan.md so that a future change which punches
# through a layer boundary is rejected automatically. A one-time cleanup
# silently unravels without enforcement; this guard is what keeps the layering
# clean over time.
#
# Target layers (bottom -> top):
#   1. common_utils  (Core Utilities)  -- STL/system only, NO miopen/ deps
#   2. MIOpen library                  -- exports the public C API
#   3. miopen_utils  (MIOpen Utilities)-- common_utils + miopen/miopen.h only
#   4. driver + tests                  -- public API + util libs only
#
# This is the per-phase guard framework introduced in Phase 1. Each later phase
# extends it with its own boundary checks (see the run_phase_* functions and the
# "extended in Phase N" markers below). Today only the Phase 1 checks are wired
# on; the rest are stubs that document where future checks attach.
#
# Run from anywhere; the script locates the MIOpen project root itself.
# Exit status: 0 = all enforced layers clean, 1 = at least one violation.

set -u

# Resolve projects/miopen/ regardless of caller cwd (script lives in script/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIOPEN_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$MIOPEN_ROOT" || exit 2

FAILURES=0

# -----------------------------------------------------------------------------
# Whitelist of documented, intentional layering exceptions.
#
# Each entry is an extended-regex matched against a path relative to
# projects/miopen/. A file whose path matches any entry is exempt from the
# include-boundary checks. ALWAYS add a one-line reason when adding an entry
# (see docs/plans/MIOpenLayering-Phase1.md section 8).
# -----------------------------------------------------------------------------
WHITELIST=(
  # RTC-compiled kernel headers register includes as flat name/content pairs
  # with no directory structure, so <common_utils/...> cannot resolve at
  # runtime. These headers therefore stay in src/kernels/ by design.
  '^src/kernels/'
  # Intentionally a *separate* lightweight throw helper (std::runtime_error /
  # COMMON_THROW), NOT a move of the heavy miopen/errors.hpp. The only "miopen"
  # text it contains is explanatory comments, not includes -- not a violation.
  '^common_utils/include/common_utils/errors\.hpp$'
  # Optional forwarding shim (only present if a common_utils consumer needs the
  # host float8 type). It references ../../src/kernels/hip_float8.hpp by
  # relative path on purpose (same RTC flat-include constraint as above).
  '^common_utils/include/common_utils/float8\.hpp$'
  # TensorDesc + tensor_holder each contain a single #include <miopen/tensor.hpp>
  # guarded by #ifdef MIOPEN_BUILD_TESTING -- the sanctioned, test-only bridge to
  # the internal type (see MIOpenLayering-Phase3.md section 4.2). Driver and
  # miopen_utils builds never define that macro, so they stay internal-free.
  '^miopen_utils/include/miopen_utils/tensor_desc\.hpp$'
  '^miopen_utils/include/miopen_utils/tensor_holder\.hpp$'
  # ---- Documented residual driver<->test cross-includes (Phase-4 follow-ups) ----
  # These could not be removed in Phase 3 without work that belongs to Phase 4
  # (driver cleanup). Tracked, not silent.
  #
  # These tests use driver/driver.hpp for GPUMem. Extracting GPUMem into
  # miopen_utils requires dropping its internal MIOPEN_LOG_CUSTOM logging (a
  # behavior-change concession) and belongs with the Phase-4 driver cleanup.
  '^test/gtest/find_mode_trust_verify\.cpp$'
  '^test/gtest/kernel_tuning_net\.cpp$'
  # layout_transpose uses driver/conv_common.hpp (driver conv test helpers with
  # internal deps). Relocating it is Phase-4 work. (#7291 also left conv_common.)
  '^test/gtest/layout_transpose\.cpp$'
)

is_whitelisted() {
  local path="$1"
  local pat
  for pat in "${WHITELIST[@]}"; do
    if [[ "$path" =~ $pat ]]; then
      return 0
    fi
  done
  return 1
}

# -----------------------------------------------------------------------------
# Core check: no real (non-commented) #include of any header under a forbidden
# include-prefix exists within a directory subtree.
#
#   $1 = directory to scan (relative to projects/miopen)
#   $2 = forbidden include prefix, e.g. 'miopen/'
#   $3 = human-readable label for messages
#   $4 = optional allowed exact include (e.g. 'miopen/miopen.h') -- a single
#        header under the forbidden prefix that IS permitted for this layer.
#
# Only matches '#include' anchored to the start of the line (after optional
# whitespace), which excludes commented-out '// #include ...' lines.
# -----------------------------------------------------------------------------
check_no_forbidden_includes() {
  local dir="$1" prefix="$2" label="$3" allowed="${4:-}"
  if [[ ! -d "$dir" ]]; then
    echo "  [skip] $dir does not exist (nothing to check for $label)"
    return 0
  fi

  local pattern="^[[:space:]]*#[[:space:]]*include[[:space:]]*[<\"]${prefix}"
  local hits
  hits="$(grep -rnE "$pattern" "$dir" 2>/dev/null)" || true

  local violations=0
  local line file inc
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    file="${line%%:*}"
    # Strip the build tree; never enforce against generated artifacts.
    [[ "$file" == */build/* || "$file" == build/* ]] && continue
    is_whitelisted "$file" && continue
    # Allow one explicitly permitted header under the forbidden prefix.
    if [[ -n "$allowed" ]]; then
      inc="$(printf '%s\n' "$line" | grep -oE "[<\"]${prefix}[^>\"]*" | head -1)"
      inc="${inc#[<\"]}"
      [[ "$inc" == "$allowed" ]] && continue
    fi
    if [[ "$violations" -eq 0 ]]; then
      echo "  [FAIL] $label must not #include <${prefix}...>${allowed:+ (except <$allowed>)}:"
    fi
    echo "         $line"
    violations=$((violations + 1))
  done <<< "$hits"

  if [[ "$violations" -eq 0 ]]; then
    echo "  [ok]   $label: no forbidden <${prefix}...> includes"
  else
    FAILURES=$((FAILURES + violations))
  fi
}

# =============================================================================
# Phase 1 -- common_utils (Core Utilities)
# Rule: common_utils depends on STL/system only; ZERO miopen/ includes.
# =============================================================================
run_phase1() {
  echo "Phase 1: common_utils (Core Utilities)"
  check_no_forbidden_includes "common_utils/include" "miopen/" "common_utils"
}

# =============================================================================
# Phase 2 -- public C API surface. No new include boundary is punched, so there
# is no forbidden-include rule here. Instead the Phase-2 assertion is additive:
# every new public entry point must be (a) declared in the public header and
# (b) exported from the built library. The header check always runs (works in
# the pre-commit lint context); the export check runs only when the built lib
# exists (build lane), so it is a no-op pre-build.
# =============================================================================
PHASE2_APIS=(
  miopenGetTensorLayout
  miopenGetTensorElementSpace
  miopenIsTensorPacked
  miopenGetTensorVectorLength
  miopenGetTensorDescriptorV2
  miopenGetConvolutionPaddingMode
  miopenGetPoolingPaddingMode
  miopenGetSolverName
  miopenGetSolverIdByName
  miopenSetDebugFlag
  miopenGetDebugFlag
)

run_phase2() {
  echo "Phase 2: public C API surface"
  local header="include/miopen/miopen.h"
  local api missing=0

  # (a) source-level: each new API is declared in the public header.
  if [[ -f "$header" ]]; then
    for api in "${PHASE2_APIS[@]}"; do
      if ! grep -q "$api" "$header"; then
        echo "  [FAIL] public API not declared in $header: $api"
        missing=$((missing + 1))
      fi
    done
    if [[ "$missing" -eq 0 ]]; then
      echo "  [ok]   all ${#PHASE2_APIS[@]} Phase-2 APIs declared in $header"
    else
      FAILURES=$((FAILURES + missing))
    fi
  else
    echo "  [skip] $header not found (nothing to check)"
  fi

  # (b) build lane: each new API is a defined, default-visibility dynamic symbol
  #     in libMIOpen.so. Opt-in via MIOPEN_LAYERING_CHECK_EXPORTS=1 (set in the
  #     build lane, after linking). Left unset in the pre-commit lint lane so a
  #     stale/partial build/ tree never produces a spurious failure there.
  local lib="${MIOPEN_LIB_PATH:-build/lib/libMIOpen.so}"
  if [[ "${MIOPEN_LAYERING_CHECK_EXPORTS:-0}" != "1" ]]; then
    echo "  [skip] export assertion off (set MIOPEN_LAYERING_CHECK_EXPORTS=1 in the build lane)"
  elif [[ -f "$lib" ]] && command -v nm >/dev/null 2>&1; then
    local exported unexported=0
    exported="$(nm -D --defined-only "$lib" 2>/dev/null)" || exported=""
    for api in "${PHASE2_APIS[@]}"; do
      if ! grep -qE "[[:space:]]${api}\$" <<< "$exported"; then
        echo "  [FAIL] public API not exported from $lib: $api"
        unexported=$((unexported + 1))
      fi
    done
    if [[ "$unexported" -eq 0 ]]; then
      echo "  [ok]   all ${#PHASE2_APIS[@]} Phase-2 APIs exported from $lib"
    else
      FAILURES=$((FAILURES + unexported))
    fi
  else
    echo "  [skip] $lib not built (export assertion deferred to build lane)"
  fi
}

# =============================================================================
# Phase 3 -- miopen_utils (MIOpen Utilities)
# Rule (when implemented): miopen_utils may include only common_utils/ and
# miopen/miopen.h, and there are 0 driver/<->test/ cross-includes.
# Wire on in the Phase 3 PR, e.g.:
#   check_no_forbidden_includes "miopen_utils/include" "miopen/" "miopen_utils" "miopen/miopen.h"
# =============================================================================
# Check that no file under $1 #includes a header living under $2 (a sibling
# layer), i.e. a cross-include. Matches the observed include forms:
#   "X.hpp"  "../X.hpp"  "../<other>/X.hpp"  <../<other>/X.hpp>
# where <other> is the sibling directory name. Commented lines are excluded
# (pattern anchored to line start after optional whitespace).
check_no_cross_includes() {
  local dir="$1" other="$2" label="$3"
  if [[ ! -d "$dir" ]]; then
    echo "  [skip] $dir does not exist (nothing to check for $label)"
    return 0
  fi
  local pattern="^[[:space:]]*#[[:space:]]*include[[:space:]]*[<\"](\.\./)*${other}/"
  local hits
  hits="$(grep -rnE "$pattern" "$dir" 2>/dev/null)" || true
  local violations=0 line file
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    file="${line%%:*}"
    [[ "$file" == */build/* || "$file" == build/* ]] && continue
    is_whitelisted "$file" && continue
    if [[ "$violations" -eq 0 ]]; then
      echo "  [FAIL] $label:"
    fi
    echo "         $line"
    violations=$((violations + 1))
  done <<< "$hits"
  if [[ "$violations" -eq 0 ]]; then
    echo "  [ok]   $label: none"
  else
    FAILURES=$((FAILURES + violations))
  fi
}

run_phase3() {
  echo "Phase 3: miopen_utils (MIOpen Utilities) + driver<->test decoupling"
  # (a) miopen_utils may include only common_utils/ and miopen/miopen.h.
  check_no_forbidden_includes "miopen_utils/include" "miopen/" "miopen_utils" "miopen/miopen.h"
  # (b) zero driver<->test cross-includes, both directions.
  check_no_cross_includes "driver" "test" "driver/ must not include test/ headers"
  check_no_cross_includes "test" "driver" "test/ must not include driver/ headers"
}

# =============================================================================
# Phase 4 -- driver. Rule (when implemented): driver/ includes of miopen/
# resolve only to miopen/miopen.h (+ miopen/config.h) and `grep MIOPEN_THROW
# driver/` is empty. Wire on in the Phase 4 PR.
# =============================================================================
run_phase4() { :; }

# =============================================================================
# Phase 5 -- tests. Rule (when implemented): intentional internal-access points
# are minimal and whitelisted. Wire on in the Phase 5 PR.
# =============================================================================
run_phase5() { :; }

echo "=== MIOpen layering guard (run from $MIOPEN_ROOT) ==="
run_phase1
run_phase2
run_phase3
run_phase4
run_phase5
echo

if [[ "$FAILURES" -ne 0 ]]; then
  echo "LAYERING GUARD FAILED: $FAILURES violation(s)."
  echo "If a violation is intentional, add the file to WHITELIST in"
  echo "  script/check_layering.sh with a one-line reason, and document it in"
  echo "  docs/plans/MIOpenLayering-Phase1.md section 8."
  exit 1
fi

echo "LAYERING GUARD PASSED: all enforced layer boundaries are clean."
exit 0
