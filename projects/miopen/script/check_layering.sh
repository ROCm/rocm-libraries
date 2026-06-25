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
# Phase 2 -- public C API surface. No new include boundary; extended later.
# =============================================================================
run_phase2() { :; }

# =============================================================================
# Phase 3 -- miopen_utils (MIOpen Utilities)
# Rule (when implemented): miopen_utils may include only common_utils/ and
# miopen/miopen.h, and there are 0 driver/<->test/ cross-includes.
# Wire on in the Phase 3 PR, e.g.:
#   check_no_forbidden_includes "miopen_utils/include" "miopen/" "miopen_utils" "miopen/miopen.h"
# =============================================================================
run_phase3() { :; }

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
