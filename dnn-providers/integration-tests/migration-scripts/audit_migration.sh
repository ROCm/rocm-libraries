#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Post-run audit: confirm the migration output is PR-ready.
# Run this AFTER run_capture_pipeline.sh has produced bundles.
#
# Checks:
#   1. Count reconciliation: census == captured == placed
#   2. Every migrated case carries `reference_source` metadata
#   3. Migrated cases have NO golden reference data (per AC)
#   4. Idempotency: git diff on the bundle tree is clean
#
# Usage:
#   audit_migration.sh [work_dir] [bundle_dir]
#   (defaults: work_dir=/tmp/almiopen2279, bundle_dir=../integration-test-bundles)
#
set -uo pipefail

WORK="${1:-/tmp/almiopen2279}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUNDLE_DIR="${2:-$(cd "$SCRIPT_DIR/.." && pwd)/integration-test-bundles}"

FAIL=0
pass() { echo "  ✅ $*"; }
fail() { echo "  ❌ $*"; FAIL=1; }
info() { echo "  ·  $*"; }

echo "=== Migration Audit ==="
echo "  work dir:   $WORK"
echo "  bundle dir: $BUNDLE_DIR"
echo ""

# ── Check 1: count reconciliation ────────────────────────────────────────
echo "--- Check 1: count reconciliation ---"
CENSUS_JSON="$WORK/census.json"
if [ -f "$CENSUS_JSON" ]; then
    CENSUS_N=$(python3 -c "import json;print(len(json.load(open('$CENSUS_JSON')).get('cases',[])))" 2>/dev/null || echo "?")
    info "census graph cases: $CENSUS_N"
else
    fail "census.json not found at $CENSUS_JSON (did the pipeline run?)"
    CENSUS_N="?"
fi

CAPTURED_N=$(find "$WORK/captured" -name '*.json' ! -name '*.meta.json' 2>/dev/null \
    | while read -r f; do [ "$(basename "$f" .json)" = "$(basename "$(dirname "$f")")" ] && echo x; done | wc -l | tr -d ' ')
info "captured cases:     ${CAPTURED_N:-0}"

# placed = sweep cases (count ids in every sweep.json) + standalone ported bundles
PLACED_N=$(python3 - "$BUNDLE_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
n = 0
for sweep in root.rglob("sweep.json"):
    try:
        data = json.load(open(sweep))
    except Exception:
        continue
    for c in data.get("cases", []):
        if (c.get("metadata") or {}).get("reference_source"):
            n += 1
for meta in root.rglob("*.meta.json"):
    try:
        m = json.load(open(meta))
    except Exception:
        continue
    if m.get("reference_source"):
        n += 1
print(n)
PY
)
info "placed (ported):    ${PLACED_N:-0}"

if [ "$CENSUS_N" != "?" ] && [ "$CENSUS_N" = "$CAPTURED_N" ] && [ "$CAPTURED_N" = "$PLACED_N" ]; then
    pass "census == captured == placed ($CENSUS_N)"
else
    fail "counts do not reconcile (census=$CENSUS_N captured=$CAPTURED_N placed=$PLACED_N)"
fi
echo ""

# ── Check 2: reference_source present ─────────────────────────────────────────
echo "--- Check 2: reference_source provenance ---"
PORTED_FILES=$(grep -rl "reference_source" "$BUNDLE_DIR" 2>/dev/null | wc -l | tr -d ' ')
if [ "${PORTED_FILES:-0}" -gt 0 ]; then
    pass "$PORTED_FILES file(s) carry reference_source metadata"
else
    fail "no files carry reference_source — migration output missing"
fi
echo ""

# ── Check 3: no golden data on migrated cases ────────────────────────────
echo "--- Check 3: migrated cases have no golden reference data ---"
# A migrated (ported) sweep/standalone must not reference golden/dvc data.
GOLDEN_ON_PORTED=$(python3 - "$BUNDLE_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
bad = []
for sweep in root.rglob("sweep.json"):
    try:
        data = json.load(open(sweep))
    except Exception:
        continue
    for c in data.get("cases", []):
        if (c.get("metadata") or {}).get("reference_source") and c.get("golden"):
            bad.append(f"{sweep}:{c.get('id')}")
print("\n".join(bad))
PY
)
if [ -z "$GOLDEN_ON_PORTED" ]; then
    pass "no ported case carries golden data"
else
    fail "ported cases with golden data (violates AC):"
    echo "$GOLDEN_ON_PORTED" | sed 's/^/       /'
fi
echo ""

# ── Check 4: idempotency ─────────────────────────────────────────────────
echo "--- Check 4: idempotency (git diff must be clean) ---"
if ! git -C "$BUNDLE_DIR" rev-parse --git-dir > /dev/null 2>&1; then
    info "SKIP: $BUNDLE_DIR is not in a git repo — cannot check idempotency via diff"
elif git -C "$BUNDLE_DIR" diff --quiet -- "$BUNDLE_DIR" 2>/dev/null; then
    pass "git diff clean on bundle tree"
else
    # untracked new files are EXPECTED (first migration); only a *diff* on
    # tracked files signals non-idempotency after a second run.
    CHANGED=$(git -C "$BUNDLE_DIR" diff --name-only -- "$BUNDLE_DIR" 2>/dev/null | wc -l | tr -d ' ')
    if [ "${CHANGED:-0}" -gt 0 ]; then
        fail "$CHANGED tracked file(s) differ — pipeline may not be idempotent"
        git -C "$BUNDLE_DIR" diff --stat -- "$BUNDLE_DIR" 2>/dev/null | sed 's/^/       /'
    else
        pass "only untracked new files (expected on first migration)"
    fi
fi
echo ""

echo "=== Audit $([ $FAIL -eq 0 ] && echo PASSED || echo FAILED) ==="
exit $FAIL
