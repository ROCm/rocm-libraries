#!/usr/bin/env bash
# rank-refresh.sh - Issue 18: refresh + snapshot the Phase-0 ranking/disposition.
#
# Re-runs rank-modules.py (Issue 17), snapshots its output under
# ranking-history/<date>-<pin>.md, and diffs the refresh against the most recent
# prior snapshot. Read-only/scaffolding: it runs NO mutation testing and edits NO
# production source. It preserves the Issue-17 formula/weights verbatim because it
# only invokes rank-modules.py (which enforces 0.40/0.25/0.15/0.20, min-max, no
# log10, no subtraction) - this wrapper never re-derives the score.
#
# Determinism: rank-modules.py output has no embedded timestamp, so the same input
# (same pin, same metrics) yields byte-identical content; re-running on the same
# day+pin overwrites the snapshot with identical bytes and reports "no change".
#
# Usage:
#   rank-refresh.sh [--out-dir <dir>] [--pin <sha>] [--metrics <json>]
#     --out-dir  where snapshots go (default: coverage/mutprod/ranking-history)
#     --pin      label for the snapshot (default: short HEAD of the source tree)
#     --metrics  optional JSON forwarded to rank-modules.py --metrics (fills the
#                PENDING cyclomatic/no_test_fraction inputs to compute scores)
set -u

die() { printf 'rank-refresh: ERROR: %s\n' "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"
[[ -n "$ROOT" ]] || ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
cd "$ROOT" || die "cannot cd to repo root: $ROOT"

RANKER="projects/hipblaslt/tensilelite/Tensile/Tests/unit/mutation/rank-modules.py"
SRC="projects/hipblaslt/tensilelite"
OUT_DIR="work/mutation/ranking-history"
PIN=""
METRICS=""

need_val() { [[ $# -ge 2 ]] || die "$1 requires a value"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) need_val "$@"; OUT_DIR="$2"; shift 2 ;;
    --pin)     need_val "$@"; PIN="$2"; shift 2 ;;
    --metrics) need_val "$@"; METRICS="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -f "$RANKER" ]] || die "ranker not found: $RANKER"
if [[ -z "$PIN" ]]; then
  PIN="$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknownpin)"
fi
# sanitize the pin for use in a filename (a branch-name pin could contain '/'):
# replace any char outside [A-Za-z0-9._-] with '-'.
PIN="${PIN//[^A-Za-z0-9._-]/-}"
[[ -n "$PIN" ]] || die "empty pin after sanitization"
DATE="$(date -u '+%Y%m%d')"
mkdir -p "$OUT_DIR" || die "cannot create out-dir: $OUT_DIR"
SNAP="$OUT_DIR/${DATE}-${PIN}.md"

# find the most recent prior snapshot (snapshots are <date>-<pin>.md, i.e. start
# with a digit; this excludes README.md and any non-snapshot .md file), excluding
# the one we're about to write.
PREV="$(ls -1 "$OUT_DIR"/[0-9]*.md 2>/dev/null | grep -v -F "$(basename "$SNAP")" | sort | tail -n1 || true)"

# generate current ranking to a temp, then place it
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
if [[ -n "$METRICS" ]]; then
  python3 "$RANKER" --metrics "$METRICS" --out "$TMP" >/dev/null 2>&1 || die "rank-modules.py failed"
else
  python3 "$RANKER" --out "$TMP" >/dev/null 2>&1 || die "rank-modules.py failed"
fi

if [[ -f "$SNAP" ]] && diff -q "$SNAP" "$TMP" >/dev/null 2>&1; then
  printf 'rank-refresh: no change - snapshot %s already up to date (deterministic re-run)\n' "$SNAP"
else
  cp "$TMP" "$SNAP" || die "failed to write snapshot: $SNAP"
  printf 'rank-refresh: wrote snapshot %s (pin=%s date=%s)\n' "$SNAP" "$PIN" "$DATE"
fi

if [[ -n "$PREV" && "$PREV" != "$SNAP" ]]; then
  if diff -q "$PREV" "$SNAP" >/dev/null 2>&1; then
    printf 'rank-refresh: no diff vs previous snapshot %s\n' "$PREV"
  else
    printf 'rank-refresh: DIFF vs previous snapshot %s:\n' "$PREV"
    diff -u "$PREV" "$SNAP" || true
  fi
else
  printf 'rank-refresh: no previous snapshot to diff against\n'
fi
