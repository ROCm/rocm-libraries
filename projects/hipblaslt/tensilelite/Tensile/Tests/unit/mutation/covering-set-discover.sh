#!/usr/bin/env bash
# covering-set-discover.sh — Issue 9: Phase-2 covering-set discovery.
#
# For a target module it discovers the candidate unit test files and
# characterization dirs that exercise it, OPTIONALLY measures their combined
# coverage of that module, and emits a covering-set.json status artifact. It is a
# DISCOVERY helper: it never runs mutation tooling (no `mutmut run`, no
# `mutmut apply`) and never edits production source.
#
# THE 80% GATE + STOP-BEFORE-MUTATION contract:
#   - measured mode: if the discovered set covers the module at >= threshold
#     (default 80%), status="ok"; otherwise status="defer".
#   - if NO candidates are discovered, status="defer" and NO pytest runs. The
#     helper NEVER falls back to the whole `-m unit` suite on its own — the
#     coverage command always constrains pytest to the discovered paths, so a
#     bare full-suite run cannot happen without an explicit user-supplied set.
#   - status is only ever "ok" or "defer"; "defer" means do not proceed to a
#     mutation slice for this module yet.
#
# DISCOVERY (read-only, no docker/pytest):
#   1. direct unit test: Tensile/Tests/unit/test_<basename>.py, if it exists.
#   2. import references in Tensile/Tests/unit/test_*.py that import the module
#      (matched by dotted path e.g. Tensile.SolutionStructs.Solution, or
#      `from <parent-package> import <basename>`).
#   3. characterization dirs under Tensile/Tests/unit/characterization/* whose
#      Python files import the module by the same patterns (the DIR is selected,
#      since pytest runs a char dir as a unit).
#
# COVERAGE COMMAND SHAPE (measured mode; also recorded verbatim in dry-run):
#   docker exec -w /work/<src> <container> \
#     pytest -m unit --cov=<module-without-.py> --cov-report=term-missing <paths...>
#
# Usage:
#   covering-set-discover.sh --module Tensile/<...>.py [--out <dir>]
#       [--container tl-mut] [--src projects/hipblaslt/tensilelite]
#       [--threshold 80] [--dry-run|--list-only]
#
# Output artifact: <out>/covering-set.json  (default out: a per-module dir under
#   work/mutation/covering/<basename>).
#
# Testability: source with COVERING_SET_LIB_ONLY=1 to expose the pure helpers
# (derive_names, discover_candidates, emit_json) without running main() — see
# tests/covering-set-discover-selftest.sh.

set -u

die() { printf 'covering-set-discover: ERROR: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------- pure helpers
# derive_names — from a module path set globals BASENAME, DOTTED, PARENT_DOTTED.
#   Tensile/SolutionStructs/Solution.py
#     -> BASENAME=Solution DOTTED=Tensile.SolutionStructs.Solution
#        PARENT_DOTTED=Tensile.SolutionStructs
#   Tensile/BenchmarkSplitter.py
#     -> BASENAME=BenchmarkSplitter DOTTED=Tensile.BenchmarkSplitter
#        PARENT_DOTTED=Tensile
derive_names() {
  local m="$1" noext dotted
  m="${m#./}"
  noext="${m%.py}"
  BASENAME="${noext##*/}"
  dotted="${noext//\//.}"
  DOTTED="$dotted"
  if [[ "$dotted" == *.* ]]; then PARENT_DOTTED="${dotted%.*}"; else PARENT_DOTTED="$dotted"; fi
}

# _re_escape_dots — escape regex metacharacters in a dotted path for grep -E.
_re_escape_dots() { printf '%s' "$1" | sed 's/[.[\*^$()+?{|]/\\&/g'; }

# _imports_module — return 0 if $1 (a file) imports the module described by the
# globals DOTTED/PARENT_DOTTED/BASENAME. Two anchored patterns:
#   - the dotted module path bounded by non-identifier chars (so
#     Tensile.SolutionStructs.Solution does NOT match SolutionFoo), and
#   - `from <parent> import ... <basename>` (word-bounded basename).
_imports_module() {
  local f="$1" ed ep
  ed="$(_re_escape_dots "$DOTTED")"
  ep="$(_re_escape_dots "$PARENT_DOTTED")"
  grep -Eq "(^|[^A-Za-z0-9_.])${ed}([^A-Za-z0-9_]|\$)" "$f" && return 0
  grep -Eq "^[[:space:]]*from[[:space:]]+${ep}[[:space:]]+import[[:space:]]+.*\b${BASENAME}\b" "$f" && return 0
  return 1
}

# parse_cov_pct — extract the coverage percent (e.g. 82.50) for a module from a
# `pytest --cov=... --cov-report=term-missing` output. Prefers the module's own
# per-file row (grep -F on the module path); falls back to the TOTAL row. Handles
# the repo's precision=2 formatting (NN.DD%), which a naive `[0-9]+%` grep would
# misread as the fractional digits. Prints the number without '%', or nothing.
#   Args: output module-path
parse_cov_pct() {
  local out="$1" mod="$2" line
  line="$(printf '%s\n' "$out" | grep -F -- "$mod" | grep -E '[0-9]+(\.[0-9]+)?%' | tail -1)"
  [[ -z "$line" ]] && line="$(printf '%s\n' "$out" | grep -E '^TOTAL' | tail -1)"
  printf '%s\n' "$line" | grep -oE '[0-9]+(\.[0-9]+)?%' | tail -1 | tr -d '%'
}

# discover_candidates — print src-relative candidate paths, one per line, sorted
# unique. Reads only the filesystem under <src>; no docker/pytest. Args: module src.
discover_candidates() {
  local module="$1" src="$2"
  derive_names "$module"
  local unit="Tensile/Tests/unit"
  local chardir="$unit/characterization"
  local out="" f d
  # 1. direct unit test file
  if [[ -f "$src/$unit/test_${BASENAME}.py" ]]; then
    out+="$unit/test_${BASENAME}.py"$'\n'
  fi
  # 2. import references in unit test_*.py
  if [[ -d "$src/$unit" ]]; then
    for f in "$src/$unit"/test_*.py; do
      [[ -f "$f" ]] || continue
      if _imports_module "$f"; then out+="$unit/$(basename "$f")"$'\n'; fi
    done
  fi
  # 2b. import references in top-level characterization/test_*.py (files that live
  #     directly under characterization/, not inside a per-feature subdir).
  if [[ -d "$src/$chardir" ]]; then
    for f in "$src/$chardir"/test_*.py; do
      [[ -f "$f" ]] || continue
      if _imports_module "$f"; then out+="$chardir/$(basename "$f")"$'\n'; fi
    done
  fi
  # 3. characterization dirs whose python files import the module -> select the dir
  if [[ -d "$src/$chardir" ]]; then
    for d in "$src/$chardir"/*/; do
      [[ -d "$d" ]] || continue
      if grep -Erlq --include='*.py' \
           -e "(^|[^A-Za-z0-9_.])$(_re_escape_dots "$DOTTED")([^A-Za-z0-9_]|\$)" "$d" 2>/dev/null; then
        out+="$chardir/$(basename "$d")"$'\n'
      elif grep -Erlq --include='*.py' \
           -e "^[[:space:]]*from[[:space:]]+$(_re_escape_dots "$PARENT_DOTTED")[[:space:]]+import[[:space:]]+.*\b${BASENAME}\b" "$d" 2>/dev/null; then
        out+="$chardir/$(basename "$d")"$'\n'
      fi
    done
  fi
  printf '%s' "$out" | sed '/^$/d' | sort -u
}

# emit_json — write covering-set.json via an embedded Python emitter (safe
# escaping in pure shell is error-prone; matches slice-preflight.sh). Values
# arrive via env vars; candidates/selected are newline-separated in
# CS_CANDIDATES / CS_SELECTED. Writes to $1.
emit_json() {
  local out_file="$1"
  CS_OUT_FILE="$out_file" python3 - <<'PY'
import json, os

def val(name):
    v = os.environ.get(name, "")
    return v if v != "" else None

def lst(name):
    return [ln for ln in os.environ.get(name, "").splitlines() if ln.strip()]

cov = os.environ.get("CS_COVERAGE", "")
coverage_percent = None
if cov != "":
    try:
        coverage_percent = float(cov)
        if coverage_percent == int(coverage_percent):
            coverage_percent = int(coverage_percent)
    except ValueError:
        coverage_percent = None

try:
    threshold = int(os.environ.get("CS_THRESHOLD", "80"))
except ValueError:
    threshold = 80

data = {
    "schema": "covering-set/1",
    "timestamp_utc": val("CS_TIMESTAMP"),
    "module": val("CS_MODULE"),
    "src": val("CS_SRC"),
    "container": val("CS_CONTAINER"),
    "threshold": threshold,
    "mode": val("CS_MODE"),
    "candidates": lst("CS_CANDIDATES"),
    "selected": lst("CS_SELECTED"),
    "coverage_percent": coverage_percent,
    "status": val("CS_STATUS"),
    "reason": val("CS_REASON"),
    "command": val("CS_COMMAND"),
    "generator": "projects/hipblaslt/tensilelite/Tensile/Tests/unit/mutation/covering-set-discover.sh",
}

out = os.environ["CS_OUT_FILE"]
tmp = out + ".tmp"
try:
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)
except Exception:
    try:
        os.unlink(tmp)
    except OSError:
        pass
    raise
PY
}

# Library-only mode: expose helpers for the selftest, skip main().
if [[ "${COVERING_SET_LIB_ONLY:-0}" == "1" ]]; then
  return 0 2>/dev/null || exit 0
fi

# ------------------------------------------------------------------- arg parse
MODULE=""; OUT=""; CONTAINER="tl-mut"; SRC="projects/hipblaslt/tensilelite"
THRESHOLD="80"; MODE="measured"
need_val() { [[ $# -ge 2 ]] || die "$1 requires a value"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --module)          need_val "$@"; MODULE="$2"; shift 2 ;;
    --out)             need_val "$@"; OUT="$2"; shift 2 ;;
    --container)       need_val "$@"; CONTAINER="$2"; shift 2 ;;
    --src)             need_val "$@"; SRC="$2"; shift 2 ;;
    --threshold)       need_val "$@"; THRESHOLD="$2"; shift 2 ;;
    --dry-run|--list-only) MODE="dry-run"; shift ;;
    -h|--help)         grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$MODULE" ]] || die "--module is required"
[[ "$THRESHOLD" =~ ^[0-9]+$ ]] || die "--threshold must be an integer, got '$THRESHOLD'"

# --------------------------------------------------------------- resolve root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"; fi
cd "$ROOT" || die "cannot cd to repo root: $ROOT"

[[ -d "$SRC" ]] || die "source dir missing: $SRC (cwd=$ROOT)"

derive_names "$MODULE"
[[ -n "$BASENAME" ]] || die "could not derive a module name from --module '$MODULE'"
[[ -f "$SRC/$MODULE" ]] || printf 'covering-set-discover: WARN: module file not found under src: %s/%s\n' "$SRC" "$MODULE" >&2

if [[ -z "$OUT" ]]; then
  OUT="work/mutation/covering/${BASENAME}"
fi

# --------------------------------------------------------------- discover
CANDIDATES="$(discover_candidates "$MODULE" "$SRC")"
SELECTED="$CANDIDATES"

# The scoped coverage command (paths constrained to the discovered set; never a
# bare full-suite run). Built even in dry-run for the record. --cov target is a
# filesystem PATH (the module's package dir), matching the repo convention
# (--cov=Tensile..., never a dotted module); the module's own row is read back
# from term-missing for its coverage.
COV_DIR="${MODULE%/*}"
[[ "$COV_DIR" == "$MODULE" ]] && COV_DIR="Tensile"
COV_TARGET="$COV_DIR"
CMD=""
if [[ -n "$SELECTED" ]]; then
  paths="$(printf '%s' "$SELECTED" | tr '\n' ' ')"
  paths="${paths% }"
  CMD="docker exec -w /work/$SRC $CONTAINER pytest -m unit --cov=$COV_TARGET --cov-report=term-missing $paths"
fi

# --------------------------------------------------------------- decide/measure
COVERAGE=""; STATUS="defer"; REASON=""
if [[ -z "$SELECTED" ]]; then
  STATUS="defer"
  REASON="no candidate unit tests or characterization dirs discovered for $MODULE; refusing to fall back to the full -m unit suite without an explicit user-supplied set"
elif [[ "$MODE" == "dry-run" ]]; then
  STATUS="defer"
  REASON="dry-run/list-only: candidates listed and coverage command recorded; coverage not measured, so the ${THRESHOLD}% gate is not evaluated and mutation must not proceed"
else
  command -v docker >/dev/null 2>&1 || die "docker is unavailable (binary not found); use --dry-run for discovery only"
  docker version >/dev/null 2>&1     || die "docker is unavailable (daemon not reachable); use --dry-run for discovery only"
  docker inspect --type container "$CONTAINER" >/dev/null 2>&1 \
    || die "container does not exist: $CONTAINER; use --dry-run for discovery only"
  printf 'covering-set-discover: measuring coverage of %s with %d discovered path(s)...\n' "$MODULE" "$(printf '%s\n' "$SELECTED" | wc -l)" >&2
  COV_OUT="$(eval "$CMD" 2>&1 || true)"
  printf '%s\n' "$COV_OUT" >&2
  pct="$(parse_cov_pct "$COV_OUT" "$MODULE")"
  if [[ -z "$pct" ]]; then
    STATUS="defer"; REASON="could not parse a coverage percent for $MODULE from the pytest --cov output; treating as below the ${THRESHOLD}% gate"
  else
    COVERAGE="$pct"
    if awk -v p="$pct" -v t="$THRESHOLD" 'BEGIN{exit !((p+0) >= (t+0))}'; then
      STATUS="ok"; REASON="discovered set covers $MODULE at ${pct}% (>= ${THRESHOLD}% gate)"
    else
      STATUS="defer"; REASON="discovered set covers $MODULE at only ${pct}% (< ${THRESHOLD}% gate); add covering tests before mutating"
    fi
  fi
fi

# --------------------------------------------------------------- emit artifact
TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
mkdir -p "$OUT" || die "cannot create output dir: $OUT"
OUT_FILE="$OUT/covering-set.json"

CS_TIMESTAMP="$TIMESTAMP" \
CS_MODULE="$MODULE" \
CS_SRC="$SRC" \
CS_CONTAINER="$CONTAINER" \
CS_THRESHOLD="$THRESHOLD" \
CS_MODE="$MODE" \
CS_CANDIDATES="$CANDIDATES" \
CS_SELECTED="$SELECTED" \
CS_COVERAGE="$COVERAGE" \
CS_STATUS="$STATUS" \
CS_REASON="$REASON" \
CS_COMMAND="$CMD" \
  emit_json "$OUT_FILE" || die "failed to write covering-set.json"

printf 'covering-set-discover: %s module=%s status=%s candidates=%s%s\n' \
  "$MODE" "$MODULE" "$STATUS" \
  "$([[ -n "$CANDIDATES" ]] && printf '%s' "$(printf '%s\n' "$CANDIDATES" | wc -l)" || printf '0')" \
  "$([[ -n "$COVERAGE" ]] && printf ' coverage=%s%%' "$COVERAGE" || printf '')"
printf 'covering-set-discover: wrote %s\n' "$OUT_FILE"
