#!/usr/bin/env bash
# covering-set-discover-selftest.sh — checks for covering-set-discover.sh.
#   - LIB_ONLY unit checks of derive_names (pure, no fs).
#   - integration --dry-run checks against the real Tensile test tree proving the
#     DoD candidate-discovery bullets, JSON validity + required fields, the 80%
#     gate/defer semantics, and the no-full-suite-fallback guarantee.
# It never runs docker/pytest/mutmut (dry-run only) and never edits source.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUT="$HERE/../covering-set-discover.sh"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)"
[[ -n "$ROOT" ]] || ROOT="$(cd "$HERE/../../../.." && pwd)"

fail=0
ok()  { printf 'ok   - %s\n' "$1"; }
bad() { printf 'BAD  - %s\n' "$1"; fail=1; }

# ---------------------------------------------------------- pure: derive_names
# shellcheck disable=SC1090
COVERING_SET_LIB_ONLY=1 source "$SUT"

check_names() {
  local m="$1" wb="$2" wd="$3" wp="$4"
  derive_names "$m"
  [[ "$BASENAME" == "$wb" && "$DOTTED" == "$wd" && "$PARENT_DOTTED" == "$wp" ]] \
    && ok "derive_names $m -> $BASENAME / $DOTTED / $PARENT_DOTTED" \
    || bad "derive_names $m -> $BASENAME / $DOTTED / $PARENT_DOTTED (want $wb / $wd / $wp)"
}
check_names "Tensile/BenchmarkSplitter.py" "BenchmarkSplitter" "Tensile.BenchmarkSplitter" "Tensile"
check_names "Tensile/SolutionStructs/Solution.py" "Solution" "Tensile.SolutionStructs.Solution" "Tensile.SolutionStructs"
check_names "Tensile/Common/Utilities.py" "Utilities" "Tensile.Common.Utilities" "Tensile.Common"

# ------------------------------------- pure: parse_cov_pct (precision=2 aware)
COV_SAMPLE="Name                              Stmts   Miss  Cover   Missing
-----------------------------------------------------------------------------
Tensile/Widget.py                    10      2  82.50%   5-6
Tensile/Other.py                      4      4   0.00%   1-4
-----------------------------------------------------------------------------
TOTAL                                14      6  57.14%"
p="$(parse_cov_pct "$COV_SAMPLE" "Tensile/Widget.py")"
[[ "$p" == "82.50" ]] && ok "parse_cov_pct reads module row NN.DD% (82.50, not fractional 50)" || bad "parse_cov_pct module row -> '$p' (want 82.50)"
p="$(parse_cov_pct "$COV_SAMPLE" "Tensile/DoesNotExist.py")"
[[ "$p" == "57.14" ]] && ok "parse_cov_pct falls back to TOTAL when no module row" || bad "parse_cov_pct TOTAL fallback -> '$p' (want 57.14)"
# gate arithmetic must be float-correct (the exact false-green the parse bug caused)
awk -v p=82.50 -v t=80 'BEGIN{exit !((p+0)>=(t+0))}' && ok "gate: 82.50% >= 80 -> ok" || bad "gate 82.50 vs 80 wrong"
awk -v p=12.99 -v t=80 'BEGIN{exit !((p+0)>=(t+0))}' && bad "gate: 12.99% must NOT clear 80 (false-green!)" || ok "gate: 12.99% < 80 -> defer (no false-green)"
awk -v p=79.99 -v t=80 'BEGIN{exit !((p+0)>=(t+0))}' && bad "gate: 79.99% must NOT clear 80" || ok "gate: 79.99% < 80 -> defer"

# --------------------------------- pure: discover_candidates branch isolation
# Fixture src where each discovery branch is the ONLY way to find its test, so a
# regression in any single branch fails a specific assertion.
FX="$HERE/fixtures/covering-set/src"
mapfile -t CANDS < <(discover_candidates "Tensile/Widget.py" "$FX")
has_c() { local x; for x in "${CANDS[@]}"; do [[ "$x" == "$1" ]] && return 0; done; return 1; }
has_c "Tensile/Tests/unit/test_Widget.py"                              && ok "fixture: direct-file branch (test_Widget.py, no import)" || bad "fixture: direct-file branch missed"
has_c "Tensile/Tests/unit/test_other.py"                              && ok "fixture: parent-import branch (from Tensile import Widget)" || bad "fixture: parent-import branch missed"
has_c "Tensile/Tests/unit/characterization/WidgetChar"               && ok "fixture: char-dir dotted branch (import Tensile.Widget)" || bad "fixture: char-dir branch missed"
has_c "Tensile/Tests/unit/characterization/test_toplevel_char.py"    && ok "fixture: top-level char-file branch (Finding 3 fix)" || bad "fixture: top-level char-file branch missed"
has_c "Tensile/Tests/unit/test_unrelated.py"                         && bad "fixture: FALSE POSITIVE test_unrelated.py included" || ok "fixture: unrelated test correctly excluded"
[[ "${#CANDS[@]}" -eq 4 ]] && ok "fixture: exactly 4 candidates (no extras)" || bad "fixture: candidate count ${#CANDS[@]} (want 4)"

# ---------------------------------------------------------- integration: dry-run
cd "$ROOT" || { bad "cannot cd to root $ROOT"; }
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# JSON field extractor (python3, no jq dependency)
jget() { python3 - "$1" "$2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
v = d
for k in sys.argv[2].split('.'):
    v = v[k]
if isinstance(v, (list, dict)):
    print(json.dumps(v))
elif v is None:
    print("__NULL__")
else:
    print(v)
PY
}
jhas() { python3 - "$1" "$2" "$3" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("yes" if sys.argv[3] in d.get(sys.argv[2], []) else "no")
PY
}

# --- BenchmarkSplitter: DoD bullet - includes test_BenchmarkSplitter.py ---
bash "$SUT" --module Tensile/BenchmarkSplitter.py --out "$tmp/bench" --dry-run >/dev/null 2>&1 \
  && ok "dry-run BenchmarkSplitter rc0" || bad "dry-run BenchmarkSplitter rc!=0"
BJ="$tmp/bench/covering-set.json"
python3 -c "import json;json.load(open('$BJ'))" 2>/dev/null && ok "BenchmarkSplitter covering-set.json is valid JSON" || bad "BenchmarkSplitter JSON invalid"
[[ "$(jhas "$BJ" candidates 'Tensile/Tests/unit/test_BenchmarkSplitter.py')" == "yes" ]] \
  && ok "candidates include test_BenchmarkSplitter.py (DoD)" || bad "missing test_BenchmarkSplitter.py"
[[ "$(jget "$BJ" mode)" == "dry-run" ]] && ok "mode is dry-run" || bad "mode not dry-run"
[[ "$(jget "$BJ" status)" == "defer" ]] && ok "dry-run status is defer (gate not evaluated)" || bad "dry-run status not defer"
[[ "$(jget "$BJ" coverage_percent)" == "__NULL__" ]] && ok "dry-run coverage_percent is null" || bad "dry-run coverage not null"
[[ "$(jget "$BJ" threshold)" == "80" ]] && ok "threshold defaults to 80" || bad "threshold not 80"
[[ "$(jget "$BJ" command)" == docker*pytest\ -m\ unit\ --cov=Tensile\ --cov-report=term-missing*test_BenchmarkSplitter.py* ]] \
  && ok "command records the plan-shape coverage command (--cov=Tensile path + scoped paths)" || bad "command shape wrong: $(jget "$BJ" command)"
# required fields all present
for k in schema module src container threshold mode candidates selected coverage_percent status reason command; do
  python3 -c "import json,sys; d=json.load(open('$BJ')); sys.exit(0 if '$k' in d else 1)" \
    && : || bad "covering-set.json missing field: $k"
done
ok "covering-set.json contains all required fields"

# --- Solution.py: DoD bullet - includes SolutionClass / SolutionDerivationSweep ---
bash "$SUT" --module Tensile/SolutionStructs/Solution.py --out "$tmp/sol" --dry-run >/dev/null 2>&1 \
  && ok "dry-run Solution rc0" || bad "dry-run Solution rc!=0"
SJ="$tmp/sol/covering-set.json"
[[ "$(jhas "$SJ" candidates 'Tensile/Tests/unit/characterization/SolutionClass')" == "yes" ]] \
  && ok "candidates include characterization/SolutionClass (DoD)" || bad "missing SolutionClass"
[[ "$(jhas "$SJ" candidates 'Tensile/Tests/unit/characterization/SolutionDerivationSweep')" == "yes" ]] \
  && ok "candidates include characterization/SolutionDerivationSweep (DoD)" || bad "missing SolutionDerivationSweep"

# --- no-full-suite-fallback: unknown module -> defer, no command, no bare -m unit ---
bash "$SUT" --module Tensile/DoesNotExist/Nope.py --out "$tmp/none" --dry-run >/dev/null 2>&1 \
  && ok "dry-run unknown-module rc0" || bad "dry-run unknown-module rc!=0"
NJ="$tmp/none/covering-set.json"
[[ "$(jget "$NJ" candidates)" == "[]" ]] && ok "unknown module -> zero candidates" || bad "unknown module has candidates"
[[ "$(jget "$NJ" status)" == "defer" ]] && ok "unknown module -> status defer" || bad "unknown module not defer"
[[ "$(jget "$NJ" command)" == "__NULL__" ]] && ok "unknown module -> no coverage command (no full-suite fallback)" || bad "unknown module emitted a command"
case "$(jget "$NJ" reason)" in *"refusing to fall back"*) ok "reason states refusal to fall back to full suite" ;; *) bad "reason does not state fallback refusal" ;; esac

# --- threshold override is honored in the artifact ---
bash "$SUT" --module Tensile/BenchmarkSplitter.py --out "$tmp/thr" --dry-run --threshold 95 >/dev/null 2>&1
[[ "$(jget "$tmp/thr/covering-set.json" threshold)" == "95" ]] && ok "--threshold override recorded" || bad "--threshold not honored"

echo
if [[ "$fail" -eq 0 ]]; then echo "ALL SELFTESTS PASSED"; exit 0; else echo "SELFTESTS FAILED"; exit 1; fi
