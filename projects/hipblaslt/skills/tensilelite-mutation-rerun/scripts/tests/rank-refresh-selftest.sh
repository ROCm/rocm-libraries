#!/usr/bin/env bash
# rank-refresh-selftest.sh - checks for rank-refresh.sh (Issue 18).
# Read-only: no mutmut, no source edits. Snapshots into a temp dir.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUT="$HERE/../rank-refresh.sh"
RANKER="$HERE/../rank-modules.py"

fail=0
ok()  { printf 'ok   - %s\n' "$1"; }
bad() { printf 'BAD  - %s\n' "$1"; fail=1; }

# syntax
bash -n "$SUT" && ok "rank-refresh.sh syntax ok" || bad "rank-refresh.sh syntax"

# the wrapper must not itself run mutmut
grep -qE 'mutmut (run|apply)' "$SUT" && bad "wrapper references mutmut run/apply" || ok "wrapper never runs mutmut"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --- first run: writes a snapshot ---
out1="$(bash "$SUT" --out-dir "$TMP" --pin testpin 2>&1)"
snaps=$(find "$TMP" -maxdepth 1 -type f -name '*-testpin.md' | wc -l)
[[ "$snaps" -eq 1 ]] && ok "first run wrote exactly one snapshot" || bad "first run snapshot count=$snaps"
printf '%s' "$out1" | grep -q 'wrote snapshot' && ok "first run reports 'wrote snapshot'" || bad "first run missing 'wrote snapshot'"
SNAP="$(find "$TMP" -maxdepth 1 -type f -name '*-testpin.md' | head -1)"

# snapshot preserves the Issue-17 formula/weights (it IS rank-modules.py output)
for tok in '0.40' '0.25' '0.15' '0.20'; do
  grep -qF "$tok" "$SNAP" && ok "snapshot preserves weight $tok" || bad "snapshot missing weight $tok"
done
grep -qiE 'no .?log10|NO `log10`' "$SNAP" && ok "snapshot states no log10" || bad "snapshot missing no-log10"
grep -qiE 'no subtraction|subtraction of' "$SNAP" && ok "snapshot states no subtraction" || bad "snapshot missing no-subtraction"

# --- second run, same pin+day: deterministic, no change, still one snapshot ---
out2="$(bash "$SUT" --out-dir "$TMP" --pin testpin 2>&1)"
printf '%s' "$out2" | grep -q 'no change' && ok "second run is deterministic (reports 'no change')" || bad "second run not deterministic: $out2"
snaps2=$(find "$TMP" -maxdepth 1 -type f -name '*-testpin.md' | wc -l)
[[ "$snaps2" -eq 1 ]] && ok "re-run did not multiply snapshots (still 1)" || bad "re-run snapshot count=$snaps2"

# --- a non-snapshot README.md in the out-dir is NOT treated as a prior snapshot ---
printf '# not a snapshot\n' > "$TMP/README.md"
out_r="$(bash "$SUT" --out-dir "$TMP" --pin testpin 2>&1)"
printf '%s' "$out_r" | grep -q 'README.md' && bad "README.md wrongly treated as a snapshot: $out_r" || ok "README.md is not treated as a prior snapshot"

# --- the DIFF-vs-previous branch is actually exercised ---
# Plant a DIFFERING snapshot with a future date so it sorts LAST (selected as the
# previous) relative to a current snapshot written for today; assert the DIFF branch.
D2="$(mktemp -d)"
printf 'COMPLETELY DIFFERENT SNAPSHOT CONTENT\n' > "$D2/20991231-futurepin.md"
out3="$(bash "$SUT" --out-dir "$D2" --pin testpin 2>&1)"
printf '%s' "$out3" | grep -q 'DIFF vs previous snapshot' && ok "DIFF-vs-previous branch reports a real diff" || bad "DIFF-vs-previous branch not exercised: $out3"
printf '%s' "$out3" | grep -q '^[-+]' && ok "unified diff body is emitted on a real diff" || bad "no unified diff body emitted"
rm -rf "$D2"

# --- README tokens ---
RME="$HERE/fixtures/ranking-history-README.md"
for tok in refresh ranking history pin CI ratchet allowlist SLICE_FLOOR '0.40' '0.25' '0.15' '0.20'; do
  grep -qiF "$tok" "$RME" && ok "README mentions $tok" || bad "README missing $tok"
done
grep -qiE 'no log10' "$RME" && ok "README states no log10" || bad "README missing no-log10"
grep -qiE 'no subtraction' "$RME" && ok "README states no subtraction" || bad "README missing no-subtraction"

echo
if [[ "$fail" -eq 0 ]]; then echo "ALL SELFTESTS PASSED"; exit 0; else echo "SELFTESTS FAILED"; exit 1; fi
