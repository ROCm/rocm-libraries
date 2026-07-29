#!/usr/bin/env bash
# pyproject-mutmut-selftest.sh — git/docker-free checks of pyproject-mutmut.sh:
# the [tool.mutmut] rewrite boundaries, preservation of unrelated content,
# idempotence, and backup/restore byte-exactness. Operates on a throwaway
# fixture; never touches the real tracked pyproject.toml.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUT="$HERE/../pyproject-mutmut.sh"

fail=0
ok()  { printf 'ok   - %s\n' "$1"; }
bad() { printf 'BAD  - %s\n' "$1"; fail=1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/src"
FIX="$tmp/src/pyproject.toml"
BAK="$tmp/pyproject.toml.bak"

cat > "$FIX" <<'EOF'
[build-system]
requires = ["setuptools"]

[tool.mutmut]
source_paths = ["Tensile"]
only_mutate = [
    "Tensile/Common/Utilities.py",
    "Tensile/TensileLogic/ValidChipId.py",
]
do_not_mutate = [
    "Tensile/Tests/*",
]
pytest_add_cli_args_test_selection = [
    "Tensile/Tests/unit/characterization/CommonUtilities",
]
also_copy = ["pytest.ini"]
mutate_only_covered_lines = false
# trailing comment preserved

[tool.other]
key = "value"
EOF

orig_sha="$(sha256sum "$FIX" | awk '{print $1}')"

# --- backup then set ---
bash "$SUT" backup --src "$tmp/src" --backup "$BAK" >/dev/null 2>&1 || bad "backup rc"
[[ -f "$BAK" ]] && ok "backup file created" || bad "backup missing"

bash "$SUT" set --src "$tmp/src" --backup "$BAK" \
  --only-mutate Tensile/LibraryIO.py \
  --test-selection Tensile/Tests/unit/characterization/LibraryIO >/dev/null 2>&1 \
  || bad "set rc"

# assert target arrays rewritten
grep -q '"Tensile/LibraryIO.py"' "$FIX" && ok "only_mutate rewritten" || bad "only_mutate not set"
grep -q '"Tensile/Tests/unit/characterization/LibraryIO"' "$FIX" && ok "test-selection rewritten" || bad "test-selection not set"
# old values gone
grep -q 'ValidChipId' "$FIX" && bad "stale only_mutate value survived" || ok "stale only_mutate removed"
grep -q 'CommonUtilities' "$FIX" && bad "stale test-selection value survived" || ok "stale test-selection removed"
# unrelated content preserved
grep -q 'source_paths = \["Tensile"\]' "$FIX" && ok "source_paths preserved" || bad "source_paths lost"
grep -q 'mutate_only_covered_lines = false' "$FIX" && ok "covered_lines=false preserved" || bad "covered_lines lost"
grep -q '# trailing comment preserved' "$FIX" && ok "comment preserved" || bad "comment lost"
grep -q '\[tool.other\]' "$FIX" && ok "other table preserved" || bad "other table lost"
grep -q 'do_not_mutate' "$FIX" && ok "do_not_mutate preserved" || bad "do_not_mutate lost"

# --- idempotence: same set twice == identical bytes ---
after1="$(sha256sum "$FIX" | awk '{print $1}')"
bash "$SUT" set --src "$tmp/src" --backup "$BAK" \
  --only-mutate Tensile/LibraryIO.py \
  --test-selection Tensile/Tests/unit/characterization/LibraryIO >/dev/null 2>&1 || bad "set(2) rc"
after2="$(sha256sum "$FIX" | awk '{print $1}')"
[[ "$after1" == "$after2" ]] && ok "set is idempotent" || bad "set not idempotent ($after1 != $after2)"

# --- comma-separated + repeated flags both parse ---
bash "$SUT" set --src "$tmp/src" --backup "$BAK" \
  --only-mutate "Tensile/A.py,Tensile/B.py" --only-mutate Tensile/C.py \
  --test-selection sel1 >/dev/null 2>&1 || bad "set(csv) rc"
for want in '"Tensile/A.py"' '"Tensile/B.py"' '"Tensile/C.py"'; do
  grep -qF "$want" "$FIX" && ok "csv/repeat entry $want" || bad "missing $want"
done

# --- restore is byte-exact to original ---
bash "$SUT" restore --src "$tmp/src" --backup "$BAK" >/dev/null 2>&1 || bad "restore rc"
restored_sha="$(sha256sum "$FIX" | awk '{print $1}')"
[[ "$restored_sha" == "$orig_sha" ]] && ok "restore byte-exact" || bad "restore not byte-exact"

echo
if [[ "$fail" -eq 0 ]]; then echo "ALL SELFTESTS PASSED"; exit 0; else echo "SELFTESTS FAILED"; exit 1; fi
