#!/usr/bin/env bash
# preflight-selftest.sh — docker-free unit checks for slice-preflight.sh pure
# helpers (derive_slug, emit_json). Does NOT touch docker/git/mutmut.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUT="$HERE/../slice-preflight.sh"

fail=0
ok()   { printf 'ok   - %s\n' "$1"; }
bad()  { printf 'BAD  - %s\n' "$1"; fail=1; }

# shellcheck disable=SC1090
MUTMUT_PREFLIGHT_LIB_ONLY=1 source "$SUT"

# --- derive_slug ---
check_slug() {
  local in="$1" want="$2" got
  got="$(derive_slug "$in")"
  if [[ "$got" == "$want" ]]; then ok "slug '$in' -> '$got'"; else bad "slug '$in' -> '$got' (want '$want')"; fi
}
check_slug "Tensile/LibraryIO.py" "libraryio"
check_slug "CommonUtilities" "commonutilities"
check_slug "Tensile/Common/Utilities.py" "utilities"
check_slug "TensileLogic" "tensilelogic"
check_slug "foo_bar.py" "foo-bar"

# --- emit_json: clean case, valid JSON with expected fields ---
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
PF_TIMESTAMP="2026-07-20T00:00:00Z" PF_SLICE="2" PF_MODULE="Tensile/LibraryIO.py" \
PF_SLUG="libraryio" PF_SRC="projects/hipblaslt/tensilelite" PF_SHA="deadbeef" \
PF_BRANCH="develop" PF_TRACKED_CLEAN="true" PF_DIRTY_FILES="" \
PF_CONTAINER="tl-mut" PF_CONTAINER_EXISTS="true" PF_CONTAINER_STATUS="exited" \
PF_IMAGE="tensilelite-char:repro" PF_IMAGE_ID="sha256:abc" PF_IMAGE_DIGEST="" \
PF_MUTMUT_VERSION="" \
  emit_json "$tmp/env.json"

python3 - "$tmp/env.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["tracked_source_clean"] is True, d
assert d["dirty_tracked_files"] == [], d
assert d["mutmut_version"] is None, d          # empty -> null
assert d["container_image_digest"] is None, d  # empty -> null
assert d["container_exists"] is True, d
assert d["slice"] == "2" and d["slug"] == "libraryio", d
assert d["schema"] == "slice-preflight/1", d
print("json-clean-case OK")
PY
[[ $? -eq 0 ]] && ok "emit_json clean case valid" || bad "emit_json clean case"

# --- emit_json: dirty list preserved, spaces in names survive escaping ---
PF_TIMESTAMP="t" PF_SLICE="9" PF_MODULE="m" PF_SLUG="s" PF_SRC="x" PF_SHA="h" \
PF_BRANCH="b" PF_TRACKED_CLEAN="false" \
PF_DIRTY_FILES=$'Tensile/A.py\nTensile/with space.py' \
PF_CONTAINER="c" PF_CONTAINER_EXISTS="true" PF_CONTAINER_STATUS="running" \
PF_IMAGE="i" PF_IMAGE_ID="id" PF_IMAGE_DIGEST="i@sha256:z" PF_MUTMUT_VERSION="3.6.0" \
  emit_json "$tmp/dirty.json"

python3 - "$tmp/dirty.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["tracked_source_clean"] is False, d
assert d["dirty_tracked_files"] == ["Tensile/A.py", "Tensile/with space.py"], d
assert d["mutmut_version"] == "3.6.0", d
assert d["container_image_digest"] == "i@sha256:z", d
print("json-dirty-case OK")
PY
[[ $? -eq 0 ]] && ok "emit_json dirty list + escaping" || bad "emit_json dirty case"

echo
if [[ "$fail" -eq 0 ]]; then echo "ALL SELFTESTS PASSED"; exit 0; else echo "SELFTESTS FAILED"; exit 1; fi
