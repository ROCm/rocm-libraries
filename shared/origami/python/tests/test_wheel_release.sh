#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$here"

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

python="${PYTHON:-python3}"
abi_tag="$("$python" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)"
echo "=== interpreter: $("$python" -V), ext suffix: ${abi_tag} ==="

build_args=()
if [ "${ORIGAMI_BUILD_FROM_SOURCE:-OFF}" = "ON" ]; then
    build_args+=(-C cmake.define.ORIGAMI_BUILD_FROM_SOURCE=ON)
fi

rm -rf dist
"$python" -m build --wheel --outdir dist ${build_args[@]+"${build_args[@]}"}
whl="$(ls dist/*.whl)"
[ -f "$whl" ] || fail "no wheel produced"
pass "built ${whl}"

listing="$("$python" -m zipfile -l "$whl")"
echo "$listing"

so_entries="$(echo "$listing" | awk '{print $1}' | grep -E '\.so($|\.)' || true)"
n_so="$(printf '%s' "$so_entries" | grep -c . || true)"
[ "$n_so" = "1" ] || fail "expected exactly one shared object in the wheel, found ${n_so}:
${so_entries}"
[ "$so_entries" = "origami/_pyorigami${abi_tag}" ] \
    || fail "wheel ships ${so_entries}, expected origami/_pyorigami${abi_tag}"
pass "wheel ships exactly one shared object, the extension for this ABI"

echo "$listing" | grep -q "origami/__init__.py" || fail "__init__.py missing from wheel"
echo "$listing" | grep -q "origami/selector.py" || fail "selector.py missing from wheel"
echo "$listing" | grep -q "origami/origami.py" || fail "origami.py alias missing from wheel"
pass "pure Python files recorded in wheel"

if echo "$listing" | grep -Eq "liborigami\.so"; then
    fail "liborigami is bundled in the wheel; it must be linked dynamically, not shipped"
fi
pass "no liborigami bundled"

if echo "$listing" | grep -q "bindings.cpp"; then
    fail "C++ source bindings.cpp leaked into the wheel"
fi
pass "no C++ source in wheel"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
"$python" -m zipfile -e "$whl" "$workdir"
ext_path="$(find "$workdir" -name "_pyorigami${abi_tag}")"
[ -f "$ext_path" ] || fail "could not extract extension from wheel"
readelf_bin=""
if command -v readelf >/dev/null 2>&1; then
    readelf_bin=readelf
elif command -v llvm-readelf >/dev/null 2>&1; then
    readelf_bin=llvm-readelf
else
    fail "neither readelf nor llvm-readelf found; cannot verify NEEDED liborigami.so.1"
fi
"$readelf_bin" -d "$ext_path" | grep -q "NEEDED.*liborigami.so.1" \
    || fail "extension does not record NEEDED liborigami.so.1"
pass "extension NEEDED liborigami.so.1"
if "$readelf_bin" -d "$ext_path" | grep -Eq "NEEDED.*liborigami\.so[^.]"; then
    fail "extension NEEDED an unversioned liborigami.so"
fi

if command -v auditwheel >/dev/null 2>&1; then
    auditwheel show "$whl" || true
fi

venv="$workdir/venv"
"$python" -m venv "$venv"
"$venv/bin/python" -m pip install --quiet --force-reinstall --no-deps "$whl"
ld_parts=()

if [ -n "${ORIGAMI_RUNTIME_LIB_DIR:-}" ]; then
    ld_parts+=("$ORIGAMI_RUNTIME_LIB_DIR")
fi

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    ld_parts+=("$LD_LIBRARY_PATH")
fi

if [ ${#ld_parts[@]} -gt 0 ]; then
    export LD_LIBRARY_PATH="$(IFS=:; echo "${ld_parts[*]}")"
fi
"$venv/bin/python" -c "import origami; print('imported origami', origami.__version__)" \
    || fail "import origami failed (is liborigami.so.1 on the loader path?)"
pass "import succeeds with runtime on loader path"

"$venv/bin/python" -W error::DeprecationWarning -c "import origami" \
    || fail "importing the origami package emits a DeprecationWarning"
pass "package import is warning-free"

"$venv/bin/python" -c "
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    import origami.origami
assert any(issubclass(w.category, DeprecationWarning) for w in caught), caught
assert origami.origami.problem_t is origami.problem_t
" || fail "origami.origami alias missing or not deprecated"
pass "origami.origami alias re-exports the extension and warns"

echo "=== all wheel-release assertions passed ==="
