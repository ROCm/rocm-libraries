#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$here"

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

python="${PYTHON:-python3}"
abi_tag=".abi3.so"
echo "=== interpreter: $("$python" -V), stable ABI: cp312 ==="

rm -rf dist
"$python" -m build --wheel --outdir dist
whl="$(ls dist/*.whl)"
[ -f "$whl" ] || fail "no wheel produced"
pass "built ${whl}"

listing="$("$python" -m zipfile -l "$whl")"
echo "$listing"

"$python" - "$whl" <<'PY'
import sys
import zipfile
from email import message_from_bytes

with zipfile.ZipFile(sys.argv[1]) as wheel:
    unexpected = [
        name for name in wheel.namelist()
        if not (name.startswith("origami/") or name.split("/", 1)[0].endswith(".dist-info"))
    ]
    metadata_name = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
    metadata = message_from_bytes(wheel.read(metadata_name))
    requirements = metadata.get_all("Requires-Dist", [])
    import os
    sdk = os.environ.get("ROCM_SDK_VERSION")
    sdk_requirements = [r for r in requirements if r.startswith("rocm[")]
    if sdk and not any(r.startswith(f"rocm[libraries]=={sdk}") for r in sdk_requirements):
        raise SystemExit("Wheel metadata lacks the exact SDK pin")
    if not sdk and sdk_requirements:
        raise SystemExit("System wheel unexpectedly depends on SDK wheels")
    wheel_metadata = wheel.read(next(n for n in wheel.namelist() if n.endswith(".dist-info/WHEEL"))).decode()
    if "Tag: cp312-abi3-" not in wheel_metadata:
        raise SystemExit("Wheel is not tagged cp312-abi3")
if unexpected:
    raise SystemExit("Unexpected wheel entries outside the package and metadata:\n" + "\n".join(unexpected))
PY
pass "all wheel entries belong to the package or distribution metadata"
pass "wheel SDK metadata and abi3 tag match the build"

so_entries="$(echo "$listing" | awk '{print $1}' | grep -E '\.so($|\.)' || true)"
n_so="$(printf '%s' "$so_entries" | grep -c . || true)"
[ "$n_so" = "1" ] || fail "expected exactly one shared object in the wheel, found ${n_so}:
${so_entries}"
[ "$so_entries" = "origami/_pyorigami${abi_tag}" ] \
    || fail "wheel ships ${so_entries}, expected origami/_pyorigami${abi_tag}"
pass "wheel ships exactly one shared object, the extension for this ABI"

echo "$listing" | grep -q "origami/__init__.py" || fail "__init__.py missing from wheel"
echo "$listing" | grep -q "origami/_rocm_init.py" || fail "_rocm_init.py missing from wheel"
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
if [ -n "${ORIGAMI_ROCM_SDK_REQUIREMENT:-}" ]; then
    "$venv/bin/python" -m pip install --quiet "$ORIGAMI_ROCM_SDK_REQUIREMENT"
fi
# Do not let loader overrides mask missing runtime setup.
unset LD_LIBRARY_PATH LD_PRELOAD
"$venv/bin/python" -c "import origami; print('imported origami', origami.__version__)" \
    || fail "import origami failed (install ROCm Python packages or configure the system runtime)"
pass "import succeeds without LD_LIBRARY_PATH or LD_PRELOAD"

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
