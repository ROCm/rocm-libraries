"""The megakernel family's lock: only code objects the committed lock names ship.

`tools/hkp_megakernel_lock.py verify` runs inside the pack step of every root that has a
`megakernel/manifest.json` and the family enabled. These cases drive the tool directly,
then through a real sub-build to show a mismatch fails the build before anything packs.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cmake_harness import PKG, SuppliedPython, consumer_preamble

pytestmark = pytest.mark.quick

TOOL = PKG / "tools" / "hkp_megakernel_lock.py"
MODEL = "qwen38-27b-mxfp4-gfx942"


def _entry(data, m, src):
    return {
        "model": MODEL,
        "src": src,
        "arch": "gfx942",
        "cus": 304,
        "kv": "ROCM_ATTN",
        "block": 784,
        "M": m,
        "tp": 1,
        "rt": 0,
        "lm_head": False,
        "symbol": f"qwen38_decode_m{m}",
        "block_threads": 512,
        "lds_bytes": 0,
        "kernarg_bytes": 20,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "file": f"{MODEL}_M{m}_{src[:16]}.co",
        "flydsl_version": "0.4.0",
        "vllm_commit": "0" * 40,
        "megakernels_commit": "0" * 40,
    }


def _bundle(folder):
    """A megakernel bundle of two objects, its KDP generated: the state a `dvc pull`
    of a correct upload leaves."""
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True)
    entries = []
    for m, src in ((1, "a" * 32), (8, "b" * 32)):
        data = bytes([m]) * 64
        entry = _entry(data, m, src)
        (artifacts / entry["file"]).write_bytes(data)
        entries.append(entry)
    lock = folder / "manifest.json"
    lock.write_text(json.dumps(entries), encoding="utf-8")
    _tool("generate", lock)
    return lock, entries


def _tool(command, lock, success=True):
    proc = subprocess.run(
        [sys.executable, str(TOOL), command, "--lock", str(lock)],
        capture_output=True,
        text=True,
    )
    assert (proc.returncode == 0) == success, proc.stdout + proc.stderr
    return proc.stderr


def test_a_bundle_whose_objects_match_the_lock_verifies(tmp_path):
    lock, _ = _bundle(tmp_path)
    assert _tool("verify", lock) == ""


def test_an_object_whose_bytes_differ_from_the_lock_fails_naming_it(tmp_path):
    lock, entries = _bundle(tmp_path)
    corrupt = tmp_path / "artifacts" / entries[1]["file"]
    data = bytearray(corrupt.read_bytes())
    data[10] ^= 0xFF  # same size, different bytes
    corrupt.write_bytes(bytes(data))

    diagnostic = _tool("verify", lock, success=False)
    assert entries[1]["file"] in diagnostic
    assert f"lock sha256 {entries[1]['sha256']}" in diagnostic
    assert entries[0]["file"] not in diagnostic


def test_an_object_the_lock_does_not_name_fails(tmp_path):
    lock, _ = _bundle(tmp_path)
    (tmp_path / "artifacts" / "stray.co").write_bytes(b"\0")
    assert "stray.co: not in the lock" in _tool("verify", lock, success=False)


def test_a_missing_object_fails(tmp_path):
    lock, entries = _bundle(tmp_path)
    (tmp_path / "artifacts" / entries[0]["file"]).unlink()
    assert "missing" in _tool("verify", lock, success=False)


def test_a_kdp_that_is_not_the_one_the_lock_generates_fails(tmp_path):
    lock, entries = _bundle(tmp_path)
    kdp = tmp_path / "artifacts" / "megakernel.kdp.json"
    doc = json.loads(kdp.read_text(encoding="utf-8"))
    doc["kernelDescriptors"][0]["metadata"]["block_threads"] = 256
    kdp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    assert "megakernel.kdp.json: not the KDP" in _tool("verify", lock, success=False)


def test_a_bundle_not_pulled_from_dvc_passes(tmp_path):
    """Configure warns about a missing DVC output; the pack has nothing to check."""
    lock, _ = _bundle(tmp_path)
    for path in (tmp_path / "artifacts").iterdir():
        path.unlink()
    (tmp_path / "artifacts").rmdir()
    assert _tool("verify", lock) == ""


@pytest.mark.parametrize(
    ("field", "value"),
    [("kernarg_bytes", 24), ("cus", True), ("file", "renamed.co")],
    ids=["other-abi", "bool-for-int", "file-not-named-by-key"],
)
def test_a_malformed_lock_fails(tmp_path, field, value):
    lock, entries = _bundle(tmp_path)
    entries[0][field] = value
    lock.write_text(json.dumps(entries), encoding="utf-8")
    assert field in _tool("verify", lock, success=False)


# Stands in for hkp_pack.py: records that the pack step reached it.
PACKER = """
import sys
from pathlib import Path
argv = sys.argv[1:]
out_root = Path(argv[argv.index('--out-root') + 1])
out_root.mkdir(parents=True, exist_ok=True)
(out_root / 'packed').write_text('yes', encoding='utf-8')
"""


@pytest.mark.parametrize("family", ["ON", "OFF"])
def test_the_pack_step_fails_on_a_lock_mismatch_only_with_the_family_enabled(
    tmp_path, cmake, cmake_make_program, family
):
    """With the family on, a corrupt object fails the build before the packer runs; with
    it off, the folder is excluded and the pack proceeds."""
    python = SuppliedPython(tmp_path, pip=False)
    source = tmp_path / "source"
    authored = source / "authored"
    _, entries = _bundle(authored / "megakernel")
    corrupt = authored / "megakernel" / "artifacts" / entries[0]["file"]
    corrupt.write_bytes(b"\xff" * 64)
    (source / "packer.py").write_text(PACKER, encoding="utf-8")
    (source / "kpack" / "rocm_kpack").mkdir(parents=True)
    exclude = "" if family == "ON" else "megakernel"
    (source / "CMakeLists.txt").write_text(
        consumer_preamble("HkpMegakernelLock")
        + f"""set(HKP_TOOL "${{CMAKE_CURRENT_SOURCE_DIR}}/packer.py")
hkp_wire_pack_target(
    NAME lock
    SOURCE_ROOT "${{CMAKE_CURRENT_SOURCE_DIR}}/authored"
    OUT_ROOT "${{CMAKE_CURRENT_BINARY_DIR}}/lock"
    ARCHES gfx942 HIPCC unused
    ROCM_KPACK_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/kpack"
    ENABLE_ROCKE OFF
    EXCLUDE_FOLDERS {exclude})
""",
        encoding="utf-8",
    )
    build = tmp_path / "build"
    python.run(
        cmake,
        "-S",
        source,
        "-B",
        build,
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={cmake_make_program}",
        f"-DPython3_EXECUTABLE={python.python}",
    )
    proc = python.run(
        cmake, "--build", build, "--target", "hkp_packaging_lock", success=family == "OFF"
    )
    packed = (build / "lock" / "packed").exists()
    if family == "ON":
        assert entries[0]["file"] in proc.stdout + proc.stderr
        assert not packed
    else:
        assert packed
