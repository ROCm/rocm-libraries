import os
import shutil
import struct
import sys
import types
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _TESTS_DIR.parent / "python"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# rocm_kpack location: CMake passes HIPKERNELPROVIDER_ROCM_KPACK_DIR; otherwise
# rely on an installed rocm_kpack already importable. No skip on absence — the
# compiler and kpack are load-bearing; a missing dependency is a hard failure.
_KPACK_DIR = os.environ.get("HIPKERNELPROVIDER_ROCM_KPACK_DIR")
if _KPACK_DIR and Path(_KPACK_DIR).is_dir() and _KPACK_DIR not in sys.path:
    sys.path.insert(0, _KPACK_DIR)

# Make the in-tree rocke platform + kernels library importable for the rocke
# producer tests, mirroring how this conftest already wires hkp_pack and
# rocm_kpack onto sys.path.
#
# Source tree, not the packs' wheel venv: this tests producer logic, and the
# packs already cover the wheel path.
_ROCKE_ROOT = _TESTS_DIR.parent.parent / "rocke"
for _rocke_sub in ("platform/python", "library"):
    _p = _ROCKE_ROOT / _rocke_sub
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Imported below the sys.path wiring above, never beside the other imports: msgpack
# ships as a rocm_kpack dependency, so on a host without it installed system-wide it
# becomes importable only once _KPACK_DIR is on the path. Importing at the top of the
# file would cache that miss and fail every synthesised-object test on exactly the
# environments HIPKERNELPROVIDER_ROCM_KPACK_DIR exists to serve.
try:
    import msgpack
except ImportError:
    msgpack = None

_ROCKE_UKD_SOURCE = "kernels/gfx950/attention_dense.py"
_ROCKE_UKD_BUILDER = "build_attention_dense"
_ROCKE_UKD_SPEC = {
    "batch": 1,
    "seqlen_q": 256,
    "seqlen_kv": 256,
    "num_query_heads": 8,
    "num_kv_heads": 8,
    "head_size": 128,
}
_ROCKE_UNAVAILABLE_HINT = (
    "provision the rocke platform and libamd_comgr; both are ingestor requirements"
)


@pytest.fixture(scope="session")
def rocm_kpack_dir():
    return _KPACK_DIR if _KPACK_DIR else None


@pytest.fixture(scope="session")
def hipcc():
    """The hipcc driver used for real --genco compilation.

    Resolved from HKP_HIPCC (set by CMake) or PATH; a missing hipcc is a hard
    failure. A real hipcc compile error is not caught here — it surfaces from
    the pipeline as a failure.
    """
    exe = os.environ.get("HKP_HIPCC")
    if not exe:
        for name in ("hipcc", "hipcc.bat"):
            found = shutil.which(name)
            if found:
                exe = found
                break
    if not exe:
        pytest.fail("hipcc not found (set HKP_HIPCC or put hipcc on PATH)")
    return exe


def _probe_rocke():
    """Attempt to import rocke/kernels and load comgr; return (ok, reason).

    Distinguishes a load failure (rocke/kernels not importable, or libamd_comgr
    not dlopen-able) from a compile failure: only the load path is probed here,
    so a ComgrError from a real compile in a test propagates rather than being
    swallowed as unavailability.
    """
    try:
        import rocke  # noqa: F401
        import kernels  # noqa: F401
        from rocke.runtime import comgr
    except Exception as exc:
        return False, f"rocke/kernels not importable: {exc}"
    try:
        comgr._resolve_lib()
    except Exception as exc:
        return False, f"libamd_comgr not loadable: {exc}"
    return True, ""


@pytest.fixture(scope="session")
def rocke_importable():
    """Session gate for rocke tests that need the CORPUS but not comgr.

    Signature and predicate guards introspect real builders without lowering
    anything, so gating them on comgr would needlessly skip them on a box that
    has rocke but no working comgr. Kept separate from rocke_available for that
    reason.

    Fails rather than skips: rocke is a requirement of the ingestor, asserted at
    configure time, so an unimportable one here is a broken build rather than an
    unprovisioned machine.
    """
    try:
        import kernels  # noqa: F401
        import rocke  # noqa: F401
    except Exception as exc:
        pytest.fail(f"rocke/kernels not importable: {exc}")
    return True


@pytest.fixture(scope="session")
def rocke_available():
    """Session gate for the comgr-dependent rocke tests.

    Returns True when rocke/kernels import and comgr loads, and fails otherwise.
    comgr ships with ROCm and configure refuses to proceed without it, so this
    tier cannot be legitimately unavailable -- skipping instead would let a ROCm
    bump that moved or dropped comgr turn the tier green by not running it. A
    real ComgrError from a compile is not gated here.
    """
    ok, reason = _probe_rocke()
    if not ok:
        pytest.fail(f"{reason} ({_ROCKE_UNAVAILABLE_HINT})")
    return True


@pytest.fixture(scope="session")
def fixtures_dir():
    return _TESTS_DIR / "fixtures"


@pytest.fixture(scope="session")
def main_fixture(fixtures_dir):
    return fixtures_dir / "main"


@pytest.fixture(scope="session")
def empty_arch_fixture(fixtures_dir):
    return fixtures_dir / "empty_arch"


@pytest.fixture(scope="session")
def rocke_fixture(fixtures_dir):
    return fixtures_dir / "rocke"


@pytest.fixture(scope="session")
def rocke_ukd():
    """The reference rocke UKD source/builder/spec shared by the producer tests."""
    return types.SimpleNamespace(
        source=_ROCKE_UKD_SOURCE,
        builder=_ROCKE_UKD_BUILDER,
        spec=_ROCKE_UKD_SPEC,
    )


# ---------------------------------------------------------------------------
# Synthesised code objects
# ---------------------------------------------------------------------------
# Shared by the signature parser tests and the rocke producer tests, whose
# stubbed compiler has to hand back something the packer can read a signature
# out of. A real comgr object is a bare ELF carrying this note, so a stub that
# is not one exercises a path no producer takes.


def requires_msgpack():
    """Fail the calling test when the note cannot be packed.

    Synthesising an object needs msgpack; most tests in these modules do not, so
    callers reaching one down a single branch call this at that branch rather
    than marking the module.

    Fails rather than skips, matching the hipcc and rocke fixtures above: msgpack
    ships as a rocm_kpack dependency, so it cannot be legitimately absent where
    the packer itself is importable, and a skip here would quietly retire the
    signature-parsing coverage instead of reporting a broken environment.
    """
    if msgpack is None:
        pytest.fail(
            "msgpack is not importable, but it ships as a rocm_kpack dependency"
        )


def _note(owner, note_type, payload):
    """One ELF note: namesz/descsz/type, then name and desc each padded to 4."""
    name = owner.encode() + b"\x00"
    pad = lambda b: b + b"\x00" * (-len(b) % 4)
    return (
        struct.pack("<III", len(name), len(payload), note_type)
        + pad(name)
        + pad(payload)
    )


def _elf(note_bytes):
    """A minimal ELF64 LSB object holding `note_bytes` in one SHT_NOTE section.

    Two section headers -- the mandatory null entry and the note -- and no
    string table: the parser selects on section type, never on section name, so
    naming the section would test nothing the real objects exercise.
    """
    header_size = 64
    entry_size = 64
    note_offset = header_size
    sh_offset = note_offset + len(note_bytes)

    header = bytearray(header_size)
    header[0:4] = b"\x7fELF"
    header[4] = 2  # ELFCLASS64
    header[5] = 1  # ELFDATA2LSB
    header[6] = 1  # EV_CURRENT
    struct.pack_into("<Q", header, 0x28, sh_offset)
    struct.pack_into("<HHH", header, 0x3A, entry_size, 2, 0)

    null_section = struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    note_section = struct.pack(
        "<IIQQQQIIQQ", 0, 7, 0, 0, note_offset, len(note_bytes), 0, 0, 4, 0
    )
    return bytes(header) + note_bytes + null_section + note_section


def _metadata(kernels):
    return msgpack.packb({"amdhsa.kernels": kernels}, use_bin_type=True)


def _arg(kind, size, offset, name=None):
    argument = {".value_kind": kind, ".size": size, ".offset": offset}
    if name is not None:
        argument[".name"] = name
    return argument


def _kernel(name, args):
    return {".name": name, ".symbol": name + ".kd", ".args": args}


def _object(kernels):
    return _elf(_note("AMDGPU", 32, _metadata(kernels)))


def _bundle(entries):
    """A clang offload bundle over `entries`, each (triple, payload)."""
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    header = magic + struct.pack("<Q", len(entries))
    for triple, _ in entries:
        header += struct.pack("<QQQ", 0, 0, len(triple)) + triple.encode()

    body = b""
    offsets = []
    for _, payload in entries:
        offsets.append(len(header) + len(body))
        body += payload

    rebuilt = magic + struct.pack("<Q", len(entries))
    for (triple, payload), offset in zip(entries, offsets):
        rebuilt += struct.pack("<QQQ", offset, len(payload), len(triple))
        rebuilt += triple.encode()
    return rebuilt + body
