import json
import os
import shutil
import sys
import uuid
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


@pytest.fixture(scope="session")
def cmake():
    """The CMake used to drive the sub-configures and sub-builds.

    Resolved from HKP_CMAKE_COMMAND (set by CMake to its own ${CMAKE_COMMAND})
    or PATH, so the tests exercise the CMake that is running them rather than
    whichever one happens to come first on PATH.
    """
    exe = os.environ.get("HKP_CMAKE_COMMAND")
    if not exe:
        exe = shutil.which("cmake")
    if not exe:
        pytest.fail("cmake not found (set HKP_CMAKE_COMMAND or put cmake on PATH)")
    return exe


@pytest.fixture(scope="session")
def cmake_make_program():
    """The build tool handed to the sub-configures' Ninja generator.

    Resolved from HKP_CMAKE_MAKE_PROGRAM (set by CMake to its own
    ${CMAKE_MAKE_PROGRAM}) or PATH.
    """
    exe = os.environ.get("HKP_CMAKE_MAKE_PROGRAM")
    if not exe:
        exe = shutil.which("ninja")
    if not exe:
        pytest.fail("ninja not found (set HKP_CMAKE_MAKE_PROGRAM or put ninja on PATH)")
    return exe


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
def hsaco_fixture_dir(fixtures_dir):
    """Committed bare-ELF code objects, one per arch, for the authored-hsaco tests."""
    return fixtures_dir / "hsaco"


@pytest.fixture(scope="session")
def rocke_fixture(fixtures_dir):
    """rocKE descriptor data, read without rocke: the disabled-kind and
    disabled-folder tests hand it to a build without rocKE, and tests/rocke/ packs
    it."""
    return fixtures_dir / "rocke"


def _restem_uuid_ids(folder, token):
    """Give every UUID id under `folder` a distinct, deterministic value.

    Stem substitution cannot reach UUID ids, so copied fixtures would collide.
    uuid5 keeps repeat packs byte-identical.
    """
    ids = set()
    for path in sorted(folder.rglob("*.json")):
        did = json.loads(path.read_text(encoding="utf-8")).get("id")
        try:
            uuid.UUID(str(did))
        except (ValueError, AttributeError, TypeError):
            continue
        ids.add(did)
    for path in sorted(folder.rglob("*.json")):
        text = path.read_text(encoding="utf-8")
        for did in sorted(ids):
            text = text.replace(
                did, str(uuid.uuid5(uuid.NAMESPACE_URL, f"hkp-test/{token}/{did}"))
            )
        path.write_text(text, encoding="utf-8")
