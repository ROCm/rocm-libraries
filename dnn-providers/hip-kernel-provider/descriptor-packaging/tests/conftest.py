import os
import shutil
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _TESTS_DIR.parent / "python"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# rocm_kpack location: CMake passes HIPKERNELPROVIDER_KPACK_PYTHON_DIR; otherwise
# rely on an installed rocm_kpack already importable. No skip on absence — the
# compiler and kpack are load-bearing; a missing dependency is a hard failure.
_KPACK_DIR = os.environ.get("HIPKERNELPROVIDER_KPACK_PYTHON_DIR")
if _KPACK_DIR and Path(_KPACK_DIR).is_dir() and _KPACK_DIR not in sys.path:
    sys.path.insert(0, _KPACK_DIR)


@pytest.fixture(scope="session")
def kpack_python_dir():
    return _KPACK_DIR if _KPACK_DIR else None


@pytest.fixture(scope="session")
def hipcc():
    """The hipcc driver used for real --genco compilation.

    Resolved from HKP_HIPCC (set by CMake) or PATH. Missing hipcc is a hard
    failure: the compiler is load-bearing and the suite compiles for real.
    """
    exe = os.environ.get("HKP_HIPCC")
    if not exe:
        for name in ("hipcc", "hipcc.bat"):
            found = shutil.which(name)
            if found:
                exe = found
                break
    if not exe:
        raise RuntimeError(
            "hipcc not found (set HKP_HIPCC or put hipcc on PATH); the suite "
            "compiles kernels for real via --genco"
        )
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
