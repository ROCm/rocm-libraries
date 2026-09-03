import os
import shutil
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
# rocm_kpack onto sys.path. Best-effort: absent trees leave import to whatever
# provisioning the environment provides, and rocke_available still gates.
_ROCKE_ROOT = _TESTS_DIR.parent.parent / "rocke"
for _rocke_sub in ("platform/python", "library"):
    _p = _ROCKE_ROOT / _rocke_sub
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

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
_ROCKE_SKIP_REASON = (
    "rocke/comgr not loadable — provision to run (deferred CI wiring per "
    "ALMIOPEN-2420)"
)


@pytest.fixture(scope="session")
def rocm_kpack_dir():
    return _KPACK_DIR if _KPACK_DIR else None


@pytest.fixture(scope="session")
def hipcc():
    """The hipcc driver used for real --genco compilation.

    Resolved from HKP_HIPCC (set by CMake) or PATH. A missing hipcc skips the
    compile-dependent tests so a toolchain-less box is a no-op, symmetric to
    rocke_available; HIPKERNELPROVIDER_KPACK_REQUIRE_HIPCC (set in CI) turns the
    miss into a hard failure so CI cannot silently skip. A real hipcc compile
    error is not caught here — it surfaces from the pipeline as a failure.
    """
    exe = os.environ.get("HKP_HIPCC")
    if not exe:
        for name in ("hipcc", "hipcc.bat"):
            found = shutil.which(name)
            if found:
                exe = found
                break
    if not exe:
        msg = "hipcc not found (set HKP_HIPCC or put hipcc on PATH)"
        if os.environ.get("HIPKERNELPROVIDER_KPACK_REQUIRE_HIPCC"):
            pytest.fail(msg)
        pytest.skip(msg)
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
    """
    try:
        import kernels  # noqa: F401
        import rocke  # noqa: F401
    except Exception as exc:
        if os.environ.get("HIPKERNELPROVIDER_KPACK_REQUIRE_COMGR"):
            pytest.fail(f"rocke/kernels not importable: {exc}")
        pytest.skip(f"rocke/kernels not importable: {exc}")
    return True


@pytest.fixture(scope="session")
def rocke_available():
    """Session gate for the comgr-dependent rocke tests.

    Returns True when rocke/kernels import and comgr loads. Otherwise skips with
    the deferred reason, or hard-fails under HIPKERNELPROVIDER_KPACK_REQUIRE_COMGR
    (set in CI) so CI cannot silently skip. A real ComgrError from a compile is
    not gated here.
    """
    ok, reason = _probe_rocke()
    if not ok:
        if os.environ.get("HIPKERNELPROVIDER_KPACK_REQUIRE_COMGR"):
            pytest.fail(reason)
        pytest.skip(_ROCKE_SKIP_REASON)
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
