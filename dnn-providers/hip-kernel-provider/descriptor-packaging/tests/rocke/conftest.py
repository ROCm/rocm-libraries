"""Fixtures for the rocKE tests, which live in this directory and nowhere else.

A build with HIPKERNELPROVIDER_ENABLE_ROCKE=OFF makes no rocKE toolchain, and its
ctest entries `--ignore` this directory, so neither this conftest nor any test
beside it is collected there. Everything here may therefore assume the build
asked for rocKE, which is why the gates below fail rather than skip.
"""

import sys
import types
from pathlib import Path

import pytest

# Make the in-tree rocke platform + kernels library importable for the rocke
# producer tests, mirroring how the parent conftest wires hkp_pack and
# rocm_kpack onto sys.path.
#
# Source tree, not the packs' private wheels: this tests producer logic, and the
# packs already cover the wheel path.
_ROCKE_ROOT = Path(__file__).resolve().parents[3] / "rocke"
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
_ROCKE_UNAVAILABLE_HINT = (
    "provision the rocke platform and libamd_comgr; both are requirements of an "
    "ingestor built with HIPKERNELPROVIDER_ENABLE_ROCKE=ON"
)


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

    Fails rather than skips: rocke is asserted at configure time whenever the
    build enables it, and rocKE tests are registered only then, so an
    unimportable one here is a broken build rather than an unprovisioned machine.
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
    comgr ships with ROCm and a rocKE-enabled configure refuses to proceed
    without it, so this tier cannot be legitimately unavailable wherever it is
    registered -- skipping instead would let a ROCm bump that moved or dropped
    comgr turn the tier green by not running it. A real ComgrError from a compile
    is not gated here.
    """
    ok, reason = _probe_rocke()
    if not ok:
        pytest.fail(f"{reason} ({_ROCKE_UNAVAILABLE_HINT})")
    return True


@pytest.fixture(scope="session")
def rocke_ukd():
    """The reference rocke UKD source/builder/spec shared by the producer tests."""
    return types.SimpleNamespace(
        source=_ROCKE_UKD_SOURCE,
        builder=_ROCKE_UKD_BUILDER,
        spec=_ROCKE_UKD_SPEC,
    )
