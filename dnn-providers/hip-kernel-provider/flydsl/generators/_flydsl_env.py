# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pinned build environment for the FlyDSL AOT generators.

Every generator in this directory compiles against *one* declared toolchain: a
pinned FlyDSL wheel, the vendored kernel sources under ``kernels_src/``, and a
recorded ROCm version. All three go into the per-arch ``SOURCE.md``, because a
checked-in code object regenerated against a different toolchain is a change
that reviews as a no-op diff. The FlyDSL release notes and this tree's Flash2
precedent both record large performance swings from the ROCm version alone on
identical source.

The environment is deliberately process-wide and single-arch. FlyDSL reads
``ARCH`` from ``os.environ`` at each compile (``flydsl/utils/env.py``:
``CompileEnvManager.arch``), so mutating it mid-run would work -- and would make
the arch a kernel object was *built* for independent of the arch its filename
claims. That is the exact mislabeling hazard the Flash2 CMake block warns about,
so one arch per invocation is enforced here and the generator additionally
checks each object's own ``amdhsa.target`` after the fact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Pins -------------------------------------------------------------------

# The FlyDSL *compiler* (pip wheel). `kernels/` is not in the wheel; the kernel
# sources this provider compiles are vendored under kernels_src/ with their own
# per-file provenance headers, so this pin covers the compiler only.
FLYDSL_VERSION = "0.3.4"

# The FlyDSL checkout the vendored kernel sources were taken from. Reproduced in
# each <op>/<arch>/SOURCE.md under CONTENT_DIR; not importable here. tools/diff_upstream.py reads
# both of these -- it is the consumer that turns them from a claim into a check.
FLYDSL_KERNELS_COMMIT = "89ad52fbbb9e252396829a8a44b12e18b7acd970"
FLYDSL_KERNELS_DESCRIBE = "v0.3.4.1-19-g89ad52f"

# The AITER checkout the attention source was taken from. AITER carries FlyDSL
# kernels that upstream FlyDSL does not -- the RDNA4 flash-attention kernel the
# gfx1151 port starts from among them -- so a vendored file names which of the
# two it came from in its own header, and this pins the second.
AITER_KERNELS_COMMIT = "8253efc4059516f492df597b80ce5311e9fda8bd"

# --- Paths ------------------------------------------------------------------

GENERATORS_DIR = Path(__file__).resolve().parent
PROVIDER_DIR = GENERATORS_DIR.parent
KERNELS_SRC_DIR = PROVIDER_DIR / "kernels_src"

# Where the checked-in objects and their descriptors live: FlyDSL's producer
# folder under the provider's production descriptor root, beside rocKE's. The
# shared packer walks that root, packs each authored `hsaco` UKD's object into
# the per-arch archive and installs it with everything else, so the objects have
# to sit inside it -- the packer refuses an object path that leaves the root.
# Laid out <op>/ (descriptors shared across arches) and <op>/<arch>/ (the KDP,
# its UKDs, the objects, manifest.json and SOURCE.md).
CONTENT_DIR = (
    PROVIDER_DIR.parent
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "descriptors"
    / "FlyDSL"
)


class FlydslEnvError(RuntimeError):
    """The pinned build environment is not the one that is installed."""


def rocm_version() -> str:
    """The ROCm version this run compiles against, for the provenance record.

    Read from ``$ROCM_PATH/.info/version``, which the SDK writes. Absent that we
    return a marker rather than guessing: an unrecorded toolchain in SOURCE.md is
    better than a wrong one, since the wrong one reads as verified.
    """
    explicit = os.environ.get("ROCM_VERSION")
    if explicit:
        return explicit.strip()
    rocm_path = os.environ.get("ROCM_PATH")
    if rocm_path:
        marker = Path(rocm_path) / ".info" / "version"
        if marker.is_file():
            return marker.read_text(encoding="utf-8").strip()
    return "unknown"


def assert_flydsl_version() -> str:
    """Import FlyDSL and confirm it is the pinned release."""
    try:
        import flydsl
    except ImportError as exc:  # pragma: no cover - environment problem
        raise FlydslEnvError(
            "flydsl is not importable. These generators need a Python environment "
            "with the pinned flydsl wheel and torch installed; see REGEN.md."
        ) from exc

    installed = getattr(flydsl, "__version__", None)
    if installed != FLYDSL_VERSION:
        raise FlydslEnvError(
            f"flydsl {installed!r} is installed but this provider pins "
            f"{FLYDSL_VERSION!r}. Regenerating against a different compiler is a "
            "change to the checked-in objects, not a refresh: install the pinned "
            "version, or update FLYDSL_VERSION here and regenerate every arch."
        )
    return installed


def prepare(arch: str) -> None:
    """Put the vendored kernel sources on the path and pin the compile env.

    Must run before any ``kernels.*`` import. ``COMPILE_ONLY=1`` is what lets a
    gfx942/gfx950 object be produced on an RDNA machine: the launcher traces and
    compiles but never dispatches, so no matching device has to be present.
    """
    if not KERNELS_SRC_DIR.is_dir():
        raise FlydslEnvError(f"vendored kernel sources missing: {KERNELS_SRC_DIR}")

    if "kernels" in sys.modules:
        raise FlydslEnvError(
            "the 'kernels' package was imported before prepare(); the vendored "
            "sources under kernels_src/ may have been shadowed by an upstream "
            "FlyDSL checkout on PYTHONPATH"
        )

    src = str(KERNELS_SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)

    os.environ["ARCH"] = arch
    # FlyDSL's codegen-time arch queries -- the WMMA atom, s_waitcnt encoding,
    # buffer-descriptor flags, and the vendored kernels' own feature checks --
    # read `get_rocm_arch()`, which is FLYDSL_GPU_ARCH or else the *host GPU*,
    # never ARCH. Unpinned, a gfx950 object built on an RDNA machine carries RDNA
    # buffer descriptors. Pin it to the compile target.
    # A generic family whose name those checks cannot classify (gfx12-generic)
    # names a member to tell them instead; the object's target stays the family.
    gpu_arch = arch
    if arch.endswith("-generic"):
        from . import _arch_families  # noqa: PLC0415

        gpu_arch = _arch_families.family(arch).get("flydsl_gpu_arch", arch)
    os.environ["FLYDSL_GPU_ARCH"] = gpu_arch
    os.environ["COMPILE_ONLY"] = "1"
    os.environ["FLYDSL_DUMP_IR"] = "1"

    if arch.endswith("-generic"):
        from . import _generic_targets  # noqa: PLC0415

        _generic_targets.install(arch)
        # Every generic family is RDNA, and a check that did not see it as RDNA
        # would build CDNA buffer descriptors without an error.
        from flydsl.runtime.device import is_rdna_arch  # noqa: PLC0415

        if not is_rdna_arch(gpu_arch):
            raise FlydslEnvError(
                f"FlyDSL does not classify {gpu_arch!r} as RDNA; building {arch} "
                "through it would emit CDNA buffer descriptors"
            )


def set_dump_dir(dump_dir: Path) -> None:
    """Point the IR dump at `dump_dir` for the next compile.

    One directory per instance. ``hsaco_from_dump`` refuses a directory holding
    more than one compiled kernel rather than picking one, so reusing a dump dir
    across instances fails loudly instead of silently packaging the first object
    under every instance's name.
    """
    dump_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_DUMP_DIR"] = str(dump_dir)


def provenance(arch: str) -> dict:
    """The toolchain record embedded in manifests and SOURCE.md.

    A generic target also records the shim it was built through, since that is
    part of how the bytes came to be.
    """
    from . import _generic_targets  # noqa: PLC0415

    record = {
        "flydsl_version": FLYDSL_VERSION,
        "flydsl_kernels_commit": FLYDSL_KERNELS_COMMIT,
        "flydsl_kernels_describe": FLYDSL_KERNELS_DESCRIBE,
        "aiter_kernels_commit": AITER_KERNELS_COMMIT,
        "rocm_version": rocm_version(),
    }
    record.update(_generic_targets.record(arch) or {})
    return record
