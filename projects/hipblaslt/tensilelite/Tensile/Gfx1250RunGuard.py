# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute-time guard: do not *run* gfx1250 revision-1 YAML on revision-0 hardware.

A YAML opts into revision 1 with ``TestParameters.RevisionID: 1`` (default 0)
or with ``skip-gfx1250v0`` in ``TestParameters.marks``. That is test metadata,
not a codegen fork. StreamK tests that set either abort when Tensile.py
executes the YAML on gfx1250 revision 0.

The check is a no-op unless this invocation is gfx1250. pytest/tox collection
uses the same helpers so skip and launch agree.
"""

import os
import sys

from Tensile.GpuRevisionTarget import detect_gpu_revision_target

# Matches Tensile.Common.Architectures.baseArchName without importing rocisa.
def _bare_arch_name(spec):
    return spec.split("[")[0].split(":")[0].strip()


GFX1250_ISA = (12, 5, 0)
DEFAULT_ASIC_REVISION = 0
SKIP_GFX1250V0_MARK = "skip-gfx1250v0"

GFX1250_V1_ON_V0_ERROR = (
    "Cannot run gfx1250 revision-1 kernels on revision-0 hardware "
    "(asicRevision==0). This device is gfx1250v0. StreamK tests that set "
    "skip-gfx1250v0 or RevisionID 1 do not run on this device. --build-only "
    "remains allowed so revision-1 code can still be cross-compiled on this "
    "machine."
)


def _is_gfx1250_family_name(name):
    return name == "gfx1250" or name.startswith("gfx1250v")


def is_gfx1250_family_compile(arch_names, isa_info_map):
    """True when this invocation targets the gfx1250 ISA (any stepping)."""
    names = {_bare_arch_name(a) for a in (arch_names or []) if a}
    if any(_is_gfx1250_family_name(n) for n in names):
        return True
    if names:
        return False
    if not isa_info_map:
        return False
    return any(tuple(key)[:3] == GFX1250_ISA for key in isa_info_map)


def _parse_asic_revision(value):
    if value is True:
        return 1
    if value is False or value is None:
        return DEFAULT_ASIC_REVISION
    try:
        return int(value)
    except (TypeError, ValueError):
        return DEFAULT_ASIC_REVISION


def config_required_asic_revision(config):
    """YAML-required RevisionID. Default 0. Not inferred from codegen knobs.

    ``TestParameters.RevisionID`` wins. ``skip-gfx1250v0`` in marks means 1.
    ``GlobalParameters.RevisionID`` is a fallback.
    """
    if not isinstance(config, dict):
        return DEFAULT_ASIC_REVISION
    test_params = config.get("TestParameters")
    if isinstance(test_params, dict):
        if "RevisionID" in test_params:
            return _parse_asic_revision(test_params["RevisionID"])
        marks = test_params.get("marks") or []
        if SKIP_GFX1250V0_MARK in marks:
            return 1
    global_params = config.get("GlobalParameters")
    if isinstance(global_params, dict) and "RevisionID" in global_params:
        return _parse_asic_revision(global_params["RevisionID"])
    return DEFAULT_ASIC_REVISION


def config_targets_gfx1250(config, filepath=None):
    """True when this YAML is a gfx1250 test.

    Used so pytest/tox revision skips do not fire on other architectures.
    """
    if not isinstance(config, dict):
        return False
    global_params = config.get("GlobalParameters")
    if isinstance(global_params, dict):
        arch = global_params.get("Architecture")
        if arch and _is_gfx1250_family_name(_bare_arch_name(str(arch))):
            return True
    test_params = config.get("TestParameters")
    if isinstance(test_params, dict):
        marks = test_params.get("marks") or []
        if SKIP_GFX1250V0_MARK in marks:
            return True
    if filepath:
        parts = os.path.normpath(str(filepath)).split(os.sep)
        name = os.path.basename(str(filepath))
        if any(_is_gfx1250_family_name(p) for p in parts):
            return True
        if "gfx1250" in name:
            return True
    return False


def requires_gfx1250_rev1(config, filepath=None):
    """True when this gfx1250 YAML is revision-1 and must not run on rev0."""
    if not config_targets_gfx1250(config, filepath):
        return False
    return config_required_asic_revision(config) >= 1


def guard_gfx1250_v1_run_on_v0(
    *,
    build_only,
    cpu_only,
    arch_names,
    isa_info_map,
    device_id=0,
    config=None,
    solutions=None,  # unused; kept so older call sites still type-check
):
    """Abort before a client/benchmark *run* of a gfx1250 rev1 YAML on rev0 HW.

    No-ops for ``--build-only``, ``--cpu-only``, non-gfx1250 compiles, YAMLs
    that default to RevisionID 0, and hardware that is not gfx1250v0.
    """
    _ = solutions
    if build_only or cpu_only:
        return
    if not is_gfx1250_family_compile(arch_names, isa_info_map):
        return
    if not requires_gfx1250_rev1(config):
        return
    hw = detect_gpu_revision_target(device_id=device_id)
    if hw != "gfx1250v0":
        return
    print("Tensile::FATAL: %s" % GFX1250_V1_ON_V0_ERROR)
    sys.stdout.flush()
    sys.exit(-1)
