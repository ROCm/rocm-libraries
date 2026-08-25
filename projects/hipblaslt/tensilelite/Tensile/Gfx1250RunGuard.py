# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Abort Tensile.py execute of a gfx1250 RevisionID 1 / skip-gfx1250v0 YAML on rev0.

Default RevisionID is 0. That field is test metadata, not a codegen fork.
``--build-only`` / ``--cpu-only`` still cross-compile.
"""

import sys

from Tensile.GpuRevisionTarget import (
    GFX1250_V0,
    arch_skip_token,
    argv_selects_gfx1250v0,
    detect_gpu_revision_target,
)

GFX1250_ISA = (12, 5, 0)
DEFAULT_ASIC_REVISION = 0
SKIP_GFX1250V0_MARK = "skip-gfx1250v0"

GFX1250_REV1_ON_REV0_REASON = (
    "This test is marked skip-gfx1250v0 / RevisionID: 1 and the selected GPU is "
    "revision 0."
)
GFX1250_V1_ON_V0_ERROR = (
    GFX1250_REV1_ON_REV0_REASON
    + " --build-only remains allowed so revision-1 code can still be "
    "cross-compiled on this machine."
)


def _is_gfx1250_family_name(name):
    return name == "gfx1250" or name.startswith("gfx1250v")


def is_gfx1250_family_compile(arch_names, isa_info_map):
    """True when this invocation targets the gfx1250 ISA (revision 0 or 1)."""
    names = {arch_skip_token(a) for a in (arch_names or []) if a}
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
    """TestParameters.RevisionID, else skip-gfx1250v0 -> 1, else 0."""
    if not isinstance(config, dict):
        return DEFAULT_ASIC_REVISION
    test_params = config.get("TestParameters")
    if isinstance(test_params, dict):
        if "RevisionID" in test_params:
            return _parse_asic_revision(test_params["RevisionID"])
        marks = test_params.get("marks") or []
        if SKIP_GFX1250V0_MARK in marks:
            return 1
    return DEFAULT_ASIC_REVISION


def config_targets_gfx1250(config):
    """True when this YAML is a gfx1250 test."""
    if not isinstance(config, dict):
        return False
    global_params = config.get("GlobalParameters")
    if isinstance(global_params, dict):
        arch = global_params.get("Architecture")
        if arch and _is_gfx1250_family_name(arch_skip_token(str(arch))):
            return True
    test_params = config.get("TestParameters")
    if isinstance(test_params, dict):
        marks = test_params.get("marks") or []
        if SKIP_GFX1250V0_MARK in marks:
            return True
    return False


def requires_gfx1250_rev1(config):
    """True when this gfx1250 YAML is revision-1 and must not run on rev0."""
    if not config_targets_gfx1250(config):
        return False
    return config_required_asic_revision(config) >= 1


def should_skip_gfx1250_rev1_on_rev0(
    config, tensile_argv=None, hardware_target=None
):
    """True when pytest/tox should skip this YAML on gfx1250 revision 0."""
    if not requires_gfx1250_rev1(config):
        return False
    if argv_selects_gfx1250v0(tensile_argv):
        return True
    hw = hardware_target
    if hw is None:
        hw = detect_gpu_revision_target()
    return arch_skip_token(hw) == GFX1250_V0


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
    """Abort a gfx1250 rev1 YAML *run* on rev0 HW.

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
