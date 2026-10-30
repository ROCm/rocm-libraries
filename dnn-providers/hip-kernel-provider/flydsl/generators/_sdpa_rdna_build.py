# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""What the RDNA SDPA generators share: the per-target LLVM options and the layout check.

The prefill objects (``gen_sdpa_rdna``) and the decode objects (``gen_sdpa_rdna_decode``)
are different kernels and different engines, but both are built for the same targets
with the same codegen options, and both assert each kernel's argument layout before an
object is written.
"""

from __future__ import annotations

from ._codeobject import GeneratorError

# Every RDNA SDPA kernel's waves are 32 lanes wide.
WAVE_SIZE = 32

# LLVM codegen options per target, added to the kernel's own (its launcher
# carries `compile_hints["llvm_options"]`) for every instance built for it; the
# effective set is recorded in each manifest row. gfx11-generic has no gfx115x SALU float ops, so
# the scalar softmax arithmetic moves to VALU and d128 tips past the 256-VGPR
# ceiling by a few registers under the default scheduler's pressure tracking.
# The AMDGPU trackers' tighter accounting fits every generic instance with no
# spill (and was measured faster on gfx1151 than the default). Per target, not
# global: on native gfx1151 the same option makes f16 d128 spill.
LLVM_OPTIONS = {
    "gfx11-generic": {"amdgpu-use-amdgpu-trackers": True},
}


def verify_layout(described, expected, segment, where, what) -> None:
    """Fail before writing an object whose kernel's argument layout, kernarg segment or
    register use is not what this family declares."""
    if described["signature"] != expected:
        raise GeneratorError(
            f"{where}: {what} kernarg layout is\n  {described['signature']}\nbut is "
            f"declared as\n  {expected}\nUpdate the layout, the semantic names and the "
            "native dispatch together."
        )
    if described["kernarg_segment_size"] != segment:
        raise GeneratorError(
            f"{where}: {what} kernarg segment is "
            f"{described['kernarg_segment_size']} bytes, expected {segment}"
        )
    if described.get("vgpr_spill_count"):
        raise GeneratorError(
            f"{where}: {what} spills {described['vgpr_spill_count']} VGPRs"
        )
