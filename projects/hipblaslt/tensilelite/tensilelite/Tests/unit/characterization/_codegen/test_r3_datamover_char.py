################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R3 — gfx1250 TensorDataMover non-wave-separated path characterization.

Target: Tensile/Components/TensorDataMover.py (miss=158 before this test).

The existing gfx1250 emit suite (test_emit_gfx1250_char.py / F4_MX.yaml) only
exercises the *wave-separated* code path because that logic file has
MIWaveGroup=[2,2].  Lines 29-116 (``calculateStartAddr``, the non-wave-sep
variant) were completely unexecuted.

This test drives the designed ``datamover.yaml`` config (F4/BF16 MX-GEMM with
TDMInst=3) using two MatrixInstruction shapes that both have MIWaveGroup=[1,1]
(prod==1), which forces the emitter down the non-wave-separated
``calculateStartAddr`` branch.  The MX-F4 data type (MXBlock=32) additionally
hits the MXS arms inside ``setTensorDim0``, ``setTensorDim1``, and
``setTensorStride0``.

Golden: order-invariant {basename, err} digest snapshot (seeded once).
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "datamover.yaml",
)

_SUBTILE_CONFIG = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "common",
        "streamk",
        "gfx1250",
        "core",
        "sk_bf16gemm_tdm_subtile.yaml",
    )
)


@pytest.fixture(scope="module")
def emitted():
    return emit_kernels_from_config(_CONFIG, limit=4, arch=_ARCH)


def test_r3_datamover_gfx1250_emits_assembly(emitted):
    """TDM non-wave-separated (MIWaveGroup=[1,1]) config emits real gfx1250 assembly."""
    assert len(emitted) >= 1, f"Expected >=1 kernel, got {len(emitted)}"
    assert all(err == 0 for (_b, _s, err) in emitted), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in emitted if e != 0])
    )
    for base, src, _err in emitted:
        assert src and len(src.splitlines()) > 100, f"Kernel {base!r} source too short"
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx1250" in src, f"Kernel {base!r} missing gfx1250 arch marker"
        assert base.startswith("Cijk_"), f"Unexpected basename: {base!r}"


def test_r3_datamover_general_batch_dereferences_inputs_before_first_load(emitted):
    """Universal TDM loads A[batch]/B[batch] before the first tensor load."""
    for base, src, err in emitted:
        assert err == 0, f"Kernel {base!r} failed to emit"
        first_load = src.index("tensor_load_to_lds")
        prologue = src[:first_load]
        for tc in ("A", "B"):
            pointer_load = f"load {tc} matrix address from pointer array"
            start_offset = f"TDM calc start addr of {tc}"
            assert pointer_load in prologue, (
                f"Kernel {base!r}: TDM {tc} loads the selected matrix from the pointer array"
            )
            assert prologue.index(pointer_load) < prologue.index(start_offset), (
                f"Kernel {base!r}: {tc}[batch] is loaded before TDM start-addr arithmetic"
            )
            batch_offset_load = f"load batchOffset{tc} from kernel args"
            batch_offset_apply = f"apply batchOffset{tc} (low)"
            assert prologue.index(pointer_load) < prologue.index(batch_offset_load)
            assert prologue.index(batch_offset_load) < prologue.index(batch_offset_apply)
        pointer_loads = sum(
            prologue.count(f"load {tc} matrix address from pointer array")
            for tc in ("A", "B")
        )
        suppressed_strides = prologue.count(
            "general batch uses an already-dereferenced matrix base"
        )
        assert suppressed_strides == pointer_loads, (
            f"Kernel {base!r} has {suppressed_strides} suppressed batch strides "
            f"for {pointer_loads} dereferenced A/B pointers; MX or metadata "
            "must retain their direct-pointer batch stride"
        )


def test_r3_datamover_subtile_resolves_pointer_before_offsets():
    """The live subtile TDM path must resolve A/B before address arithmetic."""
    base, src, err = emit_kernels_from_config(
        _SUBTILE_CONFIG, limit=1, arch=_ARCH
    )[0]
    assert err == 0, f"Kernel {base!r} failed to emit"
    first_load = src.index("tensor_load_to_lds")
    prologue = src[:first_load]
    for tc in ("A", "B"):
        pointer_load = f"load {tc} matrix address from pointer array"
        assert pointer_load in prologue
        pointer_load_idx = prologue.index(pointer_load)
        suppressed_stride_idx = prologue.find(
            "general batch uses an already-dereferenced matrix base",
            pointer_load_idx,
        )
        batch_stride_idx = prologue.find("Batch: Stride*WG", pointer_load_idx)
        assert suppressed_stride_idx != -1
        assert batch_stride_idx != -1
        assert pointer_load_idx < suppressed_stride_idx < batch_stride_idx
        assert f"load batchOffset{tc} from kernel args" in prologue


def test_r3_datamover_gfx1250_golden(snapshot, emitted):
    """P3 golden: order-invariant {basename, err} digest of the TDM datamover emit."""
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in emitted),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
