# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The fused chunkwise KDA kernel built by the direct compiler is deterministic.

The kernel keeps its state accumulators in VGPRs and scales them each chunk;
the direct compiler's ``pack`` made those multiplies ``v_pk_mul_f32``, and the
MFMA that read them next got stale or torn lanes (about 1 element in 10^4 of the
state, one bf16 ulp of the output, different on every launch), which the
bitwise tests of ``test_kda_chunkwise_gfx950_numeric`` caught only as a failure
to match COMGR's code.  It is not a wait-state rule: no count of wait states
between the packed write and the MFMA is safe.  ``target/gfx9/unpack.rs`` turns
a packed f32 instruction that an MFMA of its block reads into its two scalar
ones.  This test runs the kernel under ``ROCKE_CODEOBJ_BACKEND=direct`` (strict:
a refusal fails) and compares launches with each other, for both input-staging
variants.  It needs a gfx950 GPU and the direct compiler's library.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="ROCm torch required")

pytestmark = [pytest.mark.gpu]


def _gfx950_gpu() -> bool:
    return torch.cuda.is_available() and "gfx950" in (
        torch.cuda.get_device_properties(0).gcnArchName.lower()
    )


@pytest.mark.parametrize("prefetch_inputs", [False, True])
def test_direct_built_kernel_gives_the_same_bits_on_every_launch(
    prefetch_inputs, monkeypatch
):
    if not _gfx950_gpu():
        pytest.skip("needs a gfx950 GPU with ROCm torch")
    from rocke.runtime import code_object

    if code_object._direct_library_path() is None:
        pytest.skip("build the Rust cdylib or set ROCKE_CODEOBJ_DIRECT_LIB")

    from builders.gfx950.kda import kda_chunk_fused as fused
    from kernels.gfx950.kda_chunkwise import KdaChunkFusedSpec, KdaTileSpec

    monkeypatch.setenv("ROCKE_CODEOBJ_BACKEND", "direct")
    monkeypatch.setenv("ROCKE_DIRECT_STRICT", "1")
    monkeypatch.setattr(fused, "_LAUNCHER_CACHE", {})

    B, H, T = 2, 4, 256
    spec = KdaChunkFusedSpec(tile=KdaTileSpec(), prefetch_inputs=prefetch_inputs)
    q, k, v, g, beta = fused.make_inputs(B, H, T, spec.head_k, spec.head_v)
    first = fused.launch_packed(spec, q, k, v, g, beta)
    torch.cuda.synchronize()
    for launch in range(1, 6):
        o, ht = fused.launch_packed(spec, q, k, v, g, beta)
        torch.cuda.synchronize()
        assert torch.equal(
            o, first[0]
        ), f"launch {launch}: output differs from launch 0"
        assert torch.equal(ht, first[1]), f"launch {launch}: final state differs"
