# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device numerics of the sigmoid-gated fused output norm on GDN
(``norm_gate_act="sigmoid"``) at grouped heads (Hk 16 / Hv 48, f32 state).

Each tile with one workgroup per head runs with and without the norm knobs
(``out_lds``, ``stream_rows``, ``dpp_reduce``, ``interleave_cols``) at batches
1, 3 and 4; ``check`` scores the output (relative, as the norm's output is
O(1)), the written state and every untouched pool page against the fp32
reference, whose gate activation is the override.

Needs a real gfx950 and ROCm torch; marked ``gpu`` and skipped elsewhere.
"""

from __future__ import annotations

import pytest

ARCH = "gfx950"

torch = pytest.importorskip("torch", reason="ROCm torch required")

pytestmark = pytest.mark.gpu


def _device_is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch() == ARCH
    except Exception:
        return False


requires_gfx950 = pytest.mark.skipif(
    not _device_is_gfx950(), reason=f"needs a {ARCH} device"
)

TILES = [(1, 16, 1), (2, 16, 1), (4, 16, 1), (8, 16, 1), (4, 8, 1), (16, 16, 1)]
KNOBS = [
    dict(),
    dict(out_lds=True),
    dict(dpp_reduce=True, stream_rows=True, interleave_cols=True, out_lds=True),
    dict(state_load_hint="streaming", state_store_hint="streaming", waves_per_eu=4),
]


@requires_gfx950
@pytest.mark.parametrize("batch", [1, 3, 4])
@pytest.mark.parametrize("act", ["sigmoid", "silu"])
def test_norm_gate_override_matches_reference(batch, act):
    from builders.gfx950.gdn import gdn_decode as host
    from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

    ran = 0
    for tile in TILES:
        for knobs in KNOBS:
            spec = GdnDecodeSpec(
                num_k_heads=16,
                num_v_heads=48,
                state_dtype="f32",
                fuse_out_norm=True,
                norm_gate_act=act,
                num_warps=tile[0],
                warp_threads_k=tile[1],
                blocks_per_v_dim=tile[2],
                **knobs,
            )
            if not is_valid_spec(spec, arch=ARCH)[0]:
                continue
            out_err, state_err = host.check(spec, batch)
            assert out_err <= host.TOL, (spec.kernel_name(), out_err)
            assert state_err <= host.TOL, (spec.kernel_name(), state_err)
            # and per row relative to the reference's scale (OUT_RTOL)
            inp = host.make_inputs(spec, batch)
            ref_out, _ = host.ref_fp32(spec, inp)
            got = host.run_values(spec, inp, host.launcher_for(spec), batch)["out"]
            out_rel = host.out_rel_error(got, ref_out)
            assert out_rel <= host.OUT_RTOL, (spec.kernel_name(), out_rel)
            ran += 1
    assert ran >= len(TILES)
