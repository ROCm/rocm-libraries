# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU geometry checks of the MN-contiguous operand layout (gemm_mn_operand)."""

import itertools

import pytest

from rocke.instances.common.gemm_mn_operand import MnLayout, mn_layout_letters

# (tile_mn, tile_k, block_size, warp_ext) of the DTL ring bodies (16x16x32 atoms)
GEOMS = [
    (256, 64, 256, 128),  # beat / B3 256x256x64, 2x2 waves
    (256, 32, 256, 128),  # B3 256x256x32
    (256, 64, 1024, 64),  # DTL ring 256x256x64, 4x4 waves
    (128, 64, 512, 64),  # ring_small 128x128x64, 2x4 waves (A side)
    (128, 64, 512, 32),  # ring_small B side (4 waves along N)
    (256, 32, 256, 64),  # 256x128x32 A side, 4x1 waves
    (128, 32, 256, 128),  # 256x128x32 B side
    (128, 32, 256, 64),  # 128x128x32 2x2 waves
    (192, 64, 512, 96),  # beat body 192x128x64 2x4 waves (A side)
    (192, 64, 512, 48),  # 128x192x64 4x2 (B side)
    (128, 64, 512, 32),
    (64, 64, 256, 32),
]


def _layouts():
    for g in GEOMS:
        try:
            yield MnLayout(
                tile_mn=g[0], tile_k=g[1], block_size=g[2], wave_size=64, warp_ext=g[3]
            )
        except ValueError:
            continue


@pytest.mark.parametrize(
    "lay",
    list(_layouts()),
    ids=lambda l: f"mn{l.tile_mn}k{l.tile_k}b{l.block_size}e{l.warp_ext}W{l.W}",
)
def test_dtl_covers_stage_once(lay):
    seen = {}
    for p in range(lay.n_pass):
        for tid in range(lay.block_size):
            k, mn = lay.dtl_lane(p, tid)
            byte = lay.dtl_lds_byte(p, tid)
            for e in range(8):
                key = (k, mn + e)
                assert key not in seen, (key, p, tid)
                seen[key] = byte + 2 * e
                assert lay.lds_byte(k, mn + e) == byte + 2 * e
    assert len(seen) == lay.tile_mn * lay.tile_k
    assert max(seen.values()) < lay.stage_bytes


@pytest.mark.parametrize(
    "lay",
    list(_layouts()),
    ids=lambda l: f"mn{l.tile_mn}k{l.tile_k}b{l.block_size}e{l.warp_ext}W{l.W}",
)
def test_reader_addresses_mfma_layout(lay):
    n_w = lay.tile_mn // lay.warp_ext
    mf = lay.warp_ext // 16
    for w, idx, kk, r, lane in itertools.product(
        range(n_w), range(mf), range(lay.tile_k // 32), range(2), range(64)
    ):
        woff = w * lay.warp_ext
        addr = lay.read_addr(lane, woff, idx, r, kk)
        assert addr == lay.read_lane_bytes(lane) + lay.read_imm_bytes(
            idx, r, kk
        ) + lay.warp_bytes(w)
        # the transpose read: lane j of a 16-lane group fetches K row (j // 4)
        # of the 4-row group, MN elements 4 (j % 4) .. + 3
        k, sub = lay.read_lane_k_mn(lane, r, kk)
        for e in range(4):
            assert addr + 2 * e == lay.lds_byte(k, woff + 16 * idx + sub + e)
    # ds_read_b64_tr_b16: output lane (q, i) of a 16-lane group receives element
    # i % 4 of the 8-byte chunk addressed by source lane 16 q + 4 g + i // 4, for
    # g = 0..3 (one K row each). Two reads (r = 0, 1) -> K rows 8q .. 8q + 7 of
    # MN row i: exactly the K-contiguous ds_read_b128 fragment.
    for lane in range(64):
        q, i = divmod(lane, 16)
        rows, mns = [], set()
        for r in range(2):
            for g in range(4):
                src = 16 * q + 4 * g + i // 4
                ks, sub = lay.read_lane_k_mn(src, r, 0)
                rows.append(ks)
                mns.add(sub + i % 4)
        assert rows == list(range(8 * q, 8 * q + 8)), (lane, rows)
        assert mns == {i}, (lane, mns)


@pytest.mark.parametrize(
    "lay",
    list(_layouts()),
    ids=lambda l: f"mn{l.tile_mn}k{l.tile_k}b{l.block_size}e{l.warp_ext}W{l.W}",
)
def test_transpose_reads_conflict_free(lay):
    if not lay.conflict_free:
        pytest.skip("shallow tile: documented two-way conflicts")
    mf = lay.warp_ext // 16
    for idx, kk, r in itertools.product(range(mf), range(lay.tile_k // 32), range(2)):
        for half in range(2):
            banks = []
            for lane in range(32 * half, 32 * half + 32):
                a = lay.read_addr(lane, 0, idx, r, kk)
                banks += [(a // 4) % 64, (a // 4 + 1) % 64]
            assert len(set(banks)) == 64, (idx, kk, r, half)


def test_layout_letters():
    assert mn_layout_letters("RCR") == (False, False)
    assert mn_layout_letters("RRR") == (False, True)
    assert mn_layout_letters("CRR") == (True, True)
    assert mn_layout_letters("CCR") == (True, False)
    with pytest.raises(ValueError):
        mn_layout_letters("RCC")


# ---------------------------------------------------------------------------
# spec validity, dispatcher routing and IR build (CPU)
# ---------------------------------------------------------------------------

import dataclasses  # noqa: E402

from rocke.dispatch.gemm import (
    GemmRequest,
    dispatch_gemm_bf16,
    dispatch_gemm_fp16,
)  # noqa: E402
from rocke.dispatch.gemm.common import native_layout_alternatives  # noqa: E402
from rocke.instances.common.gemm_mn_operand import (
    GEMM_LAYOUTS,
    mn_layout_error,
)  # noqa: E402
from rocke.instances.common.gemm_universal import (
    build_universal_gemm,
    is_valid_spec,
)  # noqa: E402

_TRANS = {lay: (lay[0] == "C", lay[1] == "C") for lay in GEMM_LAYOUTS}
_SHAPES = [
    (8192, 8192, 8192),
    (1536, 1440, 576),
    (3000, 2000, 1024),
    (2048, 2048, 512),
    (7680, 6720, 1536),
    (15360, 15360, 768),
    (256, 8192, 4096),
]


def _req(M, N, K, lay, dt="fp16"):
    ta, tb = _TRANS[lay]
    return GemmRequest(
        M=M, N=N, K=K, arch="gfx950", dtype=dt, layout=lay, trans_a=ta, trans_b=tb
    )


@pytest.mark.parametrize("dt", ["fp16", "bf16"])
@pytest.mark.parametrize("lay", ["RRR", "CRR", "CCR"])
@pytest.mark.parametrize("shape", _SHAPES)
def test_dispatch_native_layout(shape, lay, dt):
    f = dispatch_gemm_fp16 if dt == "fp16" else dispatch_gemm_bf16
    r = f(_req(*shape, lay, dt))
    from rocke.instances.common import gemm_skinny as _gs

    if isinstance(r.spec, _gs.SkinnyGemmSpec):
        # the native gemm_skinny (MN-contiguous operands in place)
        assert r.spec.layout == lay
        assert _gs.is_valid_spec(r.spec, "gfx950")[0]
        assert f"_l{lay.lower()}" in r.spec.kernel_name()
        return
    assert r.spec.data.layout == lay
    assert is_valid_spec(r.spec, "gfx950")[0]
    assert f"_L{lay}" in r.spec.kernel_name()


@pytest.mark.parametrize("shape", _SHAPES)
def test_rcr_dispatch_unchanged_by_layout_path(shape):
    r = dispatch_gemm_fp16(_req(*shape, "RCR"))
    assert r.spec.data.layout == "RCR"
    alts, _ = native_layout_alternatives(_req(*shape, "RCR"), r.spec)
    assert alts == [("same", r.spec)]


def test_layout_trans_flags_must_match():
    with pytest.raises(ValueError):
        dispatch_gemm_fp16(
            GemmRequest(M=4096, N=4096, K=4096, arch="gfx950", layout="RRR")
        )  # trans_b=True default


def test_mn_leading_dimension_not_multiple_of_8_uses_generic_loader():
    # M-contiguous A with M % 8 != 0: no 16-byte DTL rows -> the padded
    # VGPR-staged body with the widest MN load that divides M (the
    # native gemm_skinny takes such shapes first; off here)
    from unittest import mock

    from rocke.dispatch.gemm import skinny as _sk

    with mock.patch.object(_sk, "SKINNY_NATIVE", False):
        r = dispatch_gemm_fp16(_req(4100, 4096, 4096, "CCR"))
        assert r.spec.data.layout == "CCR" and not r.spec.trait.direct_to_lds
        assert r.spec.trait.mn_vec == 4 and r.spec.trait.pad_m
        r = dispatch_gemm_bf16(_req(4097, 4096, 4096, "CCR", "bf16"))
        assert r.spec.trait.mn_vec == 1


@pytest.mark.parametrize("lay", ["RRR", "CRR", "CCR"])
def test_gfx942_native_layouts_use_generic_loader(lay):
    ta, tb = _TRANS[lay]
    r = dispatch_gemm_fp16(
        GemmRequest(
            M=4096, N=4096, K=4096, arch="gfx942", layout=lay, trans_a=ta, trans_b=tb
        )
    )
    assert r.spec.data.layout == lay and not r.spec.trait.direct_to_lds
    assert is_valid_spec(r.spec, "gfx942")[0]


def test_validity_gates():
    base = dispatch_gemm_fp16(_req(3000, 2000, 1024, "RCR")).spec
    for lay in ("RRR", "CRR", "CCR"):
        s = dataclasses.replace(base, data=dataclasses.replace(base.data, layout=lay))
        assert is_valid_spec(s, "gfx950")[0]
        # no transpose reads on gfx942
        assert mn_layout_error(s, "gfx942") is not None
        # the padded ring bodies bound the MN operand with the tight resource
        s2 = dataclasses.replace(
            s, trait=dataclasses.replace(s.trait, dtl_tight_rsrc=False)
        )
        assert not is_valid_spec(s2, "gfx950")[0]
        # mn_pad values
        s3 = dataclasses.replace(s, trait=dataclasses.replace(s.trait, mn_pad=24))
        assert not is_valid_spec(s3, "gfx950")[0]
    # the 16-byte vector epilogue needs a K-contiguous B
    big = dispatch_gemm_fp16(_req(8192, 8192, 8192, "RCR")).spec
    assert big.trait.rilv_vec_epilogue and big.trait.lds_row_interleave
    mr = dataclasses.replace(big.trait, sched_strategy="iterative-minreg")
    for lay, ok in (("RRR", False), ("CRR", False), ("CCR", True)):
        s = dataclasses.replace(
            big, data=dataclasses.replace(big.data, layout=lay), trait=mr
        )
        assert is_valid_spec(s, "gfx950")[0] is ok, lay
    # the interleave body with an MN operand: only the verified scheduler,
    # never with both operands MN-contiguous
    nr = dataclasses.replace(mr, rilv_vec_epilogue=False, rve_store_nt=False)
    s = dataclasses.replace(
        big, data=dataclasses.replace(big.data, layout="RRR"), trait=nr
    )
    assert is_valid_spec(s, "gfx950")[0]
    s = dataclasses.replace(s, trait=dataclasses.replace(nr, sched_strategy=""))
    assert not is_valid_spec(s, "gfx950")[0]
    s = dataclasses.replace(
        big, data=dataclasses.replace(big.data, layout="CRR"), trait=nr
    )
    assert not is_valid_spec(s, "gfx950")[0]
    # non-DTL bodies keep RCR only
    cs = dispatch_gemm_fp16(_req(3000, 2000, 1024, "RCR")).spec
    cs = dataclasses.replace(
        cs,
        trait=dataclasses.replace(
            cs.trait, direct_to_lds=False, dtl_prefetch=False, dtl_stages=0
        ),
        data=dataclasses.replace(cs.data, layout="RRR"),
    )
    assert not is_valid_spec(cs, "gfx950")[0]
    # unknown scheduler strategy
    s4 = dataclasses.replace(
        base, trait=dataclasses.replace(base.trait, sched_strategy="fastest")
    )
    assert not is_valid_spec(s4, "gfx950")[0]


@pytest.mark.parametrize("lay", GEMM_LAYOUTS)
@pytest.mark.parametrize("shape", [(3000, 2000, 1024), (8192, 8192, 8192)])
def test_build_uses_transpose_reads_only_for_mn_operands(shape, lay):
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    spec = dispatch_gemm_fp16(_req(*shape, lay)).spec
    ir = lower_kernel_to_llvm(
        build_universal_gemm(spec, arch="gfx950"), arch="gfx950", llvm_flavor="llvm22"
    )
    n_tr = ir.count("call <4 x i16> @llvm.amdgcn.ds.read.tr16.b64")
    assert (n_tr > 0) is (lay != "RCR")
    if lay == "CCR" and spec.trait.lds_row_interleave:
        assert '"amdgpu-sched-strategy"="iterative-minreg"' in ir


def test_lds_budget_counts_block_pad():
    from rocke.instances.common.gemm_mn_operand import (
        mn_lds_extra_bytes,
        spec_mn_layouts,
    )

    spec = dispatch_gemm_fp16(_req(3000, 2000, 1024, "CRR")).spec
    la, lb = spec_mn_layouts(spec)
    S = spec.trait.dtl_stages
    assert (
        mn_lds_extra_bytes(spec) == (la.n_blocks + lb.n_blocks) * spec.trait.mn_pad * S
    )


def test_binding_stores_mn_operands_transposed():
    np = pytest.importorskip("numpy")
    from rocke.dispatch.gemm.binding import gemm_rcr_binding

    class _Rt:
        def __init__(self):
            self.bufs = {}

        def alloc(self, n):
            p = len(self.bufs) + 1
            self.bufs[p] = bytearray(n)
            return p

        def memcpy_h2d(self, p, src, n):
            self.bufs[p][:n] = bytes(src)[:n]

        def memset(self, p, v, n):
            pass

    r = dispatch_gemm_fp16(_req(1536, 1440, 576, "CRR"))
    b_ = gemm_rcr_binding(r, verify=False, dtype="fp16")
    rt = _Rt()
    _, ptrs = b_.make_args(rt)
    a = np.frombuffer(bytes(rt.bufs[ptrs[0]]), dtype=np.float16)
    r0 = dispatch_gemm_fp16(_req(1536, 1440, 576, "RCR"))
    rt0 = _Rt()
    _, p0 = gemm_rcr_binding(r0, verify=False, dtype="fp16").make_args(rt0)
    a0 = np.frombuffer(bytes(rt0.bufs[p0[0]]), dtype=np.float16)
    assert np.array_equal(a.reshape(576, 1536), a0.reshape(1536, 576).T)
