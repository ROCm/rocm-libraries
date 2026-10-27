# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests of the rocKE arm of the backward timing harness (CPU only).

Configuration files, the census-mask -> band mapping (checked against the
reference mask semantics), the request built from tensor strides, and the
adapter consistency the sweep relies on: the kernel a sweep candidate
compiles is the kernel the timing plan launches for every cell of its slice.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from benchmarks.common import attention_bwd_bench as bb
from benchmarks.common import attention_bwd_rocke_arm as ra
from benchmarks.common import attention_bwd_sweep as sw
from kernels.common.attention_bwd_plan import attn_bwd_plan
from rocke.numeric.sdpa_reference import keep, resolve_diagonal_band


class _T:
    """Shape / stride / pointer stand-in for a device tensor."""

    def __init__(self, shape, strides, ptr=1 << 20, dtype="torch.bfloat16"):
        self.shape, self._st, self._p, self.dtype = shape, strides, ptr, dtype

    def stride(self, i=None):
        return self._st if i is None else self._st[i]

    def data_ptr(self):
        return self._p


def _dense(cell, layout=None):
    layout = layout or cell.layout
    out = {}
    for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
        h = {"q": cell.hq, "o": cell.hq, "do": cell.hq, "dq": cell.hq}.get(n)
        h = h or (cell.hkv if n in ("k", "dk") else cell.h_v)
        s = cell.sq if n in ("q", "o", "do", "dq") else cell.skv
        d = cell.d
        st = (h * s * d, s * d, d, 1) if layout == "bhsd" else (s * h * d, d, h * d, 1)
        out[n] = _T((cell.b, h, s, d), st)
    return out


def _thd(cell):
    out = {}
    for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
        h = {"q": cell.hq, "o": cell.hq, "do": cell.hq, "dq": cell.hq}.get(n)
        h = h or (cell.hkv if n in ("k", "dk") else cell.h_v)
        tot = sum(cell.seqlens if n in ("q", "o", "do", "dq") else cell.kv_lengths)
        out[n] = _T((tot, h, cell.d), (h * cell.d, cell.d, 1))
    out["offsets_q"] = _T((cell.b + 1,), (1,), dtype="torch.int32")
    out["offsets_kv"] = _T((cell.b + 1,), (1,), dtype="torch.int32")
    return out


def _tensors(cell):
    return _thd(cell) if cell.is_thd else _dense(cell)


# --------------------------------------------------------------------------- configs
def test_config_roundtrip_and_files(tmp_path):
    c = ra.RockeConfig("x", {"warp_grid_g4": (1, 4), "edge_tiles": True},
                       {"g_split": 2})  # fmt: skip
    back = ra.RockeConfig.from_dict(json.loads(json.dumps(c.to_dict())))
    assert back == c
    p = tmp_path / "c.json"
    ra.write_configs(p, [c, ra.SHIPPED])
    assert ra.load_configs(p) == [c, ra.SHIPPED]
    p.write_text(json.dumps({"configs": [c.to_dict(), c.to_dict()]}))
    with pytest.raises(ValueError):
        ra.load_configs(p)


# --------------------------------------------------------------------------- masks
@pytest.mark.parametrize(
    "mask,sq,skv,window",
    [
        ("none", 7, 9, None),
        ("causal_tl", 7, 9, None),
        ("causal_tl", 9, 7, None),
        ("causal_br", 7, 9, None),
        ("causal_br", 9, 7, None),
        ("swa", 9, 13, 4),
        ("swa", 13, 9, 3),
    ],
)
def test_band_fields_follow_the_census_mask(mask, sq, skv, window):
    torch = pytest.importorskip("torch")
    cell = bb.Cell("x", 64, 2, 2, 1, sq, skv, mask, "bf16", "bshd", window=window)
    f = ra.band_fields(cell)
    left, right, tl = resolve_diagonal_band(
        left_bound=f.get("left_bound"),
        right_bound=f.get("right_bound"),
        top_left=not f.get("bottom_right", False),
        causal=f.get("causal", False),
        causal_bottom_right=f.get("causal_bottom_right", False),
    )
    got = torch.tensor(
        [[keep(i, j, sq, skv, tl, left, right) for j in range(skv)] for i in range(sq)]
    )
    want = bb.allowed_mask(mask, sq, skv, window, "cpu")
    want = torch.ones(sq, skv, dtype=torch.bool) if want is None else want
    assert torch.equal(got, want)


# --------------------------------------------------------------------------- request
def test_request_from_tensor_strides():
    cells = {c.id: c for c in bb.census_cells(bb.load_census())}
    bshd = cells["d128_h32_kv8_b8_q1024_k1024_causal_tl_bf16_bshd"]
    req = ra.request_for_cell(bshd, _tensors(bshd), scale=0.1)
    assert req.strides["q"] == (1024 * 32 * 128, 128, 32 * 128, 1)
    assert req.strides["k"] == (1024 * 8 * 128, 128, 8 * 128, 1)
    assert req.causal and req.layout == "dense" and req.tensor_alignment == 256
    thd = cells["d128_h32_kv8_thd_n8_t16384_causal_tl_bf16"]
    r2 = ra.request_for_cell(thd, _tensors(thd), scale=0.1)
    assert r2.layout == "thd" and r2.b == 8 and r2.s_q == max(thd.seqlens)
    assert r2.strides["q"] == (0, 128, 32 * 128, 1)
    assert r2.ragged_offset_dtype == "int32" and r2.ragged_offset_multiplier == 1
    t = _tensors(bshd)
    t["q"]._p = (1 << 20) + 2
    assert ra.request_for_cell(bshd, t, scale=0.1).tensor_alignment == 2


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_shipped_config_plans_every_census_cell(arch):
    """The baseline rocKE arm (the shipped configuration) serves every census cell."""
    for cell in bb.census_cells(bb.load_census()):
        req = ra.request_for_cell(cell, _tensors(cell), scale=ra.scale_of(cell))
        plan = attn_bwd_plan(req, arch, policy=ra.config_policy(ra.SHIPPED))
        assert plan.specs["main"].head_size == cell.d


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_tuning_buckets_keep_every_census_cell_on_its_configuration(arch):
    """The kv-tile and grid-fill buckets of the tuned-table key change no
    census cell's configuration under the default tables (no table entry):
    every fill variant of a cell (no ``num_cus``, under, full, over, and an
    ``num_xcds`` hint) plans the same main spec, runtime values and launches
    as the cell without hints. A class-wide entry (both buckets
    ``ANY_BUCKET``) resolves every census cell of the class exactly as the
    per-class hook with the same knobs; an entry on one bucket reaches only
    the cells of that bucket."""
    import dataclasses

    from kernels.common.attention_bwd_plan import (
        ANY_BUCKET,
        FILL_BUCKETS,
        KV_TILE_BUCKETS,
    )

    shipped = ra.config_policy(ra.SHIPPED)
    assert shipped.tuned_table == {}
    kv_labels = {f"t{b}" for b in KV_TILE_BUCKETS} | {f"t>{KV_TILE_BUCKETS[-1]}"}
    knobs = {"lb_order": "reverse", "scale_placement": "convert"}
    seen_buckets = set()
    for cell in bb.census_cells(bb.load_census()):
        req = ra.request_for_cell(cell, _tensors(cell), scale=ra.scale_of(cell))
        base = attn_bwd_plan(req, arch, policy=shipped)
        key = base.tuning_key
        assert key[:-2] == base.perf_class
        assert key[-2] in kv_labels and key[-1] == "unknown"
        grid = 1
        for g in base.launches[1].grid:
            grid *= g
        for hints in (
            {"num_cus": 2 * grid + 1},
            {"num_cus": grid},
            {"num_cus": 1},
            {"num_xcds": 8},
        ):
            p = attn_bwd_plan(dataclasses.replace(req, **hints), arch, policy=shipped)
            assert p.tuning_key[-1] in FILL_BUCKETS
            seen_buckets.add(p.tuning_key[-2:])
            assert p.specs == base.specs and p.launches == base.launches, (
                cell.id,
                hints,
            )
            assert dict(p.runtime) == dict(base.runtime)
        wide = dataclasses.replace(
            shipped, tuned_table={key[:-2] + (ANY_BUCKET, ANY_BUCKET): knobs}
        )
        hook = dataclasses.replace(shipped, tuning=lambda *a: dict(knobs))
        via_table = attn_bwd_plan(req, arch, policy=wide)
        via_hook = attn_bwd_plan(req, arch, policy=hook)
        assert via_table.specs == via_hook.specs
        assert via_table.launches == via_hook.launches
        assert via_table.runtime["lb_order"] == "reverse"
        over = dataclasses.replace(req, num_cus=1)
        okey = attn_bwd_plan(over, arch, policy=shipped).tuning_key
        narrow = dataclasses.replace(shipped, tuned_table={okey: knobs})
        if okey[-1] == "over":
            assert attn_bwd_plan(req, arch, policy=narrow).launches == base.launches
            hit = attn_bwd_plan(over, arch, policy=narrow)
            assert hit.runtime["scale_placement"] == "convert"
    # the census reaches more than one kv-tile and fill bucket
    assert len({k for k, _ in seen_buckets}) > 1
    assert {f for _, f in seen_buckets} >= {"under", "full", "over"}


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("d", [64, 128])
def test_sweep_kernel_is_the_timed_kernel(arch, d):
    """For every representative cell of a primary slice, the plan the harness
    builds from a candidate's configuration launches exactly the main kernel
    the sweep compiled for the candidate (also with a head split)."""
    sl = sw.primary_slice(arch, "general", d)
    cells = sw.representative_cells(sl)
    cands = sw.geometry_stage(sl, sw.Release(frozenset({1}))).candidates[::4]
    assert cells and cands
    for c in cands:
        for x in (c, c.with_runtime(g_split=2, lb_order="reverse")):
            want = sw.legality(x).resolved
            for cell in cells:
                req = ra.request_for_cell(cell, _tensors(cell), scale=ra.scale_of(cell))
                plan = attn_bwd_plan(req, arch, policy=ra.config_policy(x.config()))
                assert plan.specs["main"] == want
                assert plan.runtime["g_split"] == x.r.get("g_split", 1)


# --------------------------------------------------------------------------- hints
def test_request_hints_default_to_the_device():
    """The arm passes the device's CU count and the target's XCD count unless
    the configuration sets them; ``None`` keeps a hint absent; the
    ``DEVICE_NUM_CUS`` sentinel needs a readable count."""
    seen = []

    def read(i):
        seen.append(i)
        return 256

    assert ra.request_hints({}, "gfx950", 3, read) == {"num_cus": 256, "num_xcds": 8}
    assert seen == [3]
    assert ra.request_hints({}, "gfx942", 0, lambda i: 304) == {
        "num_cus": 304,
        "num_xcds": 8,
    }
    # an unreadable count leaves the request as before; no XCD entry off CDNA
    assert ra.request_hints({}, "gfx1151", 0, lambda i: None) == {}
    rt = {"num_cus": 17, "num_xcds": 2, "g_split": 2}
    assert ra.request_hints(rt, "gfx950", 0, read) == {"num_cus": 17, "num_xcds": 2}
    off = {"num_cus": None, "num_xcds": None}
    assert ra.request_hints(off, "gfx950", 0, read) == {}
    dev = {"num_cus": ra.DEVICE_NUM_CUS}
    assert ra.request_hints(dev, "gfx950", 0, read)["num_cus"] == 256
    with pytest.raises(ra.Unsupported, match="num_cus"):
        ra.request_hints(dev, "gfx950", 0, lambda i: None)
    assert dict(ra.DEVICE_NUM_XCDS) == dict(sw.SWEEP_NUM_XCDS)
    # the hints are request fields, never tuning values of the policy
    pol = ra.config_policy(ra.RockeConfig("x", {}, {**rt, "lb_order": "mirror"}))
    assert pol.tuning(None, "gfx950", "general", ()) == {
        "g_split": 2,
        "lb_order": "mirror",
    }


def test_sweep_candidates_carry_the_device_cu_count():
    """Every stage R candidate of a masked slice reaches the plan with the
    device's CU count (mirror candidates pin the sentinel, the others take
    the arm's default) and the sweep's XCD count."""
    sl = sw.primary_slice("gfx950", "general", 128)
    rel = sw.Release.parse("all", "production")
    vals = sw._runtime_values(sl, sw.representative_cells(sl), rel)
    assert any(v.get("lb_order") == "mirror" for v in vals)
    for v in vals:
        cfg = ra.RockeConfig.from_dict(ra.RockeConfig("c", {}, v).to_dict())
        h = ra.request_hints(cfg.runtime, sl.arch, 0, lambda i: 256)
        assert h["num_cus"] == 256
        assert h["num_xcds"] == v.get("num_xcds", sw.SWEEP_NUM_XCDS[sl.arch])


# The gfx950 bshd quick-check cells and the arm-B knobs of the backward
# timing campaign (DMA ring, vectorised stats with FMA), each at its g_split.
QUICK_CHECK_BSHD = {
    "d128_h64_kv8_b2_q4096_k4096_causal_tl_bf16_bshd": 1,
    "d128_h32_kv32_b4_q2048_k2048_none_bf16_bshd": 1,
    "d128_h32_kv8_b2_q1024_k4096_causal_br_bf16_bshd": 1,
    "d128_h32_kv8_b8_q512_k512_causal_tl_bf16_bshd": 1,
    "d128_h32_kv8_b1_q8192_k8192_causal_tl_bf16_bshd": 4,
}
ARM_B_KNOBS = {
    "acc_in_lds": False, "agpr_alloc": None, "atom_g02": "16x16x32",
    "atom_g13": "16x16x16", "atom_g4": "16x16x32", "block_k4": 32, "block_m": 32,
    "block_n": 128, "edge_tiles": False, "global_path": "dma", "head_pack": False,
    "kt_source": "lds", "kv_residency": "reg", "pt_route": "relabel",
    "ring_depth": 2, "s_in_agpr": False, "sched": "none",
    "scheduler_strategy": None, "setprio": False, "transpose_source": "tr_read",
    "warp_grid_g02": (1, 8), "warp_grid_g13": (8, 1), "warp_grid_g4": (2, 4),
    "waves": 8, "waves_per_eu": 1, "ws_layout": "head_major", "stats_vec": 4,
    "stats_fma": True,
}  # fmt: skip
GFX950_NUM_CUS = 256  # the CU count the gfx950 timing device reports


def test_mirror_plans_on_the_gfx950_quick_check_cells():
    """With the device hints the arm passes, mirror pairing plans on every
    masked bshd quick-check cell (at the cell's g_split, or at g_split = 2
    where the g_split = 1 grid runs in one wave on the device), as the twin of
    the natural-order plan; the unmasked cell is refused for its mask. With
    no hints (the configuration opts out) the plan refuses pairing as before."""
    cells = {c.id: c for c in bb.census_cells(bb.load_census())}
    hints = ra.request_hints({}, "gfx950", 0, lambda i: GFX950_NUM_CUS)
    for cid, gs in QUICK_CHECK_BSHD.items():
        cell = cells[cid]
        req = ra.request_for_cell(cell, _tensors(cell), scale=ra.scale_of(cell))
        hinted = dataclasses.replace(req, **hints)
        for g in (gs, 2):
            mirror = ra.RockeConfig(
                "m", ARM_B_KNOBS, {"g_split": g, "lb_order": "mirror"}
            )
            natural = ra.RockeConfig("n", ARM_B_KNOBS, {"g_split": g})
            if cell.mask == "none":
                with pytest.raises(ValueError, match="mask"):
                    attn_bwd_plan(hinted, "gfx950", policy=ra.config_policy(mirror))
                break
            nat = attn_bwd_plan(hinted, "gfx950", policy=ra.config_policy(natural))
            main = nat.launches[1]
            one_wave = main.grid[0] * main.grid[1] * main.grid[2] <= GFX950_NUM_CUS
            if one_wave:
                assert g == 1, cid
                with pytest.raises(ValueError, match="overfills"):
                    attn_bwd_plan(hinted, "gfx950", policy=ra.config_policy(mirror))
                continue
            p = attn_bwd_plan(hinted, "gfx950", policy=ra.config_policy(mirror))
            assert p.runtime["lb_order"] == "mirror" and p.runtime["pair"] == 1
            spec = p.specs["main"]
            assert dataclasses.replace(spec, lb_mirror=False) == nat.specs["main"]
            assert p.launches[1].grid[0] == math.ceil(main.grid[0] / 2)
            with pytest.raises(ValueError, match="num_cus"):
                attn_bwd_plan(req, "gfx950", policy=ra.config_policy(mirror))
