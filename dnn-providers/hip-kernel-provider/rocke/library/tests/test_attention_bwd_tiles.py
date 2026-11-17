# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests of the small-tile / occupancy region of the attention backward.

The region: d64 and d128, tile pairs ``kM0 x kN0`` in {16, 32, 64} x {16, 32,
64}, two or four waves, ``waves_per_eu`` 1 to 4, on gfx942 and gfx950. Checked
here: every tile pair and wave count has a legal, built spec (small-tile warp
grids of :mod:`kernels.common._attention_bwd_tiles`); the occupancy hint is
refused only by the register budget; the grid rules keep the shipped messages;
the register estimate of the shipped grids is unchanged; the emitted step has
the MMA counts of the (grid-aware) stage-table model, no cross-wave LDS race,
and on a replicated G0 / G2 grid the P / dS stores sit behind a wave-uniform
branch. Device numerics: ``test_attention_bwd_tiles_numeric.py``.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from kernels.common._attention_bwd_body import BwdTileGeometry
from kernels.common._attention_bwd_sched import StageTable
from kernels.common._attention_bwd_tiles import (
    attn_bwd_occupancy,
    g02_grids,
    g13_grids,
    kv_operand_split,
    small_tile_grids,
)
from kernels.common.attention_bwd import (
    DKV_MODES,
    WAVES_PER_EU_VALUES,
    AttnBwdSpec,
    _not_built,
    attn_bwd_arch_facts,
    build_attn_bwd_main,
    lds_footprint_bytes,
    register_estimates,
    validate_attn_bwd_spec,
)

from .test_attention_bwd_ir import _walk, lds_hazards, step_bodies, traced_spec

ARCHS = ("gfx942", "gfx950")
HEADS = (64, 128)
TILES = (16, 32, 64)
WAVES = (2, 4)
_RESIDENCIES = (("reg", "reg"), ("reg", "lds"), ("lds", "lds"))


def _spec(d, km, kn, w, grids, *, pt="lds", kv=("reg", "reg"), wpe=None, **kw):
    """A bf16 region spec; ``block_k4`` and the G4 atom follow ``kN0``."""
    g02, g13, g4 = grids
    k4 = 32 if kn % 32 == 0 else 16
    atoms = {} if k4 == 32 else {"atom_g4": "16x16x16"}
    return AttnBwdSpec(
        head_size=d, dtype="bf16", waves=w, block_m=km, block_n=kn, block_k4=k4,
        warp_grid_g02=g02, warp_grid_g13=g13, warp_grid_g4=g4, pt_route=pt,
        kv_residency=kv[0], kt_source=kv[1], waves_per_eu=wpe, **atoms, **kw,
    )  # fmt: skip


def _structures(d, km, kn, w):
    """Candidate specs (wpe unset) in preference order: the natural grids on
    the relabel route, then every small-tile grid triple on the LDS route,
    each with the three residencies."""
    grids = small_tile_grids(d, km, kn, w)
    for triple in grids:
        routes = ("relabel", "lds") if triple[:2] == ((1, w), (w, 1)) else ("lds",)
        for pt, kv in itertools.product(routes, _RESIDENCIES):
            yield _spec(d, km, kn, w, triple, pt=pt, kv=kv)


def _first_legal(arch, d, km, kn, w, wpe=None):
    errors = []
    for s in _structures(d, km, kn, w):
        s = dataclasses.replace(s, waves_per_eu=wpe)
        try:
            r = validate_attn_bwd_spec(s, arch)
        except ValueError as ex:
            errors.append(str(ex))
            continue
        if not _not_built(r):
            return r, errors
    return None, errors


_REGION = list(itertools.product(ARCHS, HEADS, TILES, TILES, WAVES))


@pytest.mark.parametrize("arch,d,km,kn,w", _REGION)
def test_every_tile_pair_and_wave_count_is_legal_and_built(arch, d, km, kn, w):
    assert small_tile_grids(d, km, kn, w), "no grid splits the tile"
    r, errors = _first_legal(arch, d, km, kn, w)
    assert r is not None, errors[:4]
    assert (r.block_m, r.block_n, r.waves) == (km, kn, w)
    # the natural grids carry kN0 / W keys per wave: small kN0 needs the others
    if (r.warp_grid_g02, r.warp_grid_g13) == ((1, w), (w, 1)):
        assert kn % (16 * w) == 0
    else:
        assert r.pt_route == "lds"


@pytest.mark.parametrize("arch,d", list(itertools.product(ARCHS, HEADS)))
def test_occupancy_hint_is_refused_only_by_the_register_budget(arch, d):
    """For every region structure legal without a hint, ``waves_per_eu = n``
    is legal exactly when the arch-VGPR + AGPR estimate fits ``512 / n``; the
    region reaches every hint 1 to 4 somewhere."""
    facts = attn_bwd_arch_facts(arch)
    budget = facts.arch_vgprs + facts.agprs
    assert WAVES_PER_EU_VALUES == (1, 2, 3, 4)
    reached = set()
    for km, kn, w in itertools.product(TILES, TILES, WAVES):
        for s in _structures(d, km, kn, w):
            try:
                r = validate_attn_bwd_spec(s, arch)
            except ValueError:
                continue
            est = register_estimates(r, arch)
            total = est["arch_vgpr"] + est["agpr"]
            for n in WAVES_PER_EU_VALUES:
                hinted = dataclasses.replace(s, waves_per_eu=n)
                fits = n == 1 or total <= budget // n
                try:
                    validate_attn_bwd_spec(hinted, arch)
                    legal = True
                except ValueError as ex:
                    assert f"waves_per_eu={n} needs" in str(ex), str(ex)
                    legal = False
                assert legal == fits, (km, kn, w, n, total)
                if legal:
                    reached.add(n)
    assert reached == set(WAVES_PER_EU_VALUES)


def test_the_four_tiles_that_need_a_replicated_grid():
    """A 16 x 16 S tile on two or four waves, and 32 x 16 / 16 x 32 on four:
    no grid of every wave splits them; a replicated G0 / G2 grid does."""
    for km, kn, w in ((16, 16, 2), (16, 16, 4), (32, 16, 4), (16, 32, 4)):
        for d in HEADS:
            triples = small_tile_grids(d, km, kn, w)
            assert triples and all(t[0][0] * t[0][1] < w for t in triples)
    # a tile a full grid splits lists the full grids first
    assert small_tile_grids(128, 32, 32, 4)[0][0] == (2, 2)
    assert small_tile_grids(128, 64, 64, 4)[0] == ((1, 4), (4, 1), (1, 4))


def test_grid_catalogs():
    assert g02_grids(4) == ((1, 4), (4, 1), (2, 2), (1, 2), (2, 1), (1, 1))
    assert g13_grids(4) == ((4, 1), (1, 4), (2, 2))
    assert g02_grids(2) == ((1, 2), (2, 1), (1, 1))
    assert g13_grids(2) == ((2, 1), (1, 2))
    assert g02_grids(1) == ((1, 1),)
    # every catalog grid holds W waves (G1 / G3) or a divisor of W (G0 / G2)
    for w in (1, 2, 4, 8):
        assert all(r * c == w for r, c in g13_grids(w))
        assert all(w % (r * c) == 0 for r, c in g02_grids(w))


def _w4(**kw):
    base = dict(
        head_size=64, dtype="bf16", waves=4, block_m=32, block_n=32, block_k4=32,
        warp_grid_g4=(2, 2), pt_route="lds",
    )  # fmt: skip
    base.update(kw)
    return AttnBwdSpec(**base)


def test_grid_rules_and_messages():
    # natural grids keep the kN0 rule and its message
    with pytest.raises(ValueError, match="must be a multiple of 16\\*waves=64"):
        validate_attn_bwd_spec(_w4(pt_route="relabel"), "gfx942")
    # so does a stage table, whatever the grids
    with pytest.raises(ValueError, match="must be a multiple of 16\\*waves=64"):
        validate_attn_bwd_spec(
            _w4(warp_grid_g02=(2, 2), warp_grid_g13=(2, 2), sched="stage_table"),
            "gfx942",
        )
    # the small-tile grids split 32 x 32 over four waves
    validate_attn_bwd_spec(_w4(warp_grid_g02=(2, 2), warp_grid_g13=(2, 2)), "gfx942")
    validate_attn_bwd_spec(_w4(warp_grid_g02=(1, 2), warp_grid_g13=(1, 4)), "gfx942")
    # ... only on the LDS route
    with pytest.raises(ValueError, match="needs pt_route='lds'"):
        validate_attn_bwd_spec(
            _w4(pt_route="relabel", warp_grid_g02=(1, 2), warp_grid_g13=(2, 2)),
            "gfx942",
        )
    # a natural G0 / G2 grid next to a small-tile G1 / G3 grid must still split
    with pytest.raises(ValueError, match="warp_grid_g02=\\(1, 4\\) does not divide"):
        validate_attn_bwd_spec(_w4(warp_grid_g13=(2, 2)), "gfx942")
    with pytest.raises(ValueError, match="warp_grid_g13=\\(4, 1\\) does not divide"):
        validate_attn_bwd_spec(_w4(warp_grid_g02=(2, 2)), "gfx942")
    # G1 / G3 is never replicated; grids that are no grid of W waves
    for grids in (
        dict(warp_grid_g02=(2, 2), warp_grid_g13=(1, 2)),
        dict(warp_grid_g02=(4, 4), warp_grid_g13=(2, 2)),
    ):
        with pytest.raises(ValueError, match="must be"):
            validate_attn_bwd_spec(_w4(**grids), "gfx942")
    # a replicated grid must still split its tile into whole atoms
    with pytest.raises(ValueError, match="does not divide its 32 x 16 tile"):
        validate_attn_bwd_spec(
            _w4(block_n=16, block_k4=16, atom_g4="16x16x16", warp_grid_g02=(1, 2),
                warp_grid_g13=(1, 4)),
            "gfx942",
        )  # fmt: skip
    # a stage table needs a G0 / G2 grid of every wave
    with pytest.raises(ValueError, match="stage-table count model needs"):
        validate_attn_bwd_spec(
            _w4(block_n=64, warp_grid_g02=(1, 2), sched="stage_table"), "gfx942"
        )


def _old_estimate(r, arch):
    """The register estimate as written before the small-tile grids."""
    from kernels.common.attention_bwd import (
        STAGE_TABLE_SCHEDS,
        _REG_ALLOWANCE_FIXED,
        _accumulator_terms,
        _agpr_placement,
        _atom_k,
    )

    facts = attn_bwd_arch_facts(arch)
    wave, w = facts.wave_size, r.waves
    d, km, kn = r.head_size, r.block_m, r.block_n
    half = 2 * wave * w
    vgpr = _REG_ALLOWANCE_FIXED
    if r.kv_residency == "reg":
        vgpr += 2 * (kn * d // half)
    if r.kt_source == "reg":
        vgpr += kn * d // half
    vgpr += 2 * km * d // (2 * wave)
    if r.global_path == "vgpr" and (
        r.ring_depth >= 2 or (arch == "gfx942" and r.sched in STAGE_TABLE_SCHEDS)
    ):
        vgpr += km * d // (wave * w)
    vgpr += km * kn // (wave * w)
    vgpr += 2 * km * _atom_k(r.atom_g4) // (2 * wave)
    acc = 0 if r.acc_in_lds else 2 * (kn * d // (wave * w))
    terms = (acc, 2 * km * kn // (wave * w), km * d // (wave * w))
    assert terms == _accumulator_terms(r, arch)
    form = facts.matrix_path == "mfma" and r.agpr_alloc != (0, 0)
    placed = _agpr_placement(r, arch, agpr_form=form)
    agpr = 0
    for term, name in zip(terms, ("acc", "s_dp", "dq")):
        if name in placed:
            agpr += term
        else:
            vgpr += term
    return {"arch_vgpr": vgpr, "agpr": agpr}


@pytest.mark.parametrize("arch", ARCHS)
def test_estimate_of_the_shipped_grids_is_unchanged(arch):
    """(1, W) / (W, 1) and (2, W / 2) keep the shipped register estimate (so
    the AGPR policy and the legality of every existing instance stay put)."""
    n = 0
    for d, km, kn, w in itertools.product(HEADS, TILES, (32, 64, 128), (2, 4, 8)):
        if kn % (16 * w):
            continue
        alt = (2, w // 2)
        for g02, g13 in (((1, w), (w, 1)), (alt, (w, 1)), ((1, w), alt), (alt, alt)):
            pt = "relabel" if (g02, g13) == ((1, w), (w, 1)) else "lds"
            for kv in _RESIDENCIES:
                s = _spec(d, km, kn, w, (g02, g13, (1, w)), pt=pt, kv=kv)
                try:
                    r = validate_attn_bwd_spec(s, arch)
                except ValueError:
                    continue
                assert register_estimates(r, arch) == _old_estimate(r, arch)
                n += 1
    assert n > 100


def test_small_tile_estimate_terms():
    """K / V per wave: the keys of the G0 / G2 grid's columns; S / dP: the
    tile over the grid's waves (replicated grids hold more per wave)."""
    assert kv_operand_split((1, 4), 4) == 4 and kv_operand_split((2, 2), 4) == 4
    assert kv_operand_split((4, 1), 4) == 1 and kv_operand_split((1, 2), 4) == 2
    assert kv_operand_split((1, 1), 2) == 1
    arch = "gfx942"
    full = validate_attn_bwd_spec(
        _w4(warp_grid_g02=(2, 2), warp_grid_g13=(1, 4), agpr_alloc=(0, 0)), arch
    )
    one = validate_attn_bwd_spec(dataclasses.replace(full, warp_grid_g02=(1, 1)), arch)
    a, b = register_estimates(full, arch), register_estimates(one, arch)
    # (1, 1): K and V whole (32 keys, not 32 / 4), S / dP / P / dS whole
    kv = 2 * (32 * 64 // (2 * 64)) - 2 * (32 * 64 // (2 * 64 * 4))
    sdp = 3 * 32 * 32 // 64 - 3 * 32 * 32 // (64 * 4)
    assert b["arch_vgpr"] - a["arch_vgpr"] == kv + sdp


def test_stage_table_counts_follow_the_grids():
    for arch, d, (km, kn, w) in itertools.product(
        ARCHS, HEADS, ((16, 16, 2), (16, 16, 4), (32, 16, 4), (64, 16, 4), (32, 32, 4))
    ):
        r, _ = _first_legal(arch, d, km, kn, w)
        g = BwdTileGeometry.from_spec(r, arch)
        st = g.stage_table()
        assert (st.grid_g02, st.grid_g13) == (g.grid_g0, g.grid_g1)
        assert st.mfma.g0 == g.g0.mma_per_wave and st.mfma.g1 == g.g1.mma_per_wave
        assert st.mfma.g4 == g.g4.mma_per_wave
    with pytest.raises(ValueError, match="stage table needs a G0 grid of every wave"):
        StageTable(
            arch="gfx942", head_size=64, k_m0=32, k_n0=64, k_k4=32, waves=4,
            grid_g02=(1, 2), sched="stage_table",
        )  # fmt: skip
    with pytest.raises(ValueError, match="kN0 16 % \\(16 \\* waves 4\\)"):
        StageTable(
            arch="gfx942", head_size=64, k_m0=64, k_n0=16, k_k4=16, waves=4,
            grid_g02=(4, 1), grid_g13=(1, 4), sched="stage_table",
        )  # fmt: skip
    # without a stage table the small tile is fine
    StageTable(
        arch="gfx942", head_size=64, k_m0=64, k_n0=16, k_k4=16, waves=4,
        grid_g02=(4, 1), grid_g13=(1, 4), sched="none",
    )  # fmt: skip


def test_geometry_rejects_grids_of_the_wrong_wave_count():
    r, _ = _first_legal("gfx942", 64, 32, 32, 4)
    for field, grid in (("warp_grid_g02", (2, 4)), ("warp_grid_g13", (1, 2))):
        bad = dataclasses.replace(r, **{field: grid})
        with pytest.raises(ValueError, match="waves"):
            BwdTileGeometry.from_spec(bad, "gfx942")
    relabel = dataclasses.replace(r, pt_route="relabel", warp_grid_g02=(1, 1))
    with pytest.raises(ValueError):
        BwdTileGeometry.from_spec(relabel, "gfx942")


# Emitted kernels: one per grid family (replicated (1, 1) / (2, 1) / (1, 2),
# transposed (4, 1) and (2, 1), the (2, 2) grid on 32 x 32 / 4 waves).
_EMIT = (
    ("gfx942", 128, 16, 16, 2),
    ("gfx950", 64, 16, 16, 4),
    ("gfx942", 64, 32, 16, 4),
    ("gfx950", 128, 16, 32, 4),
    ("gfx942", 128, 64, 16, 4),
    ("gfx950", 64, 32, 16, 2),
    ("gfx942", 64, 32, 32, 4),
)


def _emit_spec(arch, d, km, kn, w):
    triple = small_tile_grids(d, km, kn, w)[0]
    for kv in _RESIDENCIES:
        s = _spec(d, km, kn, w, triple, kv=kv)
        try:
            validate_attn_bwd_spec(s, arch)
            return s
        except ValueError:
            continue
    raise AssertionError((arch, d, km, kn, w))


def _derives_from(value, name):
    seen, todo = set(), [value]
    while todo:
        v = todo.pop()
        if id(v) in seen or v.op is None:
            continue
        seen.add(id(v))
        if v.op.name == name:
            return True
        todo.extend(v.op.operands)
    return False


@pytest.mark.parametrize("arch,d,km,kn,w", _EMIT)
def test_emitted_step_counts_races_and_replicated_stores(arch, d, km, kn, w):
    spec = _emit_spec(arch, d, km, kn, w)
    r = validate_attn_bwd_spec(spec, arch)
    geom = BwdTileGeometry.from_spec(r, arch)
    kernel, tags = traced_spec(spec, arch)
    assert not lds_hazards(kernel, tags)[:4]
    m = geom.stage_table().mfma
    replicated = geom.g0.waves < geom.waves
    for ops in step_bodies(kernel):
        mmas = [op for op, _ in _walk(ops) if op.name == "tile.mma"]
        assert len(mmas) == m.g0 + m.g2 + m.g1 + m.g3 + m.g4
        # every matrix op runs on every wave (no MMA under a branch)
        for op, anc in _walk(ops):
            if op.name == "tile.mma":
                assert not any(a.name.startswith("scf.if") for a in anc)
        guarded = [
            op
            for op, anc in _walk(ops)
            if op.name.startswith("tile.smem_store")
            and any(
                a.name == "scf.if"
                and _derives_from(a.operands[0], "tile.readfirstlane")
                for a in anc
            )
        ]
        # P and dS: one store per accumulator slot of the wave's G0 fragments
        slots = 2 * geom.g0.frags_m * geom.g0.frags_n * geom.g0.op.c_frag_len
        assert len(guarded) == (slots if replicated else 0), (len(guarded), slots)


@pytest.mark.parametrize("arch,d,km,kn,w", _EMIT)
def test_replicated_waves_address_inside_the_tile(arch, d, km, kn, w, monkeypatch):
    """On a replicated G0 / G2 grid of ``k`` waves the G0 origin is taken
    from ``wave mod k`` (the readfirstlane'd wave index), so a wave ``w >= k``
    computes the sub-block of wave ``w mod k`` and its operand reads and lse2
    / Dsum row loads stay inside the tile (it never stores: the device
    numerics cannot tell, this pins it). Full grids use the wave index."""
    from kernels.common import _attention_bwd_gemm as gemm

    spec = _emit_spec(arch, d, km, kn, w)
    r = validate_attn_bwd_spec(spec, arch)
    g02 = tuple(r.warp_grid_g02)
    k = g02[0] * g02[1]
    seen = []
    orig = gemm.BlockGemm.wave_origin

    def spy(self, b, wave):
        if (self.m, self.n, self.k) == (km, kn, d):
            seen.append(wave)
        return orig(self, b, wave)

    monkeypatch.setattr(gemm.BlockGemm, "wave_origin", spy)
    build_attn_bwd_main(spec, arch=arch)
    assert seen
    for wave in seen:
        if k < w:
            op = wave.op
            assert op is not None and op.name == "arith.mod", op
            src, mod = op.operands
            assert src.op is not None and src.op.name == "tile.readfirstlane"
            assert mod.op.name == "arith.constant" and mod.op.attrs["value"] == k
        else:
            assert wave.op is not None and wave.op.name == "tile.readfirstlane"


def test_shipped_kernels_have_no_replicated_store_branch():
    """The default instance emits no wave branch around its P / dS stores
    (the replicated path is only taken on a replicated grid)."""
    for arch in ARCHS:
        spec = AttnBwdSpec(head_size=64, dtype="bf16", pt_route="lds")
        kernel = build_attn_bwd_main(spec, arch=arch)
        for ops in step_bodies(kernel):
            for op, anc in _walk(ops):
                if op.name.startswith("tile.smem_store"):
                    assert not any(
                        a.name == "scf.if"
                        and _derives_from(a.operands[0], "tile.readfirstlane")
                        for a in anc
                    )


@pytest.mark.parametrize("arch", ARCHS)
def test_device_sets_are_legal_and_built(arch):
    """Every device set of the region is legal and built at each of its head
    sizes (both dtypes, direct and atomic dK / dV); the bitwise groups share
    the tile, G4 grid, route, residency and hint and differ in G0 / G2 or G1
    / G3 only; the occupancy sets fit their hint by the register estimate and
    by LDS. Legality is the static rules' (a set the production compiler
    refuses on resources, ``RESOURCE_REFUSED``, raises from the shipped
    validator and is skipped by the device test with that reason); every set
    keeps at least one cell the shipped validator accepts."""
    from kernels.common._attention_bwd_refusals import resource_refusal

    from .sdpa.bwd_tile_region_sets import (
        BITWISE_GROUPS,
        REGION_CASES,
        REGION_SETS,
        region_knobs,
    )

    for name, row in REGION_SETS.items():
        accepted = 0
        for d, dt, mode in itertools.product(row[-1], ("fp16", "bf16"), DKV_MODES):
            spec = AttnBwdSpec(
                head_size=d, dtype=dt, dkv_mode=mode, **region_knobs(name)
            )
            r = validate_attn_bwd_spec(spec, arch, main_kernel_refusals=False)
            if resource_refusal(r, arch) is None:
                accepted += 1
                validate_attn_bwd_spec(spec, arch)
            else:
                with pytest.raises(ValueError, match="resource gate"):
                    validate_attn_bwd_spec(spec, arch)
            assert not _not_built(r), (name, d)
            if r.waves_per_eu > 1:
                occ = attn_bwd_occupancy(r, arch)
                assert occ.by_registers >= r.waves_per_eu, (name, d, occ)
                assert occ.hint_reachable, (name, d, occ)
        assert accepted, (name, arch, "every cell of the set is refused")
        assert set(row[-1]) <= set(REGION_CASES)
    for group in BITWISE_GROUPS:
        rows = [REGION_SETS[n] for n in group]
        keys = {(r[0], r[1], r[2], r[5], r[6], r[7], r[8], r[9]) for r in rows}
        assert len(keys) == 1, group
        assert len({(r[3], r[4]) for r in rows}) == len(rows), group


def test_occupancy_bound():
    """Whole-workgroup waves per SIMD by the register estimate and by LDS."""
    arch = "gfx942"
    small = validate_attn_bwd_spec(_emit_spec(arch, 64, 16, 16, 2), arch)
    occ = attn_bwd_occupancy(small, arch)
    est = register_estimates(small, arch)
    used = -(-(est["arch_vgpr"] + est["agpr"]) // 8) * 8
    per_simd = min(8, 512 // used)
    assert occ.by_registers == max(1, (per_simd * 4 // 2) * 2 // 4)
    lds = lds_footprint_bytes(small, arch)
    assert occ.by_lds == min(8, max(1, (65536 // lds) * 2 // 4))
    assert occ.waves == min(occ.by_registers, occ.by_lds)
    # the CDNA policy resolves the hint to 1 (the "1,1" attribute)
    assert occ.requested == 1 and occ.hint_reachable
    for n in (2, 3, 4):
        h = attn_bwd_occupancy(dataclasses.replace(small, waves_per_eu=n), arch)
        assert h.hint_reachable is (occ.by_lds >= n)
    none = attn_bwd_occupancy(dataclasses.replace(small, waves_per_eu=None), arch)
    assert none.requested is None and none.hint_reachable
    # the d128 64 x 64 four-wave tile fills more than half of gfx942 LDS (the
    # model only: the validator refuses this tile, a recorded resource-gate
    # failure, so it is resolved here, not validated)
    big = AttnBwdSpec(
        head_size=128, waves=4, block_m=64, block_n=64, block_k4=32,
        warp_grid_g4=(2, 2),
    ).resolved(arch)  # fmt: skip
    assert lds_footprint_bytes(big, arch) * 2 > 65536
    hinted = attn_bwd_occupancy(dataclasses.replace(big, waves_per_eu=2), arch)
    assert hinted.by_lds == 1 and not hinted.hint_reachable
    with pytest.raises(ValueError, match="MFMA targets only"):
        attn_bwd_occupancy(AttnBwdSpec(head_size=64).resolved("gfx1151"), "gfx1151")
