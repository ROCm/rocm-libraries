# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``kernels.gfx950.qwen38_mono.stage_b3``: the mono DeltaNet B3 stage
(gated delta rule decode) as one rocKE emitter for the isolated-stage
kernel, the hybrid device function and the layer framework. CPU only: knob
validation and names, host models (task map, B1 tiles, LDS arena), and the
emitted LLVM text of every shape x knob value.

GPU proofs (gfx950, q38 container, ATOM present): the Qwen3.8 harness's
``gdn-b3/bench_b3.py`` checks every arm bitwise against the full mono
layer's B3 outputs and state snapshots (``d8`` / ``de`` / ``core`` and every
recurrent-state pool) and times it against the mono's own B3.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import replace

import pytest
from kernels.gfx950.qwen38_mono import abi
from kernels.gfx950.qwen38_mono import stage_b3 as SB
from kernels.gfx950.qwen38_mono import stage_fn as SF
from kernels.gfx950.qwen38_mono.mono_layout import MonoLayout
from rocke.core.ir import uses_python_only_ops
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers import mailbox as M

FLAVOR = "llvm22"
SHAPES = [
    SB.B3Shape(1),
    SB.B3Shape(2),
    SB.B3Shape(3),
    SB.B3Shape(4),
    SB.B3Shape(2, True),
    SB.B3Shape(3, True),
    SB.B3Shape(4, True),
    SB.B3Shape(4, False, 2),
]
KNOB_VALUES = {
    "split": SB.SPLITS,
    "head_map": SB.HEAD_MAPS,
    "wait": SB.WAITS,
    "stores": SB.STORES,
    "state_nt": (False, True),
    "barrier": SB.BARRIERS,
    "shfl4": SB.SHFL4,
    "exp": SB.EXPS,
    "cvt": SB.CVTS,
    "state_issue": SB.STATE_ISSUES,
    "norms": SB.NORMS,
    "core_put": SB.CORE_PUTS,
    "xchg": SB.XCHGS,
    "table": SB.TABLES,
    "hoist": SB.HOISTS,
    "pre": SB.PRES,
}


def _lower(k):
    return _lower_kernel_to_llvm_python(k, llvm_flavor=FLAVOR, arch="gfx950")


def _layout():
    return MonoLayout(
        grid=256, threads=512, max_rows=4, row_pairs=abi.ROW_PAIRS, sc=dict(abi.SC), tasks={}
    )


def test_defaults_are_mono():
    k = SB.B3Knobs()
    assert (k.split, k.head_map, k.wait, k.stores) == (4, "linear", "pre_poll", "early")
    assert k.suffix() == "" and k.bitwise
    assert SB.b3_tasks(k) == 192


def test_bad_knobs():
    for kw in (
        {"split": 3}, {"head_map": "x"}, {"wait": "x"}, {"stores": "x"}, {"barrier": "x"},
        {"shfl4": "x"}, {"exp": "x"}, {"cvt": "x"}, {"split": 1, "stores": "after_put"},
        {"state_issue": "x"}, {"norms": "x"}, {"core_put": "x"}, {"split": 1, "core_put": "direct"},
        {"xchg": "x"}, {"split": 1, "xchg": "numa2"},
    ):  # fmt: skip
        with pytest.raises(ValueError):
            SB.B3Knobs(**kw)


def test_bad_shapes():
    for kw in ({"rows": 0}, {"rows": 5}, {"rows": 1, "seq": True}, {"rows": 4, "group": 3},
               {"rows": 2, "group": 2}, {"rows": 4, "seq": True, "group": 2}):  # fmt: skip
        with pytest.raises(ValueError):
            SB.B3Shape(**kw)
    assert SB.B3Shape(4, True).chain == 4
    assert SB.B3Shape(4, False, 2).chain == 2
    assert SB.B3Shape(3).chain == 1


def test_suffixes_unique():
    seen = {}
    for name, vals in KNOB_VALUES.items():
        for v in vals:
            try:
                k = SB.B3Knobs(**{name: v})
            except ValueError:
                continue
            sfx = k.suffix()
            assert (sfx == "") == (v == getattr(SB.B3Knobs(), name)), (name, v)
            if sfx:
                assert sfx not in seen, (name, v, seen.get(sfx))
                seen[sfx] = (name, v)
    k = SB.B3Knobs(mailbox=M.MailboxKnobs(pause="none"))
    assert "mbnone" in k.suffix()


@pytest.mark.parametrize("split", SB.SPLITS)
@pytest.mark.parametrize("head_map", SB.HEAD_MAPS)
def test_head_map_is_a_bijection(split, head_map):
    k = SB.B3Knobs(split=split, head_map=head_map)
    got = [SB.b3_head(t, k) for t in range(SB.b3_tasks(k))]
    assert sorted(got) == [(vh, s) for vh in range(SB.HV) for s in range(split)]
    if head_map == "xcd":
        # all siblings of a head on CTAs of one XCD (round-robin bid % 8)
        xcd = {}
        for t, (vh, _) in enumerate(got):
            xcd.setdefault(vh, set()).add(t % 8)
        assert all(len(v) == 1 for v in xcd.values())
    if head_map == "linear":
        assert got[:8] == [(t // split, t % split) for t in range(8)]


def test_tiles_match_mono_at_split4():
    # ATOM layout.b3_tiles (split 4)
    for vh in range(SB.HV):
        kh = vh // SB.GQA
        for s in range(4):
            q, k = 2 * kh, SB.KEY // 64 + 2 * kh
            v = 2 * SB.KEY // 64 + 2 * vh + s // 2
            z = SB.QKV // 64 + 2 * vh + s // 2
            assert SB.b3_tiles(kh, vh, s, 4) == (q, q + 1, k, k + 1, v, z)
    assert len(SB.b3_tiles(0, 0, 0, 1)) == 8 and len(SB.b3_tiles(0, 0, 1, 2)) == 6


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_lds_fits_mono_b3_window(rows):
    # mono's B3 LDS entries start after B1's (red, x8, xe) and end at the
    # build's arena end: the rocKE B3 must fit that window in the hybrid path.
    mono_words = {1: (1856, 22640), 2: (3808, 32256), 3: (None, None), 4: (7616, 38912)}
    for split in SB.SPLITS:
        a = SB._arena(SB.B3Shape(rows), SB.B3Knobs(split=split))
        lo, hi = mono_words[rows]
        if lo is not None:
            assert a.words <= hi - lo, (rows, split, a.words)
        if split == 4 and rows > 1:
            # the mono's row strides
            assert a.stride["b3_qk"] == 272 and a.stride["b3_vz"] == 80
            assert a.stride["b3_core"] == 80 and a.stride["b3_sums"] == 80
            assert a.stride["b3_cores"] == 144 and a.stride["b3_bax"] == 3152


def _knob_cases():
    base = SB.B3Knobs()
    out = [base]
    for name, vals in KNOB_VALUES.items():
        for v in vals:
            if v == getattr(base, name):
                continue
            try:
                out.append(replace(base, **{name: v}))
            except ValueError:
                pass
    out.append(SB.B3Knobs(split=1, stores="end", wait="barrier", head_map="xcd"))
    out.append(SB.B3Knobs(split=2, stores="after_put", head_map="xcd", barrier="full"))
    return out


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.variant)
@pytest.mark.parametrize("knobs", _knob_cases(), ids=lambda k: k.suffix() or "default")
def test_kernel_lowers(shape, knobs):
    k = SB.build_b3_kernel(shape, knobs)
    text = _lower(k)
    assert f"define amdgpu_kernel void @{k.name}(" in text
    py = uses_python_only_ops(k)
    assert k.name.endswith("_pyonly") == bool(py)
    assert ("llvm.exp.f32" in text) == (knobs.exp == "llvm")
    # one buffer store per snapshot element per row: 8 x rows x state rows/thread
    nst = 8 * shape.rows * (4 // knobs.split)
    stores = text.count("llvm.amdgcn.raw.ptr.buffer.store")
    assert stores >= nst
    if knobs.state_nt:
        assert "math.exp" in py or py  # python-only either way


def test_default_build_is_python_only_and_named():
    k = SB.build_b3_kernel(SB.B3Shape(4, True))
    assert k.name == "rk_q38_b3_r4s_pyonly"
    assert uses_python_only_ops(k) == ["math.exp"]
    k = SB.build_b3_kernel(SB.B3Shape(1), SB.B3Knobs(exp="hw"))
    assert uses_python_only_ops(k) == [] and k.name == "rk_q38_b3_r1_ehw"


def test_stamps_build():
    k = SB.build_b3_kernel(SB.B3Shape(1), stamps=True)
    text = _lower(k)
    assert k.name.startswith("rk_q38_b3_r1_tl")
    assert "s_memrealtime" in text


@pytest.mark.parametrize("variant", ["r1", "r2", "r3", "r4", "r2s", "r3s", "r4s", "r4g2"])
def test_stage_fn_variants(variant):
    rows, seq, group = SF.parse_variant(variant)
    fn = SF.lower_stage(SF.StageSpec("b3", rows, seq, group), _layout(), flavor=FLAVOR)
    assert fn.symbol == f"rocke_q38_b3_{variant}"
    assert fn.abi[-1] == ("lds_base", "i32")
    assert re.search(r"define void @rocke_q38_b3_" + variant + r"\(", fn.ir)
    # fits the mono B3 LDS window
    assert fn.lds_bytes <= 4 * (38912 - 7616)


def test_stage_fn_unsupported():
    with pytest.raises(ValueError):
        SF.build_stage_kernel(SF.StageSpec("b3", 1, True), _layout())


def test_knobs_change_only_what_they_name():
    """Numerics-preserving knobs leave every float op in place: the count of
    fused multiply-adds, exps and reciprocals is the same in every build of a
    shape (split changes how many rows a thread runs, not the per-row ops)."""
    shape = SB.B3Shape(4, True)

    def counts(k):
        t = _lower(SB.build_b3_kernel(shape, k))
        return tuple(t.count(s) for s in ("llvm.exp.f32(", "amdgcn.rcp.f32(", "amdgcn.rsq.f32("))

    ref = counts(SB.B3Knobs())
    for k in (
        SB.B3Knobs(wait="barrier"), SB.B3Knobs(stores="end"), SB.B3Knobs(head_map="xcd"),
        SB.B3Knobs(barrier="full"), SB.B3Knobs(shfl4="swizzle"),
        SB.B3Knobs(state_issue="entry"), SB.B3Knobs(core_put="direct"),
        SB.B3Knobs(hoist="gates"), SB.B3Knobs(hoist="gates_z"), SB.B3Knobs(table="global"),
        SB.B3Knobs(xchg="numa2"),
    ):  # fmt: skip
        assert counts(k) == ref, k
    # redundant norms: every wave computes the rows' 2 x rows norms (the
    # same ops per norm), instead of one wave per norm
    red = counts(SB.B3Knobs(norms="redundant"))
    assert red[:2] == ref[:2] and red[2] - ref[2] == 2 * shape.rows - 1


def test_framework_stage_spec():
    spec = SB.b3_stage_spec(SB.B3Shape(4, True), SB.B3Knobs(split=1))
    assert spec.name == "b3" and spec.stage.tasks == 48
    assert {e.name for e in spec.lds} >= {"b3_qk", "b3_vz", "b3_core", "b3_baw", "b3_bax"}
    assert "b3_cores" not in {e.name for e in spec.lds}


def test_all_combinations_validate():
    n = 0
    for combo in itertools.product(SB.SPLITS, SB.HEAD_MAPS, SB.WAITS, SB.STORES):
        try:
            SB.B3Knobs(split=combo[0], head_map=combo[1], wait=combo[2], stores=combo[3])
            n += 1
        except ValueError:
            assert combo[0] == 1 and combo[3] == "after_put"
    assert n == 3 * 2 * 2 * 3 - 2 * 2


@pytest.mark.parametrize("variant", ["r1", "r2s", "r3", "r4s", "r4g2"])
@pytest.mark.parametrize("split", SB.SPLITS)
def test_hybrid_function_lds_window(variant, split):
    """The hybrid function's entries start at the mono's B3 window and end
    inside the mono's arena (``lds_base`` is the mono arena's word 0)."""
    rows, seq, group = SF.parse_variant(variant)
    shape = SB.B3Shape(rows, seq, group)
    fn = SB.lower_b3_function(shape, SB.B3Knobs(split=split), _layout(), flavor=FLAVOR)
    assert fn.symbol == f"rocke_q38_b3_{variant}"
    assert fn.lds_bytes // 4 <= SB.MONO_LDS_WORDS[rows]
    a = SB._arena(shape, SB.B3Knobs(split=split), pad=SB.MONO_B3_LDS_START[rows])
    assert min(a.offset[e.name] for e in SB.b3_lds_entries(shape, SB.B3Knobs(split=split))) == (
        SB.MONO_B3_LDS_START[rows]
    )


def test_mono_lds_window_matches_atom():
    """``MONO_B3_LDS_START`` / ``MONO_LDS_WORDS`` against ATOM's layout (the
    ATOM tree at ``$ATOM_ROOT``; skipped without it)."""
    import os

    root = os.environ.get("ATOM_ROOT")
    if not root or not os.path.isdir(root):
        pytest.skip("ATOM_ROOT not set")
    from kernels.gfx950.qwen38_mono.mono_layout import _import_atom

    lay = _import_atom("atom.models.qwen4_exp_mono.kernels.layout", root)
    for rows in (1, 2, 3, 4):
        for seq in (False, True) if rows > 1 else (False,):
            L = lay.for_rows(rows, seq)
            assert L.lds["qk"] == SB.MONO_B3_LDS_START[rows], (rows, seq)
            assert L.lds_words == SB.MONO_LDS_WORDS[rows], (rows, seq)


def test_xchg_region_is_free_scratch():
    """``xchg="numa2"`` lives in the scratch's second half (QSA index scores
    only; a DeltaNet launch never touches it), past every request row's
    mailbox regions and inside ``mailbox_bytes(MAX_ROWS)``."""
    assert SB.XCHG_OFFSET % 16384 == 0
    assert SB.XCHG_OFFSET >= abi.MAX_ROWS * abi.SCRATCH_BYTES
    end = SB.XCHG_OFFSET + abi.MAX_ROWS * SB.XCHG_ROW_PAIRS * M.PAIR_BYTES
    assert end <= abi.mailbox_bytes(abi.MAX_ROWS)
    # every pair index of a row maps into copy 0 of its 16 KiB block
    for p in (0, 1023, 1024, 6143):
        q = (p >> 10) * 2 * SB.XCHG_BLOCK_PAIRS + (p & 1023)
        assert (q * 8) % 16384 < 8192 and q < SB.XCHG_ROW_PAIRS


def test_knob_spec_roundtrip():
    for spec in list(SB.TUNED_SPECS.values()) + ["", "split=1,exp=hw", "mb_pause=none,cvt=intrinsic"]:
        k = SB.knobs_from_spec(spec)
        assert SB.knobs_from_spec(SB.knobs_spec(k)) == k
    assert SB.knobs_from_spec("") == SB.B3Knobs()
    with pytest.raises(ValueError):
        SB.knobs_from_spec("split=3")


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.variant)
def test_tuned_knobs_build_every_variant(shape):
    """Every narrow variant has a tuned set (measured or the fallback) that
    builds as the hybrid function (LDS within the mono's window) and as the
    isolated kernel, bitwise class (no numerics-changing knob)."""
    k = SB.tuned_knobs(shape)
    assert k.bitwise
    for part in ("body", "pre") if k.pre == "dma" else ("body",):
        SB.build_b3_function_kernel(shape, k, _layout(), part)
    _lower(SB.build_b3_kernel(shape, k))


def test_tuned_table_variants_and_defaults():
    assert set(SB.TUNED_SPECS) <= {s.variant for s in SHAPES}
    assert SB.tuned_knobs(SB.B3Shape(1)) == SB.knobs_from_spec(SB.TUNED_SPECS["r1"])
    assert SB.tuned_knobs(SB.B3Shape(3)) == SB.knobs_from_spec(SB.TUNED_SPECS["r3"])
    assert SB.tuned_knobs(SB.B3Shape(4, False, 2)) == SB.knobs_from_spec(SB.TUNED_SPECS["r4g2"])
    assert SB.B3Knobs() == SB.knobs_from_spec("")  # the defaults stay the mono's B3


def test_write_b3_functions(tmp_path):
    shapes = [SB.B3Shape(1), SB.B3Shape(4, True)]
    stems = SB.write_b3_functions(str(tmp_path), shapes, SB.tuned_knobs, _layout(), flavor=FLAVOR)
    for stem in stems:
        assert (tmp_path / f"{stem}.ll").is_file() and (tmp_path / f"{stem}.json").is_file()
    assert "b3_r1" in stems and "b3_r4s" in stems
    assert ("b3pre_r1" in stems) == (SB.tuned_knobs(SB.B3Shape(1)).pre == "dma")
    ir = (tmp_path / "b3_r1.ll").read_text()
    assert "define void @rocke_q38_b3_r1(" in ir
    assert "convergent" in next(ln for ln in ir.splitlines() if ln.startswith("attributes #0"))


def test_write_b3_functions_sidecar_declares_prologue(tmp_path):
    """The body's sidecar says whether it reads its prologue's LDS (ATOM's
    ``extern_stages.resolve`` refuses a ``pre="dma"`` body linked alone);
    body and prologue carry the knob spec they were built with; a stale
    prologue of an earlier ``pre="dma"`` build is removed."""
    import json

    one = [SB.B3Shape(1)]
    dma = SB.knobs_from_spec("pre=dma")
    SB.write_b3_functions(str(tmp_path), one, lambda s: dma, _layout(), flavor=FLAVOR)
    body = json.loads((tmp_path / "b3_r1.json").read_text())
    pre = json.loads((tmp_path / "b3pre_r1.json").read_text())
    assert body["needs"] == ["b3pre"] and body["stage_knobs"] == "pre=dma"
    assert pre["prologue_of"] == "b3" and pre["stage_knobs"] == "pre=dma"
    SB.write_b3_functions(str(tmp_path), one, lambda s: SB.B3Knobs(), _layout(), flavor=FLAVOR)
    body = json.loads((tmp_path / "b3_r1.json").read_text())
    assert body["needs"] == [] and body["stage_knobs"] == ""
    assert not list(tmp_path.glob("b3pre_r1.*"))


def test_cli_refuses_unverified_variants(tmp_path, monkeypatch):
    """The CLI emits only variants with an in-situ GPU check (every narrow
    one and r4g2 today); any other needs ``--unverified``."""
    assert {s.variant for s in SHAPES} <= SB.VERIFIED_VARIANTS
    monkeypatch.setattr(SB, "VERIFIED_VARIANTS", SB.VERIFIED_VARIANTS - {"r4g2"})
    with pytest.raises(SystemExit, match="r4g2"):
        SB.main(["--out", str(tmp_path), "--variants", "r4g2"])


# ------------------------------------------------------------------ knob numa (ATOM's numa=1 mailbox layout)


def test_numa_knob_validation_and_names():
    assert SB.B3Knobs().numa == 0 and "numa" not in SB.B3Knobs().suffix()
    k = SB.knobs_from_spec("numa=1")
    assert k.numa == 1 and k.suffix() == "numa1"
    assert SB.knobs_from_spec(SB.knobs_spec(k)) == k
    for bad in ({"numa": 2}, {"numa_hdr": 256}, {"numa": 1, "numa_hdr": 100}, {"numa": 1, "numa_hdr": -256}):
        with pytest.raises(ValueError):
            SB.B3Knobs(**bad)
    assert SB.B3Knobs(numa=1, numa_hdr=1 << 24).suffix() == f"numa1_nh{1 << 24}"
    with pytest.raises(ValueError):
        SB.B3Knobs().header_offset


def test_numa_header_is_the_fused_moe_s():
    """B3's default NUMA header offset is the one the fused MoE addresses
    (``moe_fused.NUMA_HDR``, itself checked against ATOM's
    ``numa.header_offset`` where ATOM is importable) and ATOM's if present."""
    from kernels.gfx950.qwen38_mono import moe_fused as MF

    assert SB.B3Knobs(numa=1).header_offset == MF.NUMA_HDR
    assert SB.NUMA_PAGE == MF.NUMA_PAGE and SB.NUMA_CHUNK == MF.NUMA_CHUNK
    try:
        from atom.models.qwen4_exp_mono.kernels import numa
    except Exception:  # noqa: BLE001  (no ATOM / FlyDSL on this host)
        return
    assert SB.B3Knobs(numa=1).header_offset == numa.header_offset()


def test_numa_only_in_the_hybrid_function():
    k = SB.B3Knobs(numa=1)
    with pytest.raises(ValueError, match="hybrid"):
        SB.build_b3_kernel(SB.B3Shape(1), k)
    SB.lower_b3_function(SB.B3Shape(1), k, _layout(), flavor=FLAVOR)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.variant)
@pytest.mark.parametrize("xchg", SB.XCHGS)
def test_numa_addresses_the_area(shape, xchg):
    """``numa=1``: one header load (the page mask), every put of a region B3
    publishes twice (the area's two copies; ``xchg="numa2"`` already did for
    the exchange), no plain-scratch address of a region; ``numa=0`` emits the
    plain build exactly (no header load, no area)."""
    split = 2 if xchg == "numa2" else 4
    plain = SB.B3Knobs(split=split, xchg=xchg)
    n1 = replace(plain, numa=1)
    ir0 = SB.lower_b3_function(shape, plain, _layout(), flavor=FLAVOR).ir
    ir1 = SB.lower_b3_function(shape, n1, _layout(), flavor=FLAVOR).ir
    hdr = str(n1.header_offset)
    assert hdr not in ir0 and hdr in ir1
    assert str(n1.header_offset + 256 + SB.NUMA_PAGE - 1) in ir1  # the area base, rounded to 2 MiB

    def stores(ir):
        return len(re.findall(r"global_store_dwordx[24]|store <[24] x i32>", ir))

    # puts doubled: d8 (put2) and de (put) always, and the core put2 with xchg="core"
    extra = 2 if xchg == "numa2" else 3
    assert stores(ir1) - stores(ir0) == extra, (stores(ir0), stores(ir1))
    assert SB.lower_b3_function(shape, replace(n1, numa=0), _layout(), flavor=FLAVOR).ir == ir0


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.variant)
def test_tuned_numa_sets(shape):
    """``TUNED_SPECS_NUMA``: every variant of ``TUNED_SPECS``, knob ``numa=1``,
    bitwise class, builds as the hybrid function (and its prologue)."""
    assert set(SB.TUNED_SPECS_NUMA) == set(SB.TUNED_SPECS)
    k = SB.tuned_knobs(shape, numa=True)
    assert k.numa == 1 and k.bitwise
    for part in ("body", "pre") if k.pre == "dma" else ("body",):
        SB.build_b3_function_kernel(shape, k, _layout(), part)
    assert SB.tuned_knobs(shape).numa == 0  # the plain sets stay plain


def test_write_b3_functions_numa_sidecars(tmp_path):
    """Both sidecars of a numa=1 build declare the layout (ATOM keeps its lever
    only when EVERY linked function is ``numa1``); plain builds declare none."""
    import json

    one = [SB.B3Shape(1)]
    k = SB.knobs_from_spec("pre=dma,numa=1")
    SB.write_b3_functions(str(tmp_path), one, lambda s: k, _layout(), flavor=FLAVOR)
    for stem in ("b3_r1", "b3pre_r1"):
        meta = json.loads((tmp_path / f"{stem}.json").read_text())
        assert meta["mailboxes"] == "numa1" and meta["numa_hdr"] == k.header_offset, stem
        assert meta["stage_knobs"] == SB.knobs_spec(k)
    SB.write_b3_functions(str(tmp_path), one, lambda s: SB.knobs_from_spec("pre=dma"), _layout(), flavor=FLAVOR)
    for stem in ("b3_r1", "b3pre_r1"):
        meta = json.loads((tmp_path / f"{stem}.json").read_text())
        assert "mailboxes" not in meta and "numa_hdr" not in meta
