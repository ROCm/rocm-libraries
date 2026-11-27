# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side checks of the dense-attention decode classes.

A GPU numerical test cannot tell two decodes apart: every decode is a bijection
onto the work space, so any of them computes the right answer. These tests run
each class's real ``grid`` and ``emit_decode`` on Python ints instead, so a
decode that silently covers the work space in a different order, drops or
duplicates items, or bakes a shape value into a runtime-shape body is caught on
the host.

Runtime shapes are checked the way a runtime-shape binary runs: the decode is
emitted for one spec (the compile-time fields) and fed another launch's shape
(and, on the persistent grid, that launch's fast-division kernargs); it must give
the launched spec's baked mapping. The fake builder's comparisons return
``numpy.bool_``, not ``bool``, so a runtime flag takes the ``select`` path.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kernels.common.attention_dense_decode import (
    NONPERSIST_DECODES,
    PERSIST_DECODES,
    DecodeShape,
)
from kernels.gfx942.attention_dense import (
    Gfx942AttentionDenseSpec,
    attention_dense_runtime_args,
)
from kernels.gfx950.attention_dense import (
    Gfx950AttentionDenseSpec,
)
from kernels.gfx950.attention_dense import (
    attention_dense_runtime_args as gfx950_runtime_args,
)


def _sdiv(a, b):
    """i32 ``sdiv``: truncates toward zero (an unselected path may divide a
    negative value; Python's ``//`` would floor it)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


class _IntBuilder:
    """Evaluates the IR-builder integer ops the decodes emit on Python ints, with
    i32 semantics. Comparisons return ``numpy.bool_`` so a runtime flag is not a
    ``bool`` and :meth:`DecodeShape.branch` takes its ``select`` path."""

    const_i32 = staticmethod(int)
    add = staticmethod(lambda a, b: a + b)
    sub = staticmethod(lambda a, b: a - b)
    mul = staticmethod(lambda a, b: a * b)
    div = staticmethod(_sdiv)
    mod = staticmethod(lambda a, b: a - _sdiv(a, b) * b)
    lshr = staticmethod(lambda a, b: (a & 0xFFFFFFFF) >> b)
    umul_hi_i32 = staticmethod(lambda a, b: ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) >> 32)
    smax = staticmethod(max)
    land = staticmethod(lambda a, b: a & b)
    cmp_lt = staticmethod(lambda a, b: np.bool_(a < b))
    cmp_ge = staticmethod(lambda a, b: np.bool_(a >= b))
    cmp_eq = staticmethod(lambda a, b: np.bool_(a == b))
    cmp_ne = staticmethod(lambda a, b: np.bool_(a != b))
    select = staticmethod(lambda c, a, b: a if c else b)


B_ = _IntBuilder()

_SHAPE_FIELDS = ("batch", "seqlen_q", "seqlen_kv", "num_query_heads", "num_kv_heads")


def _baked(spec):
    return DecodeShape(B_, spec)


def _runtime(built, launched, fastdiv=None, extra=None):
    """``built``'s decode fed ``launched``'s shape, as one runtime-shape binary
    serves another launch."""
    values = {f: getattr(launched, f) for f in _SHAPE_FIELDS}
    values["gqa"] = launched.num_queries_per_kv
    values.update(extra or {})
    return DecodeShape(B_, built, values, fastdiv=fastdiv)


def _fastdiv(launched, dividends, runtime_args=attention_dense_runtime_args):
    """The kernel's fast division, from the launch's real host kernargs."""
    args = runtime_args(launched)

    def div(n, d):
        dividends.append(n)
        q = (B_.umul_hi_i32(n, args[f"{d}_magic"]) + n) >> args[f"{d}_shift"]
        return q

    return div


# (Hq, Hkv, B, S): GQA, MHA, and a head count that is not a power of two.
_SHAPES = ((32, 8, 2, 2048), (64, 64, 1, 1024), (40, 8, 3, 512))


def _spec(Hq, Hkv, B, S, causal, **over):
    return Gfx950AttentionDenseSpec(
        batch=B,
        seqlen_q=S,
        seqlen_kv=S,
        num_query_heads=Hq,
        num_kv_heads=Hkv,
        head_size=128,
        causal=causal,
        dtype="fp16",
        **over,
    )


def _block_linear(spec, qb, hq, bt, nqb):
    """Expected hardware dispatch index of the CTA for (qb, hq, bt), per decode.
    Written from each order's definition, independently of the decode code;
    ``blk`` is the query block before any causal reversal."""
    Hq, Hkv, B = spec.num_query_heads, spec.num_kv_heads, spec.batch
    gqa = Hq // Hkv
    name = spec.nonpersist_decode
    if name == "qb_minor":  # digits, fastest first: Q, head, B
        return qb + nqb * (hq + Hq * bt)
    if name == "bt_hkv_minor":  # B, V, G, Q; Q longest-first under causal
        blk = nqb - 1 - qb if spec.causal else qb
        hkv, hql = divmod(hq, gqa)
        return bt + B * (hkv + Hkv * (hql + gqa * blk))
    if name == "hq_minor_swz":
        # arXiv 2511.02132 Fig. 11 per batch, with batch the slowest digit:
        #   head = (w % M)*(Hq/M) + w // (M*nqb),  block = (w % (M*nqb)) // M
        # inverted to w for (hq, blk); query blocks longest-first under causal.
        M = spec.chiplet_num_xcds
        blk = nqb - 1 - qb if spec.causal else qb
        a, c = divmod(hq, Hq // M)
        return a + M * blk + M * nqb * c + Hq * nqb * bt
    raise AssertionError(f"no expected order for nonpersist decode {name!r}")


def _decode_all(decode, spec, make_shape):
    gx, gy, gz = decode.grid(spec)
    out = {}
    for bz, by, bx in itertools.product(range(gz), range(gy), range(gx)):
        d = decode.emit_decode(make_shape(spec), bx, by, bz)
        out[bx + gx * (by + gy * bz)] = (d.qb, d.hq, d.bt)
    return out


# Baked; the gfx950 grid (only seqlen_q is a param); the gfx942 runtime grid.
_GRID_SHAPES = {
    "baked": _baked,
    "seqlen_q": lambda s: DecodeShape(B_, s, {"seqlen_q": s.seqlen_q}),
    "runtime": lambda s: _runtime(s, s),
}


@pytest.mark.parametrize("name", sorted(NONPERSIST_DECODES))
@pytest.mark.parametrize("causal", (True, False))
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("xcds", (8, 4))
def test_nonpersist_decode_linearizes_to_its_documented_order(
    name, causal, shape, xcds
):
    decode = NONPERSIST_DECODES[name]
    spec = _spec(*shape, causal, nonpersist_decode=name, chiplet_num_xcds=xcds)
    nqb = (spec.seqlen_q + spec.block_m - 1) // spec.block_m
    work = nqb * spec.num_query_heads * spec.batch
    for make_shape in _GRID_SHAPES.values():
        items = _decode_all(decode, spec, make_shape)
        assert sorted(items) == list(range(work))
        assert len(set(items.values())) == work, "decode is not a bijection"
        for wid, (qb, hq, bt) in items.items():
            assert _block_linear(spec, qb, hq, bt, nqb) == wid, (name, wid)


# Launches one runtime-shape grid binary serves: other batch, seqlen and heads,
# windows shorter and longer than the keys.
_GRID_LAUNCHES = (
    (32, 8, 2, 4096, 0),
    (64, 64, 1, 1024, 0),
    (40, 8, 3, 512, 512),
    (40, 8, 1, 2048, 512),
    (16, 4, 2, 1024, 0),
)


@pytest.mark.parametrize("name", sorted(NONPERSIST_DECODES))
@pytest.mark.parametrize("window", (0, 512))
def test_nonpersist_decode_reads_the_shape_from_the_runtime_params(name, window):
    """Under a runtime shape one binary serves every shape: the decode built for
    one spec and fed another launch's shape must give that launch's baked
    mapping, not the built spec's (with a window, balancing follows the launch's
    keys)."""
    decode = NONPERSIST_DECODES[name]
    built = _spec(32, 8, 2, 2048, True, sliding_window=window, nonpersist_decode=name)
    for Hq, Hkv, B, S, w in _GRID_LAUNCHES:
        if w != window:
            continue
        launched = dataclasses.replace(
            built,
            batch=B,
            seqlen_q=S,
            seqlen_kv=S,
            num_query_heads=Hq,
            num_kv_heads=Hkv,
        )
        gx, gy, gz = decode.grid(launched)
        for bz, by, bx in itertools.product(range(gz), range(gy), range(gx)):
            got = decode.emit_decode(_runtime(built, launched), bx, by, bz)
            want = decode.emit_decode(_baked(launched), bx, by, bz)
            assert got == want, (name, (Hq, Hkv, B, S), (bx, by, bz))


def _exact_num_persistent(name, Hq, Hkv, B, S):
    """The workgroup count a persist decode needs (the formulas of the gqa_pair
    class docstrings, NQB = S // 256), written out here independently of the decode
    classes; None for an order that fixes no single count (the grid-stride orders,
    and hq_minor_swz, which needs a multiple of the XCD count)."""
    nqb = S // 256
    return {
        "gqa_pair": nqb * Hkv * B,
        "gqa_pair_2phase": nqb * Hq * B // 2,
    }.get(name)


def _persist_specs():
    """Every persist decode on every shape where it is legal, with and without
    causal masking, plus the interleave variant of qb_major."""
    return list(_iter_persist_specs())


def _iter_persist_specs():
    for name in sorted(Gfx950AttentionDenseSpec.supported_persist_decodes() - {"auto"}):
        for shape, causal in itertools.product(_SHAPES, (True, False)):
            Hq, Hkv, B, S = shape
            exact = _exact_num_persistent(name, Hq, Hkv, B, S)
            num_persistent = 256 if exact is None else exact
            for interleave in ((False, True) if name == "qb_major" else (False,)):
                try:
                    spec = _spec(
                        *shape,
                        causal,
                        persistent=True,
                        num_persistent=num_persistent,
                        persist_decode=name,
                        interleave=interleave,
                    )
                except ValueError:
                    continue  # not legal for this shape (e.g. gqa_pair on MHA)
                yield pytest.param(
                    spec,
                    id=f"{name}-{Hq}x{Hkv}-B{B}-S{S}-"
                    f"{'causal' if causal else 'full'}"
                    f"{'-intl' if interleave else ''}",
                )


@pytest.mark.parametrize("spec", _persist_specs())
def test_persist_decode_is_a_bijection(spec):
    decode = PERSIST_DECODES[spec.persist_decode]
    gqa = spec.num_queries_per_kv
    nqb = (spec.seqlen_q + spec.block_m - 1) // spec.block_m
    work = nqb * spec.num_query_heads * spec.batch
    items = set()
    for wi in range(work):
        d = decode.emit_decode(_baked(spec), wi)
        assert 0 <= d.qb < nqb and 0 <= d.hq < spec.num_query_heads
        assert 0 <= d.bt < spec.batch
        assert d.hkv is None or d.hkv == d.hq // gqa
        items.add((d.qb, d.hq, d.bt))
    assert len(items) == work, "decode drops or duplicates work items"


def test_every_persist_decode_is_covered():
    covered = {p.values[0].persist_decode for p in _persist_specs()}
    assert covered == set(PERSIST_DECODES)


@pytest.mark.parametrize("spec", _persist_specs())
def test_required_num_persistent_is_the_count_check_asks_for(spec):
    """gqa_pair and gqa_pair_2phase state the workgroup count their baked shape
    needs (``_exact_num_persistent``); every other persist order fixes no single
    count (None; hq_minor_swz only needs a multiple of the XCD count). ``check``
    accepts the required count and, one workgroup off, names it and the count it
    got."""
    decode = PERSIST_DECODES[spec.persist_decode]
    shape = (spec.num_query_heads, spec.num_kv_heads, spec.batch, spec.seqlen_q)
    required = decode.required_num_persistent(spec)
    assert required == _exact_num_persistent(spec.persist_decode, *shape)
    assert decode.check(spec) is None
    if required is None:
        return
    assert spec.num_persistent == required
    off_by_one = copy.copy(spec)
    object.__setattr__(off_by_one, "num_persistent", required + 1)
    message = decode.check(off_by_one)
    assert message is not None, "check accepted a count one workgroup off"
    assert f"({required}), got {required + 1}" in message


@pytest.mark.parametrize("causal", (True, False))
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_required_num_persistent_depends_on_the_shape_only(causal, shape):
    """The census generator asks a decode about a kernel's own spec, which has
    another order and count: the answer is a formula of the shape, whether or not
    the order is legal for it (gqa_pair on MHA)."""
    spec = _spec(*shape, causal, persistent=True, num_persistent=256)
    for name, decode in PERSIST_DECODES.items():
        assert decode.required_num_persistent(spec) == _exact_num_persistent(
            name, *shape
        ), name


_PROVIDER = Path(__file__).resolve().parents[3]
_PACKS = _PROVIDER / "src" / "tests" / "engines" / "kernel_ingestor_engine" / "packs"
_GENERATOR = _PACKS / "gen_attention_dense_served_set.py"
_PAIR_FLIPS = {"persistent_gqa_pair", "persistent_gqa_pair_2phase"}


def _require_generator():
    """Skip where rocKE is not inside the provider's source tree (an installed test
    tree); in a source checkout a missing generator is a failure, so moving it cannot
    turn its tests into skips."""
    if not (_PROVIDER / "src").is_dir():
        pytest.skip("rocKE is not inside the hip-kernel-provider source tree")
    assert _GENERATOR.is_file(), f"the census generator moved: {_GENERATOR}"


def _load_generator():
    """The provider's census generator, loaded by path."""
    _require_generator()
    spec = importlib.util.spec_from_file_location(
        "gen_attention_dense_served_set_under_test", _GENERATOR
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_catalog(tmp_path, spec):
    """A one-kernel gfx950 catalog (KDP) whose kernel has ``spec``."""
    catalog = tmp_path / "gfx950_attention_dense.kdp.json"
    descriptor = {"name": "k", "kernel_source": {"spec": dataclasses.asdict(spec)}}
    catalog.write_text(json.dumps({"kernelDescriptors": [descriptor]}))
    return catalog


def _table_labels(table_text):
    """The flip labels of the census rows in a generator's output."""
    return {
        row.split('"')[5] for row in table_text.splitlines() if row.startswith("    {")
    }


def test_census_generator_flips_every_persist_decode_the_spec_accepts():
    """The orders come from the spec class's own accepted set (not from the registry
    the generator reads), and each flip sets that order in the spec and in the KMD."""
    gen = _load_generator()
    accepted = Gfx950AttentionDenseSpec.supported_persist_decodes()
    flips = {
        label: flip
        for label, flip in gen._FLIPS["gfx950"].items()
        if label.startswith("persistent_") and label != "persistent_ragged"
    }
    assert set(flips) == {f"persistent_{decode}" for decode in accepted}
    for decode in accepted:
        spec_flips, kmd = flips[f"persistent_{decode}"]
        assert spec_flips == {"persistent": True, "persist_decode": decode}
        assert kmd == (("persistent", 1), ("persist_decode", decode))


@pytest.mark.parametrize(
    "heads, own_order, kept, dropped",
    [
        pytest.param((8, 2), "auto", _PAIR_FLIPS, set(), id="gqa4"),
        # A default-grid kernel that names a pair order: its ``persistent`` flip ends
        # on that order too, so it needs the decode's count as well.
        pytest.param(
            (8, 2),
            "gqa_pair",
            _PAIR_FLIPS | {"persistent"},
            set(),
            id="gqa4-names-gqa_pair",
        ),
        pytest.param(
            (8, 2),
            "gqa_pair_2phase",
            _PAIR_FLIPS | {"persistent"},
            set(),
            id="gqa4-names-gqa_pair_2phase",
        ),
        # MHA: the pair orders are illegal whatever the count, so those flips are left
        # out, and an order that takes any count stays.
        pytest.param((8, 8), "auto", {"persistent_qb_major"}, _PAIR_FLIPS, id="mha"),
    ],
)
def test_census_generator_gives_a_gqa_pair_flip_the_count_its_decode_requires(
    heads, own_order, kept, dropped, tmp_path, monkeypatch, capsys
):
    """The kernel's own count (256) is not the one gqa_pair needs for its shape, yet
    every flip that ends on a pair order is in the table: the count comes from the
    decode. Flips rocKE's spec rejects are left out."""
    gen = _load_generator()
    spec = _spec(*heads, 1, 512, True, persist_decode=own_order)  # two blocks of 256
    catalog = _write_catalog(tmp_path, spec)
    monkeypatch.setattr(sys, "argv", ["gen", f"gfx950={catalog}"])
    gen.main()
    labels = _table_labels(capsys.readouterr().out)
    assert kept <= labels
    assert not dropped & labels


def test_build_driver_runs_the_census_generator_with_the_decode_count(tmp_path):
    """The build's own route (kernel_ingestor_engine/CMakeLists.txt): the table driver
    runs the generator under ``python -I`` after putting rocKE's trees on ``sys.path``
    itself, and lists the rocKE modules the generator imported in the depfile, so a
    change to the decode formulas regenerates the table."""
    _require_generator()
    catalog = _write_catalog(tmp_path, _spec(8, 2, 1, 512, True))  # GQA 4
    table, depfile = tmp_path / "table.inc", tmp_path / "table.inc.d"
    run = subprocess.run(
        [
            sys.executable,
            "-I",
            str(_PACKS / "gen_gfx942_attention_unified_3d_table.py"),
            str(_GENERATOR),
            "--out",
            str(table),
            "--depfile",
            str(depfile),
            f"--generator-arg=gfx950={catalog}",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    assert _PAIR_FLIPS <= _table_labels(table.read_text())
    assert "attention_dense_decode.py" in depfile.read_text()


def test_bt_hkv_minor_follows_its_documented_order():
    """``wi = ((blk*gqa + hql)*Hkv + hkv)*B + bt`` (the class docstring), checked
    independently of the decode's Hq-first split of the head digits."""
    for Hq, Hkv, B, S in _SHAPES + ((48, 6, 2, 1024),):
        spec = _spec(
            Hq,
            Hkv,
            B,
            S,
            False,
            persistent=True,
            num_persistent=256,
            persist_decode="bt_hkv_minor",
        )
        gqa, nqb = Hq // Hkv, S // 256
        for wi in range(nqb * Hq * B):
            d = PERSIST_DECODES["bt_hkv_minor"].emit_decode(_baked(spec), wi)
            hkv, hql = d.hq // gqa, d.hq % gqa
            assert d.hkv == hkv
            assert ((d.qb * gqa + hql) * Hkv + hkv) * B + d.bt == wi


def _persist(spec_cls, Hq, Hkv, B, S, name, causal=True, window=0, np_=256, **over):
    return spec_cls(
        batch=B,
        seqlen_q=S,
        seqlen_kv=S,
        num_query_heads=Hq,
        num_kv_heads=Hkv,
        head_size=128,
        causal=causal,
        dtype="fp16",
        sliding_window=window,
        persistent=True,
        num_persistent=np_,
        persist_decode=name,
        **over,
    )


def _gfx942_persist(*args, **kw):
    return _persist(Gfx942AttentionDenseSpec, *args, **kw)


# (Hq, Hkv, B, S) launches of one persistent binary: NQB even and odd, W below,
# near and far above NP, GQA and MHA, a head count that is not a power of two.
_PERSIST_LAUNCHES = (
    (32, 8, 1, 512),
    (32, 8, 2, 2048),
    (40, 8, 1, 768),
    (64, 64, 1, 1024),
    (16, 4, 3, 1280),
    (48, 8, 4, 4096),
)


@pytest.mark.parametrize(
    "name", sorted(Gfx942AttentionDenseSpec.supported_persist_decodes() - {"auto"})
)
@pytest.mark.parametrize("causal,window", ((True, 0), (False, 0), (True, 1024)))
@pytest.mark.parametrize("interleave", (False, True))
def test_runtime_persist_decode_matches_the_launched_baked_decode(
    name, causal, window, interleave
):
    """One gfx942 persistent binary (built for the first launch) serves every
    launch through the shape params and that launch's fast-division kernargs: it
    must decode exactly as the launch's own baked decode, every fast-division
    dividend must stay within the decode's bound, and the work must be covered
    once."""
    _check_runtime_persist_decode(
        Gfx942AttentionDenseSpec,
        attention_dense_runtime_args,
        name,
        causal,
        window,
        interleave,
    )


# The gfx950 persistent orders on the runtime-shape path: every order that lists
# fast-division divisors (gqa_pair and gqa_pair_2phase need a baked shape).
_GFX950_RUNTIME_PERSIST_DECODES = sorted(
    n
    for n in Gfx950AttentionDenseSpec.supported_persist_decodes() - {"auto"}
    if PERSIST_DECODES[n].fastdiv_divisors
)


@pytest.mark.parametrize("name", _GFX950_RUNTIME_PERSIST_DECODES)
@pytest.mark.parametrize("causal,window", ((True, 0), (False, 0), (True, 1024)))
@pytest.mark.parametrize("interleave", (False, True))
def test_gfx950_runtime_persist_decode_matches_the_launched_baked_decode(
    name, causal, window, interleave
):
    """The gfx950 twin: one runtime-shape persistent binary decodes every launch
    exactly as that launch's own baked decode, through its fast-division kernargs."""
    _check_runtime_persist_decode(
        Gfx950AttentionDenseSpec, gfx950_runtime_args, name, causal, window, interleave
    )


def _check_runtime_persist_decode(spec_cls, runtime_args, name, causal, window, il):
    if il and name != "qb_major":
        pytest.skip("interleave applies to qb_major only")
    decode = PERSIST_DECODES[name]
    built = None
    for Hq, Hkv, B, S in _PERSIST_LAUNCHES:
        try:
            launched = _persist(
                spec_cls, Hq, Hkv, B, S, name, causal, window, interleave=il
            )
        except ValueError:
            continue  # not legal for this launch (e.g. hq_minor_swz, odd NQB)
        assert launched.runtime_shape, (name, (Hq, Hkv, B, S))
        built = built or launched
        nqb = S // launched.block_m
        work = nqb * Hq * B
        dividends = []
        shape = _runtime(
            built,
            launched,
            _fastdiv(launched, dividends, runtime_args),
            {"nqb": nqb, "w": work},
        )
        got = [decode.emit_decode(shape, wi) for wi in range(work)]
        want = [decode.emit_decode(_baked(launched), wi) for wi in range(work)]
        assert got == want, (name, (Hq, Hkv, B, S))
        assert len({(d.qb, d.hq, d.bt) for d in got}) == work
        assert max(dividends, default=0) <= decode.fastdiv_dividend_bound(launched)
    assert built is not None, f"no legal launch for {name}"


# Decodes that require causal masking, so have no window-free causal-free order.
_CAUSAL_ONLY = {"gqa_pair", "gqa_pair_2phase"}


@pytest.mark.parametrize("persistent", (True, False))
def test_sliding_window_orders_blocks_as_without_causal_masking(persistent):
    """A sliding window gives every query block about the same work, so no
    decode balances there (longest-first, fold, pair fold): each orders blocks
    as without causal masking. For qb_major and hkv_major that is the previous
    order (ascending, plain fold). A window at least as long as the keys cuts
    nothing off, so there the orders are the causal ones. The window-free causal
    specs are the control: some decode must still move blocks there."""
    Hq, Hkv, B, S = 32, 8, 2, 4096
    nqb = S // 256
    registry = PERSIST_DECODES if persistent else NONPERSIST_DECODES
    moved = set()
    for name in sorted(set(registry) - _CAUSAL_ONLY):
        kw = (
            dict(persistent=True, num_persistent=256, persist_decode=name)
            if persistent
            else dict(nonpersist_decode=name)
        )
        swa = _spec(Hq, Hkv, B, S, True, sliding_window=512, **kw)
        plain = _spec(Hq, Hkv, B, S, False, **kw)
        full = _spec(Hq, Hkv, B, S, True, **kw)
        long = _spec(Hq, Hkv, B, S, True, sliding_window=S, **kw)
        decode = registry[name]
        if persistent:
            orders = [
                [decode.emit_decode(_baked(s), wi) for wi in range(nqb * Hq * B)]
                for s in (swa, plain, full, long)
            ]
        else:
            orders = [_decode_all(decode, s, _baked) for s in (swa, plain, full, long)]
        assert orders[0] == orders[1], name
        assert orders[3] == orders[2], name
        if orders[2] != orders[1]:
            moved.add(name)
    assert moved, "without a window some decode should balance causal work"


@pytest.mark.parametrize("segments", (2, 4))
def test_split_kv_store_reads_the_head_count_at_runtime(segments):
    """Split-KV writes partial results to a workspace indexed by the head count,
    and ``check_kv_segments`` does not bar runtime shape, so one binary serves
    every head count on that path. The whole kernel is built here rather than
    decoded on ints: the segment store is in the epilogue, past the decode, so
    only the emitted IR shows whether it took the count from the kernarg or from
    the spec it happened to be compiled for. A baked count sends another launch's
    partials to the wrong offsets, and the reduction then reads them back as if
    they were its own, which no single-head numeric test can see."""
    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx950.attention_dense import build_attention_dense

    def spec_for(hq, hkv):
        return Gfx950AttentionDenseSpec(
            batch=2,
            seqlen_q=2048,
            seqlen_kv=2048,
            num_query_heads=hq,
            num_kv_heads=hkv,
            head_size=128,
            causal=True,
            dtype="bf16",
            persistent=False,
            kv_segments=segments,
        )

    heads = ((8, 1), (16, 2), (40, 8))
    specs = [spec_for(hq, hkv) for hq, hkv in heads]
    for (hq, _), spec in zip(heads, specs):
        assert (
            spec.runtime_shape
        ), f"Hq={hq} is not runtime-shape, so this proves nothing"

    keys = {attention_dense_cache_key(s, arch="gfx950") for s in specs}
    assert len(keys) == 1, f"expected one cache key over {heads}, got {len(keys)}"

    irs = {str(build_attention_dense(s, arch="gfx950")) for s in specs}
    assert len(irs) == 1, (
        f"segments={segments}: {heads} share one cache key but emit {len(irs)} "
        "distinct kernels, so a head count is baked into the shared binary"
    )
