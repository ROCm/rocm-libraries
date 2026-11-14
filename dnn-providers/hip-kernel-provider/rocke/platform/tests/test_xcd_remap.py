# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``rocke.helpers.grid.xcd_remap`` and friends (CPU only):

* every mode is a bijection of the grid, and of each batch plane for the
  per-plane form, over a parametrised ``(num_xcds, chunk, grid)`` sweep;
* the host evaluator ``python_xcd_remap`` equals frozen copies of the three
  implementations this helper replaced (the attention-backward gated remap,
  the GDN decode contiguous remap, and ``chiplet_transform_chunked_runtime``);
* the emitted IR, evaluated op by op, equals the host evaluator;
* the emitted op sequence is identical to the frozen copies', which is what
  keeps the kernels that moved onto the helper byte-identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PLATFORM_PY = str(Path(__file__).resolve().parents[1] / "python")
if _PLATFORM_PY not in sys.path:
    sys.path.insert(0, _PLATFORM_PY)

from rocke.core.ir import I32, IRBuilder
from rocke.helpers.grid import (
    chiplet_transform_chunked,
    chiplet_transform_chunked_dynamic,
    chiplet_transform_chunked_runtime,
    python_chiplet_transform_chunked_runtime,
    python_xcd_remap,
    xcd_remap,
    xcd_remap_plane,
)

# --------------------------------------------------------------------------- frozen oracles
# Verbatim copies of the implementations replaced by ``xcd_remap``: the
# attention-backward ``xcd_remap_id`` and the GDN decode ``_xcd_remap``.


def _old_gated_remap(b, wgid, *, num_wgs, n, chunk):
    zero, one = b.const_i32(0), b.const_i32(1)
    on = b.land(b.cmp_gt(n, zero), b.cmp_gt(chunk, zero))
    n1 = b.select(on, n, one)
    c1 = b.select(on, chunk, one)
    block = b.mul(n1, c1)
    limit = b.mul(b.div(num_wgs, block), block)
    xcd = b.mod(wgid, n1)
    local = b.div(wgid, n1)
    new = b.add(b.add(b.mul(b.div(local, c1), block), b.mul(xcd, c1)), b.mod(local, c1))
    return b.select(b.land(on, b.cmp_lt(wgid, limit)), new, wgid)


def _old_gated_remap_host(wgid, num_wgs, n, chunk):
    if n <= 0 or chunk <= 0:
        return wgid
    block = n * chunk
    if wgid >= (num_wgs // block) * block:
        return wgid
    local = wgid // n
    return (local // chunk) * block + (wgid % n) * chunk + local % chunk


def _old_gdn_remap(b, pid, npid):
    n = b.const_i32(8)
    xcd = b.mod(pid, n)
    tall = b.add(b.mod(b.sub(npid, b.const_i32(1)), n), b.const_i32(1))
    per = b.div(b.add(npid, b.const_i32(8 - 1)), n)
    short = b.smax(b.sub(xcd, tall), b.const_i32(0))
    return b.sub(b.add(b.div(pid, n), b.mul(xcd, per)), short)


def _old_gdn_remap_host(pid, npid):
    xcd = pid % 8
    tall = ((npid - 1) % 8) + 1
    per = (npid + 7) // 8
    return pid // 8 + xcd * per - max(xcd - tall, 0)


# --------------------------------------------------------------------------- IR helpers


def _signature(kernel):
    """The op stream with value names: equal signatures lower to equal IR."""
    return [
        (
            op.name,
            tuple(v.name for v in op.operands),
            tuple(v.name for v in op.results),
            tuple(sorted((k, repr(v)) for k, v in op.attrs.items())),
        )
        for op in kernel.body.ops
    ]


def _eval(kernel, env):
    """Evaluate the straight-line i32/i1 scalar ops of ``kernel`` (the remap's
    whole vocabulary) with ``env`` binding its input values."""

    def div(a, b):
        return int(a / b)  # sdiv truncates towards zero

    binop = {
        "arith.add": lambda a, b: a + b,
        "arith.sub": lambda a, b: a - b,
        "arith.mul": lambda a, b: a * b,
        "arith.div": div,
        "arith.mod": lambda a, b: a - div(a, b) * b,
        "arith.smax": max,
        "arith.and": lambda a, b: a and b,
    }
    cmp = {"lt": lambda a, b: a < b, "gt": lambda a, b: a > b}
    vals = dict(env)
    for op in kernel.body.ops:
        if op.results and op.results[0].name in vals:
            continue
        ins = [vals.get(v.name) for v in op.operands]
        if op.name == "arith.constant":
            vals[op.result.name] = op.attrs["value"]
        elif op.name in binop:
            vals[op.result.name] = binop[op.name](*ins)
        elif op.name == "arith.cmp":
            vals[op.result.name] = cmp[op.attrs["pred"]](*ins)
        elif op.name == "arith.select":
            vals[op.result.name] = ins[1] if ins[0] else ins[2]
        else:
            raise AssertionError(f"op outside the remap vocabulary: {op.name}")
    return vals


def _kernel_1d(emit, params):
    """A kernel whose one hardware id is ``block_id_x`` and whose ``params``
    are i32 kernargs; ``emit(b, wgid, **param_values)`` returns the remap."""
    b = IRBuilder("xcd_remap_test")
    values = {name: b.param(name, I32) for name in params}
    wgid = b.block_id_x()
    out = emit(b, wgid, **values)
    return b.kernel, wgid, values, out


# --------------------------------------------------------------------------- bijection sweeps

GRIDS = list(range(1, 130)) + [255, 256, 513, 1000, 1024, 4097]


@pytest.mark.parametrize("num_xcds", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("chunk", [None, 1, 2, 3, 7, 64])
def test_chunked_is_a_bijection_of_the_grid(num_xcds, chunk):
    for num_wgs in GRIDS:
        image = [
            python_xcd_remap(w, num_wgs=num_wgs, num_xcds=num_xcds, chunk=chunk)
            for w in range(num_wgs)
        ]
        assert sorted(image) == list(range(num_wgs)), (num_wgs, num_xcds, chunk)


@pytest.mark.parametrize("num_xcds", [1, 2, 3, 4, 8, 16])
def test_contiguous_is_a_bijection_and_gives_each_xcd_one_run(num_xcds):
    for npid in GRIDS:
        image = [
            python_xcd_remap(w, num_wgs=npid, num_xcds=num_xcds, mode="contiguous")
            for w in range(npid)
        ]
        assert sorted(image) == list(range(npid)), (npid, num_xcds)
        for x in range(num_xcds):  # the ids one XCD receives are consecutive
            ids = [image[w] for w in range(x, npid, num_xcds)]
            assert ids == list(range(ids[0], ids[0] + len(ids))) if ids else True


@pytest.mark.parametrize("num_xcds", [0, 1, 3, 8])
@pytest.mark.parametrize("chunk", [0, 1, 4, 64])
def test_gated_is_a_bijection_or_the_identity(num_xcds, chunk):
    for num_wgs in GRIDS:
        image = [
            python_xcd_remap(
                w, num_wgs=num_wgs, num_xcds=num_xcds, chunk=chunk, mode="gated"
            )
            for w in range(num_wgs)
        ]
        assert sorted(image) == list(range(num_wgs)), (num_wgs, num_xcds, chunk)
        if num_xcds <= 0 or chunk <= 0:
            assert image == list(range(num_wgs))


@pytest.mark.parametrize("gx", [1, 3, 8, 17])
@pytest.mark.parametrize("gy", [1, 2, 5, 16])
@pytest.mark.parametrize("n,chunk", [(0, 0), (2, 3), (4, 1), (8, 4), (8, 64), (3, 0)])
def test_per_plane_remap_is_a_bijection_of_each_plane(gx, gy, n, chunk):
    plane = gx * gy
    image = [
        python_xcd_remap(p, num_wgs=plane, num_xcds=n, chunk=chunk, mode="gated")
        for p in range(plane)
    ]
    assert sorted(image) == list(range(plane))


# --------------------------------------------------------------------------- host evaluators = frozen oracles


@pytest.mark.parametrize(
    "n,chunk", [(8, 1), (8, 4), (2, 3), (4, 16), (8, 64), (0, 4), (4, 0)]
)
def test_gated_host_model_equals_the_replaced_attention_bwd_remap(n, chunk):
    for num_wgs in GRIDS:
        for w in range(0, num_wgs, 1 if num_wgs < 200 else 13):
            assert python_xcd_remap(
                w, num_wgs=num_wgs, num_xcds=n, chunk=chunk, mode="gated"
            ) == _old_gated_remap_host(w, num_wgs, n, chunk)


def test_contiguous_host_model_equals_the_replaced_gdn_remap():
    for npid in GRIDS:
        for w in range(npid):
            assert python_xcd_remap(
                w, num_wgs=npid, num_xcds=8, mode="contiguous"
            ) == _old_gdn_remap_host(w, npid)


def test_chunked_host_model_equals_the_runtime_platform_helper_model():
    for num_wgs in GRIDS:
        for chunk in (None, 1, 5, 64):
            for w in range(0, num_wgs, 1 if num_wgs < 200 else 13):
                assert python_xcd_remap(
                    w, num_wgs=num_wgs, num_xcds=8, chunk=chunk
                ) == python_chiplet_transform_chunked_runtime(
                    w, num_wgs=num_wgs, num_xcds=8, chunk_size=chunk
                )


def test_contiguous_differs_from_chunked_off_a_multiple_of_the_xcd_count():
    # why GDN decode needs its own mode: the chunked remap leaves a tail at
    # identity, the contiguous one moves every id
    npid = 13
    chunked = [
        python_xcd_remap(w, num_wgs=npid, num_xcds=8, chunk=npid // 8)
        for w in range(npid)
    ]
    contiguous = [
        python_xcd_remap(w, num_wgs=npid, num_xcds=8, mode="contiguous")
        for w in range(npid)
    ]
    assert chunked != contiguous


# --------------------------------------------------------------------------- IR = host model


@pytest.mark.parametrize("mode", ["gated", "contiguous"])
def test_emitted_ir_equals_the_host_model(mode):
    if mode == "gated":
        kernel, wgid, p, out = _kernel_1d(
            lambda b, w, num_wgs, n, chunk: xcd_remap(
                b, w, num_wgs=num_wgs, num_xcds=n, chunk=chunk, mode="gated"
            ),
            ["num_wgs", "n", "chunk"],
        )
        cases = [
            (num_wgs, n, chunk)
            for num_wgs in (1, 5, 8, 9, 63, 64, 130, 513)
            for n, chunk in ((8, 1), (8, 4), (3, 5), (4, 64), (0, 4), (4, 0), (-1, 2))
        ]
    else:
        kernel, wgid, p, out = _kernel_1d(
            lambda b, w, npid: xcd_remap(
                b, w, num_wgs=npid, num_xcds=8, mode="contiguous"
            ),
            ["npid"],
        )
        cases = [(npid,) for npid in (1, 5, 8, 9, 13, 63, 64, 130, 1000)]
    for case in cases:
        for w in range(case[0]):
            env = {wgid.name: w, **{v.name: x for v, x in zip(p.values(), case)}}
            got = _eval(kernel, env)[out.name]
            if mode == "gated":
                want = python_xcd_remap(
                    w, num_wgs=case[0], num_xcds=case[1], chunk=case[2], mode="gated"
                )
            else:
                want = python_xcd_remap(
                    w, num_wgs=case[0], num_xcds=8, mode="contiguous"
                )
            assert got == want, (mode, case, w)


def test_emitted_per_plane_ir_equals_the_host_model():
    b = IRBuilder("plane")
    gx, gy, n, chunk = (b.param(s, I32) for s in ("gx", "gy", "n", "chunk"))
    x, y = b.block_id_x(), b.block_id_y()
    ox, oy = xcd_remap_plane(b, x, y, gx=gx, gy=gy, num_xcds=n, chunk=chunk)
    for gx_n, gy_n, n_n, c_n in [
        (3, 4, 4, 3),
        (8, 2, 8, 4),
        (5, 5, 0, 0),
        (1, 9, 2, 1),
    ]:
        for yy in range(gy_n):
            for xx in range(gx_n):
                env = {
                    gx.name: gx_n, gy.name: gy_n, n.name: n_n, chunk.name: c_n,
                    x.name: xx, y.name: yy,
                }  # fmt: skip
                vals = _eval(b.kernel, env)
                plane = python_xcd_remap(
                    xx + gx_n * yy, num_wgs=gx_n * gy_n, num_xcds=n_n, chunk=c_n,
                    mode="gated",
                )  # fmt: skip
                assert (vals[ox.name], vals[oy.name]) == (plane % gx_n, plane // gx_n)


# --------------------------------------------------------------------------- op sequence = the replaced code


def test_gated_emission_is_op_for_op_the_replaced_attention_bwd_remap():
    def new(b, w, num_wgs, n, chunk):
        return xcd_remap(b, w, num_wgs=num_wgs, num_xcds=n, chunk=chunk, mode="gated")

    def old(b, w, num_wgs, n, chunk):
        return _old_gated_remap(b, w, num_wgs=num_wgs, n=n, chunk=chunk)

    k_new, *_ = _kernel_1d(new, ["num_wgs", "n", "chunk"])
    k_old, *_ = _kernel_1d(old, ["num_wgs", "n", "chunk"])
    assert _signature(k_new) == _signature(k_old)


def test_contiguous_emission_is_op_for_op_the_replaced_gdn_remap():
    def new(b, w, npid):
        return xcd_remap(b, w, num_wgs=npid, num_xcds=8, mode="contiguous")

    def old(b, w, npid):
        return _old_gdn_remap(b, w, npid)

    k_new, *_ = _kernel_1d(new, ["npid"])
    k_old, *_ = _kernel_1d(old, ["npid"])
    assert _signature(k_new) == _signature(k_old)


def test_per_plane_emission_is_op_for_op_the_replaced_inline_decode():
    def build(emit):
        b = IRBuilder("plane")
        gx, gy, n, chunk = (b.param(s, I32) for s in ("gx", "gy", "n", "chunk"))
        x, y = b.block_id_x(), b.block_id_y()
        emit(b, x, y, gx, gy, n, chunk)
        return b.kernel

    def new(b, x, y, gx, gy, n, chunk):
        return xcd_remap_plane(b, x, y, gx=gx, gy=gy, num_xcds=n, chunk=chunk)

    def old(b, x, y, gx, gy, n, chunk):
        plane = _old_gated_remap(
            b, b.add(x, b.mul(gx, y)), num_wgs=b.mul(gx, gy), n=n, chunk=chunk
        )
        return b.mod(plane, gx), b.div(plane, gx)

    assert _signature(build(new)) == _signature(build(old))


# --------------------------------------------------------------------------- dispatch of the chunked mode


def _emit_once(call):
    b = IRBuilder("dispatch")
    num_wgs = b.param("num_wgs", I32)
    chunk = b.param("chunk", I32)
    call(b, b.block_id_x(), num_wgs, chunk)
    return _signature(b.kernel)


def test_chunked_mode_routes_to_the_matching_platform_helper():
    assert _emit_once(
        lambda b, w, n, c: xcd_remap(b, w, num_wgs=1024, num_xcds=8, chunk=64)
    ) == _emit_once(
        lambda b, w, n, c: chiplet_transform_chunked(
            b, w, num_wgs=1024, num_xcds=8, chunk_size=64
        )
    )
    assert _emit_once(
        lambda b, w, n, c: xcd_remap(b, w, num_wgs=n, num_xcds=8, chunk=64)
    ) == _emit_once(
        lambda b, w, n, c: chiplet_transform_chunked_dynamic(
            b, w, num_wgs=n, num_xcds=8, chunk_size=64
        )
    )
    assert _emit_once(
        lambda b, w, n, c: xcd_remap(b, w, num_wgs=n, num_xcds=8, chunk=c)
    ) == _emit_once(
        lambda b, w, n, c: chiplet_transform_chunked_runtime(
            b, w, num_wgs=n, num_xcds=8, chunk_size=c
        )
    )
    assert _emit_once(
        lambda b, w, n, c: xcd_remap(b, w, num_wgs=n, num_xcds=8)
    ) == _emit_once(
        lambda b, w, n, c: chiplet_transform_chunked_runtime(
            b, w, num_wgs=n, num_xcds=8, chunk_size=None
        )
    )


def test_invalid_use_is_refused():
    b = IRBuilder("bad")
    n = b.param("n", I32)
    w = b.block_id_x()
    with pytest.raises(ValueError):
        xcd_remap(b, w, num_wgs=n, num_xcds=8, chunk=1, mode="nope")
    with pytest.raises(TypeError):  # a runtime num_xcds needs the gated mode
        xcd_remap(b, w, num_wgs=n, num_xcds=n, chunk=1)
    with pytest.raises(TypeError):  # the gated mode needs a chunk
        xcd_remap(b, w, num_wgs=n, num_xcds=n, mode="gated")
    with pytest.raises(TypeError):  # contiguous takes no chunk
        xcd_remap(b, w, num_wgs=n, num_xcds=8, chunk=4, mode="contiguous")
    with pytest.raises(ValueError):
        python_xcd_remap(0, num_wgs=8, num_xcds=8, mode="nope")
