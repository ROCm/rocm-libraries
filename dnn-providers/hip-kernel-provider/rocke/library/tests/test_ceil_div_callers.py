# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host callers of the shared ``rocke.helpers.spec.ceil_div`` / ``align_up``
outside the attention backward (CPU only, no IR, no GPU).

Conv, GEMM split-K / stream-K and the gfx950 dense forward each kept a private
``_ceil_div`` / ``align_up`` before they moved to the shared helpers. Their
tile counts, grids, kernargs and workspace sizes are host arithmetic that the
IR goldens do not see, so each caller is pinned here three ways over a sweep
of integer problems:

* develop: a SHA-256 of the sweep's results, taken with the modules of develop
  ``5ea912b27b8`` (the private copies). A changed tile count, grid, kernarg or
  partition at any call site changes it;
* the removed formula: the module's result equals the result with the
  module's removed private copy patched back in for the shared helper;
* the contract: an integer-like input (``numpy`` integer, ``bool``) is
  accepted and gives a plain ``int``; a ``float`` raises ``TypeError`` (the
  removed ``(a + b - 1) // b`` copies returned a float). No caller reaches the
  helpers with a float: the problem and spec fields are integers by
  construction (``grouped_convolution._problem`` and ``select_split_k*``
  coerce with ``int``; the dense ``align_up`` coerces ``n``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools

import pytest


# --- the removed private copies, verbatim -----------------------------------


def _old_add(a, b):  # conv_args, conv_direct_grouped, dgrad, wgrad reduce
    return (a + b - 1) // b


def _old_add_int(a, b):  # helpers/split_k.py
    return (int(a) + int(b) - 1) // int(b)


def _old_neg(a, b):  # helpers/streamk.py
    return -(-a // b)


def _old_dense_align_up(n, mult):  # kernels/gfx950/attention_dense.py
    return ((int(n) + mult - 1) // mult) * mult


def _int_leaves(x):
    if isinstance(x, (tuple, list)):
        for y in x:
            yield from _int_leaves(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from _int_leaves(y)
    elif dataclasses.is_dataclass(x) and not isinstance(x, type):
        for f in dataclasses.fields(x):
            yield from _int_leaves(getattr(x, f.name))
    elif isinstance(x, int) and not isinstance(x, bool):
        yield x


def _digest(rows) -> str:
    return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]


# --- sweeps ------------------------------------------------------------------


def _conv_problems():
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    shapes = [
        # N, Hi, Wi, C, K, Y, X, stride, pad, dilation, groups
        (1, 7, 7, 8, 8, 3, 3, 1, 1, 1, 1),
        (2, 14, 14, 64, 64, 3, 3, 1, 1, 1, 1),
        (2, 13, 11, 48, 96, 3, 3, 2, 1, 1, 1),
        (1, 17, 9, 32, 16, 5, 3, 3, 2, 1, 2),
        (3, 9, 9, 24, 40, 1, 1, 1, 0, 1, 1),
        (1, 31, 29, 16, 16, 3, 3, 2, 2, 2, 4),
        (4, 5, 5, 192, 1056, 3, 3, 1, 1, 1, 1),
        (1, 8, 8, 32, 32, 4, 4, 4, 0, 1, 8),
    ]
    return [
        ConvProblem(
            N=N, Hi=Hi, Wi=Wi, C=C, K=K, Y=Y, X=X,
            sH=s, sW=s, pH=p, pW=p, dH=d, dW=d, groups=g,
        )
        for N, Hi, Wi, C, K, Y, X, s, p, d, g in shapes
    ]  # fmt: skip


_TILES = [(16, 16, 16), (32, 64, 32), (64, 64, 32), (128, 32, 64), (48, 80, 24)]


def _snap_conv_args():
    from kernels.common.conv_args import ConvArgs

    rows = []
    for prob, (tm, tn, tk) in itertools.product(_conv_problems(), _TILES):
        for direction in ("fwd", "dgrad", "wgrad"):
            a = ConvArgs.from_problem(
                prob, direction=direction, tile_m=tm, tile_n=tn, tile_k=tk
            )
            rows.append((a.num_pid_m, a.num_pid_n, a.grid()))
            for s in (1, 2, 3, 7) if direction == "wgrad" else (1,):
                rows.append(a._variant_values(s))
            if direction == "wgrad":
                for pool in (1, 7, 256):
                    rows.append(a.streamk_partition(pool))
    return rows


def _direct_problems():
    from kernels.common.conv_direct_grouped import DirectConvProblem

    return [
        DirectConvProblem(N=1, H=8, W=8, groups=8, cpg=16, kpg=16),
        DirectConvProblem(N=2, H=13, W=11, groups=16, cpg=4, kpg=4),
        DirectConvProblem(N=3, H=7, W=29, groups=64, cpg=1, kpg=1),
        DirectConvProblem(N=1, H=17, W=33, groups=3, cpg=32, kpg=48, stride=2),
        DirectConvProblem(N=2, H=5, W=67, groups=32, cpg=8, kpg=8, KH=5, KW=5, PAD=2),
    ]


# Every spec class ``direct_launch_geometry`` takes a ceiling in.
_DIRECT_SPECS = (
    "DirectConvSpec",
    "DirectDepthwiseSpatialSpec",
    "DirectDepthwiseSpec",
    "DirectDepthwiseTiledSpec",
    "DirectDepthwiseColSpec",
    "DirectDepthwiseDgradStreamSpec",
    "DirectDepthwiseDgradSpec",
    "DirectConvDgradSpec",
    "DirectConvWgradSpec",
    "DirectTransposeWeightsDgradSpec",
    "DirectReorganizeWeightsSpec",
)


def _direct_specs():
    from kernels.common import conv_direct_grouped as dg

    specs = []
    for cls_name in _DIRECT_SPECS:
        cls = getattr(dg, cls_name)
        for p in _direct_problems():
            try:
                specs.append(cls(problem=p))
            except (ValueError, TypeError, AssertionError):
                continue
            if cls_name == "DirectReorganizeWeightsSpec":
                specs.append(cls(problem=p, fold_k32=True))
    return specs


def _snap_direct():
    from kernels.common.conv_direct_grouped import direct_launch_geometry

    return [direct_launch_geometry(s) for s in _direct_specs()]


def _snap_dgrad():
    from kernels.common import conv_implicit_gemm_dgrad as dgrad

    rows = []
    for prob in _conv_problems():
        tilde = dgrad.compute_tilde(prob)
        rows.append(tilde)
        for (tm, tn, tk), sk in itertools.product(_TILES, (1, 2, 4)):
            sgs = dgrad.enumerate_sub_gemms(prob, tilde, tm, tn, tk, sk)
            rows.append(sgs)
            rows.append(dgrad.pack_sub_gemm_buffer(sgs, tm, tn))
    return rows


def _snap_wgrad_reduce():
    from kernels.common import conv_wgrad_workspace_reduce as red

    rows = []
    for prob in _conv_problems():
        for tm, tn in ((4, 64), (1, 1), (8, 32), (3, 17)):
            spec = red.WgradReduceSpec(
                problem=prob, tile_m=tm, tile_n=tn, groups=prob.groups
            )
            rows.append(red.wgrad_reduce_grid(spec))
    return rows


_GEMM = [
    (1, 1, 64),
    (7, 4096, 4096),
    (128, 128, 8192),
    (257, 333, 1024),
    (4096, 4096, 4096),
    (16, 8192, 65536),
]


def _snap_split_k():
    from rocke.helpers import split_k as sk

    rows = []
    for (M, N, K), (tm, tn, tk), batch in itertools.product(_GEMM, _TILES, (1, 3)):
        rows.append(
            sk.select_split_k(
                M=M, N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk, batch=batch, env={}
            )
        )
        rows.append(
            sk.select_split_k_wgrad(
                wg_M=M, wg_N=N, wg_K=K, tile_m=tm, tile_n=tn, tile_k=tk, groups=batch
            )
        )
    return rows


def _partitions():
    from rocke.helpers.streamk import streamk_iter_partition

    out = []
    for m, n, ipt, pool in itertools.product(
        (1, 3, 9, 17), (1, 4, 51), (1, 2, 5, 36, 64), (1, 7, 64, 256)
    ):
        try:
            out.append(
                streamk_iter_partition(
                    m_tiles=m, n_tiles=n, iters_per_tile=ipt, max_active_wgs=pool
                )
            )
        except ValueError:
            continue
    return out


def _snap_streamk():
    return [
        (p.max_linear_partners, [p.flags_bytes(w) for w in (1, 2, 4, 16)])
        for p in _partitions()
    ]


def _snap_dense_align_up():
    from kernels.gfx950.attention_dense import align_up

    return [
        align_up(n, m)
        for n, m in itertools.product(range(0, 300), (1, 16, 32, 64, 128, 256))
    ]


_SNAPSHOTS = {
    "conv_args": _snap_conv_args,
    "conv_direct_grouped": _snap_direct,
    "conv_implicit_gemm_dgrad": _snap_dgrad,
    "conv_wgrad_workspace_reduce": _snap_wgrad_reduce,
    "split_k": _snap_split_k,
    "streamk": _snap_streamk,
    "gfx950_attention_dense": _snap_dense_align_up,
}

# _digest of each sweep with the modules of develop 5ea912b27b8 (the removed
# private copies), taken on a CPU host.
_DEVELOP_DIGESTS = {
    "conv_args": "c88c3f279fb65c13",
    "conv_direct_grouped": "d42a5a12b7673434",
    "conv_implicit_gemm_dgrad": "183b949cfa989fd6",
    "conv_wgrad_workspace_reduce": "82e21bc991249030",
    "split_k": "8d18a8186c1dbc52",
    "streamk": "e7e48396bfcb2067",
    "gfx950_attention_dense": "76049912ec1260bf",
}


# --- 1. develop's results ----------------------------------------------------


@pytest.mark.parametrize("site", sorted(_SNAPSHOTS))
def test_results_match_develop(site):
    rows = _SNAPSHOTS[site]()
    assert rows
    for v in _int_leaves(rows):
        assert type(v) is int
    assert _digest(rows) == _DEVELOP_DIGESTS[site]


def test_direct_sweep_covers_every_spec_class():
    covered = {type(s).__name__ for s in _direct_specs()}
    assert covered == set(_DIRECT_SPECS), set(_DIRECT_SPECS) - covered


def test_streamk_sweep_has_streamk_ctas():
    assert any(p.sk_ctas for p in _partitions())


# --- 2. the removed formula patched back in ----------------------------------


@pytest.mark.parametrize(
    "site,module,patches",
    [
        ("conv_args", "kernels.common.conv_args", {"ceil_div": _old_add}),
        (
            "conv_direct_grouped",
            "kernels.common.conv_direct_grouped",
            {"ceil_div": _old_add},
        ),
        (
            "conv_implicit_gemm_dgrad",
            "kernels.common.conv_implicit_gemm_dgrad",
            {"ceil_div": _old_add},
        ),
        (
            "conv_wgrad_workspace_reduce",
            "kernels.common.conv_wgrad_workspace_reduce",
            {"ceil_div": _old_add},
        ),
        ("split_k", "rocke.helpers.split_k", {"ceil_div": _old_add_int}),
        (
            "streamk",
            "rocke.helpers.streamk",
            {"ceil_div": _old_neg, "align_up": lambda n, m: _old_neg(n, m) * m},
        ),
        (
            "gfx950_attention_dense",
            "kernels.gfx950.attention_dense",
            {"align_up": _old_dense_align_up},
        ),
    ],
)
def test_results_match_removed_formula(monkeypatch, site, module, patches):
    import importlib

    mod = importlib.import_module(module)
    new = _SNAPSHOTS[site]()
    for name, fn in patches.items():
        assert hasattr(mod, name)
        monkeypatch.setattr(mod, name, fn)
    assert new == _SNAPSHOTS[site]()


# --- 3. the contract ---------------------------------------------------------


@pytest.mark.parametrize("wgrad", (False, True))
def test_split_k_keeps_int_coercion_of_its_inputs(wgrad):
    """``select_split_k*`` still ``int()`` the M / N extents and tiles before
    the ceiling, as the removed private copy did: whole ``float`` and
    ``numpy`` values there give the integer result (no ``TypeError``)."""
    np = pytest.importorskip("numpy")
    from rocke.helpers.split_k import select_split_k, select_split_k_wgrad

    if wgrad:
        fn, mn, rest = select_split_k_wgrad, ("wg_M", "wg_N"), {"wg_K": 1024}
    else:
        fn, mn, rest = select_split_k, ("M", "N"), {"K": 1024, "env": {}}
    ints = {mn[0]: 257, mn[1]: 333, "tile_m": 64, "tile_n": 48}
    want = fn(**ints, tile_k=32, **rest)
    for cast in (float, np.int64, np.float32):
        got = fn(**{k: cast(v) for k, v in ints.items()}, tile_k=32, **rest)
        assert got == want
        assert type(got.base_grid) is int


def test_gfx950_dense_align_up_contract():
    """``n`` is ``int()``-coerced as before; ``mult`` must be integer-like."""
    np = pytest.importorskip("numpy")
    from kernels.gfx950.attention_dense import align_up

    assert align_up(np.int64(65), 64) == 128
    assert align_up(65.0, 64) == 128
    assert align_up(65, np.int32(64)) == 128
    assert type(align_up(65, np.int32(64))) is int
    with pytest.raises(TypeError):
        align_up(65, 64.0)


def test_conv_args_refuses_float_tiles():
    """A float tile (never produced by a spec) is refused instead of giving a
    float tile count and grid."""
    from kernels.common.conv_args import ConvArgs

    a = ConvArgs.from_problem(
        _conv_problems()[1], direction="wgrad", tile_m=64.0, tile_n=64, tile_k=32
    )
    with pytest.raises(TypeError):
        a.num_pid_m


@pytest.mark.parametrize(
    "fn_name,args",
    [
        ("ceil_div", (100.0, 16)),
        ("ceil_div", (100, 16.0)),
        ("align_up", (100.0, 16)),
        ("align_up", (100, 16.0)),
        ("ceil_div", (None, 16)),
    ],
)
def test_shared_helpers_refuse_non_integers(fn_name, args):
    from rocke.helpers import spec

    with pytest.raises(TypeError):
        getattr(spec, fn_name)(*args)


def test_shared_helpers_match_removed_copies_for_positive_divisors():
    from rocke.helpers.spec import align_up, ceil_div

    for a, b in itertools.product(range(-64, 257), (1, 2, 3, 16, 64, 256)):
        assert ceil_div(a, b) == _old_add(a, b) == _old_add_int(a, b)
        assert ceil_div(a, b) == _old_neg(a, b)
        assert align_up(a, b) == _old_add(a, b) * b
