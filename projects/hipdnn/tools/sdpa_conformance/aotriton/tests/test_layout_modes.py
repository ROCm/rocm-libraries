# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""--layout-mode: the layouts O, dO and the gradients get, using AOTriton's own allocation helpers on CPU."""

import sys
import types

import pytest
import torch

import aotriton_layout as layout
import conformance_contract as contract
import hipdnn_aotriton_plugin as plugin

B, H, SQ, SK, D = 2, 3, 16, 24, 8
DTYPE = torch.float32


@pytest.fixture
def vendored(monkeypatch):
    """The vendored `_common_test`, `attn_torch_function` and `_core_test_backward`, unloaded afterwards."""
    monkeypatch.syspath_prepend(str(layout.FLASH_TESTS))
    names = ("_common_test", "attn_torch_function", "_core_test_backward")
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    import _common_test as common
    import _core_test_backward as core
    import attn_torch_function as atf

    yield types.SimpleNamespace(common=common, core=core, atf=atf)
    for name in names:
        sys.modules.pop(name, None)


def tensors(common, perms):
    """q, k, v, b stored in `perms` = (q, k, v, b) layouts."""
    dims = {
        "q": (B, H, SQ, D),
        "k": (B, H, SK, D),
        "v": (B, H, SK, D),
        "b": (B, H, SQ, SK),
    }
    return [
        common.alloc_with_layout(dims[t], p, dtype=DTYPE, device="cpu", rand=True)
        for t, p in zip("qkvb", perms)
    ]


def allocate_outputs(atf, q, k, v, b, ext):
    """What AOTriton's forward/backward allocate for O and the gradients under `ext`."""
    alloc = dict(dtype=DTYPE, device="cpu", attn_extra_args=ext, narrow_to=None)
    return {
        "o": atf._alloc_output("o", (B, H, SQ, D), like=None, **alloc),
        "dq": atf._alloc_like(q, "dq", ext, None),
        "dk": atf._alloc_like(k, "dk", ext, None),
        "dv": atf._alloc_like(v, "dv", ext, None),
        "db": atf._alloc_like(b, "db", ext, None),
    }


def wrapped_attention(vendored, mode, seen):
    def inner(q, k, v, b, causal, sm_scale, dropout_p, ext):
        seen.append(ext)
        return None

    return plugin.make_attention(inner, mode, vendored.atf.AttentionExtraArgs)


LAYOUT_CASES = {
    "all-bshd": lambda c: (c.BSHD,) * 4,
    "all-bhsd": lambda c: (c.BHSD,) * 4,
    "mixed": lambda c: (c.HBSD, c.BSHD, c.BHSD, c.SHBD),
}


@pytest.mark.parametrize("case", LAYOUT_CASES)
@pytest.mark.parametrize("mode", ["pytorch", "efficient"])
def test_outputs_are_allocated_in_the_modes_layouts(vendored, mode, case):
    c = vendored.common
    perms = LAYOUT_CASES[case](c)
    q, k, v, b = tensors(c, perms)
    seen = []
    wrapped_attention(vendored, mode, seen)(q, k, v, b, False, 1.0, 0.0)
    got = {
        name: c.layout_of(t)
        for name, t in allocate_outputs(vendored.atf, q, k, v, b, seen[0]).items()
    }
    if mode == "pytorch":
        expected = {
            "o": perms[0],
            "dq": perms[0],
            "dk": perms[1],
            "dv": perms[2],
            "db": perms[3],
        }
    else:
        expected = {n: c.BSHD for n in ("o", "dq", "dk", "dv")} | {"db": perms[3]}
    assert got == expected


def storage_order(t):
    """`t`'s strides on the axes that have more than one element; an extent-1 axis has no place in memory."""
    return [t.stride(i) for i in range(4) if t.shape[i] > 1]


@pytest.mark.parametrize(
    "dims",
    [(1, 1, 16, 16), (2, 1, 16, 24), (1, 3, 1, 24), (1, 3, 16, 1), (2, 3, 16, 24)],
)
def test_efficient_allocates_o_and_gradients_at_any_shape(vendored, dims):
    c, atf = vendored.common, vendored.atf
    bsz, heads, sq, sk = dims
    q, k, v, b = (
        c.alloc_with_layout(d, c.BSHD, dtype=DTYPE, device="cpu", rand=True)
        for d in (
            (bsz, heads, sq, D),
            (bsz, heads, sk, D),
            (bsz, heads, sk, D),
            (bsz, heads, sq, sk),
        )
    )
    seen = []
    wrapped_attention(vendored, "efficient", seen)(q, k, v, b, False, 1.0, 0.0)
    alloc = dict(dtype=DTYPE, device="cpu", attn_extra_args=seen[0], narrow_to=None)
    # AOTriton's own layout assertion runs inside these allocations
    o = atf._alloc_output("o", (bsz, heads, sq, D), like=None, **alloc)
    grads = [
        atf._alloc_like(t, n, seen[0], None)
        for t, n in ((q, "dq"), (k, "dk"), (v, "dv"))
    ]
    bshd = lambda shape: c.alloc_with_layout(shape, c.BSHD, dtype=DTYPE, device="cpu")
    for t in (o, *grads):
        assert storage_order(t) == storage_order(bshd(tuple(t.shape)))
    if min(bsz, heads, sq) > 1:
        assert [c.layout_of(t) for t in (o, *grads)] == [c.BSHD] * 4


@pytest.mark.parametrize("mode", ["pytorch", "efficient"])
def test_a_missing_bias_gets_no_gradient_layout(vendored, mode):
    c = vendored.common
    q, k, v, _ = tensors(c, (c.BSHD,) * 4)
    seen = []
    wrapped_attention(vendored, mode, seen)(q, k, v, None, False, 1.0, 0.0)
    assert "db" not in seen[0].output_layouts


def test_a_caller_set_output_layouts_is_not_overridden(vendored):
    c = vendored.common
    q, k, v, b = tensors(c, (c.BSHD,) * 4)
    chosen = {"o": c.SHBD}
    ext = vendored.atf.AttentionExtraArgs(output_layouts=chosen)
    seen = []
    wrapped_attention(vendored, "efficient", seen)(q, k, v, b, False, 1.0, 0.0, ext)
    assert seen[0].output_layouts is chosen


def test_the_callers_extra_args_are_not_mutated(vendored):
    c = vendored.common
    q, k, v, b = tensors(c, (c.BSHD,) * 4)
    ext = vendored.atf.AttentionExtraArgs(fillnan=True)
    seen = []
    wrapped_attention(vendored, "efficient", seen)(q, k, v, b, False, 1.0, 0.0, ext)
    assert ext.output_layouts is None
    assert seen[0].fillnan is True and seen[0].output_layouts is not None


def bwd_args(flip, batch=B, heads=H, sq=SQ, sk=SK):
    return (batch, heads, D, sq, sk, False, "l1", 0.0, DTYPE, flip, "matrix")


def recording_do_test_op_bwd(got):
    """A stand-in for `_do_test_op_bwd` that records what it is called with and unpacks `args` by
    the same names the vendored function does (the plugin finds `storage_flip` through them).
    """

    def _do_test_op_bwd(request, args, device_str="cuda"):
        (
            BATCH,
            N_HEADS,
            D_HEAD,
            seqlen_q,
            seqlen_k,
            causal,
            scale,
            dropout,
            dtype,
            storage_flip,
            bias,
        ) = args
        got.append((args, device_str))

    return _do_test_op_bwd


def converted(vendored, mode, flip, **dims):
    got = []
    wrapper = plugin.make_do_test_op_bwd(recording_do_test_op_bwd(got), mode)
    wrapper("request", bwd_args(flip, **dims), "cpu")
    ((args, device_str),) = got
    assert device_str == "cpu"
    assert args[:9] == bwd_args(flip, **dims)[:9] and args[10:] == ("matrix",)
    return args[9]


def context(vendored, storage_layout=None, storage_flip=None):
    return vendored.common.SdpaContext(
        B,
        H,
        D,
        SQ,
        SK,
        DTYPE,
        bias_type="matrix",
        storage_flip=storage_flip,
        device="cpu",
        storage_layout=storage_layout,
    )


@pytest.mark.parametrize("flip", [False, True, (1, 2), (0, 1)])
@pytest.mark.parametrize("mode", ["pytorch", "efficient"])
def test_dout_and_outputs_follow_the_mode_while_inputs_keep_their_flip(
    vendored, mode, flip
):
    c = vendored.common
    sl = converted(vendored, mode, flip)
    ctx = context(vendored, storage_layout=sl)
    q, k, v, b = ctx.dev_tensors
    (dout,) = ctx.ddev_tensors
    input_perm = {
        False: c.BHSD,
        True: c.BSHD,
        (1, 2): c.BSHD,
        (0, 1): c.HBSD,
    }[flip]
    assert [c.layout_of(t) for t in (q, k, v, b)] == [input_perm] * 4
    expected_out = input_perm if mode == "pytorch" else c.BSHD
    assert c.layout_of(dout) == expected_out
    assert sl.outputs() == {
        "o": expected_out,
        "dq": expected_out,
        "dk": expected_out,
        "dv": expected_out,
        "db": input_perm,
    }
    # the same output layouts the forward wrapper derives from these inputs
    assert sl.outputs() == plugin.output_layouts(mode, q, k, v, b)


@pytest.mark.parametrize(
    "dims",
    [
        dict(heads=1),
        dict(sq=1),
        dict(sk=1),
        dict(batch=1),
        dict(batch=1, heads=1, sq=1, sk=1),
        dict(heads=(2, 1)),
    ],
    ids=str,
)
@pytest.mark.parametrize("flip", [False, True, (1, 2), (0, 1)])
@pytest.mark.parametrize("mode", ["pytorch", "efficient"])
def test_every_tensor_passes_aotriton_layout_check_with_an_extent_one_axis(
    vendored, mode, flip, dims
):
    """What `_do_test_op_bwd`'s `_check_layout` and AOTriton's output allocation assert, on CPU."""
    c, atf = vendored.common, vendored.atf
    full = dict(batch=B, heads=H, sq=SQ, sk=SK) | dims
    sl = converted(vendored, mode, flip, **full)
    heads = full["heads"]
    hq, _ = (heads, heads) if isinstance(heads, int) else heads
    ctx = c.SdpaContext(
        full["batch"],
        heads,
        D,
        full["sq"],
        full["sk"],
        DTYPE,
        bias_type="matrix",
        device="cpu",
        storage_layout=sl,
    )
    q, k, v, b = ctx.dev_tensors
    for name, t in zip("qkvb", (q, k, v, b)):
        c.assert_layout(t, sl[name], 0, name)
    odims = (full["batch"], hq, full["sq"], D)
    dout = c.alloc_with_layout(odims, sl["dout"], dtype=DTYPE, device="cpu", rand=True)
    c.assert_layout(dout, sl["dout"], 0, "dout")
    ext = atf.AttentionExtraArgs(output_layouts=sl.outputs())
    alloc = dict(dtype=DTYPE, device="cpu", attn_extra_args=ext, narrow_to=None)
    atf._alloc_output("o", odims, like=None, **alloc)
    for t, name in ((q, "dq"), (k, "dk"), (v, "dv"), (b, "db")):
        atf._alloc_like(t, name, ext, None)
    # the wrapper hands AOTriton what the forward wrapper would for the same inputs
    assert sl.outputs() == plugin.output_layouts(mode, q, k, v, b)


@pytest.mark.parametrize("flip", [False, True, (1, 2)])
@pytest.mark.parametrize("mode", ["pytorch", "efficient"])
def test_converted_inputs_hold_the_values_the_storage_flip_gave(vendored, mode, flip):
    old = context(vendored, storage_flip=(1, 2) if flip is True else (flip or None))
    new = context(vendored, storage_layout=converted(vendored, mode, flip))
    for was, now in zip(old.dev_tensors, new.dev_tensors):
        assert was.stride() == now.stride()
        assert torch.equal(was, now)


@pytest.mark.parametrize("mode", ["pytorch", "efficient"])
def test_a_storage_layout_in_args_is_left_alone(vendored, mode):
    chosen = vendored.common.StorageLayout.round_robin(2)
    got = []
    wrapper = plugin.make_do_test_op_bwd(recording_do_test_op_bwd(got), mode)
    wrapper("request", bwd_args(chosen), "cpu")
    assert got[0][0][9] is chosen


def test_storage_flip_is_found_by_name_in_the_vendored_do_test_op_bwd(vendored):
    index = plugin.unpack_names(vendored.core._do_test_op_bwd).index("storage_flip")
    flip = object()
    assert bwd_args(flip)[index] is flip


def test_storage_flip_is_found_wherever_the_vendored_function_unpacks_it(vendored):
    def moved(request, args, device_str="cuda"):
        storage_flip, BATCH, N_HEADS, D_HEAD, bias_type = args

    def renamed(request, args, device_str="cuda"):
        BATCH, N_HEADS, D_HEAD, flip, bias_type = args

    assert plugin.unpack_names(moved).index("storage_flip") == 0
    with pytest.raises(RuntimeError, match=r"renamed.*storage_flip"):
        plugin.unpack_names(renamed)


def test_a_wrapped_function_without_storage_flip_fails_loudly(vendored):
    def no_flip(request, args, device_str="cuda"):
        a, b, c = args

    def no_args(request, arguments, device_str="cuda"):
        storage_flip, b = arguments

    with pytest.raises(RuntimeError, match=r"no_flip.*storage_flip"):
        plugin.make_do_test_op_bwd(no_flip, "pytorch")
    with pytest.raises(RuntimeError, match=r"no_args.*args"):
        plugin.make_do_test_op_bwd(no_args, "pytorch")


@pytest.mark.parametrize("mode", ["pytorch", "efficient"])
def test_the_wrapper_converts_the_slot_its_function_unpacks_storage_flip_from(
    vendored, mode
):
    seen = []

    def moved(request, args, device_str="cuda"):
        (
            storage_flip,
            BATCH,
            N_HEADS,
            D_HEAD,
            seqlen_q,
            seqlen_k,
            causal,
            scale,
            dropout,
            dtype,
            bias,
        ) = args
        seen.append(args)

    flip = (1, 2)
    args = (flip, B, H, D, SQ, SK, False, "l1", 0.0, DTYPE, "matrix")
    plugin.make_do_test_op_bwd(moved, mode)("request", args, "cpu")
    (got,) = seen
    assert isinstance(got[0], vendored.common.StorageLayout)
    assert got[1:] == args[1:]


def test_a_starred_unpack_of_args_fails_loudly(vendored):
    def starred(request, args, device_str="cuda"):
        BATCH, *mid, storage_flip, bias_type = args

    with pytest.raises(RuntimeError, match=r"starred.*starred target"):
        plugin.unpack_names(starred)
    with pytest.raises(RuntimeError, match=r"starred.*starred target"):
        plugin.make_do_test_op_bwd(starred, "pytorch")


def test_a_wrapped_function_without_the_extents_fails_loudly(vendored):
    def no_extents(request, args, device_str="cuda"):
        BATCH, N_HEADS, D_HEAD, sq, sk, storage_flip = args

    with pytest.raises(RuntimeError, match=r"no_extents.*seqlen_q.*seqlen_k"):
        plugin.make_do_test_op_bwd(no_extents, "efficient")


def test_aotriton_mode_installs_nothing_and_removes_earlier_wrappers(
    vendored, monkeypatch
):
    atf, core = vendored.atf, vendored.core
    attention, do_test = atf.attention, core._do_test_op_bwd
    importer = types.ModuleType("importer_of_attention")
    importer.attention = attention
    monkeypatch.setitem(sys.modules, "importer_of_attention", importer)

    plugin._install_layout_wrappers("aotriton")
    assert (atf.attention, core._do_test_op_bwd) == (attention, do_test)

    plugin._install_layout_wrappers("pytorch")
    assert atf.attention is not attention and core._do_test_op_bwd is not do_test
    assert (
        importer.attention is atf.attention
    )  # a module that imported the name sees the wrapper
    plugin._install_layout_wrappers(
        "efficient"
    )  # re-installing never wraps a wrapper twice
    assert atf.attention._sdpa_conformance_inner is attention

    plugin._install_layout_wrappers("aotriton")
    assert (atf.attention, core._do_test_op_bwd, importer.attention) == (
        attention,
        do_test,
        attention,
    )


@pytest.mark.parametrize("flip", [False, True])
def test_aotriton_mode_passes_the_storage_flip_through_untouched(vendored, flip):
    atf, core = vendored.atf, vendored.core
    seen = []
    original = core._do_test_op_bwd
    core._do_test_op_bwd = lambda request, args, device_str="cuda": seen.append(args)
    try:
        plugin._install_layout_wrappers("aotriton")
        core._do_test_op_bwd("request", bwd_args(flip), "cpu")
    finally:
        core._do_test_op_bwd = original
    assert seen == [bwd_args(flip)]


def test_unknown_layout_mode_is_a_usage_error(monkeypatch):
    monkeypatch.setenv(contract.LAYOUT_MODE_ENV, "bshd")
    with pytest.raises(pytest.UsageError, match="bshd"):
        plugin.layout_mode()


def test_layout_mode_defaults_to_pytorch(monkeypatch):
    monkeypatch.delenv(contract.LAYOUT_MODE_ENV, raising=False)
    assert plugin.layout_mode() == "pytorch"


def test_the_mode_is_in_the_report_header(pytester, monkeypatch):
    monkeypatch.setenv(contract.LAYOUT_MODE_ENV, "efficient")
    pytester.makepyfile("def test_x():\n    pass\n")
    result = pytester.runpytest("-p", "hipdnn_aotriton_plugin")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["sdpa-conformance:*layout mode: efficient*"])


def test_a_bad_mode_aborts_at_configure(pytester, monkeypatch):
    monkeypatch.setenv(contract.LAYOUT_MODE_ENV, "nope")
    pytester.makepyfile("def test_x():\n    pass\n")
    result = pytester.runpytest("-p", "hipdnn_aotriton_plugin")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines([f"*{contract.LAYOUT_MODE_ENV}*"])
