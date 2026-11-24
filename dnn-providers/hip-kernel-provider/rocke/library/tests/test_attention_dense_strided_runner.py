# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Stride admission of the dense attention runners (host only: no torch, GPU, compiler).

Every dense body takes 12 trailing stride kernargs. The gfx942 grid, KV-segment
and persistent bodies and the gfx950 grid, KV-segment, persistent and wide-DMA
bodies read them (``body_serves_strides`` True); the gfx950 ``gqa_pair`` orders,
ragged and varlen specs, the gfx942 KDB/VPF pipeline and the additive-bias bodies of
both do not, so their runners launch packed ``[B, S, H, D]`` tensors only. These tests
drive the runners with duck-typed tensors and a recording launcher, and pin that:

* on a body that does not serve strides, a view that is not packed is refused
  with a ``ValueError`` BEFORE anything is built or compiled; a misaligned base
  is refused on every body;
* on a body that serves strides, a strided view reaches the launcher with the
  12 values the contract's :func:`dense_strides_admit` gives for the four views,
  and a view the contract declines is refused with the contract's reason;
* packed views, and objects without ``.stride()`` (the raw addresses the SDPA
  replay passes), reach the launcher with the 12 stride values of the packed layout;
  a bias launch refuses raw addresses, since it checks the bias against q's device,
  and a sinks launch refuses a raw q, since it checks the sinks against q's dtype.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

import kernels.common.attention_unified as au
from dispatch.attention import (
    AttentionRequest,
    attention_tuning_spec,
    tuning_spec_with_knobs,
)
from kernels.common._attention_dense_strides import (
    DENSE_STRIDE_NAMES,
    DENSE_STRIDE_TENSORS,
    admit_dense_launch_strides,
    dense_operand_from_view,
    dense_strides_admit,
    packed_launch_strides,
)
from kernels.common.attention_dense_spec import OCP_FP8_KV_DTYPE
from kernels.gfx942 import attention_dense as k942
from kernels.gfx950 import attention_dense as k950

_MODULES = {"gfx942": k942, "gfx950": k950}
# arm -> (arch, candidate spec_id, knobs applied on top of the candidate default)
_ARMS = {
    "gfx950-grid": ("gfx950", "gfx950_dense_grid", {}),
    "gfx950-persistent": ("gfx950", "gfx950_dense_persist", {}),
    "gfx950-wide-dma": ("gfx950", "gfx950_dense_persist_widedma", {}),
    "gfx942-grid": ("gfx942", "gfx942_dense", {"persistent": False}),
    "gfx942-persistent": ("gfx942", "gfx942_dense", {"persistent": True}),
    # The opt-in BF16 KDB pipeline: a gfx942 body that keeps packed addressing.
    "gfx942-kdb": (
        "gfx942",
        "gfx942_dense",
        {"persistent": True, "lds_num_buffers": 2},
    ),
    # The additive-bias bodies (the request carries the bias) keep packed addressing
    # too. Their bias read indexes only by the bias's own three strides, and the
    # extra stride arithmetic is not shown to fit their register budget (on gfx950
    # the D128 LSE bias body sits at the 256-VGPR budget).
    "gfx950-bias": ("gfx950", "gfx950_dense_grid", {}),
    "gfx942-bias": ("gfx942", "gfx942_dense", {"persistent": False}),
    "gfx942-bias-persistent": ("gfx942", "gfx942_dense", {"persistent": True}),
}
_BIAS_ARMS = ("gfx950-bias", "gfx942-bias", "gfx942-bias-persistent")
# The arms whose body reads the 12 strides; every other arm admits packed tensors only.
_SERVED_ARMS = (
    "gfx950-grid",
    "gfx950-persistent",
    "gfx950-wide-dma",
    "gfx942-grid",
    "gfx942-persistent",
)
_PACKED_ONLY_ARMS = tuple(a for a in _ARMS if a not in _SERVED_ARMS)
_DTYPE = {"gfx942-kdb": "bf16"}
_SCALE = 0.125
# A base address per tensor: multiples of 256 bytes, so any shift is the test's doing,
# and far enough apart that no two tensors here share a byte (O may not overlap an
# input).
_ADDRESSES = {
    t: 0x10000 + i * 0x10000000 for i, t in enumerate((*DENSE_STRIDE_TENSORS, "bias"))
}


def _spec(arm: str, *, batch: int = 2):
    """The kernel spec the dispatch factory builds for ``arm`` (GQA 32/8, D128,
    S=2048, fp16 (bf16 for the KDB pipeline), causal, an additive bias on the bias
    arms: inside every body's support set, no extent-1 axis at ``batch > 1``)."""
    arch, spec_id, knobs = _ARMS[arm]
    request = AttentionRequest(
        batch=batch,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=2048,
        seqlen_k=2048,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        dtype=_DTYPE.get(arm, "fp16"),
        mask_type=1,
        use_attn_bias=arm in _BIAS_ARMS,
    )
    # Pin the resolved arch like the dispatch wiring tests, so the factory does not
    # depend on the host GPU.
    old, au._RESOLVED_ATTENTION_ARCH = au._RESOLVED_ATTENTION_ARCH, arch
    try:
        if knobs:
            spec = tuning_spec_with_knobs(request, spec_id, knobs).kernel_spec
        else:
            spec = attention_tuning_spec(request, spec_id).kernel_spec
    finally:
        au._RESOLVED_ATTENTION_ARCH = old
    assert spec.persistent == ("persist" in spec_id or bool(knobs.get("persistent")))
    if arch == "gfx950":
        assert spec.wide_lds_dma == (arm == "gfx950-wide-dma")
    assert spec.use_additive_bias == (arm in _BIAS_ARMS)
    return spec


class _View:
    """Duck-typed tensor: the shape, element strides, base address and device that
    the runner's checks read, and nothing else."""

    is_cuda = True
    device = "cuda:0"

    def __init__(self, shape, strides, ptr):
        self.shape = tuple(shape)
        self._strides = tuple(strides)
        self._ptr = ptr

    def stride(self, dim=None):
        return self._strides if dim is None else self._strides[dim]

    def data_ptr(self):
        return self._ptr

    def is_contiguous(self):
        return self._strides == _contiguous(self.shape)

    def with_ptr(self, ptr):
        return _View(self.shape, self._strides, ptr)


def _contiguous(shape):
    strides, acc = [], 1
    for n in reversed(shape):
        strides.append(acc)
        acc *= n
    return tuple(reversed(strides))


def _shapes(spec):
    """Logical shapes of q, k, v, o: ``[B, S, H, D]``."""
    q = (spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size)
    kv = (spec.batch, spec.seqlen_kv, spec.num_kv_heads, spec.head_size)
    return {"q": q, "k": kv, "v": kv, "o": q}


def _packed_views(spec):
    return {
        t: _View(shape, _contiguous(shape), _ADDRESSES[t])
        for t, shape in _shapes(spec).items()
    }


def _bhsd_view(view):
    """``view``'s ``[B, S, H, D]`` shape over a ``[B, H, S, D]`` buffer at its
    address (``permute(0, 2, 1, 3)``)."""
    _, s, h, d = view.shape
    return _View(view.shape, (h * s * d, d, s * d, 1), view.data_ptr())


class _Bias(_View):
    """Duck-typed fp32 additive bias ``[1, 1, 1, Skv]``: what the runner's bias
    checks read on top of a view."""

    dtype = "torch.float32"

    def __init__(self, seqlen_kv):
        super().__init__(
            (1, 1, 1, seqlen_kv),
            (seqlen_kv, seqlen_kv, seqlen_kv, 1),
            _ADDRESSES["bias"],
        )

    def element_size(self):
        return 4

    def storage_offset(self):
        return 0

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: 4 * self.shape[-1])


def _sinks(spec):
    """Duck-typed per-query-head sink logits ``[Hq]``: what the sinks checks read."""
    return SimpleNamespace(
        shape=(spec.num_query_heads,),
        dtype="torch.float16",
        is_contiguous=lambda: True,
        is_cuda=True,
    )


def _run(module, spec, views):
    """Launch through the runner, with a bias for a bias spec and sinks for a
    sinks spec."""
    bias = _Bias(spec.seqlen_kv) if spec.use_additive_bias else None
    return module.run_attention_dense_torch(
        spec=spec,
        q=views["q"],
        k=views["k"],
        v=views["v"],
        out=views["o"],
        scale=_SCALE,
        additive_bias=bias,
        sinks=_sinks(spec) if spec.use_sinks else None,
    )


def _forbid_build(monkeypatch, module):
    """Make any build or compile fail loudly, on an empty launcher cache."""

    def refuse(*args, **kwargs):
        raise AssertionError("compiled")

    monkeypatch.setattr(module, "build_attention_dense", refuse)
    monkeypatch.setattr("rocke.helpers.compile.compile_kernel", refuse)
    monkeypatch.setattr(module, "_DENSE_LAUNCHER_CACHE", {})


def _record_launches(monkeypatch, module, spec):
    """Replace the build, the compile and the launcher so the runner runs to its
    launch on the host; returns the list of value dicts the launcher is called with
    and the signature it was built with."""
    launches, signatures = [], []

    class RecordingLauncher:
        def __init__(self, *, hsaco, kernel_name, signature):
            signatures.append(signature)

        def __call__(self, values, *, config):
            launches.append(dict(values))

    def fake_compile(kernel, *, arch, backend, capture_ir_text):
        return SimpleNamespace(hsaco=b"", kernel_name=spec.kernel_name())

    monkeypatch.setattr(module, "build_attention_dense", lambda spec, arch: object())
    monkeypatch.setattr("rocke.helpers.compile.compile_kernel", fake_compile)
    monkeypatch.setattr("rocke.runtime.KernelLauncher", RecordingLauncher)
    monkeypatch.setattr(module, "_DENSE_LAUNCHER_CACHE", {})
    return launches, signatures


@pytest.mark.parametrize("arm", _PACKED_ONLY_ARMS)
def test_a_body_that_does_not_read_its_strides_admits_packed_tensors_only(arm):
    """While a body does not read its 12 stride kernargs, the runner admits packed
    tensors only: its factory spec reports it."""
    arch = _ARMS[arm][0]
    assert _MODULES[arch].body_serves_strides(_spec(arm)) is False


@pytest.mark.parametrize("arm", _SERVED_ARMS)
def test_the_strided_bodies_serve_strides(arm):
    arch = _ARMS[arm][0]
    assert _MODULES[arch].body_serves_strides(_spec(arm)) is True


# gfx950 persistent orders whose workgroup count is a formula of the baked shape: they
# take the shape as constants, keep packed addressing and ignore the 12 parameters.
_GQA_PAIR_SPECS = {
    "gqa_pair": {"persist_decode": "gqa_pair", "num_persistent": 128},
    "gqa_pair_2phase": {"persist_decode": "gqa_pair_2phase"},
}


@pytest.mark.parametrize("order", _GQA_PAIR_SPECS)
@pytest.mark.parametrize("arm", ["gfx950-persistent", "gfx950-wide-dma"])
def test_a_gqa_pair_persistent_order_stays_packed_only(arm, order, monkeypatch):
    """The ``gqa_pair`` orders bake the shape, so they address packed tensors: a
    head-major view is refused with its reason before anything is built or compiled,
    and packed views launch with the packed values."""
    spec = dataclasses.replace(_spec(arm), **_GQA_PAIR_SPECS[order])
    assert not spec.runtime_shape
    assert k950.body_serves_strides(spec) is False
    _forbid_build(monkeypatch, k950)
    views = _packed_views(spec)
    views["k"] = _bhsd_view(views["k"])
    with pytest.raises(ValueError, match=r"k is not packed"):
        _run(k950, spec, views)
    launches, _ = _record_launches(monkeypatch, k950, spec)
    _run(k950, spec, _packed_views(spec))
    (values,) = launches
    assert {n: values[n] for n in DENSE_STRIDE_NAMES} == packed_launch_strides(spec)


@pytest.mark.parametrize("tensor", ["q", "k", "v", "o"])
def test_the_fp8_kv_cache_body_takes_the_strides_but_stays_packed_only(
    tensor, monkeypatch
):
    """The fp8 K/V-cache body declares the 12 stride kernargs like every dense body
    but loads K and V through plain packed pointers: a view that is not packed is
    refused with the packed-only reason before anything is built or compiled, and
    packed views launch with the scales and the packed stride values."""
    spec = dataclasses.replace(_spec("gfx950-grid"), kv_storage_dtype="fp8e4m3")
    assert k950.body_serves_strides(spec) is False
    assert [a["name"] for a in k950.attention_dense_signature(spec)][-12:] == list(
        DENSE_STRIDE_NAMES
    )

    def run(views):
        # The runner holds K and V to the OCP fp8 dtype before anything else.
        for name in ("k", "v"):
            views[name].dtype = OCP_FP8_KV_DTYPE
        return k950.run_attention_dense_torch(
            spec=spec,
            q=views["q"],
            k=views["k"],
            v=views["v"],
            out=views["o"],
            scale=_SCALE,
            k_scale=0.5,
            v_scale=0.25,
        )

    _forbid_build(monkeypatch, k950)
    views = _packed_views(spec)
    views[tensor] = _bhsd_view(views[tensor])
    with pytest.raises(
        ValueError, match=rf"{tensor} is not packed.*reads packed \[B, S, H, D\] only"
    ):
        run(views)
    launches, _ = _record_launches(monkeypatch, k950, spec)
    run(_packed_views(spec))
    (values,) = launches
    assert {n: values[n] for n in DENSE_STRIDE_NAMES} == packed_launch_strides(spec)
    assert (values["k_scale"], values["v_scale"]) == (0.5, 0.25)


def _fp8_factory_spec(dtype, d, hq, hkv, *, sq=512, sk=None, bottom_right=False):
    """The shipped fp8 K/V-cache spec of the fp8 numeric tests: the dispatch factory's
    grid candidate for ``use_fp8`` (bottom-right with the 256-CU split, as dispatch
    ships it)."""
    arch, old = "gfx950", au._RESOLVED_ATTENTION_ARCH
    au._RESOLVED_ATTENTION_ARCH = arch
    try:
        return attention_tuning_spec(
            AttentionRequest(
                batch=1,
                nhead_q=hq,
                nhead_k=hkv,
                seqlen_q=sq,
                seqlen_k=sq if sk is None else sk,
                hdim_q=d,
                hdim_v=d,
                arch=arch,
                mask_type=2 if bottom_right else 1,
                dtype=dtype,
                use_fp8=True,
                **({"num_cus": 256} if bottom_right else {}),
            ),
            "gfx950_dense_grid",
        ).kernel_spec
    finally:
        au._RESOLVED_ATTENTION_ARCH = old


@pytest.mark.parametrize(
    "case",
    [
        ("bf16", 64, 16, 4, {}),
        ("bf16", 128, 16, 4, {}),
        ("fp16", 64, 16, 4, {}),
        ("fp16", 128, 16, 4, {}),
        ("bf16", 128, 4, 1, {"sq": 512, "sk": 1024, "bottom_right": True}),
        ("bf16", 64, 4, 1, {"sq": 512, "sk": 1024, "bottom_right": True}),
    ],
    ids=lambda c: f"{c[0]}-d{c[1]}-hq{c[2]}-hkv{c[3]}" + ("-br-split" if c[4] else ""),
)
def test_packed_fp8_kv_tensors_pass_admission_at_the_fp8_numeric_shapes(case):
    """The admission the runner makes accepts the packed tensors of the fp8
    numeric tests' shipped specs (the bottom-right ones split the KV range): the
    canonical strides come back, and the body stays packed-only."""
    dtype, d, hq, hkv, extra = case
    spec = _fp8_factory_spec(dtype, d, hq, hkv, **extra)
    assert spec.kv_storage_dtype == "fp8e4m3" and spec.kv_elem_bytes == 1
    assert (spec.kv_segments > 1) == bool(extra)
    assert k950.body_serves_strides(spec) is False
    views = _packed_views(spec)
    strides = admit_dense_launch_strides(
        spec, **views, serves_strides=k950.body_serves_strides(spec)
    )
    assert strides == packed_launch_strides(spec)


@pytest.mark.parametrize(
    ("body", "field"),
    [("grid", "ragged"), ("grid", "varlen"), ("persistent", "ragged")],
    ids=lambda v: f"gfx950-{v}" if v in ("grid", "persistent") else v,
)
def test_a_gfx950_spec_with_per_sequence_or_partial_extents_stays_packed_only(
    body, field
):
    """Ragged and varlen specs bound K/V by partial tiles or per-sequence offsets,
    not by a dense slab: the grid body ignores their strides and the runner admits
    packed tensors only, so a head-major view is refused with its reason."""
    spec = dataclasses.replace(_spec(f"gfx950-{body}"), **{field: True})
    assert k950.body_serves_strides(spec) is False
    views = _packed_views(spec)
    if field == "varlen":
        # Packed [total_tokens, H, D] storage: any tensor that is not contiguous is
        # refused by the varlen branch of the admission.
        shape = (spec.batch * spec.seqlen_q, spec.num_query_heads, spec.head_size)
        views["q"] = _View(
            shape,
            (spec.head_size * 2 * spec.num_query_heads, spec.head_size * 2, 1),
            _ADDRESSES["q"],
        )
    else:
        views["q"] = _bhsd_view(views["q"])
    with pytest.raises(ValueError, match=r"q is not packed"):
        admit_dense_launch_strides(spec, **views, serves_strides=False)


def _varlen_or_paged_spec(case):
    """A factory spec turned varlen (gfx950) or paged (batch 1, 64-row pages)."""
    arm, field = case.rsplit("-", 1)
    if field == "varlen":
        return dataclasses.replace(_spec(arm), varlen=True)
    spec = _spec(arm, batch=1)
    return dataclasses.replace(
        spec, paged=True, block_size=64, num_kv_blocks=spec.seqlen_kv // 64
    )


@pytest.mark.parametrize("tensor", ["q", "k", "v", "o"])
@pytest.mark.parametrize(
    "case", ["gfx950-grid-varlen", "gfx950-grid-paged", "gfx942-grid-paged"]
)
def test_a_varlen_or_paged_spec_refuses_a_misaligned_base(case, tensor):
    """Varlen and paged bodies read packed storage and claim 16-byte aligned
    Q/K/V/O pointers like every body: a base that is not, as a view or a raw
    address, is refused with the ``pointer_alignment`` reason."""
    spec = _varlen_or_paged_spec(case)
    module = _MODULES[case.split("-")[0]]
    assert module.body_serves_strides(spec) is False
    addresses = {t: _ADDRESSES[t] for t in DENSE_STRIDE_TENSORS}
    views = _packed_views(spec)
    for operands in (addresses, views):
        assert admit_dense_launch_strides(
            spec, **operands, serves_strides=False
        ) == packed_launch_strides(spec)
    for operands, shifted in (
        (addresses, _ADDRESSES[tensor] + 8),
        (views, views[tensor].with_ptr(_ADDRESSES[tensor] + 8)),
    ):
        with pytest.raises(
            ValueError,
            match=rf"\(pointer_alignment\): {tensor} base is 8-byte aligned, needs 16",
        ):
            admit_dense_launch_strides(
                spec, **dict(operands, **{tensor: shifted}), serves_strides=False
            )


@pytest.mark.parametrize("tensor", ["q", "k", "v", "o"])
@pytest.mark.parametrize("arm", _PACKED_ONLY_ARMS)
def test_a_view_that_is_not_packed_is_refused_before_any_compile(
    arm, tensor, monkeypatch
):
    spec = _spec(arm)
    module = _MODULES[_ARMS[arm][0]]
    _forbid_build(monkeypatch, module)
    views = _packed_views(spec)
    views[tensor] = _bhsd_view(views[tensor])
    assert not views[tensor].is_contiguous()
    with pytest.raises(ValueError, match=rf"{tensor} is not packed"):
        _run(module, spec, views)


@pytest.mark.parametrize("tensor", ["q", "k", "v", "o"])
@pytest.mark.parametrize("arm", _ARMS)
def test_a_misaligned_base_pointer_is_refused_before_any_compile(
    arm, tensor, monkeypatch
):
    spec = _spec(arm)
    module = _MODULES[_ARMS[arm][0]]
    _forbid_build(monkeypatch, module)
    views = _packed_views(spec)
    views[tensor] = views[tensor].with_ptr(_ADDRESSES[tensor] + 8)
    assert views[tensor].data_ptr() % 16 == 8
    with pytest.raises(ValueError, match="pointer_alignment"):
        _run(module, spec, views)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("arm", _ARMS)
def test_packed_views_launch_with_the_canonical_packed_strides(arm, batch, monkeypatch):
    spec = _spec(arm, batch=batch)
    module = _MODULES[_ARMS[arm][0]]
    launches, signatures = _record_launches(monkeypatch, module, spec)
    views = _packed_views(spec)
    assert _run(module, spec, views) is views["o"]
    (values,) = launches
    # Extent-1 axes (batch 1) are canonicalized: batch and head stride 0, token D.
    want = packed_launch_strides(spec)
    assert {n: values[n] for n in DENSE_STRIDE_NAMES} == want
    # The launcher binds every kernarg of the signature it was built with, and no other.
    (signature,) = signatures
    assert set(values) == {a["name"] for a in signature}


# A bias launch checks the bias against q's device, which a raw address does not
# have, so the bias arms refuse raw addresses (the test after this one).
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("arm", [a for a in _ARMS if a not in _BIAS_ARMS])
def test_objects_without_strides_are_taken_as_packed(arm, batch, monkeypatch):
    """The SDPA replay passes raw addresses: no ``.stride()``, so the packed layout.
    They launch with exactly the stride values ``attention_dense_runtime_args``
    gives, which the SDPA export freezes, so a replay repeats the launch."""
    spec = _spec(arm, batch=batch)
    module = _MODULES[_ARMS[arm][0]]
    launches, _ = _record_launches(monkeypatch, module, spec)
    addresses = {t: _ADDRESSES[t] for t in DENSE_STRIDE_TENSORS}
    assert all(not hasattr(a, "stride") for a in addresses.values())
    assert _run(module, spec, addresses) == addresses["o"]
    (values,) = launches
    assert {n: values[n] for n in DENSE_STRIDE_NAMES} == packed_launch_strides(spec)
    frozen = module.attention_dense_runtime_args(spec)
    assert {n: values[n] for n in DENSE_STRIDE_NAMES} == {
        n: frozen[n] for n in DENSE_STRIDE_NAMES
    }


@pytest.mark.parametrize("use_sinks", [False, True])
@pytest.mark.parametrize("arm", _BIAS_ARMS)
def test_a_bias_launch_refuses_raw_addresses_before_any_compile(
    arm, use_sinks, monkeypatch
):
    """A bias launch checks the bias against q's device, and a raw address (the
    SDPA replay's form) has none: the runner refuses it with a ``ValueError`` that
    names what is missing, before anything is built or compiled, and before the
    sinks checks read q's dtype when the launch has sinks too. The same launch
    with views goes through (the packed-views test above)."""
    spec = dataclasses.replace(_spec(arm), use_sinks=use_sinks)
    module = _MODULES[_ARMS[arm][0]]
    _forbid_build(monkeypatch, module)
    addresses = {t: _ADDRESSES[t] for t in DENSE_STRIDE_TENSORS}
    with pytest.raises(
        ValueError, match=r"use_additive_bias=True requires q with a device.*got int"
    ):
        _run(module, spec, addresses)


@pytest.mark.parametrize("arm", ["gfx950-grid", "gfx942-grid", "gfx942-persistent"])
def test_a_sinks_launch_refuses_a_raw_q_before_any_compile(arm, monkeypatch):
    """The sinks are checked against q's dtype, which a raw address does not have:
    the runner refuses it with a ``ValueError`` that names what is missing, before
    anything is built or compiled."""
    spec = dataclasses.replace(_spec(arm), use_sinks=True)
    module = _MODULES[_ARMS[arm][0]]
    _forbid_build(monkeypatch, module)
    addresses = {t: _ADDRESSES[t] for t in DENSE_STRIDE_TENSORS}
    with pytest.raises(
        ValueError, match=r"use_sinks=True requires q with a dtype.*got int"
    ):
        _run(module, spec, addresses)


def _admitted_strides(spec, views):
    """The 12 values the contract admits for four views, computed without the runner."""
    verdict = dense_strides_admit(
        spec, {t: dense_operand_from_view(views[t]) for t in DENSE_STRIDE_TENSORS}
    )
    assert verdict.ok, verdict
    return dict(verdict.strides)


@pytest.mark.parametrize("tensor", ["q", "k", "v", "o"])
@pytest.mark.parametrize("arm", _SERVED_ARMS)
def test_a_served_body_launches_a_strided_view_with_its_admitted_strides(
    arm, tensor, monkeypatch
):
    """The same head-major view the packed-only bodies refuse is admitted by a body
    that reads the strides, and reaches the launcher with the contract's values."""
    spec = _spec(arm)
    module = _MODULES[_ARMS[arm][0]]
    launches, signatures = _record_launches(monkeypatch, module, spec)
    views = _packed_views(spec)
    views[tensor] = _bhsd_view(views[tensor])
    want = _admitted_strides(spec, views)
    keys = [n for n in DENSE_STRIDE_NAMES if n.startswith(f"stride_{tensor}_")]
    packed = packed_launch_strides(spec)
    assert any(want[n] != packed[n] for n in keys)
    assert _run(module, spec, views) is views["o"]
    (values,) = launches
    assert {n: values[n] for n in DENSE_STRIDE_NAMES} == want
    (signature,) = signatures
    assert set(values) == {a["name"] for a in signature}


@pytest.mark.parametrize("arm", _SERVED_ARMS)
def test_a_served_body_refuses_a_view_the_contract_declines(arm, monkeypatch):
    """A head-dim stride of 2 is declined by the contract with its reason, before
    anything is built or compiled."""
    spec = _spec(arm)
    module = _MODULES[_ARMS[arm][0]]
    _forbid_build(monkeypatch, module)
    views = _packed_views(spec)
    b, s, h, d = _shapes(spec)["q"]
    views["q"] = _View(
        (b, s, h, d), (s * h * d * 2, h * d * 2, d * 2, 2), _ADDRESSES["q"]
    )
    with pytest.raises(ValueError, match=r"strides declined \(head_dim_stride\)"):
        _run(module, spec, views)


@pytest.mark.parametrize("arm", ["gfx942-grid", "gfx950-grid"])
def test_the_kv_segment_body_reads_strided_q_but_writes_packed_o(arm, monkeypatch):
    """With kv_segments > 1 the grid reads strided Q/K/V but O goes through the
    combine kernel, which writes packed [B, S, H, D]: a strided O is refused."""
    module = _MODULES[_ARMS[arm][0]]
    spec = dataclasses.replace(_spec(arm), kv_segments=2)
    assert module.body_serves_strides(spec) is True
    _forbid_build(monkeypatch, module)
    views = _packed_views(spec)
    views["o"] = _bhsd_view(views["o"])
    with pytest.raises(ValueError, match=r"o is not packed: kv_segments > 1"):
        _run(module, spec, views)
    # The runner's own admission (the call run_attention_dense_torch makes) admits
    # a strided Q for the same spec and returns the contract's values.
    views = _packed_views(spec)
    views["q"] = _bhsd_view(views["q"])
    got = admit_dense_launch_strides(
        spec, **views, serves_strides=module.body_serves_strides(spec)
    )
    assert got == _admitted_strides(spec, views)
    assert got["stride_q_head"] != packed_launch_strides(spec)["stride_q_head"]
