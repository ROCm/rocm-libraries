# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Launch glue of the general attention backward.

* CPU: the packed kernel arguments of every plan launch follow the
  ``rocke.attn_bwd.v3`` lists (pointers by role, workspace sub-buffers at the
  plan's offsets, 0 for unused pointers, scalars in order); a missing tensor
  role, a short workspace and a misaligned workspace raise; the plan's
  prep / convert launches name, size and shape the shared aux kernels;
  every stride kernarg is packed on both sides of ``2^31`` (batch and head
  strides are i64, token strides are bounded by the predicate), and every
  request the predicate admits packs without error (seeded fuzz).
* Device: ``run_attn_bwd`` allocates nothing and issues no memset (the
  runtime allocator and memset are patched to raise during the run).
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from kernels.common.attention_bwd import attn_bwd_params
from kernels.common.attention_bwd_aux import (
    attn_bwd_aux_block_threads,
    attn_bwd_convert_grid,
    attn_bwd_prep_grid,
)
from kernels.common.attention_bwd_plan import (
    FEATURE_GROUPS,
    AttnBwdPolicy,
    AttnBwdRequest,
    attn_bwd_plan,
    attn_bwd_support,
)
from kernels.common.attention_bwd_run import (
    WORKSPACE_ALIGN,
    pack_attn_bwd_args,
    run_attn_bwd,
)

from .sdpa.bwd_cases import get_bwd_case
from .sdpa.bwd_kernel_cases import config_policy, request_from_case

ARCHS = ("gfx942", "gfx950", "gfx1151")
_TENSORS = {
    n: 0x1000 * (i + 1) for i, n in enumerate("q k v o do lse dq dk dv".split())
}


def _plan(arch, case_id="bwd_dtype_fp16_d64_mha"):
    req = request_from_case(get_bwd_case(case_id))
    return attn_bwd_plan(req, arch, policy=config_policy("default", arch))


def _unpack(stage, blob):
    out, pos = {}, 0
    for name, fmt in attn_bwd_params(stage):
        size = 8 if fmt in ("Q", "q") else 4
        out[name] = struct.unpack("<" + fmt, blob[pos : pos + size])[0]
        pos += size
    assert pos == len(blob)
    return out


@pytest.mark.parametrize("arch", ARCHS)
def test_packed_arguments_follow_the_abi(arch):
    plan = _plan(arch)
    ws = 0x100000
    for step in plan.launches:
        blob = pack_attn_bwd_args(
            step, tensors=_TENSORS, workspace=ws, workspace_layout=plan.workspace_layout
        )
        got = _unpack(step.stage, blob)
        for name, fmt in attn_bwd_params(step.stage):
            if fmt == "Q":
                role = step.pointers[name]
                if role is None:
                    want = 0
                elif role in plan.workspace_layout:
                    want = ws + plan.workspace_layout[role][0]
                else:
                    want = _TENSORS[role]
                assert got[name] == want, (step.stage, name)
            elif fmt == "f":
                assert got[name] == pytest.approx(step.scalars[name], rel=1e-7)
            else:
                assert got[name] == step.scalars[name], (step.stage, name)
    assert [s.stage for s in plan.launches] == ["prep", "main", "convert"]


def test_missing_role_raises():
    plan = _plan("gfx942")
    tensors = dict(_TENSORS)
    del tensors["dk"]
    with pytest.raises(ValueError, match="dk"):
        pack_attn_bwd_args(
            plan.launches[1], tensors=tensors, workspace=0, workspace_layout={}
        )


def test_short_or_misaligned_workspace_raises_before_any_launch(monkeypatch):
    from kernels.common import attention_bwd_run as run_mod

    def no_build(*a, **k):
        raise AssertionError("kernels must not be built for a rejected workspace")

    monkeypatch.setattr(run_mod, "attn_bwd_kernels", no_build)
    plan = _plan("gfx942")
    with pytest.raises(ValueError, match="smaller"):
        run_attn_bwd(
            plan, tensors=_TENSORS, workspace=0x10000,
            workspace_size=plan.workspace_bytes - 1,
        )  # fmt: skip
    with pytest.raises(ValueError, match="aligned"):
        run_attn_bwd(
            plan, tensors=_TENSORS, workspace=0x10000 + WORKSPACE_ALIGN // 2,
            workspace_size=plan.workspace_bytes,
        )  # fmt: skip


@pytest.mark.parametrize("arch", ARCHS)
def test_aux_launches_match_the_shared_kernels(arch):
    req = request_from_case(get_bwd_case("bwd_seqlen_100x17_none"))
    plan = attn_bwd_plan(req, arch, policy=config_policy("default", arch))
    prep, conv = plan.specs["prep"], plan.specs["convert"]
    p, c = plan.launches[0], plan.launches[2]
    assert p.kernel == prep.kernel_name("prep") and c.kernel == conv.kernel_name(
        "convert"
    )
    assert p.block == attn_bwd_aux_block_threads(prep, arch=arch)
    assert c.block == attn_bwd_aux_block_threads(conv, arch=arch)
    assert p.grid == attn_bwd_prep_grid(
        prep, arch=arch, s_q_max=req.s_q, s_kv_max=req.s_kv, h_q=req.h_q,
        h_k=req.h_k, h_v=req.h_v, batch=req.b, zero_kv=False,
    )  # fmt: skip
    assert c.grid == attn_bwd_convert_grid(
        conv, arch=arch, s_max=req.s_q, heads=req.h_q, batch=req.b
    )


# ---------------------------------------------------------------------------
# kernarg range: what the predicate admits packs without narrowing
# ---------------------------------------------------------------------------

# Everything declared and performance waived, so the plan and the packer are
# reached for every functional request.
_OPEN = AttnBwdPolicy(declared_groups=frozenset(FEATURE_GROUPS), waive_unverified=True)
_ALL_ROLES = {
    **_TENSORS,
    "seq_len_q": 0x20000,
    "seq_len_kv": 0x21000,
    "offsets_q": 0x22000,
    "offsets_kv": 0x23000,
}
_ROLES8 = ("q", "k", "v", "o", "do", "dq", "dk", "dv")


def _pack_all(req, arch="gfx942"):
    """``[(launch, unpacked kernargs)]`` of the plan of an admitted ``req``."""
    plan = attn_bwd_plan(req, arch, policy=_OPEN)
    out = []
    for step in plan.launches:
        blob = pack_attn_bwd_args(
            step, tensors=_ALL_ROLES, workspace=0x100000,
            workspace_layout=plan.workspace_layout,
        )  # fmt: skip
        out.append((step, _unpack(step.stage, blob)))
    return out


_EDGES = (2**31 - 1, 2**31, 2**31 + 64, 2**32 + 64, 2**40)


@pytest.mark.parametrize("which", ["b", "h"])
@pytest.mark.parametrize("value", _EDGES, ids=lambda v: f"{v:#x}")
@pytest.mark.parametrize("role", _ROLES8)
def test_batch_and_head_strides_pack_past_2_pow_31(role, value, which):
    """A batch or head stride of any tensor below, at and past ``2^31``
    elements is admitted and reaches its i64 kernarg unchanged (main and
    prep for the inputs, the convert launch for an atomic or direct output)."""
    d, s_len = 64, 16
    strides = {r: (2 * s_len * d, s_len * d, d, 1) for r in _ROLES8}
    _sb, sh, st, sd = strides[role]
    if which == "b":
        strides[role] = (value, sh, st, sd)
    else:
        strides[role] = (2 * value + s_len * d, value, st, sd)
    req = AttnBwdRequest(
        b=2, h_q=2, h_k=2, h_v=2, s_q=s_len, s_kv=s_len, d_qk=d, d_v=d,
        strides=strides,
    )  # fmt: skip
    v = attn_bwd_support(req, "gfx942", policy=_OPEN)
    assert v.ok, v
    want = req.tensor_strides(role)["bh".index(which)]
    seen = 0
    for step, got in _pack_all(req):
        if step.stage == "convert":
            if step.pointers["DST"] == role:
                assert got[f"d_{which}"] == want
                seen += 1
        elif f"{role}_{which}" in got:
            assert got[f"{role}_{which}"] == want
            seen += 1
    assert seen, f"{role}_{which} is not packed by any launch"


@pytest.mark.parametrize("value", _EDGES, ids=lambda v: f"{v:#x}")
@pytest.mark.parametrize("which", ["b", "h"])
def test_lse_batch_and_head_strides_pack_past_2_pow_31(which, value):
    lse = (2 * value, value, 1, 1) if which == "h" else (value, 64, 1, 1)
    req = AttnBwdRequest(
        b=2, h_q=2, h_k=2, h_v=2, s_q=64, s_kv=64, d_qk=64, d_v=64,
        lse_dims=(2, 2, 64, 1), lse_strides=lse,
    )  # fmt: skip
    assert attn_bwd_support(req, "gfx942", policy=_OPEN).ok
    prep = _pack_all(req)[0][1]
    assert prep[f"l_{which}"] == lse["bh".index(which)]


@pytest.mark.parametrize(
    "layout, dims, strides, want",
    [
        ("dense", (), (), (2 * 64, 64, 1)),
        ("dense", (2, 2, 64, 1), (1000, 3, 7, 5), (1000, 3, 7)),
        ("thd", (), (), (0, 1, 2)),
        ("thd", (128, 2, 1), (1, 131, 4), (0, 131, 1)),
        ("thd", (2, 2, 64, 1), (999, 1, 3, 4), (0, 1, 3)),
    ],
)
def test_every_accepted_lse_form_reaches_the_prep_kernargs(layout, dims, strides, want):
    """The packed prep arguments carry the LSE strides of every accepted form
    (THD: the B stride is unused and packs as 0)."""
    req = AttnBwdRequest(
        b=2, h_q=2, h_k=2, h_v=2, s_q=64, s_kv=64, d_qk=64, d_v=64,
        layout=layout, lse_dims=dims, lse_strides=strides,
    )  # fmt: skip
    assert attn_bwd_support(req, "gfx942", policy=_OPEN).ok
    step, prep = _pack_all(req)[0]
    assert step.stage == "prep"
    assert (prep["l_b"], prep["l_h"], prep["l_t"]) == want


@pytest.mark.parametrize("role", _ROLES8 + ("lse",))
def test_token_strides_are_declined_exactly_at_the_i32_bound(role):
    """Token strides stay i32 kernargs: ``S_max * stride + D`` (LSE:
    ``S_q * stride + 1``) just below ``2^31`` is admitted and packs; at or
    past ``2^31`` the predicate declines ``INDEX_RANGE``."""
    d, s_len = 64, 2
    extra = 1 if role == "lse" else d
    top = (2**31 - 1 - extra) // s_len
    for stride, ok in ((top, True), (top + 1, False), (2**31, False)):
        if role == "lse":
            kw = dict(
                lse_dims=(1, 1, s_len, 1),
                lse_strides=(s_len * stride, s_len * stride, stride, 1),
            )
        else:
            kw = dict(strides={role: (s_len * stride, s_len * stride, stride, 1)})
        req = AttnBwdRequest(
            b=1, h_q=1, h_k=1, h_v=1, s_q=s_len, s_kv=s_len, d_qk=d, d_v=d, **kw
        )
        v = attn_bwd_support(req, "gfx942", policy=_OPEN)
        if ok:
            assert v.ok, (role, stride, v)
            _pack_all(req)
        else:
            assert v.code == "INDEX_RANGE", (role, stride, v)


def test_a_stride_past_the_i64_range_is_declined():
    req = AttnBwdRequest(
        b=2, h_q=1, h_k=1, h_v=1, s_q=8, s_kv=8, d_qk=64, d_v=64,
        strides={"q": (2**63, 512, 64, 1)},
    )  # fmt: skip
    v = attn_bwd_support(req, "gfx942", policy=_OPEN)
    assert v.code == "INDEX_RANGE" and "i64" in v.detail


def test_packer_names_a_kernarg_out_of_its_range():
    plan = _plan("gfx942")
    step = plan.launches[1]
    bad = type(step)(
        step.stage, step.kernel, step.grid, step.block,
        {**step.scalars, "q_t": 2**31}, step.pointers,
    )  # fmt: skip
    with pytest.raises(ValueError, match="q_t"):
        pack_attn_bwd_args(
            bad, tensors=_TENSORS, workspace=0x100000,
            workspace_layout=plan.workspace_layout,
        )  # fmt: skip


def _random_request(rng):
    """One random request over the predicate's input space (strides near and
    far past ``2^31``, both layouts, every length form, unsupported dims)."""
    thd = rng.random() < 0.4
    d = int(rng.choice([32, 64, 128, 256, 48]))
    h_k = int(rng.choice([1, 2, 3]))
    h_v = h_k if rng.random() < 0.7 else int(rng.choice([1, 2, 3, 6]))
    h_q = int(np.lcm(h_k, h_v) * rng.choice([1, 2]))
    b = int(rng.choice([1, 2, 3, 70000]))
    s_q = int(rng.choice([1, 7, 64, 4096, 2**20, 2**29]))
    s_kv = int(rng.choice([1, 9, 64, 4096, 2**20]))

    def stride(natural):
        kind = rng.random()
        if kind < 0.5:
            return int(natural)
        if kind < 0.8:
            return (2**31 + int(rng.integers(-8, 8))) * int(rng.choice([1, 2, 512]))
        return int(rng.integers(1, 2**34))

    strides = {}
    for r in _ROLES8:
        if rng.random() < 0.5:
            heads = {"k": h_k, "dk": h_k, "v": h_v, "dv": h_v}.get(r, h_q)
            seq = s_kv if r in ("k", "v", "dk", "dv") else s_q
            tok = d * heads if thd else d
            strides[r] = (
                stride(heads * seq * d), stride(d if thd else seq * d),
                stride(tok) if rng.random() < 0.3 else tok, 1,
            )  # fmt: skip
    kw = {}
    if rng.random() < 0.4:
        if thd:
            kw.update(lse_dims=(0, h_q, 1), lse_strides=(stride(h_q), stride(1), 1))
        else:
            kw.update(
                lse_dims=(b, h_q, s_q, 1),
                lse_strides=(stride(h_q * s_q), stride(s_q), stride(1), 1),
            )
    if thd:
        kw.update(
            layout="thd",
            ragged_offset_dtype=str(rng.choice(["int32", "int64"])),
            ragged_offset_multiplier=int(rng.choice([1, 8, 2**31 - 1])),
        )
        if rng.random() < 0.5:
            kw.update(
                max_total_q=int(rng.integers(1, 2**31)),
                max_total_kv=int(rng.integers(1, 2**31)),
            )
    elif rng.random() < 0.3:
        kw.update(
            padding=True, has_seq_len_q=True, has_seq_len_kv=True,
            seq_len_stride=int(rng.choice([1, 3, 2**31 - 1])),
        )  # fmt: skip
    mask = str(rng.choice(["none", "tl", "br", "win"]))
    if mask == "tl":
        kw.update(causal=True)
    elif mask == "br":
        kw.update(causal_bottom_right=True)
    elif mask == "win":
        kw.update(left_bound=int(rng.integers(0, 2**30)), right_bound=0)
    return AttnBwdRequest(
        b=b, h_q=h_q, h_k=h_k, h_v=h_v, s_q=s_q, s_kv=s_kv, d_qk=d, d_v=d,
        strides=strides or None, tensor_alignment=int(rng.choice([16, 2])), **kw,
    )  # fmt: skip


@pytest.mark.parametrize("arch", ["gfx942", "gfx1151"])
def test_every_admitted_request_packs(arch):
    """Invariant: every launch of the plan of every request the predicate
    admits packs without error (each value fits its kernarg type)."""
    rng = np.random.default_rng(20261004)
    admitted = declined = 0
    for _ in range(1500):
        req = _random_request(rng)
        if not attn_bwd_support(req, arch, policy=_OPEN).ok:
            declined += 1
            continue
        _pack_all(req, arch)
        admitted += 1
    # the generator reaches both sides of the predicate
    assert admitted > 100 and declined > 100, (admitted, declined)


# ---------------------------------------------------------------------------
# device: no allocation, no memset inside the run
# ---------------------------------------------------------------------------

_ARCH = None
try:
    from rocke.runtime.hip_module import get_device_arch

    _ARCH = get_device_arch(0)
except Exception:  # noqa: BLE001 - no runtime / no device
    _ARCH = None


@pytest.mark.gpu
@pytest.mark.skipif(
    _ARCH not in ("gfx942", "gfx950", "gfx1151", "gfx1201"),
    reason=f"needs a backward device; device is {_ARCH}",
)
def test_run_allocates_nothing(monkeypatch):
    from rocke.runtime import hip_module

    from ._attention_bwd_harness import run_case
    from .sdpa.bwd_cases import compare_bwd

    case = get_bwd_case("bwd_seqlen_17x17_none")
    run_case(case, arch=_ARCH, config="debug")  # compile and load outside the check
    real_alloc, real_memset = hip_module.Runtime.alloc, hip_module.Runtime.memset
    state = {"in_run": False}
    from kernels.common import attention_bwd_run as run_mod

    real_run = run_mod.run_attn_bwd

    def guarded_run(*a, **k):
        state["in_run"] = True
        try:
            return real_run(*a, **k)
        finally:
            state["in_run"] = False

    def alloc(self, nbytes):
        if state["in_run"]:
            raise AssertionError("device allocation inside run_attn_bwd")
        return real_alloc(self, nbytes)

    def memset(self, *a, **k):
        if state["in_run"]:
            raise AssertionError("memset inside run_attn_bwd")
        return real_memset(self, *a, **k)

    monkeypatch.setattr(hip_module.Runtime, "alloc", alloc)
    monkeypatch.setattr(hip_module.Runtime, "memset", memset)
    from . import _attention_bwd_harness as harness

    monkeypatch.setattr(harness, "run_attn_bwd", guarded_run)
    run = run_case(case, arch=_ARCH, config="debug")
    assert not compare_bwd(case, run.dq, run.dk, run.dv, run.ref)
    assert np.isfinite(run.dq).all()
