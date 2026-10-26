#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host driver + fp64 parity for the GDN mode of the KDA chunkwise prefill kernel.

GDN is KDA with a scalar forget-gate. This targets the **split (raw-prep) path**:
the raw prep kernel fuses q/k L2-norm, the GDN softplus gate, and beta=sigmoid, so
inputs here are RAW (un-cooked). The oracle cooks them identically and expands q/k
from Hk key-heads to Hv value-heads (value head vh reads key head vh // kv_group).

Run (on a gfx950 box)::

    python gdn_prefill.py            # parity sweep, split path, MHA + GQA
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import warnings

try:
    import rocke  # noqa: F401
except ImportError:  # running as a bare script outside the editable install
    _HERE = os.path.dirname(__file__)
    _RK = os.path.abspath(os.path.join(_HERE, "../../../.."))
    sys.path[:0] = [_RK + "/library", _RK + "/platform/python"]

import torch  # noqa: E402

from rocke.helpers.activations import LOG2E, SOFTPLUS_THRESHOLD
from kernels.gfx950.kda_chunkwise import EXP2_CLAMP
import torch.nn.functional as F  # noqa: E402

from kernels.gfx950.kda_chunkwise import KdaChunkScanSpec, KdaTileSpec  # noqa: E402

if __package__:
    from . import kda_chunk_prep as prep_mod  # noqa: E402
    from . import kda_chunk_split as split_mod  # noqa: E402
    from .kda_chunk_fused import ref_token_serial  # noqa: E402
else:
    import kda_chunk_prep as prep_mod  # noqa: E402
    import kda_chunk_split as split_mod  # noqa: E402
    from kda_chunk_fused import ref_token_serial  # noqa: E402

TOL = 3e-2


# The generator cannot see the spec. chunk=32 is the only dispatchable chunk
# today and carries the TIGHTEST limit (a smaller chunk permits more decay), so
# sizing against it keeps every draw inside every reachable bound.
_GENERATOR_CHUNK = 32
# Stay clear of the boundary rather than on it -- see make_gdn_inputs.
_GENERATOR_MARGIN = 0.9


def make_gdn_inputs(B, Hv, Hk, T, DK, DV, gate_low=-0.5, seed=0, device="cuda"):
    """RAW GDN inputs: un-normalized q/k [B,T,Hk,DK], v [B,T,Hv,DV], raw a/beta [B,T,Hv]."""
    g = torch.Generator(device=device).manual_seed(seed)
    kw = dict(device=device, generator=g)
    q = torch.randn(B, T, Hk, DK, dtype=torch.float32, **kw).bfloat16()
    k = torch.randn(B, T, Hk, DK, dtype=torch.float32, **kw).bfloat16()
    v = (torch.randn(B, T, Hv, DV, dtype=torch.float32, **kw) * 0.2).bfloat16()
    a = (gate_low * torch.rand(B, T, Hv, dtype=torch.float32, **kw)).contiguous()
    beta = torch.rand(B, T, Hv, dtype=torch.float32, **kw).contiguous()
    dt_bias = torch.zeros(Hv, dtype=torch.float32, device=device)
    # Draw the decay rate INSIDE the bound launch_gdn enforces, so no shape can
    # generate an input the guard then refuses. peak = exp(a_log) * softplus(a +
    # dt_bias), and softplus saturates at ln2 for the a <= 0 drawn here, so the
    # rate ceiling is limit / ln2. _GENERATOR_MARGIN keeps the draw clear of the
    # boundary: the guard rejects on peak > limit, and landing exactly on it
    # would make rejection a rounding accident -- intermittent, and only on the
    # GPU-gated shapes where Hv=32 draws enough samples to reach the ceiling.
    #
    # This SUPERSEDES the earlier exp(a_log) <= 8 bound (vault note 2026-09-04,
    # job 265), which was measured where the output leaves a 3e-2 tolerance --
    # the knee, peak ~5.545. The guard promises exactness, not a tolerance, so
    # its bound is the clamp onset (5.4585 at chunk=32) and the generator
    # follows it. Two definitions of "supported" cannot both stand; this is the
    # operative one. randn() alone draws exp>20 in the tail, which made the
    # parity latently flaky as head count grew.
    rate_ceiling = _GENERATOR_MARGIN * decay_limit_for_chunk(_GENERATOR_CHUNK)
    a_log = torch.randn(Hv, dtype=torch.float32, **kw).clamp(
        max=math.log(rate_ceiling / math.log(2.0))
    )
    return q, k, v, a, beta, a_log, dt_bias


def decay_limit_for_chunk(chunk: int) -> float:
    """Largest per-token decay exponent a chunk of ``chunk`` tokens can carry.

    The kernel does not apply the decay token by token: it factors a whole
    chunk, so the exponents ACCUMULATE and the quantity that can overflow is
    decay x tokens-per-chunk. The midpoint factoring puts the worst case at
    ``chunk / 2`` tokens, and the hardware clamp is ``EXP2_CLAMP`` in base 2,
    hence ``EXP2_CLAMP / (log2(e) * chunk/2)``.

    That gives 5.46 at chunk=32 and 10.92 at chunk=16, measured on gfx950: the
    knee sits at 5.5-5.6 for chunk=32 and 11.5-12.0 for chunk=16, the same
    relationship, so the bound is derived from the clamp rather than fitted.

    The bound is the point the clamp starts to saturate -- where the answer
    stops being exact, NOT where it leaves a tolerance. That is the contract
    this guard enforces.

    ``chunk=16`` is not reachable through the dispatcher today (no
    ``value_splits`` band produces a valid chunk=16 scan spec), so the
    chunk-awareness is currently defensive rather than load-bearing.
    """
    return EXP2_CLAMP / (LOG2E * (chunk / 2.0))


def reject_if_decay_out_of_range(a_log, a, dt_bias, *, chunk):
    """Raise if the intra-chunk decay exponent exceeds what the clamp can carry.

    Past this bound ``exp2`` saturates and the kernel returns a finite, bounded,
    WRONG answer -- no NaN, no fault. Measured on gfx950: at chunk=32 the output
    relative error is 7.6e-3 at decay 5.0 and 0.93 at 6.0, while the final STATE
    error stays near 3e-3, so a serving loop that validates only its carried
    state sees nothing wrong.

    Raises rather than warns: ``warnings.warn`` is filtered by default in most
    serving stacks, and a silent wrong answer is exactly what this exists to
    stop. Cheap host reduction -- call once per problem, not inside a timed
    launch loop. Returns the observed peak decay exponent.
    """
    limit = decay_limit_for_chunk(chunk)
    x = a.float() + dt_bias.float()
    softplus = torch.where(x > SOFTPLUS_THRESHOLD, x, torch.log1p(torch.exp(x)))
    peak = (torch.exp(a_log.float()) * softplus).abs().max().item()
    if peak > limit:
        raise ValueError(
            f"GDN decay exponent peak {peak:.3f} exceeds what a {chunk}-token "
            f"chunk can carry ({limit:.3f}); past this the exp2 clamp saturates "
            f"and the kernel returns a bounded but wrong result"
        )
    return peak


def ref_gdn_raw(q, k, v, a, beta, a_log, dt_bias, scale, kv_group, h0=None):
    """fp64 oracle: cook like the kernel, expand Hk->Hv, then token-serial."""
    B, T, Hk, DK = q.shape
    Hv = v.shape[2]
    kidx = torch.arange(Hv, device=q.device) // kv_group  # value head -> key head
    qn = F.normalize(q.float(), dim=-1)[:, :, kidx, :]  # [B,T,Hv,DK]
    kn = F.normalize(k.float(), dim=-1)[:, :, kidx, :]
    x = a.float() + dt_bias.float()[None, None, :]  # [B,T,Hv]
    sp = torch.where(x > SOFTPLUS_THRESHOLD, x, torch.log1p(torch.exp(x)))
    gate = -torch.exp(a_log.float())[None, None, :] * sp  # [B,T,Hv]
    gate = gate[..., None].expand(B, T, Hv, DK)  # broadcast to DK
    bb = torch.sigmoid(beta.float())  # [B,T,Hv]
    qbh = qn.permute(0, 2, 1, 3).to(torch.bfloat16)
    kbh = kn.permute(0, 2, 1, 3).to(torch.bfloat16)
    vbh = v.float().permute(0, 2, 1, 3)
    gbh = gate.permute(0, 2, 1, 3)
    bbh = bb.permute(0, 2, 1)
    return ref_token_serial(qbh, kbh, vbh, gbh, bbh, scale, h0=h0)


def gdn_specs(DK, DV, kv_group, with_h0):
    """A plain scan spec and the raw GDN prep spec derived from it (flags added)."""
    scan = KdaChunkScanSpec(
        head_k=DK,
        head_v=DV,
        tile=KdaTileSpec(chunk=32),
        value_splits=1,
        token_major_io=True,
        has_initial_state=with_h0,
    )
    prep = dataclasses.replace(
        split_mod.prep_spec_of(scan, raw=True), gate_kind="gdn", kv_group=kv_group
    )
    return scan, prep


def _validate_prefill_inputs(scan, prep, q, k, v, a, beta, a_log, dt_bias, h0):
    """Reject tensors whose shape or dtype disagrees with the spec.

    Every address this kernel computes comes from SPEC constants -- head counts,
    head dims, chunk -- never from the tensors' own shapes, and it emits no
    buffer descriptor, so there is no ``num_records`` to clamp an over-reach. A
    tensor narrower than the spec is read past its end with nothing to catch it.

    dtype matters as much as shape and no device-side check can see it: bf16 and
    f16 are both 16 bits, so a mismatched tensor computes every address
    identically and merely decodes the bytes under the wrong rule. Mirrors
    ``_validate_decode_inputs`` in the GDN decode driver.

    Metadata only -- ``.shape`` and ``.dtype``, no ``.item()`` -- so this is
    sync-free and safe to call on every launch.
    """
    # Take the head DIMS from the spec, not from the tensors: the kernel is
    # compiled against spec constants, so "what the tensor says" is the thing
    # under test, not the reference. Deriving DK/DV from q/v would make their
    # rows compare each tensor to itself -- a check that cannot fire.
    #
    # B, T and Hv have no spec source (the request carries them, the spec does
    # not), so they come from the tensors and the rows below check every OTHER
    # tensor agrees with them.
    B, T = q.shape[0], q.shape[1]
    Hk, Hv = q.shape[2], v.shape[2]
    DK, DV = prep.head_k, scan.head_v
    io_dt, st_dt = q.dtype, torch.float32
    want = {
        "q": ((B, T, Hk, DK), io_dt),
        "k": ((B, T, Hk, DK), io_dt),
        "v": ((B, T, Hv, DV), io_dt),
        "a": ((B, T, Hv), st_dt),
        "beta": ((B, T, Hv), st_dt),
        "a_log": ((Hv,), st_dt),
        "dt_bias": ((Hv,), st_dt),
    }
    got = dict(q=q, k=k, v=v, a=a, beta=beta, a_log=a_log, dt_bias=dt_bias)
    for name, (want_shape, want_dt) in want.items():
        t = got[name]
        if tuple(t.shape) != want_shape:
            raise ValueError(
                f"{name} must be {want_shape} to agree with q/v; got "
                f"{tuple(t.shape)} -- the kernel addresses it from spec "
                f"constants and would read past its end"
            )
        if t.dtype is not want_dt:
            raise ValueError(
                f"{name} dtype {t.dtype} != {want_dt}; the kernel is compiled "
                f"with a {want_dt} pointer and would reinterpret these bytes"
            )
    if h0 is not None and tuple(h0.shape) != (B, Hv, DK, DV):
        raise ValueError(f"h0 must be {(B, Hv, DK, DV)}; got {tuple(h0.shape)}")
    if prep.kv_group != max(Hv // Hk, 1):
        raise ValueError(
            f"kv_group {prep.kv_group} disagrees with the tensors "
            f"(Hv {Hv} / Hk {Hk}); the GQA gather would read the wrong key-head"
        )


def launch_gdn(scan, prep, q, k, v, a, beta, a_log, dt_bias, h0=None):
    """Raw split GDN launch: heads=Hv (value heads); q/k carry Hk key-heads.

    Rejects a decay steeper than the chunk can carry before launching: past that
    bound the exp2 clamp saturates and the kernel returns a bounded, wrong
    answer that the final state does not reveal.
    """
    B, T, Hk, DK = q.shape
    Hv, DV = v.shape[2], v.shape[-1]
    C = scan.tile.chunk
    # T % C != 0 would silently process floor(T/C) chunks and return the tail
    # rows of `o` uninitialised -- wrong numbers, no fault. The dispatcher
    # rejects this (dispatch/gdn/prefill_common.py), but callers reaching the
    # builder directly bypass that, so the launch boundary enforces it too.
    if T % C:
        raise ValueError(
            f"seqlen {T} must be a multiple of chunk {C}; "
            f"this family has no varlen path"
        )
    _validate_prefill_inputs(scan, prep, q, k, v, a, beta, a_log, dt_bias, h0)
    reject_if_decay_out_of_range(a_log, a, dt_bias, chunk=C)
    BH, NC = B * Hv, T // C
    nt = BH * NC
    ws = prep_mod.alloc_tiles(nt, prep)
    o = torch.empty_like(v)
    ht = torch.zeros(B * Hv, DV, DK, dtype=torch.float32, device=q.device)
    h0t = None
    if h0 is not None:
        h0t = h0.transpose(-1, -2).contiguous().view(B * Hv, DV, DK)
    prep_mod.run_prep(
        prep,
        q,
        k,
        a,
        beta,
        ws,
        DK**-0.5,
        batch=B,
        heads=Hv,
        tseq=T,
        nc=NC,
        a_log=a_log,
        dt_bias=dt_bias,
    )
    split_mod.run_scan(
        scan,
        ws,
        v,
        o,
        ht,
        BH,
        NC,
        h0=h0t,
        batch=B,
        heads=Hv,
        tseq=T,
    )
    return o, ht.view(B, Hv, DV, DK).transpose(-1, -2)


def check_gdn(
    B,
    Hv,
    Hk,
    T,
    DK,
    DV,
    gate_low=-0.5,
    with_h0=False,
    seed=0,
    specs=None,
):
    # Hv % Hk != 0 would floor to a kv_group the kernel then strides q/k by,
    # reading past the end of a tensor that has fewer key-heads than the stride
    # implies -- an out-of-bounds read, not a wrong answer. Hv=6/Hk=4 gives
    # kv_group=1 and visits khead up to 5 where 3 is the maximum.
    if Hv % Hk:
        raise ValueError(
            f"num_v_heads {Hv} must be a multiple of num_k_heads {Hk}; "
            f"GQA gathers key-head head//kv_group"
        )
    kv_group = Hv // Hk
    q, k, v, a, beta, a_log, dt_bias = make_gdn_inputs(
        B, Hv, Hk, T, DK, DV, gate_low=gate_low, seed=seed
    )
    h0 = None
    if with_h0:
        gen = torch.Generator(device="cuda").manual_seed(7)
        h0 = (torch.randn(B, Hv, DK, DV, device="cuda", generator=gen) * 0.1).float()
    scan, prep = gdn_specs(DK, DV, kv_group, with_h0) if specs is None else specs
    o, ht = launch_gdn(scan, prep, q, k, v, a, beta, a_log, dt_bias, h0=h0)
    torch.cuda.synchronize()
    o_ref, s_ref = ref_gdn_raw(
        q, k, v, a, beta, a_log, dt_bias, DK**-0.5, kv_group, h0=h0
    )
    worst = 0.0
    for got, ref in ((o.permute(0, 2, 1, 3).float(), o_ref), (ht.float(), s_ref)):
        rel = (got - ref).abs().max() / ref.abs().max().clamp_min(1e-6)
        worst = max(worst, rel.item())
    return worst


def main() -> int:
    if not torch.cuda.is_available():
        print("no HIP device", file=sys.stderr)
        return 2
    worst = 0.0
    for Hv, Hk in ((4, 4), (8, 4)):
        for gate_low in (-0.5, -5.0):
            for with_h0 in (False, True):
                w = check_gdn(
                    2,
                    Hv,
                    Hk,
                    256,
                    128,
                    128,
                    gate_low=gate_low,
                    with_h0=with_h0,
                )
                worst = max(worst, w)
                tag = f"Hv{Hv}/Hk{Hk} gate[{gate_low},0] h0={with_h0}"
                print(f"  {tag:32s} rel={w:.3e} {'PASS' if w <= TOL else 'FAIL'}")
    print(f"worst={worst:.3e} tol={TOL:.1e}")
    return 0 if worst <= TOL else 1


if __name__ == "__main__":
    raise SystemExit(main())
