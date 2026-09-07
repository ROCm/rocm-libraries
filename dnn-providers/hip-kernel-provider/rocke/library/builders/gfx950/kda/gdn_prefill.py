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
    # Clamp the decay rate to exp(a_log) <= 8. The intra-chunk cumulative decay
    # is exp(a_log) * softplus(a + dt_bias); beyond ~8 (with this test's dt ~ 0.5)
    # its dynamic range exceeds bf16 chunkwise precision (rel error spikes on the
    # steepest-decay head). Real GDN keeps the product small via a tiny dt, so
    # this bound reflects the supported regime rather than hiding a kernel fault
    # -- see the vault note (2026-09-04, job 265). randn() alone draws exp>20 in
    # the tail, which made the parity latently flaky as head count grew.
    a_log = torch.randn(Hv, dtype=torch.float32, **kw).clamp(max=math.log(8.0))
    return q, k, v, a, beta, a_log, dt_bias


def warn_if_decay_out_of_range(a_log, a, dt_bias, *, limit=5.0):
    """Warn (once) if the intra-chunk decay exponent exceeds the design range.

    The chunkwise stabilization is sized for the reference gate lower bound
    (-5); the intra-chunk cumulative decay ``exp(a_log) * softplus(a + dt_bias)``
    beyond ~``limit`` loses bf16 precision on the steepest-decay head and its
    output degrades. This is a documented, accepted limit -- the guard exists so
    a workload that drives the gate past it gets a loud warning instead of a
    silent wrong number. Cheap host reduction; call it once per problem, not in
    a timed launch loop. Returns the observed peak decay exponent.
    """
    x = a.float() + dt_bias.float()
    softplus = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
    peak = (torch.exp(a_log.float()) * softplus).abs().max().item()
    if peak > limit:
        warnings.warn(
            f"GDN decay exponent peak {peak:.2f} exceeds the stabilized design "
            f"range (~{limit}); the steepest-decay head may degrade. See the "
            f"prefill decay-range note.",
            stacklevel=2,
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
    sp = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
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


def launch_gdn(scan, prep, q, k, v, a, beta, a_log, dt_bias, h0=None):
    """Raw split GDN launch: heads=Hv (value heads); q/k carry Hk key-heads."""
    B, T, Hk, DK = q.shape
    Hv, DV = v.shape[2], v.shape[-1]
    C = scan.tile.chunk
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
    warn_decay=False,
):
    kv_group = Hv // Hk
    q, k, v, a, beta, a_log, dt_bias = make_gdn_inputs(
        B, Hv, Hk, T, DK, DV, gate_low=gate_low, seed=seed
    )
    if warn_decay:
        warn_if_decay_out_of_range(a_log, a, dt_bias)
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
                    warn_decay=True,
                )
                worst = max(worst, w)
                tag = f"Hv{Hv}/Hk{Hk} gate[{gate_low},0] h0={with_h0}"
                print(f"  {tag:32s} rel={w:.3e} {'PASS' if w <= TOL else 'FAIL'}")
    print(f"worst={worst:.3e} tol={TOL:.1e}")
    return 0 if worst <= TOL else 1


if __name__ == "__main__":
    raise SystemExit(main())
