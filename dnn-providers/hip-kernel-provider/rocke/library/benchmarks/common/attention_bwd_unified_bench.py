# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Relative timing of the unified (paged) backward against the general route.

For each cell (THD query spans over a paged KV cache) the script

1. draws the inputs on the device: Q, the paged K / V caches with a random
   injective (or shared-prefix) block table, dO; the contiguous K / V of every
   sequence are gathered through the table;
2. computes an fp64 torch reference on the gathered contiguous problem (the
   forward gives O, stored in the I/O dtype, and the natural-log LSE, stored
   fp32; the backward uses exactly those stored values), and
   gates both arms against it (norm-relative bound of the backward tests,
   global and per row) before any timing;
3. times the complete backward of each arm on one stream with device events,
   paired and interleaved: the unified route (``prep [-> page_zero] -> main
   -> convert(dQ) [-> page_convert x 2]``) on the paged caches, and the
   layout-general THD route (``prep -> main -> convert(s)``) on the gathered
   contiguous K / V (the equivalent work minus the gather);
4. appends one JSON record per (cell, arm) to ``--out`` (keep it outside the
   repository) and prints only the correctness verdict and the relative
   median of the two arms.

Run on a device node: ``python -m benchmarks.common.attention_bwd_unified_bench
--out <dir>/unified_vs_general.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass

TOL = {"fp16": (4e-3, 1e-4), "bf16": (2.5e-2, 1e-3)}  # (rtol, atol), the test bound
ROW_FLOOR = 0.05


@dataclass(frozen=True)
class UnifiedCell:
    id: str
    q_lens: tuple
    kv_lens: tuple
    page: int
    h_q: int
    h_kv: int
    d: int
    dtype: str = "bf16"
    mask: str = "br"  # "br" (bottom-right causal) | "none"
    shared_prefix: int = 0  # full pages shared by every sequence (atomic mode)


CELLS = (
    UnifiedCell("mixed_d128_page16", (512, 1024, 300, 777), (1024, 2048, 900, 1500),
                16, 8, 8, 128),
    UnifiedCell("gqa_d128_page128", (1024, 2048, 64, 512), (2048, 2048, 1024, 4096),
                128, 32, 8, 128),
    UnifiedCell("mha_d64_page64_nomask", (700, 1300, 2048), (700, 1300, 2048),
                64, 16, 16, 64, "fp16", "none"),
    UnifiedCell("decode_like_gqa_page256", (16,) * 16, (2048,) * 8 + (1000,) * 8,
                256, 32, 4, 128),
    # page-to-tile decision cells (pages below the default tile's kN0)
    UnifiedCell("mha_d128_page64", (700, 1300, 2048), (700, 1300, 2048),
                64, 16, 16, 128),
    UnifiedCell("mha_d32_page64", (700, 1300, 2048), (700, 1300, 2048),
                64, 16, 16, 32),
    UnifiedCell("mha_d64_page128", (700, 1300, 2048), (700, 1300, 2048),
                128, 16, 16, 64),
)  # fmt: skip


def _torch():
    import torch

    return torch


def _band(sq, skv, mask, device):
    torch = _torch()
    if mask == "none":
        return None
    qi = torch.arange(sq, device=device)[:, None]
    ki = torch.arange(skv, device=device)[None, :]
    return ki <= qi + (skv - sq)


def make_problem(cell: UnifiedCell, *, seed: int = 7, device: str = "cuda"):
    """Device tensors of both routes plus the fp64 reference gradients."""
    torch = _torch()
    g = torch.Generator(device="cpu").manual_seed(seed)
    tdt = {"fp16": torch.float16, "bf16": torch.bfloat16}[cell.dtype]
    B, page, d = len(cell.q_lens), cell.page, cell.d
    n_blk = [-(-n // page) for n in cell.kv_lens]
    sp = cell.shared_prefix
    unique = sp + sum(max(n - sp, 0) for n in n_blk)
    num_blocks = unique + 1
    perm = torch.randperm(num_blocks, generator=g).tolist()
    mbps = max(n_blk)
    table = torch.full((B, mbps), perm[-1], dtype=torch.int32)
    it = iter(perm[sp:unique])
    for s, n in enumerate(n_blk):
        for j in range(n):
            table[s, j] = perm[j] if j < sp else next(it)
    kc = torch.randn((num_blocks, page, cell.h_kv, d), generator=g).to(tdt)
    vc = torch.randn((num_blocks, page, cell.h_kv, d), generator=g).to(tdt)
    total_q = sum(cell.q_lens)
    q = torch.randn((total_q, cell.h_q, d), generator=g).to(tdt)
    do = torch.randn((total_q, cell.h_q, d), generator=g).to(tdt)
    # gathered contiguous K / V
    pid, slot = [], []
    for s, n in enumerate(cell.kv_lens):
        t = torch.arange(n)
        pid.append(table[s, t // page].long())
        slot.append(t % page)
    pid, slot = torch.cat(pid), torch.cat(slot)
    kg, vg = kc[pid, slot], vc[pid, slot]
    scale = 1.0 / math.sqrt(d)
    grp = cell.h_q // cell.h_kv
    # fp64 reference on the stored values: the forward gives O (stored in the
    # I/O dtype) and the natural-log LSE (stored fp32); the backward then uses
    # exactly the stored O and LSE, as the kernels do (delta = rowsum(dO * O)).
    f64 = torch.float64
    o = torch.zeros((total_q, cell.h_q, d), dtype=tdt)
    lse = torch.full((total_q, cell.h_q), float("-inf"), dtype=torch.float32)
    dq = torch.zeros((total_q, cell.h_q, d), dtype=f64)
    dkg = torch.zeros(kg.shape, dtype=f64)
    dvg = torch.zeros(vg.shape, dtype=f64)
    q0 = k0 = 0
    for s in range(B):
        lq, lk = cell.q_lens[s], cell.kv_lens[s]
        qs = q[q0 : q0 + lq].to(device, f64).transpose(0, 1)
        ks = kg[k0 : k0 + lk].to(device, f64).transpose(0, 1)
        vs = vg[k0 : k0 + lk].to(device, f64).transpose(0, 1)
        dos = do[q0 : q0 + lq].to(device, f64).transpose(0, 1)
        ke = ks.repeat_interleave(grp, 0)
        ve = vs.repeat_interleave(grp, 0)
        sc = (qs @ ke.transpose(1, 2)) * scale
        band = _band(lq, lk, cell.mask, device)
        if band is not None:
            sc = sc.masked_fill(~band, float("-inf"))
        ls = torch.logsumexp(sc, dim=-1)
        live = torch.isfinite(ls)
        p = torch.where(
            live[..., None], torch.exp(sc - ls.nan_to_num(0.0)[..., None]), 0.0
        ).nan_to_num(0.0)
        os_ = (p @ ve).to(tdt)
        ls32 = ls.to(torch.float32)
        o[q0 : q0 + lq] = os_.transpose(0, 1).cpu()
        lse[q0 : q0 + lq] = ls32.transpose(0, 1).cpu()
        # backward on the stored O / LSE
        lsr = ls32.to(f64)
        liver = torch.isfinite(lsr)
        pr = torch.where(
            liver[..., None], torch.exp(sc - lsr.nan_to_num(0.0)[..., None]), 0.0
        ).nan_to_num(0.0)
        delta = (dos * os_.to(f64)).sum(-1)
        dp = dos @ ve.transpose(1, 2)
        ds = pr * (dp - delta[..., None])
        dq[q0 : q0 + lq] = ((ds @ ke) * scale).transpose(0, 1).cpu()
        dke = (ds.transpose(1, 2) @ qs) * scale
        dve = pr.transpose(1, 2) @ dos
        hk = cell.h_kv
        dkg[k0 : k0 + lk] = dke.reshape(hk, grp, lk, d).sum(1).transpose(0, 1).cpu()
        dvg[k0 : k0 + lk] = dve.reshape(hk, grp, lk, d).sum(1).transpose(0, 1).cpu()
        q0 += lq
        k0 += lk
    cu = torch.tensor([0] + list(_cumsum(cell.q_lens)), dtype=torch.int32)
    cukv = torch.tensor([0] + list(_cumsum(cell.kv_lens)), dtype=torch.int32)
    dev = {
        "q": q.to(device),
        "o": o.to(device),
        "do": do.to(device),
        "lse": lse.to(device),
        "k_cache": kc.to(device),
        "v_cache": vc.to(device),
        "k": kg.contiguous().to(device),
        "v": vg.contiguous().to(device),
        "cu_seqlens_q": cu.to(device),
        # the general THD route takes element offsets (multiplier 1)
        "offsets_q": (cu * (cell.h_q * d)).to(device),
        "offsets_kv": (cukv * (cell.h_kv * d)).to(device),
        "seq_lens_kv": torch.tensor(cell.kv_lens, dtype=torch.int32).to(device),
        "block_table": table.to(device),
    }
    ref = {"dq": dq, "dk": dkg, "dv": dvg, "pid": pid, "slot": slot}
    return dev, ref, num_blocks, mbps, scale


def _cumsum(xs):
    acc = 0
    for x in xs:
        acc += x
        yield acc


def _check(name, got, want, dtype, max_kv) -> list[str]:
    """The backward tests' bound: global ``atol + rtol * g * max|ref|`` and per
    row with the floor, ``g = max(1, sqrt(max_kv / 1024))``."""
    rtol, atol = TOL[dtype]
    rtol *= max(1.0, math.sqrt(max_kv / 1024.0))
    got, want = got.double(), want.double()
    if not bool(got.isfinite().all()):
        return [f"{name}: non-finite"]
    mx = float(want.abs().max()) if want.numel() else 0.0
    err = float((got - want).abs().max()) if want.numel() else 0.0
    probs = []
    if err > atol + rtol * mx:
        probs.append(f"{name}: max err beyond the bound")
    rows = (got - want).abs().amax(-1)
    bound = atol + rtol * want.abs().amax(-1).clamp_min(ROW_FLOOR * mx)
    if bool((rows > bound).any()):
        probs.append(f"{name}: row errors beyond the bound")
    return probs


def unified_request(cell, num_blocks, mbps, scale):
    from kernels.common.attention_bwd_unified_plan import AttnBwdUnifiedRequest

    return AttnBwdUnifiedRequest(
        num_seqs=len(cell.q_lens), h_q=cell.h_q, h_kv=cell.h_kv, d_qk=cell.d,
        d_v=cell.d, total_q=sum(cell.q_lens), max_seqlen_q=max(cell.q_lens),
        max_seqlen_kv=max(cell.kv_lens), page=cell.page, num_blocks=num_blocks,
        max_blocks_per_seq=mbps, dtype=cell.dtype,
        right_bound=0 if cell.mask == "br" else -1,
        block_table_injective=cell.shared_prefix == 0, scale=scale,
    )  # fmt: skip


def general_request(cell, scale):
    from kernels.common.attention_bwd_plan import AttnBwdRequest

    d = cell.d
    st = {n: (0, d, cell.h_q * d, 1) for n in ("q", "o", "do", "dq")}
    st.update({n: (0, d, cell.h_kv * d, 1) for n in ("k", "v", "dk", "dv")})
    return AttnBwdRequest(
        b=len(cell.q_lens), h_q=cell.h_q, h_k=cell.h_kv, h_v=cell.h_kv,
        s_q=max(cell.q_lens), s_kv=max(cell.kv_lens), d_qk=d, d_v=d,
        dtype=cell.dtype, strides=st, layout="thd",
        right_bound=0 if cell.mask == "br" else -1, bottom_right=True,
        max_total_q=sum(cell.q_lens), max_total_kv=sum(cell.kv_lens), scale=scale,
    )  # fmt: skip


def unified_tile_arms(cell, arch) -> list[str]:
    """``unified:<config>`` arms: every shipped tile configuration that can
    serve the cell's page (multi tiles dividing it, the matching sub tile, the
    fallback)."""
    from kernels.common.attention_bwd_unified import (
        AttnBwdUnifiedSpec,
        validate_attn_bwd_unified_spec,
    )
    from kernels.common.attention_bwd_unified_plan import (
        attn_bwd_unified_tile_configs,
    )

    out = []
    for name, (cls, knobs) in attn_bwd_unified_tile_configs(arch, cell.d, 8).items():
        spec = validate_attn_bwd_unified_spec(
            AttnBwdUnifiedSpec(head_size=cell.d, dtype=cell.dtype, **knobs), arch
        )
        if cls == "sub" and spec.sub_page != cell.page:
            continue
        if cls == "multi" and cell.page % spec.block_n:
            continue
        out.append(f"unified:{name}")
    return out


def _arm(cell, dev, num_blocks, mbps, scale, arm, arch):
    """``(launch(), outputs)``: one complete backward of ``arm``."""
    torch = _torch()
    tdt = dev["q"].dtype
    if arm.startswith("unified"):
        from kernels.common.attention_bwd_unified_plan import (
            AttnBwdUnifiedPolicy,
            attn_bwd_unified_plan,
            attn_bwd_unified_tile_configs,
            shipped_tuning,
        )
        from kernels.common.attention_bwd_unified_run import run_attn_bwd_unified

        knobs = {}
        if ":" in arm:
            cfg = arm.split(":", 1)[1]
            knobs = dict(attn_bwd_unified_tile_configs(arch, cell.d, 8)[cfg][1])
        # "unified": the shipped plan (tuned table included);
        # "unified:<config>": that tile configuration forced
        tuning = shipped_tuning if not knobs else (lambda r, a, pc: dict(knobs))
        pol = AttnBwdUnifiedPolicy(validated_archs=frozenset({arch}), tuning=tuning)
        plan = attn_bwd_unified_plan(
            unified_request(cell, num_blocks, mbps, scale), arch, policy=pol
        )
        out = {
            "dq": torch.empty_like(dev["q"]),
            "dk_cache": torch.zeros_like(dev["k_cache"]),
            "dv_cache": torch.zeros_like(dev["v_cache"]),
        }
        run = run_attn_bwd_unified
    else:
        from kernels.common.attention_bwd_plan import AttnBwdPolicy, attn_bwd_plan
        from kernels.common.attention_bwd_plan import FEATURE_GROUPS
        from kernels.common.attention_bwd_run import run_attn_bwd

        pol = AttnBwdPolicy(
            declared_groups=frozenset(FEATURE_GROUPS), waive_unverified=True
        )
        plan = attn_bwd_plan(general_request(cell, scale), arch, policy=pol)
        out = {
            "dq": torch.empty_like(dev["q"]),
            "dk": torch.empty(dev["k"].shape, dtype=tdt, device=dev["k"].device),
            "dv": torch.empty(dev["v"].shape, dtype=tdt, device=dev["v"].device),
        }
        run = run_attn_bwd
    ws = torch.empty(
        max(plan.workspace_bytes, 256), dtype=torch.uint8, device=dev["q"].device
    )
    tensors = {k: v.data_ptr() for k, v in {**dev, **out}.items()}
    cache: dict = {}

    def launch():
        run(plan, tensors=tensors, workspace=ws.data_ptr(),
            workspace_size=ws.numel(), cache=cache)  # fmt: skip

    return launch, out, plan


def _time(launch, iters: int) -> float:
    torch = _torch()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iters):
        launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="JSONL path outside the repository")
    ap.add_argument("--attempts", type=int, default=7)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--cells", default="", help="comma-separated cell ids (default all)"
    )
    args = ap.parse_args(argv)
    torch = _torch()
    from rocke.runtime.hip_module import get_device_arch

    arch = get_device_arch(0)
    want = set(filter(None, args.cells.split(",")))
    rc = 0
    for cell in CELLS:
        if want and cell.id not in want:
            continue
        dev, ref, num_blocks, mbps, scale = make_problem(cell)
        arms = {}
        for arm in ["unified", "general"] + unified_tile_arms(cell, arch):
            launch, out, plan = _arm(cell, dev, num_blocks, mbps, scale, arm, arch)
            launch()
            torch.cuda.synchronize()
            mk = max(cell.kv_lens)
            probs = _check("dq", out["dq"].float().cpu(), ref["dq"], cell.dtype, mk)
            if arm.startswith("unified"):
                pid, slot = ref["pid"], ref["slot"]
                exp_k = torch.zeros(dev["k_cache"].shape, dtype=torch.float64)
                exp_v = torch.zeros_like(exp_k)
                exp_k.index_put_((pid, slot), ref["dk"].double(), accumulate=True)
                exp_v.index_put_((pid, slot), ref["dv"].double(), accumulate=True)
                live = torch.zeros(dev["k_cache"].shape[:2], dtype=torch.bool)
                live[pid, slot] = True
                kc = out["dk_cache"].float().cpu()[live]
                vc = out["dv_cache"].float().cpu()[live]
                probs += _check("dk", kc, exp_k[live], cell.dtype, mk)
                probs += _check("dv", vc, exp_v[live], cell.dtype, mk)
            else:
                probs += _check(
                    "dk", out["dk"].float().cpu(), ref["dk"], cell.dtype, mk
                )
                probs += _check(
                    "dv", out["dv"].float().cpu(), ref["dv"], cell.dtype, mk
                )
            if probs:
                # an arm that fails the gate is never timed
                print(f"{cell.id}: {arm} CORRECTNESS FAIL {probs}")
                rc = 1
                continue
            arms[arm] = (launch, plan)
        if "general" not in arms or "unified" not in arms:
            continue
        times = {a: [] for a in arms}
        for _ in range(3):
            for a, (launch, _p) in arms.items():
                launch()
        torch.cuda.synchronize()
        for _ in range(args.attempts):
            for a, (launch, _p) in arms.items():
                times[a].append(_time(launch, args.iters))
        med = {a: statistics.median(t) for a, t in times.items()}
        with open(args.out, "a", encoding="utf-8") as f:
            for a, (launch, plan) in arms.items():
                rec = {
                    "cell": asdict(cell), "arm": a, "arch": arch, "ms": times[a],
                    "median_ms": med[a], "route_tag": plan.route_tag,
                    "kernels": [s.kernel for s in plan.launches],
                }  # fmt: skip
                f.write(json.dumps(rec) + "\n")
        for a in arms:
            if a == "general":
                continue
            ratio = med[a] / med["general"]
            tag = arms[a][1].route_tag
            print(
                f"{cell.id}: {a} ({tag}) gate PASS; {a}/general median ratio {ratio:.2f}"
            )
    return rc


if __name__ == "__main__":
    sys.exit(main())
