"""fp8 decode routing gate A/B — is `_enable_fp8_decode_3d` still needed after #10583?

#10583 (feat(rocke): resolve gfx950 num_cus from the device) now resolves the
LIVE gfx950 CU count instead of the legacy 120. The fp8 routing gate
`_enable_fp8_decode_3d` was added to compensate for that legacy undersizing. This
A/B asks whether the gate still changes anything now that the resolver is fixed:

  for each cohort shape, at the PRODUCTION-resolved num_cus, compare
    * gate ON  (shipped): select_path() forces 3D
    * gate OFF (monkeypatched to False): select_path() falls to the num_cus
      heuristic -- does it still pick 3D on its own?

If gate-OFF routes to 3D for every cohort shape, the routing gate is redundant
(the kernel is identical) and should be dropped, leaving only waves_per_eu=3.
Where gate-OFF routes 2D, the resolved num_cus tells us the regime (a partitioned
part floored to 120, vs a full part ~304).

Reports resolved num_cus, ON/OFF path, correctness (vs fp32 paged-dequant ref),
and latency. Measured latencies are for the local decision only -- they go to
Confluence, never the repo.

Run (rocke .venv, gfx950 node):
    python fp8_decode_gate_ab.py
    python fp8_decode_gate_ab.py --warmup 50 --iters 500 --repeat 3
"""

from __future__ import annotations

import argparse
import sys

_BATCHES = (1, 64)
_KV_LENS = (2048, 8192)
_NHQ, _NHK, _HD, _BS = 64, 8, 64, 16
_TOL = 5e-2
_K_SCALE, _V_SCALE = 1.0, 1.0


def _build_inputs(batch, sk, fp8_dtype, use_sinks, seed):
    import torch

    torch.manual_seed(seed)
    num_blks = (sk + _BS - 1) // _BS
    pool = num_blks * batch + 8
    q = torch.randn(batch, _NHQ, _HD, dtype=torch.bfloat16, device="cuda") * 0.1
    k_f32 = torch.randn(pool, _BS, _NHK, _HD, dtype=torch.float32, device="cuda") * 0.5
    v_f32 = torch.randn(pool, _BS, _NHK, _HD, dtype=torch.float32, device="cuda") * 0.5
    kc = k_f32.to(fp8_dtype).contiguous()
    vc = v_f32.to(fp8_dtype).contiguous()
    cu_q = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda")
    kv_lens = torch.full((batch,), sk, dtype=torch.int32, device="cuda")
    block_table = torch.randint(0, pool, (batch, num_blks), dtype=torch.int32, device="cuda")
    sinks = (
        torch.randn(_NHQ, dtype=torch.bfloat16, device="cuda") * 0.1 if use_sinks else None
    )
    return dict(q=q, kc=kc, vc=vc, cu_q=cu_q, kv_lens=kv_lens,
                block_table=block_table, sinks=sinks)


def _reference(data, batch, sk, sinks):
    import torch

    q = data["q"].float()
    scale = _HD ** -0.5
    nrep = _NHQ // _NHK
    out = torch.empty(batch, _NHQ, _HD, dtype=torch.float32, device="cuda")
    for b in range(batch):
        bt = data["block_table"][b]
        kd = (data["kc"][bt].float() * _K_SCALE).reshape(-1, _NHK, _HD)[:sk]
        vd = (data["vc"][bt].float() * _V_SCALE).reshape(-1, _NHK, _HD)[:sk]
        for h in range(_NHQ):
            kh = h // nrep
            s = (q[b, h] @ kd[:, kh, :].t()) * scale
            if sinks is not None:
                m = torch.maximum(s.max(), sinks[h].float())
                p = torch.exp(s - m)
                denom = p.sum() + torch.exp(sinks[h].float() - m)
            else:
                m = s.max()
                p = torch.exp(s - m)
                denom = p.sum()
            out[b, h] = (p / denom) @ vd[:, kh, :]
    return out


def _run(prob, data, backend, out, stream):
    from kernels import run_unified_attention_torch

    run_unified_attention_torch(
        problem=prob, q=data["q"], k=data["kc"], v=data["vc"], out=out,
        cu_seqlens_q=data["cu_q"], seqused_k=data["kv_lens"],
        softmax_scale=_HD ** -0.5, block_table=data["block_table"], softcap=0.0,
        sinks=data["sinks"], backend=backend, k_scale=_K_SCALE, v_scale=_V_SCALE,
        stream=stream,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1

    from rocke.runtime import synchronize_and_release, time_launches
    from rocke.core.arch import ArchTarget
    import kernels.common.attention_unified as au
    from kernels.common.attention_unified import _resolve_attention_arch
    from kernels.common.fmha_fwd_fp8 import _FNUZ_FP8_TARGET_FAMILIES
    from dispatch.attention.common import _resolve_num_cus, AttentionRequest

    arch = _resolve_attention_arch()
    fnuz = ArchTarget.from_gfx(arch).target_family in _FNUZ_FP8_TARGET_FAMILIES
    fp8_dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    stream = int(torch.cuda.current_stream().cuda_stream)

    def _resolved_cus(batch, sk, use_sinks):
        req = AttentionRequest(
            batch=batch, nhead_q=_NHQ, nhead_k=_NHK, seqlen_q=1, seqlen_k=sk,
            hdim_q=_HD, hdim_v=_HD, arch=arch, kv_block_size=_BS, dtype="bf16",
            use_sinks=use_sinks, use_fp8=True, fp8_fnuz=fnuz, num_cus=0,
        )
        return _resolve_num_cus(req)

    def _problem(batch, sk, use_sinks, num_cus):
        return au.UnifiedAttentionProblem(
            total_q=batch, num_seqs=batch, num_query_heads=_NHQ, num_kv_heads=_NHK,
            head_size=_HD, block_size=_BS, max_seqlen_q=1, max_seqlen_k=sk,
            dtype="bf16", use_fp8=True, fp8_fnuz=fnuz, use_sinks=use_sinks,
            num_cus=num_cus,
        )

    print(f"arch={arch}  fp8={'e4m3fnuz' if fnuz else 'e4m3fn'}  "
          f"warmup={args.warmup} iters={args.iters} repeat={args.repeat}")
    print(f"{'shape':<20} {'num_cus':>7}  {'ON':>4} {'us':>8}   {'OFF':>4} {'us':>8}  "
          f"{'gate_redundant?':>15}")
    print("-" * 78)

    all_off_3d = True
    for use_sinks in (False, True):
        tag = "sink" if use_sinks else "flash"
        for batch in _BATCHES:
            for sk in _KV_LENS:
                label = f"{tag}_b{batch}_kv{sk}"
                cus = _resolved_cus(batch, sk, use_sinks)
                data = _build_inputs(batch, sk, fp8_dtype, use_sinks, args.seed)
                ref = _reference(data, batch, sk, data["sinks"])

                def _measure(path):
                    out = torch.empty(batch, _NHQ, _HD, dtype=torch.bfloat16, device="cuda")
                    backend = "tiled" if path == "2d" else "3d"
                    _run(_problem(batch, sk, use_sinks, cus), data, backend, out, stream)
                    torch.cuda.synchronize()
                    err = (out.float() - ref).abs().max().item()
                    nan = int(torch.isnan(out).sum() + torch.isinf(out).sum())
                    us = []
                    for _ in range(args.repeat):
                        ms = time_launches(
                            lambda: _run(_problem(batch, sk, use_sinks, cus), data,
                                         backend, out, stream),
                            warmup=args.warmup, iters=args.iters, stream=stream)
                        us.append(ms * 1e3)
                    us.sort()
                    return err, nan, us[len(us) // 2]

                # gate ON (shipped): select_path forces 3D.
                path_on = _problem(batch, sk, use_sinks, cus).select_path()
                err_on, nan_on, us_on = _measure(path_on)

                # gate OFF: does the num_cus heuristic still pick 3D?
                orig = au._enable_fp8_decode_3d
                au._enable_fp8_decode_3d = lambda _p: False
                try:
                    path_off = _problem(batch, sk, use_sinks, cus).select_path()
                    err_off, nan_off, us_off = _measure(path_off)
                finally:
                    au._enable_fp8_decode_3d = orig

                off_is_3d = path_off == "3d"
                all_off_3d = all_off_3d and off_is_3d
                ok = (nan_on == 0 and nan_off == 0 and err_on < _TOL and err_off < _TOL)
                verdict = "REDUNDANT" if off_is_3d else "gate flips 2D->3D"
                flag = "" if ok else "  !CORRECTNESS"
                print(f"{label:<20} {cus:>7}  {path_on:>4} {us_on:>8.2f}   "
                      f"{path_off:>4} {us_off:>8.2f}  {verdict:>15}{flag}")

    synchronize_and_release(stream)
    print("-" * 78)
    if all_off_3d:
        print("VERDICT: gate-OFF routes 3D for every cohort shape at the resolved "
              "num_cus -> _enable_fp8_decode_3d is REDUNDANT (drop it; keep wpe3).")
    else:
        print("VERDICT: gate-OFF routes 2D for some shapes (see rows) -> the gate "
              "still flips those. Check the resolved num_cus column for the regime "
              "(full ~304 vs partitioned floored 120).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
