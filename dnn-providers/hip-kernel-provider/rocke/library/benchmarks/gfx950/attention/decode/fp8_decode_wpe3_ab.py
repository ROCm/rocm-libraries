"""waves_per_eu A/B for the fp8 decode cohort (post-#10583 -- routing is not ours).

After dropping the redundant routing gate, this PR's only contribution is
``waves_per_eu = 3`` on the gfx950 fp8 3D split-KV decode kernel. This A/B toggles
ONLY waves_per_eu (3 = shipped, vs the LLVM-default ``None`` = baseline), both on
the 3D path #10583 routes the cohort to, and records per-shape WINS AND LOSSES.

waves_per_eu is a pure AMDGPU occupancy hint (kernel attribute), so both variants
are numerically identical -- the only difference is latency / run-to-run variance.
Because the launcher cache_key excludes waves_per_eu, the 3D kernel cache is
cleared between the two builds so each compiles with the intended value.

Measured latencies are for the local honest-loss verdict only -- they go to
Confluence, never the repo.

Run (rocke .venv, gfx950 node):
    python fp8_decode_wpe3_ab.py
    python fp8_decode_wpe3_ab.py --warmup 50 --iters 500 --repeat 5
"""

from __future__ import annotations

import argparse
import sys

_BATCHES = (1, 64)
_KV_LENS = (2048, 8192)
_NHQ, _NHK, _HD, _BS = 64, 8, 64, 16
_TOL = 5e-2
_K_SCALE, _V_SCALE = 1.0, 1.0
_NOISE = 0.02  # |delta| below this fraction is called neutral, not a win/loss


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1

    from rocke.runtime import synchronize_and_release, time_launches
    from rocke.core.arch import ArchTarget
    import kernels.common.attention_unified as au
    from kernels import run_unified_attention_torch
    from kernels.common.attention_unified import _resolve_attention_arch
    from kernels.common.fmha_fwd_fp8 import _FNUZ_FP8_TARGET_FAMILIES
    from dispatch.attention.common import _resolve_num_cus, AttentionRequest

    arch = _resolve_attention_arch()
    if arch != "gfx950":
        print(f"[skip] targets gfx950; device is {arch}")
        return 0
    fnuz = ArchTarget.from_gfx(arch).target_family in _FNUZ_FP8_TARGET_FAMILIES
    fp8_dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    stream = int(torch.cuda.current_stream().cuda_stream)

    def _num_cus(batch, sk, use_sinks):
        return _resolve_num_cus(AttentionRequest(
            batch=batch, nhead_q=_NHQ, nhead_k=_NHK, seqlen_q=1, seqlen_k=sk,
            hdim_q=_HD, hdim_v=_HD, arch=arch, kv_block_size=_BS, dtype="bf16",
            use_sinks=use_sinks, use_fp8=True, fp8_fnuz=fnuz, num_cus=0))

    def _problem(batch, sk, use_sinks, cus):
        return au.UnifiedAttentionProblem(
            total_q=batch, num_seqs=batch, num_query_heads=_NHQ, num_kv_heads=_NHK,
            head_size=_HD, block_size=_BS, max_seqlen_q=1, max_seqlen_k=sk,
            dtype="bf16", use_fp8=True, fp8_fnuz=fnuz, use_sinks=use_sinks, num_cus=cus)

    def _run(prob, data, out):
        run_unified_attention_torch(
            problem=prob, q=data["q"], k=data["kc"], v=data["vc"], out=out,
            cu_seqlens_q=data["cu_q"], seqused_k=data["kv_lens"],
            softmax_scale=_HD ** -0.5, block_table=data["block_table"], softcap=0.0,
            sinks=data["sinks"], backend="3d", k_scale=_K_SCALE, v_scale=_V_SCALE,
            stream=stream)

    print(f"arch={arch}  fp8={'e4m3fnuz' if fnuz else 'e4m3fn'}  "
          f"warmup={args.warmup} iters={args.iters} repeat={args.repeat}")
    print(f"{'shape':<18} {'cus':>4}  {'wpe=3':>9} {'default':>9}  {'verdict':>9}  {'max_abs':>9}")
    print("-" * 72)

    losses = []
    for use_sinks in (False, True):
        tag = "sink" if use_sinks else "flash"
        for batch in _BATCHES:
            for sk in _KV_LENS:
                label = f"{tag}_b{batch}_kv{sk}"
                cus = _num_cus(batch, sk, use_sinks)
                data = _build_inputs(batch, sk, fp8_dtype, use_sinks, args.seed)
                ref = _reference(data, batch, sk, data["sinks"])

                def _measure(wpe):
                    orig = au._select_3d_waves_per_eu
                    au._select_3d_waves_per_eu = lambda _p: wpe
                    au._ATTN_3D_TILED_CACHE.clear()  # cache_key excludes wpe -> force rebuild
                    try:
                        out = torch.empty(batch, _NHQ, _HD, dtype=torch.bfloat16, device="cuda")
                        _run(_problem(batch, sk, use_sinks, cus), data, out)
                        torch.cuda.synchronize()
                        err = (out.float() - ref).abs().max().item()
                        nan = int(torch.isnan(out).sum() + torch.isinf(out).sum())
                        us = []
                        for _ in range(args.repeat):
                            ms = time_launches(
                                lambda: _run(_problem(batch, sk, use_sinks, cus), data, out),
                                warmup=args.warmup, iters=args.iters, stream=stream)
                            us.append(ms * 1e3)
                        us.sort()
                        return err, nan, us[len(us) // 2]
                    finally:
                        au._select_3d_waves_per_eu = orig
                        au._ATTN_3D_TILED_CACHE.clear()

                err_on, nan_on, us_on = _measure(3)
                err_off, nan_off, us_off = _measure(None)
                delta = (us_off - us_on) / us_off if us_off else 0.0  # >0 => wpe3 faster
                if abs(delta) < _NOISE:
                    verdict = "neutral"
                elif delta > 0:
                    verdict = "win"
                else:
                    verdict = "LOSS"
                    losses.append(label)
                bad = max(err_on, err_off) >= _TOL or nan_on or nan_off
                print(f"{label:<18} {cus:>4}  {us_on:>9.2f} {us_off:>9.2f}  {verdict:>9}  "
                      f"{max(err_on, err_off):>9.3e}{'  !CORRECTNESS' if bad else ''}")

    synchronize_and_release(stream)
    print("-" * 72)
    if losses:
        print(f"HONEST LOSSES: waves_per_eu=3 regresses {len(losses)} shape(s): "
              f"{', '.join(losses)}")
    else:
        print("No regressions: waves_per_eu=3 is a win-or-neutral across the cohort.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
