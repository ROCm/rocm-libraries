"""fp8 KV-dequant decode — rocKE vs AITER / Triton comparison harness.

Ticket criterion 3: bench our fp8 (e4m3) decode against AITER and Triton *where
a baseline exists*. Same gpt-oss decode workload as the baseline harness (D64,
64/8 GQA, Sq=1, flash + sink, kv_len {2048, 8192}, batch {1, 64}).

Design:
  * Every backend is correctness-gated against ONE shared fp32 paged-dequant
    reference before it is timed -- never time a wrong kernel (runbook rule).
  * rocKE is driven through the dispatcher: an ``AttentionRequest(use_fp8=True)``
    is dispatched to confirm the selected path, then launched via
    ``run_unified_attention_torch`` -- so the number reflects what production
    routing picks (the fp8 long-KV -> 3D split-KV gate).
  * AITER / Triton are optional: each is *probed* at runtime. If the module is
    absent, or exposes no fp8 paged-decode entry we can drive, the backend is
    SKIPPED with a reason (and any discovered candidate entry points printed) --
    the run continues with whatever backends are available.

Compliance: this file bakes in NO measured numbers. It PRINTS latency at
runtime; those figures go to the protected Confluence page, never into the repo.

Run (rocke .venv, on a gfx942 or gfx950 node):
    python fp8_decode_vs_baselines.py
    python fp8_decode_vs_baselines.py --warmup 50 --iters 500 --repeat 3 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys

# gpt-oss decode workload: (batch, kv_len) x (flash, sink), D64 64/8 e4m3.
_BATCHES = (1, 64)
_KV_LENS = (2048, 8192)
_NHQ, _NHK, _HD, _BS = 64, 8, 64, 16
_TOL = 5e-2  # bf16 correctness band (runbook 1.4)
_K_SCALE, _V_SCALE = 1.5, 0.75


# --------------------------------------------------------------------------- #
# Shared workload: inputs + fp32 reference (mirrors fp8_decode_baseline.py).
# --------------------------------------------------------------------------- #
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
    block_table = torch.randint(
        0, pool, (batch, num_blks), dtype=torch.int32, device="cuda"
    )
    sinks = (
        torch.randn(_NHQ, dtype=torch.bfloat16, device="cuda") * 0.1
        if use_sinks
        else None
    )
    return dict(q=q, kc=kc, vc=vc, cu_q=cu_q, kv_lens=kv_lens,
                block_table=block_table, sinks=sinks)


def _reference(data, batch, sk, sinks):
    """fp32 paged decode reference mirroring the kernel dequant."""
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


class BackendUnavailable(Exception):
    """Raised by a backend probe when it cannot run this workload."""


def _signature(fn) -> str:
    """Best-effort ``(params)`` string for a discovered entry, for handoff."""
    import inspect

    try:
        return str(inspect.signature(fn))
    except (TypeError, ValueError):
        return "(<signature unavailable>)"


# --------------------------------------------------------------------------- #
# rocKE backend: dispatch -> run_unified_attention_torch.
# --------------------------------------------------------------------------- #
class RockeBackend:
    name = "rocke"

    def __init__(self, arch, fnuz):
        self._arch = arch
        self._fnuz = fnuz

    def probe(self):
        # Always available on a supported arch; the format guard is what gates it.
        return True, "ok"

    def _request(self, batch, sk, use_sinks):
        from dispatch.attention import AttentionRequest

        return AttentionRequest(
            batch=batch, nhead_q=_NHQ, nhead_k=_NHK, seqlen_q=1, seqlen_k=sk,
            hdim_q=_HD, hdim_v=_HD, arch=self._arch, kv_block_size=_BS,
            dtype="bf16", use_sinks=use_sinks, use_fp8=True, fp8_fnuz=self._fnuz,
        )

    def make_launch(self, data, batch, sk, use_sinks, out, stream):
        """Return (launch_callable, path) after confirming dispatch routing."""
        from dispatch.attention import dispatch_attention
        from kernels import UnifiedAttentionProblem, run_unified_attention_torch

        # Confirm what the production dispatcher selects for this shape.
        result = dispatch_attention(self._request(batch, sk, use_sinks))
        path = result.spec.path

        prob = UnifiedAttentionProblem(
            total_q=batch, num_seqs=batch, num_query_heads=_NHQ, num_kv_heads=_NHK,
            head_size=_HD, block_size=_BS, max_seqlen_q=1, max_seqlen_k=sk,
            dtype="bf16", use_fp8=True, fp8_fnuz=self._fnuz, use_sinks=use_sinks,
        )

        def launch():
            run_unified_attention_torch(
                problem=prob, q=data["q"], k=data["kc"], v=data["vc"], out=out,
                cu_seqlens_q=data["cu_q"], seqused_k=data["kv_lens"],
                softmax_scale=_HD ** -0.5, block_table=data["block_table"],
                softcap=0.0, sinks=data["sinks"],
                backend=("tiled" if path == "2d" else path),
                k_scale=_K_SCALE, v_scale=_V_SCALE, stream=stream,
            )

        return launch, path


# --------------------------------------------------------------------------- #
# External backends: probe-based adapters. Left as thin, honest seams because
# the exact AITER / Triton fp8 paged-decode entry point cannot be verified from
# this authoring box (module absent / off-target arch). Each probe reports the
# candidate entry points it discovered so the call can be finalized on the run
# box if the default guess does not match.
# --------------------------------------------------------------------------- #
_AITER_DECODE_CANDIDATES = (
    "paged_attention_rocm",
    "paged_attention",
    "pa_fwd",
    "mha_batch_prefill",
)


class AiterBackend:
    name = "aiter"

    def __init__(self, fp8_dtype):
        self._fp8_dtype = fp8_dtype
        self._entry = None

    def probe(self):
        try:
            import aiter  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"import aiter failed: {exc!r}")
        found = [n for n in _AITER_DECODE_CANDIDATES if hasattr(aiter, n)]
        if not found:
            avail = [n for n in dir(aiter) if "atten" in n.lower() or "pa" in n.lower()]
            raise BackendUnavailable(
                "no known fp8 paged-decode entry on `aiter`; "
                f"attention-like symbols seen: {avail[:12]}"
            )
        self._entry = found[0]
        return True, f"aiter.{self._entry}{_signature(getattr(aiter, self._entry))}"

    def make_launch(self, data, batch, sk, use_sinks, out, stream):
        # The concrete AITER call is finalized once probe() confirms the entry on
        # the run box; until an entry is wired, surface it rather than fabricate.
        raise BackendUnavailable(
            f"aiter.{self._entry} discovered but its call signature is not wired "
            "yet -- rerun after confirming the entry, or paste `help(aiter."
            f"{self._entry})` so the adapter can be completed."
        )


class TritonBackend:
    name = "triton"

    def __init__(self, fp8_dtype):
        self._fp8_dtype = fp8_dtype
        self._entry = None

    def probe(self):
        try:
            import triton  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"import triton failed: {exc!r}")
        # Prefer AITER's Triton ops if present (AOTriton-style decode kernels).
        entry = None
        try:
            import aiter.ops.triton as at  # noqa: F401

            cands = [n for n in dir(at) if "decode" in n.lower() or "attention" in n.lower()]
            if cands:
                entry = f"aiter.ops.triton.{cands[0]}{_signature(getattr(at, cands[0]))}"
        except Exception:  # noqa: BLE001
            pass
        if entry is None:
            raise BackendUnavailable(
                "triton present but no fp8 paged-decode kernel located "
                "(looked under aiter.ops.triton); point the adapter at the "
                "AOTriton/Triton decode entry to enable this backend."
            )
        self._entry = entry
        return True, entry

    def make_launch(self, data, batch, sk, use_sinks, out, stream):
        raise BackendUnavailable(
            f"{self._entry} discovered but not wired; confirm its signature on "
            "the run box to complete the adapter."
        )


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def _measure(backend, data, batch, sk, use_sinks, ref, timing, stream):
    """Correctness-gate then time one backend on one shape. Returns a row dict."""
    import torch

    from rocke.runtime import time_launches

    out = torch.empty(batch, _NHQ, _HD, dtype=torch.bfloat16, device="cuda")
    try:
        launch, path = backend.make_launch(data, batch, sk, use_sinks, out, stream)
    except BackendUnavailable as exc:
        return dict(status="SKIP", path="-", reason=str(exc), max_abs=None, us=None)

    try:
        launch()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        return dict(status="ERR", path=path, reason=repr(exc), max_abs=None, us=None)

    n_bad = int(torch.isnan(out).sum() + torch.isinf(out).sum())
    err = (out.float() - ref).abs().max().item()
    if n_bad or err >= _TOL:
        return dict(status=("NaN" if n_bad else "DIFF"), path=path,
                    reason=f"max_abs={err:.3e}", max_abs=err, us=None)

    med = []
    for _ in range(timing["repeat"]):
        ms = time_launches(launch, warmup=timing["warmup"], iters=timing["iters"],
                           stream=stream)
        med.append(ms * 1e3)
    med.sort()
    return dict(status="OK", path=path, reason="", max_abs=err,
                us=med[len(med) // 2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeat", type=int, default=1, help="runs to median over")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default="", help="write structured rows here")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1

    from rocke.core.arch import ArchTarget
    from rocke.runtime import synchronize_and_release
    from kernels.common.attention_unified import _resolve_attention_arch
    from kernels.common.fmha_fwd_fp8 import _FNUZ_FP8_TARGET_FAMILIES

    arch = _resolve_attention_arch()
    fnuz = ArchTarget.from_gfx(arch).target_family in _FNUZ_FP8_TARGET_FAMILIES
    fp8_dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    fmt = "e4m3fnuz" if fnuz else "e4m3fn(OCP)"
    stream = int(torch.cuda.current_stream().cuda_stream)
    timing = dict(warmup=args.warmup, iters=args.iters, repeat=args.repeat)

    backends = [RockeBackend(arch, fnuz), AiterBackend(fp8_dtype),
                TritonBackend(fp8_dtype)]

    # Probe once up front and report availability.
    active, skipped = [], []
    for b in backends:
        try:
            _, detail = b.probe()
            active.append(b)
            print(f"backend {b.name:<8} available: {detail}")
        except BackendUnavailable as exc:
            skipped.append(b.name)
            print(f"backend {b.name:<8} SKIPPED:  {exc}")

    print(f"\narch={arch}  fp8={fmt}  warmup={args.warmup} iters={args.iters} "
          f"repeat={args.repeat}  (D{_HD} {_NHQ}x{_NHK} bs{_BS})")
    header = f"{'shape':<18} {'backend':<8} {'path':<4} {'max_abs':>10} {'us':>9}  status"
    print(header)
    print("-" * len(header))

    rows = []
    failed = False
    for use_sinks in (False, True):
        tag0 = "sink" if use_sinks else "flash"
        for batch in _BATCHES:
            for sk in _KV_LENS:
                shape = f"{tag0}_b{batch}_kv{sk}"
                data = _build_inputs(batch, sk, fp8_dtype, use_sinks, args.seed)
                ref = _reference(data, batch, sk, data["sinks"])
                for b in active:
                    r = _measure(b, data, batch, sk, use_sinks, ref, timing, stream)
                    rows.append(dict(shape=shape, backend=b.name, **r))
                    us = f"{r['us']:.2f}" if r["us"] is not None else "--"
                    ma = f"{r['max_abs']:.3e}" if r["max_abs"] is not None else "--"
                    print(f"{shape:<18} {b.name:<8} {r['path']:<4} {ma:>10} "
                          f"{us:>9}  {r['status']}")
                    if r["status"] not in ("OK", "SKIP"):
                        failed = True
    synchronize_and_release(stream)

    print("-" * len(header))
    if skipped:
        print(f"skipped backends: {', '.join(skipped)} (see reasons above)")
    print("comparison INCOMPLETE — a backend errored/diverged above" if failed
          else "comparison complete for all available backends")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dict(arch=arch, fp8=fmt, timing=timing, rows=rows), fh, indent=2)
        print(f"wrote {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
