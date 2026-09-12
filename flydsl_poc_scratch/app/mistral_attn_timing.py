#!/usr/bin/env python3
"""Attention-isolated timing: base PyTorch SDPA vs hipDNN injection, per Mistral shape.

Answers the user's timing questions:
  1. What backend does base PyTorch pick by DEFAULT on ROCm (AOTriton flash? efficient?
     math?) -- and is it the same for prefill vs decode? We identify it by forcing each
     SDPBackend in turn (FLASH/EFFICIENT/CUDNN/MATH) and matching the default's time to
     the forced backend that reproduces it (and reporting which backends even succeed).
  2. Which hipDNN engine WINS each shape (autotune, TUNE=tune).
  3. Attention-level uplift = base_default_ms / injected_ms.

Shapes = Mistral-7B GQA 32/8 D128 bf16, BSHD views, default scale:
  PREFILL  Sq=Skv=S causal      S in {512,1024,2048,4096}
  DECODE   Sq=1 Skv=S noncausal S in {1024,2048,4096}
CUDA-event timed, warmup + many iters. Base runs with injection UNINSTALLED (F.sdpa is
native torch); the monkeypatch is global so we capture the native fn before install().
"""
import math
import os
import statistics
import sys

sys.path.insert(
    0, "/home/AMD/brpepers/wt/hipdnn-flydsl-poc/projects/hipdnn/tools/hipdnn_torch"
)
import hipdnn_torch  # noqa: E402

HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
WARMUP, ITERS = 20, 100


def main() -> int:
    if not hipdnn_torch.provider_ready():
        print("provider/torch NOT ready", file=sys.stderr)
        return 1
    os.environ.setdefault("HIPDNN_TORCH_TUNE", "tune")
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    native_sdpa = F.scaled_dot_product_attention  # capture BEFORE any install
    dev = torch.device("cuda")
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(f"torch   = {torch.__version__}  hip={getattr(torch.version, 'hip', None)}")
    print(f"backends: flash={torch.backends.cuda.flash_sdp_enabled()} "
          f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} "
          f"math={torch.backends.cuda.math_sdp_enabled()}")
    print("(on ROCm the FLASH SDPA backend is AOTriton)")
    print()

    def _bshd(b, h, s, d):
        return (torch.randn(b, s, h, d, dtype=torch.bfloat16, device=dev) * 0.1).transpose(1, 2)

    def _time(fn):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(ITERS):
            st, en = torch.cuda.Event(True), torch.cuda.Event(True)
            st.record()
            fn()
            en.record()
            torch.cuda.synchronize()
            samples.append(st.elapsed_time(en))
        return statistics.median(samples)

    BACKENDS = [
        ("FLASH(AOTriton)", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
        ("MATH", SDPBackend.MATH),
    ]

    # (label, Sq, Skv, causal)
    cases = [("prefill", s, s, True) for s in (512, 1024, 2048, 4096)] + \
            [("decode", 1, s, False) for s in (1024, 2048, 4096)]

    rows = []
    for phase, sq, skv, causal in cases:
        q = _bshd(1, HEADS, sq, HEAD_DIM)
        k = _bshd(1, KV_HEADS, skv, HEAD_DIM)
        v = _bshd(1, KV_HEADS, skv, HEAD_DIM)

        def call():
            return F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=causal, enable_gqa=True)

        # base default (no injection)
        hipdnn_torch.uninstall(["sdpa"])
        base_ms = _time(call)
        # which backend does default match? force each; record success+time
        per_backend = {}
        for name, be in BACKENDS:
            try:
                with sdpa_kernel([be]):
                    per_backend[name] = _time(call)
            except Exception as ex:  # noqa: BLE001
                per_backend[name] = f"ERR:{type(ex).__name__}"
        # default backend = the succeeding forced backend whose time is closest to base
        numeric = {n: t for n, t in per_backend.items() if isinstance(t, float)}
        default_be = min(numeric, key=lambda n: abs(numeric[n] - base_ms)) if numeric else "?"

        # injected (hipDNN, autotune)
        hipdnn_torch.reset(["sdpa"])
        hipdnn_torch.install(["sdpa"])
        try:
            inj_ms = _time(call)
            eng = hipdnn_torch.overrides()["sdpa"]._last_engine
        finally:
            hipdnn_torch.uninstall(["sdpa"])

        rows.append((phase, sq, skv, base_ms, default_be, per_backend, inj_ms, eng))

    # engine id -> friendly
    NAMES = {"0x45d33a15c6a70e0b": "FlydslAttention"}
    print(f"{'phase':8} {'Sq':>5} {'Skv':>5} {'base_ms':>9} {'default_be':>16} "
          f"{'inj_ms':>8} {'winner':>22} {'uplift':>7}")
    for phase, sq, skv, base_ms, default_be, per_be, inj_ms, eng in rows:
        w = NAMES.get(str(eng), str(eng))
        up = base_ms / inj_ms if inj_ms else float('nan')
        print(f"{phase:8} {sq:>5} {skv:>5} {base_ms:9.4f} {default_be:>16} "
              f"{inj_ms:8.4f} {w:>22} {up:6.2f}x")
    print()
    print("per-backend forced times (ms; ERR = backend refused the shape):")
    for phase, sq, skv, base_ms, default_be, per_be, inj_ms, eng in rows:
        pretty = "  ".join(f"{n}={v:.4f}" if isinstance(v, float) else f"{n}={v}"
                           for n, v in per_be.items())
        print(f"  {phase} Sq={sq} Skv={skv}: {pretty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
