# Validation for the promoted std-QK D256 kernel now living in
# kernels/gfx942/attention_tiled_2d.py (StdQKAttentionSpec / build_stdqk_attention).
# Correctness vs torch causal SDPA (fp32 ref) + perf + AOTriton ratio. Exits nonzero
# on a correctness failure so it doubles as a smoke test.
import sys, torch
import kernels.common.attention_unified as au

au._RESOLVED_ATTENTION_ARCH = "gfx942"
from kernels.gfx942.attention_tiled_2d import (
    StdQKAttentionSpec,
    build_stdqk_attention,
    stdqk_attention_grid,
)
from rocke import compile_kernel
from rocke.helpers import SignatureBuilder
from rocke.runtime import (
    KernelLauncher,
    LaunchConfig,
    time_launches,
    synchronize_and_release,
)

SQ = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
H = int(sys.argv[2]) if len(sys.argv) > 2 else 16
HKV = 2
D = 256
GQ = H // HKV

spec = StdQKAttentionSpec(seqlen_q=SQ, seqlen_k=SQ, num_query_heads=H, num_kv_heads=HKV)
art = compile_kernel(build_stdqk_attention(spec), arch="gfx942")
print(f"built {spec.kernel_name()}", flush=True)

torch.manual_seed(0)
q = torch.randn(H, SQ, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(HKV, SQ, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(HKV, SQ, D, device="cuda", dtype=torch.bfloat16)
o = torch.zeros(H, SQ, D, device="cuda", dtype=torch.float32)
sig = (
    SignatureBuilder()
    .ptr("Q", "bf16")
    .ptr("K", "bf16")
    .ptr("V", "bf16")
    .ptr("O", "f32")
    .build()
)
L = KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)
hs = torch.cuda.current_stream().cuda_stream
cfg = LaunchConfig(grid=stdqk_attention_grid(spec), block=(64, 1, 1), stream=hs)
L({"Q": q, "K": k, "V": v, "O": o}, config=cfg)
torch.cuda.synchronize()

ke = k.repeat_interleave(GQ, dim=0)
ve = v.repeat_interleave(GQ, dim=0)
ref = torch.nn.functional.scaled_dot_product_attention(
    q.float()[None], ke.float()[None], ve.float()[None], is_causal=True
)[0]
err = (o - ref).abs().max().item()
ok = err < 0.2
print(f"stdqk D256 SQ={SQ} H={H} max_abs_err={err:.4e}  {'CORRECT' if ok else 'WRONG'}")


def once():
    L({"Q": q, "K": k, "V": v, "O": o}, config=cfg)


ms = time_launches(once, warmup=10, iters=50, stream=hs)
synchronize_and_release(hs)
flop = 2.0 * (2.0 * SQ * SQ * D) * 0.5 * H
tf = flop / (ms * 1e-3) / 1e12
print(f"time={ms*1e3:.1f}us  TF/s(causal)={tf:.2f}")
try:

    def aot():
        torch.nn.functional.scaled_dot_product_attention(
            q[None], k[None], v[None], is_causal=True, enable_gqa=True
        )

    aot()
    torch.cuda.synchronize()
    ms_a = time_launches(aot, warmup=10, iters=50, stream=hs)
    synchronize_and_release(hs)
    print(f"AOTriton TF/s={flop/(ms_a*1e-3)/1e12:.2f}  ratio={ms_a/ms:.3f}x")
except Exception as e:
    print("aot bench skipped:", str(e).splitlines()[0][:80])

sys.exit(0 if ok else 1)
