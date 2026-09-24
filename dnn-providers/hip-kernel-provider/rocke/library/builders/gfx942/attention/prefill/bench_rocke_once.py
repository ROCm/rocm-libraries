import argparse
import math
import os

import torch

from builders.gfx942.attention.prefill.attention_dense_prefill import (
    dense_request,
    resolve_dense_spec,
    run,
)

from kernels.gfx942.attention_dense import (
    build_attention_dense,
    attention_dense_signature,
    attention_dense_grid,
    attention_dense_block,
)

from rocke.helpers.compile import compile_kernel
from rocke.runtime import KernelLauncher, LaunchConfig


# Shape is controlled by environment variables.
B = int(os.environ.get("B", "1"))
S = int(os.environ.get("S", "4096"))
HQ = int(os.environ.get("HQ", "32"))
HKV = int(os.environ.get("HKV", "8"))
D = int(os.environ.get("D", "128"))

WARMUP = int(os.environ.get("WARMUP", "10"))
ITERS = int(os.environ.get("ITERS", "50"))

VALIDATE = os.environ.get("VALIDATE", "0") == "1"
MAX_ABS_TOL = 2e-2


args = argparse.Namespace(
    persistent=None,
    num_persistent=None,
    persist_decode=None,
    block_n=None,
    waves_per_eu=None,
    interleave=None,
    lds_k_group_pad=None,
    sliding_window=None,
)


req = dense_request(
    args,
    batch=B,
    seqlen_q=S,
    seqlen_kv=S,
    num_query_heads=HQ,
    num_kv_heads=HKV,
    head_size=D,
    causal=True,
    dtype="bf16",
)

spec = resolve_dense_spec(req, {})


print(f"shape: B={B} S={S} HQ={HQ} HKV={HKV} D={D}")
print("kernel:", spec.kernel_name())
print("cfvst:", spec.resolved_use_cfvst())
print("v_swizzle:", spec.resolved_use_v_swizzle())
print("v_row_pad:", spec.resolved_v_row_pad())
print("wpe:", spec.resolved_waves_per_eu())


# ------------------------------------------------------------
# Numerical validation
#
# Example:
#
# B=1 S=4096 HQ=32 HKV=8 D=128 \
# VALIDATE=1 python bench_rocke_once.py
#
# Validation is kept completely separate from performance timing.
# ------------------------------------------------------------

if VALIDATE:
    _, _, err = run(
        spec,
        warmup=0,
        iters=1,
        check=True,
        overrides={},
    )

    if err >= MAX_ABS_TOL:
        raise SystemExit(
            f"VALIDATION=FAIL max_abs_error={err:.6e}"
        )

    print(f"VALIDATION=PASS max_abs_error={err:.6e}")
    raise SystemExit(0)


# ------------------------------------------------------------
# Performance benchmark
# ------------------------------------------------------------

art = compile_kernel(
    build_attention_dense(spec, arch="gfx942"),
    arch="gfx942",
    backend="python",
    capture_ir_text=False,
)

launcher = KernelLauncher(
    hsaco=art.hsaco,
    kernel_name=art.kernel_name,
    signature=attention_dense_signature(spec),
)


torch.manual_seed(0)

q = (
    torch.randn(
        B,
        S,
        HQ,
        D,
        device="cuda",
        dtype=torch.bfloat16,
    )
    * 0.2
).contiguous()

k = (
    torch.randn(
        B,
        S,
        HKV,
        D,
        device="cuda",
        dtype=torch.bfloat16,
    )
    * 0.2
).contiguous()

v = (
    torch.randn(
        B,
        S,
        HKV,
        D,
        device="cuda",
        dtype=torch.bfloat16,
    )
    * 0.2
).contiguous()

o = torch.empty_like(q)


vals = {
    "q_ptr": q,
    "k_ptr": k,
    "v_ptr": v,
    "o_ptr": o,
    "scale": 1.0 / math.sqrt(D),
}


cfg = LaunchConfig(
    grid=attention_dense_grid(spec),
    block=attention_dense_block(spec),
    stream=torch.cuda.current_stream().cuda_stream,
)


# ------------------------------------------------------------
# Warmup
# ------------------------------------------------------------

for _ in range(WARMUP):
    launcher(vals, config=cfg)

torch.cuda.synchronize()


# ------------------------------------------------------------
# Timed launches
# ------------------------------------------------------------

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

for _ in range(ITERS):
    launcher(vals, config=cfg)

end.record()
end.synchronize()


ms = start.elapsed_time(end) / ITERS

print(f"RESULT_MS={ms:.6f}")