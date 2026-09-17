import torch
from flydsl.compiler.kernel_function import kernel
from flydsl.compiler.jit_function import jit
from flydsl.expr import gpu, arith, math, range_constexpr
from flydsl.expr.gpu import block_idx
from flydsl.expr.typing import Tensor, T

# Throwaway Phase-A kernel: RMSNorm, one thread per row, D hardcoded.
# out[row, i] = x[row, i] * rsqrt(mean_i(x^2) + eps) * w[i]
# Rely entirely on flyDSL scalar coercion: python int -> i32, python float -> f32.
# Only the block-index cast needs an explicit (bare) T.i32, matching working vadd.
D = 64
EPS = 1e-6


@kernel
def rmsnorm(out: Tensor, x: Tensor, w: Tensor):
    row = block_idx.x  # already i32 on this path (no cast)
    base = row * D  # i32 * python-int -> i32
    ss = x[base] * x[base]  # seed accumulator as DSL f32
    for k in range_constexpr(1, D):  # constexpr -> unroll, k is python int
        v = x[base + k]
        ss = ss + v * v
    inv = math.rsqrt(ss * (1.0 / D) + EPS)  # DSL f32 * / + python floats
    for k in range_constexpr(D):
        out[base + k] = x[base + k] * inv * w[k]


ROWS = 8  # throwaway: hardcode grid to avoid tracing rows as an index arg


@jit
def run(out, x, w, stream):
    rmsnorm(out, x, w).launch(grid=(ROWS, 1, 1), block=(1, 1, 1), stream=stream)


def main():
    rows = ROWS
    torch.manual_seed(0)
    x = torch.randn(rows, D, dtype=torch.float32, device="cuda")
    w = torch.randn(D, dtype=torch.float32, device="cuda")
    out = torch.empty(rows, D, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream()
    # Pass 1-D flat views so linear base+k indexing matches rank-1 tensors
    # (rank-2 tensors inject an index-typed stride -> addi(i32,index) verify fail).
    run(out.view(-1), x.view(-1), w, stream)
    torch.cuda.synchronize()
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w
    err = (out - ref).abs().max().item()
    print(
        ("OK" if err < 1e-3 else "MISMATCH"),
        "| max_abs_err=",
        err,
        "| out[0,:4]=",
        out[0, :4].tolist(),
    )


if __name__ == "__main__":
    main()
