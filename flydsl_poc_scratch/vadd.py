import torch
from flydsl.compiler.kernel_function import kernel
from flydsl.compiler.jit_function import jit
from flydsl.expr import gpu, arith
from flydsl.expr.typing import Tensor, T


@kernel
def vadd(out: Tensor, a: Tensor, b: Tensor):
    i = arith.index_cast(T.i32, gpu.thread_id("x"))
    out[i] = a[i] + b[i]


@jit
def run(out, a, b, stream):
    vadd(out, a, b).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)


def main():
    N = 256
    a = torch.arange(N, dtype=torch.float32, device="cuda")
    b = torch.ones(N, dtype=torch.float32, device="cuda")
    out = torch.empty(N, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream()
    run(out, a, b, stream)
    torch.cuda.synchronize()
    exp = a + b
    ok = torch.allclose(out, exp)
    print(
        "OK" if ok else "MISMATCH",
        "| out[:5]=",
        out[:5].tolist(),
        "exp[:5]=",
        exp[:5].tolist(),
    )


if __name__ == "__main__":
    main()
