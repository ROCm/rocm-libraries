import argparse
import os
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

assert torch.cuda.is_available(), "cuda is not available"


class GemmRMSNorm(nn.Module):
    def __init__(self, n: int, eps: float = 1e-6, scale: float = 1.0):
        super().__init__()
        self.norm = nn.RMSNorm(n, eps=eps)
        self.scale = scale

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with record_function("gemm"):
            c = torch.mm(a, b)
        with record_function("rmsnorm"):
            out_bf16 = self.norm(c)
        with record_function("quantize_fp8"):
            # 240.0 is the max finite value of E4M3FNUZ; clamp before cast to avoid undefined saturation.
            out_fp8 = out_bf16.mul(self.scale).clamp(-240.0, 240.0).to(torch.float8_e4m3fnuz)
        return out_fp8, out_bf16


def main():
    parser = argparse.ArgumentParser(description="GEMM + RMSNorm benchmark")
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations before profiling")
    parser.add_argument("--steps", type=int, default=10, help="profiled iterations")
    parser.add_argument("--scale", type=float, default=1.0, help="per-tensor fp8 quantization scale")
    parser.add_argument("--save-kernels", metavar="DIR", help="directory to save generated Triton kernels")
    args = parser.parse_args()

    if args.save_kernels:
        os.makedirs(args.save_kernels, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = args.save_kernels
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        os.environ["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"] = "1"

    a = torch.randn(args.m, args.k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(args.k, args.n, dtype=torch.bfloat16, device="cuda")

    model = torch.compile(GemmRMSNorm(args.n, scale=args.scale).cuda().to(torch.bfloat16))

    for _ in range(args.warmup):
        model(a, b)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(args.steps):
            with record_function("forward"):
                model(a, b)
            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    trace_path = f"trace_m{args.m}_n{args.n}_k{args.k}.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace saved to {trace_path}")


if __name__ == "__main__":
    main()
