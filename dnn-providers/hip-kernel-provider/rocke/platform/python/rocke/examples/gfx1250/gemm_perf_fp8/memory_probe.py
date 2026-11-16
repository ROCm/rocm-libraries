# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Memory-side probes for the gfx1250 fp8 GEMM: streaming bandwidth and TDM feed.

``bandwidth``
    Streaming read and copy over a sweep of buffer sizes (default 8 MiB to
    8 GiB), so both the L2-resident and the HBM regime show. Every lane loads
    16 B per step, grid-stride; copy counts read plus written bytes. The copy
    destination is checked afterwards.
``tdm``
    The case study's GEMM with the math taken out: the champion's grid of
    256x256 output tiles, tile_k 256, double-buffered LDS, wave 0 filling the A
    tile and wave 1 the B tile by TDM with one fill in flight, and a workgroup
    barrier per K-step -- no WMMAs, no LDS reads (``--lds-reads`` adds a fixed
    amount of LDS read traffic per step). Its time per K-step is the floor the
    TDM fill alone sets for that loop; compare it with the GEMM's own step time
    (GEMM time / (K / 256)). An untimed first launch checks that every
    workgroup's last filled buffer holds the bytes it should, and the probe
    reports no rate if any did not: a descriptor that moved nothing would
    otherwise look fast. The timed launches skip that check.

Both use the peak protocol (idle gap before each sample, short burst, median
over samples) and report throughput; keep absolute figures out of the
repository. Needs ``hipcc`` (``$HIPCC``, ``PATH`` or ``$ROCM_PATH/bin``) and a
gfx1250 device.

  python3 -m rocke.examples.gfx1250.gemm_perf_fp8.memory_probe
  python3 -m rocke.examples.gfx1250.gemm_perf_fp8.memory_probe bandwidth --sizes 512,8192
  python3 -m rocke.examples.gfx1250.gemm_perf_fp8.memory_probe tdm --k 65536
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .wmma_roofline_probe import build

#: The case study's tile: output tile and K tile, in elements (1-byte fp8).
TILE = 256
THREADS = 512

_BANDWIDTH_SRC = r"""
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  printf("ERROR %s at %d\n", hipGetErrorString(e), __LINE__); exit(2); } } while (0)

__global__ __launch_bounds__(256) void rd(const uint4* __restrict__ p, size_t n, uint4* sink) {
  uint4 acc = {0, 0, 0, 0};
  for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) {
    uint4 v = p[i];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if ((acc.x | acc.y | acc.z | acc.w) == 0x12345678u) sink[blockIdx.x * 256 + threadIdx.x] = acc;
}

__global__ __launch_bounds__(256) void cp(const uint4* __restrict__ p, uint4* __restrict__ q, size_t n) {
  for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) q[i] = p[i];
}

int main() {
  const std::vector<size_t> mib = {SIZES};
  const int repeats = REPEATS, gap_us = GAP_US;
  hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, 0));
  int blocks = prop.multiProcessorCount * 32;
  size_t maxb = *std::max_element(mib.begin(), mib.end()) << 20;
  uint4 *a, *b, *sink;
  CHECK(hipMalloc(&a, maxb)); CHECK(hipMalloc(&b, maxb));
  CHECK(hipMalloc(&sink, (size_t)blocks * 256 * 16));
  CHECK(hipMemset(a, 0x5a, maxb)); CHECK(hipMemset(b, 0, maxb));
  hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
  for (int kind = 0; kind < 2; ++kind) {
    for (size_t m : mib) {
      size_t bytes = m << 20, n = bytes / 16;
      int iters = std::max(3, (int)std::min<size_t>(200, (size_t)(4096ull << 20) / bytes));
      auto launch = [&]() {
        if (kind == 0) hipLaunchKernelGGL(rd, dim3(blocks), dim3(256), 0, 0, a, n, sink);
        else hipLaunchKernelGGL(cp, dim3(blocks), dim3(256), 0, 0, a, b, n);
      };
      std::vector<float> s;
      for (int r = 0; r < repeats; ++r) {
        CHECK(hipDeviceSynchronize()); usleep(gap_us);
        for (int w = 0; w < 3; ++w) launch();
        CHECK(hipEventRecord(e0)); for (int i = 0; i < iters; ++i) launch(); CHECK(hipEventRecord(e1));
        CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
        s.push_back(ms / iters);
      }
      std::sort(s.begin(), s.end());
      double moved = (double)bytes * (kind == 0 ? 1 : 2);
      printf("BW %s %zu %.6f %.4f\n", kind == 0 ? "read" : "copy", m,
             moved / (s[s.size() / 2] * 1e-3) / 1e9, s[s.size() / 2] / s[0]);
      fflush(stdout);
    }
  }
  std::vector<unsigned char> h(4096); CHECK(hipMemcpy(h.data(), b, h.size(), hipMemcpyDeviceToHost));
  bool ok = std::all_of(h.begin(), h.end(), [](unsigned char c) { return c == 0x5a; });
  printf("CHECK copy %s\n", ok ? "OK" : "WRONG");
  return ok ? 0 : 1;
}
"""

_TDM_SRC = r"""
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  printf("ERROR %s at %d\n", hipGetErrorString(e), __LINE__); exit(2); } } while (0)

typedef unsigned v4u __attribute__((ext_vector_type(4)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define THREADS THREADS_
#define T TILE_
constexpr int M = M_, N = N_; constexpr long K = K_;
constexpr int TILES_N = N / T;

// One 2D TDM fill of a [T rows, T bytes] tile at g into LDS address lds:
// tensor extents dim0 (bytes left in the row) x dim1 (rows left), row pitch K.
__device__ void tdm(const char* g, unsigned lds, unsigned dim0, unsigned dim1) {
  unsigned long ga = (unsigned long)g;
  v4u g0 = {1u, lds, (unsigned)ga, (unsigned)((ga >> 32) & 0x1FFFFFF) | (2u << 30)};
  v8i g1 = {0, (int)(dim0 << 16), (int)((dim0 >> 16) | (dim1 << 16)),
            (int)((dim1 >> 16) | (T << 16)), T, (int)K, 1 << 16, 0};
  v4i z4 = {0, 0, 0, 0}; v8i z8 = {0, 0, 0, 0, 0, 0, 0, 0};
  __builtin_amdgcn_tensor_load_to_lds(g0, g1, z4, z4, z8, 0);
}

__global__ __launch_bounds__(THREADS) void feed(const char* A, const char* B, int* bad, int lds_reads,
                                                int check) {
  extern __shared__ __attribute__((aligned(128))) unsigned char smem[];
  unsigned base = (unsigned)(unsigned long)(__attribute__((address_space(3))) unsigned char*)smem;
  const int tm = blockIdx.x / TILES_N, tn = blockIdx.x % TILES_N;
  const int wave = threadIdx.x / 32;
  auto fill = [&](int s, int buf) {
    long k0 = (long)s * T;
    unsigned la = base + buf * 2 * T * T;
    if (wave == 0) tdm(A + (long)tm * T * K + k0, la, (unsigned)(K - k0), (unsigned)(M - tm * T));
    if (wave == 1) tdm(B + (long)tn * T * K + k0, la + T * T, (unsigned)(K - k0), (unsigned)(N - tn * T));
  };
  const int steps = K / T;
  fill(0, 0);
  for (int s = 0; s < steps; ++s) {
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __syncthreads();
    if (s + 1 < steps) fill(s + 1, (s + 1) & 1);
    if (lds_reads) {  // 64 x 16 B LDS reads per thread from the buffer just filled
      const v4i* rb = (const v4i*)(smem + (s & 1) * 2 * T * T);
      v4i acc = {0, 0, 0, 0};
#pragma unroll
      for (int j = 0; j < 64; ++j) { v4i v = rb[(threadIdx.x * 7 + j * 512) % (2 * T * T / 16)]; acc += v; }
      if (acc.x == 0x7fffffff) bad[1] = acc.y;
    }
  }
  __builtin_amdgcn_s_wait_tensorcnt(0);
  __syncthreads();
  if (!check) return;
  // The last fill landed in buffer (steps-1)&1: A bytes are 1, B bytes are 2.
  const unsigned char* last = smem + ((steps - 1) & 1) * 2 * T * T;
  for (int i = threadIdx.x; i < T * T; i += THREADS)
    if (last[i] != 1 || last[T * T + i] != 2) { atomicOr(bad, 1); break; }
}

int main(int argc, char** argv) {
  const int lds_reads = LDS_READS, repeats = REPEATS, gap_us = GAP_US;
  const int wgs = (M / T) * (N / T);
  char *A, *B; int* bad;
  CHECK(hipMalloc(&A, (size_t)M * K)); CHECK(hipMalloc(&B, (size_t)N * K)); CHECK(hipMalloc(&bad, 8));
  CHECK(hipMemset(A, 1, (size_t)M * K)); CHECK(hipMemset(B, 2, (size_t)N * K)); CHECK(hipMemset(bad, 0, 8));
  size_t lds = 4 * T * T;
  CHECK(hipFuncSetAttribute((const void*)feed, hipFuncAttributeMaxDynamicSharedMemorySize, (int)lds));
  // The first, untimed launch checks the LDS contents; the timed ones do not.
  auto go = [&](int check) {
    hipLaunchKernelGGL(feed, dim3(wgs), dim3(THREADS), lds, 0, A, B, bad, lds_reads, check);
  };
  go(1); CHECK(hipDeviceSynchronize());
  int flag = 0; CHECK(hipMemcpy(&flag, bad, 4, hipMemcpyDeviceToHost));
  printf("CHECK tdm %s\n", flag ? "WRONG" : "OK");
  if (flag) return 1;
  std::vector<float> v;
  hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
  for (int r = 0; r < repeats; ++r) {
    usleep(gap_us);
    CHECK(hipEventRecord(e0)); for (int i = 0; i < 3; ++i) go(0); CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); v.push_back(ms / 3);
  }
  std::sort(v.begin(), v.end());
  const double steps = K / T, bytes = 2.0 * T * T * steps * wgs;
  const float med = v[v.size() / 2];
  printf("TDM %d %.6f %.6f %.6f %.4f\n", wgs, med, med * 1e3 / steps, bytes / (med * 1e-3) / 1e12,
         med / v[0]);
  return 0;
}
"""


def bandwidth_source(sizes_mib: list[int], repeats: int, gap_ms: float) -> str:
    return (
        _BANDWIDTH_SRC.replace("SIZES", ", ".join(f"{s}ull" for s in sizes_mib))
        .replace("REPEATS", str(repeats))
        .replace("GAP_US", str(int(gap_ms * 1000)))
    )


def tdm_source(
    m: int, n: int, k: int, *, lds_reads: bool, repeats: int, gap_ms: float
) -> str:
    for name, dim in (("M", m), ("N", n), ("K", k)):
        if dim % TILE:
            raise ValueError(f"{name}={dim} must be a multiple of the {TILE} tile")
    src = _TDM_SRC
    for key, val in (
        ("THREADS_", THREADS),
        ("TILE_", TILE),
        ("M_", m),
        ("N_", n),
        ("K_", k),
        ("LDS_READS", int(lds_reads)),
        ("REPEATS", repeats),
        ("GAP_US", int(gap_ms * 1000)),
    ):
        src = re.sub(rf"\b{key}\b", str(val), src)
    return src


def _run(src: str, work: Path) -> list[str]:
    exe, _ = build(src, work, False)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=False)
    lines = out.stdout.splitlines()
    if out.returncode or any(
        line.startswith(("ERROR", "CHECK")) and "OK" not in line for line in lines
    ):
        sys.stderr.write(out.stdout + out.stderr)
        raise SystemExit(
            f"[memory_probe] {exe.name} failed its check (rc {out.returncode})"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "probe", nargs="?", default="all", choices=("all", "bandwidth", "tdm")
    )
    p.add_argument(
        "--sizes", default="8,32,128,512,2048,8192", help="bandwidth buffer sizes, MiB"
    )
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=65536)
    p.add_argument(
        "--lds-reads", action="store_true", help="tdm: add LDS read traffic per step"
    )
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--gap-ms", type=float, default=25.0)
    p.add_argument(
        "--work",
        type=Path,
        default=Path(tempfile.gettempdir()) / "rocke_memory_probe",
        help="build directory (binaries are cached by source hash)",
    )
    args = p.parse_args(argv)

    if args.probe in ("all", "bandwidth"):
        sizes = [int(s) for s in args.sizes.split(",") if s]
        lines = _run(bandwidth_source(sizes, args.repeats, args.gap_ms), args.work)
        print(f"{'kind':<5} {'MiB':>6} {'GB/s':>10} {'best/med':>8}")
        for line in lines:
            tok = line.split()
            if tok and tok[0] == "BW":
                print(
                    f"{tok[1]:<5} {tok[2]:>6} {float(tok[3]):10.1f} {float(tok[4]):8.3f}"
                )
    if args.probe in ("all", "tdm"):
        src = tdm_source(
            args.m,
            args.n,
            args.k,
            lds_reads=args.lds_reads,
            repeats=args.repeats,
            gap_ms=args.gap_ms,
        )
        for line in _run(src, args.work):
            tok = line.split()
            if tok and tok[0] == "TDM":
                wgs, ms, us_step, tbs, spread = tok[1:]
                print(
                    f"tdm feed {args.m}x{args.n}x{args.k}, {wgs} workgroups"
                    f"{', with LDS reads' if args.lds_reads else ''}: "
                    f"{float(ms):.4f} ms, {float(us_step):.3f} us/K-step, "
                    f"{float(tbs):.1f} TB/s delivered, best/med {float(spread):.3f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
