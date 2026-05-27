# okl: Optimal Kernel Lookup + Standalone Packaging

Experimental tooling for two related tasks:

1. **Query hipBLASLt** for the optimal kernel (heuristic winner) for a given GEMM shape + dtypes + arch, without writing any hipBLASLt code yourself.
2. **Package** that kernel as a standalone executable that loads the shipped `.co`, runs the exact kernel hipBLASLt would have chosen, and times it — with **no link dependency on libTensile or libhipblaslt**.

The point of #2 is apples-to-apples benchmarking against non-hipBLASLt kernels (cuBLAS, custom assembly, etc.). The bench-runtime tax that `hipblaslt-bench` includes is excluded, so the number reflects just the kernel.

## Files in this folder

| File | Purpose |
|---|---|
| `okl.py` | CLI tool. Default: print the heuristic winner as JSON. With `--package OUT_DIR`: extract the kernel + emit a self-describing `kernel.conf`. |
| `okl_run.cpp` | Standalone HIP runner. Reads `kernel.conf`, loads the `.co`, packs the kernarg from a metadata-driven slot list, launches with hipblaslt-bench-style timing. |
| `kernel-packaging-research.md` | Design notes + reference. Sections §1–§9 cover .co on-disk format, SolutionAdapter, kernarg ABI, the loading interface, and the metadata-driven argument-feeding interface. Read this before changing either tool. |
| `AGENTS.md` | Guidance for AI agents working in this folder. |

## Requirements

- ROCm install (tested on 6.4.3 and 7.2.1) with hipBLASLt's shipped device library available
- `hipblaslt-bench` binary (ships with hipBLASLt; `okl.py` shells out to it)
- `clang-offload-bundler` (in any of `/opt/rocm/lib/llvm/bin/`, `/opt/rocm-*/lib/llvm/bin/`)
- `llvm-readobj` (in `/usr/bin/` or any ROCm `lib/llvm/bin/`)
- For building `okl_run`: `hipcc`

Versions of bench, library, and HIP runtime must be ABI-consistent — mixing a hipblaslt-bench from one ROCm with a `TensileLibrary_lazy_*.dat` from another typically segfaults. `okl.py` defaults to whichever `hipblaslt-bench` is on `$PATH` and whichever library directory `HIPBLASLT_TENSILE_LIBPATH` (or auto-discovery) finds; pass `--bench` and `--libpath` to pin a known-good pair.

## Quick start

### Query the heuristic

```bash
python3 okl.py -m 512 -n 512 -k 512 --transa T --transb N \
    --a-type bf16_r --b-type bf16_r --c-type bf16_r --d-type bf16_r \
    --compute-type f32_r
```

Prints JSON with `solution_name`, `solution_index`, achieved `timing` (gflops, GB/s, microseconds), and the verbatim `bench_args` so the call is reproducible by hand.

### Package the chosen kernel

```bash
python3 okl.py -m 512 -n 512 -k 512 --transa T --transb N \
    --a-type bf16_r --b-type bf16_r --c-type bf16_r --d-type bf16_r \
    --compute-type f32_r --package /tmp/mypkg
```

Produces:

- `/tmp/mypkg/kernel.co` — copy of the shipped `.co` shard containing the chosen kernel
- `/tmp/mypkg/kernel.conf` — key=value config: kernel symbol, kernarg size, buffer roles to allocate, slot list (offset/size/kind/value or buffer ref) extracted from the `.co`'s `amdhsa.kernels[*].args` metadata

### Build and run the standalone benchmark

```bash
/opt/rocm/bin/hipcc -O3 -std=c++17 okl_run.cpp -o okl_run
./okl_run /tmp/mypkg/kernel.conf
```

Output:

```
conf:      /tmp/mypkg/kernel.conf
co:        /tmp/mypkg/kernel.co
kernel:    Cijk_Alik_Bljk_BBS_BH_UserArgs_MT32x32x128_MI16x16x1_...
resources: regs=256 lds=51200 bytes
problem:   M=512 N=512 K=512 batch=1
kernarg:   104 bytes, 22 slots, 4 buffers
grid:      256 workgroups x 256 threads = 65536 global threads
iters:     500 hot (after 500 cold), single sync, CPU wall clock
time:      4.9 us / iter   (hot window: 2450 us / 500 calls)
perf:      54000 gflops
verify:    OK (D fully overwritten and zero, as expected for A=B=C=0)
```

`okl_run` links only against `libamdhip64` / `libhsa-runtime64`. No Tensile, no hipBLASLt.

## How it differs from `hipblaslt-bench`

Both load the same `.co` and run the same kernel. The difference is what gets timed:

- `hipblaslt-bench` wraps each launch in `hipblaslt_ext::Gemm::run(stream)`, which validates args, looks up the algorithm, prepares the kernarg, then launches. The reported time includes that per-call API overhead.
- `okl_run` builds the kernarg once at startup, then loops on raw `hipExtModuleLaunchKernel`. The reported time is just the kernel.

For a 3072×1536×256 bf16 GEMM on MI300X, that difference is ~1.7 µs per call — visible at this shape, dominant for tiny shapes, negligible for huge ones.

## Caveats

- **Legacy in-kernarg ABI only.** Most shipped GEMM kernels use this. Grouped-gemm (multiple problems via `DeviceUserArguments` in HBM) is not implemented.
- **One kernel per package.** A package is (one kernel, one problem). Sweeping shapes means re-packaging.
- **Faithful replay, not generic GEMM.** The conf captures the constants Tensile actually pushed for the captured problem — `internalArgs`, `numWG`, strides, etc. Changing M/N/K in the conf without re-running `okl.py --package` will produce undefined behavior.
- **No correctness check against a reference.** `verify_d_buffer` only asserts the kernel wrote to D and (since A=B=C=0) D ended up zero. For real numerical comparison against cuBLAS / custom kernels, add a reference path.
- **StreamK / MX block scales / sparsity** kernels package without error but their workspace/scale buffer sizing in `okl.py` is a guess and was never exercised. Treat results from those with caution; check `kernel-packaging-research.md` §9.8 for what's documented.

## Where to learn more

`kernel-packaging-research.md` in this folder. It has the bundle format, the launcher API path through Tensile's `HipSolutionAdapter`, the kernarg ABI (legacy and DeviceUserArguments variants), the design rationale for both the loading and argument-feeding interfaces, and worked dumps from real `.co` files.
