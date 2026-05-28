# okl: Optimal Kernel Lookup + Standalone Packaging

Two **independent** tools that talk to each other through an intermediate file format. The reason this matters is that they have very different dependency requirements and can be used separately:

- **`okl.py`** — query hipBLASLt to find the optimal kernel for a GEMM shape, optionally package the chosen kernel as a self-describing config + a copy of the shipped `.co`. **Requires a full hipBLASLt install.**
- **`okl_run.cpp`** — standalone HIP-only program that loads one of those packages and runs the kernel directly. **No hipBLASLt, no Tensile — just HIP runtime.** The package can be produced on a hipBLASLt-equipped box, copied to a HIP-only box, and run there.

`okl.py` is *one way* to produce the input that `okl_run` consumes. The package format (`kernel.conf` + `kernel.co`) is the contract between them; anything that emits a conformant conf and a valid `.co` can drive `okl_run`. `okl_run.cpp` doubles as a reference implementation of how a third-party (cuBLAS comparison harness, custom kernel framework, etc.) would load and dispatch a hipBLASLt-tuned kernel from raw HIP.

## Files in this folder

| File | Role | Hard deps |
|---|---|---|
| `okl.py` | Driver: query hipBLASLt heuristic, optionally package the result | hipBLASLt install (`hipblaslt-bench`, shipped device library), `clang-offload-bundler`, `llvm-readobj`, Python 3 |
| `okl_run.cpp` | Standalone runner: load a package, launch the kernel, time it | `hipcc` to build; `libamdhip64` + `libhsa-runtime64` to run. **No hipBLASLt, no Tensile.** |
| `package_examples.py` | Convenience: drive `okl.py --package` for several canonical shapes | Same as `okl.py` |
| `compare_okl_vs_bench.py` | Run both `okl_run` and `hipblaslt-bench` on the same package and write a comparison table | Both stacks |
| `packages/` | Output of `package_examples.py`: one subdir per packaged kernel | — |
| `kernel-packaging-research.md` | Design notes + reference. Sections §1–§9 cover .co on-disk format, SolutionAdapter, kernarg ABI, the loading interface, and the metadata-driven argument-feeding interface. Read this before changing either tool. | — |
| `AGENTS.md` | Guidance for AI agents working in this folder | — |

## Part 1 — `okl.py` (needs hipBLASLt)

Drives `hipblaslt-bench` to find the heuristic-winning kernel for a problem and optionally packages it.

### Requirements
- A hipBLASLt install with its shipped device library available. Tested with ROCm 6.4.3 and 7.2.1.
- `hipblaslt-bench` binary (ships with hipBLASLt; `okl.py` shells out).
- `clang-offload-bundler` (in any of `/opt/rocm/lib/llvm/bin/`, `/opt/rocm-*/lib/llvm/bin/`) — used to unbundle the `.co` for metadata extraction in `--package` mode.
- `llvm-readobj` (in `/usr/bin/` or any ROCm `lib/llvm/bin/`) — used to read the kernel's HSA `.args` metadata.

Versions of `hipblaslt-bench`, the device library, and the HIP runtime must be ABI-consistent — mixing a `hipblaslt-bench` from one ROCm with a `TensileLibrary_lazy_*.dat` from another typically segfaults. Pass `--bench` and `--libpath` to pin a known-good pair.

### Query the heuristic (no packaging)

```bash
python3 okl.py -m 512 -n 512 -k 512 --transa T --transb N \
    --a-type bf16_r --b-type bf16_r --c-type bf16_r --d-type bf16_r \
    --compute-type f32_r
```

Prints JSON with `solution_name`, `solution_index`, achieved `timing` (gflops, GB/s, microseconds), and the verbatim `bench_args` so the call is reproducible by hand. This is the same number hipblaslt-bench reports; `okl.py` is just a friendlier CLI around it.

### Package the chosen kernel

```bash
python3 okl.py -m 512 -n 512 -k 512 --transa T --transb N \
    --a-type bf16_r --b-type bf16_r --c-type bf16_r --d-type bf16_r \
    --compute-type f32_r --package /tmp/mypkg
```

Produces a self-contained directory:

- `/tmp/mypkg/kernel.co` — copy of the shipped `.co` shard containing the chosen kernel.
- `/tmp/mypkg/kernel.conf` — key=value config: kernel symbol, kernarg size, buffer roles to allocate, slot list (offset/size/kind/value or buffer ref) extracted from the `.co`'s `amdhsa.kernels[*].args` metadata.

That directory is the handoff to `okl_run`. **It can be tar'd up and copied to another machine** (provided that machine has the same GPU arch and a compatible HIP runtime).

## Part 2 — `okl_run.cpp` (HIP-only, hipBLASLt **not** required)

Standalone HIP program that loads a `kernel.conf` + `kernel.co` package and runs the kernel with hipblaslt-bench-style timing methodology. Implements the load + kernarg-pack + launch path from scratch over the HIP runtime — useful as the kernel-launching equivalent of how a third-party benchmark harness (cuBLAS comparison, custom kernel framework, etc.) would drive a hipBLASLt-tuned kernel.

### Requirements
- **Build:** `hipcc` (any recent ROCm; tested with 7.2.1).
- **Run:** `libamdhip64` and `libhsa-runtime64`. That's it. **No `libhipblaslt`, no `libTensile`, no shipped device library.** Confirm with `ldd okl_run` — there should be no hipBLASLt or Tensile entries.
- A `.co` file built for the GPU you'll run on. The `.co` shipped by hipBLASLt is one source; a `.co` you compile yourself from any hipcc-compiled GEMM source would also work, as long as `kernel.conf` describes its kernarg layout correctly.

### Build

```bash
/opt/rocm/bin/hipcc -O3 -std=c++17 okl_run.cpp -o okl_run
```

### Run a packaged kernel

`okl_run` takes one positional argument: the path to a `kernel.conf`. The `.co` file is found via the `co_file = ...` line inside the conf and is resolved **relative to the conf file's directory**, so you point at the conf and the binary follows the breadcrumbs:

```bash
# Run one of the five pre-packaged kernels from ./packages/
./okl_run packages/small_square/kernel.conf

# Or any package directory anywhere on disk
./okl_run /home/me/kernels/bf16-2048cube/kernel.conf

# A package can be moved or renamed; only its internal layout matters.
# As long as `kernel.co` (or whatever `co_file =` points at) sits next to
# the conf in the relative path the conf declares, okl_run will find it.
```

### What you get

```
conf:      packages/small_square/kernel.conf
co:        packages/small_square/kernel.co
kernel:    Cijk_Alik_Bljk_BBS_BH_UserArgs_MT64x64x128_MI16x16x1_SN_LDSB1_AFC...
resources: regs=256 lds=33024 bytes              # from hipFuncGetAttribute
problem:   M=1024 N=1024 K=1024 batch=1          # echoed from conf
kernarg:   104 bytes, 22 slots, 4 buffers        # what the runner built
grid:      256 workgroups x 256 threads = 65536 global threads
iters:     500 hot (after 500 cold), single sync, CPU wall clock
time:      7.7 us / iter   (hot window: 3850 us / 500 calls)
perf:      280000 gflops                         # 2*M*N*K*batch / time
verify:    OK (D fully overwritten and zero, as expected for A=B=C=0)
```

Exit code 0 on success, 1 on load/launch error, 2 on verify failure (kernel didn't write D).

### What `okl_run` demonstrates

The cpp file is the reference for **how a hipBLASLt-tuned kernel expects to be loaded and called** from raw HIP:

- Module loading via `hipModuleLoad` + symbol resolution via `hipModuleGetFunction`.
- Resource probe via `hipFuncGetAttribute(NUM_REGS|SHARED_SIZE_BYTES)`.
- Kernarg byte-buffer construction driven by the slot list (offsets/sizes/types from the kernel's own HSA `.args` ELF metadata — no hardcoded ABI knowledge).
- Buffer allocation per the role list (D/C/A/B and any feature-gated buffers like `bias`, `Synchronizer`, etc.).
- Driver-style launch via `hipExtModuleLaunchKernel` with a `HIP_LAUNCH_PARAM_BUFFER_POINTER` array.
- Steady-state timing (500 cold + 500 hot iters, single sync, CPU wall clock).

If you're integrating a hipBLASLt kernel into your own benchmark harness, `okl_run.cpp` is the minimal code path you'd reimplement.

## Package format (the contract between the two tools)

A package is a directory with two files:

- **`kernel.co`** — clang-offload-bundle containing one gfx-target ELF with the kernel symbol. Position-independent; loadable via `hipModuleLoad`.
- **`kernel.conf`** — plain-text key=value config describing what the kernel expects in kernarg memory and what buffers to allocate for it.

You can produce these with `okl.py --package`, with your own tooling, or by hand. As long as `kernel.conf` correctly describes the kernel in `kernel.co`, `okl_run` (or anything else implementing the same contract) will launch it correctly.

### Anatomy of `kernel.conf` (real example: `packages/small_square/kernel.conf`)

```
# --- Scalars ---
co_file                 = kernel.co                    # path relative to conf's dir
kernel_symbol           = Cijk_Alik_Bljk_BBS_BH_UserArgs_MT64x64x128_..._WG32_8_1
kernarg_size            = 104                          # total kernarg buffer bytes
workgroup_size_threads  = 256                          # threads per workgroup

# Diagnostic only (not used by the launch; echoed in the runner preamble):
m                       = 1024
n                       = 1024
k                       = 1024
batch                   = 1
macro_tile_0            = 64
macro_tile_1            = 64

# --- Buffers ---
# One line per device buffer the kernel reads/writes. `name` is the "role" --
# a string slot lines refer to. `init` is 'zero' or 'poison' (0xee).
buffer = name=D bytes=2097152 init=poison
buffer = name=C bytes=2097152 init=zero
buffer = name=A bytes=2097152 init=zero
buffer = name=B bytes=2097152 init=zero

# --- Slots ---
# One line per kernarg field. The runner writes them in ANY order at their
# declared `offset`; the only thing that matters is offsets cover the kernarg
# buffer without overlap. Each slot is either `kind=value` (a literal byte
# pattern encoded per `ctype`) or `kind=buffer` (the 8-byte device pointer of
# a buffer declared above, referenced by role name).
slot = offset=0   size=4 kind=value  ctype=u32 value=0x1          name=Gemm_info
slot = offset=4   size=4 kind=value  ctype=u32 value=0x20080001   name=kernel_info0
slot = offset=8   size=4 kind=value  ctype=u32 value=0x4c010000   name=kernel_info1
slot = offset=12  size=4 kind=value  ctype=u32 value=0x100        name=numWG
slot = offset=16  size=4 kind=value  ctype=u32 value=0x400        name=SizesFree0
slot = offset=20  size=4 kind=value  ctype=u32 value=0x400        name=SizesFree1
slot = offset=24  size=4 kind=value  ctype=u32 value=0x1          name=SizesFree2
slot = offset=28  size=4 kind=value  ctype=u32 value=0x400        name=SizesSum0
slot = offset=32  size=8 kind=buffer buffer=D                     name=D
slot = offset=40  size=8 kind=buffer buffer=C                     name=C
slot = offset=48  size=8 kind=buffer buffer=A                     name=A
slot = offset=56  size=8 kind=buffer buffer=B                     name=B
slot = offset=64  size=4 kind=value  ctype=u32 value=0x400        name=strideD0
... (more strides + alpha + beta)
```

Reserved scalar keys: `co_file`, `kernel_symbol`, `kernarg_size`, `workgroup_size_threads`, `m`, `n`, `k`, `batch`, `macro_tile_0`, `macro_tile_1`. Repeatable line forms:

- `buffer = name=<role> bytes=<N> init=<zero|poison>` — one per logical buffer the kernel reads/writes.
- `slot = offset=<O> size=<S> kind=<value|buffer> [ctype=<u32|i32|u64|i64|u16|i16|u8|i8|f32|f64|pkf16> value=<literal>] [buffer=<role>] name=<n>` — one per kernarg field.

`#` starts a line comment. Order doesn't matter except as a convention.

### Writing your own runner (language-agnostic)

Implementing the package contract is ~150 lines in C++ and even less in Python (via cuda-python / pyhip / nanobind). What a runner has to do, in order:

1. **Parse `kernel.conf`** into scalars + a list of buffers + a list of slots.
2. **Resolve `co_file`** relative to the conf's directory (if it's a relative path).
3. **Load the module:** `hipModuleLoad(&module, co_path)`.
4. **Resolve the kernel:** `hipModuleGetFunction(&fn, module, kernel_symbol)`.
5. **(Optional)** Probe resources: `hipFuncGetAttribute(NUM_REGS|SHARED_SIZE_BYTES, fn)`.
6. **Allocate every buffer**, recording the role → device pointer mapping:
   ```
   for b in buffers:
       hipMalloc(&p, b.bytes)
       hipMemset(p, 0xee if b.init == "poison" else 0, b.bytes)
       ptrs[b.role] = p
   ```
7. **Build the kernarg byte buffer** (size = `kernarg_size`, zero-initialized):
   ```
   for s in slots:
       if s.kind == "buffer":
           memcpy(kernarg + s.offset, &ptrs[s.buffer], 8)
       else:  # kind == "value"
           bytes = encode(s.ctype, s.value, s.size)   # u32 -> 4 LE bytes, etc.
           memcpy(kernarg + s.offset, bytes, s.size)
   ```
8. **Compute the launch grid.** Find the slot named `numWG` in the slot list — that's the workgroup count (Tensile collapses 2D/3D grids into a single dimension via its internal `numWG` field). Global thread count = `numWG * workgroup_size_threads`.
9. **Launch with the driver-style param buffer:**
   ```
   void* launch_params[] = {
       HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg.data(),
       HIP_LAUNCH_PARAM_BUFFER_SIZE,    &kernarg_size,
       HIP_LAUNCH_PARAM_END,
   };
   hipExtModuleLaunchKernel(
       fn, globalX, 1, 1, workgroup_size_threads, 1, 1,
       /*sharedMemBytes=*/0, stream, nullptr, launch_params, nullptr, nullptr);
   ```
10. **Time it.** Warmup, then a tight loop with a single sync at the end; total / iters = per-call cost.
11. **Cleanup:** `hipFree` every buffer, `hipModuleUnload(module)`.

That's the entire contract. There are no special calls back into hipBLASLt or Tensile.

### Where `okl_run.cpp` implements each step

| Step | `okl_run.cpp` function | Roughly |
|---|---|---|
| 1, 2 | `load_config()`, `co_path` resolution in `main()` | top half of file |
| 3, 4, 5 | `load_kernel()` | one helper, ~40 lines |
| 6 | `allocate_buffers()` | returns `BufferSet` |
| 7 | `build_kernarg()` (walks slots) + `encode_value()` (per-ctype) | two helpers |
| 8, 9 | `time_kernel()` | inner launch lambda |
| 10 | `time_kernel()` | cold + hot loop |
| 11 | `cleanup()` | one helper |

Read `main()` for the call sequence; read each helper for the implementation. Total: ~600 lines including header comments. If you're porting to another language, that's your spec.

See `kernel-packaging-research.md` §9 for the full slot-list design and rationale (why this format, why metadata-driven, what the `value_kind` values mean, dtype-aware encoding, etc.).

## How `okl_run` differs from `hipblaslt-bench`

Both load the same `.co` and run the same kernel. The difference is what gets timed:

- `hipblaslt-bench` wraps each launch in `hipblasLtMatmul` (or `hipblaslt_ext::Gemm::run`), which validates args, looks up the algorithm, prepares the kernarg, and launches. The reported time includes that per-call API overhead, plus any on-stream work hipBLASLt does around the kernel.
- `okl_run` builds the kernarg once at startup, then loops on raw `hipExtModuleLaunchKernel`. The reported time is just the kernel.

For an 8192³ bf16 GEMM on MI300X the gap is ~474 µs/iter (~40% on TFLOPS). See `packages/comparison.md` and `packages/timing-gap-investigation.md` for measurements and partial analysis.

For comparing a kernel against a non-hipBLASLt implementation (cuBLAS, custom), the `okl_run` number is the fairer one. For predicting what a real hipBLASLt user observes, the bench number is.

## Caveats

- **Legacy in-kernarg ABI only.** Most shipped GEMM kernels use this. Grouped-gemm (multiple problems via `DeviceUserArguments` in HBM) is not implemented.
- **One kernel per package.** A package is (one kernel, one problem). Sweeping shapes means re-packaging.
- **Faithful replay, not generic GEMM.** The conf captures the constants Tensile actually pushed for the captured problem (`internalArgs`, `numWG`, strides, etc.). Changing M/N/K in the conf without re-running `okl.py --package` will produce undefined behavior.
- **No correctness check against a reference.** `verify_d_buffer` only asserts the kernel wrote to D and (since A=B=C=0) D ended up zero. For real numerical comparison, add a reference path.
- **StreamK / MX block scales / sparsity** kernels package without error but their workspace/scale buffer sizing in `okl.py` is a guess and was never exercised. Treat results from those with caution; check `kernel-packaging-research.md` §9.8.

## Where to learn more

`kernel-packaging-research.md` in this folder. It has the bundle format, the launcher API path through Tensile's `HipSolutionAdapter`, the kernarg ABI (legacy and DeviceUserArguments variants), the design rationale for both the loading and argument-feeding interfaces, and worked dumps from real `.co` files.
