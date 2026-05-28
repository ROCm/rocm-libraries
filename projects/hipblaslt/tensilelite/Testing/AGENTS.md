# AI Agent Guidance: `okl` tooling

Read `README.md` first for what these tools do and how to drive them. This file covers what's load-bearing about the design and what you should not break.

## Two separate tools with different dependency surfaces

This is the first thing to keep clear when editing here:

- **`okl.py`** drives `hipblaslt-bench` and reads the shipped `.co` files. It **requires a full hipBLASLt install** (the bench binary + the device library + the LLVM tools to unbundle and read kernel metadata).
- **`okl_run.cpp`** loads a `kernel.conf` + `kernel.co` package and runs the kernel. It **requires only the HIP runtime** (`libamdhip64` + `libhsa-runtime64`). No hipBLASLt, no Tensile, no shipped device library. It is the kernel-launching equivalent of how a third-party harness (cuBLAS comparison, custom kernel framework) would drive a hipBLASLt-tuned kernel.

The package (`kernel.conf` + `kernel.co`) is the only contract between them. Any edit that quietly tightens that contract on the C++ side — adding a runtime call into hipBLASLt, requiring the shipped device library, etc. — collapses the two tools into one and defeats the design.

## What lives here

| File | Role | Hard deps |
|---|---|---|
| `okl.py` | Driver: query hipBLASLt heuristic, optionally package the chosen kernel | hipBLASLt install (`hipblaslt-bench` + shipped device library), `clang-offload-bundler`, `llvm-readobj`, Python 3 |
| `okl_run.cpp` | Standalone runner: load a package, launch the kernel, time it | Build: `hipcc`. Run: `libamdhip64` + `libhsa-runtime64`. **NOT** hipBLASLt, **NOT** Tensile. |
| `package_examples.py` | Convenience: drive `okl.py --package` for several canonical shapes | Same as `okl.py` |
| `compare_okl_vs_bench.py` | Run both runners on each package and write a comparison file | Both stacks |
| `packages/` | Output of `package_examples.py` (one subdir per packaged kernel) and of `compare_okl_vs_bench.py` (`comparison.json` / `comparison.md`) | — |
| `kernel-packaging-research.md` | Reference doc (~1900 lines). Sections §1–§9 cover everything from on-disk `.co` layout to the metadata-driven argument-feeding design. Always check the relevant section before changing either tool. | — |

## Load-bearing invariants

These are the design choices that hold the system together. Don't undo them without a deliberate refactor.

1. **`okl_run.cpp` links only against `libamdhip64` / `libhsa-runtime64`.** No Tensile, no hipBLASLt, no shipped device library. The whole point of the standalone exe is to be the kernel-launching equivalent of how a third party (cuBLAS comparison, custom kernel) would run things — and to be usable on machines that don't have hipBLASLt installed at all. Adding a `#include <Tensile/...>`, a `-lhipblaslt`, or a runtime dependency on `HIPBLASLT_TENSILE_LIBPATH` would collapse the two tools into one and defeat the project. Verify with `ldd okl_run` after any change to the build line.

2. **No kernel-specific constants in the C++.** Every kernel-dependent value (kernarg offsets, `internalArgs` bit-packing, MT/WG sizes, buffer roles, alpha/beta encoding) lives in `kernel.conf`, populated by `okl.py` from either the runtime `TENSILE_DB=0x40` dump or the `.co`'s `amdhsa.kernels[*].args` ELF metadata. If you're about to hardcode `0x20080001` or `put_u32(96, alpha)` somewhere, you're walking back the metadata-driven design from §9.

3. **Conf-format changes are symmetric.** A new conf field needs a writer in `okl.py` (typically in `write_package` or `build_slots`) AND a reader in `okl_run.cpp` (typically in `Config` + `load_config`). Adding to only one side is a silent break.

4. **The slot list is the source of truth for the kernarg.** `okl_run.cpp::build_kernarg` walks `c.slots` in order. To add a kernel feature (bias, activation, scaleA/B, …), add the buffer role to `KNOWN_BUFFER_ROLES` (or the by-value field to `BY_VALUE_FROM_PROBLEM`) in `okl.py`. The C++ should not learn new field names.

5. **Faithful replay, not generic GEMM.** A package is captured for one (kernel, problem) pair. The kernel's tile is fixed; the dump's `internalArgs`, `numWG`, strides, etc. are baked in. Changing M/N/K in `kernel.conf` without re-running `--package` is not supported behavior — it will compile and run but produce undefined results.

## Major API surfaces in each file

### `okl.py`

- `parse_dump(stdout)`: extract kernel symbol, `.co` path, kernarg u32 fields, workgroup size, grid from a `TENSILE_DB=0xF0` bench run. Lossy by design — only fields the runtime printed survive.
- `unbundle_co(co_path)`: shell out to `clang-offload-bundler` to get the amdgcn ELF.
- `parse_kernel_args(elf, symbol)`: hand-rolled YAML walker over `llvm-readobj --notes` output. Returns `(arg_list, kernarg_segment_size)` where each arg has `name`, `offset`, `size`, `value_kind`, `value_type`.
- `KNOWN_BUFFER_ROLES`: maps `value_kind=global_buffer` arg names to (role, init mode, sizing hint). Extend here for new pointer slots.
- `BY_VALUE_FROM_PROBLEM`: maps `value_kind=by_value` arg names to user-problem fields (M, N, K, alpha, strides…). Extend here for new scalar slots.
- `build_slots(...)`: assembles the slot+buffer list. Slots whose values come from the bench's runtime dump (`internal_args`, `numWG`, etc.) are emitted as `ctype=u32 value=0x…` regardless of the kernel-side `value_type` — preserves the exact bit pattern (critical for floats whose dump reading was u32-based).
- `write_package(out_dir, args, dump_info)`: top-level. Calls everything above, copies `.co`, writes conf, returns JSON metadata.

### `okl_run.cpp`

After argument parsing, `main` is 9 numbered stages calling one helper each. The helpers:

- `load_config(path)`: parses the key=value file plus the `buffer = ...` and `slot = ...` mini-DSLs into a `Config` struct.
- `allocate_buffers(c)`: returns `BufferSet { ptrs, sizes }` keyed by role name.
- `load_kernel(co_path, symbol)`: `hipModuleLoad` + `hipModuleGetFunction`, plus a `hipFuncGetAttribute` probe for regs/LDS. Produces clean error messages on missing file / missing symbol / wrong arch.
- `build_kernarg(c, bs)`: walks the slot list, packs the byte buffer. Bounds-checks every slot.
- `time_kernel(fn, kernarg, wg, gthr)`: 500 cold + 500 hot launches, single sync per window, CPU wall clock. Returns `TimingResult`.
- `print_report(...)`: the preamble + perf lines.
- `verify_d_buffer(bs)`: reads D back, checks not-poison + all-zero.
- `cleanup(bs, module)`: `hipFree` + `hipModuleUnload`.

## Validation commands

End-to-end test on the validated 512³ bf16 TN problem:

```bash
/opt/rocm/bin/hipcc -O3 -std=c++17 okl_run.cpp -o okl_run

rm -rf /tmp/okl-validate && python3 ./okl.py \
    -m 512 -n 512 -k 512 --transa T --transb N \
    --a-type bf16_r --b-type bf16_r --c-type bf16_r --d-type bf16_r \
    --compute-type f32_r \
    --bench /opt/rocm-6.4.3/bin/hipblaslt-bench \
    --libpath /opt/rocm-6.4.3/lib/hipblaslt/library \
    --package /tmp/okl-validate

./okl_run /tmp/okl-validate/kernel.conf
```

Expected: `verify: OK`, non-zero gflops in the ~50 TFLOPS range on MI300X.

A bias-bearing kernel (different `.args` shape) is exercised by adding `--bias-vector` to the okl.py call. The 160-byte kernarg variant must package and run without C++ changes.

## Known gaps (documented in research §9.8)

- **DeviceUserArguments path.** Triggered by grouped-gemm with N>1. Not implemented. Adding it means a new conf section for the device-side `DeviceUserArguments` struct + a new packing path in `okl_run.cpp`.
- **StreamK / MX block scales / sparsity.** `okl.py` will package these but the workspace / scale buffer sizing is a guess (e.g., 64 MiB hardcoded for `Synchronizer`). May fault at launch on unusual shapes. To fix: read per-solution workspace size from the library logic YAML before packaging.
- **Reference correctness check.** `verify_d_buffer` only proves the kernel wrote AND the result is zero (cheap trick: zero inputs → zero output). For real numerical comparison vs cuBLAS / custom kernels, add a random-init + host-side reference path.
- **Multi-arch packaging.** A package is one shard. Multi-arch is left as "a directory of packages."

## Coding standards

- C++ in `okl_run.cpp`: C++17. `/** ... */` docstrings on functions and structs. Per-stage numbered comments in `main`. Helpers are `static` and file-local.
- Python in `okl.py`: stdlib only. PEP 8 spacing is fine but don't reformat unrelated code in a diff.
- Conf-file format: key=value with `#` comments. Reserved keys: `co_file`, `kernel_symbol`, `kernarg_size`, `workgroup_size_threads`, `m`, `n`, `k`, `batch`, `macro_tile_0`, `macro_tile_1`, `buffer`, `slot`. New keys are fine; add a reader in `load_config` and a writer in `write_package`.

## Build & test loop

```bash
cd projects/hipblaslt/tensilelite/Testing
/opt/rocm/bin/hipcc -O3 -std=c++17 okl_run.cpp -o okl_run
# diagnostics from the editor's clangd about "hip/hip_runtime.h not found" are
# spurious — clangd doesn't know hipcc's include paths. trust hipcc.
```

Edits to `okl.py` don't need a rebuild — just rerun `--package` and check the conf.

## When in doubt

The research doc (`kernel-packaging-research.md`) is the definitive reference. It cites file:line into Tensile / HIP / HSA for every claim about behavior. If your change feels like it's reinventing something there, re-read the relevant section before proceeding.
