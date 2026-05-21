# Lowering Pipeline

`ck_dsl` has three code-emission paths. They serve different purposes; do not confuse them.

```text
Production:
  KernelDef SSA IR
    -> AMDGPU LLVM IR text       (core/lower_llvm.py)
    -> libamd_comgr               (runtime/comgr.py)
    -> HSACO bytes
    -> hipModuleLoadData + hipModuleLaunchKernel  (runtime/hip_module.py)

Debug:
  KernelDef SSA IR -> HIP C++ text  (core/lower_hip.py)

CK Tile parity:
  selected spec dataclasses -> CK Tile C++ text  (core/lower_cktile.py)
```

## Production Path

Top-level entry point:

```python
from ck_dsl.helpers import compile_kernel
art = compile_kernel(
    kernel,
    isa="amdgcn-amd-amdhsa--gfx950",   # default
    capture_ir_text=True,
    optimize_ir=False,
)
```

Implementation (`helpers/compile.py`):

```text
1. optimize_kernel(kernel) if optimize_ir         # passes (constant fold + CSE + DCE)
2. print_ir(kernel) if capture_ir_text            # MLIR-style debug dump
3. lower_kernel_to_llvm(kernel)                   # AMDGPU LLVM IR text
4. build_hsaco_from_llvm_ir(llvm_text, isa=...)  # libamd_comgr
5. return KernelArtifact(kernel, ir_text, llvm_text, hsaco, timings, pass_stats, isa)
```

`KernelArtifact.timings` (all in milliseconds):

```text
ir_opt              # only populated when optimize_ir=True
ir_build            # print_ir
ir_lower_llvm       # lower_kernel_to_llvm
comgr_bc            # LLVM IR -> bitcode
comgr_relocatable   # bitcode -> relocatable ELF
comgr_executable    # relocatable -> HSACO
total
```

Typical total times observed during validation (gfx950, ROCm 7.0.2, warm `libamd_comgr`):

```text
implicit-GEMM conv (this repo, build_implicit_gemm_conv):   ~17 ms
universal GEMM (smallest hero shape):                       ~10-30 ms
elementwise / reduce / norm / transpose:                    ~5-15 ms
```

## LLVM Lowering Details

Entry point: `core/lower_llvm.py::lower_kernel_to_llvm(kernel: KernelDef) -> str`.

`_Lowerer` walks regions and emits LLVM IR text:

- Datalayout (`_DATALAYOUT`) is the clang-emitted gfx950 layout, copied verbatim. If the ROCm version changes, regenerate via `clang -target amdgcn-amd-amdhsa -mcpu=gfx950 -emit-llvm -S`.
- Target triple: `amdgcn-amd-amdhsa`.
- Only intrinsics actually used by the kernel are declared (see `_INTRINSIC_DECLS`).
- LDS allocations from `tile.smem_alloc` become module-level `addrspace(3)` globals collected in a pre-pass; uses become GEPs.
- The kernel function is `define amdgpu_kernel void @kernel_name(...) #0` with at minimum:

```llvm
attributes #0 = {
  "uniform-work-group-size"="true"
  "amdgpu-flat-work-group-size"="64,N"     ; N = kernel.attrs["max_workgroup_size"]
  ...
  ; optional, only emitted when kernel.attrs["waves_per_eu"] is set:
  "amdgpu-waves-per-eu"="lo,hi"
}
```

## Op-to-LLVM Mapping

| IR op                          | LLVM emission                                                              |
|--------------------------------|----------------------------------------------------------------------------|
| `gpu.thread_id x`              | `call i32 @llvm.amdgcn.workitem.id.x()`                                    |
| `gpu.block_id x/y/z`           | `call i32 @llvm.amdgcn.workgroup.id.*()`                                   |
| `tile.lane_id`                 | `mbcnt.hi(-1, mbcnt.lo(-1, 0))`                                            |
| `tile.smem_alloc`              | `internal addrspace(3) global <[NxT]> undef` + GEP at use sites             |
| `tile.smem_load_vN_f16`        | `load <Nxhalf>, ptr addrspace(3) ...`                                      |
| `tile.smem_store_vN_f16`       | `store <Nxhalf>, ptr addrspace(3) ...`  (lowers to `ds_write_b{16..128}`)  |
| `tile.smem_addr_of` / `smem_ptr_add` | `ptrtoint` / `add` / `inttoptr` on addrspace(3)                       |
| `memref.global_load_typed`     | `getelementptr` + typed `load addrspace(1)`                                |
| `memref.global_load_vN`        | typed vector load in `addrspace(1)`                                        |
| `memref.global_store_typed`    | typed `store addrspace(1)`                                                 |
| `tile.buffer_rsrc`             | `llvm.amdgcn.make.buffer.rsrc.p1(ptr, i16 0, i32 num_bytes, i32 159744)`  |
| `tile.buffer_load_vN_f16`      | `llvm.amdgcn.raw.ptr.buffer.load.{i32,v2i32,v4i32}` + `bitcast` to halves  |
| `tile.async_buffer_load_lds`   | `llvm.amdgcn.raw.ptr.buffer.load.lds`                                      |
| `tile.s_waitcnt`               | `llvm.amdgcn.s.waitcnt(i32 encoded_imm)`                                   |
| `tile.sync`                    | `s.waitcnt(vmcnt=0, lgkmcnt=0)` + `s.barrier`                              |
| `tile.sync_lds_only`           | `s.waitcnt(lgkmcnt=0)` + `s.barrier`                                       |
| `tile.sched_group_barrier`     | `llvm.amdgcn.sched.group.barrier(i32 mask, i32 count, i32 sync_id)`        |
| `tile.s_setprio`               | `llvm.amdgcn.s.setprio(i16 prio)`                                          |
| `tile.mfma_f32_16x16x16_f16`   | `llvm.amdgcn.mfma.f32.16x16x16f16(<4xhalf>, <4xhalf>, <4xfloat>, 0,0,0)`   |
| `tile.mfma_f32_16x16x32_f16`   | `llvm.amdgcn.mfma.f32.16x16x32.f16(<8xhalf>, <8xhalf>, <4xfloat>, 0,0,0)`  |
| `tile.mfma_f32_16x16x16_bf16`  | `llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4xi16>, <4xi16>, <4xfloat>, 0,0,0)` |
| `tile.mfma_f32_16x16x32_bf16`  | `llvm.amdgcn.mfma.f32.16x16x32.bf16(<8xbfloat>, <8xbfloat>, <4xfloat>, 0,0,0)` |
| `tile.mfma_f32_32x32x8_f16`    | `llvm.amdgcn.mfma.f32.32x32x8f16(<4xhalf>, <4xhalf>, <16xfloat>, 0,0,0)`   |
| `tile.mfma_f32_32x32x16_f16`   | `llvm.amdgcn.mfma.f32.32x32x16.f16(<8xhalf>, <8xhalf>, <16xfloat>, 0,0,0)` |
| `tile.mfma_f32_4x4x4_f16`      | `llvm.amdgcn.mfma.f32.4x4x4f16(<4xhalf>, <4xhalf>, <4xfloat>, 0,0,0)`      |
| `tile.readfirstlane`           | `llvm.amdgcn.readfirstlane.{i32,i64}`                                      |
| `tile.pin_sgpr`                | `asm volatile("" : "+s"(x))`                                               |
| `tile.wave_all` / `wave_any`   | `ballot.i64` + compare                                                     |
| `tile.wave_ballot`             | `llvm.amdgcn.ballot.i64`                                                   |
| `tile.ds_bpermute`             | `llvm.amdgcn.ds.bpermute(i32 addr, i32 data)`                              |
| `tile.ds_read_tr16_b64`        | `llvm.amdgcn.ds.read.tr16.b64(ptr addrspace(3))`                           |
| `arith.cvt_fp8_to_f32`         | `llvm.amdgcn.cvt.f32.fp8(i32, i32 imm)`                                    |
| `arith.cvt_f32_to_fp8`         | `llvm.amdgcn.cvt.pk.fp8.f32(f32, f32 0, i32 imm)` (low-byte extract)       |
| `scf.for` / `scf.for_iter`     | header / body / latch / exit blocks with phi nodes                         |
| `scf.if`                       | conditional branch + then/else/join blocks                                 |

## The 0x00027000 Buffer Resource Flag

`buffer_rsrc(ptr, num_bytes)` lowers to:

```llvm
call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p1(
  ptr addrspace(1) %ptr, i16 0, i32 %num_bytes, i32 159744)
```

`159744 = 0x00027000` encodes DWORD3 of the AMDGPU buffer resource descriptor:

- `TYPE = 2` (BUFFER_RESOURCE)
- `DATA_FORMAT = 4` (32-bit dword)
- `NUM_FORMAT = 4` (UINT)

This is the canonical bounds-checked configuration. Out-of-range byte offsets silently return zero on load and are dropped on store. The "OOB sentinel" pattern used everywhere in the DSL relies on this: a `select(valid, real_off_bytes, INT32_MAX)` makes false-mask lanes safe without a software branch.

`INT32_MAX = 2147483647 = (1 << 31) - 1` is the default `oob_sentinel` in `AsyncTileLoader.issue` and the loader helpers.

A wrong DW3 flag (the `0` default in the LLVM intrinsic without these encoded bits) yields a single dword load with no bounds check, which then misreads padded boundary positions as the next row of A. This was a real correctness bug fixed during the implicit-GEMM conv bake-off (see `RUNBOOK_COMPLIANCE.md` empirical pass).

## COMGR

Entry point: `runtime/comgr.py::build_hsaco_from_llvm_ir(ir_text, isa=..., options=...)`.

The pipeline:

```text
amd_comgr_create_data_set
amd_comgr_create_data(SOURCE)
  set body = ir_text (utf-8) + name = "kernel.ll"
amd_comgr_action_info_set_isa_name(isa)             # default "amdgcn-amd-amdhsa--gfx950"
amd_comgr_action_info_set_language(LLVM_IR)
amd_comgr_action_info_set_option_list(["-O3"])       # default

amd_comgr_do_action(COMPILE_SOURCE_TO_BC)            # bitcode
amd_comgr_do_action(CODEGEN_BC_TO_RELOCATABLE)       # relocatable ELF
amd_comgr_do_action(LINK_RELOCATABLE_TO_EXECUTABLE)  # HSACO
amd_comgr_action_data_get_data(EXECUTABLE)           # extract bytes
```

Libraries are loaded from the default ROCm library locations or the dynamic
linker search path. The HSACO bytes returned are ready for `hipModuleLoadData`.

`ComgrTimings(bc, relocatable, executable)` returns per-stage seconds; `compile_kernel` converts to ms in the artifact.

## HIP Debug Lowering

Entry point: `core/lower_hip.py::lower_kernel_to_hip(kernel, launch_bounds=None, include_prologue=True) -> str`.

This backend emits readable HIP C++:

- `extern "C" __global__ __launch_bounds__(N)` for kernel signature;
- typedef aliases for ext-vector types: `f16x4`, `f32x16`, `i32x4`, `boolx8`, etc.;
- builtin shims for clang-unfriendly intrinsics (`ds.read.tr16.b64`, `raw.ptr.buffer.load.lds`);
- `__shared__ <elem> name_storage[...]` for LDS allocations;
- `__builtin_amdgcn_make_buffer_rsrc((void*)ptr, /*stride=*/(short)0, /*num_records=*/num_bytes, /*flags=*/0x00027000)`;
- `__builtin_amdgcn_mfma_*` for matrix ops;
- inline `asm volatile("" : "+s"(x))` for `pin_sgpr`;
- normal C++ `for` / `if` for structured control flow.

Use cases:

- side-by-side reading of generated code;
- making a smaller `clang -x hip` repro for an LLVM bug;
- visual comparison of IR intent to a readable source shape.

Caveats:

- unsupported ops raise `NotImplementedError`;
- op coverage is narrower than LLVM lowering;
- some debug shims work around clang builtin limitations;
- bf16 / fp8 vector constant emission paths are less battle-tested;
- this is not the benchmark path; do not infer production behavior from HIP output.

## CK Tile Spec Emission

Entry point: `core/lower_cktile.py::lower_spec_to_cktile(spec, kernel_name=None)`.

Supported specs:

- `UniversalGemmSpec` -> `lower_universal_gemm_to_cktile(...)`
- `ImplicitGemmConvSpec` -> `lower_implicit_gemm_conv_to_cktile(...)`

For each, the backend assembles CK Tile-style C++:

- CK Tile headers;
- `GemmConfig` / `ConvConfig` static constexpr fields populated from the spec;
- dtype/layout aliases;
- `TileGemmShape`, `GemmPipeline`, `CShuffleEpilogue`, `GemmKernel` (and the grouped conv equivalents);
- `extern "C" float launch_<name>(...)` host launcher.

Current scope is narrow on purpose:

- GEMM is fp16 in/out + fp32 acc with a three-character `R`/`C` layout string;
- conv is fp16 NHWC/KRSC/NHWK, 2D spatial;
- unsupported pipeline/scheduler/epilogue choices raise.

Do not build new production features only in this backend.

## Analysis After Lowering

The runbook requires evidence that a speed claim maps to real generated code. Use:

```python
from ck_dsl import analyze_llvm_ir, analyze_hsaco

ir_stats   = analyze_llvm_ir(art.llvm_text)
hsaco_stats = analyze_hsaco(hsaco_path)
```

Look for:

- intended MFMA shapes and counts;
- `raw.ptr.buffer.load.lds` calls when expecting async DMA;
- vector stores (`global_store_dwordx{2,4}`, `buffer_store_dwordx{2,4}`);
- waitcnt placement;
- `s_barrier` count;
- VGPR / SGPR usage;
- static LDS bytes;
- unexpected scalarization or missing cshuffle vectorization.

See `runtime/limitations.md` for what these tools can and cannot tell you.

## Lowering Invariants

Production lowering assumes:

- wave64 (`wave_size=64` in helpers); MFMA lane mappings depend on this.
- valid address space and vector type combinations.
- explicit synchronization inserted by the builder/helper; lowering does not add barriers.
- structured control flow regions are well formed (matching `scf_yield` operand count, etc.).
- LDS allocations are visible before consumer ops (the pre-pass walks the body).
- operation names are in the lowering vocabulary.

If you add a new builder operation, update in this order:

1. `core/ir.py` — builder method and the op name + result type.
2. `core/ir_print.py` — only if you want a clean MLIR-style debug print.
3. `core/lower_llvm.py` — production lowering and any new `_INTRINSIC_DECLS` entry.
4. `core/lower_hip.py` — optional readable debug output.
5. tests/examples that lower the op and assert the expected LLVM/ISA shape.
