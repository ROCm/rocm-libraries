# Direct Convolution in CK DSL — Architecture and Porting Guide

This document explains:

1. What the CK DSL (Python eDSL under [projects/composablekernel/python/ck_dsl](../../python/ck_dsl)) is and how it works.
2. How CK DSL relates to the [FlyDSL](https://github.com/ROCm/FlyDSL) project.
3. How the existing C++ direct-convolution library at [projects/composablekernel/include/ck_tile/ops/direct_convolution](../../include/ck_tile/ops/direct_convolution) maps onto CK DSL primitives.
4. What is already implemented in CK DSL ([instances/conv_direct_grouped.py](../../python/ck_dsl/instances/conv_direct_grouped.py)) and a concrete plan for porting the remaining 8c / 32c / Dgrad variants.

It is intended as a working reference for someone moving the direct-conv kernels from C++ templates into the Python authoring layer.

> **Note on placement.** The project [CLAUDE.md](../../CLAUDE.md) points detailed direct-convolution docs at [docs/direct_convolution/](../direct_convolution/). That subdirectory is currently read-only on this checkout, so this guide lives under `docs/ck_dsl/` instead. When the permission situation is sorted, move this file to `docs/direct_convolution/ck_dsl_porting_guide.md` and update the link from [include/ck_tile/ops/direct_convolution/README.md](../../include/ck_tile/ops/direct_convolution/README.md).

---

## Part 1 — CK DSL in 10 Minutes

### 1.1 What CK DSL is

CK DSL is a **Python authoring layer for CK Tile kernels** on AMDGPU. It keeps the CK Tile programming model — tile distributions, tile windows, MFMA atoms, LDS staging, software pipelining — but moves the authoring surface from C++ template metaprogramming into Python. The motivation is iteration speed: a CK DSL kernel compiles in ~5–30 ms (plus ~1 ms HSACO emit via `libamd_comgr`) versus minutes for a fully-instantiated CK Tile C++ template.

CK DSL is **not** a string-templating layer that emits HIP C++. The mental model in [dsl_docs/architecture/mental_model.md](../../python/ck_dsl/dsl_docs/architecture/mental_model.md) puts it clearly:

> Do NOT think: "Python generates a HIP string."
> DO think: "Python builds a typed SSA `KernelDef`, then a backend lowers that object to AMDGPU LLVM IR and `libamd_comgr` turns it into HSACO."

### 1.2 The compile pipeline

```
Python authoring (IRBuilder + helpers + instances)
         │
         ▼
   KernelDef (SSA IR — typed Values, Ops, Regions, source-pinned)
         │
         ├─► lower_llvm   ── AMDGPU LLVM IR ── comgr ──► HSACO ──► hipModuleLaunchKernel   (production)
         ├─► lower_hip    ── readable HIP C++                                              (debug)
         └─► lower_cktile ── CK Tile C++                                                   (parity)
```

The IR lives in [core/ir.py](../../python/ck_dsl/core/ir.py) (~2,400 LoC, stdlib-only). Three side-by-side lowerers live in [core/lower_llvm.py](../../python/ck_dsl/core/lower_llvm.py), [core/lower_hip.py](../../python/ck_dsl/core/lower_hip.py), and [core/lower_cktile.py](../../python/ck_dsl/core/lower_cktile.py). Only the LLVM lowerer is on the production path; the other two exist so developers can read the same kernel as HIP C++ or CK Tile C++ when chasing bugs.

The runtime in [runtime/](../../python/ck_dsl/runtime) is a thin ctypes wrapper over `libamd_comgr.so` and `libamdhip64.so` plus tensor-arg packing for PyTorch — no HIP toolchain shell-out, no temp `.cpp` files.

### 1.3 The four authoring layers

| Layer       | Where                                                | What                                                                                                                                       |
| ----------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `core/`     | [core/](../../python/ck_dsl/core)                    | SSA IR (`Type`, `Value`, `Op`, `Region`, `KernelDef`, `IRBuilder`), op vocabulary, lowering backends.                                       |
| `helpers/`  | [helpers/](../../python/ck_dsl/helpers)              | Reusable patterns — tile windows, tile distributions, MFMA atoms, coalesced/async loaders, software pipeline, epilogues, autotune, reduce. |
| `instances/`| [instances/](../../python/ck_dsl/instances)          | ~48 parametric kernel builders: GEMM family, convolution family, attention family, MoE family, small ops, quantization.                    |
| `examples/` | [examples/](../../python/ck_dsl/examples)            | End-to-end bake-offs and parity harnesses (e.g. [bake_off_direct_conv_16c.py](../../python/ck_dsl/examples/bake_off_direct_conv_16c.py)).  |

The op vocabulary is split between a high-level "tile dialect" (matching CK Tile concepts — `tile.smem_load_v4`, `tile.mfma_f32_16x16x16_f16`, `tile.s_waitcnt`, `tile.buffer_load_v4_f16`, `tile.async_buffer_load_lds_addr`) and a low-level arith/scf dialect (`arith.add`, `arith.cvt_fp8_to_f32`, `scf.for`, `scf.if`, `scf.yield`). See [dsl_docs/reference/op_vocabulary.md](../../python/ck_dsl/dsl_docs/reference/op_vocabulary.md).

### 1.4 What writing a kernel looks like

The smallest possible kernel (illustrative, from the helpers README):

```python
from ck_dsl import IRBuilder, I32

b = IRBuilder("my_add")
m = b.param("M", I32)
n = b.param("N", I32)
v = b.add(m, n)         # emits arith.add, returns SSA Value %v0
```

A realistic kernel is constructed via a *spec dataclass* + *builder function* pair in `instances/`:

```python
from ck_dsl.instances.conv_direct_grouped import (
    DirectConvProblem, DirectConv16cSpec, build_direct_conv_16c)
from ck_dsl import compile_kernel

problem = DirectConvProblem(N=32, H=200, W=200, groups=16,
                            cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1)

spec    = DirectConv16cSpec(problem=problem, block_q=16,
                            block_groups=4, fold_k32=True)

kernel    = build_direct_conv_16c(spec)        # returns KernelDef
artifact  = compile_kernel(kernel,
                           isa="amdgcn-amd-amdhsa--gfx950")
# artifact.hsaco       : bytes -> hipModuleLoadData
# artifact.llvm_text   : AMDGPU LLVM IR (for inspection)
# artifact.ir_text     : MLIR-style dump of the SSA KernelDef
```

The artifact is launched through `KernelLauncher` (in [runtime/launcher.py](../../python/ck_dsl/runtime/launcher.py)) with a packed argument dict and a `LaunchConfig(grid, block, stream)`.

### 1.5 Coordinate-transform DAG

A piece worth highlighting because it dominates how convolutions are authored: addressing is expressed through a small algebra of transforms (`unmerge`, `embed`, `pad`, `merge`, `indirect`) in [transforms.py](../../python/ck_dsl/transforms.py). The implicit-GEMM convolution turns input addressing into:

```python
TensorDescriptor.naive("A_nhwc", lengths=[N, Hi, Wi, C]).transform(
    unmerge("m", into=["n", "ho", "wo"], dims=[N, Ho, Wo]),
    embed (["ho", "r"], "hi", strides=[sH, dH], offset=-pH, lo=0, hi=Hi),
    embed (["wo", "s"], "wi", strides=[sW, dW], offset=-pW, lo=0, hi=Wi),
    unmerge("k", into=["r", "s", "c"], dims=[R, S, C]),
    pad   ("r", lo=0, hi=R),
    pad   ("s", lo=0, hi=S),
)
```

At use site `A_desc.offset(b, m=m_val, k=k_val)` returns an SSA `(element_offset, valid_predicate)` pair. This is the direct analogue of CK Tile's `TensorDescriptor` + `make_naive_tensor_view` machinery, but evaluated lazily into IR rather than expanded by the C++ compiler.

### 1.6 Intermediate formats and how they relate to the C++ toolchain

The four-arrow pipeline in §1.2 hides a lot of detail. This section names every intermediate format CK DSL produces and lines each one up against the equivalent stage of the `hipcc` / `rocm-clang` pipeline that a CK Tile C++ kernel goes through. Knowing which artifact is which makes it much easier to diff a CK DSL kernel against its C++ twin when chasing a performance gap.

The two pipelines side-by-side:

```
CK Tile C++ kernel (.hpp templates)              CK DSL kernel (Python builder)
─────────────────────────────────────            ────────────────────────────────
       │                                                  │
       │ rocm-clang front-end                             │ IRBuilder
       │ (preprocess + template instantiate               │ (emits Ops/Values/
       │  + Sema + AST)                                   │  Regions directly)
       ▼                                                  ▼
   Clang AST                                       KernelDef   (SSA IR,
       │                                                       in-memory)
       │ ClangCodeGen                                     │
       │                                                  │ print_ir()
       │                                                  ▼
       │                                            ir_text  (MLIR-style dump —
       │                                                      inspection only,
       │                                                      NO lowerer reads it)
       │                                                  │
       │ -emit-llvm                                       │ lower_llvm
       ▼                                                  ▼
   *.ll / *.bc  ◄── same kind of artifact ──►  artifact.llvm_text
   (AMDGPU LLVM IR)                            (AMDGPU LLVM IR text)
       │                                                  │
       └──────────────────┐                  ┌────────────┘
                          ▼                  ▼
                 LLVM optimizer + AMDGPU backend
                 (InstCombine, GVN, SIScheduler,
                  SIInsertWaitcnts, register alloc, …)
                          │
                          ▼
                 AMDGPU assembly / object  (*.s, *.o)
                          │
                          ▼
                 amd_comgr ELF linker
                          │
                          ▼
                 HSACO  ◄── same kind of artifact ──►  artifact.hsaco
                          │
                          ▼
                 hipModuleLoadData → hipModuleLaunchKernel
```

The key fact: **both paths converge at AMDGPU LLVM IR**. CK DSL's `llvm_text` and the `.ll` file `rocm-clang -S -emit-llvm` produces for an equivalent C++ kernel are the same kind of object and feed the same downstream pipeline (LLVM optimizer → AMDGPU backend → `amd_comgr` ELF linker). They differ only in:

- **Surface form.** CK DSL emits compact, named SSA (`%v3 = call i32 @llvm.amdgcn.workgroup.id.x()`); the C++ front-end emits LLVM IR with mangled symbol names, debug metadata, and a lot of inlining markers. Both reduce to the same machine instructions.
- **What's *not* there.** The C++ LLVM IR carries a long tail of template-instantiated helpers and inlined CK Tile boilerplate. CK DSL's IR contains only what the IRBuilder actually emitted — much shorter, easier to scan.

#### The four formats CK DSL exposes

| Format             | Where                       | Lifetime                | What it's for                                                                                                                                                | C++ analogue                                                                             |
| ------------------ | --------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| **`KernelDef`**    | Returned by `build_*(spec)` | In-memory Python object | The fundamental authoring artifact. All lowerers walk *this*, not any text representation. The thing every helper, optimizer pass, and lowerer consumes.    | Clang AST — also in-memory, also the thing the backend actually walks.                    |
| **`ir_text`**      | `artifact.ir_text`          | String                  | Human-readable MLIR-style dump of `KernelDef`. Useful for `print_ir()` debugging, golden-IR tests, code review. **Inspection-only — no lowerer reads it.**   | `clang -Xclang -ast-dump` — a pretty-printed view that's never re-parsed by the backend.  |
| **`llvm_text`**    | `artifact.llvm_text`        | String                  | The actual AMDGPU LLVM IR that goes into `libamd_comgr`. The thing to diff against the C++ kernel's `.ll`. Each line carries a `; <py-file>:<line>` source pin. | `clang -S -emit-llvm` output (`.ll`) for the matching C++ kernel.                         |
| **`hsaco`**        | `artifact.hsaco`            | `bytes`                 | Ready for `hipModuleLoadData`. Same ELF layout as the HSACO embedded in a `.hipfb` fat binary — `roc-obj-ls`, `llvm-objdump -d --arch=amdgcn`, `rocm-gdb` all work. | The HSACO bundled inside `hipcc`'s `.hipfb` fat binary.                                   |

A fifth artifact, `artifact.timings`, is a dict of per-stage compile times — useful when an autotuner runs hundreds of variants and you want to spot a regression in `lower_llvm` vs. `comgr`.

#### How to look at each stage

| Question                                | C++ tool                                                            | CK DSL tool                                                                |
| --------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| "What did the front-end produce?"       | `clang++ -Xclang -ast-dump`                                          | `print_ir(kernel)` → `ir_text`                                             |
| "What LLVM IR feeds the backend?"       | `hipcc -S -emit-llvm` (or `hipcc --save-temps`)                      | `artifact.llvm_text`                                                       |
| "What machine code did we get?"         | `llvm-objdump -d --arch=amdgcn foo.o`                                | `analyze_hsaco(artifact)` or `llvm-objdump -d --arch=amdgcn` on the HSACO  |
| "How many VGPRs / SGPRs / LDS bytes?"   | `roc-obj-ls -v foo.hipfb` → kernel descriptor                        | `ck_dsl.analysis.summarize(artifact)` (reads the HSA kernel descriptor)    |
| "Where do these instructions come from?"| DWARF debug info (`-g`) → `addr2line`                                | Source-pin comments in `llvm_text` and assembly; every `Op` records `(file, line)`. |

When a performance gap shows up between a CK DSL kernel and its C++ twin, the first useful diff is `llvm_text` vs. the C++ `.ll`. The two will not match line-for-line — different naming, different inlining — but the **instruction counts** (especially `mfma` issue counts, `s_waitcnt` placements, `ds_read`/`ds_write` density) should match. If they don't, the divergence is upstream of LLVM and lives in the CK DSL builder. If they match but the assembly diverges, the divergence is in how each path configures the LLVM/AMDGPU backend (most commonly `waves_per_eu`, `amdgpu_flat_work_group_size`, or `target-cpu` attributes).

#### What CK DSL deliberately skips

Two stages from the C++ pipeline have no analogue in CK DSL — that's the point:

1. **Preprocessing + template instantiation.** This is most of the wall-clock cost of compiling a CK Tile C++ kernel; a single instance can pull in megabytes of header text and instantiate hundreds of class templates before an AST exists. CK DSL has no preprocessor and no templates — Python `for` loops do compile-time unrolling at IRBuilder time, so the equivalent expansion happens in milliseconds.
2. **Clang Sema and AST→LLVM lowering.** CK DSL emits LLVM IR directly. There is no Clang-level type checker, no implicit conversions, no C++ ABI mangling. Type checking happens in the Python builder (`Value.__bool__()` raising `TypeError`, op-result type validation in `IRBuilder`), but it is much narrower than C++ Sema.

This is also why CK DSL ships a `lower_hip` side-path: when you want to read the kernel as C++, run `lower_hip` on the same `KernelDef` and you get debuggable HIP source. That source is **not** on the production path — production goes `KernelDef → lower_llvm → llvm_text → comgr → HSACO`, bypassing C++ entirely.

---

## Part 2 — Relation to FlyDSL

[FlyDSL](https://github.com/ROCm/FlyDSL) ("Flexible Layout DSL") is a separate AMD project. The two share part of the same intent — Python authoring of high-performance AMDGPU kernels with explicit tiling — but they take different routes.

| Dimension                | FlyDSL                                                           | CK DSL                                                              |
| ------------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Frontend**             | Python (`@flyc.kernel`, `@flyc.jit`)                              | Python (`IRBuilder` + spec/builder pairs, helpers, instances)        |
| **IR**                   | MLIR — custom `fly` dialect with first-class **layout algebra**  | Custom in-tree SSA IR (typed Ops/Values/Regions) — no MLIR roundtrip|
| **Lowering stack**       | `fly-layout-lowering` → `convert-fly-to-rocdl` → LLVM/ROCDL → bin| Direct AMDGPU LLVM IR text → `libamd_comgr` → HSACO                  |
| **Layout model**         | Shape/Stride/Layout algebra (compose/product/divide)              | CK Tile model: `TensorDescriptor` + `TileDistributionEncoding`       |
| **Target hardware**      | MI300X/MI308X (gfx942), MI350/MI355X (gfx950)                     | gfx942, gfx950 (LLVM-flavour autodetect for ROCm 7.0/7.1 vs 7.2+)   |
| **Programming model**    | "Cute-like" — tiled MMA, partitioned tensors, preshuffle GEMM    | CK Tile — tile windows, MFMA atoms, pipelines, epilogues             |
| **Examples shipped**     | vector add, tiled copy, tiled MMA, preshuffle GEMM               | GEMM, implicit/direct conv, attention (~10 variants), MoE, quant    |
| **Runtime integration**  | MLIR-managed; project provides its own runners                    | ctypes shim over comgr+HIP; native `torch.compile` backend           |

### 2.1 Where the projects touch each other

CK DSL is *aware of* FlyDSL but is not derived from it. Grep across the tree turns up four loose references:

- [runtime/__init__.py](../../python/ck_dsl/runtime/__init__.py): "launcher: long-lived launch abstractions (CK Tile / FlyDSL / Triton inspired)" — the persistent `KernelLauncher` pattern is shared.
- [helpers/README.md](../../python/ck_dsl/helpers/README.md): notes that CK Tile's `fmha_bwd_launcher`, FlyDSL's `_TorchReduceWrapper`, and CK DSL's helpers share a workspace/launcher design.
- [instances/attention_tiled_2d.py](../../python/ck_dsl/instances/attention_tiled_2d.py): "FlyDSL-inspired subset: exact bf16 QK logits, native fp8 PV".
- [core/ir.py](../../python/ck_dsl/core/ir.py): comments calling out the "FlyDSL pattern" for fp8 quantization in the IR vocabulary.

### 2.2 How to think about the split

- **FlyDSL** is a *general* GPU-kernel DSL whose distinguishing feature is its layout algebra and its commitment to the MLIR lowering ecosystem. It is the right starting point if you want a portable, MLIR-native authoring stack and don't already think in CK Tile vocabulary.
- **CK DSL** is a *CK Tile authoring layer*. It commits to CK Tile primitives, the AMDGPU LLVM IR backend, and direct comgr-based HSACO generation. It is the right starting point if you already have CK Tile kernels (like the direct convolutions in this directory) and want to author equivalents in Python without losing the CK Tile mental model or paying MLIR compile time.

For the direct convolution port, **CK DSL is the natural target**: the existing C++ kernels are explicitly built on CK Tile abstractions ([direct_convolution/kernel/grouped_4c_tile_conv_impl_v3.hpp:1-25](../../include/ck_tile/ops/direct_convolution/kernel/grouped_4c_tile_conv_impl_v3.hpp) shows the includes — `tile_distribution`, `tile_window`, `load_tile`, `store_tile`, `tensor_view`, `buffer_view`), all of which have direct CK DSL helpers.

---

## Part 3 — Direct Convolution in CK Tile / HIP Today

A condensed view of what is in [include/ck_tile/ops/direct_convolution](../../include/ck_tile/ops/direct_convolution) and its pure-HIP twin in [projects/miopen/src/hipconv](../../../miopen/src/hipconv), oriented around what a CK DSL port needs to reproduce.

### 3.1 Supported configurations

From the [direct_convolution README.md](../../include/ck_tile/ops/direct_convolution/README.md):

- **Data types**: fp16, bf16
- **Layouts**: NHWC input, KYXC weights, NHWK output
- **Directions**: Fprop (Forward), Dgrad (Backward Data) — Wgrad not yet supported
- **Filter**: 3×3, stride 1, dilation 1
- **Group sizes** (channels per group): 4, 8, 16, 32
- **Constraint**: `c == k` per group (relaxed by padding variants)
- **Architectures**: gfx942 + gfx950 for 4c/16c; gfx950 only for 8c (Toeplitz fold) and 32c (C-reduction)

### 3.2 Algorithm — direct sliding-window with circular accumulators

Unlike implicit-GEMM, the C++ direct-conv kernels do **not** flatten the problem into an `(M=N·Ho·Wo, N=K, K=R·S·C)` GEMM. They stream input rows top-to-bottom, hold a depth-`KH=3` circular accumulator buffer over output H, and flush a completed output row whenever it has received contributions from all three filter rows. The relevant control logic lives in [direct_convolution/kernel/grouped_conv_compute_loop.hpp:128-223](../../include/ck_tile/ops/direct_convolution/kernel/grouped_conv_compute_loop.hpp).

Tiling — common skeleton across the 4c / 8c / 16c / 32c variants:

```
Workgroup geometry:
  block_q       output columns per workgroup       (default 16)
  block_groups  groups handled per workgroup       (default 4)
  one wave per group (64 threads)
  N is the grid_z dimension

Grid:
  grid_x = ceil_div(out_W, block_q)
  grid_y = ceil_div(groups, block_groups)
  grid_z = N
```

The per-variant difference is the MFMA atom and the per-row inner loop:

| Variant | Per-group MFMA shape (M·N·K) | MFMA atom                       | Notes                                              |
| ------- | ---------------------------- | -------------------------------- | -------------------------------------------------- |
| 4c      | 4·16·4                       | `mfma_f32_4x4x4_f16`             | One wave runs 16 independent 4×4×4 (batch by lane/4)|
| 8c      | 16·16·32                     | `mfma_f32_16x16x32_f16`          | Toeplitz fold — S=0,1 collapsed into K-dimension    |
| 16c     | 16·16·16                     | `mfma_f32_16x16x16_f16`          | 9 MFMAs per 3×3 filter per output row              |
| 32c     | 16·16·32                     | `mfma_f32_16x16x32_f16`          | Explicit C-reduction inner loop                    |

The full file-level layout is documented in the direct-convolution [README.md](../../include/ck_tile/ops/direct_convolution/README.md). The shared infrastructure — `grouped_conv_kernel_base.hpp`, `grouped_conv_input_loader.hpp`, `grouped_conv_weight_loader.hpp`, `grouped_conv_output_writer.hpp`, `grouped_conv_descriptors.hpp` — is what we need CK DSL analogues for.

### 3.3 CK Tile primitives that need a CK DSL analogue

| CK Tile primitive (C++)                  | CK DSL analogue                                                                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `buffer_view`                            | `tile.buffer_rsrc` op + `BufferDesc` wrapper                                                                                              |
| `tensor_view`                            | `make_naive_tensor_view_packed`, `TensorDescriptor` ([transforms.py](../../python/ck_dsl/transforms.py))                                  |
| `tile_distribution` / `tile_distribution_encoding` | `TileDistributionEncoding` ([helpers/distribution.py](../../python/ck_dsl/helpers/distribution.py))                              |
| `tile_window`                            | `make_tile_window` + helpers in [helpers/](../../python/ck_dsl/helpers)                                                                   |
| `load_tile` / `store_tile`               | `CoalescedTileLoader`, `AsyncTileLoader` + `store_*` helpers                                                                              |
| `static_for`                             | Python `for` loops over compile-time ranges (unrolled at IRBuilder time)                                                                  |
| MFMA intrinsics (`__builtin_amdgcn_mfma_*`) | `MfmaAtom` catalog ([helpers/atoms.py](../../python/ck_dsl/helpers/atoms.py)) + `tile.mfma_*` ops                                       |
| `wait_vmcnt<N>()`                        | `b.s_waitcnt(vmcnt=N)`                                                                                                                   |
| Swizzle layout                           | `LdsLayout.padded_k(...)`, XOR/cyclic-shift in consumer-read arithmetic                                                                   |
| `MatrixLayout<M, K, Batch, T>`           | `MfmaAtom.lane_to_output(...)`                                                                                                            |
| `Mfma16x16x32` with C-reduction loop     | Python loop calling the 16×16×32 `MfmaAtom` per C-batch                                                                                   |

All of these already exist in CK DSL — the 16c direct-conv builder ([instances/conv_direct_grouped.py](../../python/ck_dsl/instances/conv_direct_grouped.py)) exercises essentially the whole list.

---

## Part 4 — What's Already in CK DSL

[instances/conv_direct_grouped.py](../../python/ck_dsl/instances/conv_direct_grouped.py) (~827 LoC) already provides two of the four kernel variants and the supporting infrastructure. The contract and algorithm are summarized in [dsl_docs/instances/convolution.md](../../python/ck_dsl/dsl_docs/instances/convolution.md).

### 4.1 Spec types

```python
@dataclass(frozen=True)
class DirectConvProblem:
    N: int
    H: int           # input/output height (assumes equal in/out spatial)
    W: int
    groups: int
    cpg: int         # channels per group
    kpg: int         # filters per group (= cpg in bake-off)
    KH: int = 3
    KW: int = 3
    PAD: int = 1
    stride: int = 1
```

Note this diverges from `ConvProblem` used by implicit-GEMM — here `H`/`W` (not `Hi`/`Wi`), `KH`/`KW` (not `R`/`S`), and a single `PAD`/`stride` int (no per-axis split, no dilation). Match the convention if you add new variants.

### 4.2 16c variant — `DirectConv16cSpec` / `build_direct_conv_16c`

Per-group kernel shape `(M=16, N=block_q=16, K=16)`, MFMA `f32_16x16x16_f16`. The compute loop maintains the depth-3 circular accumulator and folds adjacent S values into a single K=32 MFMA when `fold_k32=True`:

```text
for each input/filter row:
    1. Load needed input row/window into LDS slabs
    2. Apply H/W padding predicates via TensorDescriptor.pad()
    3. Read input + weight fragments
    4. If fold_k32:
         combine S=0 and S=1 into mfma_f32_16x16x32_f16,
         handle S=2 with K=16 mfma_f32_16x16x16_f16
       else use 16x16x16 per S
    5. When an output row is complete:
         buffer_store_dwordx2 per lane (4 halves)
         reset that circular accumulator slot
```

The performance levers documented in CK DSL `RUNBOOK_COMPLIANCE.md`:

- K=32 MFMA fold: ~92 → ~108 TFLOPS
- Wide direct epilogue (1 `buffer_store_dwordx2` / lane = 4 halves): ~108 → ~210 TFLOPS
- `BLOCK_GROUPS=4`: ~210 → ~214 TFLOPS

These figures are validated via [examples/bake_off_direct_conv_16c.py](../../python/ck_dsl/examples/bake_off_direct_conv_16c.py).

### 4.3 4c variant — `DirectConv4cSpec` / `build_direct_conv_4c`

Per-group kernel shape `(M=4, N=block_q=16, K=4)`, MFMA `f32_4x4x4_f16`. One wave packs 16 independent 4×4×4 matmuls — `batch = lane / 4` — so multiple groups are processed per wave. No LDS row pipeline (channels are tiny enough that direct vectorization is cleaner):

```text
1. Pack multiple groups across the 16 wave batches
2. For each output coordinate assigned to the wave:
     - Load input vectors with padding masks
     - Load 4-channel weights
     - Issue mfma_f32_4x4x4_f16
3. Accumulate across KH*KW
4. Vector-store 4 output channels as one buffer_store_vN_f16
```

The vec2-dword epilogue (1 store per lane, 4 halves fused) takes the kernel from ~44 → ~48 TFLOPS.

### 4.4 What is *not* yet in CK DSL

| C++ kernel                                              | Direction | Channels | gfx     | CK DSL status                                                |
| ------------------------------------------------------- | --------- | -------- | ------- | ------------------------------------------------------------ |
| `grouped_4c_tile_conv_impl_v3.hpp` (Fprop)              | Fprop     | 4        | 942/950 | Done — `build_direct_conv_4c`                                |
| `grouped_4c_tile_conv_impl_v3.hpp` (Dgrad)              | Dgrad     | 4        | 942/950 | Not ported                                                   |
| `grouped_8c_tile_conv_impl_v2.hpp`                      | Fprop     | 8        | 950     | Not ported (Toeplitz S-fusion)                               |
| `grouped_8c_tile_conv_impl_v2.hpp` (Dgrad)              | Dgrad     | 8        | 950     | Not ported                                                   |
| `grouped_16c_tile_conv_impl_v2.hpp` (Fprop)             | Fprop     | 16       | 942/950 | Done — `build_direct_conv_16c`                               |
| `grouped_16c_tile_conv_impl_v2.hpp` (Dgrad)             | Dgrad     | 16       | 942/950 | Not ported                                                   |
| `grouped_32c_tile_conv_impl_v2.hpp`                     | Fprop     | 32       | 950     | Not ported (C-reduction inner loop)                          |
| `grouped_32c_tile_conv_impl_v2.hpp` (Dgrad)             | Dgrad     | 32       | 950     | Not ported                                                   |
| `conv_32c_tile_impl_v1.hpp` (non-grouped 32c)           | Fprop     | 32       | 950     | Not ported                                                   |
| bf16 across all variants                                | both      | all      | varies  | Not exposed — spec dataclasses currently fp16-only            |

---

## Part 5 — Porting the Remaining Variants

This section walks through the concrete steps to bring the missing kernels into CK DSL, using the existing 16c port as the template.

### 5.1 Recommended order

1. **bf16 for existing 4c/16c** — smallest change, validates the typing/atom selection path. Generalize `DirectConv{4,16}cSpec` to take a `dtype` field, pick the bf16 MFMA atom in the builder, swap fp16→bf16 in load/store ops.
2. **8c Fprop (gfx950)** — single new MFMA atom (`mfma_f32_16x16x32_f16` with Toeplitz S-fusion). Largely a remix of the 16c builder.
3. **32c Fprop (gfx950)** — same MFMA atom as 8c but with an explicit C-reduction inner loop.
4. **Non-grouped 32c Fprop** — drops the per-group wave assignment.
5. **Dgrad family** — needs the transposed LDS read pattern (CDNA4 `ds_read_tr16_b64`). Author this once as a helper, then reuse across Dgrad/4c/16c/8c/32c.

### 5.2 Step-by-step for a new variant (e.g. 8c Fprop)

#### Step 1 — read the reference

Read [direct_convolution/kernel/grouped_8c_tile_conv_impl_v2.hpp](../../include/ck_tile/ops/direct_convolution/kernel/grouped_8c_tile_conv_impl_v2.hpp) end-to-end, then re-read alongside [direct_convolution/kernel/grouped_conv_compute_loop.hpp](../../include/ck_tile/ops/direct_convolution/kernel/grouped_conv_compute_loop.hpp) to see what is shared vs. what is variant-specific.

Note the Toeplitz pattern: two consecutive S values are interleaved along the K dimension of the MFMA so that one `mfma_f32_16x16x32_f16` does the work of two `mfma_f32_16x16x16_f16`. This is essentially the same `fold_k32` trick used in 16c, but at a different per-S granularity.

#### Step 2 — sketch the spec dataclass

In [instances/conv_direct_grouped.py](../../python/ck_dsl/instances/conv_direct_grouped.py), add:

```python
@dataclass(frozen=True)
class DirectConv8cSpec:
    problem: DirectConvProblem
    name: str = "direct_conv_8c"

    block_q: int = 16
    block_groups: int = 4
    dtype: str = "f16"          # "f16" | "bf16"

    fold_k32: bool = True       # Toeplitz fold of S=0,1
    waves_per_eu: Optional[int] = None
```

#### Step 3 — write the builder

```python
def build_direct_conv_8c(spec: DirectConv8cSpec) -> KernelDef:
    p = spec.problem
    assert p.cpg == 8 and p.kpg == 8, "8c kernel: cpg=kpg=8 required"
    assert spec.dtype in ("f16", "bf16")

    io_ty = io_ir_type(spec.dtype)
    b = IRBuilder(kernel_name_join("direct_conv_8c", spec.dtype,
                                   f"G{p.groups}H{p.H}W{p.W}"))
    b.kernel.attrs["max_workgroup_size"] = spec.block_groups * 64

    # ... param declarations (A, B, D, *_bytes, problem dims) ...
    # ... grid math: gx=ceil(W/block_q), gy=groups/block_groups, gz=N ...
    # ... wave/lane decomposition ...
    # ... tensor descriptors for A (NHWC + pad), B (KRSC grouped), D (NHWK) ...
    # ... circular accumulator setup (depth = KH = 3) ...
    # ... main row loop calling the chosen mfma atom ...
    # ... epilogue: store via buffer_store_vN ...

    return b.kernel
```

Reuse helpers wherever possible. Most of the per-row loop, padding-predicate generation, and epilogue logic in `build_direct_conv_16c` can be lifted into shared helpers and parameterized by `cpg`, `kpg`, and the MFMA atom — that refactor is worth doing before/alongside the second variant rather than after the fourth.

#### Step 4 — register the variant

Add `DirectConv8cSpec` and `build_direct_conv_8c` to the top-level exports in [instances/__init__.py](../../python/ck_dsl/instances/__init__.py).

#### Step 5 — write a bake-off / parity example

Create `examples/bake_off_direct_conv_8c.py` modeled on the existing [examples/bake_off_direct_conv_16c.py](../../python/ck_dsl/examples/bake_off_direct_conv_16c.py):

- Build the kernel.
- Generate inputs with NumPy / PyTorch.
- Compute reference output via the existing CK Tile C++ kernel through a pybind wrapper (or via PyTorch's `conv2d` as a numerical reference).
- Compile via `compile_kernel`, launch via `KernelLauncher`.
- Verify bitwise / within-tolerance equality.
- Benchmark with `time_launches` and compare TFLOPS against the CK Tile reference.

#### Step 6 — capture performance levers in `RUNBOOK_COMPLIANCE.md`

Mirror the lever table from the 16c bake-off so future tuning has a trail.

### 5.3 Special considerations for Dgrad

Dgrad needs the **transposed LDS read** pattern — the CK Tile C++ code uses [direct_convolution/utils/transpose_lds_layout.hpp](../../include/ck_tile/ops/direct_convolution/utils/transpose_lds_layout.hpp) for CDNA4 `ds_read_tr16_b64`. CK DSL does not have a direct equivalent helper today; the port should:

1. Add a `tile.ds_read_tr16_b64` op to the IR vocabulary (if not already present — check [dsl_docs/reference/op_vocabulary.md](../../python/ck_dsl/dsl_docs/reference/op_vocabulary.md)).
2. Lower it in `lower_llvm.py` to the appropriate `llvm.amdgcn.ds.read.tr16.b64` intrinsic for gfx950.
3. Add a `TransposedLdsView` helper that wraps the op with the right LDS layout assumptions.
4. Provide a "Dgrad mixin" — the `SizeView` swap of spatial/channel roles is straightforward; the trickier piece is the inverted padding logic (which the C++ `SizeView<Direction::Dgrad>` already gets right).

Once that helper exists, the per-variant Dgrad builder is a small modification of the Fprop builder: same tiling, same MFMA atom, transposed weight load, inverted-padding output write.

### 5.4 Validation strategy

Each new variant should pass three layers of validation before it goes into `instances/__init__.py`:

1. **Static IR test** — a stub call in [tests/static_ir/](../../python/ck_dsl/tests/static_ir/) that builds the kernel and asserts the IR text matches a golden file. Catches accidental ABI / op-order changes.
2. **Parity test** — bitwise / within-tolerance match against the existing CK Tile C++ kernel for a small set of canonical shapes from the bake-off harness.
3. **Performance gate** — record TFLOPS in `dsl_docs/optimization/measured_results.md`. The CK DSL version should be within ~5% of the CK Tile C++ version once optimized; the bake-off file is the place to document each lever's contribution.

### 5.5 Risks and gotchas

The convolution failure modes listed in [dsl_docs/instances/convolution.md](../../python/ck_dsl/dsl_docs/instances/convolution.md) apply directly. The ones most relevant to this port:

- **Descriptor returns element offsets, buffer ops expect byte offsets.** Shift left by 1 for fp16, 2 for f32 accumulator stores. The 16c builder gets this right; reuse the same helpers.
- **False lanes masked *after* a faulting pointer load.** Use the buffer-rsrc sentinel `INT32_MAX` pattern (`safe = select(valid, off_bytes, INT32_MAX)`) so OOB loads silently return zero.
- **Direct-conv circular accumulator slot not reset after store.** Easy to forget when adding the 8c epilogue — copy the reset from the 16c builder.
- **Toeplitz / K-fold packs S or C channels in wrong order.** Numerically close but not bit-exact. Always parity-check against the C++ reference before claiming a speedup.
- **LLVM flavour skew** — ROCm 7.0/7.1 (LLVM 20) vs 7.2+ (LLVM 22) changed signatures for `make.buffer.rsrc`, fp8/bf8 MFMA operands. CK DSL auto-detects, but a new MFMA atom must be tested under both flavours.

---

## Part 6 — Benchmarking CK DSL Against the C++ Profiler

The natural apples-to-apples comparison for a CK DSL direct-conv port is `ckProfiler` — the same binary used to benchmark the C++ CK Tile kernels. It already understands the direct-conv instances we want to compete with, so the only thing we need to add on the CK DSL side is a wrapper that drives the Python kernel through the *same* problem shapes, datatype, and validation tolerances.

### 6.1 What lives where in the profiler

The three headers the user pointed at are the per-direction "run every instance for this signature" routines. They are pure orchestration — they take a problem shape, iterate over every registered tile instance (implicit-GEMM + direct), time each one, validate against the reference, and report the best.

| Header                                                                                                                                                                                              | Direction               | Entry point                                                                |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------- |
| [profiler/include/profiler/grouped_convolution_forward_tile_algs.hpp](../../profiler/include/profiler/grouped_convolution_forward_tile_algs.hpp)                                                    | Fprop                   | `ckp::run_grouped_conv_forward_tile_algs<SIGNATURE>(...)`                  |
| [profiler/include/profiler/grouped_convolution_backward_data_tile_algs.hpp](../../profiler/include/profiler/grouped_convolution_backward_data_tile_algs.hpp)                                        | Dgrad (backward data)   | `ckp::run_grouped_conv_backward_data_tile_algs<SIGNATURE>(...)`            |
| [profiler/include/profiler/grouped_convolution_backward_weight_tile_algs.hpp](../../profiler/include/profiler/grouped_convolution_backward_weight_tile_algs.hpp)                                    | Wgrad (backward weight) | `ckp::run_grouped_conv_backward_weight_tile_algs<SIGNATURE>(...)`          |

Each `run_*_tile_algs` does the same thing:

1. Allocate a *reference* output and run `ConvBuilder<SIGNATURE, ConvAlgorithm_Reference{}>::Instance` to populate it.
2. Iterate over every registered tile instance for `SIGNATURE`. For direct conv these come from the `get_fwd_direct_instances_nhwgc_*c()` / `get_bwd_data_direct_instances_*` / `get_bwd_weight_direct_instances_*` factory lists, which `forward_tile_algs.hpp:179-201` shows are split per channels-per-group (4c/8c/16c/32c) and per datatype (fp16/bf16).
3. For each instance, call it with `ck_tile::stream_config{time_kernel_=true, flush_cache_=true}`, validate the output against the reference (tolerances from `ck::profiler::get_rtol<DataType>()` / `get_atol<DataType>()`), and compute TFLOPS + GB/s from `conv_param.GetFlops()` / `GetByte<>()`.
4. Return a tuple `(valid, best_avg_time_ms, best_tflops, best_gbs, best_op_name, best_instance_index)`.

The driver `int profile_grouped_conv_fwd_tile(int argc, char* argv[])` (in [profiler/src/profile_grouped_conv_fwd_tile.cpp](../../profiler/src/profile_grouped_conv_fwd_tile.cpp)) parses the CLI, picks the right `SIGNATURE`, and calls into the header above. The matching Dgrad/Wgrad drivers are `profile_grouped_conv_bwd_data_tile.cpp` and `profile_grouped_conv_bwd_weight_tile.cpp` in the same directory. All three register into the `ckProfiler` executable built from [profiler/src/CMakeLists.txt](../../profiler/src/CMakeLists.txt) (target `ckProfiler`).

### 6.2 Running the existing C++ baseline

For a concrete shape — `G=16, N=32, K=16, C=16, Y=X=3, Hi=Wi=200, stride=1, pad=1, dilation=1, NHWGC layout, fp16` — the `ckProfiler` invocation that produces the C++ baseline numbers is:

```bash
# From the build directory after `cmake --build . --target ckProfiler`
./bin/ckProfiler grouped_conv_fwd_tile \
    1            \   # data type: fp16
    1            \   # layout:    NHWGC_GKYXC_NHWGK
    0            \   # index:     32-bit
    1            \   # verify:    yes
    1            \   # init:      integer
    0            \   # log:       no
    1            \   # time:      yes
    2            \   # spatial dims: 2D
    16 32 16 16  \   # G N K C
    3 3          \   # filter Y X
    200 200      \   # input  Hi Wi
    1 1          \   # stride Sy Sx
    1 1          \   # dilation Dy Dx
    1 1          \   # left  pad  LeftPy LeftPx
    1 1              # right pad  RightPy RightPx
```

The argument layout is documented inline in `print_helper_msg()` ([profile_grouped_conv_fwd_tile.cpp:47-83](../../profiler/src/profile_grouped_conv_fwd_tile.cpp)). Useful flags for a comparison run:

| Flag                         | Effect                                                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------------------------- |
| `arg5 = 1` (verify)          | Compare against the reference. Required when you want the printed `[Valid]` line.                       |
| `arg8 = 1` (time)            | Enable kernel timing — without this, `avg_time` is the cold-launch number and not directly comparable.  |
| `--instance <id>`            | Restrict the run to a single instance (0-indexed within the order the algs header iterates them).        |

The per-instance output line is:

```
[Valid] Perf:    0.0723 ms,   214.5 TFlops, 1342.6 GB/s, direct_nhwgc_fp16_16c[ch16,bq16,bg4,fold32] (instance 9)
```

That is the line a CK DSL run must reproduce, ideally with the same `op_name` style so reports diff cleanly.

### 6.3 Running the CK DSL kernel on the same shape

CK DSL ships its own bake-off / parity harness in [examples/bake_off_direct_conv_16c.py](../../python/ck_dsl/examples/bake_off_direct_conv_16c.py), built on `KernelLauncher` + `time_launches`. The minimal driver matching the C++ invocation above is:

```python
# benchmarks/ck_dsl_vs_ckprofiler/run_direct_conv_16c.py
import torch
from ck_dsl import compile_kernel
from ck_dsl.instances.conv_direct_grouped import (
    DirectConvProblem, DirectConv16cSpec, build_direct_conv_16c,
    direct_conv_16c_signature, direct_conv_16c_grid)
from ck_dsl.runtime import KernelLauncher, LaunchConfig, time_launches

problem = DirectConvProblem(N=32, H=200, W=200, groups=16,
                            cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1)

spec      = DirectConv16cSpec(problem=problem, block_q=16,
                              block_groups=4, fold_k32=True)
kernel    = build_direct_conv_16c(spec)
artifact  = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")
launcher  = KernelLauncher(hsaco=artifact.hsaco,
                           kernel_name=artifact.kernel_name,
                           signature=direct_conv_16c_signature(spec))

# Allocate NHWGC inputs and KYXC weights matching the C++ profiler's layout
A = torch.randn(problem.N, problem.H, problem.W,
                problem.groups, problem.cpg,
                dtype=torch.float16, device="cuda")
B = torch.randn(problem.groups, problem.kpg,
                problem.KH, problem.KW, problem.cpg,
                dtype=torch.float16, device="cuda")
D = torch.empty_like(A).reshape(problem.N, problem.H, problem.W,
                                 problem.groups, problem.kpg)

grid    = direct_conv_16c_grid(spec, problem)
block   = (spec.block_groups * 64, 1, 1)
config  = LaunchConfig(grid=grid, block=block, stream=0)

# Correctness — compare against torch.nn.functional.conv2d as reference
import torch.nn.functional as F
launcher({"A": A, "B": B, "D": D,
          "A_bytes": A.numel() * 2,
          "B_bytes": B.numel() * 2,
          "D_bytes": D.numel() * 2}, config=config)

ref = F.conv2d(A.permute(0, 3, 4, 1, 2).reshape(problem.N,
              problem.groups * problem.cpg, problem.H, problem.W),
              B.reshape(problem.groups * problem.kpg, problem.cpg,
                        problem.KH, problem.KW),
              padding=problem.PAD, stride=problem.stride,
              groups=problem.groups)
torch.testing.assert_close(D.permute(0, 3, 4, 1, 2)
                            .reshape_as(ref).float(),
                           ref.float(), rtol=1e-2, atol=1e-2)

# Performance — same TFLOPS formula as the profiler
stats = time_launches(launcher, args={"A": A, "B": B, "D": D, ...},
                      config=config, warmup=10, iters=200)
flops    = 2 * problem.N * problem.groups * problem.kpg * problem.cpg \
           * problem.H * problem.W * problem.KH * problem.KW
tflops   = flops / 1e9 / stats.median_ms
print(f"[CK DSL] {stats.median_ms:.4f} ms  {tflops:.1f} TFlops  "
      f"direct_conv_16c[bg{spec.block_groups},bq{spec.block_q},"
      f"fold{32 if spec.fold_k32 else 16}]")
```

Two things to keep equivalent with the profiler so the numbers diff cleanly:

- **TFLOPS formula.** The profiler uses `conv_param.GetFlops() / 1e9 / avg_time_ms` where `GetFlops()` is `2 * N * G * K * C * Ho * Wo * Y * X` (one MAC = 2 FLOPs). Match that exact formula in the Python driver; do not use a "useful FLOPs" variant.
- **Timing mode.** The profiler runs each instance once as a warm-up before timing (`dummy_run_executed` logic in `forward_tile_algs.hpp:114-121`) and uses `flush_cache_=true`. `time_launches` already does multi-iter timing with warm-up; ensure both have the same warm-up count and a cache flush between iterations if you want strict comparability. Otherwise note the methodology difference in the report.

### 6.4 A side-by-side harness

For a sweep across the canonical direct-conv shapes from `RUNBOOK_COMPLIANCE.md`, the most maintainable setup is a thin Python script that:

1. Reads the shape list from a single YAML/JSON file (`shapes.yaml`).
2. For each shape:
   - shells out to `ckProfiler grouped_conv_fwd_tile ... --instance -1` and parses the best `[Valid] Perf:` line (regex `r"Perf:\s+([\d.]+) ms,\s+([\d.]+) TFlops"`);
   - imports the CK DSL builder, runs `time_launches`, validates against `torch.nn.functional.conv2d`;
   - writes a CSV row `(shape, ck_tile_tflops, ck_tile_op_name, ck_dsl_tflops, ck_dsl_kernel_name, speedup)`.
3. Prints a Markdown summary table and stashes it under `bench_results/direct_conv/<date>-<gfx>.md`.

The output table should look like:

```text
| Shape                             | CK Tile (TFLOPS) | CK DSL (TFLOPS) | Δ      | Notes        |
| --------------------------------- | ----------------:| ---------------:| ------:| ------------ |
| G16N32H200W200_C16K16R3S3_p1      |           214.5  |          212.8  | -0.8%  | within noise |
| G32N32H200W200_C16K16R3S3_p1      |           208.1  |          203.4  | -2.3%  |              |
| G16N8H56W56_C16K16R3S3_p1         |           102.6  |           99.1  | -3.4%  |              |
| ...                               |                  |                 |        |              |
```

`bench_results/` is a sensible home alongside `dsl_docs/optimization/measured_results.md`. The Markdown table is what gets pasted into PR descriptions; the CSV is what feeds longer-running comparison plots.

### 6.5 Per-instance ablation

When investigating a regression, you want to compare against a *specific* C++ instance rather than its best. `--instance <id>` is the lever — the algs header iterates the direct instances in a fixed order (`get_fwd_direct_instances_nhwgc_fp16_4c()`, then `_16c`, then `_8c`, then `_32c`), so the printed `instance N` number is stable across runs of the same `ckProfiler` build.

In practice:

1. Run the full sweep to find the best C++ instance for a shape, e.g. `instance 9 = direct_nhwgc_fp16_16c[ch16,bq16,bg4,fold32]`.
2. Pin to that instance: `ckProfiler grouped_conv_fwd_tile <args...> --instance 9` and re-time five times for a tight median.
3. Build the matching CK DSL `DirectConv16cSpec(block_q=16, block_groups=4, fold_k32=True)` and benchmark with the same warm-up/iter count.
4. If the gap > 5%, dump both kernels' AMDGPU LLVM IR — for CK DSL use `compile_kernel(...).llvm_text`; for CK Tile use the `--save-temps` rocm-clang flag. Diff register usage and MFMA ordering. The most common source of divergence is `waves_per_eu`, which the C++ instance sets via `__attribute__((amdgpu_waves_per_eu(...)))` and CK DSL exposes through the `waves_per_eu` field on the spec dataclass.

### 6.6 Plugging the CK DSL kernel *into* the profiler (optional)

If we want CK DSL kernels to appear alongside C++ instances in `ckProfiler`'s tile sweep (rather than living in a separate Python script), the cleanest hook is to register them through the same `direct_conv_instance_registry.hpp` / `direct_conv_profiler_bridge.hpp` mechanism that the existing direct instances use. This requires:

1. Compiling each CK DSL kernel to HSACO once at profiler startup (via a small ctypes call into `libamd_comgr`, or by pre-building the HSACO offline and embedding it as a binary blob).
2. Wrapping the launcher in the `KernelVariant` interface from [include/ck_tile/ops/direct_convolution/utils/kernel_variant.hpp](../../include/ck_tile/ops/direct_convolution/utils/kernel_variant.hpp) (the same `is_applicable / get_launch_params / launch / get_workspace_size` function-pointer table the C++ kernels use).
3. Adding `get_fwd_dsl_instances_nhwgc_fp16_16c()` and the like, and calling them from the `if constexpr (SIGNATURE == SIGNATURE_NHWGC_FP16_FWD)` branch in `forward_tile_algs.hpp` next to the existing direct factories.

This is a nice-to-have; the shell-out approach in §6.4 is sufficient for first-pass comparison and avoids dragging a Python runtime into `ckProfiler`.

---

## Appendix — Quick References

### CK DSL entry points relevant to this port

- Top-level README: [python/ck_dsl/README.md](../../python/ck_dsl/README.md)
- Mental model: [python/ck_dsl/dsl_docs/architecture/mental_model.md](../../python/ck_dsl/dsl_docs/architecture/mental_model.md)
- Convolution instances doc: [python/ck_dsl/dsl_docs/instances/convolution.md](../../python/ck_dsl/dsl_docs/instances/convolution.md)
- Op vocabulary: [python/ck_dsl/dsl_docs/reference/op_vocabulary.md](../../python/ck_dsl/dsl_docs/reference/op_vocabulary.md)
- MFMA atom catalog: [python/ck_dsl/dsl_docs/reference/mfma_atom_catalog.md](../../python/ck_dsl/dsl_docs/reference/mfma_atom_catalog.md)
- Existing 16c builder: [python/ck_dsl/instances/conv_direct_grouped.py](../../python/ck_dsl/instances/conv_direct_grouped.py)
- 16c bake-off: [python/ck_dsl/examples/bake_off_direct_conv_16c.py](../../python/ck_dsl/examples/bake_off_direct_conv_16c.py)
- IRBuilder + KernelDef: [python/ck_dsl/core/ir.py](../../python/ck_dsl/core/ir.py)
- LLVM lowering: [python/ck_dsl/core/lower_llvm.py](../../python/ck_dsl/core/lower_llvm.py)
- Tile distribution / tile window helpers: [python/ck_dsl/helpers/](../../python/ck_dsl/helpers)
- Coordinate transforms: [python/ck_dsl/transforms.py](../../python/ck_dsl/transforms.py)
- Runtime launcher: [python/ck_dsl/runtime/launcher.py](../../python/ck_dsl/runtime/launcher.py)

### CK Tile C++ references for the port

- Direct convolution README: [include/ck_tile/ops/direct_convolution/README.md](../../include/ck_tile/ops/direct_convolution/README.md)
- Per-kernel details (4c): [docs/direct_convolution/kernel_4c_fp16.md](../direct_convolution/kernel_4c_fp16.md)
- Tile distribution encoding: [docs/direct_convolution/tile_distribution_encoding.md](../direct_convolution/tile_distribution_encoding.md)
- Utility primitives: [docs/direct_convolution/utils.md](../direct_convolution/utils.md)
- Compute loop (circular accumulators): [include/ck_tile/ops/direct_convolution/kernel/grouped_conv_compute_loop.hpp](../../include/ck_tile/ops/direct_convolution/kernel/grouped_conv_compute_loop.hpp)
- Input loader: [include/ck_tile/ops/direct_convolution/kernel/grouped_conv_input_loader.hpp](../../include/ck_tile/ops/direct_convolution/kernel/grouped_conv_input_loader.hpp)
- Output writer: [include/ck_tile/ops/direct_convolution/kernel/grouped_conv_output_writer.hpp](../../include/ck_tile/ops/direct_convolution/kernel/grouped_conv_output_writer.hpp)
- Weight loader: [include/ck_tile/ops/direct_convolution/kernel/grouped_conv_weight_loader.hpp](../../include/ck_tile/ops/direct_convolution/kernel/grouped_conv_weight_loader.hpp)
- Kernel descriptors: [include/ck_tile/ops/direct_convolution/kernel/grouped_conv_descriptors.hpp](../../include/ck_tile/ops/direct_convolution/kernel/grouped_conv_descriptors.hpp)

### CK Profiler (for benchmarking)

- Forward tile algs header: [profiler/include/profiler/grouped_convolution_forward_tile_algs.hpp](../../profiler/include/profiler/grouped_convolution_forward_tile_algs.hpp)
- Backward-data tile algs header: [profiler/include/profiler/grouped_convolution_backward_data_tile_algs.hpp](../../profiler/include/profiler/grouped_convolution_backward_data_tile_algs.hpp)
- Backward-weight tile algs header: [profiler/include/profiler/grouped_convolution_backward_weight_tile_algs.hpp](../../profiler/include/profiler/grouped_convolution_backward_weight_tile_algs.hpp)
- Fprop driver: [profiler/src/profile_grouped_conv_fwd_tile.cpp](../../profiler/src/profile_grouped_conv_fwd_tile.cpp)
- Dgrad driver: [profiler/src/profile_grouped_conv_bwd_data_tile.cpp](../../profiler/src/profile_grouped_conv_bwd_data_tile.cpp)
- Wgrad driver: [profiler/src/profile_grouped_conv_bwd_weight_tile.cpp](../../profiler/src/profile_grouped_conv_bwd_weight_tile.cpp)
- Direct-instance registry (used by the tile algs headers): [profiler/include/profiler/direct_conv_instance_registry.hpp](../../profiler/include/profiler/direct_conv_instance_registry.hpp)
- `KernelVariant` ABI used by direct instances: [include/ck_tile/ops/direct_convolution/utils/kernel_variant.hpp](../../include/ck_tile/ops/direct_convolution/utils/kernel_variant.hpp)
- `ckProfiler` executable: built from [profiler/src/CMakeLists.txt](../../profiler/src/CMakeLists.txt) (`add_executable(ckProfiler ...)`)

### FlyDSL

- Upstream: <https://github.com/ROCm/FlyDSL>
