# RFC 0023: Runtime Kernel Compilation (JIT) Requirements

- Contributors: Brian Harrison
- **Status**: Planned
- **Status notes**: Defines requirements and evaluation criteria, including kernel performance and the variant space a compiler must expose, and records first compile-time measurements. No compiler is selected and no API is specified. The hipDNN-side work these requirements depend on is planned separately. RFC 0017 §9.3 defers JIT to a separate RFC; this is its first stage.
- **Last verified**: 2026-10-30 against `develop` (cd0d5ae038a)
- Audience: kernel authors, hipDNN project leads, ROCm component owners

## Summary

hipDNN needs a runtime compiler: a native library, shipped in TheRock, that compiles a
kernel while hipDNN builds an execution plan. It has three uses:

1. **Specialize** an authored kernel for the exact problem (shape, dtype, architecture)
   instead of shipping every variant prebuilt. The variants it can produce still form a
   bounded space that hipDNN can rank.
2. **Generate** fused kernels from the operation graph, starting with pointwise operations
   before or after a convolution.
3. **Precompile** plans ahead of time for chosen GPU targets, including on a machine with no
   GPU, and serialize them for reuse.

This RFC states what that compiler must provide and how each requirement is checked, then
records where the options reviewed so far stand, including first compile-time measurements.
A candidate must compile quickly, produce kernels that reach parity with hipDNN's best
existing engines, and expose a variant space that hipDNN can rank.
This RFC does not choose a compiler. hipDNN project leads make that choice with the owners
of the selected component, after the remaining measurements and the fusion proof of concept
in [Next steps](#next-steps).

Reviewers are asked to:

- correct facts about their component;
- challenge any requirement that would exclude a workable option;
- name options missing from the list.

## Background

**Prebuilt kernels scale by count.** hipDNN's kernel providers ship most kernels compiled
ahead of time and packed per GPU architecture. Covering more problems means shipping more
variants, and every value a variant bakes in must be checked when the kernel is matched to
a problem (RFC 0017 §5). A runtime compiler moves this cost but does not remove it: the
variants it can produce must still form a bounded space that hipDNN can rank (see
[Variant space and selection](#variant-space-and-selection)).

**Fused graphs need generated kernels.** hipDNN graphs can express `conv → pointwise` and
`pointwise → conv` chains. Fused execution today covers fixed patterns: MIOpen's
conv-bias-activation fusion, hipBLASLt matmul epilogues and authored hip-kernel-provider
(HKP) kernels. A general pointwise chain cannot be enumerated in advance. cuDNN covers this
case with runtime fusion engines.

**hipDNN already compiles at runtime in places.** There is no shared compiler contract and
no fusion representation.

| Path today | When code is compiled | Cache of compiled code |
|---|---|---|
| HKP embedded HIP source (hipRTC) | Plan build | HKP: in-process. amd-comgr may also cache on disk. |
| HKP prebuilt kernels (kpack archives; rocKE kernels when enabled) | Library build | In-process module cache |
| MIOpen fusion plans | Applicability check and plan build | MIOpen binary cache on disk |
| hipBLASLt epilogues | hipBLASLt library build | Not applicable |

Under RFC 0009 a provider serializes its own plans. Today only HKP's kernel ingestor does,
and only when every kernel comes from a kpack archive, so a plan with a hipRTC-compiled
kernel cannot be serialized. hipDNN cannot yet build a plan for a GPU that is not present.

**hipDNN sits low in the ROCm stack.** TheRock ships hipDNN in the `rocm-sdk-libraries`
package with the other math libraries (MIOpen, hipBLASLt). hipRTC and amd-comgr ship one
layer below, in `rocm-sdk-core`. PyTorch, Triton and AOTriton are built on top of these
packages in TheRock's separate framework builds. A math library at this layer must be
self-contained and light on dependencies: it cannot depend on Python packages or on anything
built above it. hipDNN runs rocKE, whose kernels are authored in Python, only at build time
for this reason. This constraint shapes the choice of runtime compiler.

## Terms

| Term | Meaning |
|---|---|
| Plan build | The step in which hipDNN turns a selected engine configuration into an executable plan. |
| Applicability check | The query in which a provider reports whether it supports a graph. |
| Code object | The compiled GPU binary that HIP loads and launches. |
| amd-comgr | ROCm's code-object library. HIP and hipRTC use it to compile and link. |
| kpack | Per-architecture archive of prebuilt code objects that hipDNN loads at runtime. |
| Generic target | One ISA target, such as `gfx11-generic`, that runs on every GPU in a family. |
| Prologue / epilogue | Operations fused into a kernel at its input load or its output store. |
| Cold compile | No cached code object exists; the compiler is already loaded in the process. |
| One-time cost | Loading and initializing the compiler in a new process. |
| `rocm-sdk-core`, `rocm-sdk-libraries` | TheRock's Python wheel packages for the ROCm runtime and the math libraries. Frameworks such as PyTorch install on top of them. |
| LDS | Local data share: on-chip memory shared by a workgroup. |
| GEMM / GEMV | General matrix-matrix / matrix-vector multiply. |
| Implicit GEMM | A convolution computed as a GEMM without materializing the unfolded input. |
| Matrix core | GPU units that execute matrix-multiply (MFMA) instructions. |
| IR, MLIR | Intermediate representation; MLIR is LLVM's multi-level IR framework. |
| Tracing | Running Python kernel code once to record it as IR. |
| Kernel ingestor | The HKP component that loads kernels described by RFC 0017 descriptors. |
| hipGraph | HIP's mechanism for capturing a sequence of GPU work once and replaying it. |
| Deviceless compile | Compiling for a named GPU target on a machine that has no GPU. |
| Best existing engine | The engine and knob setting that hipDNN autotuning (RFC 0013) selects for a graph, across every installed provider's engines and their knob variants, on the same device and hipDNN build. It may be prebuilt, compiled at runtime or fused. |
| Unfused execution | The best existing engine for each operation of a fused graph, run in sequence, with each intermediate tensor stored in the data type the graph declares for it. |
| Variant | One compilable configuration of a kernel: fixed values for its tuning parameters, such as tile sizes. |
| Variant space | The variants hipDNN may request for one operation input and target: an enumerated catalog, or a bounded generator that produces them. |
| Best-of-autotune | The fastest variant found by autotuning over the variant space. |
| Per-shape ratio | Run time of the JIT kernel divided by run time of the baseline, for one problem shape. The geomean summarizes the ratios over a set of shapes; the worst shape is the one with the largest ratio. |
| Parity margin, worst-shape cap | The largest geomean, and the largest single per-shape ratio, above 1 that still count as parity. Set per operation family (D10). |
| Tuning budget | The bound on how many candidates, or how much compile time, autotuning spends on JIT variants for one graph. |
| Heuristic | A model that ranks candidate engines or kernels without running them (RFC 0019). |

## Requirements

**Must** blocks adoption. **Must (POC)** must be shown by a proof of concept before
selection. **Should** distinguishes between options that meet every Must.

### Packaging and platform

| ID | Level | Requirement | Checked by |
|---|---|---|---|
| P1 | Must | The compiler, and everything it needs, is available to a math library in `rocm-sdk-libraries`: it ships at that layer or below (for example in `rocm-sdk-core`), follows TheRock's generic wheel packaging, and adds no install or release dependency that this layer cannot satisfy. No Python interpreter, Python package or framework built above the SDK is required to compile or launch. | Install only the `rocm` package with its `libraries` and `device-<target>` extras, with no other Python packages; build and run a JIT plan. |
| P2 | Must | Built and shipped by TheRock, with declared build and runtime dependencies, and tested from the installed tree. | TheRock build and installed-tree tests. |
| P3 | Must | Follows TheRock packaging: target-neutral host libraries; any per-architecture content in the per-architecture packages. | Two-architecture and one-architecture installs both work. |
| P4 | Must | Supports every architecture TheRock releases for, including `gfx11-generic` and `gfx12-generic`. | Compile and run on each architecture family. |
| P5 | Must | All third-party dependencies come from TheRock's pinned third-party sources, with licenses cleared for ROCm. Builds need no network access once inputs are supplied. | Offline TheRock build; dependency and license list. |
| P6 | Must | Optional at build time. A build without the JIT keeps every prebuilt kernel working. | Build with the JIT disabled; existing tests pass. |

### API and stability

| ID | Level | Requirement | Checked by |
|---|---|---|---|
| A1 | Must | Follows ROCm versioning: no breaking API or ABI change outside a ROCm major release. The same applies to the hipDNN API that exposes the JIT (RFC 0005). | Older same-major client runs against a newer library. |
| A2 | Must | A stable, documented interface that returns a code object and what is needed to launch it: entry symbol, argument layout, resource usage (registers, LDS, scratch) and launch constraints such as workgroup size. The interface language and input level are open (D2). | Interface review; hipDNN uses only this interface. |
| A3 | Must | Failures return an error with a diagnostic. No silent fallback to another compiler, path or architecture. | Fault-injection tests. |
| A4 | Must | Safe to call concurrently from parallel plan builds. | Parallel plan-build stress test. |
| A5 | Must | Deterministic: identical inputs give identical code objects, and the compiler reports a version usable in a cache key. | Repeat-compile comparison. |
| A6 | Must | A named owning team and a release cadence aligned with ROCm. | Recorded support agreement. |
| A7 | Should | After compiling, hipDNN can confirm from the code object alone, without loading it or using a GPU, that the kernel is valid for the plan's target. See [Kernel validity](#kernel-validity). | A kernel built for the wrong target or over the LDS limit is rejected at plan build on a CPU-only machine. |

#### Kernel validity

A7 uses values that AMDGPU code objects already record in their metadata. A kernel is valid
for a plan when every check below passes:

| Check | Code-object metadata field |
|---|---|
| Built for the plan's target | `amdhsa.target` |
| Entry symbol is present | `.name`, `.symbol` |
| Static LDS plus any dynamic LDS the plan requests fits the target's limit | `.group_segment_fixed_size` |
| The plan's workgroup size is within the kernel's maximum, and the wavefront size matches | `.max_flat_workgroup_size`, `.wavefront_size` |
| The arguments hipDNN passes match the kernel's argument size | `.kernarg_segment_size` |
| Scratch memory per work-item is within the plan's budget | `.private_segment_fixed_size`, `.uses_dynamic_stack` |

These are readable on the CPU with amd-comgr (`amd_comgr_get_data_metadata`). A code object
compiled by hipRTC for gfx950, without using a GPU, carries all of them, so the requirement is
realistic for any compiler that emits standard code objects. The same metadata also records
register counts and spills; those affect performance, not validity.

### Compile time

Plan build sits on a user's first-call path, so compile time is visible to users. L1 and L2
bound that cost.

| ID | Level | Requirement | Checked by |
|---|---|---|---|
| L1 | Must | Cold compile under one second per kernel for the reference set, on each reference architecture. | Latency harness. |
| L2 | Should | Tens of milliseconds cold for GEMM, convolution and pointwise kernels. | Latency harness. |
| L3 | Must | One-time cost is paid once per process and reported separately. | Harness reports first and later compiles. |

Reference set: one GEMM, one implicit-GEMM convolution, one pointwise kernel, one dense
attention forward, and the two fusion kernels (F3). Each in fp16 and bf16, on gfx942, gfx950
and one RDNA architecture.

### Kernel performance

A compiler that compiles quickly but produces slower kernels than hipDNN already runs is not
a candidate. The baseline is hipDNN's best existing engine for the same problem, not the
compiler's own untuned or unfused output.

Fusion saves at most the intermediate tensor's round trip through memory and, outside
hipGraph replay, one kernel launch. What that is worth depends on the problem: it is large
for short, memory-bound kernels and small for long, compute-bound convolutions and GEMMs. A
fused kernel whose main loop is slower than the best existing engine's can lose the whole
saving on large problems, so fused performance depends on the main kernel reaching parity
first. For this reason the RFC sets no required fusion gain, and judges a fused kernel
against the unfused execution.

| ID | Level | Requirement | Checked by |
|---|---|---|---|
| E1 | Must (POC) | A specialized kernel performs within the parity margin of the best existing engine for the same problem. | The geomean of per-shape ratios is within the margin, and the worst shape within the cap (D10). |
| E2 | Must (POC) | A fused kernel reaches parity with the unfused execution of the same graph, and is within the parity margin of any existing fused engine for the pattern. | Against the unfused execution, the geomean of per-shape ratios does not exceed parity and the worst shape is within the cap (D10). Against an existing fused engine, the E1 check. |
| E3 | Must (POC) | E1 and E2 are judged on the best-of-autotune kernel over the variant space (V1), and every variant compiled to find it meets the per-kernel bound of L1. | The harness records the compile time of every variant compiled during the search, and the compile and run time of the selected code object. |

Performance is measured on real model shapes that the baseline serves, in fp16 and bf16, on
the L1 reference architectures. Both arms run on the same node in one session, interleaved,
with a repeated baseline run to bound measurement noise. Results report the geomean, the
worst shape and the coverage: how many of the requested shapes were measured.

These requirements weigh on options differently. A compiler with a fixed cost on every
compile can try fewer variants within a tuning budget, so its best-of-autotune comes from a
smaller search. A compiler that already builds kernels in the baseline can meet E1 by
producing the same code, so E1 says less about it than about a new code generator.

### Variant space and selection

A runtime compiler can produce far more variants than hipDNN can select from. hipDNN picks a
kernel by ranking candidates with a heuristic and, when the user asks, by autotuning (RFC
0013). Every variant in the space adds cost to both, and to building the heuristic. More
variants do not guarantee better kernels: in one hipDNN attention study on gfx942, a small
curated set came close to the best of a larger one, while dropping the wrong tuning
parameter cost heavily. The compiler must therefore work within a space that hipDNN can
bound and rank.

| ID | Level | Requirement | Checked by |
|---|---|---|---|
| V1 | Must | For each supported operation input and target, the variants hipDNN may request form a bounded variant space, enumerated or produced by a bounded generator (RFC 0019 §4.3, §14.4). The compiler compiles any variant in it. | Each reference-set input maps to a bounded set of candidates, and the set's size is reported. |
| V2 | Must | Each variant is identified by parameters known before it is compiled, so a heuristic can score it without compiling. RFC 0017 §9.3 describes one form for this metadata. | Candidates are ranked with no compilation (I1). |

hipDNN also needs a heuristic that quickly picks a high-quality candidate from that space,
and a bound on how many candidates autotuning compiles. Both are hipDNN work, planned
separately and outside this RFC. The selection mechanism and its latency budget are open in
RFC 0019 (§4.3, §9.4 and Open Question 8). JIT kernels use the same selection path as
prebuilt kernels (I2).

### hipDNN integration

| ID | Level | Requirement | Checked by |
|---|---|---|---|
| I1 | Must | The JIT compiler runs only during plan build, never during applicability checks, heuristics or execution (RFC 0013). Existing provider compile paths are outside this RFC. | Instrumented test. |
| I2 | Must | Compiled kernels are matched, autotuned, selected and launched like prebuilt kernels, with no JIT-specific launch or selection path. | Code review; existing tests. |
| I3 | Must | Specialized and generated kernels are matched as strictly as prebuilt ones: dtype, layout, architecture and every baked value (RFC 0017 §5). | Mutation tests. |
| I4 | Must | hipDNN provides one persistent cache for compiled code objects, shared by every compiler it uses, in a location the user controls (today `HIPDNN_CACHE_DIR`). The compiler does not need its own cache; it must provide deterministic output and a version for the cache key (A5). | A second process reuses the cached code object; a corrupted entry is recompiled. |

### Targets and deployment

hipDNN must be able to build plans ahead of time for chosen targets, serialize them, and
reuse them, as well as build them for the GPU in use. These are hipDNN capabilities: this
RFC requires only that the compiler makes them possible. The hipDNN-side implementation of
T1–T4 is planned separately and is not part of this RFC.

| ID | Level | Requirement | Checked by |
|---|---|---|---|
| T1 | Must | Compiles for any supported target named by hipDNN, not only the GPU present. | Compile for every P4 target from one machine. |
| T2 | Must | Compiles on a CPU-only machine with no GPU or GPU driver. | Compile for a P4 target on a machine without a GPU. |
| T3 | Must | A compiled kernel can be stored and restored without the compiler or any compiler runtime state, so that plans containing JIT kernels can be serialized, including plans built for another target (RFC 0009). | A code object and its launch details (A2), saved on a CPU-only machine, load and launch on the target GPU without the compiler. |
| T4 | Must | JIT kernels can be captured into a hipGraph and replayed, like prebuilt kernels. | The kernel launches through `hipModuleLaunchKernel` with no compiler-owned launcher, and replays under hipGraph capture. |

These follow directly when the compiler returns a standard code object with its launch
details (A2): hipDNN stores the bytes and launches them with `hipModuleLaunchKernel`, which
works under hipGraph capture. If a runtime, loader or launcher layer sits between hipDNN and
the compiled kernel, that layer must itself meet T1–T4.

### Fusion

The compiler does not need to fuse at adoption. It must show that it can, through a proof
of concept, before it is selected.

| ID | Level | Requirement | Checked by |
|---|---|---|---|
| F1 | Must (POC) | Pointwise operations attach as a prologue on an input or an epilogue on the output of a convolution or GEMM, without writing a new kernel per chain. | Both POC kernels come from one convolution template. |
| F2 | Must (POC) | A prologue applies before padding, so padded positions stay zero rather than `f(0)`. | Test with padding and an operation where `f(0) ≠ 0`. |
| F3 | Must (POC) | Two reference patterns, fp16 and bf16, NHWC (hipDNN's primary convolution layout): `conv → MUL` with a per-channel scale, and `MUL → conv` with a per-channel scale on the input. | Matches the CPU reference; meets L1, E2 and E3. |
| F4 | Should | Intermediate values keep the graph's compute precision between fused operations. | Comparison with an fp32 reference. |

hipDNN normalizes the graph into a fusion request before calling the compiler. The compiler
does not interpret hipDNN's graph schema.

## Options reviewed so far

The list is not exhaustive, and no option is ruled out. Each entry covers what the option
is, how hipDNN uses it today, whether it is available at hipDNN's layer (P1), measured
compile time, fusion support and the work needed to meet the requirements. This RFC has not
yet measured any option against the kernel performance (E1–E3) or variant space (V1–V2)
requirements; results published by component owners use other baselines and are not
comparable.

| Option | Native compile path | Available at the `rocm-sdk-libraries` layer (P1) | Fusion today | Main work to meet the requirements |
|---|---|---|---|---|
| hipRTC | Yes (C API) | Yes, in `rocm-sdk-core` | None in hipDNN; CK tests generate fused source | Compile time (about 0.4 s floor per kernel), fusion templates |
| rocKE | Partial: native to LLVM IR; code-object step in Python | No: build-time tool; native library not in the runtime packages | GEMM epilogues; convolution output operations | Shipped native library, compile API, remaining native builders, adapter |
| Loom | Yes (C API, own code generator) | No: not in TheRock | Matmul kernels with fused epilogues | ROCm component status, DNN kernels, generated-kernel performance data |
| FlyDSL | No (Python front end) | No: Python wheel that imports PyTorch | Authored GEMM epilogues | Native front end, stable native API, packaging |
| Triton | No (Python front end) | No: built with PyTorch, above the SDK | Generated by PyTorch Inductor | Native compiler library, fusion generator, generic targets |

All five options produce standard AMDGPU code objects, and all five compiled for gfx942 and
gfx950 without using a GPU (T1, T2). A code object with its launch details can be serialized
and launched under hipGraph capture like a prebuilt kernel (T3, T4). Two options add a layer
that hipDNN would need to handle: Loom emits a separate launch-configuration artifact that
must be saved with the code object or replaced, and FlyDSL and Triton launch through Python
launchers today, which a native path would replace with `hipModuleLaunchKernel`.

### Measured compile times

The process is set up first: libraries loaded, compiler initialized and, where the option
needs it, Python modules imported. Each cell then times one compile of one kernel, median in
milliseconds, with every compiler cache disabled (L1). Ranges span gfx942 and gfx950. The
last two columns give the setup costs excluded from the compile times.

| Option | Pointwise | GEMM | Convolution | Attention fwd, D128 | Initialization | Python import |
|---|---|---|---|---|---|---|
| hipRTC | 374–405 (trivial kernels); 416–446 (RMSNorm) | 1,091–1,110 (tiled); 9,118–9,158 (CK template) | 398–430 (direct) | Not measured | 20–30 | None |
| rocKE | 18–20 | 62–81 | 78–87 (implicit GEMM) | 218 (gfx942 tiled); 1,430 (gfx950 dense) | About 20 | 155–274 (Python kernel builders) |
| Loom | 0.7–0.8 | 8.5–8.8 (bf16 GEMV) | No example kernel | Not measured (needs optional C++ importer) | About 0.4 | None |
| FlyDSL 0.3.4 | 45 | 440–518 | 1,692 (gfx950 only, bf16) | 1,280–1,686; 5,184 (gfx950 dual-wave) | About 35 | About 1,000 (includes PyTorch) |
| Triton 3.8.0 | 45–52 | 229–346 | 488–729 (implicit GEMM) | 733–740 | About 390 | About 105 |

Measured on a development machine for framing purposes: compile only, no GPU used, fp16
unless noted.

- Each option compiled different kernels, so rows compare order of magnitude only. Compile
  time does not show how fast the compiled kernel runs; that is judged by E1–E3.
- The hipRTC tiled GEMM and the Triton convolution were written for this measurement, and the
  hipRTC convolution is HKP's direct (non-GEMM) kernel; none is a production kernel.
- Loom generates code without LLVM. Its GEMM entry is a GEMV, not a tiled matrix-core GEMM.
- rocKE kernels were built by its Python builders and lowered by its native library; the
  compile times include kernel construction. A native rocKE path would not import Python.
- FlyDSL and Triton compile times include their Python front ends (tracing or IR building).

### Assessment

**hipRTC.** hipRTC compiles HIP C++ source at runtime. HKP uses it at plan build for its
hip_mlops kernels (batch, layer and RMS normalization, and resampling) and for
kernel-ingestor embedded-source kernels, targeting the GPU present. It ships in
`rocm-sdk-core`, meets P1 and has a versioned C API. Every compile pays a front-end cost of
about 350–380 ms, even for a trivial kernel, so cold compiles start near 0.4 s; a tiled GEMM
takes about 1.1 s and a Composable Kernel template GEMM about 9 s. hipDNN has no fused
hipRTC kernels today; fusion would come from convolution and GEMM source templates with
pointwise insertion points, as Composable Kernel's hipRTC code-generation tests do. Work
needed: lower per-compile cost to meet L1 for GEMM-class kernels, confirmed generic-target
output, and the fusion templates.

**rocKE.** rocKE is a kernel-authoring library inside HKP; kernels are authored in Python.
With `HIPKERNELPROVIDER_ENABLE_ROCKE` on (off by default, including in TheRock), hipDNN
compiles rocKE kernels at library build and ships them in kpack archives. rocKE is not in
the runtime packages, so it does not meet P1 today. Its native C++ library, with a C
interface, builds kernel IR and lowers it to LLVM IR without Python, with native builders
for GEMM, implicit-GEMM convolution and elementwise kernels; amd-comgr, called from Python
today, produces the code object. Measured cold compiles are 18–87 ms for pointwise, GEMM and
convolution, and 0.2–1.4 s for attention. Fusion covers GEMM epilogue chains and convolution
output scaling, bias and activation; rocKE does not support input-side (prologue) fusion.
Work needed: a native library shipped at the `rocm-sdk-libraries` layer with a stable
compile-to-code-object API, native builders for the remaining families (gfx942 tiled and MLA
attention, dense attention on gfx942 and gfx950) and for dispatch selection, and a
kernel-ingestor source adapter (RFC 0017).

**Loom.** Loom (`loomc`, in ROCm/hrx-system) is a native compiler with a C API and its own
AMDGPU code generator. It compiles source held in memory for a GPU profile read from the
device and emits a code object plus a separate launch-configuration artifact; its target
list includes gfx942, gfx950 and the gfx11 and gfx12 generic targets. hipDNN does not use
Loom today. It is not part of TheRock: the C library is version 0.1.0 with a versioned
shared library, and the project describes itself as early-access and not an official
component of the ROCm stack. Measured compiles are under 10 ms for its elementwise and GEMV
examples. Its examples include matmul kernels with fused residual and SwiGLU epilogues.
Work needed: a stability and support agreement, a TheRock component with cleared dependency
licenses, convolution and tiled matrix-core GEMM kernels (its attention example needs the
optional C++ importer), and performance data for the code it generates.

**FlyDSL.** FlyDSL (ROCm/FlyDSL) is a Python kernel language built on MLIR: Python traces
the kernel into MLIR, and C++ passes with FlyDSL's bundled LLVM lower it to a code object.
hipDNN does not use FlyDSL today. FlyDSL ships as a Python wheel that imports PyTorch, so a
math library in `rocm-sdk-libraries` cannot depend on it at runtime. Measured cold compiles
are 45 ms for pointwise, 0.4–0.5 s for GEMM, 1.7 s for convolution (gfx950 only) and
1.3–5.2 s for attention; Python tracing is the largest
stage, and the majority for convolution and attention. Fusion today is limited to epilogues
authored into its GEMM kernels. A runtime path needs a native front end. One route is to
capture kernel templates at build time and specialize them in C++ at plan build; FlyDSL does
not define a stable saved format or replay API for this, and tile sizes and dtypes are fixed
when a kernel is traced. Work needed: the native front end, a stable native API, and a native
library with its LLVM shipped at the `rocm-sdk-libraries` layer.

**Triton.** Triton compiles Python-authored kernels: Python builds Triton IR, and C++ MLIR
passes with Triton's bundled LLVM lower it to a code object. hipDNN does not use Triton
today; AOTriton compiles with Triton at build time and ships a C++ runtime that loads the
prebuilt kernels. TheRock builds Triton together with PyTorch, on top of the ROCm SDK
packages, so a math library in `rocm-sdk-libraries` cannot depend on it. Measured cold
compiles are 45–52 ms for pointwise, 0.2–0.35 s for GEMM and 0.5–0.75 s for convolution and
attention. PyTorch Inductor generates fused Triton kernels from graph nodes, so generated
pointwise fusion with Triton is in production use through PyTorch. Upstream Triton does not
emit the gfx11 and gfx12 generic targets. As with FlyDSL, a runtime path needs a native
compiler library without Python, shipped at the `rocm-sdk-libraries` layer and fed with
Triton IR captured at build time (that IR is not versioned across Triton releases), plus a
hipDNN fusion generator and a solution for generic targets.

## Decisions

| ID | Decision | Status |
|---|---|---|
| D1 | amd-comgr may be on the runtime path. It ships in `rocm-sdk-core`, and HIP and hipRTC already load it. | Decided |
| D2 | The compiler contract: its interface language (for example C or C++), and its input level (HIP source, LLVM IR, a kernel IR, or a kernel specification). | Open |
| D3 | Ownership: a separate ROCm component, or a library inside a hipDNN provider. | Open |
| D4 | The fusion request format, and whether it extends the RFC 0017 matching and dispatch descriptors. | Open |
| D5 | The compile-time budget applies per kernel (L1). | Decided |
| D6 | hipDNN owns caching of compiled code objects through one shared mechanism (I4); a compiler's built-in cache is not required. | Decided |
| D7 | Kernel performance is judged on the best-of-autotune kernel: against hipDNN's best existing engine (E1), and for fused kernels against the unfused execution and any existing fused engine (E2). | Decided |
| D8 | The compiler works within a bounded variant space whose variants can be ranked before compiling (V1, V2). | Decided |
| D9 | The selection mechanism, its latency budget, and the tuning budget. Tracked with RFC 0019 (§4.3, §9.4, Open Question 8) and RFC 0013. | Open |
| D10 | The parity margin and the worst-shape cap for each operation family. hipDNN project leads set them before the performance study, and they are recorded in this RFC. | Open |

## Next steps

1. Extend the compile-time measurements to the full reference set: fp16 and bf16, one RDNA
   architecture, and the fusion kernels.
2. Define the variant space (V1) for each reference-set input, record the parity margins and
   worst-shape caps (D10) in this RFC, then measure kernel performance as described in
   [Kernel performance](#kernel-performance).
3. Build the fusion proof of concept on the option the measurements support. It must meet
   F1–F3.
4. Build that option inside TheRock and check P1–P6 and the compiler-side checks of T1–T4 on
   the installed tree.
5. Update this RFC with the results and close D2–D4.

hipDNN-side support for T1–T4 (precompiling and serializing plans for another target,
hipGraph capture), the selection heuristic and the tuning budget (D9) are planned separately.

## References

- hipDNN RFCs: [0005 Versioning](0005_Versioning.md),
  [0009 Execution plan serialization](0009_CompiledPlanSerialization.md),
  [0013 Autotuning](0013_Autotune.md),
  [0017 Universal kernel descriptors](0017_UniversalKernelDescriptor.md),
  [0019 Universal heuristic descriptors](0019_UniversalHeuristicDescriptor.md)
- [hip-kernel-provider compilation](../../../../dnn-providers/hip-kernel-provider/src/compilation/):
  hipRTC path and module caches
- [rocKE documentation](../../../../dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/):
  pipeline, recipes, measured results and fusion
- [MIOpen kernel cache](../../../../projects/miopen/docs/conceptual/cache.rst)
- [Composable Kernel hipRTC code generation tests](../../../../projects/composablekernel/codegen/test/rtc/)
- [HIP RTC guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html)
- [AMDGPU code object metadata](https://llvm.org/docs/AMDGPUUsage.html#code-object-v5-metadata)
- TheRock: [Python packaging](https://github.com/ROCm/TheRock/blob/main/docs/packaging/python_packaging.md),
  [Dependencies](https://github.com/ROCm/TheRock/blob/main/docs/development/dependencies.md),
  [Versioning](https://github.com/ROCm/TheRock/blob/main/docs/packaging/versioning.md),
  [rocm_sdk package map](https://github.com/ROCm/TheRock/blob/main/build_tools/packaging/python/templates/rocm/src/rocm_sdk/_dist_info.py),
  [PyTorch framework build](https://github.com/ROCm/TheRock/tree/main/external-builds/pytorch)
- [ROCm/hrx-system](https://github.com/ROCm/hrx-system) (Loom), [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL),
  [triton-lang/triton](https://github.com/triton-lang/triton)
