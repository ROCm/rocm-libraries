# INSIGHTS.md -- FMHA BWD rocm_ck Example

Design decisions and lessons learned from mapping the CK Tile FMHA backward
kernels (OGradDotO, DqDkDv, ConvertDQ) to the rocm_ck kpack pattern.

## IGLP Pipeline Crash on clang 22.0.0git (ROCm Mainline)

**Severity: BLOCKER for pad_hdim_q=8 / pad_hdim_v=8 configs**

The `BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLP` pipeline variant produces
`HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` at runtime on gfx942 when compiled
with clang 22.0.0git (ROCm mainline, commit `c849bc16`). The kernel compiles
without errors but crashes immediately on launch.

**Pipeline selection logic** (from `block_fmha_bwd_dq_dk_dv_pipeline_selector.hpp`):
- `kUseTrLoad=false` AND `has_dpad1=false` → selects IGLP variant (crashes)
- `kUseTrLoad=false` AND `has_dpad1=true`  → selects non-IGLP variant (works)
- `has_dpad1 = (kPadHeadDimQ == 1 || kPadHeadDimV == 1)`

**Workaround**: Use `pad_hdim_q=1` and `pad_hdim_v=1` instead of `8` for all
DqDkDv variants. This forces `has_dpad1=true`, selecting
`BlockFmhaBwdDQDKDVPipelineKRKTRVR` (the non-IGLP variant) which works
correctly and passes numerical verification.

**Impact on performance**: The `pad=1` variant adds minimal bounds checking
that `pad=8` (vector-aligned padding) would optimize away. For the demo
this is acceptable. For production use, this needs to be investigated with the
CK Tile team — the IGLP pipeline may have a compiler-specific bug with the
`amd-mainline` clang branch.

**How we found this**: The OGradDotO kernel (simple 1D reduction) worked
perfectly with generic Args. The DqDkDv kernel (5-GEMM template chain)
crashed on every launch. Bisecting by pipeline variant isolated the IGLP
pipeline as the crash source. The non-IGLP pipeline with identical Kargs
initialization passes all tests.

**TODO**: File a bug against the CK Tile IGLP pipeline with this compiler
version. Test with release ROCm compilers (6.x, 7.x) to determine if this
is a mainline regression.

## Generic Args with Named Slot Constants

All kernel families use the same `rocm_ck::Args` struct (1408 bytes, 34% of
the 4096-byte HSA kernarg budget). This eliminates per-mode flat structs,
`__builtin_bit_cast`, and all Kargs ABI tracking.

### Named Slot Constants

Each kernel family defines a slot namespace (e.g., `fmha_bwd_dqdkdv_slots`)
with named constants for tensor and scalar indices. This prevents off-by-one
errors in the 50-parameter DqDkDv kernel argument mapping:

```cpp
namespace S = fmha_bwd_dqdkdv_slots;
const auto& t_q = args.tensors[S::Q];   // not args.tensors[0]
const auto& t_k = args.tensors[S::K];   // not args.tensors[1]
```

Optional tensor slots (BIAS=9, DBIAS=10, RANDVAL=11) have **fixed indices**
regardless of which features are enabled. Unused slots are simply not
populated — no slot remapping.

### Aggregate Kargs Initialization

The device bridge constructs CK Tile's Kargs via aggregate initialization
(matching the inheritance order), rather than `__builtin_bit_cast`. This
works because CK Tile's own `MakeKargsImpl` uses the same pattern.

For the DqDkDv kernel, the Kargs struct uses multiple inheritance with 5
conditional base classes (bias, dbias, mask, dropout, deterministic). When
a feature is disabled, the base resolves to `FmhaBwdEmptyKargs<N>` (empty
struct). The aggregate init uses `{}` placeholders for these, then assigns
optional fields via `if constexpr`:

```cpp
typename T::Kargs kargs{
    {/* CommonKargs: 32 fields */},
    {},  // bias (empty or BiasKargs)
    {},  // dbias
    {},  // mask
    {},  // dropout
    {},  // deterministic
    /* batch-mode: 8 stride fields */
};
if constexpr(hasMask(K)) {
    kargs.window_size_left  = -1;
    kargs.window_size_right = 0;  // causal
    kargs.mask_type = MASK_FROM_TOP_LEFT;
}
```

The mask parameters are runtime-configurable via the scalar slots, so a
mask-enabled spec can be re-purposed for different mask shapes without
recompilation. Host-side population for a top-left causal mask:

```cpp
namespace S = fmha_bwd_dqdkdv_slots;
args.scalars[S::WINDOW_SIZE_LEFT].i32  = -1; // unlimited left context
args.scalars[S::WINDOW_SIZE_RIGHT].i32 = 0;  // causal
args.scalars[S::MASK_TYPE].i32         = static_cast<int>(
    ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
```

The device bridge reads these scalars and assigns them to `kargs` after
the aggregate init -- the spec-time `mask_type` enum only gates which
inheritance base is active (NO_MASK vs any other family), not the
runtime mask shape itself.

### 1D Tensor Stride Convention

Tensors without a row stride (D, LSE) pack strides directly:
- `strides[0] = nhead_stride`
- `strides[1] = batch_stride`

NOT `strides[0] = 1, strides[1] = nhead_stride`. A spurious `1` in
`strides[0]` shifts all subsequent strides and causes wrong results.

### Problem Dimensions via Scalars

Each tensor carries only its own natural dimensions in `lengths[]`:
- Q: `lengths[0]=seqlen_q, lengths[1]=hdim_q`
- K: `lengths[0]=seqlen_k, lengths[1]=hdim_q`
- V: `lengths[0]=seqlen_k, lengths[1]=hdim_v`

Problem-level dimensions that don't belong to any single tensor are passed
as scalars: `scalars[NUM_HEAD_Q].i32`, `scalars[NHEAD_RATIO_QK].i32`.

## HIP Device Consteval Limitations

The `.hip` files use `static constexpr <ExplicitType> kernel = make_kernel(...)`
as an NTTP for the device bridge template. Two HIP compiler limitations apply:

1. **`constexpr auto` fails on device** — `static constexpr auto kernel = ...`
   causes "const variable cannot be emitted on device side due to dynamic
   initialization." Use an explicit type (`FmhaBwdOGradDotOKernel`,
   `FmhaBwdDQDKDVKernel`, etc.) instead of `auto`.
2. **`__launch_bounds__` with struct members** —
   `__launch_bounds__(kernel.block_size, kernel.block_per_cu)` works with
   explicit kernel descriptor types. It fails only with `auto`-deduced types.

### `<array>` in API Headers — NOT a Problem

Earlier versions of this document stated that including `<rocm_ck/args.hpp>`
(which transitively includes `<array>`) in API headers would break device
consteval. **This is incorrect.** The GEMM example (`gemm_api.hpp`) includes
`<rocm_ck/resolve.hpp>` → `<rocm_ck/signature.hpp>` → `<rocm_ck/args.hpp>`
→ `<array>` in its API header, and all .hip files compile successfully.

The consteval evaluator runs entirely in the compiler frontend and does not
interact with non-constexpr static initializers from `<array>`. The include
order is irrelevant for consteval evaluation.

The FMHA BWD API headers currently do not include `args.hpp` or `resolve.hpp`,
but this is an implementation choice, not a constraint. If unified Signature
support is added to FMHA BWD API headers in the future, including
`resolve.hpp` will work without issues.

## Transpose Load (TrLoad) Support

CK Tile supports transpose loads for FMHA BWD DqDkDv on architectures with
hardware transpose load capability:

- **gfx908/gfx90a/gfx942**: No TrLoad support. `kUseTrLoad=false`.
- **gfx950**: Hardware TrLoad available. The codegen generates 6 additional
  TrLoad-specific tile configs. TrLoad requires `pad_hdim=8` and selects
  `BlockFmhaBwdDQDKDVPipelineTrLoadKRKTRVR` or the decode-mode
  `BlockFmhaBwdDQDKDVPipelineTrLoadQRQTRDOR` variant.
- **gfx11xx/gfx12xx (RDNA)**: No TrLoad support.

Currently `kUseTrLoad` is hardcoded to `false`. Enabling it for gfx950
requires adding a `use_tr_load` field to the Algorithm struct and
TrLoad-specific tile configs.

## Group Mode

Group mode uses variable-length sequences within a batch:
- **Batch**: `[batch, nhead, seqlen_q, hdim_v]` with fixed batch strides
- **Group**: `[total_seq, nhead, hdim_v]` where `total_seq = sum(seqlen_q_i)`

The kernel computes per-batch offsets from `seqstart_q_ptr` (cumulative sequence
lengths) rather than fixed batch strides. Group mode requires `pad_seqlen_q=true`
(OGradDotO, ConvertDQ) or nonzero `pad_hdim` (DqDkDv) because partial tiles
at sequence boundaries are inherent.

The host test must re-layout data when testing group-mode variants against a
batch-mode CPU reference.

## D Shares LSE Stride Layout

Both D (OGradDotO result) and LSE (log-sum-exp) are 1D per (batch, head, seqlen_q).
CK Tile uses `nhead_stride_lsed` and `batch_stride_lsed` for both. In the generic
Args, both tensors use the same stride packing convention:
`strides[0]=nhead_stride, strides[1]=batch_stride`.

## Unified Signature Migration — Decision

Evaluated migrating FMHA BWD from per-kernel Signature/Algorithm/Config types
to the unified `rocm_ck::Signature` + `FmhaBwdOp` framework used by the GEMM 
example.

**Decision: Do not migrate.** The current per-kernel types are appropriate
domain specialization, not structural debt. Key findings:

- **FMHA BWD breaks the GEMM pattern**: 3-kernel pipeline (not 1 kernel),
  feature-gated optional tensors (BIAS/DBIAS/RANDVAL), and 3 structurally
  different kernel families that are not variations of one operation.
- **`resolve()` adds no new information**: Unlike GEMM where `resolve()`
  provides dtype cascade and layout propagation, FMHA tensors have explicitly
  known properties. `resolve()` just confirms what's already stated.
- **No compiler blockers**: The `<array>` include chain, consteval resolution,
  and NTTP derivation all work (proven by GEMM). The question is design fit,
  not compilation.

`FmhaBwdOp` was added to the shared infrastructure (`ops.hpp`, `resolve.hpp`)
and is available for future use (Signature-based dispatch, kernel selection,
pipeline description). Full analysis:

## Naming Convention

We follow CK Tile internal naming (`FmhaBwdOGradDotOKernel`,
`FmhaBwdDQDKDVKernel`, `FmhaBwdConvertQGradKernel`) rather than the legacy
dispatcher naming (`bwd_dot_do_o`). This aligns type names across our API and
the CK Tile template chain, reducing confusion when tracing the template
instantiation path.

## PhysicalTensor Does Not Apply to FMHA BWD

The GEMM example uses `PhysicalTensor` and `GemmSpec` to build a physical tensor
table mapping named tensors to `Args::tensors[]` slots. This was analyzed for
FMHA BWD applicability and **deliberately rejected**. The rationale:

**GEMM's problem is different.** In GEMM, the tensor set is *configurable* — the
user names tensors via a Signature graph ("A", "B", "bias"), the epilogue chain
determines which tensors exist, and the output tensor's name changes depending
on the chain ("C" → "D" → "E"). PhysicalTensor provides dynamic name-to-slot
resolution because slots are assigned at `make_kernel()` time from the graph.

**FMHA BWD has a fixed domain-specific ABI.** Q is always slot 0, K is always
slot 1, etc. (`fmha_bwd_dqdkdv_slots`). Optional tensors (BIAS=9, DBIAS=10,
RANDVAL=11) have fixed indices regardless of which features are enabled. The
slot namespace already provides the same safety as PhysicalTensor's `args_slot`
field — named constants prevent off-by-one errors.

**Specific reasons against adoption:**
- `kMaxPhysicalTensors = 8` is insufficient for FMHA BWD's 16 slots
  (Q..DV + BIAS/DBIAS/RANDVAL + 4 group-mode SEQSTART/SEQLEN slots). Bumping
  it globally bloats every `GemmSpec` NTTP with unused padding.
- FMHA BWD tensors have heterogeneous types (Q/K/V are fp16, LSE/D/DQ_ACC are
  fp32, RANDVAL is uint8). PhysicalTensor's single `dtype` per tensor adds
  redundant metadata already handled by `FmhaBwdDQDKDVTypes<K>`.
- Layout (Row/Col) is not meaningful for FMHA tensors — they are attention
  matrices, not GEMM operands with Row/Col layout semantics.
- `requiredTensors(k)` / `requiredScalars(k)` already encode "which slots are
  active" — the information that PhysicalTensor's table would provide.

**In summary:** PhysicalTensor is designed for operator-graph kernels (GEMM,
elementwise) where the tensor ABI emerges from a user-provided Signature.
FMHA BWD has a domain-specific fixed ABI — forcing PhysicalTensor would be
like using a map when you need a struct.

*(analysis: 2026-03-27)*
