# gfx950 vs gfx1250 Instance & Feature Coverage Gaps

Evidence-based gap analysis of Composable Kernel (CK) support for **gfx950**
(MI350 series, CDNA4, native MFMA/XDL matrix instructions) versus **gfx1250**
(RDNA-next / "MI400"-adjacent family per dispatcher code comments, native WMMA
matrix instructions, mapped to `rdna4` in codegen arch tables). Every claim
below cites the exact file (and line, where available) it was verified
against. Where a gap could not be conclusively verified, it is marked
`[UNVERIFIED]`.

This document reflects the repository state as of 2026-09-15 and should be
re-validated after any change to the files it cites.

## 1. Executive Summary

gfx950 and gfx1250 are **not symmetric, drop-in-equivalent targets** in CK
despite both being "newest generation" archs and both gaining
`CK_USE_NATIVE_MX_SUPPORT`. gfx950 is CK's most mature non-CDNA3 target
(MFMA/XDL native, extensively hand-tuned, multi-stage CI hardware testing).
gfx1250 is a **structurally different, less mature, and less tested** target:
it is pinned to CK Tile's legacy (pre-unification) WarpGemm framework, uses
WMMA instead of MFMA, has a growing but still-partial static instance
library, is under-represented or entirely absent in the dispatcher/tile_engine
markdown docs (though present in the underlying generated codegen data), has
several explicitly-disabled correctness-sensitive instance families, and —
most significantly — **CI never executes the test suite on gfx1250 hardware**
(build-verification only, via a software HSA emulator), whereas gfx950 has
five dedicated hardware test-execution CI stages.

## 2. Architectural Baseline

| | gfx950 | gfx1250 |
|---|---|---|
| Family | CDNA4 (MI350 series) | RDNA-next ("rdna4" in codegen; "MI400" in some dispatcher comments) |
| Native matrix instruction | MFMA/XDL (`ck_tile::MfmaOp`, `include/ck_tile/core/arch/mma/mfma/`) | WMMA (`ck_tile::WmmaOp`, `include/ck_tile/core/arch/mma/wmma/`, e.g. `scale_gfx125.hpp`) |
| ck_tile unified-framework status | Default ON (`USE_NEW_UNIFIED_FRAMEWORK=1`, `include/ck_tile/core/arch/arch.hpp:49-51`) | **Force-disabled**: `CMakeLists.txt:~226-233` unconditionally sets `USE_NEW_UNIFIED_FRAMEWORK=0` whenever `GPU_TARGETS MATCHES "gfx1250"` |
| Host capability classification | `is_gfx9_supported()` family; not classified XDL via gfx12 path | `is_gfx12_supported()` returns true for gfx1250 too (`include/ck/host_utility/device_prop.hpp:73-77`), so gfx1250 is host-classified "XDL supported" via `is_xdl_supported() = is_gfx12_supported()||is_gfx11_supported()` (`device_prop.hpp:109-114`) even though its native ISA is WMMA — this is why gfx950 and gfx1250 sometimes share the *same* XDL C++ template family with `#if defined(__gfx950__) \|\| defined(__gfx125__)` device branches (e.g. `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp:129-131,252-254`) |
| ck_tile-internal target identity | n/a | `include/ck_tile/core/arch/arch.hpp` gives gfx1250 its **own** `amdgcn_target_family_id::GFX1250`, distinct from GFX11/GFX12, with the explicit comment: "its MMA builtins and data-type ABI differ, so it must not be treated as a GFX12-family device" — i.e. `ck/` and `ck_tile/` layers classify gfx1250 *differently* relative to "gfx12" |

**Net implication**: code that queries capability via `include/ck/` (`is_gfx12_supported`) treats gfx1250 as part of the gfx12/XDL family; code inside `include/ck_tile/core/arch/arch.hpp` treats it as its own family. Both views are simultaneously true in different parts of the codebase — a source of subtlety when reasoning about "is this code path active for gfx1250?".

## 3. Compile-Time Feature Macro Comparison (root `CMakeLists.txt`)

| Macro | gfx950 | gfx1250 | Gate (CMakeLists.txt) |
|---|---|---|---|
| `CK_USE_XDL` | ON | ON (string-match artifact: `gfx1250` contains substring `gfx12`) | L444-447, regex `gfx9\|gfx11\|gfx12` |
| `CK_USE_GFX94` | OFF | OFF | L448-452 (`gfx94\|gfx95`; note gfx950 matches `gfx95` too, see next row) |
| `CK_USE_GFX950` | **ON** | **OFF** | L456-460, `SUPPORTED_GPU_TARGETS MATCHES "gfx950"` |
| `CK_USE_WMMA` / `CK_TILE_USE_WMMA` / `CK_USE_WMMA_FP8` | OFF | **ON** | L474-486, `gfx11\|gfx12` |
| `CK_USE_OCP_FP8` / `CK_TILE_USE_OCP_FP8` | ON | ON | L487-492, `gfx12 \| gfx950` |
| `CK_USE_FNUZ_FP8` | OFF | OFF | L493-496, gfx90a/gfx94 only |
| `CK_USE_NATIVE_MX_SUPPORT` | **ON** | **ON** (shared) | L497-509 |
| `CK_GFX950_SUPPORT` | **ON** | OFF | L497-502 |
| `CK_GFX1250_SUPPORT` / `CK_USE_GFX1250` | OFF | **ON** (separate `if` block, not folded into the gfx950 block) | L503-509 |
| `CK_GFX12_SUPPORT` | OFF | ON | L510-512, `gfx12` |
| `CK_ENABLE_TF32` | **ON** (gfx942/gfx95 match) | **OFF** (gfx1250 matches neither) | L514-520 |
| `DL_KERNELS` | OFF (neither arch) | OFF | L295-299 |

**Key finding**: `CK_USE_GFX950`/`CK_GFX950_SUPPORT` and `CK_GFX1250_SUPPORT` are
mutually exclusive, purpose-built macros — code gated on the gfx950 macro
(the "double-rate MFMA"/`eightwaves`/preshuffle-B fast paths, see §5) does
**not** automatically extend to gfx1250; per
`dispatcher/codegen/python/grouped_gemm_abquant_utils.py:865-880`, gfx1250
explicitly falls back to "the standard CompV3 pipeline (NOT the gfx950-native
eightwaves/preshuffleb paths, which require `CK_GFX950_SUPPORT`)".

TF32 is a clean, permanent gap: gfx1250 gets **no** TF32 GEMM/conv instances
at all (`library/src/tensor_operation_instance/gpu/CMakeLists.txt:134-140`
gates tf32 instances to `gfx942|gfx950` only), consistent with `CK_ENABLE_TF32`
being off for it.

## 4. The `USE_NEW_UNIFIED_FRAMEWORK` Fork (biggest structural gap)

Root cause, `CMakeLists.txt:~226-233`:
```cmake
if(GPU_TARGETS MATCHES "gfx1250")
    add_compile_definitions(USE_NEW_UNIFIED_FRAMEWORK=0)
    message(STATUS "gfx1250 target detected: forcing USE_NEW_UNIFIED_FRAMEWORK=0 ...")
endif()
```
The comment explains this must be a global `add_compile_definitions` (not a
device-only `__gfx1250__` header guard) because the flag is consumed by
preprocessor logic in headers that **host** code also instantiates (via
`getCMakeCompilerTarget()`); a device-pass-only guard would desync host/device
compilation.

Default value is `1` (`include/ck_tile/core/arch/arch.hpp:49-51`), so **every
other arch including gfx950 uses the new unified framework**; only gfx1250 is
force-pinned to the legacy path. Concretely, with the flag OFF (gfx1250):

- `include/ck_tile/ops/gemm/warp/warp_gemm.hpp:64-622` — the **legacy** full
  catalog of named WarpGemm structs is compiled (vs. a smaller curated new set
  when ON).
- `include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp:13-351` — the entire
  legacy `impl::warp_gemm_dispatcher` namespace only exists when the flag is
  OFF; this is gfx1250's active dispatch code path.
- `include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher_unification.hpp:13-210` —
  the new `UnificationDispatcher` is **entirely absent** from gfx1250 builds.
- `include/ck_tile/ops/gemm/warp/warp_wmma_gemm.hpp:20-261` — named WMMA
  WarpGemm structs are **only** defined in the legacy (`!USE_NEW_UNIFIED_FRAMEWORK`)
  branch; the new-framework branch has a comment stating no named WMMA
  warpgemms are defined there yet. This is why gfx1250 (a WMMA target) is
  pinned to the legacy path — the new framework simply doesn't support WMMA
  yet.
- `include/ck_tile/core/arch/mma/mfma/mfma_gfx9.hpp:535-545` — comment
  documents that the two framework modes are not output-identical even for
  archs that could toggle the flag (numeric/ISA differences, e.g. gfx90a int8
  accumulation).

**Consequence**: gfx1250 cannot benefit from any future `UnificationDispatcher`/
new-framework codegen improvements until CK adds explicit WMMA support to that
framework — a standing architectural gap, not a temporary bug.

## 5. Static Instance Library Coverage (`library/src/tensor_operation_instance/gpu/`)

### 5.1 Instance families unique to gfx950
- "Double-rate MFMA" (`_2x`) tuned instance variants, runtime-gated by
  `if(ck::get_device_name()=="gfx950")`, in `batched_gemm/`, `gemm/`,
  `gemm_add_add_fastgelu/`, `conv2d_fwd*/` (~15 files) — no gfx1250 equivalent.
- smfmac (sparse MFMA) kernels: `test/CMakeLists.txt:403-404` gates the
  `smfmac_op` subdir to `gfx942|gfx950`; gfx1250 explicitly stripped from
  `_smfmac` test targets (`test/CMakeLists.txt:210,314`).
- TF32 instances (§3).
- `mx_mfma_op` test subdir gfx950-only (`test/CMakeLists.txt:406-407`).

### 5.2 Instance families unique to gfx1250
- WMMA CShuffleV3 families: `DeviceGemm_Wmma_CShuffleV3`,
  `DeviceGemmMultipleD_Wmma_CShuffleV3`, `DeviceBatchedGemm_Wmma_CShuffleV3`,
  plus grouped-conv WMMA "large_tiles" instances
  (`library/src/tensor_operation_instance/gpu/grouped_conv2d_{fwd,bwd_data}/.../wmma/*_large_tiles_instance.cpp`).
  These get a **gfx1250-only compiler flag**
  `-enable-post-misched=1;--amdgpu-mfma-vgpr-form` wired in their CMakeLists
  when `GPU_TARGETS MATCHES "gfx1250"`.
- `mx_wmma_op` / `s_prefetch_op` / `prefetch_op` test subdirs, gated to
  `gfx125` (`test/CMakeLists.txt:409-421`) — no gfx950 equivalent.
- A dedicated `_gfx1250_instance.cpp` suffix pattern exists for some grouped
  conv bwd-data bf16 instances
  (`library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_data/nhwgc/xdl/device_grouped_conv2d_bwd_data_xdl_v3_nhwgc_gkyxc_nhwgk_bf16_gfx1250_instance.cpp`).

### 5.3 Instances shared (MX) but architecturally mismatched
`library/src/tensor_operation_instance/gpu/CMakeLists.txt:79-83,361-364` gates
MX (microscaling FP8/FP4/FP6/BF6) instance compilation on
`INST_TARGETS MATCHES "gfx950|gfx125"` — both archs get MX GEMM support — but
the `gemm_mx` subdirectory's instances are **exclusively XDL-shaped**
(`device_gemm_mx_xdl_*`); there is no separate WMMA-flavored `gemm_mx`
instance set even though gfx1250 is not MFMA-native. This works because those
specific instance files are dual-compiled through the shared XDL device-op
template (§2), but individual tile configs within them are frequently
`#if !defined(__gfx125__)`-excluded (see §6).

### 5.4 Explicitly gfx1250-excluded instance configs (correctness/perf reasons)
Found via `#if !defined(__gfx125__) && !defined(CK_USE_GFX1250)` guards:

| File | Reason (from comment) |
|---|---|
| `gemm_multiply_multiply/device_gemm_multiply_multiply_wmma_c_shuffle_{f8_f8_bf16,f8_f8_f16,i8_i8_bf16,i8_i8_f16}_mk_nk_mn.cpp` | "tile too small for gfx1250" (256x128x128/32-tile variant excluded) |
| `gemm_mx/device_gemm_mx_xdl_f4_f4_f16_mk_nk_mn.hpp:52` | "FIXME: This 96x256x128 instance is disabled on gfx1250 due to illegal memory access for the 5120x5120x4096 problem" |
| `gemm_mx/device_gemm_mx_xdl_{bf6_bf6_bf16,f6_f6_f16,f8_f8_bf16,f8_f8_f16}/*.hpp` | Several tile configs excluded, similar numerical/access-fault reasons |
| `gemm_universal/device_gemm_wmma_universal_{bf16_i4_bf16,f16_f8_f16,f8_f8_bf16}/*.hpp` | "numerical issues on gfx1250" / "tile size is not supported on gfx1250" |
| `batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gnk_gmn_instance.cpp:61-64` | "produces incorrect results on gfx1250 with BatchCount=1" |
| `gemm_universal/CMakeLists.txt:147-153,324-328` | Withholds `-mllvm;-greedy-reverse-local-assignment=1` compiler flag from 3 WMMA universal-GEMM source files on gfx1250 due to "multiple unit-test failures" |

The inverse also exists (gfx950-exclusion, for symmetry/completeness):
`batched_gemm_softmax_gemm_permute/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp:72-166`
excludes two instance blocks on gfx950 ("instances not working on gfx950").

### 5.5 Confirmed regression: grouped conv "large tensors" disabled on gfx1250
Matches `CHANGELOG.md` 1.3.0 entry verbatim ("Disabled the large tensor XDL
grouped convolution backward weight instances on gfx1250, where they could
produce intermittent memory access faults") — **and extends further than the
changelog states**:

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle_v3.hpp:1439-1447` —
  `IsSupportedArgument()`: `if constexpr(LargeTensors) { if(is_gfx125_supported()) return false; }`, comment "Memory access runtime error on gfx1250 (inconsistent across runs) // TODO: need fix".
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v3.hpp:992-1000` —
  identical pattern for **backward data** (not named in the CHANGELOG entry).
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp:1456-1464` —
  identical pattern for **forward** (also not named in the CHANGELOG entry).

**Net gap**: large-tensor grouped convolution (fwd, bwd-data, bwd-weight, all
XDL cshuffle-v3 variants) is unconditionally unsupported on gfx1250 at
runtime — the instances still compile into the binary but always reject the
argument via `IsSupportedArgument() == false`. This is broader than the single
CHANGELOG line suggests.

### 5.6 Shared MX device-op gate
`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp:692-696` —
"Only gfx950 and gfx1250 architectures support MX GEMMs" — runtime guard
`ck::get_device_name() != "gfx950" && !is_gfx125_supported()` rejects MX
dispatch on every other target, confirming MX is intentionally a *shared*
gfx950/gfx1250 feature at the device-op level.

### 5.7 ck4inductor (PyTorch Inductor backend)
`python/ck4inductor/{universal_gemm,batched_universal_gemm,grouped_conv_fwd}/gen_instances.py`
each maintain **two separate instance-harvesting paths**: `gen_ops_library()`
(greps XDL classes like `DeviceGemm_Xdl_CShuffleV3`) and a distinct
`gen_ops_library_wmma()` (greps `DeviceGemm_Wmma_CShuffleV3`-family gfx1250
instances, tagging them `is_wmma=True` in `op.py:77-81`). This is **additive**
(WMMA path added alongside XDL, not a per-arch conditional branch), and no
dedicated gfx950-only ck4inductor path was found — gfx950 support is implicit
in the generic XDL path. `[UNVERIFIED further]`: whether every op family that
has an XDL ck4inductor path also has a WMMA one (only confirmed for the three
listed above).

## 6. Dispatcher / Tile Engine / Codegen Coverage

### 6.1 Three layers, three different levels of gfx1250 maturity
1. **Markdown docs** (`tile_engine/operation_support_matrix.md`,
   `dispatcher/README.md`, `dispatcher/STREAMK.md`,
   `dispatcher/codegen/ADDING_NEW_GPU.md`) — gfx1250 is **completely absent**.
   Columns/examples stop at gfx1201. These docs are demonstrably stale versus
   the code (see 6.2).
2. **Generated codegen config** (`dispatcher/codegen/arch_specs_generated.py`,
   generated from `arch_specs.json`, timestamp 2026-06-01) — gfx1250 **is**
   present: `ARCH_FAMILY_MAP['gfx1250'] = 'rdna4'`,
   `WARP_TILE_SUPPORTED_COMBINATIONS['gfx1250']` has its own WMMA tile profile
   (`fp16/bf16: 16x16x32`, `fp8/bf8: 16x16x64`) distinct from both gfx1201's
   generic 16x16x16 tiles and gfx950's rich MFMA tile set (including
   16x16x128, pk_fp4). No preshuffle warp-tile entry exists for gfx1250.
3. **ctypes bridge + Python sweep-config layer** — gfx1250 enablement is
   **per-op and uneven** (§6.3).

### 6.2 Documentation staleness (concrete evidence)
- `tile_engine/operation_support_matrix.md` — full read confirms GPU columns
  are exactly `90a | 942 | 950 | 1201`; gfx1250 is not mentioned anywhere.
- `dispatcher/README.md:151-156` ("Supported architectures") lists
  gfx90a/gfx942/gfx950/gfx1101/gfx1201 only; the Warp Tile Combination tables
  (~L754-786) use the same 4-column set; footnote [7] names gfx908/gfx1100/
  gfx1200 as present in `arch_specs.json` but not shown as columns — gfx1250
  is absent even from that footnote.
- `dispatcher/README.md:141` — "Newer GPU targets (gfx950, gfx1201) require
  ROCm 6.3+" names gfx1201, **not** gfx1250.
- `dispatcher/STREAMK.md` — "Known limitations" section covers gfx942
  (validated) and gfx950 (fp8/bf8 FNUZ-vs-OCP mismatch); gfx1250 is not
  mentioned at all — Stream-K status on gfx1250 is **undocumented**.
- `dispatcher/codegen/ADDING_NEW_GPU.md` — generic onboarding guide; its
  example GPU-family table (cdna2/cdna3/cdna4/rdna3/rdna4 →
  gfx90a/gfx942/gfx950/gfx1100/gfx1201) never names gfx1250 as an example.
- `dispatcher/codegen/arch_filter.py` — the `ArchFilter` class falls back to a
  **hardcoded minimal subset** (gfx90a/gfx942/gfx950/gfx1201 only) if
  `arch_specs_generated.py` is missing/stale — meaning a broken codegen
  regeneration would silently drop gfx1250 support.

### 6.3 ctypes bridge layer: inconsistent per-op gfx1250 gating

| Bridge | gfx1250 status | Evidence |
|---|---|---|
| `quant_bridge_common.hpp::validate_supported_arch()` (used by gemm_abquant/aquant/bquant, grouped_gemm_abquant/aquant/bquant) | **Included** | L113-135, explicit `arch.rfind("gfx1250",0)==0` |
| `grouped_gemm_rowcolquant_ctypes_lib.cpp` | **Excluded** | Hardcodes own `kSupportedArchs[] = {"gfx942","gfx950"}` (L77), bypasses the shared header |
| `grouped_gemm_tensorquant_ctypes_lib.cpp` | **Excluded** | Same pattern, own `kSupportedArchs[] = {"gfx942","gfx950"}` (L80) |
| `mx_gemm_ctypes_lib.cpp` | **Hard-excluded, architecturally** | Compile-time `static_assert(GFX_ARCH=="gfx950")` (L68-70) + runtime device check (L179-195); depends on gfx950-only host helper `preShuffleScaleBuffer_gfx950` (no e8m0 microscaling pre-shuffle hardware path exists for gfx1250) |
| `batched_contraction_ctypes_lib.cpp` | **Included** (generic, `#error`-gated, no allow-list) | Python-side `_SUPPORTED_ARCHS = ("gfx90a","gfx942","gfx950","gfx1250")`, dedicated `default_ci_config_gfx1250.json` |
| `fmha_ctypes_lib.cpp` | `[UNVERIFIED]` — no explicit exclusion found, but no gfx1250-specific enablement code found either; inferred low support since gfx1201's FMHA column in `tile_engine/operation_support_matrix.md` is ❌ | — |
| Generic bridges (`gemm_ctypes_lib`, `streamk_gemm`, `multi_d_gemm`, `gemm_multi_abd`, `grouped_gemm`) | Included (generic `#error`-gated, no hardcoded allow-list) | Gated only by whether the Python codegen layer emits valid gfx1250 kernels |

Note the asymmetry: `grouped_gemm_{aquant,bquant,abquant}` **do** support
gfx1250 via the shared header, while their sibling
`grouped_gemm_{rowcolquant,tensorquant}` explicitly do **not** — an
inconsistency within the same op family, not an architectural necessity.

### 6.4 Python sweep-config layer: gfx1250 is real but requires bespoke tuning
`dispatcher/python/{batched_gemm_utils.py, gemm_utils.py,
grouped_gemm_{abquant,aquant,bquant}_utils.py}` all list gfx1250 in
`_SUPPORTED_ARCHES` and ship dedicated `default_*_config_gfx1250()` helper
functions plus `default_ci_config_gfx1250.json` files, each with extensive
empirical-tuning comments, e.g.:

- `grouped_gemm_abquant_utils.py:862-925` ("MI400" section): gfx1250 uses the
  standard CompV3 pipeline, not gfx950's `eightwaves`/preshuffle-B fast path
  (which requires `CK_GFX950_SUPPORT`); only `warp_tile_k=128` (FlatMM,
  16x16x128) produces correct results for fp8/bf8 on gfx1250 — `warp_tile_k=32`
  (gfx9 MFMA) and `warp_tile_k=16` (gfx12 WMMA) silently return wrong/zero
  results.
- `grouped_gemm_aquant_utils.py:790-851` and `grouped_gemm_bquant_utils.py:1219-1271` —
  same tuning; **explicitly document that fp8i4/bf8i4 (packed-int4 quant)
  variants do not compile on gfx1250 at any warp_tile_k** — "no gfx12
  WMMA/FlatMM instruction for the packed-int4 quant path" — a genuine,
  documented functional gap versus gfx950 which supports these variants.
- `batched_gemm_utils.py:2424-2432` — a documented, currently-unfixed
  correctness bug: the CompV3/intrawave pipeline miscompiles for 8-warp
  block arrangements (`2x4x1`/`4x2x1`) on gfx1250 (max_rel error 0.14–0.87),
  citing tracking issue **ROCm/rocm-libraries#11161** — matches the CHANGELOG
  1.3.0 fix "Fixed incorrect results from the CK Tile compute v3 intrawave
  GEMM pipeline with eight-warp block arrangements on gfx1250" (verify fix is
  actually landed vs. still tracked — code comment suggests it may be a
  documented workaround/gate rather than a full fix; treat as
  `[UNVERIFIED]` whether fully resolved).
- `unified_grouped_conv_codegen.py:2356` — CLI `--gpu-target` choices include
  gfx1250, and `grouped_config_rules_full.py:495-499` has an explicit
  rdna4/gfx1250 fallback (falls back to all tile_math-valid pairs since curated
  CDNA-derived wave combos don't apply) — grouped-conv codegen treats gfx1250
  as supported (if less-curated) even though `tile_engine/operation_support_matrix.md`
  shows grouped_conv as ❌ across all documented GPU columns.

## 7. FMHA (Attention) Pipeline Coverage Gap

FMHA is the single largest **feature-parity** gap between gfx950 and gfx1250,
via the `KernelComponentFactory*` codegen classes in
`example/ck_tile/01_fmha/codegen/ops/{fmha_fwd.py,fmha_bwd.py}`:

| Capability | gfx950 (`KernelComponentFactoryGfx950`) | gfx1250 (`KernelComponentFactoryGfx125`) |
|---|---|---|
| Dtypes | fp16/bf16/fp8/fp8bf16/fp8fp32 **+ `mxfp8`, `mxfp4`** | fp16/bf16/fp8/fp8bf16/fp8fp32 only — **no MX dtypes** |
| Pipelines | qr, qr_hpad, qr_async, **qr_async_trload, qr_async_trload_v3 (V3/`FmhaFwdV3Kernel`)** | qr, qr_hpad, **qr_tdm** (gfx1250's own dedicated LDS-resident-QKV pipeline) — no V3, no trload |
| `fmha_fwd.py:994-1024,1174-1260,1369,1440,1789-1801` | V3 pipeline tag (`qr_async_trload_v3`) only emitted for gfx950 traits | Never emits `qr_async_trload_v3`; `fmha_fwd_v3` API split only triggers for that tag |
| trload-kernel availability | `fmha_fwd_kernel.hpp:173-176`/`fmha_bwd_kernel.hpp:488-491`: `kIsAvailable` unconditionally true on `__gfx950__` | Disabled whenever `!kUseTrLoad` is required — **gfx1250 cannot use trload-based fwd/bwd kernel variants** |
| hdim256-optimized single-buffer path | `block_fmha_pipeline_qr_ks_vs_async_trload.hpp:154-1876`, gfx950-only | Not available |

**CHANGELOG.md cross-check**: no entry across the full history ever adds MX
or V3 FMHA support to gfx12/gfx1250; existing gfx1250 FMHA entries are
performance-only (LDS padding for qr_tdm head-dim 128) or arch-agnostic bug
fixes. Base WMMA (gfx12) FMHA support originates from a single older entry
("Added WMMA (gfx12) support for FMHA") and has not been extended with MX/V3
since.

**Known bug in gfx1250's unique pipeline**: `example/ck_tile/01_fmha/script/smoke_test_fwd.sh:321-325`
documents that `qr_tdm` (gfx1250-only) has "a known ping-pong K/V prefetch bug
for prefill s>=2048 with sink masks" — gfx950 is unaffected because it doesn't
use `qr_tdm`.

**Early-silicon (A0) test skips**: `test/ck_tile/gemm/test_gemm_pipeline_util.hpp:443-464`
`GTEST_SKIP`s TDM cluster-launch pipelines and 32-wide WMMA F4 instructions on
gfx1250 when `asicRevision==0` (A0 silicon), i.e. gfx1250 test coverage is
further reduced by hardware-revision gating independent of software maturity.

## 8. Test Suite Gaps

### 8.1 `KNOWN_FAILING_TESTS` (gfx1250-emulator-only, `test/CMakeLists.txt:124-136`)
Comment: "List of tests currently failing on gfx1250 emulator. All of the
failing tests must be fixed asap." Full list (10 entries, all MX/microscaling
related, no gfx950 equivalent list exists):
```
test_ck_tile_mx_scale
test_ck_tile_mx_gemm_pipeline_tdm_wmma
test_gemm_blockscale_wp_fp8
test_fp6
test_bf6
test_mx_fp4
test_mx_fp6
test_mx_bf6
test_mx_fp8_pk4scale
test_mx_bf8_pk4scale
```
This confirms MX/microscaling is CK's **least mature functional area on
gfx1250** at present.

### 8.2 Named test files
- gfx1250-named: `test/ck_tile/core/arch/mma/test_amdgcn_mma_layout_gfx1250`
  (`test/core/arch/mma/CMakeLists.txt`, built only when `GPU_TARGETS` matches
  `gfx1250` exactly, no `amdgcnspirv` fallback), and
  `test/ck_tile/warp_gemm/test_wmma_bf16_16x16x32_gfx1250.cpp`. Only **two**
  gfx1250-named test files exist.
- gfx950-named: `test_amdgcn_scale_mma`, `test_amdgcn_mma_layout_gfx950`
  (has an `amdgcnspirv` fallback gfx1250's variant lacks),
  `test/ck_tile/epilogue/test_cshuffle_epilogue_fp8_gfx950.cpp` (no gfx1250
  counterpart).

### 8.3 Arch-exclusion asymmetries in `test/CMakeLists.txt`
- L294,311-312: `monitor_mwait`/`async_lds_load_store` (sync tests) are
  **gfx1250-only** — stripped from every other arch including gfx950
  ("only build sync tests for gfx1250+").
- L315-318: `_mx` tests built for gfx950 and gfx125x; **nested** exclusion —
  `_pk4scale` MX variants are gfx1250-**only**, explicitly removed for gfx950
  even though the general `_mx` filter would otherwise include it.
- L208,310: `_wmma` tests strip gfx950 (no WMMA hardware).
- L210,314: `_smfmac` tests strip gfx1250 (sparse MFMA is gfx942/gfx950-only).
- `example/CMakeLists.txt:204`: comment says "only build mx example for
  gfx950" but the actual `REMOVE_ITEM` list never strips gfx1250 — a
  **stale/inaccurate comment**, not a functional exclusion (gfx1250 passes
  this filter too).

## 9. CI Coverage Gap (Jenkinsfile / groovy/vars/ck.groovy)

This is the most consequential, unambiguous gap found.

**gfx950** gets 5 dedicated stages, each scheduling onto real
`rocmnode("gfx950")` hardware and invoking real test execution:
1. "Run AITER Tests on gfx950" (`Jenkinsfile:422`)
2. "Run FA Tests on gfx950" (`Jenkinsfile:456`)
3. "Run CK_TILE_FMHA Tests on gfx950" (`Jenkinsfile:603`)
4. "Run TILE_ENGINE_GEMM Tests on gfx950" (`Jenkinsfile:695`)
5. "Build CK and run Tests on gfx950" (`Jenkinsfile:760`, calls
   `ck.runBuildCKAndTests("gfx950")`)

**gfx1250** gets exactly **one** stage: "Build CK for gfx1250"
(`Jenkinsfile:906`, gated by `params.BUILD_GFX1250`) — and it:
- Schedules onto `rocmnode("gfx90a")` hardware, **not gfx1250 silicon**.
- Calls the same `ck.runBuildCKAndTests("gfx1250")` helper, but
  `groovy/vars/ck.groovy:927-951` special-cases any `setup_args` containing
  `"gfx1250"`: it explicitly **skips** `ninja install check` (the real
  test-execution target) and instead runs `ninja -j${nt} install smoke`
  (build-verification only) under a **software HSA model/emulator**
  (`HSA_MODEL_LIB=/libhsakmtmodel.so`, `HSA_MODEL_TOPOLOGY=/topology/mi450`),
  with the code comment "do not run tests on gfx1250, just build everything".
- `groovy/vars/ck.groovy:1017` — performance-test triggering is also
  explicitly disabled for gfx1250 (`!setup_args.contains('gfx1250')` guard).
- Uses `-DDISABLE_DL_KERNELS=ON` and a dedicated, more restrictive Docker
  image (`ck_ub24.04_gfx1250_ffm`) not shared with gfx950's build.
- No post-build trace visualization is archived for gfx1250
  (`Jenkinsfile:927-937` archives traces for gfx11/gfx12/gfx90a/gfx942/gfx950
  only).

Note: gfx1201 (a **different**, already-released gfx12 target) *does* get
real hardware test stages ("Run CK_TILE_FMHA Tests on gfx1201",
`Jenkinsfile:621`; "Run TILE_ENGINE_GEMM Tests on gfx1201", `Jenkinsfile:713`)
— so the gap is specific to gfx1250, not "all gfx12 targets."

**Conclusion**: as CI is currently configured, **no CK code path ever executes
correctness tests on real gfx1250 silicon**; all gfx1250 CI validation is
build-only, under a software HSA emulator, on non-gfx1250 hardware.

## 10. Summary Table — Gap Inventory by Category

| # | Category | Gap | Severity | Type |
|---|---|---|---|---|
| 1 | ck_tile framework | gfx1250 pinned to legacy WarpGemm framework (`USE_NEW_UNIFIED_FRAMEWORK=0`); no access to `UnificationDispatcher` | High — structural | Standing architectural limitation |
| 2 | TF32 | No TF32 instances on gfx1250 at all | Medium | Missing feature |
| 3 | smfmac | No sparse-MFMA support on gfx1250 (expected — no hardware) | N/A | Architectural (not a bug) |
| 4 | Grouped conv "large tensors" | fwd/bwd-data/bwd-weight XDL v3 large-tensor instances unconditionally rejected at runtime on gfx1250 | High — broader than CHANGELOG states | Confirmed regression |
| 5 | Packed-int4 quant (fp8i4/bf8i4) | Does not compile on gfx1250 (no WMMA/FlatMM int4 instruction) for grouped_gemm aquant/bquant | Medium | Missing feature (documented) |
| 6 | 8-warp CompV3/intrawave pipeline | Miscompiles on gfx1250 for `2x4x1`/`4x2x1` block arrangements (tracked: ROCm/rocm-libraries#11161) | Medium | Correctness bug, tracked |
| 7 | ctypes bridges (rowcolquant/tensorquant) | Hardcoded `kSupportedArchs` excludes gfx1250 while sibling ops (abquant/aquant/bquant) include it | Low-Medium | Inconsistent gating (fixable) |
| 8 | MX-GEMM ctypes bridge | gfx950-only by design (`preShuffleScaleBuffer_gfx950` dependency); no gfx1250 path | Medium | Architectural exclusion |
| 9 | FMHA MX dtypes + V3 pipeline | Not available on gfx1250 (`KernelComponentFactoryGfx125` has no MX types, no V3 tag) | High — major feature gap | Missing feature |
| 10 | FMHA `qr_tdm` prefetch bug | Known bug for prefill seqlen≥2048 with sink masks, gfx1250-only pipeline | Medium | Correctness bug, documented |
| 11 | MX/microscaling test suite | 10 tests in `KNOWN_FAILING_TESTS` on gfx1250 emulator | High | Confirmed test failures |
| 12 | A0-silicon test skips | TDM/32-wide WMMA F4 tests `GTEST_SKIP`ped on gfx1250 A0 revision | Low-Medium | Hardware-revision limitation |
| 13 | tile_engine/dispatcher docs | gfx1250 entirely absent from `operation_support_matrix.md`, `dispatcher/README.md`, `STREAMK.md`, `ADDING_NEW_GPU.md` despite being present in generated codegen data | Medium | Documentation staleness |
| 14 | CI test execution | gfx1250 CI is build-only (software HSA emulator, non-gfx1250 hardware); gfx950 has 5 real-hardware test stages | **Highest — no gfx1250 correctness signal in CI** | CI/process gap |
| 15 | Preset support | No `dev-gfx1250` CMake preset exists (gfx950 has one); gfx1250 never appears in HIP-version-gated default `CK_GPU_TARGETS` lists | Low | Tooling/DX gap |

## 11. Where To Look When Extending gfx1250 Support

- Instance additions: mirror the `_gfx1250_instance.cpp` / WMMA
  `*_large_tiles_instance.cpp` pattern in
  `library/src/tensor_operation_instance/gpu/<op>/` and register via that
  op's `CMakeLists.txt`.
- FMHA: extend `KernelComponentFactoryGfx125` in
  `example/ck_tile/01_fmha/codegen/ops/{fmha_fwd.py,fmha_bwd.py}` to add MX
  dtypes / a V3-equivalent pipeline tag; verify against `kIsAvailable` gating
  in `include/ck_tile/kernel/fmha_{fwd,bwd}_kernel.hpp`.
- Dispatcher ctypes: align `grouped_gemm_{rowcolquant,tensorquant}_ctypes_lib.cpp`
  to use the shared `quant_bridge_common.hpp::validate_supported_arch()`
  (already includes gfx1250) instead of their own hardcoded `kSupportedArchs`.
- Docs: update `tile_engine/operation_support_matrix.md`,
  `dispatcher/README.md`, `dispatcher/STREAMK.md`, and
  `dispatcher/codegen/ADDING_NEW_GPU.md` to add a gfx1250 column/mention —
  the underlying `arch_specs_generated.py` data already supports it.
- CI: adding a real hardware test-execution stage for gfx1250 (paralleling
  the 5 gfx950 stages) requires changing `groovy/vars/ck.groovy`'s
  `setup_args.contains('gfx1250')` branch (currently forces `ninja install
  smoke` instead of `ninja install check`) once gfx1250 hardware/CI capacity
  is available.
