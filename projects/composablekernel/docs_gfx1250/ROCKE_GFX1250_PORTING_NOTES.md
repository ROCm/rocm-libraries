# gfx1250 Support in rocKE: Porting Notes for Composable Kernel

This document inventories what **rocKE** (`dnn-providers/hip-kernel-provider/rocke/`)
has built for **gfx1250** and evaluates each piece against the gfx1250 gaps
already documented for **Composable Kernel** (CK,
`projects/composablekernel/GFX950_VS_GFX1250_COVERAGE.md`). It is organized
around CK's open gaps, one section per gap, each ending in a **Verdict**:

- **PORT** — rocKE has a concrete, evidence-backed solution or design pattern CK should adopt.
- **PARTIAL** — rocKE has a relevant but incomplete/unshipped answer; useful as a lead, not a drop-in fix.
- **SHARED GAP** — rocKE has the identical open problem; nothing to port, but confirms the gap is a real hardware/toolchain limitation rather than a CK-only bug.
- **CK IS AHEAD** — CK has already made progress rocKE hasn't reached yet.

All rocKE paths are relative to `dnn-providers/hip-kernel-provider/rocke/`
unless given as absolute. All CK paths are relative to
`projects/composablekernel/`. rocKE's gfx1250 work reflects its state as of
the 2026-06 to 2026-09 planning/case-study snapshots read here; re-verify
before acting on anything marked in-progress.

## 0. What rocKE Is (context for the comparison)

rocKE is a Python (and mirrored C++20) kernel-authoring stack: a kernel spec
is built into a typed SSA `KernelDef`, then a lowering engine emits AMDGPU
LLVM IR directly — no C++ template metaprogramming, no external HIP compiler
driver in the hot path (`README.md`). It keeps CK-Tile's tile abstractions
(tensor descriptors, coordinate-transform DAGs, tile windows, MFMA/WMMA atoms,
software pipelines) but as first-class Python/SSA objects. It targets both
CDNA (gfx942/gfx950, MFMA) and RDNA/GFX12 (gfx1151/gfx1201/**gfx1250**, WMMA)
architectures from one framework, which makes its per-arch abstraction
boundaries directly comparable to CK's `ck::`/`ck_tile::` per-arch code.

**Important classification note**: rocKE explicitly treats gfx1250 as its
*own* architecture family (`target_family: gfx12_cdna` in
`platform/python/rocke/core/arch/data/arch_specs.json`), deliberately **not**
aliased to gfx1201/RDNA4 despite both being wave32/WMMA — this mirrors CK's
own `ck_tile::amdgcn_target_family_id::GFX1250` special-casing
(`include/ck_tile/core/arch/arch.hpp`) and is a second independent
confirmation that gfx1250 needs its own classification bucket, not a
"gfx12-generic" one.

## 1. LDS Transpose-Load Builtin Hazard (`ds_load_tr16_b128`) — **PORT**

**CK's gap**: gfx1250 core arch code (`include/ck_tile/core/arch/mma/mma_wavewise.hpp`,
`scale/scale_mma_pipeline.hpp`) has multiple `// TODO`/`// Dubious for
gfx1250, needs attention` comments about LDS transpose-load builtin handling,
and CK has several WMMA GEMM instance configs disabled on gfx1250 for
"numerical issues" or "illegal memory access" with no root cause identified
(§5.4 of `GFX950_VS_GFX1250_COVERAGE.md`).

**rocKE's finding — a concrete, reproduced compiler hazard**:

> The AMDGPU LLVM backend, when it sees a plain sequential `load <8 x half>`
> (or bf16) from `addrspace(3)` (LDS) that feeds a WMMA operand, **silently
> substitutes** the `ds_load_tr16_b128` (hardware transpose-LDS-read)
> instruction for it — this assumes the LDS tile is stored column-major. If
> the kernel actually stored the tile row-major (as is common for coalesced
> global→LDS writes), the substitution silently corrupts the WMMA input.

Measured magnitude: **"producing wrong WMMA inputs (factor ~136x wrong in
single-element tests)"**
(`platform/python/rocke/core/isa/backend.py:534-539`).

**rocKE's fix** (`Gfx1250Backend.blocks_ds_load_tr16 = True`,
`platform/python/rocke/core/isa/backend.py:517-563`; C++ mirror
`platform/cpp/core/lower_llvm/core.cpp:580-593`, emission at
`platform/cpp/core/lower_llvm/mem.cpp:574-580`): every ordinary (non-transpose)
8-vector LDS load destined to feed WMMA is emitted with the LLVM `volatile`
keyword. `volatile` is opaque to the backend's pattern-matching substitution
pass, forcing the plain sequential `ds_read_b128` instead. Documented as
**zero-cost** ("LDS is sequentially consistent within a wave").

**When the transpose read is actually wanted**, rocKE calls the intrinsic
explicitly instead of relying on backend auto-substitution:
`ds_tr16_b128_spec` selects the element-typed opcode
(`llvm.amdgcn.ds.load.tr16.b128.v8f16` / `.v8bf16`, overloaded on result
type — unlike gfx950's type-agnostic `ds_read_b128_tr_b16` which returns
`<8 x i16>` and needs a bitcast, `backend.py:170-176`), gated per-kernel by
explicit opt-in flags (`use_ds_tr_reads`).

**Decoded lane-mapping semantics** (directly reusable if CK hits the same
instruction): `ds_load_tr16_b128` transposes an 8×8 element block **within
each group of 8 lanes** — the 8 lanes of a group each read 8 contiguous
elements, and lane `j` of the group receives element `j` from all 8 of those
reads (`platform/dsl_docs/instances/convolution.md:403-407`,
`platform/dsl_docs/optimization/optimization_runbook.md:1942-1945`). This is
wave32's 8-elements/lane regime, distinct from gfx950's wave64
`ds_read_tr16_b64` 4-elements/lane regime — a 16-element WMMA fragment on
gfx1250 needs **two** `ds_load_tr16_b128` reads. A worked example combining
two reads + a `ds_bpermute (lane^8)` stitch for the FMHA P·V operand is in
`library/kernels/gfx1250/_wmma_attention_common.py:474-503`. A shipped,
real-hardware-verified conv application (not just attention) is
`platform/cpp/instances/common/conv_implicit_gemm_dgrad.cpp:502-506` /
`conv_implicit_gemm_wgrad.cpp:472-476`, gated on wave size (64 ⇒
`ds_read_b64_tr_b16`, 32 ⇒ `ds_load_tr16_b128`), with the wrong combination
rejected at `validate()` time
(`conv_implicit_gemm_wgrad.cpp:590-593`).

**A hard-won methodology lesson** (`optimization_runbook.md:1859-1949`):
rocKE's own conv K-outer port initially got the WMMA B-operand lane map
*wrong* by reasoning from the gfx950 analogue instead of probing hardware;
the fix was a small per-lane hardware probe kernel (fill LDS with a known
pattern, read back via the intrinsic, decode the permutation) run once before
trusting any transpose-read lane map. Byte-identity between the Python and
C++ lowering engines did **not** catch the wrong lane-map bug, because both
engines emitted the *same wrong* IR — a general lesson that dual-engine
parity tests are not a substitute for hardware-verified numerics.

**Action for CK**: audit every `#if !defined(__gfx125__) && !defined(CK_USE_GFX1250)`-disabled
WMMA GEMM/MX instance config (§5.4 of the CK coverage doc — "tile too small",
"illegal memory access", "numerical issues") for whether it is *actually*
hitting this exact backend auto-substitution hazard on a row-major LDS tile,
rather than a genuine hardware/tile-size limitation. If so, the fix is
either (a) mark the offending LDS loads `volatile` at the point CK emits
them, or (b) verify CK's inline-asm/intrinsic path already forces the
non-transpose opcode. This is the single most actionable, concrete lead in
this investigation.

## 2. `s_wait_asynccnt` vs `s_waitcnt vmcnt` / Split Wait-Counter Model — **PORT**

**CK's gap**: gfx1250 needs `s_wait_asynccnt` handling distinct from gfx950's
`s_waitcnt vmcnt` idiom; CK's own comments in
`include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp` note gfx1250
"has no `global_load_async_to_lds_b96`" and needs workaround synthesis.

**rocKE's finding**: gfx1250 exposes **three distinct counter families**,
matching real AMDGPU GFX12 ISA, and rocKE models each as an explicit
capability rather than an arch-string special case:

1. **Legacy monolithic `s_waitcnt`** — UNAVAILABLE on gfx1250.
   `Gfx1250Backend.emits_legacy_s_waitcnt = False`
   (`platform/python/rocke/core/isa/backend.py:517-521`): "the
   `llvm.amdgcn.s.waitcnt` intrinsic is NOT selectable on gfx1250."
2. **Split VMEM/LDS counters** (`s_wait_loadcnt` / `s_wait_dscnt`) — the
   ordinary traffic-class counters. `Gfx1250Backend.emit_lds_barrier_drain`
   (`backend.py:565-573`) emits `s_wait_loadcnt(0)` (only if `drain_vmem`)
   then always `s_wait_dscnt(0)` before a workgroup barrier — clang fuses
   these into one `s_wait_loadcnt_dscnt 0`.
3. **Dedicated `ASYNCcnt`**, drained by `s_wait_asynccnt` — used **only** for
   the new `global_load_async_to_lds`/`global_store_async_from_lds` family
   (`platform/python/rocke/core/ir.py:3404-3423`): "tracks completion on a
   DEDICATED ASYNCcnt counter — separate from loadcnt and dscnt; gives true
   partial-drain control for an async-DMA ping-pong: issue the next tile's
   copies, then `s_wait_asynccnt(n_next)` to drain the current tile while the
   next tile's copies keep streaming." On every non-gfx1250 backend,
   `s_wait_asynccnt` is a documented no-op — the counter simply doesn't
   exist there (`platform/cpp/include/rocke/lower_llvm_internal.h:192-197`).

Modeled as a capability flag: `has_async_lds_counter`
(`backend.py:158-162,526-528`) — **True only for gfx1250** — rather than an
`if (arch == "gfx1250")` string check scattered through call sites.

**A critical correctness rule rocKE learned the hard way**: a bare
`s_barrier`/named barrier on gfx1250 does **not** auto-insert a pre-barrier
drain — `emit_lds_barrier_drain`'s docstring states: "The raw
`llvm.amdgcn.s.barrier` does NOT get an auto-inserted pre-barrier
`s_wait_dscnt`, so the LDS write→read handoff across the barrier would race
and read stale LDS → NaN." Every LDS-producing region before a barrier must
therefore emit an explicit `s_wait_dscnt(0)` (+ `s_wait_loadcnt(0)` if a
VMEM→LDS chain is in flight). This was **functionally verified on real
gfx1250 hardware** with a two-wave cross-wave LDS producer/consumer handoff
(`platform/python/rocke/examples/gfx1250/isa_features/barrier_verify.py`).

**Confirmed hardware fact, not a CK-only quirk**: rocKE independently
discovered that the gfx9-style `buffer_load_lds`/`global_load_lds`
DirectToLDS intrinsic family is **entirely unselectable on gfx1250** — a
negative-assertion test enforces this
(`library/tests/test_gfx1250_attention.py:281-317`: the compiled LLVM must
contain `llvm.amdgcn.global.load.async.to.lds.b128` +
`llvm.amdgcn.s.wait.asynccnt` and must **not** contain
`raw.ptr.buffer.load.lds`/`global.load.lds`). This corroborates CK's finding
that gfx950's async-copy mechanism cannot be reused verbatim on gfx1250 — it
is a real, hardware-level instruction-family split, not an implementation gap
either project could paper over.

**Action for CK**: replace any code path that assumes a single `vmcnt`-style
counter applies uniformly across archs with a capability query analogous to
`has_async_lds_counter`, and specifically audit CK's `qr_tdm`/gfx1250 FMHA
pipeline for a missing `s_wait_dscnt`/`s_wait_loadcnt` drain immediately
before any workgroup barrier that follows an LDS write — this exact
stale-LDS-read race is a plausible root cause for a ping-pong-style
correctness bug (see §4 below).

## 3. Async DRAM↔LDS Transfer Width Gap (the "b96" issue) — **SHARED GAP**, informative

**CK's gap**: "no native `global_load_async_to_lds_b96`; synthesize a
12-byte async LDS load from a b64+b32 pair."

**rocKE's finding**: rocKE has the **identical restriction** and has **not**
built a b96-synthesis workaround. Both the Python (`core/ir.py:3526-3529`)
and C++ (`platform/cpp/core/ir/ir_flow.cpp:788-790`) width validators enforce
`width_bytes ∈ {1,4,8,16}` for `global_load_async_to_lds` /
`global_store_async_from_lds` — there is no 12-byte case anywhere in rocKE's
model or its test matrix
(`platform/python/rocke/examples/gfx1250/isa_features/async_store_verify.py`
exercises exactly `((0,1),(4,4),(8,8),(16,16))`). rocKE's own
attention V-tile prefetch (`library/kernels/gfx1250/attention_tiled_3d.py:978-1010`)
uses only the 16-byte (b128) width, never 12.

Separately, rocKE's *older* CDNA-style `async_buffer_load_lds` **does**
accept a 3-dword (b96) width, but this is explicitly the gfx9-family
DirectToLDS path that is "NOT selectable on gfx1250"
(`core/ir.py:3511-3524`) — i.e. the one regime that supports b96 is precisely
the regime gfx1250 cannot use, and the regime gfx1250 does use has no b96.

**Verdict**: nothing to port for the b96 gap specifically — treat this as
confirmation the gap is a genuine hardware limitation on gfx1250, not a
CK implementation bug. The transferable *pattern*, though, is rocKE's
discipline of **never emitting an unsupported width** (validate at IR-build
time, fail loudly) rather than attempting synthesis; if CK's `cluster_load`
family doesn't already validate widths this strictly, adding that guard is
worthwhile defensively.

## 4. FMHA `qr_tdm` Ping-Pong Prefetch Bug — **PARTIAL** (no direct fix, but relevant technique inventory)

**CK's gap**: the `qr_tdm` pipeline (CK's gfx1250-specific, LDS-resident-QKV
FMHA pipeline) has a known ping-pong K/V prefetch bug for prefill
seqlen≥2048 with sink masks.

**rocKE's attention architecture is structurally different** and does not
directly resolve this: rocKE has **no shipped TDM-based attention pipeline**
at all. Its gfx1250 attention stack is three hand-written kernel modules
(`library/kernels/gfx1250/{wmma_attention_fwd,attention_tiled_2d,attention_tiled_3d}.py`)
sharing common building blocks
(`library/kernels/gfx1250/_wmma_attention_common.py`), gated by opt-in
dataclass flags rather than named pipeline tiers (`qr`/`qr_hpad`/`qr_tdm`/V3
have no rocKE analogue). Its `has_tdm`/`has_async_global_lds` capability
flags are reserved for a **not-yet-built Phase-3 tier**
(`library/builders/gfx1250/attention/gfx1250_universal_attention_plan.md`).

rocKE **does** test the exact CK-bug-analogous shape — `prefill_big =
([2048], [0], sliding_window=0, sinks=True)` and `prefill_big_sw` (with
sliding window) in
`library/builders/gfx1250/attention/tiled_2d_verify.py` — and reports
"correctness-clean" at that shape. **This is not proof CK's bug class is
fixable the same way**: rocKE's prefill kernel has no async/double-buffered
K/V prefetch loop shipped (`use_dtla_prefetch` is opt-in, unshipped,
single-buffer-only, and explicitly excludes fp8 KV per
`attention_tiled_3d.py`'s `__post_init__` validator) — the bug class CK has
*requires* a double-buffered async prefetch loop to manifest, and rocKE
simply hasn't built one for prefill yet. The comparison is evidence of a
different, simpler, currently-bug-free design, not a fix for CK's design.

**Concrete, portable techniques found** (from
`library/builders/gfx1250/attention/gfx1250_mha_optimization_case_study.md`,
334 lines, and `gfx1250_universal_attention_plan.md`, 460 lines — both read
in full):

1. **Dependent-WMMA co-execution hazard, documented as CK's likely bug
   class.** rocKE independently found and reproduced an "intermittent
   garbage P·V accumulator at high occupancy" bug on gfx1250, attributed to
   the backend not correctly scheduling/waiting between dependent WMMA
   instructions. Three remediation options were ranked: **(1) [preferred,
   not yet built] a proper backend WMMA scheduler that emits correct
   inter-WMMA `s_nop`/wait — this is the "right" fix but doesn't exist yet in
   rocKE or (as far as this investigation found) anywhere; (2) [interim,
   what rocKE actually ships] an empirically-swept `v_nop`-spacing knob
   (`wmma_spacing`, auto-forced ≥1 whenever the DPP-softmax lever is
   enabled) inserted after every QK/PV WMMA
   (`_wmma_attention_common.py::_wmma_spacing()`); (3) [not built] restructure
   to shorten the dependent P·V chain via interleaved accumulators.**
   rocKE's own dense FMHA-fwd bring-up kernel
   (`library/kernels/gfx1250/wmma_attention_fwd.py`) has an **open,
   unresolved** causal-masking NaN bug from this exact hazard class — the
   `v_nop`-count workaround "did not robustly fix the causal NaN across
   shapes," so Phase-1 correctness there falls back to compiling via
   `hipcc`/clang instead of rocKE's direct LLVM path, explicitly deferring
   the real fix to "a proper gfx1250 WMMA scheduler."
   **Actionable for CK**: this dependent-WMMA scheduling hazard is a strong
   candidate root cause for CK's `qr_tdm` ping-pong bug and possibly some of
   CK's disabled WMMA GEMM instance configs (§5.4/§1 above) too — worth
   checking whether CK's generated ISA has back-to-back dependent WMMAs
   without an intervening wait/nop on the code paths that fail.
2. **Online-softmax `-inf`/NaN sentinel guards.** rocKE found and fixed a
   real intermittent bug: a `~1e30` sentinel value leaking into ~128 output
   elements at large batch (`num_seqs=256, kv_len=4096, fp8, seg=16`), root
   caused to insufficiently-guarded softmax merges across split-KV segments.
   Fix: workspace pre-initialization + explicit `fcmp ogt(rowmax, -inf)`
   guards in the online-softmax row-max/row-sum merge
   (`softmax_row_update` in `_wmma_attention_common.py`). Directly
   applicable to any split-KV/segmented CK pipeline doing an online-softmax
   merge on gfx1250.
3. **`binary_search_iters` floor.** A KV-cache paged-sequence-index binary
   search needs a minimum iteration floor (32) to avoid under-iterating at
   large batch — a narrow but concrete edge-case lesson for any paged-KV
   dispatch code.
4. **DPP `row_xmask` VALU softmax reduction** (default-on in production,
   1.4–1.56× speedup at low occupancy): moves the softmax row-reduction off
   the LDS port onto VALU via fused `v_max_num_f32_dpp`/`v_add_f32_dpp`
   emitted as inline asm (plain `row_xmask` movs don't get folded by LLVM's
   `GCNDPPCombine` automatically). Portable in spirit; CK targets wave64
   MFMA where the lane-count arithmetic differs, so this needs re-deriving,
   not copy-pasting.
5. **What rocKE tried and explicitly rejected** (useful negative
   information, avoids wasted CK effort): native fp8 K=64 WMMA P·V atom
   (ceiling-probed at <1% of segment kernel time — not worth building);
   register-resident P via `ds_bpermute` cross-lane gather (**2× slower**,
   not faster, at batched decode — cross-lane traffic tax dominates);
   wide/double-buffered V staging (regresses — occupancy loss);
   multi-wave at large batch (4–10× slower — device already saturated).
6. **Residual hardware ceiling, not fixable in software**: rocKE's own
   conclusion is that gfx1250 attention trails gfx950 by 1.07–4× depending on
   batch size due to **no async global→LDS DMA** (mostly true, see §3),
   **no `ds_read_tr`** (false — see §1, it exists but rocKE found it "neutral,
   not adopted" for attention specifically because the `ds_bpermute`
   lane-stitch tax eats the savings), and **wave32 vs wave64** lane-count.
   CK should not expect to close 100% of any gfx950-vs-gfx1250 attention
   performance gap through software alone.

**Verdict**: no direct fix for CK's specific bug is available from rocKE,
but the dependent-WMMA scheduling hazard (item 1) is the strongest lead —
CK's `qr_tdm` bug is very plausibly the same class of issue rocKE is
fighting with `wmma_spacing`, and CK should specifically inspect whether an
inter-WMMA wait/nop is missing around its K/V prefetch boundary.

## 5. MX/Block-Scale Support on gfx1250 GEMM — **PORT** (design pattern + hardware existence proof)

**CK's gap**: no MX-FP8/FP4 support on gfx1250 anywhere in CK (MX is
gfx950-MFMA-only in CK today); CK's own device-op guard
(`device_gemm_xdl_cshuffle_v3_mx.hpp:692-696`) explicitly rejects gfx1250 for
MX GEMM ["Only gfx950 and gfx1250 architectures support MX GEMMs" — wait,
CK's guard actually *accepts* both `gfx950` and `is_gfx125_supported()`; the
instance-*library* gap is that `gemm_mx`'s only shipped instances are
XDL-shaped and several tile configs are disabled on gfx1250 for correctness
reasons, not that the device op rejects gfx1250 outright].

**rocKE's finding — gfx1250 WMMA hardware genuinely has native low-precision
matrix instructions**:

- **fp8/bf8 16×16×64 WMMA atoms** (all 4 A/B dtype combinations:
  fp8×fp8, fp8×bf8, bf8×fp8, bf8×bf8) — confirmed in
  `platform/python/rocke/core/isa/backend.py:290-350`
  (`_GFX1250_WMMA_FP8` table) and `arch_specs.json:144-165`. This exactly
  matches CK's own documented gfx1250 WMMA tile set
  (`arch_specs_generated.py` in CK's dispatcher).
- **Native block-scaled (MX) WMMA forms, K=128**:
  `wmma.scale.f32.16x16x128.f8f6f4` (E8M0 scale packed as a 32-bit int) and
  a `wmma.scale16` variant (scale packed as 64-bit, 8 E8M0 bytes) — gated on
  `llvm23`/ROCm 7.13+ (`backend.py:627-629`). This is real, hardware-backed
  MX support on gfx1250 WMMA that CK does not currently exploit for its
  gfx1250 instance library.
- Shipped implementation: `platform/python/rocke/instances/gfx1250/block_scaled_gemm.py`
  has **two matrix paths**: a legacy `'wmma'` path (K=64 FP8/BF8 atoms +
  software post-scaling, fp16/fp32 A/B scales applied per group) and native
  `'wmma_scale'`/`'wmma_scale16'` paths (K=128 native FP8 WMMA instructions
  consuming packed E8M0 scale bytes directly, block_k=32 or 16 E8M0 groups).
  **This K=128-native-is-correct / K=64-legacy-is-fallback split is
  structurally identical to CK's own empirical finding** that only
  `warp_tile_k=128` (FlatMM-style, 16×16×128) produces correct fp8/bf8
  results on gfx1250, while `warp_tile_k=32` (MFMA-style) and
  `warp_tile_k=16` (generic-WMMA-style) return wrong/zero results
  (`dispatcher/python/grouped_gemm_abquant_utils.py:865-925` in CK). rocKE
  arrived at the same K=128 conclusion independently and treats it as the
  *intended fast path* (K=64 is a slower, pre-native-instruction
  compatibility fallback, not a broken path rocKE is working around) — this
  is a strong second confirmation that CK's dispatcher-layer K=128 rule is
  correct and CK's own `gemm_mx` static instance library (which is
  XDL-shaped, not WMMA-native) should be extended with a native K=128 WMMA
  MX path analogous to `wmma_scale`/`wmma_scale16`.
- **Explicitly excluded even in rocKE's more mature model**: FP4/FP6/BF6
  block-scaled WMMA forms ("FP4/FP6/BF6 block-scaled forms remain
  intentionally omitted" — `arch_specs.json:145` comment) and packed
  int4/int8 WMMA (`iu8`/`iu4`, present only on the `gfx11-generic` row, not
  gfx1250). So CK should not expect a gfx1250 WMMA path for FP4/FP6/BF6 MX or
  packed-int4 either — see §6.

**Verdict**: port the **K=128-native-scale-is-correct** design pattern (already
independently confirmed by both projects) into CK's static `gemm_mx`
instance library as a proper native WMMA MX path (rather than only
XDL-shaped instances with several gfx1250 tile configs disabled), and treat
rocKE's `block_scaled_gemm.py` + its E8M0-scale-packing logic as a reference
implementation for the WMMA-native MX instruction encoding.

## 6. Packed-Int4 (fp8i4/bf8i4) Quantized GEMM on gfx1250 — **SHARED GAP**

**CK's gap**: packed-int4 quantized GEMM does not compile on gfx1250 (no
gfx12 WMMA/FlatMM int4 instruction path).

**rocKE's finding**: identical, unresolved gap. rocKE's packed-int4 GEMM
builder (`platform/python/rocke/instances/common/_matmul_nbits_common.py:63-65,148-151`)
has `SUPPORTED_ARCHES = frozenset({"gfx1151", "gfx1201"})` — **gfx1250 is
explicitly absent**, and `platform/dsl_docs/instances/index.md:29-31`
confirms: "`matmul_nbits.py` ships `MatMulNBitsSpec` for fp16 activations and
packed-int4 weights with group size 32. Its validator accepts gfx1151 and
gfx1201" (not gfx1250). rocKE's WMMA atom catalog for gfx1250 (§5 above) has
no `iu4`/`iu8` entries at all — this is architecturally the same "no gfx12
WMMA int4 instruction on gfx1250" limitation CK documents.

**Verdict**: nothing to port; confirms this is a genuine, currently-unsolved
hardware/toolchain gap on gfx1250 shared across both projects, not a
CK-specific implementation deficiency. If either project resolves it first,
the fix should be cross-pollinated then.

## 7. 8-Warp CompV3/Intrawave Pipeline Miscompile (ROCm/rocm-libraries#11161) — **PARTIAL** (avoidance pattern, not a fix)

**CK's gap**: an 8-warp block arrangement (`2x4x1`/`4x2x1`) in the
CompV3/intrawave GEMM pipeline miscompiles on gfx1250 (max_rel error
0.14–0.87), tracked as issue #11161.

**rocKE's finding**: rocKE's shared, arch-polymorphic universal-GEMM builder
(`platform/python/rocke/instances/common/gemm_universal.py:499-510`) gates
gfx1250 (like gfx1151) to **only** the `mem` pipeline + `default` epilogue +
a single supported atom `{(16,16,32)}` — explicitly because "the richer
pipelines (compv3/compv4 scheduler interleave, cshuffle LDS-staged C, DTLA,
preshuffle) encode MFMA-shaped assumptions and are gated off until ported."
rocKE therefore **cannot** structurally reach the code path where CK's
#11161 bug lives — it simply hasn't enabled a compv3/compv4-equivalent
scheduling tier for WMMA yet.

The **one** place rocKE did try an 8-warp arrangement was in a bespoke,
hand-written kernel (not the shared/portable builder): `fused_moe_mega_wmma.py`'s
`warp_n=8` (256-thread block) configuration, documented in
`platform/python/rocke/examples/gfx1250/fused_mega_moe/fused_moe_case_study.md`.
This **passed correctness** (max rel ≤1.4e-6) but **regressed performance**
("tile_m=16 can't feed 8 warps") — a different symptom class than CK's
numerical corruption, and not evidence that 8-warp WMMA arrangements are
inherently correctness-safe on gfx1250 (rocKE's kernel and CK's pipeline are
structurally different code).

**Verdict**: the design lesson (not a fix) is rocKE's discipline of **not
enabling a general multi-warp scheduling tier for WMMA until it has been
explicitly ported and verified for that instruction family**, rather than
assuming an MFMA-era scheduling tier generalizes. CK's #11161 bug is
consistent with exactly the class of "MFMA-shaped assumption baked into a
shared scheduler tier, silently wrong on WMMA" that rocKE's gating comment
warns about. Worth checking whether CK's CompV3/intrawave pipeline scheduler
makes any MFMA-specific assumption (issue latency, register-file layout,
dependent-instruction spacing — cf. §4's dependent-WMMA hazard) that doesn't
hold for WMMA.

## 8. Arch-Classification Cleanliness (`is_xdl_supported()` / `is_gfx12_supported()` quirk) — **PORT** (architectural pattern)

**CK's gap**: CK's host capability layer classifies gfx1250 as "XDL
supported" via `is_xdl_supported() = is_gfx12_supported() || is_gfx11_supported()`
(`include/ck/host_utility/device_prop.hpp:109-114`), even though gfx1250's
native instruction set is WMMA, not MFMA/XDL. This is why gfx950 and gfx1250
sometimes share the same XDL C++ template family via
`#if defined(__gfx950__) || defined(__gfx125__)` device branches — a
misclassification that makes it easy to accidentally apply an MFMA-shaped
assumption to a WMMA target (arguably related to §7's bug class).

**rocKE's finding**: `platform/dsl_docs/architecture/multi_arch_data_layout.md`
(58KB, read in full) describes an already-implemented architecture that
avoids this class of confusion **by construction**:

- **Family is an explicit parameter**, never an inferred boolean. Every MMA
  catalog query takes `family="mma"|"wmma"` explicitly:
  `ArchTarget.mma.enumerate(family="wmma", ...)` — there is no derived
  `is_xdl_supported()`-style boolean anywhere that a caller could
  accidentally rely on for an arch it doesn't actually describe.
  `block_scaled_gemm.py` asserts `target.has_wmma and not target.has_mfma`
  for gfx1250 as an explicit validity gate at the point of use.
- **Strict separation of hardware facts from kernel-family policy**:
  `ArchTarget` carries only hardware facts (wave_size, `lds_capacity_bytes`,
  the MMA atom catalog, memory-capability bits like `has_async_lds`/
  `has_ds_read_tr`, waitcnt-encoding variant). Kernel-family *pipeline
  policy* (pipeline-name lists, warp-tile tables, LDS budget factors) is
  explicitly barred from the core arch layer and instead lives in per-family
  `instances/common/<family>_policy.py` modules that *compose* the core
  predicates. Direct quote:
  > "CK Tile C++ shows that pipeline *skeleton* is generic but pipeline
  > *validity and performance* are architecture-bound. That does **not**
  > mean pipeline metadata belongs in core: it means each kernel family must
  > filter its own pipeline space using the hardware facts that core
  > exposes." (`multi_arch_data_layout.md:355-358`)
- Physical operand/accumulator layout is split into three explicit protocols
  (`OperandLayout`, `LdsProducerLayout`, `LdsConsumerLayout`) so register maps
  are target- and MMA-op-selected *data*, not hardcoded per instance file.
- `instances/<gfx>/` subfolders are reserved for genuinely arch-divergent
  *algorithms* (staging strategy, K-loop shape); shared algorithms live in
  `instances/common/` and parametrize purely over `ArchTarget`.

**Verdict**: **port this design pattern**, not any specific code. CK's
device-op layer conflating "is XDL supported" with "is gfx12/gfx11" is the
structural root of needing ad-hoc `#if defined(__gfx950__) ||
defined(__gfx125__)` branches scattered through device-op headers. Replacing
the derived boolean with an explicit `family` parameter threaded through
capability queries — and moving pipeline-tier policy (which pipelines are
valid/enabled per arch, cf. §7) out of the device-op template itself and
into an arch-parametrized policy table — would let CK add or fix gfx1250's
WMMA path without touching gfx950's MFMA path, and vice versa, by
construction rather than by discipline.

## 9. Device-vs-CU-Scope Buffer Atomics (split-K correctness) — **CK IS AHEAD**

**CK's gap (already fixed)**: gfx1250 needs DEVICE-scope (16) rather than
CU-scope (0) coherence for cross-CU buffer atomics; CK's fix is documented in
`include/ck_tile/core/arch/amd_buffer_addressing.hpp:2327-2333` and
`amd_buffer_addressing_builtins.hpp:2393-2400`.

**rocKE's finding**: rocKE has **not** solved this and, by its own
documentation, has explicitly **deferred** it. `Gfx1250Backend`'s class
docstring (`platform/python/rocke/core/isa/backend.py:504-516`) states
verbatim: "the inherited buffer SRD word3 and gfx11 `s_waitcnt` layout are
placeholders adequate for flat-global WMMA GEMM bring-up; the gfx1250 57-bit
SRD and split wait-counter model are deferred." `platform/dsl_docs/development/known_gaps.md:97-100`
repeats this: "the buffer SRD word3 ... are placeholders, exactly as in
Python's `Gfx1250Backend`. The gfx1250 57-bit SRD is deferred." rocKE
currently uses **the RDNA buffer-descriptor word3 (`0x31014000`) as an
unvalidated bring-up placeholder** for gfx1250 — not a validated 57-bit SRD.
No code anywhere in rocKE sets an explicit atomic scope=16/DEVICE bit in the
buffer-descriptor construction for gfx1250, and no comment mentions
"device scope" vs "CU scope" by name. What rocKE *does* model instead is a
related-but-distinct concern: fine/coarse-grained memory-model metadata
(`!amdgpu.no.fine.grained.memory` / `!amdgpu.no.remote.memory`) to make
`atomicrmw fadd` select the native hardware instruction instead of a CAS
loop — not the buffer-descriptor scope bits CK's fix addresses.

**Verdict**: **CK is ahead of rocKE here.** There is nothing to port; if
anything, CK's `amd_buffer_addressing*.hpp` DEVICE-scope fix and its
underlying reasoning would be a useful contribution *back* to rocKE, whose
57-bit-SRD/atomic-scope work is explicitly still a placeholder.

## 10. Real-Hardware CI Coverage — **PORT** (process pattern)

**CK's gap**: CI never runs correctness tests on real gfx1250 hardware
(build-only, via a software HSA emulator on gfx90a nodes — see §9 of
`GFX950_VS_GFX1250_COVERAGE.md`).

**rocKE's finding**: rocKE's dual-backend unification RFC
(`platform/dsl_docs/architecture/dual_backend_unification_rfc.md:331-339`)
documents a concrete multi-arch CI plan already run on real hardware: **"WS15
— multi-arch execution. Run the tiers across gfx942/gfx950/gfx1151/gfx1201/
**gfx1250** on the multi-arch GPU CI SLURM cluster (per-arch nodes)."** Its
ISA-feature verification scripts
(`platform/python/rocke/examples/gfx1250/isa_features/*.py`) are also
launched on real gfx1250 hardware where a safe/deterministic reference
exists (async store, barriers, scalar controls all byte-compare real device
output), not just compiled and disassembled. Its conv K-outer transpose-read
port is explicitly "verified on gfx1250 hardware" with a bitwise-identical
numeric check (`optimization_runbook.md:1859-1949`).

**Verdict**: port the *practice*, not code — CK's Jenkins CI should add a
real-hardware gfx1250 test-execution stage (paralleling gfx950's 5 stages)
once capacity allows, rather than continuing build-only/emulator validation.
rocKE's per-arch-SLURM-node pattern is a reasonable template for how to
structure that.

## 11. TDM / Cluster-Load — **SHARED GAP** (rocKE is actually less mature here than CK)

**CK's gap**: gfx1250-only `cluster_load`/`cluster_load_async_to_lds`
intrinsic family (`include/ck_tile/core/arch/amd_cluster_load.hpp:13-183`)
with a `static_assert(sizeof(T)==0, "cluster_load is only supported on
gfx1250")` else-branch — implying CK already has a *working* gfx1250-only
cluster-load path, distinct from its separate `qr_tdm` correctness bug.

**rocKE's finding**: rocKE exposes the matching low-level IR ops
(`tensor_load_to_lds`/`tensor_store_from_lds`/`s_wait_tensorcnt`, lowering to
`llvm.amdgcn.tensor.load.to.lds` etc.) and confirms these assemble/disassemble
correctly on real gfx1250 hardware
(`platform/python/rocke/examples/gfx1250/isa_features/tdm_verify.py`), **but**
this is compile/ISA-verification only — descriptor groups are dummy zero
vectors, never launched, because "ROCKE does not yet expose construction of
a valid D# global-memory/LDS descriptor"
(`isa_features/README.md`). rocKE's own capability flag `has_tdm: false`
for gfx1250 in `arch_specs.json` confirms rocKE's own model does not
consider TDM production-ready.

**Verdict**: nothing to port — rocKE's TDM support is at an earlier maturity
stage than CK's (CK at least has a working, if buggy, TDM-adjacent pipeline;
rocKE has never launched a real TDM kernel). The only value is confirming
the correct LLVM intrinsic names/`s_wait_tensorcnt` usage, which CK already
has.

## 12. Early-Silicon (A0) Gating — **NOT FOUND in rocKE**

**CK's gap**: `test/ck_tile/gemm/test_gemm_pipeline_util.hpp:443-464`
`GTEST_SKIP`s TDM cluster-launch pipelines and 32-wide WMMA F4 instructions
on gfx1250 A0 (early) silicon revisions.

**rocKE's finding**: searched `known_gaps.md`, `limitations.md`, and the
whole rocKE tree for "A0", "early silicon", "silicon rev", "revision" —
**no hits**. rocKE uses only informal language ("current bring-up
silicon"/"bring-up board") with no programmatic revision-gating mechanism.

**Verdict**: nothing to port; CK's A0-gating mechanism is, if anything, more
mature than rocKE's informal handling of the same concern.

## Summary Table

| # | CK Gap | rocKE Status | Verdict |
|---|---|---|---|
| 1 | LDS transpose-load builtin / disabled WMMA tile configs | Found + fixed root-cause hazard (`blocks_ds_load_tr16`, `volatile` trick) | **PORT** |
| 2 | `s_wait_asynccnt` vs `s_waitcnt vmcnt` | Modeled as explicit 3-tier capability (`has_async_lds_counter`) | **PORT** |
| 3 | b96 async-load gap | Same restriction, no synthesis; validates widths strictly | SHARED GAP |
| 4 | `qr_tdm` K/V ping-pong prefetch bug | No TDM attention shipped; but has dependent-WMMA hazard + `wmma_spacing` workaround, softmax-sentinel fix | PARTIAL (strong lead) |
| 5 | No MX-FP8/FP4 on gfx1250 GEMM | Native fp8/bf8 16×16×64 + K=128 block-scaled WMMA atoms, shipped `block_scaled_gemm.py` | **PORT** |
| 6 | Packed-int4 GEMM unsupported on gfx1250 | Identical gap (`gfx1151`/`gfx1201` only) | SHARED GAP |
| 7 | 8-warp CompV3 miscompile (#11161) | Avoids by not enabling compv3/compv4-equivalent tier on WMMA yet | PARTIAL (design lesson) |
| 8 | `is_xdl_supported()` misclassification | Explicit `family` parameter + hardware-facts/policy separation | **PORT** (pattern) |
| 9 | DEVICE- vs CU-scope atomics | Deferred/placeholder in rocKE; CK already fixed | **CK IS AHEAD** |
| 10 | CI never runs gfx1250 hardware tests | Real per-arch SLURM hardware CI (WS15) | **PORT** (process) |
| 11 | `cluster_load`/TDM | rocKE is less mature (ISA-probe only, never launched) | SHARED GAP (CK ahead) |
| 12 | A0-silicon test skips | No equivalent mechanism in rocKE | N/A |

## Prioritized Action List for CK

1. **Audit disabled gfx1250 WMMA GEMM/MX instance configs for the
   `ds_load_tr16_b128` auto-substitution hazard** (§1) — cheapest, highest-
   confidence lead; a `volatile`-load fix is a one-line change per site if
   this is indeed the root cause.
2. **Model `s_wait_asynccnt` as a capability, not a string-matched special
   case** (§2), and audit `qr_tdm` for a missing `s_wait_dscnt` drain before
   its post-prefetch barrier.
3. **Extend `gemm_mx`'s gfx1250 instances with a native K=128 WMMA
   block-scaled path** analogous to rocKE's `wmma_scale`/`wmma_scale16`
   (§5), replacing/complementing the current XDL-shaped-only instances.
4. **Investigate whether CK's CompV3/intrawave scheduler carries an
   MFMA-specific assumption** that silently breaks on WMMA (§7), given
   rocKE's explicit refusal to enable that scheduling tier for WMMA at all.
5. **Adopt the explicit-`family`-parameter / hardware-facts-vs-policy
   separation pattern** (§8) as a refactor target for CK's `is_xdl_supported()`
   host-classification layer, to prevent future MFMA-assumption leakage into
   gfx1250 code paths.
6. **Contribute CK's DEVICE-scope atomic fix back to rocKE** (§9) — a rare
   case where CK is the more mature reference implementation.
7. Longer-term: push for a real-hardware gfx1250 CI stage (§10), mirroring
   rocKE's per-arch SLURM approach and gfx950's existing 5 CI stages.
