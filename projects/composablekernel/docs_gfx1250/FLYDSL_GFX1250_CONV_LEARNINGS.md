# FlyDSL gfx1250 Learnings for Composable Kernel Convolution

FlyDSL (`/home/sgundabo/FlyDSL`, branch `main`) is a Python-embedded MLIR
compiler for GPU kernel authoring (GEMM, MoE, attention, conv, norm) — a
full compiler stack with custom MLIR dialects (`Fly`/`FlyROCDL`), C++
lowering passes, a Python JIT frontend, and its own LLVM build. Of the four
gfx1250 kernel-authoring projects investigated in this series (MISA —
hand-written assembly; rocKE — Python SSA IR; hipConv — templated HIP C++;
FlyDSL — MLIR compiler), **FlyDSL's ISA-lowering layer
(`lib/Dialect/FlyROCDL/GFX1250/`) is the closest architectural analogue to
CK's own `include/ck/utility/amd_wmma.hpp` / `include/ck_tile/core/arch/mma/wmma/wmma_gfx12.hpp`
ISA-wrapper headers** — both express hardware constraints as compile-time
verifiers (FlyDSL: TableGen + C++ `verify()`; CK: `static_assert`/SFINAE),
making cross-referencing unusually direct.

**Scoping correction (found before delegating)**: FlyDSL's `kernels/conv/`
(conv3d_implicit.py, conv3d_implicit_fp8.py) has **zero** gfx1250
references — FlyDSL's gfx1250 work is entirely in GEMM/MoE/attention, not
convolution. Since CK implements convolution as implicit-GEMM using the same
WMMA/TDM building blocks it uses for plain GEMM, the GEMM/ISA-level findings
below are still directly applicable to CK's gfx1250 conv code.

**Methodology**: 3 parallel research agents (one with full git/bash access,
two without — flagged per-finding below) plus my own direct source
cross-checks against CK.

## 1. The gfx1250 Naming Question — Resolved More Clearly Than Any Prior Investigation

Across this investigation series, three different projects have used three
different names for gfx1250: MISA/rocKE-adjacent sources say "MI400"-ish,
hipConv calls it **"CDNA5"** (`hipconv/src/arch/cdna5/`, mapped to gfx1250 in
CMake), and CK's own `ck_tile` layer gives it a bespoke
`amdgcn_target_family_id::GFX1250` distinct from both GFX9 and GFX12.

**FlyDSL's documentation is the most extensively tabular of any project
investigated** (`docs/architecture_guide.md` §6 "Target hardware",
`CLAUDE.md:172-176`, `README.md:384-385` "Verified Platforms") — and it
**deliberately leaves gfx1250's product-name column blank ("—")** in every
one of these tables, while every other architecture gets an explicit
marketing name (gfx942→MI300A/MI300X, gfx950→MI350/MI355X,
gfx1201→Radeon AI PRO R9700, gfx90a→MI250X). `CLAUDE.md`'s platform table
entry reads verbatim: `gfx1250 | — | 32 | WMMA / TDM | FP8/FP4 GEMM, MoE,
async/TDM copy helpers, 320KB LDS.`

Cross-referencing commit history: `#384` ("Add MI355X/gfx1201 and
**MI450**/gfx1250 to platform docs") shows FlyDSL **did** once write "MI450"
into its docs, followed later by `#781` ("Refer to gfx1250 by arch name
only in docs and comments") which scrubbed it. The most likely reading
`[INFERENCE, source-grounded but commit diffs not directly read by this
agent]`: FlyDSL initially used the internal/rumored codename MI450, then
deliberately reverted to the bare arch name once that codename proved
unconfirmed or wrong — today's blank-name cells are the visible residue.

**Architecturally**, FlyDSL's own MLIR negative-tests
(`tests/mlir/Conversion/wmma_gfx120x_neg.mlir`) explicitly prove gfx1250 and
RDNA4/gfx1201 are enforced as **separate ISA targets** despite sharing a
common v8-operand register ABI: "16x16x32 BF16 is a gfx1250 instruction and
must not be accepted" on a gfx120x (RDNA4) offload target. A skill doc
(`.claude/skills/add-target-atom-op/SKILL.md`) uses "CDNA5" only as a
**hypothetical placeholder name** in a generic tutorial about adding new
backend targets — not a factual classification — and separately, correctly,
classifies gfx1250 as GFX12-lineage (wave32/WMMA/TDM), architecturally
distinct from the CDNA/MFMA/wave64 line (gfx942/gfx950/gfx90a).

**Conclusion for CK's own documentation**: gfx1250 has no settled, universally-agreed
product name across the industry/ecosystem as of this investigation — CK's
own docs should continue treating it as "gfx1250" (arch name only) rather
than adopting any of "MI400"/"MI450"/"CDNA5", consistent with FlyDSL's own
considered, walked-back position after trying "MI450" once.

## 2. My Own Independent CK Cross-Check: Wave-Size Handling (clean, no action needed)

FlyDSL's commit `3d8fbf1d` ("Fix wave size for gfx1250 (wave32, not
wave64)") fixed a real bug where FlyDSL's own ROCDL dialect defaulted to
wave64 for gfx1250, corrupting DPP-based warp reductions/shuffle-mask
widths. FlyDSL's docs (`kernel_tuning_guide.md:414-425`,
`tests/arch_compat.py:9-16`) treat "assumed wave64" as an active landmine
class requiring ongoing vigilance on gfx1250.

**I directly checked CK's own wave-size dispatch** (per the same audit
discipline used in the MISA/hipConv investigations — verify by reading
source, not by assuming):

```cpp
// include/ck_tile/core/arch/arch.hpp:1108-1111
CK_TILE_HOST_DEVICE constexpr index_t get_warp_size()
{
    return static_cast<index_t>(core::arch::get_compiler_target().WAVE_SIZE_ID);
}
```
correctly derives wave size from the per-target `WAVE_SIZE_ID` enum (not
hardcoded), and the legacy layer:
```cpp
// include/ck/utility/get_id.hpp:10-21
__device__ constexpr index_t get_warp_size()
{
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__GFX9__)
    return 64;
#else
    return 32;   // correctly covers gfx1250
#endif
#else
    return 64;
#endif
}
```
also correctly gates on `__GFX9__` (gfx1250 excluded, falls to 32).

**Verdict**: CK does **not** have the wave64-hardcoded-for-gfx1250 bug class
FlyDSL found and fixed in its own dialect. No action needed — included here
as a clean negative-check demonstrating the bug class is real (another
independent project hit it) but CK's core arch layer already handles it
correctly. Worth spot-checking any newer or less-central CK code path that
might independently hardcode 64 (occupancy formulas, DPP/`ds_permute` mask
widths, shuffle widths) rather than calling `get_warp_size()`.

## 3. WMMA Register-Layout and K-Tile-Size Facts — a Direct Challenge to a CK Assumption

FlyDSL's `lib/Dialect/FlyROCDL/GFX1250/MmaAtom.cpp` `verify()` (~line 128)
and `docs/kernel_authoring_guide.md:279-373` document gfx1250's WMMA support
matrix: `f32(K=4)`, `f16/bf16(K=32)`, `fp8/bf8 any E4M3FN/E5M2 mix (K=64 or
K=128, both native OCP fp8)`, `i8(K=64)`, `i4(K=32, sign_a/sign_b/clamp
kwargs)`.

**Critical nuance for CK**: FlyDSL's compiler and documentation state, with
**no correctness caveat**, that **both K=64 and K=128 are valid, correct**
symmetric fp8/bf8 WMMA configurations on gfx1250 — FlyDSL chose K=128 in its
own production kernels (`gemm_a8w4_mxscale_gfx1250.py:53`,
`gemm_a8w8_gfx1250.py:52`) purely for throughput (matching its MX-scale
block granularity — see §4), not because K=64/K=32 are hardware-broken.

This directly contradicts the framing of CK's own documented finding (per
`GFX950_VS_GFX1250_COVERAGE.md` §6.4 / `dispatcher/python/grouped_gemm_abquant_utils.py:865-925`)
that "only `warp_tile_k=128` produces correct results on gfx1250 for
fp8/bf8; `warp_tile_k=32`/`16` return wrong/zero results." **FlyDSL's
independent, hardware-verified compiler ground truth suggests CK's
K=64/K=32 wrong-result behavior is far more likely a CK-side codegen or
scheduling bug than a genuine gfx1250 hardware limitation.** CK should audit
its own K=64/K=32 fp8 WMMA code generation against FlyDSL's register-layout
formula (`getThrValLayoutAB`: `K = block*16 + (lane/16)*8 + within_block`, a
block-of-8 lane-group interleave) rather than continuing to treat K=64/K=32
as categorically unsupported on gfx1250.

**Exception, a genuine hardware constraint**: the *native MX-scaled*
`WMMAScale` instruction (not plain WMMA) is hardware-native only at
`16x16x128` (or `32x16x128`, fp4-only) per `docs/api/dsl.rst:196` — for that
specific scaled-instruction family, K=128 genuinely is the only shape. This
is a real hardware constraint distinct from plain (unscaled) WMMA, where
K=64 is equally valid.

**Signed-integer WMMA, confirmed real** (`MmaAtom.cpp:112-140`,
commit `32ce75ae`): gfx1250 has native `wmma_i32_16x16x64_iu8` /
`wmma_i32_16x16x32_iu4` instructions with explicit `signA`/`signB`/`clamp`
bits. This is a **wholly separate instruction class from atomic-add** — it
does not contradict MISA's independent finding that gfx1250 has no
dedicated signed-integer *atomic-add*; it simply confirms gfx1250 has
first-class signed-integer *matrix* math while still lacking signed atomic
accumulation.

**`modC`/`reuseA`/`reuseB` clarified** (commit `0b92602b`): these are
**not** a hazard workaround — `modC` is a plain C-operand modifier (negate/abs)
and `reuseA`/`reuseB` are literal ISA operand-reuse-cache scheduler hints
forwarded verbatim to the ROCDL WMMA op (`MmaAtom.td:88-96`). No relation to
any operand-liveness hazard.

## 4. MX-Scale (E8M0) Encoding — A Concrete Reference for CK's Missing gfx1250 MX Path

`lib/Dialect/FlyROCDL/GFX1250/MmaAtomScale.cpp:44-56, 250-280` packs E8M0
block-scale bytes into an **i32 integer for block-32 granularity, or an i64
for block-16 granularity**, carried as opaque atom state and passed directly
to the ROCDL `wmma_scale`/`wmma_scale16` intrinsic's `scaleA`/`scaleB`
operands. This is a concrete, directly transferable reference implementation
for CK's currently-nonexistent native gfx1250 MX-FP8/FP4 GEMM path (per
`GFX950_VS_GFX1250_COVERAGE.md` §5, CK has zero MX-FP8/FP4 on gfx1250 today).

**Important scoping correction, itself informative**: despite this MX-scale
lowering infrastructure existing, **FlyDSL has ZERO MXFP8/MXFP4 GEMM
kernels on gfx1250** — `fp4_gemm_4wave.py`, `fp8_gemm_4wave.py`,
`mxfp4_preshuffle.py`, `mxfp8_gemm_8wave.py` are all gfx950/CDNA4-only
(MFMA-based, wave64; `mxfp8_gemm_8wave.py` even asserts
`arch.startswith('gfx950')`). FlyDSL's only gfx1250 quantized GEMM paths are
**A8W4** (fp8-activation × MXFP4-weight, MXScale) and **A8W8** (fp8×fp8, in
raw/MX-blockscale/PTPC variants) — both WMMA-based, neither a "classic"
symmetric MXFP8×MXFP4 GEMM. **This means even a project with full ISA and
compiler control chose not to build a symmetric gfx1250 MXFP4×MXFP8 GEMM,
concentrating that specific combination on gfx950** — suggesting either a
genuine ISA/toolchain maturity gap for that exact configuration on gfx1250,
or a deliberate product-priority choice, rather than a CK-specific omission.
CK's own MX gap on gfx1250 is therefore consistent with, not uniquely behind,
the broader ecosystem for that specific precision combination — but the
A8W4/A8W8 asymmetric-precision path (§4 above, `WMMAScale` at K=128) *is*
mature in FlyDSL and remains directly transferable.

**Two scale-handling bugfixes with a direct analogy to CK's own unexplained
disabled MX instances**:

- `#705` ("Fix A-scale VGPR and optimize decode GEMM"): the bug was
  VGPR-resident A-scale (E8M0) loads assuming a full 32-row scale
  "super-row" when the actual M tail was shorter — a **ragged/non-tile-aligned-M
  interaction producing wrong or out-of-bounds scale values on the last
  partial tile**.
- `#679` ("Make mxscale B-scale preshuffle tile-independent"): the B-scale
  preshuffle addressing silently depended on the specific `tile_n`/`tile_k`
  chosen at compile time — changing GEMM tile shape changed which scale
  bytes a given (N,K) block read, without any explicit error.

**Both bugs share the exact failure signature** ("illegal memory
access"/"numerical issues" that manifest only for particular tile-shape
combinations or ragged M) that CK's own disabled gfx1250 WMMA MX-GEMM
instance configs exhibit (per `GFX950_VS_GFX1250_COVERAGE.md` §5.4). **CK
should specifically audit its own E8M0 scale index/stride math for (a)
unintended tile-shape coupling that should be factored out of the scale
addressing (the `#679` pattern), and (b) M-tail/OOB handling that assumes
full 32-row scale granularity (the `#705` pattern)** as the two most likely
concrete, fixable root causes — rather than treating the disabled configs as
unexplained hardware limitations.

## 5. TDM Prefetch Scheduling: A Generalizable Rule Conditioned on Tile Width

`kernels/gemm/gemm_bf16_gfx1250.py:216-237`, commit `a7d7c4a9` ("Issue tdm
prefetch earlier for wide tiles"):

```
if ks == 0 and prefetch_kt is not None and wmma_m_rep > 1:
    issue(prefetch_kt, ...)   # BEFORE the first K-step's WMMA
# ...
# for wmma_m_rep == 1, the issue happens AFTER the first K-step's WMMA instead
```

**Generalizable rule**: place the next-tile TDM-prefetch issue point in the
K-step unroll based on how much independent WMMA compute follows it. Wide
tiles (`wmma_m_rep > 1`, multiple WMMA M-repeats per wave) have enough
subsequent MMA issue slots to fully absorb the TDM instruction's front-end
issue cost if prefetch is issued *before* the first K-step's MMAs; narrow
tiles (`wmma_m_rep == 1`) have only one MMA group per K-step, so the
prefetch must be issued *after* it, letting the MMA's own latency (not the
prefetch's issue overhead) sit on the critical path instead. **Directly
transferable to CK's own gfx1250 WMMA GEMM/conv pipelines**: condition
prefetch-issue placement on tile width in WMMA units, not a fixed
loop-top/loop-bottom position.

Related shared-helper pattern (`gemm_common_gfx1250.py`):
`pipeline_fence()` fuses `tensor_wait()` (issues `s_wait_tensorcnt`) with a
workgroup barrier; a split variant (`pipeline_fence_signal()`/`_wait()`
using `s_barrier_signal(-1)`/`s_barrier_wait(-1)`) lets the TDM-wait latency
be issued early and overlapped with compute before the actual barrier wait
resolves — a software-pipelining technique for hiding TDM completion
latency behind independent compute, directly comparable to the split-barrier
techniques found in the rocKE and hipConv investigations.

**Confirms and sharpens a prior finding**: `pipeline_fence()` explicitly
issues `tensor_wait()` (the TDM tensorcnt wait) **then a separate**
workgroup/cluster barrier — i.e. the TDM counter alone does not stand in
for cross-wave LDS visibility and must always be paired with an actual
barrier. This corroborates, with FlyDSL's own lowering-level ground truth,
the prior investigations' finding that "TDM completion doesn't cross-order
against LDS-store completion" — here solved via an explicit paired barrier
rather than a separate `s_wait_dscnt`.

**A sharpened, compiler-enforced ISA fact** (`.claude/skills/flydsl-tile-programming/SKILL.md:373-394`):
gfx1250 is **not in the arch-dispatch list** for the legacy positional
`s_waitcnt(vmcnt=, lgkmcnt=, expcnt=)` form at all — that form dispatches
only to gfx942/gfx950/gfx11xx/gfx120x. gfx1250 kernels **must** use the
split wait counters (`s_wait_loadcnt`, `s_wait_storecnt`, `s_wait_dscnt`,
`s_wait_tensorcnt`) exclusively, because the raw positional waitcnt bitfield
packing differs per architecture (CDNA3: `lgkmcnt<<8|expcnt<<4`; RDNA3:
`vmcnt<<10|lgkmcnt<<4|expcnt`) and a shared literal would silently produce
different waits on different targets. This confirms gfx1250's split-counter
scheme is a genuinely new ISA feature, compiler-enforced as mandatory, not
a convention any of these projects merely chose to follow.

## 6. Register-Pressure Management for Wide (256-per-side) WMMA Tiles: A Different Technique Than MISA's

FlyDSL's quantized (fp8/fp4) gfx1250 kernels **do** successfully use
256×256 WMMA tiles in production, tested and merged (`tests/kernels/test_gemm_fp8fp4_gfx1250.py:220-227,323-331,434-438`:
A8W4 at M=256,N=256,K=512,tile=256×256×256; PTPC/blockscale fp8 at
256×256×128), contrasting with **MISA's finding that 256-per-side WMMA tiles
via VGPR-MSB register-bank-doubling were a consistent net performance loss
at every scale tested** (§4.2 of `MISA_GFX1250_CONV_LEARNINGS.md`).

**The mechanisms are fundamentally different, which likely explains the
different outcome**: FlyDSL (`gemm_a8w8_gfx1250.py`'s
`use_quadrant`/`compute_ktile_quad`, `gemm_a8w4_mxscale_gfx1250.py`'s
`_FRONT`/`_BACK` M-row split) uses a **temporal/streaming decomposition** —
splitting the accumulator M/N grid into halves or quadrants and computing
one at a time, **never materializing more than one quadrant's live A/B
register fragments simultaneously** — rather than MISA's approach of
bit-packing two logical accumulator values into the high/low halves of one
physical VGPR bank (`S_SET_VGPR_MSB`) to double density within a fixed
budget. Origin commit `69c82298` ("optimize FP4 GEMM for large tile
256x256"): "use single LDS arena to reduce epilogue usage ... FP4
large-tile-specific optimization to reduce vgpr pressure (x2) ... fix wave
specific tdm bugs."

**Caveat, important for interpreting the contrast correctly**: this
256×256 win in FlyDSL is specific to its **quantized (fp8/fp4) kernels**,
where narrow-precision operands already relieve register pressure relative
to bf16/fp16; FlyDSL's plain bf16 kernel test suite
(`test_gemm_bf16_gfx1250.py`) shows no equivalent 256×256×256 case as of
this investigation. **This may not contradict MISA's finding at all** — MISA's
negative result was likely scoped to plain (non-quantized) bf16/fp16 WMMA,
where FlyDSL has not (yet) attempted or published a 256-tile result either.
**For CK**: if CK ever considers large (256-per-side) WMMA tiles on
gfx1250 for its own quantized (fp8/fp4/MX) instance paths specifically, the
quadrant/temporal-split register-liveness technique is a real, tested,
merged alternative to VGPR-bank-doubling worth evaluating — but this is not
evidence that a 256-tile plain-bf16 conv/GEMM kernel would perform any
differently on gfx1250 than MISA already found.

## 7. Non-Tile-Aligned Shapes and Split-K-Conditioned Store Paths (directly relevant to CK's wrw/small-shape gaps)

Commit `5e97cfc8` ("Add PTPC FP8/A8W4, non-tile-aligned M, and strided A/C
support"): uses gfx1250's **TDM engine's native per-dimension OOB-clip
field** (`tensor_dim1`/`oob_outer_bound`, visible as `mn_oob = M - blk_m`
passed as the TDM atom's "outer" bound in `gemm_bf16_gfx1250.py` and
`gemm_a8w8_gfx1250.py`) to clip A/scale-A loads to valid M rows **with no
host-side padding required**. This is a hardware-native mechanism, distinct
from software EXEC-masking.

Store-path branching, with a **measured regression from doing it the "safe"
way unconditionally**: commit body states verbatim: *"A whole-output buffer
clip regressed aligned production prefill by +15%..+82%, while `tdm_tail`
stays within ~2% of the no-clip path, so a static buffer default was
wrong."* The surviving design branches the output-store path specifically
on split-K: a fast TDM tail-store for aligned tiles / non-split-K, and a
predicated atomic buffer store (needed because TDM itself cannot atomically
accumulate) only when `split_k > 1` forces it. The final state: **"Remove
`m_oob_clip` flag: non-tile-aligned M is now the default GEMM path"** — not
an opt-in edge case, the unconditional default.

**Relevance to CK's wrw/small-shape gaps** (per `GFX950_VS_GFX1250_COVERAGE.md`'s
documented wrw small-output-channel and grouped-conv large-tensor gaps):
(a) if CK's gfx1250 WMMA path uses TDM for global loads, using the TDM
engine's native OOB-clip field for irregular conv/wrw M-dimension shapes
would avoid host-side padding/masking machinery entirely; (b) CK should
specifically branch its own output-store path on split-K presence — a fast
tile-aligned/near-aligned store when `split_k==1`, falling back to a
predicated atomic path only when split-K genuinely requires it — rather
than applying one generic OOB-safe path unconditionally, given FlyDSL's own
measured 15–82% cost from doing exactly that.

## 8. Scale-Handling Design Split: LDS-Staged vs. LDS-Bypassed

Commit `07668a6f` and the surviving `gemm_a8w8_gfx1250.py` code establish a
clean design principle for quantization-scale handling that bifurcates on
**whether the scale participates in the WMMA instruction itself**:

- **E8M0 MX-blockscale** (32-row granular, consumed by the `WMMAScale`
  opcode as operand state) — must be staged through LDS like the A/B
  operands, because the hardware instruction itself consumes it.
- **PTPC** (per-row/per-column, plain fp32 scalar, applied as a
  post-accumulation multiply) — never touches LDS at all. It is
  `buffer_load`'d directly to VGPR late in the K-loop drain
  (`issue_ptpc_scale_loads`) and applied as a plain fp32 multiply in the
  epilogue (`epilogue_apply_ptpc_scale`), fully decoupled from the
  TDM/barrier-synchronized main-loop pipeline.

Commit `07668a6f`'s broader scheduling-experiment log also records that
FlyDSL evaluated both LLVM's automatic `iglp_opt`/`MFMASmallGemmOpt`
instruction-interleaving pass and a hand-emitted `sched_group_barrier`
DS:MFMA-ratio template (default `8:8`) as opt-in alternatives to its default
schedule, and **kept neither as the default** — both were left available
but not promoted, implying neither beat the hand-scheduled
`sched_dsrd`/`sched_mfma`/`sched_barrier(0)` sequence already shipped. This
is a fourth independent data point (alongside MISA's, rocKE's, and CK's own
`#if 0`'d WMMA-v3 scheduler discussed in `MISA_GFX1250_CONV_LEARNINGS.md`
§1.5) that **automatic or generically-templated instruction interleaving
tends not to beat a hand-tuned baseline on gfx1250 WMMA** — reinforcing that
CK should A/B test any interleaving scheduler it has, rather than assume it
helps.

**Relevance to CK**: any CK per-tensor/per-channel scale WMMA path should
consider the same LDS-bypass for broadcast-scalar scales that never need to
be a WMMA operand — buffer-load straight to VGPR, apply as a late multiply,
and keep it off the TDM/LDS/barrier-synchronized critical path.

## 9. Cluster/Multicast TDM: A Real Mechanism, Not Yet Safe to Copy As-Is

`lib/Dialect/FlyROCDL/GFX1250/CopyAtom.cpp:429-460` +
`kernels/common/gfx1250_cluster.py`: each TDM descriptor carries a 16-bit
`workgroup_mask` field, OR'd into bits [15:0] of the hardware descriptor's
GROUP1 config word — the literal hardware multicast bitmask selecting which
workgroups in a launch cluster receive a shared load. Masks are derived from
the flat in-cluster workgroup ID (`local_x + local_y*cluster_m`), with an
A-tile mask spanning workgroups sharing an M-row and a B-tile mask spanning
workgroups sharing an N-column — directly mirroring how a GEMM tile is
shared across a 2D cluster, and citing "gfx1250 Shader Programming, TTMP6
layout, section 3.5.5.1" as the authoritative source.

**Important caveat — this is not yet a mature, safe-to-copy pattern**:
FlyDSL's own end-to-end cluster+multicast+WMMA-GEMM test is **explicitly
skipped** (`tests/unit/test_cluster_mcast_gemm_gfx1250.py:48`: *"Hangs
during JIT compilation with cluster params — deferred to another PR"*), and
a separate test file notes a single wide (16-lane) cluster dimension is
"known to wedge the queue on gfx1250." No committed bandwidth numbers exist
for the multicast benefit either (`tests/perf/bench_tdm_bandwidth_gfx1250.py`
has a multicast mode but ships no logged results). **CK relevance**: if CK
ever considers cluster-launch/TDM-multicast for gfx1250, treat this as an
active, unresolved risk area even in a project with full compiler control —
not a proven, ready-to-adopt technique yet.

## 10. Additional Corroborated Hardware Facts and Defensive Patterns

- **TDM OOB and LDS-padding fields are native hardware descriptor bits, not
  software emulation** (`CopyAtom.cpp:24-46`): per-dimension `extent`
  fields implement hardware OOB zero-fill/clamping directly; padding is a
  dedicated bitfield (`encoded_interval`, `encoded_amount`) applied to LDS
  stride for both load and store directions. FlyDSL's code comments
  explicitly cross-reference a "Triton reference" implementation encoding
  the same fields identically — **a fourth independent confirmation
  (MISA/hipConv/FlyDSL/Triton) that this is a real, stable hardware
  contract**, not a project-specific abstraction choice.
- **TDM padding must be dword-aligned and a power of two, or FlyDSL emits a
  hard compile error** rather than silently producing a wrong hardware
  bitfield (`CopyAtom.cpp:33-66`). **CK should verify its own TDM/cluster_load
  descriptor-construction code rejects (rather than silently mis-encodes) an
  invalid pad interval/amount** the same way.
- **Explicit sentinel-vs-zero disambiguation** (`CopyAtom.cpp:88-92`): a
  large negative sentinel (`0x80000000`) distinguishes an "unset" outer
  stride field from a legitimate stride-0 broadcast, avoiding an aliasing
  bug. Worth checking any CK code that uses `0` as both a valid stride and
  an "unset" sentinel for the same field.
- **Explicit WAR/RAW barrier-pairing comments** in FlyDSL's MoE GEMM
  pipelines (`gemm2.py:261-264`): `gpu.barrier()  # WAR: all waves finished
  reading the prior tile's A-LDS` paired with a later `gpu.barrier()  # RAW:
  A-LDS write visible before the read below`. A good documentation
  discipline CK's own WMMA/TDM pipeline code could adopt for hazard clarity.
- **Gap, not a contradiction**: no FlyDSL source describes the specific
  narrow-D-vs-C WMMA WAR hazard (the LLVM `GCNHazardRecognizer::hasWMMAToVALURegOverlap`
  bug hipConv found and worked around, per `HIPCONV_GFX1250_CONV_LEARNINGS.md`
  §1.1) — searched for "WAR", "read-after-write", "LLVM bug", "miscompile"
  in FlyDSL's kernels/ and lib/Dialect/FlyROCDL/GFX1250, found only generic
  ping-pong-buffer WAR/RAW comments, not that specific hazard class. This
  should be read as **absence of evidence, not evidence of absence** — flag
  as `[UNVERIFIED whether FlyDSL is exposed]`, not as contradicting
  hipConv's finding.

## 11. Testing/CI Status: Same Gap as CK, hipConv, and MISA

FlyDSL's GitHub Actions runner matrix
(`.github/workflows/{flydsl,ci,flydsl-atom-integration}.yaml`) includes only
`linux-flydsl-mi325-1`, `linux-flydsl-mi355-1/8`, `linux-flydsl-mi35x-1`,
`linux-flydsl-navi-2`, and `build-only-flydsl` — **no runner name containing
"1250", "mi450", or "mi400" appears anywhere**. FlyDSL has **no automated
real-hardware CI for gfx1250**, matching (not exceeding) CK's own documented
gap.

More strikingly, one of FlyDSL's own gfx1250 test files admits being
authored hardware-blind: `tests/kernels/test_gfx1250_atoms_device.py:5-20`
is explicitly headed `"!!! UNVALIDATED SCAFFOLD !!!"` and states it was
"authored on a non-gfx1250 host (MI300/gfx942)... hardware-specific fragment
layouts and TDM async fencing have not been confirmed on real gfx1250
silicon... A gfx1250 owner must run and, if needed, correct these before
treating them as regression coverage." **This is direct, self-admitted
evidence that FlyDSL's gfx1250 development happens partly without gfx1250
hardware access** — the same access-scarcity problem implicitly affecting
CK's own gfx1250 test coverage and CI gap (per `GFX950_VS_GFX1250_COVERAGE.md`
§9), now confirmed as an industry-wide, not CK-specific, constraint.

**Testing philosophy worth adopting regardless**: FlyDSL runs a two-tier
test split — MLIR-level FileCheck lit tests
(`tests/mlir/Conversion/{tdm,wmma,mma_scale}_gfx1250*.mlir`) verify the
*ISA-encoding contract* (no GPU needed: pins exact expected
`rocdl.wmma.scale.f32.16x16x128.f8f6f4` / `rocdl.tensor.load.to.lds`
lowering output), fully separate from GPU-executed pytest kernel tests that
verify *numerical correctness*. CK's C++ header-only library has no
MLIR-IR-equivalent static layer to FileCheck, but the underlying
**principle — verify template-instantiation/constraint legality
independently from end-to-end numerical correctness** — maps to "CK should
have compile-time tests asserting which WMMA/TDM template instantiations
are legal on gfx1250 (mirroring FlyDSL's negative tests like
`wmma_gfx120x_neg.mlir`), kept separate from device numerical tests."

## Summary Table

| # | Finding | Source | CK Actionability |
|---|---|---|---|
| 1 | gfx1250 has no settled product name anywhere (FlyDSL tried "MI450", walked it back) | `architecture_guide.md`, `CLAUDE.md`, commits `#384`/`#781` | Informational — keep calling it "gfx1250" in CK docs |
| 2 | CK's own wave-size dispatch is already correct for gfx1250 | Direct CK source read, cross-checked against FlyDSL's own fixed bug | No action — confirms CK is clean here |
| 3 | K=64 fp8/bf8 WMMA is hardware-correct per FlyDSL's compiler; only K=128 for the *scaled* instruction is a real constraint | `MmaAtom.cpp` `verify()`, `kernel_authoring_guide.md` | **High** — re-investigate CK's own K=64/32 "wrong result" finding as likely a CK-side bug, not hardware |
| 4 | MX-scale (E8M0) packing into i32/i64, direct ROCDL intrinsic reference | `MmaAtomScale.cpp:44-56,250-280` | **High** — concrete template for CK's missing gfx1250 MX-FP8/FP4 path |
| 5 | FlyDSL itself has zero symmetric MXFP8×MXFP4 GEMM on gfx1250 | `fp4_gemm_4wave.py` etc. are gfx950-only | Informational — this specific gap may be ecosystem-wide, not CK-unique |
| 6 | Two scale bugs (#705 ragged-M VGPR OOB, #679 tile-shape-coupled preshuffle addressing) match CK's disabled-instance failure signature | commits `0b248798`, `5d4f7727` | **High** — concrete root-cause hypotheses for CK's own disabled MX configs |
| 7 | TDM prefetch-issue timing should condition on tile width (`wmma_m_rep`) | `gemm_bf16_gfx1250.py:216-237`, commit `a7d7c4a9` | **High** — directly transferable scheduling rule |
| 8 | 256×256 WMMA tiles work via temporal/quadrant register-liveness split (quantized kernels only) | `gemm_a8w8_gfx1250.py`, commit `69c82298` | Medium — alternative to MISA's failed VGPR-banking approach, scoped to quantized paths |
| 9 | TDM native OOB-clip field replaces host-side M-padding; split-K-conditioned store path avoids 15-82% regression | commit `5e97cfc8` | **High** — directly relevant to CK's wrw/small-shape gaps |
| 10 | PTPC scale bypasses LDS entirely (buffer_load→VGPR→late multiply) vs MX-blockscale must stage through LDS | commit `07668a6f`, `gemm_a8w8_gfx1250.py` | Medium — design principle for CK's own scale-handling paths |
| 11 | Automatic/templated instruction interleaving doesn't beat hand-tuned scheduling on gfx1250 (4th independent confirmation) | commit `07668a6f` | Reinforces: A/B test CK's own WMMA interleave scheduler |
| 12 | Cluster/multicast TDM is real but has an open, unresolved compiler hang in FlyDSL itself | `CopyAtom.cpp`, skipped test `test_cluster_mcast_gemm_gfx1250.py` | Caution — not yet a safe pattern to copy |
| 13 | TDM OOB/pad fields are hardware-native, 4-project-confirmed (MISA/hipConv/FlyDSL/Triton) | `CopyAtom.cpp:24-66` | Confirms hardware contract; verify CK rejects invalid encodings the same way |
| 14 | No real-hardware gfx1250 CI in FlyDSL either; one gfx1250 test admits being hardware-blind | `.github/workflows/`, `test_gfx1250_atoms_device.py` | Confirms industry-wide gap, not CK-unique |

## Prioritized Recommendations for CK

1. **(Highest priority, re-scopes an existing finding) Re-investigate CK's
   "only `warp_tile_k=128` works" finding as a probable CK-side codegen bug,
   not a gfx1250 hardware limitation** (§3) — FlyDSL's own compiler treats
   K=64 as fully correct and uses it in production elsewhere; audit CK's
   K=64/K=32 fp8 WMMA lowering against FlyDSL's register-layout formula.
2. **(High value) Audit CK's disabled gfx1250 MX-GEMM instance configs for
   the two specific bug patterns FlyDSL found and fixed** (§4): tile-shape-coupled
   scale-preshuffle addressing, and ragged-M-tail scale VGPR loads assuming
   full block granularity.
3. **(High value) Adopt tile-width-conditioned TDM prefetch scheduling**
   (§5) in CK's own gfx1250 WMMA main loop.
4. **(High value, closes a documented gap) Implement the E8M0-in-i32/i64
   MX-scale encoding pattern** (§4) as a starting point for CK's own native
   gfx1250 MX-FP8/FP4 GEMM support.
5. **(High value, directly addresses documented gaps) Use gfx1250 TDM's
   native per-dimension OOB-clip field instead of host-side padding for
   CK's wrw/small-shape irregular M dimensions, and branch the output-store
   path on split-K presence** rather than applying one generic safe path
   unconditionally (§7).
6. **(Medium) Adopt LDS-bypass for broadcast-scalar (PTPC-style)
   quantization scales** in any CK per-tensor/per-channel scale WMMA path
   (§8).
7. **(Caution, not yet actionable) Do not adopt cluster-launch/TDM-multicast
   for gfx1250 as a mature pattern** — FlyDSL itself has an open, unresolved
   compiler hang in this exact combination (§9).
8. **(Process) Adopt the principle of static/compile-time legality tests
   for WMMA/TDM template instantiations, separate from device numerical
   tests** (§11) — mirroring FlyDSL's MLIR FileCheck / GPU-pytest split
   within CK's existing C++ test infrastructure.
