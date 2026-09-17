# MISA gfx1250 Bring-Up: Learnings for Composable Kernel Convolution

MISA (`/home/sgundabo/MISA`, README calls it "iGEMMgen") is a Python-driven,
hand-authored **AMDGPU assembly** code generator for implicit-GEMM convolution
kernels (fwd / bwd-data / bwd-weight, used by MIOpen). Its branch
`users/SreecharanGundaboluAMD/gfx1250_bringup` (226 commits ahead of
`develop`) is a from-scratch WMMA bring-up for gfx1250, with 60+ `docs/*.md`
files recording phase-by-phase hardware discoveries, correctness bugs, and
performance experiments — all against **real gfx1250 silicon**.

This document extracts what is relevant to **CK's own gfx1250 grouped-conv
WMMA code paths** (`device_grouped_conv_{fwd,bwd_data,bwd_weight}_*_wmma_cshuffle_v3*`,
`gridwise_gemm_wmma_cshuffle_v3*`, `blockwise_gemm_pipeline_wmmaops*`,
`epilogue_cshuffle_v3_wmma*`), since CK's convolution correctness and
performance on gfx1250 is the objective.

**Methodology note**: per explicit instruction, this document does **not**
rely on the two pre-existing MISA-authored files that already compare MISA
against CK (`/home/sgundabo/MISA/docs/gfx1250_ck_deep_dive.md`,
`/home/sgundabo/rocm-libraries/gfx1250_ck_audit.md`). Findings below come from
(a) MISA's other raw docs/commits, read independently, and (b) my **own**
direct audit of CK's source — including real `--offload-arch=gfx1250`
compiles with `/opt/rocm/lib/llvm/bin/clang++` (AMD clang 23, ROCm 7.x
toolchain present on this workstation) and disassembly inspection, not
inference from documentation. Every CK-side claim below is backed by either
a compile/disassembly result or an exact file:line citation I read directly.

## 1. My Own Independent CK Audit (real compiles + direct source reads)

### 1.1 The `gfx12`-regex XDL/WMMA CMake overlap degrades gracefully for at least one device-op family — more nuanced than "broken build"

CK's build gates XDL instance compilation with
(`library/src/tensor_operation_instance/gpu/CMakeLists.txt:75`):
```
if(((NOT INST_TARGETS MATCHES "gfx9" AND NOT INST_TARGETS MATCHES "gfx11"
     AND NOT INST_TARGETS MATCHES "gfx12") OR FORCE_DISABLE_XDL) AND source_name MATCHES "_xdl")
```
and the root `CMakeLists.txt:446` sets `CK_USE_XDL` via
`SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx11|gfx12"` — both regexes match
`gfx1250` (substring `gfx12`), so `_xdl`-suffixed instances are compiled for
gfx1250 builds, and the device-op bodies themselves are gated
`#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)`
(`device_grouped_conv_bwd_weight_xdl_cshuffle.hpp:68`, confirmed by direct
read) — which also includes gfx1250.

**I compiled this directly** rather than assume it breaks:
```
/opt/rocm/lib/llvm/bin/clang++ -x hip -c \
  library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_weight/nhwgc/xdl/nhwgc_gkyxc_nhwgk/device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_bf16_instance.cpp \
  -I include -I library/include --offload-arch=gfx1250 --offload-device-only \
  -DCK_USE_XDL -std=c++17 -o /tmp/test_xdl_gfx1250.o
```
Result: **compiles cleanly, 2.2MB object, and the emitted kernel
(`kernel_batched_gemm_xdlops_bwd_weight<GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_bwd_weight<...>>`)
disassembles to 1440 `v_wmma_*` instructions and ZERO `v_mfma_*`
instructions.** This confirms CK's `include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`
abstraction (the "XdlopsGemm" warp-level GEMM primitive used by every
`_xdl`-named device op) is itself arch-polymorphic: on gfx1250 it silently
routes to native WMMA instructions rather than emitting MFMA. **The
`gfx12`-regex inclusion of gfx1250 in the "XDL" instance bucket is, for this
device-op family at least, not a build-breaking bug — it's the mechanism by
which gfx1250 gets a (older, non-`_wmma`-suffixed) WMMA-backed kernel at all.**

Caveat: I also tried compiling an instance using the newer
`DeviceGroupedConvBwdWeight_Xdl_CShuffleV3` class (the one carrying the
`LargeTensors`/`is_gfx125_supported()` reject discussed in
`GFX950_VS_GFX1250_COVERAGE.md` §5.5) via the only real instance file using it
(`..._bf16_pipev1_instance.cpp`) — that compiled to a **7.5KB object with zero
kernel symbols at all** (no WMMA, no MFMA — just a generic utility function).
This needs further investigation before drawing a conclusion (likely an
empty-instance-tuple SFINAE outcome for this specific pipeline/precision
combination, unrelated to gfx1250) — flagged as `[UNVERIFIED]`, not asserted
as a bug.

**Actionable for CK**: don't assume every `_xdl`-named instance is
MFMA-exclusive or broken on gfx1250 — some device-op families silently
degrade to WMMA via `xdlops_gemm.hpp`'s per-arch dispatch. Before "fixing" the
CMake regex to exclude gfx1250 from the XDL bucket wholesale, verify which
specific `_xdl` instance files/pipelines actually produce MFMA vs WMMA vs
empty output for gfx1250, since collapsing the regex could silently remove a
currently-functioning code path.

### 1.2 CK already implements gfx1250's DEVICE-scope atomic requirement — but MISA independently found a *different* atomic instruction needs *SYSTEM* scope

Direct read, `include/ck/utility/amd_buffer_addressing.hpp:600-605`:
```cpp
#if defined(__gfx125__)
    // gfx1250 requires DEVICE scope for cross-CU buffer atomics; CU scope is sufficient elsewhere.
    constexpr int coherence_flag = static_cast<int>(AmdBufferCoherenceEnum::DEVICE);
#else
    constexpr int coherence_flag = static_cast<int>(AmdBufferCoherenceEnum::DefaultCoherence);
#endif
```
applied to `float`/`half_t`/`int32_t` (and, gfx1250-only, `double`)
`raw_buffer_atomic_add`. Per `include/ck/utility/amd_buffer_coherence.hpp:24-27`
(gfx12 branch), scope encodings are `CU=0, SE=8, DEVICE=16, SYSTEM=24`. This
is CK's fix for exactly the class of hazard grouped-conv `wrw`'s split-K
atomic-accumulate epilogue depends on
(`device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp` uses
`InMemoryDataOperationEnum::AtomicAdd` when `KBatch>1`, confirmed by direct
read of lines 918-969).

**MISA independently discovered the same class of hardware requirement**
(`docs/gfx1250_streamk_design.md`, cross-referenced via `PerfTechniques`
agent): "`global_atomic_add_f32`, scope:SCOPE_SYS load-bearing — bare atomics
silently drop cross-CU updates on this HW" (found/fixed Phase 17). This is
strong **third-party, independently-derived confirmation** that gfx1250
cross-CU atomics need wider-than-CU scope — it is a real silicon fact, not
CK-specific caution.

**Open verification item, not a confirmed bug**: MISA's finding was about
`global_atomic_add_f32` (a flat/global atomic) requiring **SYSTEM** (24)
scope; CK's fix is on `raw_buffer_atomic_add` (a buffer-resource-based
atomic, a different instruction class) using **DEVICE** (16) scope. These
may have different scope requirements on the same hardware (buffer atomics
route through the buffer descriptor's cache-coherence path, global atomics
through the L2 directly) — but this divergence (16 vs 24) is exactly the
kind of discrepancy worth testing directly: **run CK's wrw split-K on a
large-CU-count gfx1250 config with an atomic-heavy shape (small M/N, large
K, bf16/fp32 output) and verify no dropped cross-CU updates at high split-K
counts**, since MISA's finding suggests DEVICE scope might not always be
wide enough depending on instruction class.

### 1.3 CK's WMMA epilogue has zero LDS bank-conflict padding — MISA's single largest measured performance lever

Direct read, `include/ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_wmma_base.hpp`:
**zero occurrences** of any pad/bank-conflict-avoidance mechanism (grepped for
`Pad|pad|BankConflict|bank` — no matches). CK's WMMA epilogue does the
standard per-thread-accumulator → LDS (unpadded, tile-linear) → barrier →
vectorized LDS read → global store sequence with no row padding.

MISA independently found, and quantified, **LDS row padding for
bank-conflict avoidance as the single largest performance lever in its
entire gfx1250 bring-up** (`docs/gfx1250_w3_transposed_pitch_sweep.md`,
`docs/gfx1250_w4_tile_size_comparison.md`): because `macro_tile_n` is
typically a power-of-2 multiple of 64 and gfx1250 LDS is 64 banks × 4 bytes,
the unpadded tile-linear address `(row*macro_tile_n+col)*4` collapses
`bank = col mod 64` identically for every row — a **maximal 64-way bank
conflict**, not merely a 2-way one. Adding byte padding to break the
power-of-2 stride (`lds_row_pad=16`, giving a conflict-free stride via
`gcd(stride_dwords, 64) == 4`) measured **+6.4% to +70.2%** across bwd/wrw
shapes (wrw benefits most since more operands are LDS-staged), and combined
with 128×128 tiling specifically, **+41% to +98% relative TFLOP/s** — 4-9×
larger than the next-best lever MISA found (cross-tile LDS double-buffering,
+10-20%). MISA's own recommendation: `lds_row_pad=16`, never unpadded, for
every transposed-operand LDS layout.

**Actionable for CK**: this is the highest-confidence, highest-value,
directly-quantified recommendation in this document. CK's WMMA epilogue
(`epilogue_cshuffle_v3_wmma_base.hpp`) and any WMMA main-loop LDS staging
for transposed operands (bwd's weight-B operand, wrw's grad_output-A /
input-B operands — the layouts CK stages via LDS for grouped conv) should be
audited for the same power-of-2-row-stride bank-conflict pattern and, if
present, given row padding. Given MISA's magnitude (up to 70-98%), this is
worth prioritizing over most other items in this document.

### 1.4 Wave-transfer/direct-store epilogue path confirmed excluded for gfx1250 — CK gets the slow fallback, with no compensating optimization

Direct read, `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp:197`:
```cpp
#if defined(__gfx120__)
    static constexpr bool IsAWaveTransferApplicable = AWaveTransferApplicable();
    static constexpr bool IsBWaveTransferApplicable = BWaveTransferApplicable();
```
`__gfx120__` excludes gfx1250 (only gfx1200/1201/gfx12-generic) — confirmed
by reading `include/ck/ck.hpp`'s alias definitions directly. So gfx1250 never
gets this optimized direct-store/wave-transfer path; it always uses the
default (LDS-reshuffle) epilogue described in §1.3, with no padding either.

MISA (§ below, "Direct store") independently confirms direct per-lane store
(bypassing LDS reshuffle entirely) is a real, valuable optimization for
gfx1250 WMMA specifically because of its lane geometry (16 consecutive lanes
already cover 16 consecutive output columns — the store is already
coalesced without LDS staging). **This is two independent findings pointing
at the same gap**: CK's fastest epilogue path is unavailable on gfx1250 by
construction (wrong tile-layout assumption per `GFX950_VS_GFX1250_COVERAGE.md`
§1), and the fallback it uses instead has no padding either. Closing either
gap independently (padding the existing fallback, or building a
gfx1250-correct direct-store path per MISA's/FlyDSL's/hipconv's converging
design — see §5) is worth pursuing; padding is far lower engineering risk.

### 1.5 CK's WMMA v3 main-loop scheduler is active, not disabled

Direct read, `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp`:
`HotLoopScheduler()` (defined lines 182-288, using
`__builtin_amdgcn_sched_group_barrier` to interleave DS-write/DS-read/VMEM/WMMA
issue groups) is called at lines 495, 569, 826, and 908 — i.e. it **is** part
of the live compute path, not `#if 0`'d out. (The one `#if 0` block in the
neighboring `gridwise_gemm_wmma_cshuffle_v3_common.hpp:593` is an unrelated,
disabled M/N GEMM-padding tensor-descriptor variant with a
"TODO: investigate why this path is not used in the original
gridwise_gemm_xdl_cshuffle_v3.hpp" comment — not a scheduler.)

**Actionable for CK**: if CK's WMMA hot-loop scheduling is suspected of
under-performing on gfx1250, treat it as a live, tunable mechanism to
profile/adjust — not dead code to resurrect. Cross-reference with MISA's own
finding (§4) that hand-scheduled instruction interleaving is a **confirmed
regression** (4-7%) on gfx1250 WMMA specifically — worth A/B testing CK's
`HotLoopScheduler()` on/off on real gfx1250 hardware, since MISA's
independent negative result on a structurally similar interleave scheme is a
signal (not proof) that CK's scheduler could also be a net loss on this
silicon generation.

## 2. Hardware Facts from MISA's Bring-Up (apply to any gfx1250 WMMA kernel, CK included)

These were independently verified by MISA via standalone, WMMA-independent
or ISA-documented reproducers — not codegen quirks specific to MISA's
Python-to-assembly pipeline. Each is something CK's own gfx1250 grouped-conv
WMMA code should be checked against.

### 2.1 Last-lane LDS-visibility gap across a split barrier at high occupancy
`docs/gfx1250_fp32_wmma_occupancy_race.md`, commits `0823d8c`/`663cb1b`.
A single-buffered LDS main loop (`ds_write` → `s_wait_dscnt 0x0` →
`s_barrier_signal`/`s_barrier_wait` → cross-wave `ds_read`) silently returns
**1-loop-iteration-stale data from the last lane of the last wave**, once
enough workgroups are concurrently resident (~1500-12000+ depending on
tile). Reproduced with a **standalone, WMMA-free** hand-written-assembly
repro (`docs/gfx1250_fp32_wmma_race_repro/`): 100% of 618k captured
mismatches were exactly 1 iteration stale, >99.9% were reads of the last
lane. Root cause not resolved at the silicon level (escalated); only
mitigation found is **`lds_double_buffer=1`** (disjoint read/write LDS
regions per iteration eliminates it in every retested config).
**CK relevance**: any gfx1250 WMMA conv/GEMM main loop reusing a single LDS
buffer across K-iterations with only a barrier as the producer→consumer
fence is at risk once occupancy is high. Any CK gfx1250 pipeline variant
that opts into single-buffered LDS (e.g. to save capacity for large tiles)
needs validation at realistic full-GPU occupancy — this bug is invisible in
low-occupancy/unit-test-sized runs.

### 2.2 TDM descriptor LDS base is a fixed SGPR — invisible to VGPR-based double-buffer toggling
`docs/gfx1250_fp32_tdm_nan_regression.md`, fix `efe84c0` + a follow-up wrw
port. `tensor_load_to_lds` (TDM) writes to LDS using a descriptor whose LDS
base address is a **plain SGPR constant** set once in the kernel prologue —
completely independent of the VGPR-based offsets ordinary `ds_write`/`ds_read`
addressing uses. A generic double-buffer toggle that only XORs VGPR offsets
silently leaves TDM writing to buffer 0 forever while VGPR-addressed reads
alternate — a stale-read NaN. Fix requires an explicit
`s_xor_b32 s[tdm_lds_base], lds_single_size, s[tdm_lds_base]` in lockstep
with the VGPR toggle. MISA missed this independently twice (fwd/bwd, then
again for wrw's separate emitter).
**CK relevance**: this is directly pertinent to CK's `cluster_load`/TDM
family (`include/ck_tile/core/arch/amd_cluster_load.hpp`, per
`ROCKE_GFX1250_PORTING_NOTES.md` §11) — if CK's TDM usage is ever combined
with double/multi-buffered LDS, the TDM descriptor's LDS-base field must be
treated as separate double-buffer state requiring its own explicit toggle;
it will not automatically participate in whatever generic buffer-select
mechanism the rest of the pipeline uses.

### 2.3 Atomic-claim return value not reliably visible to a same-wave LDS broadcast after `s_wait_loadcnt`
`docs/gfx1250_streamk_design.md`, commit `1fd71fb`. An atomic tile-claim
(`global_atomic_add_u32` with `TH_ATOMIC_RETURN`, waited with
`s_wait_loadcnt 0x0`) followed by a same-wave `ds_write_b32` broadcasting the
claimed value to other waves via LDS (barrier-gated) raced on real gfx1250
silicon — "despite the ISA spec stating loadcnt covers atomic-with-return,
propagation from the atomic unit to the VGPR, then to the LDS write port,
races with the barrier sequence." Fixed by eliminating the dynamic
atomic-claim mechanism entirely in favor of static shard indexing.
**CK relevance**: if CK ever implements a dynamic/atomic-based Stream-K-style
tile-claim-and-broadcast for gfx1250 grouped-conv (rather than static
partitioning), it inherits this exact hazard class: an atomic return value,
even after the documented wait-counter, is not guaranteed visible to a
subsequent same-wave LDS write in time for a barrier-gated cross-wave read.

### 2.4 Prefetch/cache-scope hint: `SCOPE_DEV` vs `SCOPE_CU` — an order-of-magnitude perf trap, not a correctness bug
`commit 697dbab`. `SCOPE_DEV` on a WGP-local prefetch bypasses the local
cache and forces every prefetch to the shared L2, causing **~10x
regression** once 256+ workgroups are concurrently resident (`gfx1250 ISA
§10.5`: `Scope=0` pulls into all cache levels on miss; `Scope=2`/DEVICE
brings data only into GL2). **CK relevance**: any CK gfx1250 code issuing
explicit-scope global loads/prefetches/non-temporal hints in a GEMM/conv
main loop must use CU/WGP-local scope, not device scope, for anything
intended to stay local — and must validate at full-occupancy grid sizes
since the effect is invisible on small shapes.

### 2.5 VGPR accumulator bank-select bit (`S_SET_VGPR_MSB`) is sticky across instructions, per operand slot
`docs/gfx1250_wmma_vgpr_msb_wip_status.md`. gfx1250 exposes 1024 physical
VGPRs but every instruction encoding only carries an 8-bit register field;
`S_SET_VGPR_MSB <imm>` retargets which physical 256-register bank a
*subsequent* instruction's DST/SRC0/SRC1/SRC2 slots resolve to — and this
selection **does not auto-reset**. Confirmed hardware-real (not an assembler
quirk): a VOP2 `v_add_u32`'s implicit VSRC1 operand slot silently read stale
bank-1 data because an unrelated `ds_write_b32`'s bank-1 selection (set for
a *different* operand's encoding slot) was still active. **CK relevance**:
any CK gfx1250 WMMA kernel needing >256 accumulator VGPRs (any tile
exceeding the base 256-VGPR addressing range) must reset the bank-select bit
immediately after every WMMA burst and audit every VOP2 instruction issued
while any operand is bank-1-selected, since an ordinary address-increment
can silently inherit the wrong bank via its implicit operand-slot placement.
Separately, a genuine **hardware near-hang** was found for 4+ back-to-back
same-register-dependent WMMA calls with zero intervening instructions
(required a machine reboot) — MISA explicitly warns this pattern doesn't
resemble real generated code (which always has intervening barriers/reads)
but flags it as a real hazard class to never test carelessly on live
hardware.

### 2.6 No dedicated signed 32-bit integer atomic-add on gfx1250
`commit 2834700`. gfx1250 has no signed-int atomic-add; only
`global_atomic_add_u32` (plain two's-complement add, correct for both signed
and unsigned bit patterns) exists. Using `global_atomic_add_f32` on an int32
accumulator (bit-reinterpreting the sum as IEEE float) is only coincidentally
correct for small non-negative sums. **CK relevance**: any CK grouped-conv
split-K/atomic-reduction epilogue for int8 (or other integer) accumulation
on gfx1250 must issue the u32 atomic-add directly on the int32 bit pattern,
never a float-based atomic trick.

## 3. WMMA Instruction/Register-Layout Facts (hardware-verified by MISA, portable)

Per-lane operand/accumulator mappings for gfx1250 WMMA atoms (wave32, lane
`l` = 0..31), hardware-verified via a round-trip probe
(`docs/gfx1250_wmma_layout.md` lines 1-90):

| Atom | A operand | B operand | C/D operand |
|---|---|---|---|
| `v_wmma_f32_16x16x32_f16`/`_bf16` (8 VGPR/lane, 2 packed/dword) | `row=l%16`, `k=(l/16)*16+a*2+s` | `col=l%16`, `k=(l/16)*16+a*2+s` | `row=(l/16)*8+j`, `col=l%16` (fp32, 8 VGPR/lane) |
| `v_wmma_i32_16x16x64_iu8` (8 VGPR/lane, 4 packed int8/dword) | `row=l%16`, `k=(l/16)*32+a*4+s` | `col=l%16`, same k formula | same formula as above, int32 |
| `v_wmma_f32_16x16x4_f32` (**2 VGPR/lane, no packing** — structurally distinct) | `row=l%16`, `k=(l/16)*2+a` | `col=l%16`, same | identical D formula to the others |

**`v_wmma_f32_16x16x128_fp8_fp8` (K=128) row/k mapping was never hardware-probed
by MISA** — explicitly flagged `[UNVERIFIED]`; only the D-operand layout is
*expected*, not confirmed, to carry over. If CK's fp8/bf8 gfx1250 WMMA
instances need independent verification, MISA's data does not cover this
specific instruction.

**Accumulator has no hardware clear op** — must be explicitly zeroed by
software before the K-loop.

**gfx1250 workgroup-ID delivery**: `blockIdx.x/y` arrive via `ttmp9`/`ttmp7`
trap-temporary registers, not classical system SGPRs, regardless of
`.amdhsa_system_sgpr_workgroup_id_*` kernel-descriptor flags (discovered by
disassembling `hipcc`-compiled HIP code; no official documentation found).
MISA flags this as toolchain-version-fragile — re-verify if CK's generated
code or ROCm/LLVM version changes. `blockIdx.z` is folded into a bit-split of
the same ttmp register, not a separate one.

**Toolchain gotchas confirmed on real hardware**: gfx1250/wave32 requires
`vcc_lo` (not `vcc`) and `v_add_co_ci_u32` (not `v_addc_co_u32`); waitcnt is
split into `s_wait_loadcnt`/`s_wait_storecnt`/`s_wait_dscnt`/`s_wait_kmcnt`
rather than one `s_waitcnt`; `.amdhsa_ieee_mode`/`.amdhsa_dx10_clamp`/
`.amdhsa_workgroup_processor_mode` kernel-descriptor directives are rejected
by the assembler build MISA used; the full `amdgcn-amd-amdhsa` triple is
required (the vendor-less `amdgcn--amdhsa` triple fails to assemble).

## 4. Performance Techniques — What Worked, What Didn't (avoid re-deriving negative results)

### 4.1 Confirmed wins
- **LDS row padding** (§1.3) — the dominant lever, +41-98% relative, up to
  +70% on wrw. Highest-priority recommendation in this document.
- **Epilogue store-width/dtype techniques** (`docs/gfx1250_c1c2_fp16_output.md`,
  `docs/gfx1250_c2_widen_store.md`): packing fp32 accumulator pairs into
  fp16x2/bf16x2 for direct-store (+9-15%), then widening the packed store to
  `global_store_dwordx2` (further +2.5-2.9% compute-bound, **+45.6% to +51.3%**
  on bandwidth-bound/shallow-K shapes — exactly the shape family grouped-conv
  wrw with small output channels produces). General technique (narrow output
  dtype + widen vectorized store), gfx1250-specific in the exact
  lane-layout/coalescing details.
- **Direct per-lane store, skipping LDS reshuffle entirely**
  (`docs/gfx1250_direct_store_plan.md`): gfx1250's WMMA C-tile lane geometry
  (16 consecutive lanes = 16 consecutive output columns) is already coalesced
  for scalar per-lane global stores — independently confirmed by MISA,
  FlyDSL, and hipconv (three separate projects converging on the same
  design). hipconv ships this at the ISA level: `ds_store_b128` into padded
  LDS scratch, then `global_store_async_from_lds_b128` issued directly from
  the LDS address, skipping the VGPR-readback step MISA's own
  `coalescing_store_wmma.py` (and, per §1.3-1.4, CK's default epilogue) still
  does.
- **Liveness-based VGPR reuse** (commits `f100a56`/`30eefef`): a
  compiler-adjacent register-allocation technique overlapping dead
  main-loop address registers with epilogue-only scratch. Freed 2-6 VGPRs at
  the 256-VGPR ceiling — small in isolation but disproportionately valuable
  on gfx1250 because fp16/bf16/int8 128×128 WMMA tiles sit pinned at that
  exact ceiling (a few freed registers can be the difference between a
  config building at all).

### 4.2 Confirmed negative/neutral results — do not re-derive
- **Main-loop chunk/compute interleaving**: measured a reproducible **4-7%
  regression** on gfx1250 WMMA (commit `3fc9e18`), kept disabled by explicit
  policy (not deleted, since gfx1250 is pre-production and future steppings
  might behave differently). See §1.5 for CK's own active scheduler as a
  candidate for the same A/B test.
- **LDS double-buffering alone**: performance-**neutral** (not the fix hoped
  for) — only becomes valuable when *combined with* interleaving, which
  itself regressed. It remains a **correctness requirement** for fp32 (§2.1)
  independent of its performance neutrality.
- **Workgroup swizzle** (L2-locality remapping): only **<1%** gain (vs. a
  hoped-for 5-15% seen on other architectures), and actively hurts small
  grids (-1.4%). **Do not assume classic swizzle wins transfer to gfx1250
  without re-measurement.**
- **Incremental-gather address strength-reduction**: an initial **+19.9%**
  measurement was a **GPU-contention artifact** — re-measured at only ~+1%
  on an uncontended machine. General lesson: always re-measure gfx1250 perf
  claims on an uncontended GPU with multiple runs before trusting a
  percentage.
- **VGPR-level prefetch depth 2**: could not even be shipped for fp16/bf16/int8
  128×128 tiles — no VGPR budget remains (would need +64 registers with zero
  headroom at the 256-VGPR ceiling). A hard resource ceiling, not a tuning
  choice.
- **256×256/256×128 WMMA tiles via VGPR-MSB banking or 8-wave splitting**:
  hardware-validated correct, but a **consistent net performance loss at
  every tested scale** (2-2.7x slower small, still 1.35-1.4x slower even at a
  deliberately huge, occupancy-saturating shape — the ratio plateaus, it
  doesn't close with scale). The register math (`total_acc_c =
  gemm_m_per_block*gemm_n_per_block/block_size`) means **any** tile with one
  dimension at 256 and the other <256 hits an identical VGPR ceiling — not
  worth CK pursuing >128-per-side WMMA tiles on gfx1250 via a similar banking
  trick without new evidence.
- **`atomic_cascade` (cascading atomic)**: **permanently hangs real gfx1250
  hardware** in MISA's usage (missing companion release/fence for the
  deferred scope-completion signal) — deleted from MISA's codebase entirely.
  Note: hipconv's grouped-wgrad kernel uses `TH_ATOMIC_CASCADE_RT`
  successfully in production, contradicting MISA's finding — an open,
  unresolved discrepancy (§5). **If CK ever considers a cascading atomic for
  gfx1250 split-K reduction, treat it as unsafe until the calling-convention
  difference between MISA's and hipconv's usage is understood.**

## 5. Cross-Project Hardware Facts (FlyDSL, hipconv — independently converging with MISA)

MISA's own external research surveyed two other independent gfx1250
kernel-authoring efforts. Facts confirmed by **multiple independent
projects converging on the same encoding/behavior** are the strongest form
of evidence in this document (stronger than any single project's finding),
since three unrelated codebases agreeing exactly rules out project-specific
error.

- **TDM (Tensor Data Mover) descriptor format is byte-identical across all
  three independent implementations** (MISA hand assembly, FlyDSL's
  `CopyAtom.cpp`, hipconv's `bunnies_mi400.hpp`). `TENSOR_LOAD_TO_LDS`/
  `TENSOR_STORE_FROM_LDS` do whole-tile global↔LDS transfer in one async
  instruction with **hardware-native OOB handling**: loads zero-fill
  out-of-bounds rows in LDS, stores silently drop out-of-bounds writes —
  controlled entirely by descriptor `tensor_dim`/`tile_dim` fields, no EXEC
  masking needed. **Load-bearing correctness detail for any K-tail-via-TDM
  loop** (relevant to CK's `cluster_load` family): for a looping K-reduction
  with an advancing global address, `tensor_dim0` must be **decremented by
  the tile width every iteration** — holding it constant only zero-fills
  correctly at iteration 0; later iterations read past-the-end memory.
  Hardware LDS padding is also descriptor-native (`pad_interval`/
  `pad_amount` fields) — hipconv uses this in production, replacing software
  pad-stride computation entirely (a cleaner alternative to CK's currently
  entirely-absent padding, §1.3, if CK's WMMA path uses TDM for LDS staging).
- **Workgroup Cluster multicast** (`workgroup_mask` field, `CLUSTER_LOAD_ASYNC`):
  multiple workgroups sharing an M-tile-row or N-tile-column get one HBM
  read fanned out via TDM — confirmed in FlyDSL's production 256×256 GEMM
  kernels (4×4/16-workgroup hardware cap clusters), with a distinct
  cluster-scope barrier (`s_barrier_signal(-3)`/`s_barrier_wait(-3)`).
- **Packed 2-wide atomics for split-K epilogues** (FlyDSL, gfx1250-confirmed
  production code): builds a `vector<2xf16>` from two adjacent output
  columns and issues one packed atomic-add instead of two scalar ones,
  halving atomic-op count. MISA's own wrw atomic epilogue is scalar-per-element
  f32 — flagged by MISA itself as an untested, plausible adoption candidate.
  Relevant to CK's grouped-conv wrw atomic epilogue (§1.2/§2.6) if CK ever
  narrows accumulation precision.
- **Block-diagonal WMMA channel packing for small-group/grouped convolution**
  (hipconv's shipped `grouped_multi_g_wgrad` kernel): instead of padding a
  tiny (G×G) weight-gradient block up to the full 16×16 WMMA tile and wasting
  most lanes, packs `GPW=16/G` independent groups diagonally into one WMMA
  instruction, discarding off-diagonal cross-terms in the epilogue. A
  register-tile-level fix for small-M/N occupancy — directly relevant to any
  WMMA kernel (CK's grouped conv with small per-group channel counts
  included) needing to handle small-group convolutions efficiently.
- **Split-K via a separate reduction kernel, as an alternative to atomics**
  (hipconv, shipped side-by-side with an atomic-cascade variant in a static
  per-shape config table): each K-split partition writes to a private
  workspace slice with zero atomics in the main kernel; a second, trivial
  elementwise-sum kernel reduces partitions. A concrete alternative
  architecture to CK's atomic-only wrw split-K epilogue.
- **DMA-engine parity and issuer-role assignment** (hipconv, measured): gfx1250
  WGPs have two DMA engines selected by wave parity (even/odd wave id);
  assigning load-issuer and store-issuer roles to opposite parities measured
  a **19% win**; having only one wave issue a given TDM transfer (not every
  wave redundantly) measured **3-7%** better than naive all-waves-issue.
- **Scheduling/priority hints, several distinct and untested-in-MISA**:
  `s_setprio(1)`/`s_setprio(0)` bracketing a WMMA sequence (hipconv, standard
  ISA-documented priority bump); `disable_xdl_arb_stall`
  (`S_SETREG_B32` write to `SCHED_MODE` bit 2, FlyDSL) — trades away
  co-execution with neighboring waves, a plausible win only for
  low-occupancy-per-SIMD kernels (e.g. wrw's small split-K workgroups),
  explicitly flagged as needing isolated A/B testing before adoption, not a
  guaranteed win; `sched_group_barrier`-style IR-level scheduling hints
  (narrower blast radius than MISA's own regressed full-instruction-reorder
  interleave, §4.2).
- **Non-power-of-2 / sub-64 WMMA tile shapes are legal and production-used**:
  FlyDSL ships a tested `tile_n=96` gfx1250 GEMM tile (not 128-aligned, not
  power-of-2), with an explicit compiler guard branching to a general
  N_BLOCKS computation. Counter-evidence against assuming WMMA tiles must
  stay at powers-of-2 ≥64 — relevant to CK's own WMMA tile-shape enumeration
  for small/irregular conv shapes (the wrw small-M/N gap discussed in
  `GFX950_VS_GFX1250_COVERAGE.md`).
- **N-stage software pipelining beyond double-buffering, parametrized by LDS
  headroom** (FlyDSL): `num_buffers` (2/3/4, chosen per-shape, gated by an
  explicit `check_smem_capacity` check) with `s_wait_tensorcnt(num_buffers-2)`
  in steady state. A generalization of the double-buffering both MISA and
  CK currently use — worth checking whether CK's WMMA pipeline depth is
  hardcoded at 2 vs. parametrized by available LDS headroom (CK's gfx1250
  LDS is 320KB per `arch.hpp`/`get_smem_capacity()`, confirmed independently
  by MISA's hardware read of the same figure).

## 6. Generic XDL→WMMA Porting Principles (abstracted from MISA's own two generators)

MISA maintains both an XDL and a WMMA code generator internally and has its
own porting backlog between them — the principles below are abstracted from
that experience and apply directly to CK's own XDL→WMMA gap (CK, per
`GFX950_VS_GFX1250_COVERAGE.md`, also has a mature XDL/MFMA path and a less
mature WMMA path for gfx1250):

1. **Check within-family (WMMA-to-WMMA) parity before porting from the
   MFMA-era codebase.** MISA found several optimizations that already exist
   in one WMMA conv direction but were never cross-ported to the other two —
   cheaper to close than porting fresh from XDL, since the WMMA-specific
   concerns (wave32, VGPR-only accumulate, WMMA-specific addressing) are
   already solved by the sibling direction. **For CK**: before porting an
   XDL-era optimization to CK's WMMA path, check whether CK's WMMA path
   already has it in a different conv direction.
2. **A ported feature is not automatically a win — measure per direction.**
   MISA's `main_loop_interleave` ported cleanly (correct in all 3 directions)
   but measured 7-10% *slower* on bwd/wrw despite winning on fwd, because
   interleave/latency-hiding tricks interact differently with each
   direction's transposed-vs-contiguous operand access pattern. **An
   XDL-era latency-hiding scheme that wins on CK's fwd is not guaranteed to
   win when mechanically ported to CK's bwd-data/bwd-weight WMMA path.**
3. **Removing AGPR (accumulator register file) is a net simplification but
   reintroduces register contention that needs a *new* compensating
   mechanism.** WMMA has no AGPR — the accumulator lives in plain VGPR,
   competing directly with operand registers. XDL-era AGPR-to-VGPR transfer
   code should not be ported (correctly absent), but the register-pressure
   problem AGPR used to solve for free (isolating accumulator space from
   operand space) needs a WMMA-native answer (MISA's: VGPR-MSB banking +
   narrower intermediate accumulation types). **Don't just delete AGPR-era
   code during a WMMA port — identify and replace the problem it was
   solving.**
4. **FMA-orthogonal XDL features are the ones most likely to be silently
   dropped during a WMMA rewrite, because they aren't "front and center"
   during instruction-set bring-up.** MISA's own unported-after-years list:
   skip-LDS-for-one-operand pass-through on skinny/small-K shapes, folding
   multi-tap filter extent into GEMM_K to eliminate a runtime tap-loop
   (`merge_e`), M-major-vs-N-major dispatch order for L2 reuse, configurable
   vector-store width, main-loop (not just epilogue) LDS padding. **CK's own
   XDL→WMMA port plan should explicitly enumerate every FMA-orthogonal XDL
   tunable/mechanism as its own checklist**, separate from WMMA-instruction
   work.
5. **Buffer-descriptor (SRD) addressing has no 1:1 WMMA equivalent and must
   be independently re-derived per operand/direction.** XDL's
   `buffer_load_dwordx4` against a hardware-clamping SRD has no WMMA
   analogue; software EXEC-masking is the replacement, and even MISA's own
   closest analogue (`saddr_global_load`) needed independent re-derivation
   for A vs. transposed-B operands and per conv direction. **Don't assume an
   XDL-era buffer-descriptor addressing trick has a mechanical WMMA
   translation.**

## 7. Tunable/Feature Exclusions — What's Unsafe on WMMA That Was Safe on MFMA

From MISA's own collapsed exclusion list (`docs/gfx1250_tunable_exclusions.md`,
each entry a real, reproduced build/assembly/hardware failure, not a
hypothetical):

- **Instruction interleaving requires double-buffered LDS** — single-buffered
  interleaving races across waves (confirmed on real hardware). Applies to
  any WMMA kernel attempting latency-hiding overlap of compute with the next
  tile's load.
- **64KB/workgroup LDS is a hard ceiling** that any large-tile WMMA epilogue
  padding scheme must respect (128×128 tiles are already at the wall with
  zero headroom in MISA's own layout).
- **Cascading atomics hang gfx1250 hardware** in at least one calling
  convention (§4.2) — contradicted by hipconv's differently-coded successful
  usage; treat as unsafe until the discrepancy is understood.
- **Asymmetric-tile + tail-masking combinations can fail at the assembler
  level** even when every Python/template-level precondition passes — a
  register-range formula that assumes symmetric-tile addressing doesn't
  always generalize to asymmetric tiles. **Generalizable warning for CK**:
  validate asymmetric-tile+tail combinations at actual-assembly granularity,
  not just compile-time asserts.
- **Not every listed exclusion is a real hardware constraint** — several
  MISA exclusions were later found to be stale heuristics or simple bugs
  (e.g. a double-buffer+row-pad interaction that was just a missed offset
  toggle, not a fundamental incompatibility) and were removed once
  root-caused. **Lesson for CK**: don't assume an inherited "known
  incompatible" combination in CK's own tunable-validity tables is a
  permanent hardware constraint without re-testing against the current
  implementation.

## 8. Prioritized Recommendations for CK

1. **(Highest confidence, highest measured value) Add LDS row/bank-conflict
   padding to CK's WMMA epilogue and any transposed-operand LDS staging on
   gfx1250** (§1.3). MISA measured +41-98% from this alone; CK currently has
   none. Lowest-risk, highest-payoff item in this document.
2. **(High value, needs real-hardware verification) Re-test CK's WMMA main-loop
   `HotLoopScheduler()` (§1.5) on/off on real gfx1250 hardware.** MISA's
   independent negative result on a structurally similar interleave scheme
   (§4.2) is a signal, not proof, that CK's active scheduler could be a net
   loss on this silicon generation.
3. **(Correctness-critical, verify before scaling split-K) Test CK's
   DEVICE-scope buffer-atomic fix (§1.2) against wrw split-K at large CU
   count / high split-K factor**, given MISA's independent finding that a
   *different* atomic instruction class needed the wider SYSTEM scope on
   the same hardware.
4. **(Medium effort, converging 3-project evidence) Evaluate a gfx1250-correct
   direct-per-lane-store epilogue** (§4.1, §5) to replace or supplement the
   LDS-reshuffle fallback CK is stuck with since its wave-transfer path
   excludes gfx1250 (§1.4) — three independent projects (MISA, FlyDSL,
   hipconv) converge on this design being valuable specifically for
   gfx1250's WMMA lane geometry.
5. **(Medium priority, closes a documented wrw gap) Consider CK's wrw
   small-M/large-K tile coverage** using MISA's/FlyDSL's converging evidence
   that non-power-of-2, non-≥64 WMMA tiles are legal and production-viable
   (§5) — relevant to the wrw small-output-channel gap already noted in
   `GFX950_VS_GFX1250_COVERAGE.md`.
6. **(Low risk, mechanical) Verify CK's `cluster_load`/TDM usage (if any)
   correctly advances the TDM descriptor's SGPR-resident LDS base in lockstep
   with any VGPR-based double/multi-buffering** (§2.2) — an easy-to-miss
   interaction MISA hit twice independently.
7. **(Process) Do not assume CK's gfx1250 `_xdl`-named instances are
   uniformly broken or uniformly fine** — verify per device-op family via
   real compilation (§1.1 method is reusable: `--offload-arch=gfx1250
   --offload-device-only`, then `llvm-objdump -d | grep -c v_mfma` vs
   `v_wmma`) before changing CMake gating.
8. **(Avoid wasted effort) Do not re-attempt**: hand-scheduled instruction
   interleaving without double-buffered LDS (races), cascading atomics
   without understanding hipconv's differing calling convention (hangs),
   256-per-side WMMA tiles via VGPR banking (consistently net-slower at
   every scale tested), or workgroup swizzle expecting a 5-15% win (measured
   <1% on this hardware) — all confirmed negative results in §4.2.
