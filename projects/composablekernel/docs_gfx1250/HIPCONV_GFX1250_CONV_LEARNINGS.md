# hipConv gfx1250 (CDNA5) Learnings for Composable Kernel

hipConv (`/home/sgundabo/hipconv`, branch `main`) is a production HIP C++
implicit-GEMM convolution kernel library that feeds MIOpen, developed under
heavy PR review with an explicit multi-architecture packaging discipline.
Unlike MISA (hand-written assembly) or rocKE (a Python SSA/IR DSL), hipConv
is ordinary templated HIP C++ — architecturally closer to CK's own style —
which makes its gfx1250 findings unusually directly comparable.

**Architecture naming**: hipConv classifies its four architecture buckets by
real GPU target (`hipconv/src/arch/<name>/CMakeLists.txt`,
`hipconv_add_arch_lib(<name> TARGETS <gpu>)`, confirmed by direct read):

| hipConv name | GPU | Notes |
|---|---|---|
| `cdna3` | gfx942 | |
| `cdna4` | gfx950 | |
| **`cdna5`** | **gfx1250** | Target of this investigation |
| `rdna5` | gfx1310 | NPI/bring-up-only, out of scope |

hipConv calls gfx1250 **"CDNA5"** — an Instinct/data-center-line successor
to gfx950/CDNA4, not an RDNA-family part — despite gfx1250 using WMMA and
wave32 like RDNA/gfx12. This is consistent with CK's own `ck_tile` layer
treating `gfx1250_t` as its own family distinct from both `gfx9_t` and
`gfx12_t` (per `GFX950_VS_GFX1250_COVERAGE.md` §2): three independent
projects now agree gfx1250 needs its own classification bucket rather than
inheriting CDNA-family or RDNA-family assumptions wholesale.

**Methodology**: as with the MISA investigation, this document is built from
(a) 3 parallel research agents reading hipConv's own docs/source/commit
messages directly, and (b) my own independent audit of CK's source —
including a direct grep-based cross-check of CK's actual WMMA intrinsic
definitions against hipConv's exact hazard description. Two headline
correctness hazards were found by me directly from hipConv's commit history
before delegating the rest.

## 1. Two Headline gfx1250 Hazards (found directly, hardware-verified by hipConv)

### 1.1 WMMA C-operand WAR hazard — an LLVM bug, still present upstream

`hipConv commit 1b0e3e7`, `hipconv/src/arch/cdna5/bunnies_mi400.hpp:405-421`:

> `v_wmma_bf16f32_16x16x32_bf16` writes a D narrower than C, so C dies at the
> MMA and its registers are free for reuse while the hardware is still
> reading them. The scheduler hoists an unrelated VALU write onto them; on
> gfx1250 it lands on the dword holding accumulator element 6, which then
> reads as zero and fails `wmma_test`. The root cause is in LLVM:
> `GCNHazardRecognizer::hasWMMAToVALURegOverlap` adds `src2` to the WAR
> check only for SWMMAC, so a plain WMMA's C operand is never checked.
> **Still true on upstream main.**

Workaround: `__builtin_amdgcn_sched_barrier(0)` immediately after the WMMA
issue, blocking the compiler from hoisting a VALU write into the hazard
window (costs pipelining across that point; does not protect an instruction
that already follows in source order — a per-site fix, not a general one).
Verified on real gfx1250: `wmma_test` 25/25 after the fix.

**My own independent cross-check against CK**: this is the "bf16 input,
bf16 (not fp32) accumulate" WMMA overload — narrower D than the full-width C
a wider-accumulate variant would use. I grepped CK's source directly for
this exact overload class:

```
include/ck_tile/core/arch/mma/wmma/wmma_gfx12.hpp:417:
struct amdgcn_mma<bf16_t, bf16_t, bf16_t, 16u, 16u, 32u, CompilerTarget,
                  MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
```
```
include/ck_tile/core/arch/mma/wmma/wmma_gfx12.hpp:1067:
struct amdgcn_mma<fp16_t, fp16_t, fp16_t, 16u, 16u, 32u, CompilerTarget,
                  MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
```

**CK ships gfx1250-specific `amdgcn_mma` specializations for exactly this
narrow-accumulate (bf16→bf16, fp16→fp16) overload class** — gated
`enable_if_target_gfx1250_t`, i.e. real, live, gfx1250-only code, not dead
template branches. I checked the ISA-wrapper header itself
(`wmma_gfx12.hpp:400-440`) for any `sched_barrier` protection at the
intrinsic-definition site: **none found** — which is architecturally
expected, since a raw ISA-wrapper header is the wrong layer to insert
pipeline-scheduling hints; that would belong in the calling GEMM
pipeline/block-level code (`include/ck_tile/ops/gemm/pipeline/*.hpp`,
`include/ck_tile/ops/gemm/block/*.hpp` — confirmed via grep to contain
`sched_barrier` usage generally, but whether any specific placement
incidentally guards this exact WAR hazard was not verified in the time
available for this investigation — flagged `[UNVERIFIED, high-priority]`).

**Actionable for CK**: audit every call site of the gfx1250 bf16→bf16 and
fp16→fp16 `amdgcn_mma` specializations (lines 417, 1067) in CK's WMMA GEMM
pipelines for a `sched_barrier(0)` (or equivalent) immediately after the MMA
issue, using hipConv's exact fix as the template. If absent, CK's own
gfx1250 code compiled with a recent-enough LLVM/clang could be silently
exposed to the identical accumulator-corruption bug hipConv found — this is
a **live LLVM bug** ("still true on upstream main"), not an
already-patched-elsewhere issue, so CK's exposure depends purely on whether
its own generated code sequence happens to place a VALU write in the hazard
window, which is compiler-scheduling-dependent and could appear or
disappear across ROCm/LLVM versions. This is the single most concrete,
actionable, and urgent finding in this document.

### 1.2 Sub-dword TDM hazard racing a same-wave transpose-read gather

`hipConv commit ebf2180` (direct dgrad kernel):

> The direct dgrad c-advance loader issues its TDM tensor load into the LDS
> half the current c trip does not read, so the load is built to stay in
> flight across that trip's gathers. When the dgrad reduction count is odd
> at a 2 byte element width the load's innermost extent ends mid dword, and
> while such a load is outstanding the same wave's `ds_load_tr16_b128`
> gather of the weight tile returns a neighbouring value. Only dgrad gathers
> the weights transposed, which is why fprop and tf32 are clear of it, and
> the wait at the top of the next c trip is too late to protect this trip.

Fix: `s_wait_tensorcnt(0)` inserted narrowly — guarded to dgrad, the
non-`IsKTile1` path, wave 0, and an odd reduction count at that element
width — so the common (even-count) case keeps its prefetch overlap.

**Why this matters for CK**: this is a real, hardware-verified example of an
**in-flight TDM load racing a same-wave transpose-read gather**, gated on a
narrow (odd sub-dword) condition invisible at typical test shapes. CK's
documented, still-open `qr_tdm` FMHA pipeline bug (per
`GFX950_VS_GFX1250_COVERAGE.md` §7: "a known ping-pong K/V prefetch bug for
prefill seqlen≥2048 with sink masks") is structurally the same failure
class — an outstanding async/TDM prefetch racing a subsequent same-wave
read. hipConv's fix pattern (narrowly-scoped `s_wait_tensorcnt`/wait-counter
drain, gated on the exact condition that produces a sub-dword/misaligned
extent) is a concrete template CK should try against its own `qr_tdm`
repro shape, rather than treating the bug as an unexplained, unfixable
hardware mystery.

## 2. Three Additional Hazards (found by research agents, not yet in any prior investigation)

### 2.1 gfx1250 vs CDNA4 sparse-encoding slot shift (silicon behavior difference, not a bug)
`hipconv/src/arch/cdna5/bunnies_mi400.hpp:251-253`, in `matrix_cast` (1:4→2:4
sparsity conversion, lines ~349-361):

> gfx1250 keeps the live value in the odd slot where CDNA4 uses the even
> one, so the index shifts up by one 2-bit field.

**CK relevance**: this is a correctness-critical encoding difference — if
CK's own sparse-MMA (`amd_smfmac.hpp`) or any structured-sparsity WMMA
support is ever extended from CDNA4/gfx950 to gfx1250 by analogy, this exact
odd/even slot shift must be accounted for or values land in the wrong 2:4
slot silently.

### 2.2 TDM descriptor mutation mid-pipeline deadlocks (not just corrupts)
`hipconv/src/arch/cdna5/grouped/grouped_multi_g_wgrad/kernel.hpp:174-177`:

> Channel extent is loop-invariant per wave ... So set the TDM channel clamp
> ONCE in init — mutating the descriptor between in-flight loads desyncs the
> tensor count and **deadlocks** on the cycle-accurate model.

This is a distinct hazard class from §1.2: not silent corruption but a
**deadlock** — mutating a TDM descriptor field that affects issue/retire
accounting while a prior load against that descriptor is still in flight.
**CK relevance**: any CK `cluster_load`/TDM descriptor reuse across loop
iterations (`include/ck_tile/core/arch/amd_cluster_load.hpp`) must treat
descriptor fields affecting the tensor-count accounting as write-once per
in-flight-load lifetime, not safely mutable mid-loop.

### 2.3 `s_wait_tensorcnt` is per-wave, not per-workgroup — cross-counter ordering gaps
`hipconv/src/arch/cdna5/grouped/grouped_multi_g/kernel.hpp:632-633, 1291-1292`
(`zero_slot`) and independently confirmed in
`docs/algorithms/toeplitz/depthwise-1d-toeplitz-cdna5.md`:

- **`s_wait_tensorcnt` does not order against `ds_write`/LDS-store
  completion** — zeroing an LDS ring slot via `ds_write` before a TDM load
  into the same slot requires a *separate* `s_wait_dscnt` fence, because the
  TDM engine's completion counter and the LDS-store completion counter are
  independent and do not cross-order each other. The depthwise design doc
  makes the identical point: "LDS writes and tile-DMA writes are separate
  memory types with no ordering between them."
- **`s_wait_tensorcnt` is a per-wave counter of that wave's own in-flight TDM
  ops** — if any wave other than the one that issued the load tries to wait
  on it, the wait is meaningless for tracking the real transfer. hipConv's
  depthwise/grouped kernels therefore designate exactly **one named "ring
  wave"** as the sole issuer+waiter for a given TDM stream, publishing
  completed data to other waves via an ordinary `s_barrier` afterward (never
  via the tensorcnt wait itself). This was **measured, not just
  theorized**: round-robin issuing (every wave issues its own TDM and
  waits) cost **-3% to -7%** versus one dedicated issuer wave.

**CK relevance, directly actionable**: audit whether any CK gfx1250
`cluster_load`/TDM-issuing code assumes any-wave-can-wait semantics, or
mixes LDS-store zeroing with TDM-load reuse of the same LDS region without
an explicit `s_wait_dscnt`. This is exactly the kind of TDM synchronization
discipline gap that could produce an intermittent, occupancy- or
timing-dependent correctness bug indistinguishable from "flaky hardware" —
the same symptom class as CK's own `qr_tdm` bug (§1.2).

## 3. Kernel Architecture: Asymmetric Wave-Role Specialization (a different design than CK's symmetric-wave pipelines)

hipConv's cdna5 `direct` kernel (`hipconv/src/arch/cdna5/direct/kernel.hpp`)
launches 8 waves per block and splits them into two **wave groups by
`wave_id_k`**, not by symmetric per-thread cooperative loading:

- `wave_id_k==0` acts as **"CU0"**, owning the input-activation TDM prefetch
  pipeline.
- `wave_id_k==1` acts as **"CU1"**, owning the weight TDM prefetch pipeline.

Both wave groups execute the *same* WMMA accumulation loop, reading shared
LDS tiles via `load_tile<...>`, but which operand each group is responsible
for *prefetching* is templated (`IsKTile1=true/false`) and diverges per
group. This is a genuine **operand-specialized producer/consumer split
baked into the wave-ID partition** — entire wave groups get disjoint
responsibilities — rather than CK's conventional approach of every wave
symmetrically loading and computing its own subtile with a uniform
prologue/steady-state/epilogue pipeline stage split.

Fine-grained scheduling cadence: the main loop
(`kernel.hpp:463-624`) recurs `s_barrier()`+`sched_barrier(0)` pairs after
essentially every phase transition (load → cast → MMA → next-load), and
brackets each `mma()` call with `s_setprio(1)`/`s_setprio(0)` — a much
finer-grained synchronization/scheduling-hint cadence than a typical CK v3
pipeline's larger-grained staging. Notably `s_setprio(1)`/`(0)` bracketing
the MMA issue is the **same technique** independently found in CK's own
`blockwise_gemm_pipeline_wmmaops_v1.hpp` (per the MISA investigation) and in
rocKE's/FlyDSL's/hipconv's own bunnies-level work — **four independent
projects now converge on `s_setprio` bracketing around matrix-instruction
issue as a real, adoptable technique** on gfx1250/gfx12-class hardware.

Output epilogue chooses between a checked (bounds-predicated) and unchecked
async global-store path based on whether the whole output tile is provably
in-bounds at compile/launch time (`kernel.hpp:820-860`) — a
static/dynamic-dispatch optimization worth comparing against CK's own
gfx1250 epilogue bounds-checking (which, per prior investigation, has no
equivalent direct-store path available on gfx1250 at all, since CK's
wave-transfer optimization is `__gfx120__`-gated and excludes gfx1250).

## 4. Block-Diagonal WMMA Packing for Small-Group Convolution (third independent confirmation)

`hipconv/src/arch/cdna5/grouped/grouped_multi_g/kernel.hpp` and
`grouped_multi_g_wgrad/kernel.hpp` handle grouped convolution with small
channels-per-group (G ∈ {4, 8, 16}) by **packing `GPW = 16/G` independent
groups diagonally into a single 16×16×32 WMMA instruction**, structurally
zeroing the off-diagonal cross-group products in the A operand construction
(comment, `kernel.hpp:285-286`: "g0 rows only dot g0 K, g1 rows only g1 K")
rather than computing and discarding them. The K=32 dimension is
*additionally* packed with two adjacent convolution taps (K_lo/K_hi
covering K[0..15]/K[16..31]) so one WMMA call does the work of what would
otherwise be two separate 16-wide-K MMAs — two independent packing axes
compounded into one instruction. G=32 gets a separate, unpacked "one group
per wave, full K=32 contraction" code path since it already saturates the
operand.

**This is the third independent codebase this investigation series has
found using exactly this design** for gfx1250 grouped/small-channel
convolution (hipconv here; a separate implementation was found independently
re-derived in FlyDSL/hipconv cross-reference during the MISA investigation,
per `MISA_GFX1250_CONV_LEARNINGS.md` §5). Three convergent, independent
implementations of the same technique is strong evidence it is *the*
correct approach for small-group WMMA convolution on gfx1250, not an
idiosyncratic choice — **directly relevant to CK's own grouped-conv WMMA
instance library**, which per prior investigation has no documented
small-group-packing strategy and instead relies on its wide static instance
library (tile-shape enumeration) to cover small shapes.

## 5. TF32 on gfx1250: Real and Shipped in hipConv — CK Has None

hipConv's cdna5 (gfx1250) build ships **genuine, tested TF32 support** via a
3-way bf16-pair emulation (gfx1250 WMMA has no native TF32 mode, matching
CK's own finding that gfx1250 lacks native TF32):

- `hipconv/src/arch/cdna5/direct/config.hpp:32-35` adds an `elem_bytes`
  field (2 for fp16/bf16, 4 for tf32) because "the tile sizes are picked per
  width (tf32 halves tile_size_c and tile_size_k)" — driven by the TDM
  descriptor's pad-field bit-width limit.
- `hipconv/src/arch/cdna5/direct/kernel.hpp:66-80`: tf32 is stored as plain
  fp32 and split into a "(big, small) bf16 pair" on the way into the MMA
  (`compute_fmt = e8m10_e8m7x2split`) — the standard 3-pass-bf16 TF32
  emulation.
- Covered end-to-end by both device-unit tests
  (`test/cdna5/wmma_test.cpp`'s `TF32_F32_F32` config: "three bf16 WMMAs
  emulating tf32"; `test/cdna5/swmmac_test.cpp`'s
  `SwmmacTf32SplitTest.SplitCarriesResidual`, validating the residual-carrying
  split against both an exact double-precision reference and a naive
  bf16-rounded one) and the full spec-driven integration suite.
- Wired into dispatch as a first-class branch
  (`Direct_F16_ConvKernel::is_applicable`'s `ok_tf32` predicate), not a stub.
- Also shipped in `grouped_multi_g_wgrad` ("tf32 is stored as fp32 and
  multiplied as a bf16 (big, small) pair, so its delta is fp32 too and every
  staged byte count doubles").

**This directly contrasts with CK**: per `GFX950_VS_GFX1250_COVERAGE.md`
§3, CK has **zero** TF32 support on gfx1250 (`CK_ENABLE_TF32` gates on
`gfx942|gfx95` only; gfx1250 matches neither). hipConv's working
implementation is proof this is a **closeable software gap, not a hardware
ceiling** — CK could adopt the identical bf16-pair-split emulation technique
(the same one CK's own `xdlops_gemm.hpp` uses for XDL-path TF32 on other
architectures, applied here at the WMMA level) rather than treating gfx1250
TF32 as architecturally infeasible.

## 6. Testing/CI Methodology: A Concrete Pattern for CK's gfx1250 Real-Hardware Gap

hipConv's `test/spec_driven_test.cpp` implements a pattern directly portable
to CK's own documented gfx1250 CI gap (per `GFX950_VS_GFX1250_COVERAGE.md`
§9: CK's Jenkins CI never runs correctness tests on real gfx1250 hardware):

1. A hand-curated YAML corpus (`specs/features/**/*.yaml`, small
   corner-case-oriented layer descriptors; `specs/workloads/**/*.yaml`,
   larger production-shape-oriented, including a dedicated
   `specs/workloads/fremont/gfx1250_qual.yaml` qualification workload)
   describes convolution *layer shapes*, not kernel configs.
2. At test-binary startup (between `InitGoogleTest` and `RUN_ALL_TESTS`),
   `register_spec_tests` walks the corpus and crosses every layer × 3
   directions (fprop/dgrad/wgrad) × 3 dtypes × **every valid config the
   library itself reports** via `get_valid_configs(arch, shape,
   ALL_RANKED_CONFIGS)` — nobody hand-writes per-kernel test cases; coverage
   is entirely driven by what the library's own dispatch logic claims
   applies to a shape.
3. The **architecture is resolved from the live device** at test-registration
   time (`hipGetDeviceProperties` + `resolve_arch`); if no GPU of a given
   arch is physically present, that arch's cases are never registered (a
   single skip, ctest exit code 77 — standard ctest SKIP semantics), rather
   than either failing or silently not compiling.
4. On gfx950 this produces **11,068 GTest cases from 443 layers in ~4.5
   minutes at 32 workers** — a `ReferenceCache` memoizes CPU-reference
   computation per distinct (layer, direction, dtype) tuple (11,068 cases
   rest on only 2,805 distinct references — a measured 2.5× wall-time win).
5. A `--patience`/`HIPCONV_TEST_PATIENCE` bound prevents a runaway CPU
   reference from hanging the whole suite (times out to a skip instead).

**Directly portable to CK**: CK could declare a similar shape corpus (or
reuse its own `DeviceOp::GetInstances()`/`IsSupportedArgument` instance
metadata as the "valid config" source, analogous to hipConv's
`get_valid_configs`) and auto-generate one GTest case per (shape × instance
× precision), registered/skipped based on whether the CI runner's actual
device is gfx1250 — closing the loop between "an instance claims to support
this shape" and "a real test executed it on real gfx1250 hardware," which is
exactly the gap CK's current Jenkins CI (build-only, software HSA emulator)
leaves open.

**Caveat, applies equally to CK**: hipConv's own `docs/architecture-guards.md`
states plainly: **"hipconv runs no CI of its own, and its multi-GPU builds
are downstream."** There is no automated real-hardware CI *execution* gate
inside hipConv either — the spec-driven suite is comprehensive and
hardware-capable, but the only pre-publication safety net is a **manually
run** "foreign-target build" check (§7). This is the same class of gap CK
has (a good test methodology existing without automated hardware execution
behind it), not something hipConv has already solved end-to-end — treat §6
as a design pattern to adopt, not evidence CK is uniquely behind.

## 7. Build Governance: hipConv's Docs Name Composable Kernel Directly

`docs/multi-arch-convergence.md` documents hipConv's core build discipline —
**host code must be byte-identical across `GPU_TARGETS` shards; only device
kernel bodies vary per `--offload-arch`** — and explicitly calls out CK by
name as a prior offender of the pattern it fixed:

> [CMake-time branching on `GPU_TARGETS` that alters the host-side build
> graph] is **the same violation recorded against composable_kernel and
> MIOpen**.

The practical mechanism (`hipconv_add_arch_lib(<name> TARGETS <gpu>)`
intersecting an architecture's declared `TARGETS` with the build's actual
`GPU_TARGETS`, with a "stand-in" offload target for architectures serving
none of the build's GPUs) exists specifically to prevent per-shard host-symbol
drift breaking MIOpen/TheRock's shard-overlay assumptions. A measured
side-effect: fixing the CDNA4/CDNA5 target intersection dropped a
`gfx950;gfx1250` build's `.hip_fatbin` size from **44.8MB to 38.8MB** —
gfx1250 binaries had been bloated with gfx950-targeted device code compiled
for the wrong offload-arch (and vice versa) before the fix.

**Directly actionable audit item for CK**: verify whether CK's own
multi-target CMake logic (per `GFX950_VS_GFX1250_COVERAGE.md`'s build-gating
findings, e.g. the `gfx12`-regex overlaps discussed there) similarly
over-compiles or under-intersects per-target instance selection across a
combined `gfx950;gfx1250` (or any multi-target) build — this is a real,
previously-documented violation pattern in CK's own history per hipConv's
own citation, worth re-checking whether it's been fully resolved or still
recurs in the specific gfx950/gfx1250 combination given the `gfx12`-regex
issues already independently found in this investigation series.

## 8. Config-Ranking Design Lens (transferable audit pattern, not gfx1250-specific)

`docs/config-ranking.md` documents hipConv's `get_weighted_throughput_index`
(WTI) — a `[0,1]` self-reported utilization estimate per kernel-config used
to rank candidates for a shape. Two transferable disciplines:

- **Self-graded honesty**: "A flat index of 1 outranks every honest
  fraction, which is a reason to make a family's index real" — hipConv
  explicitly audits and documents which kernel families report an honest
  fractional utilization estimate versus a lazy flat 1.0, since a dishonest
  flat score can wrongly out-rank a genuinely better-fitting config.
- **A real, caught performance bug**: `direct_wgrad` once computed an
  argmin *inside* its per-config `is_valid_config` predicate (called once
  per table entry), making config ranking accidentally quadratic — 120µs/layer
  versus 0.7µs once fixed. The documented rule: `is_valid_config` must stay
  cheap and side-effect-free (no env reads, no table scans).

**CK relevance**: if CK's own gfx1250 instance-selection/dispatch heuristics
(the `DeviceOp::GetInstances()` + `IsSupportedArgument` ranking walk
discussed in `AGENTS.md`) have any non-trivial per-shape cost function for
choosing among competing gfx1250 WMMA instances, audit it against both
disciplines above: is any "always applicable" instance family's selection
weight actually just a lazy constant rather than a real fit estimate, and
does any per-instance validity/ranking predicate do expensive
table-scanning work that should instead be precomputed once.

## Summary Table

| # | Finding | Source | CK Actionability |
|---|---|---|---|
| 1 | WMMA C-operand WAR hazard (LLVM bug, still upstream) | hipConv commit `1b0e3e7` | **High** — CK ships the exact hazardous overload (`wmma_gfx12.hpp:417,1067`); audit for `sched_barrier` protection |
| 2 | Sub-dword TDM load races same-wave transpose gather | hipConv commit `ebf2180` | **High** — structurally matches CK's own open `qr_tdm` ping-pong bug; try the same narrow `s_wait_tensorcnt` fix pattern |
| 3 | gfx1250 vs CDNA4 sparse-slot odd/even shift | `bunnies_mi400.hpp:251-253` | Medium — relevant only if CK extends sparse-MMA to gfx1250 |
| 4 | TDM descriptor mutation mid-pipeline deadlocks | `grouped_multi_g_wgrad/kernel.hpp:174-177` | Medium — audit CK's `cluster_load` descriptor reuse discipline |
| 5 | `s_wait_tensorcnt` is per-wave; needs separate `s_wait_dscnt` vs LDS | `grouped_multi_g/kernel.hpp`, toeplitz doc | **High** — same symptom class as CK's `qr_tdm` bug; audit multi-wave TDM issue/wait discipline |
| 6 | Asymmetric wave-role specialization (CU0/CU1 producer split) | `direct/kernel.hpp` | Design idea — differs from CK's symmetric-wave pipelines |
| 7 | Block-diagonal WMMA packing for small-group conv | `grouped_multi_g/kernel.hpp` (3rd independent confirmation) | **High** — directly addresses CK's small-group gfx1250 conv gap |
| 8 | TF32 on gfx1250 via bf16-pair emulation — shipped, tested | `direct/config.hpp`, `kernel.hpp`, tests | **High** — proves CK's TF32-on-gfx1250 gap is closeable, not a hardware ceiling |
| 9 | Spec-driven, live-arch-gated GTest generation | `docs/spec-driven-tests.md` | Process pattern for CK's gfx1250 real-hardware CI gap |
| 10 | No automated real-hardware CI in hipConv either | `docs/architecture-guards.md` | Confirms this is an industry-wide gap, not CK-unique |
| 11 | CK named directly as a past shard-convergence violator | `docs/multi-arch-convergence.md` | Re-audit CK's multi-target CMake instance-selection logic |
| 12 | WTI self-honesty + O(n²) predicate anti-pattern | `docs/config-ranking.md` | Audit lens for CK's own dispatch cost functions |

## Prioritized Recommendations for CK

1. **(Urgent, concrete) Audit CK's gfx1250 bf16→bf16 / fp16→fp16 `amdgcn_mma`
   specializations (`wmma_gfx12.hpp:417,1067`) for the LLVM WMMA
   C-operand WAR hazard** (§1.1). This is a live upstream LLVM bug; CK's
   exposure depends on compiler-scheduling behavior that could change across
   ROCm versions. Add `sched_barrier(0)` at each MMA issue site if not
   already incidentally present.
2. **(High value) Re-examine CK's `qr_tdm` ping-pong prefetch bug against
   both hipConv TDM hazard patterns** (§1.2, §2.3): an in-flight TDM load
   racing a same-wave transpose-read gather, and/or a multi-wave
   `s_wait_tensorcnt` misuse. Both are hardware-verified, narrowly-scoped,
   *fixable* hazard classes on real gfx1250 hardware — not unexplained
   silicon mysteries.
3. **(High value, closes a documented gap) Adopt block-diagonal WMMA
   packing for CK's small-group/small-channel grouped-conv WMMA instances**
   (§4) — independently confirmed correct and valuable by three separate
   projects now.
4. **(High value, proven feasible) Implement gfx1250 TF32 via the bf16-pair
   emulation technique** (§5) hipConv ships and tests — closes a gap CK
   currently treats as unsupported by construction.
5. **(Process) Build a spec-corpus-driven, live-arch-gated GTest generation
   harness for CK's gfx1250 correctness tests** (§6), and separately pursue
   real-hardware CI execution (hipConv has the former but not the latter
   either — CK should aim to have both).
6. **(Audit) Re-verify CK's multi-target CMake instance-selection logic
   for the shard-convergence violation hipConv's own docs cite CK for by
   name** (§7), specifically for the `gfx950;gfx1250` combination given the
   `gfx12`-regex overlap issues already found in this investigation series.
