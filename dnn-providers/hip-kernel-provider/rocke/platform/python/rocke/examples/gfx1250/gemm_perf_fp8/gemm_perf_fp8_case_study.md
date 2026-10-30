# Case Study: fp8 GEMM on gfx1250 -- load width, then the K-loop

This file records two pieces of work on the gfx1250 fp8e4m3 A/B -> bf16 C
universal GEMM (layout RCR, wave32). **Only ratios are reported**; absolute
figures are kept in the internal record, per `platform/AGENTS.md` §Compliance.
Every compared arm passed an on-silicon numeric gate in the measured run.

* **Part 1 (§1-§11): global-load width.** The GEMM emitted 8-byte A/B loads
  where the hardware permits 16. Widen them, and establish whether that is
  worth shipping. Measured on the 16x16x64 WMMA atom, `pipeline=mem` /
  `scheduler=intrawave` / `epilogue=default`, at 4096x4096x8192 (compute-bound
  by arithmetic intensity) and 64x16384x4096 (its memory-bound counterpart),
  against the same kernel with the load width pinned to its previous value,
  built from the same source tree in the same binary. HIP-event-timed
  (20 iterations, 5 warmup). **Result:** at the measured configurations the
  wider load is faster, and the margin depends strongly on the tile geometry
  it is paired with. It is shipped on by default.
* **Part 2 (§12-§19): the K-loop.** On the dense 16x16x128 atom at
  4096x4096x65536, three opt-in K-loop traits (split barrier, alternating TDM
  issuers, early refill) against the same kernel without them, all in one
  process, and the kernel against a WMMA ceiling measured on the same operand
  data. Every result in §13-§15 comes from the two tools in this folder and
  can be regenerated with the commands in §19; §16 is labelled separately.

---

## 1. The change

One argument. `helpers/spec.py::choose_load_vec` caps the vector at
`16 // elem_bytes` — the 4-dword `buffer_load` limit — and
`gemm_universal._choose_load_vec` never passed `elem_bytes`. The picker's
default of 2 therefore capped a **1-byte** operand at 8 elements, spending
half of every A/B load:

```
before:  load <8  x i8>, ptr addrspace(1) ..., align 8
after:   load <16 x i8>, ptr addrspace(1) ..., align 16
```

The fix resolves the width from `data.dtype_a` via `_ab_dtype_bytes`, which
already existed. `TraitSpec.ab_load_elem_bytes: Optional[int] = None` overrides
it; `None` resolves from the dtype, an explicit `2` reproduces the older
emission. The knob exists so both widths can be built in one binary and
compared — which, per §3, turned out to be the only trustworthy way to measure
anything on this machine.

**Blast radius is narrow.** f16/bf16 resolve to 2 either way, so every 2-byte
kernel is byte-identical. Only 1-byte operands move. The C++ picker's
hardcoded `{8,4,2,1}` candidate ladder (permanently `elem_bytes=2`) was
replaced with `16/elem_bytes`; the conv and MoE adapters pass `2` explicitly so
their emission is unchanged.

---

## 2. What we got wrong first

Three times, and each mistake is more instructive than the fix.

**(a) The shape prediction was backwards.** An arithmetic-intensity analysis
said 4096×4096×8192 sits far above the roofline ridge (AI reduces to the
harmonic mean of M and N when K is large, so a square shape is compute-bound),
therefore a memory-side lever should do nothing there and should pay off on a
skinny shape. The opposite was observed: a clear effect on the square shape,
flat on the skinny one. AI correctly predicted which side of the ridge the
shape sits on; it did not predict that load *width* would matter there anyway.

**(b) A cross-run comparison manufactured a regression that did not exist.**
The first evaluation compared a 16-byte ladder run against an 8-byte ladder run
from hours earlier and concluded the change was ~5% slower. It was not — the
environment had changed between the two runs (§5). The sign of the result was
wrong, not just the magnitude.

**(c) The first explanation for (b) was also wrong.** The drift was attributed
to a drop between two memory-clock levels, with a roofline argument built on
top. A machine-config snapshot taken days earlier showed `pp_dpm_mclk` offering
**one** level, identical to the current state; the higher figure had been the
shader clock (SCLK). There had been no clock change, and the real cause (§5)
was in the software stack, not the hardware. The lesson is narrow and worth stating: a
plausible mechanism that explains the size of an effect is not evidence that
the mechanism occurred.

---

## 3. Method: A/B must be interleaved in one process

This is the durable finding.

Measuring arm A, then arm B, then comparing, is invalid on this machine at the
scale of effect being chased. An identical kernel — same spec, same build path,
same measurement code — was re-measured hours apart and differed by more than
any effect in this study.

The experiment that settled the question builds all arms once and times them
**round-robin inside a single process**, cold round discarded, repeated with
the arm order alternating per round to rule out an ordering effect. Whatever
degrades the machine then hits every arm equally, and the ratio stays valid
even though the absolute level does not.

A second trap sits in the harness itself. The sweep's `lever` stage A/Bs arms
**on the champion geometry the search found** — and that search ran under one
arm's emission. When a lever changes which geometry is optimal, as this one
does, that answers "what does the width do to *this* geometry", not "is the
best achievable configuration better". Those differ by a large factor here
(§4). Both are true statements; only the second answers "should we ship it".

---

## 4. Results (ratios, 4096×4096×8192)

Two geometries × two widths, interleaved, two independent runs:

| geometry | 16-byte vs 8-byte |
|---|---|
| `tile 256×128×128, warp 4×2` | 1.015× – 1.032× |
| `tile 128×128×128, warp 2×2` | **1.586× – 1.589×** |

**Best achievable vs best achievable: 1.074× – 1.088×.**

Both runs agree within ~1% on every arm but one (4.5% spread). The width helps
on both geometries and helps substantially on the one the 16-byte search
selects — which is why the single-champion A/B in §3 reports ~1.59× while the
honest best-vs-best figure is ~1.08×. Quote the latter.

At 64×16384×4096 the lever measured 0.9867×, i.e. flat within a ~2–3% spread.
These are bounded claims about these measured configurations on this target;
no general superiority is implied.

---

## 5. The toolchain changed underneath the measurements

An identical spec, re-measured through the identical code path used for the
original figure, came back **~10–15% slower** hours later. Three explanations
were considered and two were wrong.

*Not thermal*: junction and memory temperatures were well below any throttling
point. *Not clocks*: the DPM tables were
byte-identical to a snapshot taken days earlier (§2c). *Not another tenant*:
`rocm-smi --showpids` and `/sys/class/kfd/kfd/proc/` both report **zero**
compute processes attached to the device.

One thing that definitely changed is the compiler. The ROCm installation that
`$ROCM_PATH` resolves to was repointed to a newer toolchain partway through the
day, and `/proc/<pid>/maps` confirms the kernel-build process now loads its
`libamd_comgr` from that new tree. So every figure recorded before the switch
was produced by one compiler and every figure after it by another.

**But the compiler is not the whole story, and assuming it was would have been
the third wrong explanation.** The toolchains can be A/B'd properly: emit the
LLVM IR once (identical input, hash-checked), compile it under each `comgr` in
separate processes, then time the resulting blobs round-robin in a single
process under one HIP runtime. Done that way the newer toolchain is **0.5–2.3%
slower** on these two kernels — and replaying one fixed blob under each
`libamdhip64` shows **no measurable runtime effect** at all. The generated ISA
differs only in scalar branch lowering: identical instruction count, identical
VGPR and LDS, nothing touched in the vector, MMA or memory stream.

That leaves most of the gap unattributed. The same spec built with the *same*
compiler still measures materially slower than it did before the update, and
the remaining differences in the environment — notably a driver version change
— cannot be tested without reinstalling. Cross-process measurement noise here
is around 4–5%, which absorbs part of the gap but not all of it.

The honest position is therefore: the environment changed in at least two ways,
one of them measurable and small, and **a pre-update number cannot be compared
with a post-update number** regardless of which part dominates. That is the
operative lesson, and it does not depend on resolving the cause.

Consequences, which apply to any future work in this folder:

- Ratios from a single interleaved process are sound. Absolute levels are not
  comparable across sessions, and a toolchain switch makes them incomparable
  even on an otherwise idle, thermally stable, clock-stable machine.
- A stored champion from an earlier ladder is **not** a valid baseline. Re-run
  the ladder in full, or interleave.
- **The HSACO cache key does not include the toolchain.** `sweep._spec_hash`
  covers the spec and the arch, so a cache populated before a toolchain change
  will happily serve stale objects to a run after it. Separate output
  directories per run avoided this here, but a resumed run (`--stages`)
  spanning an update would silently mix two compilers' output.
- The harness records no environment alongside its results. It should capture
  the resolved ROCm path and comgr version, clocks, temperature and
  utilisation at the start and end of every run. Until it does, a code effect
  and a toolchain that moved look identical in the output — which is exactly
  how §2(b) happened.

---

## 6. Correctness

Every arm in every comparison passed the exact numeric gate **in the measured
run** — `max_abs_diff = 0.0`, not a tolerance. Inputs are integers in −5..5,
each exact in e4m3; with `|sum| ≤ 25·K` the fp32 accumulator is exact for any
K below ~670k, so summation order cannot change the result and only the final
RNE to bf16 rounds, which the reference reproduces. The harness refuses a K
that would break the argument rather than widening the gate.

Dual-engine byte-identity holds: 69 GREEN / 1 DRIFT, the DRIFT being a
documented pre-existing failure in an unrelated attention family. Parity
configs 13 (resolved width) and 14 (pinned width) are both byte-identical
between the Python and C++ engines, so the knob is covered in both states.

---

## 7. Levers tried, and one that is not available

Swept: tile geometry (`tile_m/n/k`, `warp_m/n`), `lds_k_pad`, `lds_swizzle`,
`waves_per_eu`. Of these only `lds_k_pad` and the tile geometry moved the
result materially; `waves_per_eu` spans under a percent at the champion.

**`direct_to_lds` is not a lever on this path.** Every combination setting it
is rejected with *"WMMA path does not support direct_to_lds on gfx1250"* — 80
of 120 trait candidates per finalist died on that one line before it was
removed from the config. Listing it implied a knob the kernel does not have.

**Not attempted:** an ISA/occupancy inspection of why the wider load helps. The
plausible mechanisms (VGPR pressure from a 4-register vector, a halved
`a/b_buffer_load_inst_num` changing the `sched_group_barrier` interleave) are
unverified and are deliberately not asserted here.

---

## 8. The widening changed what a good `lds_k_pad` is

`lds_k_pad` is in **elements**, not bytes — `_lds_k = block_k + lds_k_pad`, and
the LDS row stride in bytes is that times the element width. Bank conflicts are
a byte-level effect (32 banks × 4 B = a 128 B cycle), so **the same numeric pad
means something different for a 1-byte operand than for a 2-byte one**. The
values here were inherited from a bf16 sweep and were wrong for fp8 three ways.

**They sampled half the space.** At `tile_k=64`, bf16 pads of
`[0, 8, 24, 40, 56]` give bank offsets `{0, 16, 48, 80, 112}`. The same numbers
on fp8 at `tile_k=128` give `{0, 8, 24, 40, 56}` — only the lower half of the
bank range is explored.

**The winner sat at the top of the range.** Every top-five finalist took the
largest value in the list. A winner at the edge of a search range is the
signature of a truncated search, not of an optimum.

**And the widening made all the non-zero values misaligned.** This is the part
created by this change. Phase 1 made the LDS store `<16 x i8>`, so a row stride
that is not a multiple of 16 puts alternate rows off a 16-byte boundary. Every
non-zero pad in the inherited list is ≡ 8 (mod 16). The emitter claims
`align 16` on that store **regardless of the pad**:

```
lds_k_pad=56  (row 184 B, 184 % 16 = 8):   store <16 x i8> ... align 16
```

Before the widening the store was `<8 x i8>` and 184 % 8 = 0, so the claim held.
It is now untrue for odd rows. Every measured arm still passed the exact numeric
gate, so there is no observed miscompile — the hardware tolerates the access —
but an alignment attribute is an assertion to LLVM, not a formatting detail, and
a different compiler version is entitled to act on it.

### The good pads can be derived instead of swept

LDS has 32 banks of 4 bytes. Row `r` of the staged tile begins at bank
`(r·S/4) mod 32` for a row stride of `S = tile_k + pad` bytes, so the per-row
bank stride is `(pad/4) mod 32` and the number of distinct banks the rows land
on is `32 / gcd(stride, 32)`. More distinct banks is better; one bank is a
full 32-way conflict.

Swept interleaved at the champion geometry, every class the model predicts
shows up, and the two values picked *because* the model predicted them —
32 and 64 — behave as predicted:

| pad | distinct banks | 16B-aligned | measured, relative to best |
|---:|---:|:--:|---:|
| 0 | 1 | yes | 3.38× worse |
| 16 | 8 | yes | 1.01× |
| 32 | 4 | yes | 1.32× worse |
| 48 | 8 | yes | 1.02× |
| 56 | 16 | **no** | 1.24× worse |
| 64 | 2 | yes | 1.81× worse |
| 80 | 8 | yes | **1.00×** |
| 96 | 4 | yes | 1.28× worse |
| 112 | 8 | yes | 1.04× |

Three things follow.

**The pathology is the bank offset, not the number zero.** `pad = 128` gives
the same offset 0 as `pad = 0` and is just as catastrophic (worse, in fact —
it also costs LDS). So the space is periodic in 128 bytes for a 1-byte operand,
which *bounds* the search: there is nothing useful beyond 112.

**The good set is computable.** Wanting `gcd((pad/4) mod 32, 32) == 4` (eight
distinct banks) and `pad % 16 == 0` (so the `<16 x i8>` store's alignment claim
is true) yields exactly `[16, 48, 80, 112]` — which is what the configs here
sweep, plus `0` kept as the known-bad anchor that makes the others
interpretable. Not as the screening pin, though (§9); that is 48.

**Banking and alignment are in tension, and alignment wins.** Sixteen distinct
banks would need `gcd == 2`, i.e. `pad ∈ {8, 24, 40, 56, 72, 88, 104, 120}` —
*every one* of which is ≡ 8 (mod 16) and therefore misaligns the 16-byte store.
`pad = 56` is the measured proof: best-in-class banking, worst-in-class
alignment, and it lands 24% behind `pad = 80`. That is also an independent
confirmation of the alignment finding above, reached from a different
direction, and it explains why 56 won the inherited sweep — the aligned values
simply were not in it.

The practical upshot for anyone porting this list to another operand width:
**do not copy the numbers, recompute them.** `pad` is in elements, so the whole
table shifts with the element size.

**Fixed.** The emitter now derives the alignment attribute from the actual row
stride — `gcd(vec, innermost_dim) × elem_bytes` — for both the LDS store and
the LDS load, so a misaligned pad produces a weaker-but-true claim instead of
a stronger false one.

Fixing it surfaced a second, worse defect in the same place. The **load** path
kept its own element-size map and that map omitted the 8-bit floats, so they
fell through to 2 bytes and a 16-element fp8 read claimed `align 32` for a
16-byte access. Unlike the store defect, **no padding choice avoided it**: the
16-aligned pads this file recommends mitigate the store and never touched the
load, so the configuration recommended above was itself emitting a false
`align 32` until this was fixed. The two element-size maps are now a single
shared table, since their having drifted apart is what caused it.

Parity config 15 covers the case: `lds_k_pad=56` gives a row stride the vector
width does not divide. Every other config in the family has a stride it does
divide, so they all pass with the bug present — which is how it survived.

## 9. The screening pin nearly hid the answer

Covered in full in [README.md](README.md); summarised because it shaped every
number above. Stage 1 ranks tile geometry with traits held at
`screening_traits`. Pinned at `lds_k_pad = 0`, the configuration that went on to
win ranked **18th of 180**, two places above the `tile_finalists = 20` cutoff,
and **no** `tile_k = 256` geometry reached the finalists at all (best rank 43).
Re-screened at a non-zero pin, the ranking inverted:

| `tile_k` | top-20 at pin 0 | top-20 at pin 56 |
|---|---:|---:|
| 64 | 17 | 2 |
| 128 | 3 | 14 |
| 256 | 0 | 4 |

The search was fragile, not wrong — after stage 2 tuned traits, both screens
finished within noise of each other. But a cutoff two ranks lower would have
silently reported a worse configuration as the answer, with nothing in the
output indicating anything had been missed.

---

## 10. Remaining work

1. Re-measure once the machine is healthy, and confirm the §4 ratios hold at a
   stable operating point. The §4 numbers were taken with the old, half-range
   `lds_k_pad` list (§8), so the re-run should also establish whether the new
   16-aligned values move the champion.
2. *Done.* The LDS store's alignment attribute is derived from the actual row
   stride (§8).
3. *Done.* The sweep records machine state per run (`machine_state.json`,
   written as each stage starts and at a fault) (§5).
4. *Documented, not enforced.* The README states that a `lever` ratio is an
   attribution on the champion's geometry, not a speedup; the stage does not
   yet refuse to print one when an arm's own champion differs (§3).
5. Investigate *why* the wider load helps, before generalising it to other
   1-byte paths — notably `moe_gemm_fused`, which still passes the default
   width and would inherit the same halving.
6. The DirectToLDS load path hardcodes 2 bytes per element, in **both** engines
   (`_DTL_HALVES = _DTL_DWORDS * 2` with `_DTL_BYTES_PER_LANE = _DTL_DWORDS * 4`;
   `a_half_bytes = block_m * block_k * 2` in `gemm_build_load.cpp`). For a
   1-byte operand it budgets 16 bytes per lane chunk while moving 8, and it
   disagrees with its own LDS allocation, which *is* sized correctly from
   `_ab_dtype_bytes`.

   It is **not** reachable from this path — `direct_to_lds` is rejected on the
   gfx1250 WMMA pipeline — but it **is** reachable with fp8/bf8 on gfx950 and
   gfx942, so this is a live defect on CDNA rather than a latent one. It is not
   fixed here because the arches that can reach it are not the arch this work
   has hardware for: changing both engines would turn the byte-identity gate
   GREEN by making them agree, without any numeric check that they agree on the
   *right* answer — which is how the defect arose in the first place.

---

## 11. Artifacts

- Harness, configs and the staged-sweep design: this folder, see
  [README.md](README.md).
- Kernel change: `instances/common/gemm_universal.py` (`_choose_load_vec`,
  `TraitSpec.ab_load_elem_bytes`) and its C++ mirrors in
  `cpp/helpers/spec.cpp`, `cpp/instances/common/gemm_mma.cpp`.
- Parity coverage: `tests/instances/parity/gemm_emit.{py,c}` configs 13 and 14.
- Lowering assertions, including the counterfactual that the override still
  produces the narrow load:
  `tests/instances/test_gemm_universal_fp8_lowering.py`.
- Measured results are kept in the internal record, not in this repository.

---

# Part 2 -- The K-loop at 4096×4096×65536

## 12. Scope and the fixed geometry

Problem 4096×4096×65536, fp8e4m3 A/B, bf16 C, fp32 accumulation, RCR, on
gfx1250. Every arm shares one geometry and differs only in K-loop traits:

| | |
|---|---|
| WMMA atom | dense 16×16×128 `f8f6f4` (format immediate = fp8), `<16 x i32>` operands |
| tile | 256×256×256, 4×4 warps (16 waves, 512 threads) |
| A/B staging | TDM, depth 2 (ping-pong), `lds_k_pad = 48` |
| epilogue | `cshuffle`, `pipeline=mem`, `scheduler=intrawave` |

The two LDS halves of this tile leave room for one workgroup per compute
unit. The geometry was chosen with the staged sweep (Part 1,
[README.md](README.md)) and is held fixed here: this part asks what the K-loop
can recover once the geometry is settled. `fp8_trait_ab.champion_spec()` is the
single definition of it.

## 13. The ceiling depends on the operand data

The usual roofline is a register-resident probe: many independent WMMA chains,
no memory traffic, one operand bit pattern in every register. On this target
that last detail is not neutral. Repeating the probe with per-lane operand
bytes drawn from the GEMM's own input sets, alternating between two operand
sets so consecutive WMMAs see different bits, gives a lower ceiling for the
same instruction:

| probe operand data | ceiling vs constant data (peak) |
|---|---|
| one constant pattern | 1.00 |
| integers in −5..5 (the exact-check inputs) | 0.83 |
| `normal(0, 0.1)` rounded to e4m3 | 0.74 |

More toggling in the datapath draws more power and the clock comes down; the
cycle count per WMMA does not change. The GEMM shows the same effect: the
same kernel runs the same cycles on both input sets, at a lower clock on the
random one.

**So a GEMM is compared only against the ceiling measured on matching data,
in the same session.** Comparing an integer-input GEMM with the constant-data
ceiling, as was done at first, understates the kernel by a wide margin, and
the error is not even consistent between input sets.

## 14. The three K-loop traits

All three are opt-in `TraitSpec` fields, default off (the kernel is
byte-identical when off), gated to the TDM WMMA `mem` pipeline, and mirrored
in the C++ engine. Instruction-level traces (ATT) of the depth-2 ping-pong
showed the WMMA pipe idle for a large share of the K-loop, with the
workgroup barrier the largest single cause: every wave stops at a full
`s_barrier` each K-step, and the step lasts about as long as one TDM fill plus
the tail of the issuing waves' work.

1. **`split_barrier`.** The per-K-step barrier becomes a signal and a wait.
   A wave signals as soon as its last fragment reads of the tile have drained
   (and the fill it needs has landed), runs the tile's last k-atom of WMMAs,
   and waits only at the top of the next step, so the barrier skew between
   waves is absorbed by those WMMAs. Depth 2 and the deeper ring both have it.
2. **`tdm_alternate_issuers`.** The fills of LDS half *p* are issued by waves
   2*p* and 2*p*+1 instead of always waves 0 and 1. The waves that wait for one
   fill then never issue the next, so the refill goes out as soon as the
   barrier releases rather than after those waves' last WMMAs. Each issuer
   still has one fill outstanding, so every TENSORcnt wait stays exact.
3. **`tdm_early_refill`.** A half is refilled as soon as every wave has read
   it, not at the top of the next iteration: after its signal, the pair owning
   the half waits on that barrier and issues the fill of the tile two ahead, so
   two fills are in flight within the same two halves. The wait sits right
   before the TDM issue, after the descriptor arithmetic, so that arithmetic
   overlaps it. The loop runs two K-tiles per iteration with static LDS parity
   (immediate fragment-read offsets) over an even tile count; with an odd
   count the extra tile lies past K and is zero-filled. **Enable it for long
   K.** Against `alternate_issuers` it adds about 1% at peak at both measured
   K -- 1.013× / 1.011× (integer / random) at K = 65536 and 1.013× / 1.008×
   at K = 8192 -- but under sustained load it measured 1.012× at K = 65536 and
   0.989× at K = 8192. A short loop spends a larger share on the second
   prologue fill and, for an odd tile count, on the zero-filled tile.

### Results

`fp8_trait_ab`, one process, arms interleaved with the order alternating per
round, every arm verified on both input sets first (§17). Ratios are against
the `tdm_d2` arm; peak is the median of 15 samples, sustained of 3. The
WMMA ceilings measured before and after the A/B agreed within 0.3%, and no
other GPU process appeared during the run.

| arm | peak, integer inputs | peak, random inputs | sustained, random inputs |
|---|---|---|---|
| `tdm_d2` (ping-pong, joint barrier) | 1.000× | 1.000× | 1.000× |
| `split_barrier` | 1.016× | 1.012× | 1.025× |
| `alternate_issuers` (split + alternating) | 1.047× | 1.041× | 1.032× |
| `early_refill` (all three) | **1.061×** | **1.052×** | **1.044×** |

Against the same-session ceiling on matching data (§13), the `early_refill`
arm reaches **75.3%** of the integer-data ceiling and **80.3%** of the
random-data ceiling at peak.

## 15. Peak and sustained are different regimes

The peak protocol idles the GPU for 25 ms before every sample and times 1
warmup plus 3 launches, which fits inside the clock's boost window. Run
back-to-back (20 warmup, 400 timed), the same kernel settles at a much lower
clock: it keeps about 65% of its peak throughput on random data, while the
register-resident probe keeps about 96%. A shader-clock counter
read inside the kernel confirmed it: equal cycle counts in both regimes, a
lower clock under sustained load (`clock_probe --sustained`). Trait ratios are similar in both regimes,
but **an absolute figure means nothing without its protocol**, and a
percentage of the ceiling is only meaningful at peak, against the peak
ceiling.

## 16. What did not help

**Diagnostic, historical, not reproducible from this folder:** these variants
were never committed. Each was built, verified exact and timed interleaved
against the champion of its day, on the same device; the ratios are those
sessions' and are listed so nobody repeats the experiments without a new
reason.

* **K=64 atom with fences or prefetch.** Fewer cycles in the trace did not
  survive in-kernel clock measurement: equal cycles to the K=128 atom.
* **8 waves per SIMD** (two workgroups per CU at smaller tiles): about 0.94×
  of the best 4-wave configuration. 1 wave per SIMD reaches well under half
  the WMMA rate; 4 is the working point.
* **Wider workgroups** (4×8 warps on the dense atom) collapse, and "fat"
  waves holding more of the tile run out of VGPRs.
* **Direct-to-LDS loads instead of TDM:** exact, about 0.94×. Asynchronous
  global loads in the GEMM loop order deliver the same as TDM; cluster
  multicast merges requests and gains nothing.
* **Unrolling two K-tiles without the early refill** is slower: at the
  256-VGPR budget the allocator shuffles fragment registers with copies and
  hazard waits. **Early refill on the non-unrolled loop** is slightly slower.
* **Per-wave priority changes, per-slot priority, and carrying WMMAs across
  the barrier:** no measurable change.
* **Hand-edited assembly.** Reordering the compiler's output (refill ahead of
  independent WMMAs, the barrier wait moved after the descriptor arithmetic)
  matched the committed `tdm_early_refill` schedule within noise; that trait
  is the same schedule expressed in rocKE's IR, with zero spills.
* **`lds_swizzle`** is rejected on the WMMA path: a no-op under TDM and wrong
  without it.

## 17. Measurement protocol

What made the ratios above stable, in the order the mistakes were made:

1. **Interleave in one process** (§3). Separate runs of the same kernel can
   differ by more than any effect here.
2. **Verify every arm in the measured run, before timing any.** Integer inputs
   must match the host reference bit for bit; random inputs must stay within
   2^-7 of the largest output (bf16 output rounding is up to 2^-8); each on 5
   launches, so a race shows up as launch-to-launch drift. `fp8_trait_ab`
   refuses to time anything if one arm fails.
3. **Measure the ceiling in the same session, on matching data** (§13), before
   and after the GEMM: if the two disagree by more than about 1%, the session
   is suspect.
4. **Make sure the GPU is yours.** Another user's kernels share the device and
   skew every arm, and short jobs are invisible to a one-shot check. Use
   `rocm-smi --showpids` (it also lists processes owned by other users), wait
   until it has been empty for a minute, and poll it throughout the run;
   discard the run if anything appears. `fp8_trait_ab` reports the process
   list at its start and end, which is necessary, not sufficient.
5. **Watch for a degraded device.** Once, a few compute units drifted into a
   state where they needed 13–25% more cycles for identical work, at an equal
   or higher clock; it followed the hardware location, not the tile, and a
   reboot cleared it. The GEMM arms and the TDM feed probe both run exactly
   one round of workgroups, so a few slow units set their time while the
   WMMA probe's large grid hides them: the symptoms are GEMM arms and a TDM
   feed time per K-step far off an earlier session while the WMMA ceiling
   looks normal. `clock_probe` checks for it directly: it stamps each wave's
   shader cycles and realtime at start and end, and flags the slowest
   workgroup when it needs more than 10% more cycles than the rest for the
   same work. Compare realtime only within a wave: on one boot a fixed half
   of the workgroups read realtime about half a millisecond ahead of the
   other half, with identical lifetimes, so cross-workgroup start and end
   times were meaningless; the probe detects that against the event-timed
   launch and says so.

## 18. Where the remaining gap is

After the three traits, an ATT trace of the `early_refill` arm on random
inputs still shows the WMMA pipe idle for about a quarter of the K-loop, and
the barrier remains the largest cause at about 14% of time (diagnostic: the
bubble-accounting scripts behind these two figures are not in this folder).
With two fills in flight the issuing waves now wait several times longer for
each fill: the K-step is bounded by the rate at which TDM fills LDS, not by
instruction placement. The TDM feed probe, which is in this folder, agrees:
the same loop with the WMMAs removed (`memory_probe tdm`) takes about 77%
(random inputs) to 80% (integer inputs) of the `early_refill` arm's time per
K-step, so most of each step is the fill itself. Further gains at this geometry need a faster fill (or fewer bytes per
step), not a better schedule.

To look for yourself, capture an ATT trace with the WaveScope tool
(`dsl_docs/optimization/utilities/tools/wavescope/capture_wavescope_trace.py`,
see `platform/AGENTS.md` §Profiling) around a script that builds one arm with
`fp8_trait_ab.arm_spec(...)` and launches it once.

## 19. Reproducing Part 2

Needs a gfx1250 device, a ROCm install (`ROCM_PATH`; the probe compiles with
its `hipcc`), and numpy. Keep output outside the repository.

```sh
export PYTHONPATH=<platform>/python ROCM_PATH=<rocm>
P=rocke.examples.gfx1250.gemm_perf_fp8

# 0. the GPU must be yours (§17.4): empty for a minute, and stays empty;
#    and healthy (§17.5): the probe should end with "health: ok"
rocm-smi --showpids
python3 -m $P.clock_probe

# 1. ceilings before, on each input set, peak and sustained (§13, §15)
for d in const int rand; do
  python3 -m $P.wmma_roofline_probe -i f32_16x16x128_f8f6f4.fp8 --data $d
  python3 -m $P.wmma_roofline_probe -i f32_16x16x128_f8f6f4.fp8 --data $d \
      --gap-ms 0 --warmup 20 --iters 50
done

# 1b. memory side: streaming read/copy bandwidth, and the TDM feed floor (§18)
python3 -m $P.memory_probe

# 2. the four arms: verify all, then peak (int, rand) and sustained (rand)
python3 -m $P.fp8_trait_ab --json <out>/ab.json

# 2b. the early refill at a shorter K (§14)
python3 -m $P.fp8_trait_ab --k 8192 --arms alternate_issuers,early_refill

# 3. ceilings after, as in step 1; before and after should agree within ~1%
```

`fp8_trait_ab` prints each arm's ratio to the first arm and its own figure
(for your record); `--arms` takes any subset in any order, the first being the
reference, `--data` and `--rounds` narrow the run, and `--k` changes the
problem (the exact integer check needs 25·K < 2^24). The whole sequence takes
on the order of ten minutes, most of it the host references and the
sustained rounds. The
percentage of the ceiling is the `early_refill` arm's peak figure divided by
the same-session probe figure on the same data.

**Artifacts.** `fp8_trait_ab.py` (arms, verification, interleaved timing),
`fp8_data.py` (the two input sets and the random-set bound),
`wmma_roofline_probe.py` (the ceiling), `memory_probe.py` (bandwidth and the
TDM feed floor), `clock_probe.py` (in-kernel clock and the health check);
their CPU tests are `tests/test_fp8_trait_ab.py`, `tests/test_memory_probe.py`
and `tests/test_clock_probe.py`. The traits are in
`instances/common/gemm_universal.py` and its C++ mirror; byte-identity configs
22–26 in `tests/instances/parity/gemm_emit.{py,c}` cover them.
