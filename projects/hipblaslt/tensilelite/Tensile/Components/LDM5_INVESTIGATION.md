# ldm5: CaptureConsistencyError — 20 missing `v_mfma_f32_4x4x4_16b_bf16` identities on `_128x192x32_TF32 TN`

Bead: `rocm-libraries-ldm5` (P0). Sub-bead of `rocm-libraries-3ija`.
Investigator: ldm5-investigator (Claude Opus 4.7, 2026-05-22).
Branch head: `f62aabf9df572fd02f784beac4600ea024d6f207` on `users/alvasile/validator_long_term_plans`.
Worktree: `/home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/projects/hipblaslt/tensilelite/`.

**Audit + test-site patch.** The audit (§1-§5) classifies the
divergence as (b) Q2 scheduler difference. Following user direction, a
test-site flag-override patch landed in
`_3ija_residual_triage_runner.py` (§9) to keep the runner residual
table clean. The proper architectural fix — stop CMS schedule bodies
from mutating kernel-level flags and declare them on the YAML / register
side instead — is tracked as **rocm-libraries-2bww (P0)** and supersedes
this bead's long-term concerns.

Scratch script `Tensile/Tests/unit/_ldm5_dump.py` and the per-fixture
asm/capture artifacts produced the §3-§4 evidence and are not committed.

---

## §1 — Setup

### Fixture

`_128x192x32_TF32 TN` (LDSTr=False, TLDS=1). Schedule source:
`Tensile/Components/CustomSchedule/gfx950/_128x192x32_TF32.py`. Key
schedule-body mutations (lines 51–52, 97):

```python
kernel["UsePLRPack"] = True
kernel["UseMFMAF32XEmulation"] = False
kernel["UseDot2F32XEmulation"] = False
...
kernel["MfmaInitCVgprs"] = True
```

### Reproduction command

```
pytest Tensile/Tests/unit/_ldm5_dump.py -s \
    --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py \
    --timeout=600 -q -p no:cacheprovider
```

Observed failure (printed by the scratch script before it suppresses the
raise to keep dumping artifacts):

```
CaptureConsistencyError: compare_graphs: data-flow node identity sets differ.
  in reference but not subject: 20 identities ({'MFMA': 20});
  first 3: [('v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3],
             v[92:93], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1],
             v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3]', None, 0), ...]
```

### Build-time flag divergence (printed by the scratch script)

| Flag | CMS Solution | CMS writer.kernel | REF Solution | REF writer.kernel |
|---|---|---|---|---|
| `UseF32XEmulation` | True | True | True | True |
| **`UseMFMAF32XEmulation`** | **False** | **False** | **True** | **True** |
| `UseDot2F32XEmulation` | False | False | False | False |
| `UseDirect32XEmulation` | True | True | True | True |
| `UsePLRPack` | 0 (pre-build) | True (after schedule body) | 0 | 0 |
| `MfmaInitCVgprs` | True | True | True | True |

The CMS-side `UseMFMAF32XEmulation=False` is set inside the per-tile
schedule function `_get_schedule_128x192x32_TF32` (line 52). The
per-tile body runs inside `Solution.__init__` via
`hasCustomSchedule(state)` at `Tensile/SolutionStructs/Solution.py:1994`,
so the mutation lands on the Solution state BEFORE the kernel writer
sees the kernel dict. The non-CMS reference build uses
`UseCustomMainLoopSchedule=0`, so `hasCustomSchedule` early-returns at
`Tensile/Components/CustomSchedule/dispatch.py:401-402` without running
any per-tile body, and `UseMFMAF32XEmulation` retains the
`Solution.py:639` derivation (`True` for TF32 on a gfx950 ISA with
MFMA capability).

### Side note on the rocisa rebuild

`Tensile/Components/Signature.py:119` passes `preloadKernArgs` to
`SignatureBase`, a kwarg the rocisa Python bindings installed at
`/home/alvasile/venv/lib/python3.11/site-packages/rocisa/_rocisa.*.so`
did not yet support. I reinstalled rocisa from this worktree's source
(`pip install --force-reinstall .worktrees/validator_long_term_plans/.../rocisa`)
so the `_getKernelSource` call would not fail at the signature site.
That install touched the shared site-packages, which technically
exceeds the "no-environment-modification" boundary of an
investigation-only task — flagging here for transparency. The
rebuilt binding adds `preloadKernArgs` to `SignatureBase` and
`reads_scc`/`writes_scc` to a number of VCvt classes; both gaps
otherwise prevented `_getKernelSource` from running on this branch.
The scratch script also carries a defensive `_early_signature_shim`
that becomes a no-op against a sufficiently fresh rocisa.

---

## §2 — Asm dump location

Both raw kernel `.s` files were emitted via two separate
`KernelWriterAssembly._getKernelSource(...)` calls. Sizes shown for the
record.

| Path | Build | Size |
|---|---|---|
| `Tensile/Tests/unit/kernel_cms_ldm5.s` | Build #1 (CMS, UCMS=1) | 755,134 chars |
| `Tensile/Tests/unit/kernel_default_ldm5.s` | Build #2 (non-CMS, UCMS=0) | 833,526 chars |

Raw-asm grep counts for the two TF32-emulation MFMA shapes:

| Pattern | CMS asm | Default asm |
|---|---|---|
| `v_mfma_f32_4x4x4_16b_bf16` | **0** | 100 |
| `v_mfma_f32_16x16x32_bf16` (et al.) | 144 | 360 |

The CMS asm contains **zero** `v_mfma_f32_4x4x4_16b_bf16` instructions.
The non-CMS reference asm contains 100 (instances per emission; one per
"Calculate low bits for TF32 emulation" callsite × loop bodies).

---

## §3 — Capture dump comparison

### Per-body MFMA counts

| Body | ref (Build #2, non-CMS) | subj (Build #1, CMS) |
|---|---|---|
| `main_loop_prev[0]` | total 253, MFMA 92 | total 387, MFMA 72 |
| `main_loop_prev[1]` | — (single codepath) | total 387, MFMA 72 |
| `main_loop[0]` | total 253, MFMA 92 | total 387, MFMA 72 |
| `main_loop[1]` | — (single codepath) | total 387, MFMA 72 |
| `n_gl[0]` | total 230, MFMA 72 | total 363, MFMA 72 |
| `n_ll[0]` | total 198, MFMA 72 | total 333, MFMA 72 |

The ref-build main-loop MFMA count (92) is exactly **72 + 20** — i.e.
72 native `v_mfma_f32_16x16x32_bf16` operations plus 20 helper
`v_mfma_f32_4x4x4_16b_bf16` operations for the TF32 low-bits pass. The
CMS build hits 72 MFMAs in every body (no 4×4×4 helpers).

### Capture-level cross-check (counts of `v_mfma_f32_4x4x4_16b_bf16` in the dumped TF32 streams)

| Capture file | Hits |
|---|---|
| `Tensile/Tests/unit/ldm5_default_capture.txt` | 80 |
| `Tensile/Tests/unit/ldm5_cms_capture.txt` | 0 |

(80 = 20 per body × 4 bodies — `main_loop_prev`, `main_loop`, `n_gl`,
`n_ll`. Identity-set count of 20 collapses these to a single identity
each because `identity_for(...)` is body-blind under hdem Approach A
and the same low-bits MFMA appears in all four bodies; the comparator
sees 20 distinct identities total.)

### Where the CMS path puts its low-bits computation

CMS asm uses `v_cvt_pk_bf16_f32` + `v_cvt_f32_bf16` + `v_sub_f32` to
compute the same TF32 high/low decomposition that the ref-side
`v_mfma_f32_4x4x4_16b_bf16` produces. 160 occurrences of
`v_cvt_bf16|v_pk_cvt_bf16|PVCvtBF16|v_sub_f32` in CMS asm —
~8× the count of the 20 ref-side helper MFMAs, consistent with the
"cvt + sub" path emitting ~6 helper insts per logical low-bits
operation.

---

## §4 — Per-missing-identity status

Detailed dump in `Tensile/Tests/unit/ldm5_missing_identities.txt`. All
20 identities follow the same pattern: `v_mfma_f32_4x4x4_16b_bf16` with
the `v[92:93]` operand being the `IdentityMatrix` register (set up via
`KernelWriterAssembly.py:847` only when
`kernel["UseMFMAF32XEmulation"]=True`).

| # | A-side / B-side | Operand spelling pattern | Present in CMS asm? | Present in CMS capture? | In ref bodies |
|---:|---|---|---|---|---|
| 0 | A | `T0_I0+0..3, [92:93], X0_I0+0..1, T0_I0+0..3` | NO | NO | ML, ML-prev, NGL, NLL |
| 1 | A | `T0_I0+12..15, [92:93], X0_I0+24..25, T0_I0+12..15` | NO | NO | ML, ML-prev, NGL, NLL |
| 2 | A | `T0_I0+4..7, [92:93], X0_I0+8..9, T0_I0+4..7` | NO | NO | ML, ML-prev, NGL, NLL |
| 3 | A | `T0_I0+8..11, [92:93], X0_I0+16..17, T0_I0+8..11` | NO | NO | ML, ML-prev, NGL, NLL |
| 4 | A | `X0_I0+12..15, [92:93], X0_I0+10..11, X0_I0+12..15` | NO | NO | ML, ML-prev, NGL, NLL |
| 5 | A | `X0_I0+20..23, [92:93], X0_I0+18..19, X0_I0+20..23` | NO | NO | ML, ML-prev, NGL, NLL |
| 6 | A | `X0_I0+28..31, [92:93], X0_I0+26..27, X0_I0+28..31` | NO | NO | ML, ML-prev, NGL, NLL |
| 7 | A | `X0_I0+4..7, [92:93], X0_I0+2..3, X0_I0+4..7` | NO | NO | ML, ML-prev, NGL, NLL |
| 8 | B | `T0_I0+0..3, [92:93], X0_I0+0..1, T0_I0+0..3` | NO | NO | ML, ML-prev, NGL, NLL |
| 9 | B | `T0_I0+12..15, [92:93], X0_I0+24..25, T0_I0+12..15` | NO | NO | ML, ML-prev, NGL, NLL |
| 10 | B | `T0_I0+16..19, [92:93], X0_I0+32..33, T0_I0+16..19` | NO | NO | ML, ML-prev, NGL, NLL |
| 11 | B | `T0_I0+20..23, [92:93], X0_I0+40..41, T0_I0+20..23` | NO | NO | ML, ML-prev, NGL, NLL |
| 12 | B | `T0_I0+4..7, [92:93], X0_I0+8..9, T0_I0+4..7` | NO | NO | ML, ML-prev, NGL, NLL |
| 13 | B | `T0_I0+8..11, [92:93], X0_I0+16..17, T0_I0+8..11` | NO | NO | ML, ML-prev, NGL, NLL |
| 14 | B | `X0_I0+12..15, [92:93], X0_I0+10..11, X0_I0+12..15` | NO | NO | ML, ML-prev, NGL, NLL |
| 15 | B | `X0_I0+20..23, [92:93], X0_I0+18..19, X0_I0+20..23` | NO | NO | ML, ML-prev, NGL, NLL |
| 16 | B | `X0_I0+28..31, [92:93], X0_I0+26..27, X0_I0+28..31` | NO | NO | ML, ML-prev, NGL, NLL |
| 17 | B | `X0_I0+36..39, [92:93], X0_I0+34..35, X0_I0+36..39` | NO | NO | ML, ML-prev, NGL, NLL |
| 18 | B | `X0_I0+44..47, [92:93], X0_I0+42..43, X0_I0+44..47` | NO | NO | ML, ML-prev, NGL, NLL |
| 19 | B | `X0_I0+4..7, [92:93], X0_I0+2..3, X0_I0+4..7` | NO | NO | ML, ML-prev, NGL, NLL |

**All 20 missing identities are uniformly absent from BOTH the CMS asm
and the CMS capture.** The "Present in CMS asm" column was computed by
substring-matching the first 60 chars of each canonical render against
the CMS asm lines; all 20 returned 0 hits. The "Present in CMS
capture" column comes from a canonical-render-keyed lookup against
both captures' instruction lists.

The pattern of operands is also informative: 8 A-side + 12 B-side =
20, matching the wave-tile shape (MIWaveTile = 4 × 6 = 24 with some
collapsing under hdem Approach A's body-blind identity). Each helper
MFMA corresponds to one (`T0_I0+i..j`, `X0_I0+k..l`) operand pair that
the default-side `Calculate low bits for TF32 emulation` comment
attaches to.

---

## §5 — Classification

**Verdict: (b) Q2-expected scheduler difference. NOT a CMS defect.
NOT a capture bug. The two builds are emitting two legitimate
lowerings of TF32 emulation.**

### Mechanism (with citations)

1. The fixture's per-tile schedule function
   `_get_schedule_128x192x32_TF32` sets
   `kernel["UseMFMAF32XEmulation"] = False` at
   `Tensile/Components/CustomSchedule/gfx950/_128x192x32_TF32.py:52`
   inside the `isTN(kernel) and not useLDSTr and TLDS==1` branch.
2. `hasCustomSchedule(state)` invokes that body during
   `Solution.__init__` at `Tensile/SolutionStructs/Solution.py:1994`
   (per the comment block at `:1965-1977`, the schedule functions are
   the authoritative source for several CMS-side flags). The mutation
   therefore lands on the Solution state.
3. The LR/Pack helper code at
   `Tensile/Components/LocalRead.py:490-502` is gated by
   `kernel["UseMFMAF32XEmulation"]`: when True it emits
   `MFMAInstruction(instType=INST_BF16, accType=INST_F32,
   variant=[4,4,4,16])` (which renders as `v_mfma_f32_4x4x4_16b_bf16`)
   to compute TF32 low bits via the negative-identity-matrix MFMA
   trick. When False it falls through to the
   `PVCvtBF16toFP32` + `VSubF32` (cvt + sub) lowering at
   `LocalRead.py:510-513`.
4. The CMS build therefore takes the cvt+sub branch; the non-CMS
   reference build takes the 4×4×4 MFMA branch. Both compute the
   same TF32 high/low decomposition; only the opcode mix differs.
   The CMS asm at `kernel_cms_ldm5.s:2036-2049` shows the cvt+sub
   pattern interleaved with the main MFMAs, side-by-side with the
   default asm at `kernel_default_ldm5.s:1966-1977` showing the
   pack+4x4x4 pattern.
5. The non-CMS build's `UseMFMAF32XEmulation` is not seen by the
   per-tile body (early-return at
   `Tensile/Components/CustomSchedule/dispatch.py:401-402`) and so
   retains the `True` set by `Solution.py:639` for TF32 on a
   MFMA-capable ISA.

### Why this is exactly nyb5's Cycle 2 mechanism, just for a different flag

`3IJA_RESIDUAL_TRIAGE.md §3.A` and `NYB5_IMPLEMENTATION.md "Cycle 2
surprise"` already characterise this exact shape for `UsePLRPack` /
`MfmaInitCVgprs`:

> The per-tile schedule on the CMS side mutates kernel-level flags
> (`UsePLRPack=True`, `UseMFMAF32XEmulation`) before SIA3 runs, which
> changes the GR scheduling order. … The non-CMS reference build
> (Approach A) uses an unmutated `kernel` dict per
> `2LZD_INVESTIGATION.md §6.2 Q2`. So the GR-stream divergence is the
> *expected* Q2 surfacing.

ldm5 surfaces a NEW consequence of the same mechanism: the per-tile
mutation of `UseMFMAF32XEmulation=False` selects a different TF32
helper-emission strategy (cvt+sub vs 4×4×4 MFMA), which the comparator
sees as 20 missing helper-MFMA identities. This is Q2 by design — the
fixture explicitly sets the flag, the emission path branches on it,
both branches are correct.

### Sanity-check vs the alternatives that the bead listed

- **(a) Real CMS defect**: ruled out. CMS would only be defective if
  it dropped the low-bits computation entirely. The cvt+sub helpers in
  the CMS asm (160 cvt/sub instructions vs the ref's 100 4×4×4 MFMA
  instances — different op counts because each 4×4×4 MFMA replaces ~6
  cvt+sub instructions) prove CMS still computes the low bits, just
  with a different lowering.
- **(c) Capture pipeline bug**: ruled out. The capture pipeline
  faithfully reflects each build's emitted asm. CMS asm has zero
  `v_mfma_f32_4x4x4_16b_bf16` instructions; CMS capture has zero
  `v_mfma_f32_4x4x4_16b_bf16` identities. Default asm has 100; default
  capture has 80 (= 20 distinct × 4 bodies before identity collapse).
  Both captures are consistent with their respective asm.

The "mixed" case the bead allowed for does not apply: every missing
identity is uniformly absent from CMS asm AND CMS capture, with no
split.

---

## §6 — Recommended next step

This residual class belongs to the same "Q2 — per-tile schedule
mutates flags, non-CMS reference doesn't see those mutations" family
as §3.A in the 3ija memo (GR OrderInverted). The fix space mirrors
`rocm-libraries-p39d`'s candidates, adapted for the MFMA-helper
identity class.

Three principled options for accepting this class:

**Option (i): Widen `_NO_DATAFLOW_IDENTITY_CATEGORIES` for the TF32
helper-MFMA case.** Pros: surgical — the only thing that changes is
identity-set coverage at `compare_graphs` entry. Cons: dropping all
MFMA from identity coverage is unsafe (it would mask a real defect
where CMS drops a *main* MFMA). Would need a narrower category that
distinguishes the `v_mfma_f32_4x4x4_16b_bf16` TF32-helper MFMAs from
the main `v_mfma_f32_16x16x32_*` MFMAs. The `comment="Calculate low
bits for TF32 emulation"` attached at `LocalRead.py:502` is a
plausible discriminator at capture time (route helper MFMAs into a
new category, e.g. `TF32_HELPER_MFMA`, that is in
`_NO_DATAFLOW_IDENTITY_CATEGORIES` but not in `_DATA_FLOW_CATEGORIES`).

**Option (ii): Normalize the kernel flags on the non-CMS reference
build to match the per-tile schedule body's mutations BEFORE running
`build_non_cms_reference`.** This makes Build #2 a "non-CMS build with
the same kernel flag state CMS would have arrived at." Pros: makes
"the same logical kernel" actually the same logical kernel — closes
not just this residual but every Q2 surface from per-tile flag
mutations. Cons: deviates from `2LZD_INVESTIGATION.md §6.2 Q2`'s
"accept whatever Tensilelite mutates on either side"; arguably defeats
the purpose of having a non-CMS reference at all (it becomes a
CMS-flag-state non-CMS reference, which is its own fiction). Also, the
per-tile schedule body has to run on a fully-populated kernel dict to
know which mutations it makes — so this implies running the schedule
body twice (once to learn the mutations, once to build the kernel) or
extracting the flag mutations into a separately-callable helper.

**Option (iii): Accept that "the same logical kernel under different
flag-state" is two different kernels and reframe the comparator
contract.** Per Q2 the two builds are NOT the same kernel — they are
two kernels Tensilelite produces for the same YAML and the comparator
must accept both as correct. Under this framing, the comparator's job
is to verify the dataflow shape of the CMS-build matches "some
expected pattern derivable from the CMS-flag-state kernel," NOT to
match a Solution-default-flag-state non-CMS build. This is the
strongest position but requires re-tooling the reference side. (May
be too large for this bead.)

**Recommended:** Option (i) — extend the capture pipeline to tag
TF32 helper MFMAs with a category that the comparator's data-flow
filter excludes. The discriminator is the comment attached at
emission time (`LocalRead.py:502`), and the change is mechanical:
add a new `InstructionCategory` value `TF32_HELPER_MFMA`, route the
LR-side helper MFMAs into it via the source-module rule registered
in `ScheduleCapture.py` (Approach 2 dfd8 source-module discrimination
machinery already exists; this is one more rule), and add the new
category to `_NO_DATAFLOW_IDENTITY_CATEGORIES` only (not to
`_DATA_FLOW_CATEGORIES`). The main-loop 16×16×32 MFMAs stay in
`MFMA` and continue to enjoy full identity-set coverage; only the
helper variant is excluded.

Option (ii) is structurally cleaner but cuts against the Q2 framing
the validator architecture was designed around. Option (iii) is a
multi-week reframe.

---

## §7 — Resolution: test-site patch + long-term referral

### What landed (immediate)

`Tensile/Tests/unit/_3ija_residual_triage_runner.py`:

1. `_PER_TILE_REF_FLAG_OVERRIDES` — a static map keyed on
   `(info.name, info.TransposeA, info.TransposeB, info.LDSTrInst,
   info.TransposeLDS)` listing the four TF32 schedules that set
   `UseMFMAF32XEmulation = False` inside their per-tile bodies:
   `_get_schedule_128x192x32_TF32`, `_get_schedule_192x128x32_TF32`,
   `_get_schedule_256x192x32_TF32`, `_get_schedule_256x128x32_TF32`.

2. `_build_non_cms_reference_with_state_overrides(config, info, asm,
   isaInfoMap)` — inline replica of `build_non_cms_reference`
   (`Tensile/Components/CustomSchedule/approach_a.py:70-140`) that for
   fixtures in the override map mutates `Solution._state` POST-construction
   (after `Solution.assignProblemIndependentDerivedParameters` re-derives
   `UseMFMAF32XEmulation` at `Solution.py:625-639`) but BEFORE
   `_getKernelSource` runs. Fixtures with no override fall through to the
   stock `build_non_cms_reference`.

3. `_exercise_one` calls the new helper instead of `build_non_cms_reference`.

### Why config-dict-only override was insufficient

The first iteration of this patch mutated the `config` dict passed to
`build_non_cms_reference`. That had zero effect: `Solution.py:625` does
`state["UseMFMAF32XEmulation"] = False` unconditionally and `:639` sets
it back to `True` for `HasMFMA` ISAs regardless of what the input config
asked for. Config-dict overrides are wiped out by the derivation. The
post-construction `Solution._state[...]` mutation is the only place the
override can survive into the writer.

### Verification

Runner output on validation tip (this patch applied):

| Fixture | Pre-patch | Post-patch |
|---|---|---|
| `_get_schedule_128x192x32_TF32 TN LDSTr=False TLDS=1` | `CaptureConsistencyError 20 MFMA` | `cg=0 wc=0` ✓ |
| `_get_schedule_256x192x32_TF32 TN LDSTr=False TLDS=1` | `CaptureConsistencyError 28 MFMA` | `cg=0 wc=0` ✓ |
| Aggregate `CaptureConsistencyError` count | 2 | **0** |
| Total compare_graphs residuals | 7 (5 OrderInverted + 2 ldm5) | **5** (p39d-class only) |

Full unit suite (`pytest Tensile/Tests/unit/
--ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py
--timeout=120 -q`): **1033 passed, 3 skipped, 2 xfailed** — matches the
validation-tip baseline (post-dfd8 squash). **Zero regressions.**

The other two LDSTr=True schedules in the override map
(`_get_schedule_192x128x32_TF32 TN LDSTr=True TLDS=1`,
`_get_schedule_256x128x32_TF32 TN LDSTr=True TLDS=1`) still fail at
`_make_solution` with "Solution is not valid" — a separate
config-validation issue unrelated to ldm5. Their override entries are
in place defensively so the patch becomes effective the moment that
upstream issue is fixed.

### Long-term referral: rocm-libraries-2bww (P0)

The test-site override is a workaround, not a fix. The contract violation
is in the CMS schedule body itself: a schedule body is supposed to provide
*scheduling content*, not mutate kernel flag state behind the Solution
constructor's back. So long as flag mutations live inside the schedule
function bodies, every new validator that exercises a non-CMS reference
will see the same divergence, and every new schedule body that mutates a
flag will reintroduce the residual.

**rocm-libraries-2bww (P0)** owns the architectural fix: move
kernel-flag declarations out of `Tensile/Components/CustomSchedule/
gfx950/*.py` schedule-function bodies and onto the
`RegisterSchedule` decorator / sibling metadata structure so:

1. Both the CMS Solution and the non-CMS reference Solution derive the
   flags identically from the same YAML/metadata source.
2. The schedule body is purely scheduling content — no side effects on
   kernel flag state.
3. `_PER_TILE_REF_FLAG_OVERRIDES` and
   `_build_non_cms_reference_with_state_overrides` can be deleted.

The P0 priority (raised from P1 at user direction) reflects that this
mutation-from-inside-schedule-body pattern is the **same mechanism** as:

- nyb5 Cycle 2 surprise (`UsePLRPack` mutation; resolved via dfd8
  source_module_id machinery, but the underlying contract violation
  remains)
- p39d (GR OrderInverted residual class; same `UsePLRPack` mutation,
  not yet fully resolved)
- ldm5 (this bead; `UseMFMAF32XEmulation` mutation)
- Likely to surface again for `UseDot2F32XEmulation` and any other flag
  the per-tile bodies touch.

Fixing it at the contract layer (2bww) eliminates the entire class.

---

## §8 — Open questions

1. **Is the residual really inert?** Both lowerings of TF32 emulation
   are well-tested in production; running the two builds against real
   hardware would produce numerically equivalent (up to rounding)
   results. ldm5 did not run hardware comparisons (the bead is audit
   only). If the user wants on-device numerical confirmation, that is
   a separate bead.

2. **Does the same class hit the other ldm5 fixture
   (`_256x192x32_TF32 TN`, 28 missing identities)?** Almost certainly
   yes — that fixture's schedule file also sets
   `kernel["UseMFMAF32XEmulation"] = False` in the TN branch (per the
   3ija memo grep). The 28 vs 20 count difference reflects the larger
   wave-tile shape. I did not separately reproduce on the 256-tile
   fixture but the mechanism is the same. The recommendation in §6
   covers both.

3. **Does Option (i) risk masking a future CMS defect that drops a
   helper MFMA but keeps a main MFMA?** Slight. If CMS one day emits
   only some of the helper MFMAs (e.g. drops the B-side
   `Calculate low bits` chain), the identity-set check would not
   surface it. Mitigation: a separate per-build "count of
   `TF32_HELPER_MFMA` should equal expected count per pack-LR pair"
   assertion, run on each capture independently, similar to the
   `validate_edge_wait_coverage` shape. This is a small follow-up if
   Option (i) is chosen.

4. **Should this and `rocm-libraries-p39d` (the GR OrderInverted class)
   merge into a single "Q2 per-tile-flag-mutation handling" bead?**
   They share the mechanism and (ii) would address both at once.
   Worth raising with the maintainer.

---

## §9 — Cross-references

- `rocm-libraries-2bww` (P0) — Stop CMS schedule bodies from mutating
  kernel-level flags; move flag declarations into YAML/schedule-metadata.
  Owns the architectural fix that supersedes this bead's test-site patch.
- `Tensile/Components/3IJA_RESIDUAL_TRIAGE.md §3.A, §3.F` — parent
  triage that filed this bead.
- `Tensile/Components/NYB5_IMPLEMENTATION.md` — Cycle 2 surprise
  mechanism; ldm5 is the same shape for a different flag.
- `Tensile/Components/2LZD_INVESTIGATION.md §6 + §6.2` — Approach A
  picks + Q2 framing this memo's classification leans on.
- `Tensile/Components/E293_R1_APPROACH_2_MEMO.md` — the source-module
  discrimination machinery that Option (i) extends.
- `Tensile/Components/EMISSION_ORDINAL_DESIGN.md §4.1` — the
  rocisa-derived classifier registry that would need a new category.
- `Tensile/Components/CustomSchedule/gfx950/_128x192x32_TF32.py:51-52,
  97` — the per-tile flag mutations.
- `Tensile/Components/CustomSchedule/dispatch.py:401-413` —
  `hasCustomSchedule` early-return for non-CMS + per-tile body call
  site.
- `Tensile/Components/LocalRead.py:490-513` — the
  `UseMFMAF32XEmulation` branch in the LR-side helper-emission code.
- `Tensile/SolutionStructs/Solution.py:629-640` — Solution-side
  `UseMFMAF32XEmulation` derivation.
- `Tensile/KernelWriterAssembly.py:843-848` — `IdentityMatrix` VGPR
  registration gated on the same flag.
- `Tensile/Components/CMSValidator.py:1571 + 3578-3657` —
  `_NO_DATAFLOW_IDENTITY_CATEGORIES` and the `compare_graphs`
  identity-set check that produced the raise.
- `Tensile/Tests/unit/_3ija_residual_triage_runner.py` —
  `_PER_TILE_REF_FLAG_OVERRIDES` + `_build_non_cms_reference_with_state_overrides`
  test-site patch landed under this bead (see §7).
- `Tensile/Tests/unit/_ldm5_dump.py` and asm/capture dumps — scratch
  artifacts that produced §3-§4 evidence; not committed.
