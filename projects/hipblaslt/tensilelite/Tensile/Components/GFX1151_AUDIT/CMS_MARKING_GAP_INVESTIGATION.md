# CMS-Marking Gap in Shipping gfx1151 HHS YAML

Bead: `rocm-libraries-w9p`. Investigation only — no code changes proposed under this bead.

## Executive Summary

The shipping yaml
`projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality/gfx1151_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml`
contains 25 solutions. Only 2 end up with `UseCustomMainLoopSchedule=1`
(MT128x64x64 PGR=1 PLR=0 SIA=3 and MT64x32x64 PGR=2 PLR=1 SIA=3). The
other 23 solutions have CMS=0. Because the validator's auto-activation
hook in `KernelWriter.py:4692-4693` only fires when CMS=1, 23/25 of the
shipping kernels are never validated.

The gap is not a dispatcher bug — `hasCustomSchedule(state)` works as
intended. The reason 23 kernels show CMS=0 is simply that **no
`@RegisterSchedule` exists for their (MT0, MT1, DepthU, PGR, PLR, DTL,
DPLB, MIWaveGroup, GRVW, LRVW, MatrixInstruction) tuple** in
`Tensile/Components/CustomSchedule.py`. 31 gfx1151 schedules are
registered, but their parameter tuples don't cover the shapes the
shipping yaml ships.

The gap is **accidental, not by design**: some unmatched shapes are off
by a single field (PLR or MIWG) from a registered schedule, and the
naming convention (`_pgr1_plr0_...`, `_pgr2_plr1_...`) suggests every
viable PGR×PLR combination per MT was meant to be covered, with several
combinations simply not yet contributed.

## 1. Where `UseCustomMainLoopSchedule` is set

### 1.1 Default (`-1` sentinel)

`projects/hipblaslt/tensilelite/Tensile/Common/GlobalParameters.py:504`
```
{"UseCustomMainLoopSchedule": [-1]},
```

The valid values are `[-1, 0, 1]`
(`Tensile/Common/ValidParameters.py:960`). YAML-supplied values that
aren't `-1` are honored (the comment at line 986 notes the parameter is
"internally set" but inspection of the dispatcher shows yaml-supplied
`1` is also re-resolved against the registry).

### 1.2 The single dispatcher

`projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Solution.py:2054-2062`
```python
if state["UseCustomMainLoopSchedule"] in [-1, 1]:
  state["SwapGlobalReadOrder"] = 0
  state["UsePLRPack"] = 0
  hasCMS,_ = hasCustomSchedule(state)
  if state["UseCustomMainLoopSchedule"] == 1 and not hasCMS:
    reject(state, printRejectionReason, "UseCustomMainLoopSchedule=1 but CMS is not supported")
  state["UseCustomMainLoopSchedule"] = 1 if hasCMS else 0
```

This is the *only* place the flag is mutated. `hasCustomSchedule(state)`
lives at `Tensile/Components/CustomSchedule.py:534` and walks the
`_SCHEDULE_REGISTRY` list of `RegisterSchedule`-wrapped functions.

### 1.3 Why this runs at library-build time even for "AssignedDerivedParameters: True" yamls

`assignDerivedParameters` (Solution.py:1227) does have an early-return
gate at line 1291-1293 when `state["AssignedDerivedParameters"]` is
True. Shipping logic yamls *do* set `AssignedDerivedParameters: True`
on every solution, so on the surface line 2062 should never run.

However, `LibraryIO.py` forces re-derivation when loading a logic file:

`projects/hipblaslt/tensilelite/Tensile/LibraryIO.py:393-395`
```python
# force redo the deriving of parameters, make sure old version logic yamls can be validated
solutionState["AssignedProblemIndependentDerivedParameters"] = False
solutionState["AssignedDerivedParameters"] = False
```
(also at LibraryIO.py:513). Because of this, the dispatcher at
Solution.py:2062 *does* run for every solution loaded from a shipping
yaml, and the `UseCustomMainLoopSchedule` value baked into the yaml
(either `-1` or `1`) is overridden by the live registry lookup result.

### 1.4 The dispatcher predicate (`hasCustomSchedule`)

`Tensile/Components/CustomSchedule.py:534-559`:

1. Returns `False, None` if `kernel["UseCustomMainLoopSchedule"]` is
   falsy (i.e., kernel state already says CMS off — a YAML that hard-
   codes CMS=0 short-circuits here).
2. Returns `False, None` if `not EnableMatrixInstruction`.
3. Returns `False, None` if ISA is not gfx950 (`(9,5,0)`) or gfx1151
   (`(11,5,1)`). On `develop` only gfx950 is allowed; gfx1151 was
   added on `users/alvasile/validator_long_term_plans` (commit
   `9cc8bcfdbf` and earlier).
4. Returns `False, None` if `isMixed(kernel)` (different bytes for
   DataTypeA vs DataTypeB).
5. Otherwise iterates `_SCHEDULE_REGISTRY` and returns the first match.

Each registered schedule is wrapped by
`RegisterSchedule.__call__` (`Tensile/Components/CustomSchedule.py:794-833`)
which compares the kernel state against the registration's
`TileConfig`, `dtype_predicate`, `vector_widths`, `matrix_inst`, and
`mfma_wave_group`. All of:
`(MT0, MT1, DU, PGR, PLR, DTL, DPLB, WSGRA, WSGRB, ISA, WavefrontSize)`
plus the dtype predicate, the `[GRVWA, GRVWB, LRVWA, LRVWB]` quadruple
(extended from the registration's `[GRVW_A, GRVW_B, LRVW]`), the
`MatrixInstruction` list, and `MIWaveGroup` must all match exactly.

## 2. The 31 registered gfx1151 schedules

Discovered by parsing `@RegisterSchedule` decorators that name an
`isa=(11, 5, 1)` `TileConfig` (lines 5586-7635 of CustomSchedule.py
on the validator branch):

| Line | Function | MT | PGR | PLR | DTL | MIWG |
| --- | --- | --- | --- | --- | --- | --- |
| 5594 | `_get_schedule_96x128x32_16bit_gfx1151` | 96x128x32 | 2 | 1 | 0 | [2,2] |
| 5657 | `_get_schedule_128x96x32_16bit_gfx1151` | 128x96x32 | 2 | 1 | 0 | [2,2] |
| 5719 | `_get_schedule_192x64x32_16bit_gfx1151` | 192x64x32 | 2 | 1 | 0 | [4,1] |
| 5781 | `_get_schedule_64x192x32_16bit_gfx1151` | 64x192x32 | 2 | 1 | 0 | [1,4] |
| 5844 | `_get_schedule_32x128x64_16bit_gfx1151` | 32x128x64 | 2 | 1 | 0 | [1,4] |
| 5922 | `_get_schedule_96x128x64_plr0_14_16bit_gfx1151` | 96x128x64 | 2 | 0 | 0 | [2,2] |
| 5995 | `_get_schedule_128x64x64_plr1_16bit_gfx1151` | 128x64x64 | 2 | 1 | 0 | [2,2] |
| 6068 | `_get_schedule_128x80x64_16bit_gfx1151` | 128x80x64 | 2 | 1 | 0 | [4,1] |
| 6145 | `_get_schedule_128x80x64_pgr1_16bit_gfx1151` | 128x80x64 | 1 | 1 | 0 | [4,1] |
| 6212 | `_get_schedule_64x128x32_14_16bit_gfx1151` | 64x128x32 | 2 | 1 | 0 | [1,4] |
| 6268 | `_get_schedule_128x64x32_16bit_gfx1151` | 128x64x32 | 2 | 1 | 0 | [4,1] |
| 6324 | `_get_schedule_64x128x32_22_16bit_gfx1151` | 64x128x32 | 2 | 1 | 0 | [2,2] |
| 6384 | `_get_schedule_96x96x32_16bit_gfx1151` | 96x96x32 | 2 | 1 | 0 | [2,2] |
| 6463 | `_get_schedule_64x64x32_22_16bit_gfx1151` | 64x64x32 | 2 | 1 | 0 | [2,2] |
| 6529 | `_get_schedule_16x16x128_16bit_gfx1151` | 16x16x128 | 2 | 1 | 0 | [1,1] |
| 6616 | `_get_schedule_32x16x128_16bit_gfx1151` | 32x16x128 | 2 | 1 | 0 | [2,1] |
| 6703 | `_get_schedule_128x96x64_plr0_41_16bit_gfx1151` | 128x96x64 | 2 | 0 | 0 | [4,1] |
| 6765 | `_get_schedule_96x128x64_plr0_14_16bit_gfx1151_tn` | 96x128x64 | 2 | 0 | 0 | [1,4] |
| 6827 | `_get_schedule_128x112x32_plr0_16bit_gfx1151` | 128x112x32 | 2 | 0 | 0 | [4,1] |
| 6885 | `_get_schedule_128x80x64_plr0_16bit_gfx1151` | 128x80x64 | 2 | 0 | 0 | [4,1] |
| 6952 | `_get_schedule_128x80x64_pgr1_plr1_16bit_gfx1151` | 128x80x64 | 1 | 1 | 0 | [4,1] |
| 7020 | `_get_schedule_80x128x64_pgr2_plr1_16bit_gfx1151` | 80x128x64 | 2 | 1 | 0 | [1,4] |
| 7084 | `_get_schedule_80x128x64_pgr2_plr0_16bit_gfx1151` | 80x128x64 | 2 | 0 | 0 | [1,4] |
| 7153 | `_get_schedule_64x224x32_pgr2_plr0_16bit_gfx1151` | 64x224x32 | 2 | 0 | 0 | [2,2] |
| 7212 | `_get_schedule_128x64x32_pgr2_plr0_16bit_gfx1151` | 128x64x32 | 2 | 0 | 0 | [2,2] |
| 7271 | `_get_schedule_128x48x32_pgr2_plr1_16bit_gfx1151` | 128x48x32 | 2 | 1 | 0 | [4,1] |
| 7331 | `_get_schedule_64x32x64_pgr2_plr1_16bit_gfx1151` | 64x32x64 | 2 | 1 | 0 | [2,2] |
| 7401 | `_get_schedule_128x96x32_pgr1_plr0_16bit_gfx1151` | 128x96x32 | 1 | 0 | 0 | [2,2] |
| 7458 | `_get_schedule_128x96x64_pgr2_plr0_22_16bit_gfx1151` | 128x96x64 | 2 | 0 | 0 | [2,2] |
| 7521 | `_get_schedule_128x64x64_pgr2_plr0_22_16bit_gfx1151` | 128x64x64 | 2 | 0 | 0 | [2,2] |
| 7584 | `_get_schedule_128x64x64_pgr1_plr0_22_16bit_gfx1151` | 128x64x64 | 1 | 0 | 0 | [2,2] |

(All 31 use ISA=(11,5,1), wavefront_size=32, DTL=0, DPLB=0, WSGRA=0,
WSGRB=0, MatrixInstruction=[16,16,16,1], LRVW=16. Most use
`vector_widths=[8,8,16]`; #18 (`128x112x32_plr0`) and #25
(`128x48x32_pgr2_plr1`) use `[8,4,16]`.)

## 3. Per-kernel classification

All 25 yaml solutions (TN layout, half-precision A/B with FP32
compute, ISA=(11,5,1), WavefrontSize=32, EnableMatrixInstruction=True,
DTL=0, DPLB=0, WSGRA=WSGRB=0, MI=[16,16,16,1], GRVWA=GRVWB=8, LRVW=16):

| # | MT | PGR | PLR | MIWG | YAML CMS | Probe CMS | SIA | Classification |
|---|---|---|---|---|---|---|---|---|
| 0 | 16x16x128 | 1 | 1 | [1,1] | -1 | 0 | 3 | no registered schedule (only PGR=2/PLR=1 exists for this MT, line 6529) |
| 1 | 16x16x128 | 1 | 0 | [1,1] | -1 | 0 | 3 | no registered schedule (only PGR=2/PLR=1 exists for this MT) |
| 2 | 64x16x64 | 1 | 1 | [4,1] | -1 | 0 | 3 | no registered schedule (no MT=64x16 of any DU registered) |
| 3 | 32x16x64 | 1 | 1 | [2,1] | -1 | 0 | 1 | no registered schedule (only MT=32x16x128 exists, line 6616) |
| 4 | 128x64x64 | 1 | 0 | [2,2] | 1 | 1 | 3 | **MATCHES `_get_schedule_128x64x64_pgr1_plr0_22_16bit_gfx1151` (line 7584)** |
| 5 | 32x32x128 | 1 | 1 | [2,2] | -1 | 0 | 3 | no registered schedule (no MT=32x32 of any DU registered) |
| 6 | 128x96x64 | 1 | 0 | [2,2] | -1 | 0 | 3 | no registered schedule (line 7458 covers PGR=**2**/PLR=0; PGR=1 variant missing) |
| 7 | 64x112x128 | 1 | 0 | [4,1] | -1 | 0 | 3 | no registered schedule (no MT=64x112 of any DU registered) |
| 8 | 128x128x32 | 1 | 0 | [2,2] | -1 | 0 | 3 | no registered schedule (no MT=128x128 of any DU registered) |
| 9 | 96x96x64 | 1 | 0 | [2,2] | -1 | 0 | 3 | no registered schedule (only MT=96x96x32 exists, line 6384) |
| 10 | 128x48x64 | 1 | 0 | [4,1] | -1 | 0 | 3 | no registered schedule (only MT=128x48x32 exists, line 7271) |
| 11 | 64x16x128 | 1 | 1 | [4,1] | -1 | 0 | 1 | no registered schedule (no MT=64x16 of any DU registered) |
| 12 | 96x128x64 | 1 | 0 | [1,4] | -1 | 0 | 3 | no registered schedule (line 6765 covers PGR=**2**/PLR=0; PGR=1 variant missing) |
| 13 | 64x96x64 | 1 | 0 | [2,2] | -1 | 0 | 3 | no registered schedule (no MT=64x96 of any DU registered) |
| 14 | 112x128x64 | 1 | 0 | [1,4] | -1 | 0 | 3 | no registered schedule (no MT=112x128 of any DU registered) |
| 15 | 128x112x64 | 1 | 0 | [4,1] | -1 | 0 | 3 | no registered schedule (line 6827 covers MT=128x112x**32**; DU=64 variant missing) |
| 16 | 128x96x32 | 1 | 0 | [4,1] | -1 | 0 | 3 | no registered schedule (line 7401 covers PGR=1/PLR=0 but MIWG=[2,2]; MIWG=[4,1] variant missing) |
| 17 | 128x80x64 | 1 | 0 | [4,1] | -1 | 0 | 3 | no registered schedule (lines 6068/6145/6885/6952 register PGR×PLR={2,1}×{1,0,1,1}; PGR=1/PLR=0 variant missing) |
| 18 | 64x96x32 | 1 | 1 | [2,2] | -1 | 0 | 1 | no registered schedule (no MT=64x96 of any DU registered) |
| 19 | 64x96x32 | 1 | 0 | [2,2] | -1 | 0 | 3 | no registered schedule (same — no MT=64x96 registered) |
| 20 | 64x128x64 | 1 | 0 | [1,4] | -1 | 0 | 3 | no registered schedule (no MT=64x128 of DU=64 registered; only DU=32 at lines 6212/6324) |
| 21 | 128x128x32 | 1 | 0 | [4,1] | -1 | 0 | 3 | no registered schedule (no MT=128x128 of any DU registered) |
| 22 | 80x128x64 | 1 | 0 | [1,4] | -1 | 0 | 3 | no registered schedule (lines 7020/7084 cover PGR=**2**; PGR=1 variant missing) |
| 23 | 128x16x64 | 1 | 0 | [4,1] | -1 | 0 | 3 | no registered schedule (no MT=128x16 of any DU registered) |
| 24 | 64x32x64 | 2 | 1 | [2,2] | 1 | 1 | 3 | **MATCHES `_get_schedule_64x32x64_pgr2_plr1_16bit_gfx1151` (line 7331)** |

### 3.1 The two CMS=1 kernels

- **#4 MT128x64x64 PGR=1 PLR=0 SIA=3** matches the registration at
  `CustomSchedule.py:7584`
  (`_get_schedule_128x64x64_pgr1_plr0_22_16bit_gfx1151`). The inner
  function checks `isTN(kernel)` (the yaml is TN), confirms via
  PGR/PLR/MIWG match.
- **#24 MT64x32x64 PGR=2 PLR=1 SIA=3** matches the registration at
  `CustomSchedule.py:7331`
  (`_get_schedule_64x32x64_pgr2_plr1_16bit_gfx1151`). Same TN-only
  inner check, all wrapper criteria match.

### 3.2 The 23 CMS=0 kernels — registration coverage breakdown

**Group A — partial-coverage MTs** (a sibling registration exists for
the same MT but a different PGR/PLR/MIWG combination): 8 kernels
- #0, #1 (MT16x16x128: only PGR=2/PLR=1 registered; need PGR=1 PLR=0/1)
- #6 (MT128x96x64: PGR=1 PLR=0 needed; only PGR=2 registered)
- #12 (MT96x128x64: PGR=1 PLR=0 needed; only PGR=2 registered)
- #16 (MT128x96x32: PGR=1 PLR=0 needed at MIWG=[4,1]; only [2,2] registered)
- #17 (MT128x80x64: PGR=1 PLR=0 needed; PGR=1 PLR=1 and PGR=2 PLR=0/1 already registered)
- #22 (MT80x128x64: PGR=1 PLR=0 needed; only PGR=2 PLR=0/1 registered)

**Group B — DepthU mismatch** (same MT0×MT1 exists at a different DU):
3 kernels
- #3 (MT32x16x64 vs registered 32x16x128 at line 6616)
- #10 (MT128x48x64 vs registered 128x48x32 at line 7271)
- #15 (MT128x112x64 vs registered 128x112x32 at line 6827)
- #20 (MT64x128x64 vs registered 64x128x32 at lines 6212/6324) — note
  these registrations cover different MIWGs from the yaml's [1,4]
  anyway, so this is also Group A.

**Group C — wholly unregistered macro-tiles** (no `MT0×MT1` anywhere
in the gfx1151 schedule list): 12 kernels
- #2, #11 (MT64x16 — never registered)
- #5 (MT32x32 — never registered)
- #7 (MT64x112 — never registered)
- #8, #21 (MT128x128 — never registered)
- #9 (MT96x96 of DU=64 — only DU=32 at line 6384)
- #13 (MT64x96 — never registered)
- #14 (MT112x128 — never registered)
- #18, #19 (MT64x96 of DU=32 — never registered)
- #23 (MT128x16 — never registered)

## 4. Is the gap intentional or accidental?

**Accidental.** Evidence:

1. **Registration naming convention is exhaustive-style.** The
   ad-hoc names like `_get_schedule_128x80x64_pgr1_16bit_gfx1151`
   (line 6145), `_get_schedule_128x80x64_pgr1_plr1_16bit_gfx1151`
   (line 6952), `_get_schedule_128x80x64_plr0_16bit_gfx1151` (line
   6885), and `_get_schedule_128x80x64_16bit_gfx1151` (line 6068,
   the default PGR=2 PLR=1 form) show that authors register each
   PGR×PLR combination as it gets hand-tuned. The MT128x80x64 series
   is missing only PGR=1 PLR=0 — exactly what kernel #17 needs. The
   pattern is "register each variant as its hand-tuned schedule
   becomes available" — a coverage-by-accumulation approach.

2. **The two CMS=1 kernels were specifically added.** The shipping
   yaml hard-codes `UseCustomMainLoopSchedule: 1` only for the two
   shapes that have been hand-tuned and registered — meaning the
   tuning team explicitly opts in *after* they've added the
   corresponding `_get_schedule_*` function. The remaining 23 just
   never had matching schedules contributed.

3. **No documentation says "PGR=1/PLR=0 was deliberately not
   registered for MT128x80x64".** The CustomSchedule.py docstrings
   describe each registered schedule's instruction sequence in
   detail; nothing reads as "this shape uses default scheduling on
   purpose".

4. **The audit-doc ancestor file `Tensile/Components/GFX1151_AUDIT.md`
   on the validator branch (commit 31a7b17e8a) frames CMS coverage as
   work-in-progress.** The probe README in this same directory
   ("Effective validator coverage on shipping HHS gfx1151 = 1/25")
   treats the gap as something to close.

The gap is therefore **accidental**: missing registrations rather
than a design decision to use default scheduling. There is no
defensive code path that "intentionally falls back" — `hasCMS=False`
just means "no one has hand-tuned this shape yet".

## 5. Recommended next steps

(All as separate beads — no code changes under `rocm-libraries-w9p`.)

1. **File one bead per missing registration in Group A** (8 kernels):
   these are the easiest wins — siblings in the same MT family already
   exist and their structure can guide the new variant. Suggested bead
   titles:
   - "Register CMS schedule for gfx1151 MT16x16x128 PGR=1 PLR=0/1"
   - "Register CMS schedule for gfx1151 MT128x96x64 PGR=1 PLR=0"
   - "Register CMS schedule for gfx1151 MT96x128x64 PGR=1 PLR=0"
   - "Register CMS schedule for gfx1151 MT128x96x32 PGR=1 PLR=0
     MIWG=[4,1]"
   - "Register CMS schedule for gfx1151 MT128x80x64 PGR=1 PLR=0"
   - "Register CMS schedule for gfx1151 MT80x128x64 PGR=1 PLR=0"
   - "Register CMS schedule for gfx1151 MT128x112x32 -> DU=64 variant"
   - "Register CMS schedule for gfx1151 MT128x48x32 -> DU=64 variant"

2. **For Group B (DU mismatches)** investigate whether the existing
   DU=32 schedules can be parameterised over DU or whether DU=64
   needs an independent registration.

3. **For Group C (wholly unregistered macro-tiles)** — 12 kernels —
   defer until tuning data identifies which actually deliver
   meaningful speedups under CMS. These macro-tiles (32x32, 64x16,
   64x96, 64x112, 64x128 DU=64, 80x128 DU=64, 96x96 DU=64, 112x128,
   128x16, 128x128) are arguably out of scope for hand-tuning until
   benchmarks justify the effort.

4. **Document the validator-coverage expectation** in
   `Tensile/Components/CustomSchedule.py` near `hasCustomSchedule`:
   note that absence of a matching registration silently disables
   both CMS *and* validator coverage for that solution. (Today it's
   only inferable by reading `KernelWriter.py:4692-4693` together
   with `Solution.py:2062`.)

5. **Consider a CI check** that flags shipping-yaml solutions whose
   live `hasCustomSchedule(state)` returns False — at minimum to
   produce a JSON report of validator-coverage gaps per
   library/build, similar to what
   `Tensile/Components/GFX1151_AUDIT/validator_coverage_probe.py`
   already does for one yaml.

## 6. Method used

1. Cherry-picked
   `projects/hipblaslt/tensilelite/Tensile/Components/CustomSchedule.py`
   and `projects/hipblaslt/tensilelite/Tensile/Components/GFX1151_AUDIT/`
   from `users/alvasile/validator_long_term_plans` into this
   develop-based worktree.
2. Parsed every `@RegisterSchedule(...)` decorator in
   `CustomSchedule.py` with `isa=(11, 5, 1)` to enumerate the 31
   registered gfx1151 schedules and their (TileConfig,
   matrix_inst, mfma_wave_group, vector_widths) tuples.
3. Loaded the 25 solutions from
   `gfx1151_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml` (the
   `Equality/` copy that lives on the validator branch, identical to
   the `GridBased/` copy on develop except for directory) and
   extracted the same fields per solution.
4. Re-implemented the wrapper match predicate
   (CustomSchedule.py:794-833 — TileConfig equality, matrix_inst
   equality, mfma_wave_group equality, extended vector-widths
   equality) and applied it for every (solution, schedule) pair.
5. Cross-checked the result against
   `/tmp/probe_perk_report.json` (output of the
   `validator_coverage_probe.py` run on the validator-branch live
   build): both methods agreed on which 2 kernels are CMS=1 and which
   23 are CMS=0.

## 7. Critical-review pass

A general-purpose subagent re-read this draft and CustomSchedule.py
with the per-kernel classification table to verify each row.

**Findings from review:**

- Verified the two CMS=1 matches by re-reading
  `CustomSchedule.py:7331` (`_get_schedule_64x32x64_pgr2_plr1_16bit_gfx1151`)
  and `CustomSchedule.py:7584`
  (`_get_schedule_128x64x64_pgr1_plr0_22_16bit_gfx1151`). Both are
  `isTN`-gated and the yaml is TN; both register
  `MIWaveGroup=[2,2]`, `MatrixInstruction=[16,16,16,1]`,
  `vector_widths=[8,8,16]` matching the kernel.
- Verified Group A entries by spot-checking line numbers — each cited
  same-MT registration exists at the cited line and differs by
  exactly the field flagged.
- Re-confirmed Group C ("wholly unregistered MTs") by grepping
  CustomSchedule.py for each MT0×MT1 pair (`64x16`, `32x32`,
  `64x112`, `128x128`, `64x96`, `112x128`, `128x16`) — all return
  zero gfx1151 hits.
- Verified the static-analysis predicate against the live probe: the
  two methods agree on every kernel's CMS bit.
- Reviewed the "intentional vs accidental" framing — agree the
  exhaustive-style naming convention
  (`_pgr1_plr0_`, `_pgr1_plr1_`, `_pgr2_plr0_`, `_pgr2_plr1_` per MT)
  is strong evidence of incremental coverage rather than deliberate
  fallback.

**No mis-classifications found.** Recommended steps in §5 stand.

(One note worth flagging in a future bead: the wrapper at
`CustomSchedule.py:813-819` reads `kernel["LocalReadVectorWidthA/B"]`,
but `Solution.py:2547-2550` only normalises those from
`LocalReadVectorWidth` *after* the `hasCustomSchedule` call at
`Solution.py:2059`. In principle that should make every match fail
when the yaml omits LRVWA/B — yet kernels #4 and #24 do match in the
live probe. Either there's an upstream setter not found in this
investigation, or the wrapper is comparing `-1 == -1` due to both
sides seeing default values. Not load-bearing for this bead's
classification — both kernels' match was confirmed by the probe — but
worth opening as a code-review follow-up.)
