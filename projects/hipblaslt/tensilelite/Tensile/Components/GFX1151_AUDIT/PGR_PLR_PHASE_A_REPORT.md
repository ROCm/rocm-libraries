# Phase A: (PGR, PLR) cross-product audit (rocm-libraries-wuk)

**Bead:** rocm-libraries-wuk (sub of `rocm-libraries-ot2`).
**Scope:** Phase A — enumerate valid PGR / PLR values, the constraints between
them, and which body-emit phases (ML-1, ML, NGL, NLL) each legal pair triggers
in `KernelWriter.kernelBody`. Investigation only; no validator/capture changes.
**Source baseline read for this report:** worktree
`/home/alvasile/rocm-libraries/.claude/worktrees/agent-af817a1e7ddfa0940/projects/hipblaslt/tensilelite`
(branch `worktree-agent-af817a1e7ddfa0940`, descended from upstream develop
commit `43d06c68ec`). Validator/CMS/CMSValidator code paths additionally
cross-referenced against the active validator branch worktree at
`/home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/projects/hipblaslt/tensilelite`
(branch `users/alvasile/validator_long_term_plans`, latest `678b75ed21`),
because `Tensile/Components/CMSValidator.py`, `ScheduleCapture.py` and
`CustomSchedule.py` only exist on that branch. All file-line citations below
are against the develop-baseline tree unless explicitly prefixed with
`(validator-branch)`.

---

## 1. Enumerated valid values

### 1.1 `PrefetchGlobalRead`

**Schema-allowed values (the `ValidParameters` list):** `[0, 1, 2, 3, 4, …, 16]`
— see `Tensile/Common/ValidParameters.py:273`:

```python
"PrefetchGlobalRead": [0, 1, 2] + list(range(3,16 + 1)),
```

The neighbouring documentation block at `ValidParameters.py:261-272` defines the
semantics groupwise (no separate per-N description for N=3..16):

* **PGR=0**: no global prefetch. Inside the unroll loop the order is
  `wait → sync → local-write → local-read → mac` and "GR,LW,sync,LR will put at
  front of loop" (`KernelWriter.py:3596` comment, `:3601-3627` emit).
* **PGR=1**: one extra prefetch buffer ("Requires 2X LDS space, and VGPRs for
  buffering data on way into LDS"). One unroll-loop body is the steady-state;
  one final `NoLoadLoop` (NLL) drains the in-flight prefetch.
  (`ValidParameters.py:261-263`.)
* **PGR=2**: double-buffered with overlap. "Do another prefetch while writing
  data from vgpr to lds." (`ValidParameters.py:265-268`.) Requires both an
  `NGLL` (NoGlobalLoadLoop, drains the *first* in-flight set) and an `NLL`
  (drains the second). Body emit confirmation in
  `KernelWriter.py:4994-5011` (NGLL emission loop) and
  `KernelWriter.py:5017-5040` (NLL emission).
* **PGR≥3**: "DirectToLds only. Do PGR times prefetch global read before main
  loop. Need to allocate PGR+1 or PGR LDS buffer" (`ValidParameters.py:269-272`).
  Code path heavily restricted; see §3.1.

The schema admits PGR up to 16 by construction, but the validation in
`Tensile/SolutionStructs/Solution.py` rejects PGR≥3 unless several other
conditions hold simultaneously (DTLA+DTLB, PLR≥1, SIA=3, non-Sparse). I am
unaware of any in-tree solution YAML that exercises PGR>2; the gfx1151 sweep
captured by the existing probe (`validator_long_term_plans` branch,
`Tensile/Components/GFX1151_AUDIT/validator_coverage_probe.py:140-200`) records
PGR observed in {1, 2} only. **The schema is broader than the empirical
universe; PGR>2 is allowed by `ValidParameters.py:273` but I found no
in-tree evidence it is currently tuned for any architecture.** Flagged as an
implicit assumption — see "open questions" §6.

### 1.2 `PrefetchLocalRead`

**Schema-allowed values:** `range(128 + 1)` = `[0, 1, 2, …, 128]`
(`ValidParameters.py:275`):

```python
"PrefetchLocalRead": list(range(128 + 1)),
```

Documentation comment is terse: "number of iteration prefetch local reads from
lds to VGPRs buffer = PLR" (`ValidParameters.py:274`). The semantic:
PLR is the number of unroll-iterations worth of LR data prefetched into VGPR
buffers ahead of the MFMA that consumes them. It is *not* a boolean.

The runtime-effective PLR is bounded by `LoopIters` (the number of MFMA
iterations per unroll). At kernel-init time
`KernelWriter.py:6052-6056`:

```python
if kernel["EnableMatrixInstruction"] and kernel["LocalReadVectorWidthA"] >= kernel["MIInputPerThread"]:
  WLR = int(max(kernel["LocalReadVectorWidthA"]//kernel["MIInputPerThread"], 1))
  self.states.numItersPLR = kernel["PrefetchLocalRead"]%(kernel["LoopIters"]//WLR)
else:
  self.states.numItersPLR = kernel["PrefetchLocalRead"] % (kernel["LoopIters"])
```

So `numItersPLR` (the value actually consumed by the scheduler) is
`PLR mod (LoopIters / WLR)`. **Important consequence:** `PLR == LoopIters`
collapses to `numItersPLR == 0`. `PLR == 0` and `PLR == LoopIters` are
*not* distinguished here; downstream, `numItersPLR == 0` is the trigger for
"no-PLR" code paths (e.g. `KernelWriter.py:4780` `if self.states.numItersPLR:`,
`:4883 elif self.states.numItersPLR == 0 and kernel["UseCustomMainLoopSchedule"]:`).
This is a real footgun: a yaml with `PrefetchLocalRead: <LoopIters>` will not be
rejected but will silently behave like PLR=0. No assertion guards this.
Flagged in §6.

For non-MI (`UseDotInstruction`, dot2 path) `PLR` is further capped at
`LoopIters - 1` and `LoopIters % (PLR+1) == 0` is required
(`Solution.py:4424-4427`).

For `ForceUnrollSubIter` kernels, `numItersPLR` is hard-overridden to 1
regardless of the YAML PLR value (`KernelWriter.py:6222-6225`). This is an
assignment, not a reject; user-specified PLR is silently ignored in this
mode — flagged as another implicit assumption.

### 1.3 `OptNoLoadLoop`

For completeness (relevant in §4): `"OptNoLoadLoop": [0, 1, 2]`
(`ValidParameters.py:368`).

* 0 = no interleaving of stores into NLL.
* 1 = one store interleaved.
* 2 = two stores interleaved.

This is an NLL *shape* dimension, not an NLL on/off switch. NLL emission is
driven by `kernel["PrefetchGlobalRead"]` and `kernel["SuppressNoLoadLoop"]`
(see §4).

---

## 2. Pairwise legality of (PGR, PLR)

The cross-product space is large — 17 PGR values × 129 PLR values = 2 193
pairs. The vast majority of (PGR≥3, PLR=*) and (PGR=*, PLR≥some-LoopIters-bound)
pairs are rejected at solution-validation time. Below I enumerate every
constraint I located that mentions PGR and/or PLR. Each is a `reject(...)` (or
silent override) call in `Tensile/SolutionStructs/Solution.py` unless noted
otherwise.

### 2.1 Hard rejects (mutual-incompatibility)

| ID | Constraint | Source |
|----|-----------|--------|
| H1 | `BufferLoad=0 → reject if PGR ≥ 2` | `Solution.py:1635-1637` |
| H2 | `PGR ≥ 3 → reject if not (DirectToLdsA and DirectToLdsB)` | `Solution.py:4475-4478` |
| H3 | `PGR ≥ 3 → reject if PLR == 0` | `Solution.py:4479-4481` |
| H4 | `PGR ≥ 3 → reject if SIA != 3` | `Solution.py:4482-4484` |
| H5 | `PGR ≥ 3 → reject if Sparse` | `Solution.py:4485-4487` |
| H6 | `GlobalReadPerMfma > 1 → reject if PGR ≥ 2` | `Solution.py:4523-4524` |
| H7 | `UnrollLoopSwapGlobalReadOrder=1 → reject if PGR < 2` | `Solution.py:4548-4549` |
| H8 | `UseSubtileImpl=1 → reject unless PGR ∈ {0, 2}` (excludes PGR=1) | `Solution.py:708-710` |
| H9 | `1LDSBuffer=1 → reject if PGR == 0` ("PGR=0 already use 1 LDS buffer only") | `Solution.py:4082-4084` |
| H10 | `DirectToVgpr*` (single-side) `→ reject if PLR == 0 and TLU%c == False` | `Solution.py:955-957` |
| H11 | `DirectToVgpr*` `→ reject if PGR == 0` | `Solution.py:1056-1057` |
| H12 | `DirectToVgpr` (any) `→ reject if SIA < 3` | `Solution.py:1026-1027` |
| H13 | `dot2` (`UseDotInstruction`) `→ reject if LoopIters % (PLR+1) != 0` | `Solution.py:4426-4427` |
| H14 | `EnableMatrixInstruction and PLR > 0 → reject if LoopIters - PLR*wlrMultiple < 0` (A-side) | `Solution.py:4446-4448` |
| H15 | same for B-side | `Solution.py:4465-4467` |
| H16 | `DirectToVgpr ^ (DirectToVgprA xor DirectToVgprB) → reject if PLR ≥ LoopIters and ClusterLocalRead=1` | `Solution.py:4414-4418` |
| H17 | `ClusterLocalRead and PLR >= LoopIters and SIA != 2 and not ForceUnrollSubIter → silently force ClusterLocalRead=0, PLR=0` (NOT a reject; an override — flagged) | `Solution.py:4419-4420` |

### 2.2 Silent overrides (non-rejecting)

These are *not* rejects but quietly mutate the YAML-supplied PGR/PLR values:

| ID | Override | Source |
|----|----------|--------|
| O1 | `DirectToVgprA and DirectToVgprB → set PGR=1, PLR=0` (and EPS=False, 1LDSBuffer=0) | `Solution.py:931-936` |
| O2 | `SIA == 2 → set PLR=1, 1LDSBuffer=1` | `Solution.py:3237-3242` |
| O3 | `ForceUnrollSubIter → numItersPLR=1` (KernelWriter, not Solution) | `KernelWriter.py:6222-6225` |
| O4 | `ClusterLocalRead and PLR >= LoopIters and SIA != 2 and not ForceUnrollSubIter → set PLR=0` (already cited as H17) | `Solution.py:4419-4420` |
| O5 | `UseCustomMainLoopSchedule != 0 path: backup PLR; if not CMS-applicable, restore; if CMS-applicable AND DirectToLds==1 AND SIA==3 AND ForceUnrollSubIter etc., enable UsePLRPack — but PGR==0 OR PLR==0 disables UsePLRPack` | `Solution.py:2051-2102` |

### 2.3 Cross-product summary table (MFMA, the common case)

For the common MFMA path (`EnableMatrixInstruction=1`,
`BufferLoad=1`, no DTV, no DTL, non-Sparse, single-buffer LDS), the legal
(PGR, PLR) combinations are:

| PGR | PLR=0 | PLR=1 | PLR≥2 | PLR≥LoopIters |
|-----|-------|-------|-------|----------------|
| 0   | legal (SIA∈{0,1,3}, see §5; SIA=2 forces PLR=1 per O2) | legal | legal (subject to H14/H15) | overridden to 0 (H17) or rejected (H16) |
| 1   | legal | legal (most common case) | legal (H14/H15) | overridden / rejected |
| 2   | legal | legal | legal (H14/H15) | overridden / rejected |
| ≥3  | rejected (H3 — needs PLR≥1) | requires DTLA+DTLB+SIA=3+non-Sparse (H2/H4/H5) | same | overridden / rejected |

For DTV/DTL/Subtile/CMS sub-paths, additional constraints prune further. See
§5 for SIA-3-specific eligibility.

### 2.4 Forbidden pairs explicitly cited in source comments

* `(PGR=0, PLR=0)` is *legal* by all rejects above but the SIA3 scheduling
  literature in `Components/SIA.py` quietly assumes either PGR or PLR is
  non-zero in several spots — e.g. `SIA.py:304, 497`:
  `if writer.states.scheduleGROverBarrier or writer.states.numItersPLR == 0:`
  is a special-case branch. Not a reject; just a fall-through. No correctness
  bug located, but worth flagging.
* `(PGR=2, PLR=0)` is legal — `Solution.py` doesn't reject it. CMS
  registers a TileConfig at exactly this combination (validator-branch
  `Components/CustomSchedule.py:1321`):
  ```python
  tile_config=TileConfig(256, 256, 128, 2, 0, 1, False, 0, 0),
  ```
  i.e. `(MT0=256, MT1=256, DU=128, PGR=2, PLR=0, DTL=1, ...)`. So PGR=2 +
  PLR=0 is empirically supported. It triggers the "numItersPLR==0 + CMS"
  half-prefetch path at `KernelWriter.py:4883-4914`.

---

## 3. PGR-specific scheduling/asm helpers

### 3.1 PGR≥2 multi-prefetch emit

`KernelWriter.kernelBody` only enters the multi-prefetch loop when
`PGR ≥ 2` (`KernelWriter.py:4690-4770`):

```python
if kernel["PrefetchGlobalRead"] >= 2:
  for idxPgr in range(1, kernel["PrefetchGlobalRead"]):
    module.add(self.openPrefetchGlobalRead2orMore(kernel, idxPgr))
    ...
  for idxPgr in range(0, kernel["PrefetchGlobalRead"] + 1):
    module.add(self.closePrefetchGlobalRead2orMore(...))
```

The helper at `KernelWriterAssembly.py:15793-15822` emits the early-exit
`s_cmp_eq_u32 LoopCounter, idxPgr; s_cbranch_scc1 label_skipPGRn` ladder
visible in the in-tree `Custom_Cijk_Alik_Bljk_F8BS_BH_SAB_UserArgs_*.s`
captures (ripgrep hit at `CustomKernels/...:1618` `s_cmp_eq_u32 ... // PGR=2 but
only 1 loop`).

### 3.2 PGR≥3 LDS-buffer count

`Solution.py:3982-3989`:

```python
numLdsBlk = 1 if state["1LDSBuffer"] == 1 else 2
if state["PrefetchGlobalRead"] >= 3:
  numLdsBlk = state["PrefetchGlobalRead"]
  state["1LDSBuffer"] = 0
if state["PrefetchGlobalRead"] >= 2 and state["DtlPlusLdsBuf"]:
  numLdsBlk = state["PrefetchGlobalRead"] + 1
```

So `numLdsBlk` ∈ {1, 2, PGR, PGR+1}. PGR=2 + DtlPlusLdsBuf gives 3 LDS
buffers; PGR=3 gives 3 (or 4 with DtlPlusLdsBuf); PGR=k gives k (or k+1) for
k≥3.

### 3.3 ExpandPointerSwap × PGR

`Solution.py:1794-1802`:

```python
if state["ExpandPointerSwap"]:
  if not (bufferLoad and ( state["PrefetchGlobalRead"] == 1 \
          or (state["PrefetchGlobalRead"] > 1 and \
              (state["ProblemType"]["DataType"].isDouble() or state["ProblemType"]["DataType"].isDoubleComplex()))
          or (state["ProblemType"]["Sparse"] and state["PrefetchGlobalRead"] > 0))):
    state["ExpandPointerSwap"] = False
```

EPS is silently disabled for PGR≥2 *unless* (D/Z gemm) or (Sparse). Yet another
silent override; comment at `:1795-1796` is the only documentation.

`Solution.py:1046-1048` additionally forces `ExpandPointerSwap = False` when
`PGR >= 2 and ExpandPointerSwap` *inside* `Solution.isDirectToVgprDoable` —
i.e. only when DirectToVgpr is being evaluated. This is a DTV-specific
silent override, not a general one; the general EPS-vs-PGR override at
`:1794-1802` is separate.

---

## 4. Body-emit phase mapping per (PGR, PLR)

The four body phases referenced in the bead description (ML-1, ML, NGL, NLL)
correspond to:

* **ML-1 ("main loop minus one")** — the standard unroll-loop body emitted
  inside `_loopBody`, possibly emitted twice when `loopCopies==2` (driven by
  `expand = ExpandPointerSwap` or `needSecondLoop` for DTV/ULSGRO).
* **ML ("main loop")** — same `_loopBody` machinery; the "ML vs ML-1"
  distinction in this codebase comes from
  `expand = kernel["ExpandPointerSwap"]` ⇒ `loopCopies = 2` at
  `KernelWriter.py:4918`. With EPS, even and odd unroll-iter copies of the
  body are emitted; without EPS, a single copy (1 `_loopBody`).
* **NGL = NGLL ("NoGlobalLoadLoop")** — emitted only when `PGR ≥ 2`
  (`KernelWriter.py:4994` `for remainPgr in range(kernel["PrefetchGlobalRead"]-1, 0, -1):`
  is a no-op range when PGR ≤ 1). Loops `PGR-1` times → 1 NGLL for PGR=2,
  2 NGLLs for PGR=3, etc. May be doubled (`needSecondNGLL`) for DTV /
  ULSGRO with PGR≥2 (`:4924`).
* **NLL = NoLoadLoop (Ord/Opt label variants)** — emitted iff
  `PGR > 0 and not SuppressNoLoadLoop` (`KernelWriter.py:5017-5018`):
  ```python
  if kernel["PrefetchGlobalRead"]:
    if not kernel["SuppressNoLoadLoop"]:
      ...
      module.add(self.noLoadLoop(..., isNGLL=False, ...))
  ```
  Loop count = `NLLnum = 2 if NeedNLLOddEven else 1` (`:5024`), where
  `NeedNLLOddEven = isDTV` (DirectToVgprA or DirectToVgprB). The Opt vs Ord
  variant is selected by `isOptNLL` in the call (always `False` from
  `kernelBody`, but the GSU subcomponent — `Component.GSU.find(self).noLoadLoop`,
  invoked at `:5026` — and `OptNoLoadLoop` >0 may emit OptNLL labels;
  see `KernelWriter.py:3379-3387, 3455-3475` for the Opt/Ord label tag
  emission). **`SuppressNoLoadLoop` is already conditioned on
  `bufferLoad and PGR == 1 and (GSU == 1 or GSU == -1)` at
  `Solution.py:1646-1648`, so PGR != 1 always emits NLL when PGR > 0**
  (i.e. SNLL cannot suppress the PGR≥2 NLL).

The conditional NGL emit logic the bead specifically asked about is the loop
at `KernelWriter.py:4994` — the range collapses to empty for PGR ∈ {0, 1},
producing exactly the "shape difference" the existing probe runs detected for
`(PGR=1, PLR=0) vs (PGR=2, PLR=1)`.

### 4.1 Emit-phase table

Below, for each (PGR, PLR) pair that survives §2's reject filter under the
"common MFMA / non-DTV / non-DTL / non-Sparse / non-CMS" path:

| PGR | PLR | ML copies (loopCopies) | NGL copies | NLL copies | Notes |
|-----|-----|------------------------|-----------|------------|-------|
| 0 | 0 | 1 (or 2 if EPS) | 0 | 0 (NLL gated on `PGR > 0`) | Pure non-prefetch path, GR/LW/sync inside main loop (`KernelWriter.py:3601-3627`). |
| 0 | ≥1 | 1 (EPS auto-disabled per §3.3 unless D/Z) | 0 | 0 | Same as above; PLR>0 just enables LR prefetch inside the body. |
| 1 | 0 | 1 (or 2 if EPS) | 0 (range PGR-1..1 empty) | 1 (or 2 if DTV; but DTV needs PGR≥1, not 0 — both A+B DTV forces PGR=1, see O1) | "OrdNLL" only; NLL may be suppressed if `SuppressNoLoadLoop=1` ∧ buffer-load ∧ GSU∈{1,-1} (Solution.py:1646). |
| 1 | ≥1 | 1 (or 2 if EPS) | 0 | 1 (or 2 if DTV) | The classic MFMA+PLR=1+PGR=1 layout. |
| 2 | 0 | 1 (or 2 if EPS, but EPS forced off unless D/Z — §3.3) | 1 (or 2 if `needSecondNGLL`) | 1 (or 2 if DTV) | CMS schedule at `validator-branch CustomSchedule.py:1321` exists for this. Skips PLR-prefetch chunk in `kernelBody`, takes the `:4883` branch. |
| 2 | 1 | 1 / 2 | 1 / 2 | 1 / 2 | The most-common gfx950/gfx1151 PGR=2 layout. |
| 2 | ≥2 | 1 / 2 | 1 / 2 | 1 / 2 | Subject to H14/H15 (LoopIters bound) and H17 (PLR>=LoopIters silent demote). |
| ≥3 | ≥1 (PLR=0 rejected H3) | 1 / 2 | PGR-1 (≥2) | 1 / 2 | Each NGLL gets `LabelNGLL = "NoGlobalLoadLoop_<remainPgr>"` (`KernelWriter.py:3392-3399`). |

Caveat on ML/ML-1: `loopCopies` is always 1 or 2 in `KernelWriter.kernelBody`.
The `lc==0` body inside the loop is "ML-1" (not the final iteration), `lc==1`
is "ML" (final). `loopCopies == 1` means just one body that serves both roles
(`KernelWriter.py:4918, 4960-4965`).

The loop-counter check
`s_cmp_le_u32 loopCounter, hex(endCounter); s_cbranch_scc1 NoGRloopAfterABLoop`
at `KernelWriter.py:4951-4955` uses `endCounter = kernel["PrefetchGlobalRead"]`.
This is the runtime guard that selects "early-exit to no-GR loop" vs
"continue main loop".

### 4.2 OptNLL vs OrdNLL labels

`KernelWriter.py:3384-3387`:
```python
isOptNLLComment = "Opt" if isOptNLL else "Ord"
startComment = "%s. %s - Begin " % (isOptNLLComment, LoopNameComment)
```

So the assembly comment at the start of every NLL says either `Opt. NoLoadLoop`
or `Ord. NoLoadLoop`. The single direct call from `kernelBody` (line 5039,
5060) always passes `isOptNLL=False` → "Ord". The OptNLL is emitted by the
GSU subcomponent (`gsuComponent.noLoadLoop` at line 5026); its emit logic was
not re-traced for this report (out of scope — Phase A is PGR/PLR-centric).

### 4.3 SubtileBasedKernel path

`KernelWriter.py:9326-9329`:
```python
if not kernel["UseSubtileImpl"]:
  (error, kb) = self.kernelBody(kernel, tensorParametersA, tensorParametersB)
else:
  (error, kb) = self.kernelBodySubtile(kernel, tensorParametersA, tensorParametersB)
```

`UseSubtileImpl=1` only fires on gfx950 (`Solution.py:669-670`), so it is
NOT relevant to gfx1151 (the parent ot2 investigation), but it is worth noting
that this path uses a *separate* body-emit function (`kernelBodySubtile`,
defined at `KernelWriter.py:4201`) and its NLL/NGL emit story is not
described by §4.1. PGR is constrained to `{0, 2}` for subtile per H8.

---

## 5. SIA-3 / CMS eligibility per (PGR, PLR)

### 5.1 SIA = 3 eligibility

SIA=3 is the validator-eligible algorithm. Constraints found:

* SIA must be in `{0, 1, 2, 3}` (`ValidParameters.py:330`).
* `DirectToVgpr*` requires SIA=3 (`Solution.py:1026-1027`, H12).
* `Stream-K` requires SIA ∈ {2, 3} (`Solution.py:1350-1351`).
* `PGR ≥ 3` requires SIA=3 (H4).
* `1LDSBuffer=1` requires SIA ∈ {2, 3} (or SIA=1 + no SLW)
  (`Solution.py:4086-4087`).
* `DtlPlusLdsBuf` requires SIA=3 (`Solution.py:3970-3972`).
* `UsePLRPack` requires SIA=3 (`Solution.py:2092-2093`).
* `UseSubtileImpl` rejects SIA ∈ {1, 2} (`Solution.py:713-714`); accepts 0
  or 3.

There is **no rule that forbids any specific (PGR, PLR) pair from running
under SIA=3**. SIA=3 is the most permissive algorithm: all 12 of the legal
combinations from §2.3 are SIA=3-eligible. Therefore **every legal
(PGR, PLR) pair the validator might encounter is SIA=3-eligible**. Conversely
SIA ∈ {0, 1, 2} restricts the space (e.g. SIA=2 forces PLR=1 per O2; PGR≥3
forbidden under non-3 SIA per H4; etc.).

### 5.2 `UseCustomMainLoopSchedule = 1` (CMS) eligibility

CMS resolution is at `Solution.py:2054-2066`:
```python
if state["UseCustomMainLoopSchedule"] in [-1, 1]:
  state["SwapGlobalReadOrder"] = 0
  state["UsePLRPack"] = 0
  hasCMS, _ = hasCustomSchedule(state)
  if state["UseCustomMainLoopSchedule"] == 1 and not hasCMS:
    reject(state, printRejectionReason, "UseCustomMainLoopSchedule=1 but CMS is not supported")
  state["UseCustomMainLoopSchedule"] = 1 if hasCMS else 0
```

CMS-eligibility is determined by whether a CMS schedule for the current
`TileConfig` is registered. The TileConfig captures
`(MT0, MT1, DU, PGR, PLR, DTL, DPLB, WSGRA, WSGRB)` per
(validator-branch) `Components/CustomSchedule.py:808-809`:
```python
kernel_tile_config = TileConfig(MT0, MT1, DU, PGR, PLR, DTL, DPLB, WSGRA, WSGRB,
                                isa=kernel_isa, wavefront_size=kernel["WavefrontSize"])
```

There are 70 `tile_config=TileConfig(...)` literals registered in the
validator-branch `Components/CustomSchedule.py` (one additional `TileConfig(...)`
appears as a constructor at `:808` for the runtime kernel-config; that is not
a registered schedule). Distribution of (PGR, PLR) across the 70 schedules
(extracted with
`grep "tile_config=TileConfig(" CustomSchedule.py | sed -E 's/.*TileConfig\(([0-9]+), *([0-9]+), *([0-9]+), *([0-9]+), *([0-9]+).*/PGR=\4 PLR=\5/' | sort | uniq -c`):

```
2  PGR=1 PLR=0
2  PGR=1 PLR=1
11 PGR=2 PLR=0
55 PGR=2 PLR=1
```

Concrete examples for each combination:

| (PGR, PLR) | First file:line | Tile geometry |
|------------|-----------------|----------------|
| (1, 0)     | `:7394` | `TileConfig(128, 96, 32, 1, 0, 0, False, 0, 0, ...)` |
| (1, 0)     | `:7577` | `TileConfig(128, 64, 64, 1, 0, 0, False, 0, 0, ...)` |
| (1, 1)     | `:6138` | `TileConfig(128, 80, 64, 1, 1, 0, False, 0, 0, ...)` |
| (1, 1)     | `:6945` | `TileConfig(128, 80, 64, 1, 1, 0, False, 0, 0, ...)` |
| (2, 0)     | `:1321` | `TileConfig(256, 256, 128, 2, 0, 1, False, 0, 0)` |
| (2, 1)     | `:872`  | `TileConfig(256, 96, 64, 2, 1, 1, False, 0, 0)` |

So **CMS-registered schedules cover four (PGR, PLR) combinations:
(1, 0), (1, 1), (2, 0), (2, 1)**. PGR=0 and PGR≥3 have *no* CMS schedules
registered. CMS+SubtileImpl is explicitly rejected
(`Solution.py:2068-2069`). `UseCustomMainLoopSchedule=-1` (auto) is therefore
equivalent to `1` only when (PGR, PLR, plus tile geometry, layout, dtype,
DTL, DPLB, WSGRA, WSGRB, vector widths, MatrixInstruction, MIWaveGroup)
matches one of the 70 TileConfigs (predicate at validator-branch
`Components/CustomSchedule.py:794-833`).

In practice the dominant combination is (PGR=2, PLR=1) at 55/70 = ~79% of
registered schedules, with (PGR=2, PLR=0) at 11/70 = ~16% and the four
PGR=1 schedules covering edge cases.

---

## 6. Open questions and implicit assumptions

These are the things I flagged during the audit that warrant follow-up:

1. **PGR > 2 is schema-allowed but I found no in-tree solution YAML that
   specifies it.** `ValidParameters.py:273` admits PGR up to 16; the
   `range(3, 17)` extension was added with a comment block describing
   "DirectToLds only" semantics. I did NOT exhaustively grep the
   `gfx*/Equality/*.yaml` library logic files for explicit
   `PrefetchGlobalRead: 3..16`, so this is "not observed in my read,"
   not "guaranteed absent." The `validator_coverage_probe` on the validator
   branch (`Tensile/Components/GFX1151_AUDIT/validator_coverage_probe.py:140-141`)
   *captures* per-kernel PGR/PLR but I have not read its run-output logs to
   know what PGR values it actually observed; the README at
   `Tensile/Components/GFX1151_AUDIT/README.md` only describes a single
   gfx1151 HHS sweep and does not cite specific PGR values seen.

2. **PLR mod-LoopIters silent collapse.** `numItersPLR = PLR % LoopIters` at
   `KernelWriter.py:6054, 6056` means a yaml requesting `PLR == LoopIters`
   silently behaves like `PLR == 0`. There is no assertion that
   `PLR < LoopIters`. The reject H14/H15 only fires when
   `LoopIters - PLR*wlrMultiple < 0` (i.e. when wlrMultiple is involved), not
   for plain `PLR == LoopIters`. This is a real footgun for tuners.

3. **Override O2 (SIA=2 forces PLR=1).** A YAML that asks for PLR=0 and
   SIA=2 is silently rewritten to PLR=1, with only a `print2(...)` at
   `Solution.py:3242` as the trace. No reject; users see a different schedule
   than they asked for.

4. **Override O1 (DTV-A+DTV-B forces PGR=1, PLR=0).** Same pattern at
   `Solution.py:931-936` — silent rewrite, no reject.

5. **Override O3 (`ForceUnrollSubIter` overrides numItersPLR=1).** At
   `KernelWriter.py:6222-6225`, regardless of yaml PLR. This is *post*
   Solution validation, so the YAML PLR is "stored" but never consumed.

6. **`(PGR=0, PLR=0)` is legal but barely defended.** `SIA.py:304, 497` have
   `or writer.states.numItersPLR == 0` branches that suggest the author
   expected this to be exercised, but I did not find any rule asserting
   correctness in this configuration. The validator (Phase B's job) should
   confirm whether real schedules are ever produced and whether the validator
   has rules for this corner.

7. **`SuppressNoLoadLoop` only applies under `PGR == 1`** (per
   `Solution.py:1646-1648`). For PGR≥2, NLL is *always* emitted when PGR>0;
   the user-visible `SuppressNoLoadLoop=True` setting is silently overridden
   to `False`. Comment at `Solution.py:1644-1645` documents this.

8. **PGR≥3 acceptance is only enforced as four separate rejects** (H2-H5)
   rather than a single guard. If a future `Solution.py` change adds new
   PGR-conditional branches without re-examining all four, regressions are
   easy. Recommend a single
   `assert PGR<3 or (DTLA and DTLB and PLR>=1 and SIA==3 and not Sparse)`
   gate.

---

## 7. Probe extensions

I did not extend the existing `validator_coverage_probe.py` for this phase;
all evidence above is from static reading of `Solution.py`,
`KernelWriter.py`, `KernelWriterAssembly.py`, `ValidParameters.py`,
`Common/GlobalParameters.py`, (validator-branch) `Components/CustomSchedule.py`,
and (validator-branch) `Components/CMSValidator.py`. No new files committed
under `Tensile/Components/GFX1151_AUDIT/` other than this report. The probe
stays pristine; the per-kernel logging it already does
(`PGR=…  PLR=…  DTL=…/…  CMS=…  SIA=…`,
`Components/GFX1151_AUDIT/validator_coverage_probe.py:196-198`) is already
sufficient to corroborate the §4.1 emit table once a sweep is run, which is
Phase B's responsibility.

---

## 8. Source citation index

Files referenced (all paths absolute, but listed as repo-relative for
brevity):

| Path | Purpose |
|------|---------|
| `Tensile/Common/ValidParameters.py:255-330, 368` | PGR/PLR/SIA/OptNoLoadLoop schema |
| `Tensile/Common/RequiredParameters.py:83-87` | PGR/PLR named in required-params list |
| `Tensile/Common/GlobalParameters.py:411-412` | Default value `PrefetchGlobalRead: [1]` and `PrefetchLocalRead: [1]` |
| `Tensile/SolutionStructs/Solution.py:638-647, 669-716` | SIA=3 reorder logic; UseSubtileImpl PGR/SIA constraints |
| `Tensile/SolutionStructs/Solution.py:893, 931-957, 1026-1057` | DTV constraints on PGR/PLR/SIA/TLU |
| `Tensile/SolutionStructs/Solution.py:1635-1648` | BufferLoad ∧ PGR ≥ 2; SuppressNoLoadLoop gating |
| `Tensile/SolutionStructs/Solution.py:1790-1802, 4082-4087` | SIA=2 PLR/EPS overrides; 1LDSBuffer/SIA constraints |
| `Tensile/SolutionStructs/Solution.py:2051-2102` | UseCustomMainLoopSchedule resolution; UsePLRPack gating |
| `Tensile/SolutionStructs/Solution.py:3866-3893, 3957-3989, 4014-4045` | ScheduleGROverBarrier, DtlPlusLdsBuf, NumLdsBlk per PGR |
| `Tensile/SolutionStructs/Solution.py:4414-4467` | LoopIters/PLR/wlrMultiple rejects |
| `Tensile/SolutionStructs/Solution.py:4475-4549` | PGR ≥ 3 hard rejects; UnrollLoopSwapGlobalReadOrder gating |
| `Tensile/KernelWriter.py:451, 629-652` | makeSchedule isNGLL parameter; perIterLocalWriteCodeNGLL field |
| `Tensile/KernelWriter.py:2910-3370` | noLoadLoopBody (full body) |
| `Tensile/KernelWriter.py:3377-3480` | noLoadLoop (Opt/Ord/NGLL label dispatch) |
| `Tensile/KernelWriter.py:3488-3640` | _loopBody (per-unroll-iter body), PGR=0 GR-inside-loop emit |
| `Tensile/KernelWriter.py:4581-5070` | kernelBody (full) — driver of ML/NGL/NLL emit |
| `Tensile/KernelWriter.py:6052-6056, 6222-6225` | numItersPLR derivation; ForceUnrollSubIter override |
| `Tensile/KernelWriter.py:9320-9329` | kernelBody vs kernelBodySubtile dispatch |
| `Tensile/KernelWriterAssembly.py:15793-15822` | openPrefetchGlobalRead2orMore / closePrefetchGlobalRead2orMore (skipPGRn label emission) |
| `Tensile/KernelWriterModules.py:143-145` | "Skip force waitcnt0" PGR≥2 branch |
| `Tensile/Contractions.py:640-732` | PGR plumbed through to host-side contractions metadata |
| `Tensile/Components/SIA.py:304, 467, 494, 537, 602, 624-700` (develop) | SIA per-PGR scheduling branches |
| (validator-branch) `Tensile/Components/CustomSchedule.py:75-115, 690-1649` | TileConfig schema + all 9 PGR=2 CMS schedules |
| (validator-branch) `Tensile/Components/CMSValidator.py:1-100` | ISA_CONCERN_CATALOG; PGR/PLR not directly used (only through TileConfig predicate) |
| (validator-branch) `Tensile/Components/GFX1151_AUDIT/validator_coverage_probe.py:140-200` | PGR/PLR per-kernel logging |
| (validator-branch) `Tensile/CustomKernels/Custom_Cijk_Alik_Bljk_F8BS_BH_SAB_UserArgs_*.s:1618-1696, 2250` | Captured asm labels `skipPGR2`, `toPGR1` confirming PGR=2 emit shape |

---

## Critical review notes

A second-pass critical review was performed against the draft above. The
review checked: (1) every claim in §2 against the cited line numbers,
(2) the §4.1 emit table rows against `kernelBody` line 4994, 5017,
(3) the §5.2 CMS PGR/PLR enumeration against actual `TileConfig(...)`
literal occurrences in `Components/CustomSchedule.py`.

Findings of the review pass and resolutions:

* **R1.** Original draft of §1.1 said "PGR has values 0, 1, 2 only" — this
  was contradicted by `ValidParameters.py:273` which extends to 16. Fixed:
  §1.1 now correctly enumerates `[0, 1, 2, 3..16]` and §6 explicitly flags
  the `>2` extension as "schema admits, in-tree YAMLs may not exercise".
* **R2.** Original draft of §2.1 listed H1 as "PGR≥2 → reject if BufferLoad=0";
  rechecked source (`Solution.py:1635-1637`) — the implication is correct
  (`if not bufferLoad: ... if state["PrefetchGlobalRead"] >= 2: reject`).
  Wording sharpened to make the conditional direction unambiguous.
* **R3.** Original §4.1 claimed `loopCopies` is always derived from
  `ExpandPointerSwap`. Rechecked `KernelWriter.py:4918-4928` — `loopCopies`
  is *also* forced to 2 when `needSecondLoop` is true (DTV or ULSGRO), even
  when `expand=False`. Added the `needSecondLoop` clause to §4.
* **R4.** Original §4.2 attributed OptNLL emission to `kernelBody` directly;
  rechecked — `kernelBody` always passes `isOptNLL=False`, so OptNLL labels
  come from the GSU subcomponent at line 5026. Wording corrected.
* **R5.** Original §5.2 said "all CMS schedules are PGR=2, PLR=1". Rechecked
  the `TileConfig(...)` literals: 70 schedules total, with distribution
  (PGR=1, PLR=0)=2, (PGR=1, PLR=1)=2, (PGR=2, PLR=0)=11, (PGR=2, PLR=1)=55.
  My earlier draft also undercounted by inspecting only the first nine
  `TileConfig(...)` lines via head-truncated grep output, missing the PGR=1
  schedules at `:6138, :6945, :7394, :7577`. Fixed: §5.2 now lists all four
  combinations with examples and proportions, and the misattributed citation
  to `Solution.py:411-412` (which is unrelated code) was corrected to
  `GlobalParameters.py:411-412`.
* **R6.** Original §3.3 cited only `Solution.py:1797-1802` for EPS+PGR
  interaction; reviewer noted `Solution.py:1046-1048` also touches it
  (PGR≥2 + ExpandPointerSwap branch), but the second branch is *inside*
  `isDirectToVgprDoable` and only fires for DTV kernels — NOT a general
  EPS-vs-PGR rule. Wording corrected to make the DTV scoping explicit.
* **R7.** Reviewer noted that §2.2's override O5 description was vague.
  Fixed: now cites the exact PGR==0/PLR==0 disable conditions at
  `Solution.py:2101-2102`.
* **R8.** Reviewer asked whether `(PGR=0, PLR=0)` is truly legal. Re-grepped
  Solution.py — no reject mentions both PGR==0 and PLR==0 simultaneously, but
  `H17` (`Solution.py:4419-4420`) is the closest thing (silently demotes).
  Added §6 item 6 documenting this corner.
* **R9.** Reviewer asked about Sparse + PGR/PLR interactions. Rechecked
  `Solution.py:4485-4487` (PGR≥3 + Sparse rejected) and `Solution.py:4179-4182`
  (Sparse forces `AssertSummationElementMultiple = 8`). Decided not to expand
  §2 with Sparse-only rules — out of scope for Phase A's primary
  (PGR, PLR) cross-product, but referenced as a constraint in H5.

A second restructuring review pass was not performed because the first-pass
fixes were localized (each touched a single section) and did not change the
report's overall structure.
