---
name: rocke-tiling-api
description: Design and author a GPU kernel with the rocKE tiling API, end-to-end. Use for "I want to design a kernel with tiles", "design a new kernel for this algorithm/pipeline", turning an algorithm + problem into a tiled IRBuilder kernel, or iterating on an existing tiled kernel. Dispatches the Tiling Kernel Architect (lead) who consults MMA/LDS/Profiling specialists and drives layout-viz / bank-conflict / build / profile. GEMM-family today; open to novel algorithms (offers to learn + persist them).
argument-hint: <algorithm or "design a kernel for ..."> [problem/tile/pipeline details]
---

# rocKE Tiling-API Kernel Design Skill

You are the **coordinator/hands**. The **Tiling Kernel Architect** (Tiling Expert) is the **lead brain**: it
makes the design decisions and hands you directives; you execute them and report. **Authoring is yours;
JUDGMENT is not.** You may write files, run tools and assemble reports. You may NOT decide anything called
*free / cheap / optimal / sound / a NOP / a reorder / cross-lane*, and you may NOT invent a descriptor,
encoding, tile or transform — those come from an expert, every time.

## The output contract — what the USER sees (NON-NEGOTIABLE)

Expert output is for the architect and for you. A user watching a design run has none of the context that
makes expert prose meaningful, so streaming it buries the decision.

- **"Relay" means relay BETWEEN EXPERTS.** What reaches the user is a *reported design*, never a forwarded
  consult.
- **A finding reaches the user as a table row or a verdict line** — condensed, with its rationale, in the
  report. Never as prose, never mid-run, never quoted.
- **This governs what is STREAMED, not what is KEPT.** Verbatim expert output (encodings, Rs/Hs/Ps/Ys, expert
  tables) is retained for the design record, which is required to be reproduction-complete.
- **One line per dispatch BATCH when it starts** — what, why, rough duration, and that it cannot stream.
  Nothing when it returns.
- **Interrupt immediately for:** anything that invalidates a locked parameter, any expert **STOP** or **ASK**
  (the LDS Expert must ASK if its mode is missing; a failed `selftest` is a STOP), and any **provenance
  downgrade** (a number becoming simulated/unvalidated). Say what broke, give the options, and name the phase
  you will resume in.

```
✓  "Consulting MMA + LDS on the A/B operand layouts (~2 min, can't stream)."
✗  pasting a subagent's findings block or a tool's raw stdout into the conversation
```

## Non-negotiables

- **Clean slate — DERIVE, never COPY.** Never source a descriptor, encoding, parameter or shape from a
  sibling kernel, a nearby example or a prior session — re-derive it from THIS design's spec and stage table.
  A near-miss kernel is the most dangerous input there is. **Scope:** this applies to every kernel EXCEPT the
  subject of an EXISTING run, which is the baseline you are changing; it still applies to that kernel's
  siblings. Carry this rule into every subagent prompt. **A hand-typed tool config is a copy** — see "Derive
  tool inputs" below.
- **User-given lengths/strides are FIXED FACTS.** Interpret exactly as given; never re-interpret to fit a
  familiar kernel. If strides are NOT given, the major sweep is a QUESTION resolved in Frame — the user picks,
  or the architect picks and states why. Design&Emit always carries exactly one major triple.
- **Brief experts with ORDERED shape + strides, verbatim** — `A=(M×K) strides=(sM,sK)`, `B=(K×N)
  strides=(sK,sN)`, `C=(M×N) strides=(sM,sN)`. The contiguous axis is the stride-1 axis, PER TENSOR. Pass
  strides, never "row/col-major".
- **Take expert parameters VERBATIM** into the design. If an expert contradicts a locked param, CHALLENGE it
  with the expert — never silently reconcile.
- **Derive tool inputs from the RECORDING, never type them.** `/bank-conflict`'s config (`strides`, `origin`,
  `lds_swizzle`, `dtype_name`, `wtag`) must be pulled off the recorded transaction, exactly as
  `verify_lds_roundtrip` does. A hand-filled config is a second source of truth that drifts, and its failure
  is silent — it analyzes a different kernel and stamps yours on the answer. Hand-supplied ⇒ assert against
  the recording and say so in the report.
- Domain theory lives in the SOT docs the architect cites (`tiling_api_surface.md §2`,
  `label_flow_and_transforms.md`, `tiling_interleaving_design.md`, `mma_is_machinery.md`, `lds_banks.md`).
  Do not restate it here; the architect reasons from it, you do not.

## Validation — what is proven, and when (the design is not done until this is)

Validation is not one step at the end; it is a LADDER, and each rung proves something narrower than the
next. The failure to avoid is reading a low rung as if it were a high one — a structurally gated design that
computes the wrong answer passes every CPU-side check in this skill.

| Rung | Gate | Proves | Does NOT prove | When |
|---|---|---|---|---|
| 1 | arithmetic + budget screen | the design is EXPRESSIBLE (macro%wave, wave%atom, K%atom, vw vs dtype/stride-1, LDS fits) | anything about behaviour | Design&Emit, before emitting |
| 2 | `witness(pipe, kernel)` | the RECORDING IS COMPLETE — nothing emitted outside the decorated verbs | that what was recorded is right | every record |
| 3 | `verify_lds_roundtrip` (per space, non-vacuous) | each LDS space ROUND-TRIPS: the consumer reads the datum the store wrote | the value is the right value | every record |
| 4 | `verify_mma_soundness` (non-vacuous) | operand pairs form a SOUND MAC | the operands hold the right data | every record |
| 5 | **bit-exact vs a numpy golden** | **the kernel COMPUTES THE RIGHT ANSWER** | anything about speed | Finish (needs a GPU) |
| 6 | `/ck-profile`, `/bank-conflict --mode investigate` | the MEASURED perf + conflict numbers | correctness | Finish (needs the target GPU) |

**Rungs 1–4 are structural and CPU-only. Rung 5 is the correctness gate and there is no substitute for it.**
Passing 1–4 means the design is well-formed and self-consistent; it says nothing about arithmetic. So be
explicit: **throughout Frame→Iterate a design is structurally gated and arithmetically UNPROVEN.** Never
describe an un-run rung 5 as "validated" — Open decisions carries "correctness: not yet proven" until it
passes.

**The golden.** Random floats cannot prove bit-exactness — f16 accumulation reorders and the comparison
drowns in rounding. Use **small integers** (`rng.integers(-3, 4)`, the house pattern), which are exactly
representable, so the bar is `max_abs_diff == 0.0` — *not* `allclose`. Any non-zero difference is a real bug,
never rounding. A probe or kernel that is not bit-exact makes every downstream number meaningless.

**Re-validate after anything that changes the recording** — completion, and every Iterate change. Rungs 2–4
are cheap; re-run them every time. If completion changed the LDS stride, pad, swizzle or buffer count, the
LDS findings are void as well (Finish step 2).

**No GPU, or a host whose arch ≠ the target?** Rungs 5–6 cannot run. That is a legitimate stopping point —
but the design ships **UNVALIDATED**, and you say so in the report and in the design record rather than
letting the structural passes imply correctness.

**Correctness and performance are different gates.** Bit-exactness is the correctness gate; wall-time/TFLOPS
is the perf gate and only applies when the goal is performance. A faster kernel that is not bit-exact is not
a kernel.

## Prerequisites (Read First)

1. `../shared/prerequisites.md` — path resolution, dispatch rules, "Consult, Don't Improvise", fail-fast.
2. `../shared/temporary_file_policy.md` — probes/build dirs/renders are temporary; track + offer cleanup.

## Experts & sub-skills

Request by generic ROLE via the dispatch table (filenames below are pointers, not addresses): **Tiling
Expert** (lead architect), **MMA Expert** (operand/accumulator layouts, interleave, C-shuffle soundness),
**LDS Expert** (bank conflicts, swizzle/pad, LDS budget), **Profiling Expert** (`"hardware counters"`),
**GPU Expert** (non-MMA compute). Sub-skills you invoke: `/layout-viz`, `/bank-conflict`, `/ck-build`,
`/ck-profile`. Dispatch with model `opus`.

**Required brief, per expert** — beyond the ordered tensors above. Mark every field GIVEN or PROPOSED:

| Expert | Must also carry |
|---|---|
| **MMA** | MMA atom (16×16 vs 32×32 — interleave requires 16×16), wave size, macro/wave/thread tiles, waves_m×waves_n, issue order (M-outer/N-outer), **and WHICH STAGE the question is about** (constraints come from the consumer) |
| **LDS** | **`mode`** (`simulate` in Frame→Iterate, `investigate` in Finish — its rule 0 requires this and it will ASK if missing), LDS-space stride (`tile_free + lds_pad`), origin, swizzle, access width, waves in the cooperative store, NB + wave size **for the target** |

**Coupled ownership.** "Re-consult the owning expert" is not enough: an LDS lever that changes **vector width,
LDS stride, or the distribution** (narrowing swizzle, redistribute) also re-consults **MMA**; an MMA change
that alters any LDS descriptor also re-consults **LDS**. Only a pad preserving `b128` alignment is
single-owner.

## Workflow

```
  NEW      ─→ Frame ─→ Design&Emit ─────────────→ Iterate ─→ Finish
              (ask +   (architect → CONFIRM →     (viz,      (complete,
               confirm)  specialists → skeleton    re-record)  verify, offers)
                         → record → witness → report)
  EXISTING ─→ Recover ──────────────────────────→ Iterate ─→ Finish
              (pin config, record, verify claim)

  Frame..Iterate are CPU-only (emit, record, render, simulate). The GPU is needed only in Finish.
```

**Pick the arm first.** Names an existing kernel/file ⇒ EXISTING; do not re-interview. Ambiguous ⇒ ASK
(guessing NEW and re-interviewing is the annoying failure).

### Frame (NEW)

Settle the output location: propose `kernels/<kernel>/` (code + `docs/` + `docs/viz/` + `tmp/`) in one line
and proceed unless the user objects. Only rocke-CORE changes touch the rocke source tree.

**Round 1 — REQUIRED, never invented.** `AskUserQuestion` takes at most FOUR questions per call, so this is
two calls, not one:
- **1a (the four that shape everything):** goal (correctness / peak TFLOPS / a target / learning) ·
  algorithm · **ordered tensor descriptors** (lengths, strides, dtype, memory space) · arch + wave size.
- **1b (two, and both are cheap to answer):** **kernel name** (names the output folder and every report) ·
  **caller-side constraints** — grid/launch convention, pointer/ABI shape, batch stride, any fusion it must
  absorb, any fixed occupancy or LDS budget it shares. "None" is a fine answer to the second; ask anyway,
  because discovering a caller constraint at authoring time is the expensive place to discover it.

Required items are asked until answered — the two-round cap below is on the OPTIONAL levers, not on these.

**Round 2 — OPTIONAL levers**, offered once: layout style · tiles · waves_m × waves_n · pipeline (named
pattern or step list). "Skip, you decide" is a first-class answer. **Round-2 answers are PROPOSALS subject to
expert validation** — an expert may return one INVALID, which is a blocking issue.

**If the user has no preference on layout style, do not just name the three — give them the trade-off.** The
MMA Expert owns the detailed comparison and the concrete encoding; surface the choice, then defer:
- *canonical (basic):* direct hardware placement (label == position) — simplest, no derivation or relabel.
  But it is locked to native placement, so a store- or coalesce-friendly layout can force strided or
  cross-lane movement elsewhere in the chain.
- *interleaved:* labels flow for wide coalesced load/store, LDS reuse, and a store-friendly derived C — at
  the cost of a derivation (position ≠ label) plus a dtype-graded reorder, which is often free via a symmetry.
- *custom:* the user supplies their own `make_tile_desc` encodings; the MMA Expert validates soundness and
  vectorization before anything is built on them.

Then dispatch the architect to **CLASSIFY**: GEMM-family → Tiling+MMA+LDS · non-MMA → Tiling+GPU · NOVEL →
the architect offers to learn it *with the user* (derive data-movement, compute structure, tensor signature,
pipeline) before Design&Emit. Route non-GEMM explicitly; do not force-fit.

### Design&Emit (NEW)

**1. Architect proposes (cheap).** Dispatch with the round-1 params. It returns the pipeline design + its
`GIVEN / ASSUMED / CONFIRM` lists.

**2. CONFIRM before spending.** Put the architect's `CONFIRM` list to the user as one `AskUserQuestion`, each
assumed value shown as the default so "looks right" is one click. This is a decision list, not an expert
relay — it does not violate the output contract. Anything the user skipped in round 2 is ASSUMED-pending-
confirm, never settled. Do not run specialists before this returns.

**3. Specialists.** Execute the architect's directives (MMA/LDS/…). They may construct candidate encodings and
run `mma_pair_compatible` / `classify_transform` / `RegisterMapper` standalone — these need only encodings, not
a kernel — and must state the target a `cross_lane` verdict was classified against. Feed results back until
the plan is concrete: every judgment word owned by an expert, no bare assumptions.

**4. Exit gate — arithmetic + budget.** Before emitting: macro%wave, wave%atom, K%atom, vector width vs dtype
and the stride-1 axis, and the LDS budget vs capacity/occupancy. Report as a pass/fail row. A design that
fails does not proceed.

**5. The architect AUTHORS the skeleton; you write it verbatim.** Choosing encodings, origins, vector widths
and swizzle IS design — not yours. The skeleton is the smallest runnable IRBuilder + tiling-API function
satisfying:

> **Skeleton invariant — every distinct TRANSACTION is emitted at least once, at the real descriptors,
> strides, dtype, swizzle and distribution — including the steady-state buffer swap and the full epilogue /
> C-shuffle chain.** Only *repetition* (loop trip counts, multi-tile iteration), host glue, and *edge-case*
> predication may be omitted.
>
> Why each clause: omit the swap and `verify_lds_roundtrip` covers half the design; omit the epilogue and the
> C side — the derived C, the de-interleave, the CShuffle's own LDS space and its budget — is unanalysed;
> predication changes the emitted op kind and can change the achievable vector width, so wherever it is
> omitted the coalescing verdict is **provisional** and must be labelled so.
>
> **LDS stride, pad, swizzle and buffer COUNT are part of the skeleton, not of completion.** They set the
> bank map; changing them later voids every LDS finding.

The build fn must `return b.kernel, mma` (house convention — the recorder returns its result verbatim, and
`mma` is what carries arch/wave size into the recording).

**6. Record + witness.**
```python
(kernel, mma), pipe = tiling_recorder.record_build(build_fn, *args, **cfg)
tiling_recorder.witness(pipe, kernel)        # CoverageError = the recording is SHORT
```
`record_build` only sees the decorated verbs; anything emitted outside them (a raw `b.mma`) is invisible, and
every render and conflict number would inherit the hole. A `CoverageError` is a hard stop back to the
architect and an API-gap candidate.

**7. Gates — CALL them, and require them NON-VACUOUS.** `render_sweep`/`view` do NOT run them:
```python
halves = auto_pipeline.verify_lds_roundtrip(pipe, space_id, tile_k=<K>)   # per LDS space
n_mma  = auto_pipeline.verify_mma_soundness(pipe)
```
`verify_lds_roundtrip` returns `[]` when a space has no store or no read; `verify_mma_soundness` returns `0`
when no MMA was emitted. **Both are PASS BY OMISSION** — exactly what an incomplete skeleton produces. Require
a non-empty half list for EVERY LDS space and a count matching the design's MMA issues. Empty/zero is **NOT
RUN**, and goes in Open decisions.

**8. Report** (the Output Format below), built from the RECORDING. **The recording is authoritative about the
CODE, not about what the design should be** — a disagreement means either the skeleton mis-expresses the
design or the design was wrong. STOP, hand both to the architect, and let it say which. Never report a design
the architect has not reconciled.

### Recover (EXISTING)

1. **Pin the config.** Build fns are parameterised FAMILIES. Enumerate the parameters, get the exact values
   from the user (or read them off the call site / test they named), echo them back. **Never record at
   defaults** — that analyses a kernel the user is not working on while labelling it theirs.
2. **Get the goal** — "change X" and "why is it slow" need different work.
3. **Record + witness + gates** at the pinned config (Design&Emit steps 6–7), and reflect the recovered
   parameters as the report's parameters table + pipeline for confirmation.
4. **Verify any asserted defect BEFORE fixing it.** "The A store is conflicting" is a claim; run the owning
   sub-skill and confirm it. Fixing an unverified defect is the failure `/bank-conflict` exists to prevent.
5. **Dispatch the architect once** with the recovered pipeline + the requested change, so it classifies the
   change (structural → architect; layout → MMA; memory → LDS) and emits directives. Then → Iterate. Do not
   redesign anything the user did not ask you to touch.

### Iterate

**If the request already names the analysis, run it and skip the menu.** Otherwise ask, defaulting to the
architect's directives, each tied to the Open decision it settles ("coalescing — settles open decision #2"):

| Option | Runs | Covers |
|---|---|---|
| **Sweep** | `/layout-viz render_sweep` — L0 overview + L1 flows (roles are DERIVED per kernel; a GEMM yields prefetch A/B, lds_read A/B, compute, epilogue — a non-MMA kernel differs) | dataflow |
| **Bank conflicts** | `/bank-conflict <access> --mode simulate --arch <target>` | store **and** read, by different rules — the store via the write-port model, the read via the read-port model (`conflicts/access = max_depth − 1`). The read is **gfx90a-only and envelope-gated** (2 dwords/lane, no broadcast); outside that it is geometry-only and the cost is UNKNOWN until `investigate` — carry that in Open decisions |
| **Coalescing** | `/layout-viz` coalescing, one image per output major; ASM cross-check only once the kernel COMPILES (CPU-side `compile_kernel` + `llvm-objdump`, no GPU) — otherwise the diagram ships without it and must say so | global access |
| **All three** | the sweep + both L2 analyses | |

**Simulate-mode discipline.** Requires a REGISTERED model for that arch whose `selftest` passes; if the target
has none, `/bank-conflict` hard-stops — the Bank-conflicts option is **UNAVAILABLE**, and the choices are
build+validate a model on that hardware (a Finish task) or proceed with conflicts UNKNOWN in Open decisions.
Never substitute another target's constants. A SIMULATED number may **rank** candidates and **flag** one for
Finish; it may **not** disqualify a layout or justify adopting a fix. Simulate sees replays but not
instruction count or occupancy, so on simulated evidence alone you may adopt only zero-instruction levers
(free symmetry, `b128`-preserving pad, contiguity-preserving swizzle) — **never a narrowing swizzle or a
redistribute**, whose cost is instructions. Each adoption enters Open decisions as "unvalidated, re-test in
Finish". Driving a SIMULATED conflicts/access down across iterations IS the BC→0 antipattern wearing a caveat.

**If the goal is performance, take ONE baseline `/ck-profile` before the first change** — a perf run that
never measured "before" cannot report an improvement. Thereafter on request.

Each change: re-consult the owning expert(s) (coupled ownership above) **silently**, have the architect
re-author the affected skeleton part, **re-record + re-witness + re-gate**, update **only the affected report
section**. Every render comes from a fresh recording; a carried-over render is stale data.

**Exit: ASK.** "Is this design final, or keep iterating?" Nothing else triggers Finish — the user will not
spontaneously say FINAL.

### Finish

1. **Complete the kernel** (NEW) — grow the skeleton into the full kernel: repetition, host glue, edge-case
   predication. Do NOT rewrite it; the recorded pipeline IS the approved design. *(EXISTING: skip — the kernel
   already exists; verify your edits re-record to the approved pipeline.)*
2. **Re-record + witness + gates.** If completion changed the LDS stride, pad, swizzle or buffer count,
   **every Iterate LDS verdict is VOID and must be re-run.** Say so rather than carrying the old number.
3. **Rung 5 — verify bit-exact** vs a numpy golden (see Validation): integer inputs, bar is
   `max_abs_diff == 0.0`. Needs a GPU whose arch matches the target — check (`rocminfo`). If it cannot run,
   the design ships UNVALIDATED and the report must say so.
4. **Measure, then record.** In this order, because the design record is specified as reproduction-complete
   and carries measured numbers — offered earlier it gets written with `N/A` where its value lives:
   `/ck-build` + `/ck-profile` → `/bank-conflict --mode investigate` (promotes simulated numbers to measured;
   host GPU must BE the target arch) → **then offer** the design record. Spec + path: the Tiling Expert's
   "Per-kernel design record" + `docs/kernels/_TEMPLATE.md`; it carries expert reasoning and tables
   **verbatim** (the output contract governs streaming, not this).
5. **Persist NOVEL learnings** — signature + workflow into the right expert (pipeline/structure → Tiling;
   layout → MMA; memory → LDS), one concise entry, number-free. These are shared checked-in files: **show the
   diff and get the user's approval** before writing.
6. **API-gap proposals** — one file per gap under `docs/api_proposals/` using the template there. Propose
   only. (Record frictions as they happen, throughout — a `CoverageError` is one.)
7. **Cleanup** — after whichever offers the user accepted: promote keeper renders to `docs/viz/`, fold scratch
   findings into the report, clear `tmp/`, offer to remove probes/build dirs/containers.

## Output Format

The report is the primary user-facing output; Iterate updates sections in place rather than reprinting it.

```
## Tiling-API Kernel Design — <kernel>

- arm/phase: <NEW: Frame | Design&Emit | Iterate N | Finish   /   EXISTING: Recover | Iterate N | Finish>
             goal: <...> ; arch+wave: <...>
- inputs: GIVEN <...> | ASSUMED-confirmed <...> | DERIVED by <expert> <...> | OPEN <...>
- validation: rungs 1-4 <witness mem n/n mma n/n · round-trip halves per space · soundness n/n ·
              arith+budget pass/fail>   (empty/zero = NOT RUN, not pass)
  **correctness (rung 5): <bit-exact max_abs_diff 0.0 | NOT YET PROVEN | UNVALIDATED (no target GPU)>**
  perf (rung 6): <measured | not run>

### Design parameters
| dtypes | MMA atom | macro | wave | thread | threads | waves (m×n) | buffering | LDS stride/pad/swizzle |

### Pipeline
<pictograph: every stage, its function, the resources it uses>

### Layouts (all proposed, including the user's own)
| stage | layout | transform in | conditions | expert | provenance | verdict / rationale |
(conditions = what the verdict depends on, e.g. dtype/atom — never compress a conditional to an
 unconditional word. provenance inline on every number, e.g. "4-way (SIMULATED, pending counters)".)

### Visualizations
- sweep: <dir> | store conflicts: <SIMULATED c/a> | read conflicts: <SIMULATED c/a, or GEOMETRY if out of envelope> | coalescing: <verdict [provisional if unpredicated]>

### Open decisions
- <what is open, what would settle it>  · stages ABSENT from the recording  · unvalidated adoptions

### Glossary
- <every term of art used above>
```

Appended in Finish: `### Kernel` (bit-exact yes/skipped) · `### Learnings persisted` · `### API-gap proposals`.

## Glossary (for YOUR use — define these for the user on first use)

- **emit space** — the ops the build function actually emits. The authoritative record of the design: the
  design *is* the emitted ops, so it is inspected by recording, not by reading source intent.
- **transaction** — one mem↔reg movement (a load/store/fill), recorded as an object by `tiling_recorder`.
  Each transaction independently chooses its own distribution.
- **round-trip** — the per-LDS-space correctness invariant: each consumer register gets the datum stored at
  the address it reads.
- **served group** — the lanes the LDS arbitrates together (half-wave × dword phase).
- **binding stage** — the pipeline stage currently limiting wall-time. BC is a diagnostic; this is the objective.
- **K-stride aliasing** — LDS rows a whole number of banks apart, so stepping K lands on the same bank.
- **major sweep (RRR/RRC/CRC…)** — one letter per tensor (A,B,C) for which axis is stride-1; used when the
  user gives no strides.
- **MMA atom** — the hardware matrix instruction shape (e.g. 16×16×16). Interleave requires the 16×16 atom.
- **derived C / C-shuffle (CShuffle)** — the accumulator's post-MMA relabel + epilogue gather into a
  store-friendly order; may use its own LDS space.
- **Rs/Hs/Ps/Ys** — the encoding factorisation of a tile distribution.
- **L0/L1/L2/L3** — `/layout-viz` levels: overview · flows · analyses (coalescing, bank-conflict) · single panels.
- **BC** — `SQ_LDS_BANK_CONFLICT`, the hardware counter of LDS replay cycles. A diagnostic, never the objective.
- **NB** — number of LDS banks on the target (32 on CDNA). `bank(dword d) = d mod NB`.
- **vw / access width (`b32`/`b64`/`b128`)** — how many bits one lane moves per memory instruction:
  b32 = 1 dword, b64 = 2, b128 = 4. Wider is fewer instructions but a coarser bank pattern.
- **wtag** — the access-width tag (`"b64"`, `"b128"`) a bank-conflict config carries.
- **coalescing** — whether the lanes of one instruction hit a contiguous global range, so the memory
  system serves them in as few transactions as possible.
