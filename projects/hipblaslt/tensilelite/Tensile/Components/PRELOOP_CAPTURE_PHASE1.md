# Pre-mainloop prologue capture — Phase 1 investigation memo

Bead: `rocm-libraries-oram` (Phase 1 — investigation only, no code).
Branch base: `users/alvasile/validator_long_term_plans` (vlt), tip `ee723da495`.
Reference architecture: `rocm-libraries-dj1g`, commit `c1054a019c` (deferred-expansion pattern; the template for any prologue extension).

This memo answers the four investigation questions for whether the validator should
extend its current `FourPartCapture` (steady-state mainloop only) to also capture
the pre-mainloop prologue — the GR warmup, prefetch LRs, and (optionally) usePLRPack
Pack chain emitted between `setupNewTile` and `openLoop`.

References below cite line numbers in the worktree at tip `ee723da495`.

---

## Q1: Where is the prologue emitted?

**Walk of `KernelWriter.kernelBody` (`KernelWriter.py:4697`–`5067`) in emission order.**
The "prologue" — everything that lands in `module` between the kernel header and
the `openLoop` call at `:5067` — is built in this sequence. The CMS dispatcher
that the bead points at (`customMainLoopSchedule` -> `simdSpecDispatch` at
`:4581`/`:4587`) is the *body* of `_loopBody`, which is itself called from the
`for lc in range(0, loopCopies)` loop at `:5096`–`:5101`, well *after* the
prologue has already been appended to `module`. The bead's "between line 4581 and
4587" framing is correct for the **CMS macro dispatch site**, but the prologue
emission itself sits earlier — at the `kernelBody` top level (`:4697`–`:5066`),
not inside `_loopBody`.

| # | Code-emission call | KernelWriter.py | What it emits | Gating flags | Natural capture hook |
|---|---|---|---|---|---|
| 1 | `defineAndResources` | `:4721` | sgpr/vgpr declarations; not dataflow-bearing | always | n/a |
| 2 | `disableWmmaArbStall` | `:4722` | one s_setreg | always | n/a |
| 3 | `skComponent.preLoop` | `:4729` | StreamK loop init (sgpr arithmetic, label) | `kernel["StreamK"]>0` | n/a (out of scope per bead) |
| 4 | `createNegIdentityMatrix` | `:4733` | one-time VGPR init for TF32 emulation | `kernel["UseMFMAF32XEmulation"]` | n/a |
| 5 | `loopComponent.openPersistentLoop` | `:4738` | persistent-loop label/header | `PersistentKernel` | n/a |
| 6 | **`setupNewTile(... isOptNLL=False)`** | `:4739` | The biggest single emit. Calls `graWorkGroup`, `graTileAssignment{A,B,...}`, `graUnrollAssignment`, `graTileOffsets`, `graUnrollOffsets`, `graShift`, `graFinalOffsets`, `graAddresses`, then in `PrefetchGlobalRead` branch (`KernelWriter.py:2987–`) emits the **first round of GR** via `globalReadDo(kernel, 0, ...)` for A/B (and MX siblings). | always; PGR sub-branch gated `kernel["PrefetchGlobalRead"]` | wrap the returned `Module` and hand it to a builder |
| 7 | `openShadowInit` ... `closeShadowInit` | `:4750`–`:4759` | shadow-init block (initC, etc.) — no GR/LR/Pack | `doShadowInit and PGR` | n/a |
| 8 | `getWaitcntCodeForPGR` | `:4762` | s_waitcnt for the first-round GR | `PGR` | hook |
| 9 | optional `SBarrier` | `:4767` | barrier for streamK / multi-summation | `StreamK>0 or actualSummationLoops>1` | hook |
| 10 | **`preLoopLocalWriteDo`** | `:4771` | First round of **LDS writes** that consume the GRs in (6) | `PGR and not NoLdsWriteCode` | hook |
| 11 | `localWriteSwapOffsets`/`tdmSwapLdsOffset` (A, MXSA, MXSB, B) | `:4779`–`:4806` | swap LWA/LWB pointer state | always under PGR | n/a (sgpr ptr math) |
| 12 | **PGR>=2 prefetch chain** (the `for idxPgr in range(1, PGR):` loop) | `:4813`–`:4892` | For each additional PGR generation: `directToLdsM0Update` + `globalReadDo(kernel, 0, ...)` for A/B/MXSA/MXSB; `globalReadIncrementAB` for all but the last; per-iter `localWriteSwapOffsets`. **This is a second/third round of GR** (and any DTL LWs). | `PGR>=2` | hook |
| 13 | `lraAddressesInitFor3LDSBlk` | `:4898` | LR addr sgpr init | `IncLdsBufSwitch` | n/a |
| 14 | **prefetch-local block** (`if self.states.numItersPLR:`) | `:4902`–`:5018` | Sequential per-`plrIdx` emission of: `_wait`+`_syncThreads` (`:4905`–`:4906`), then for each `plrIdx`: **`localReadDo`** for A (`:4932`), for MXSA (`:4952`), for MXSB (`:4958`), for metadata (`:4964`), for B (`:4971`); pack-pre/pack code accumulated in `packPrePrefetchA/B`; `localReadInc` for each (`:4990`–`:5005`); then `_interleavePackAB` lays out the pack chain into `packPrePrefetchItems` and appends them to `module` (`:5008`–`:5012`). **This is the prefetch LR + the usePLRPack Pack chain.** | `numItersPLR>0` | the cleanest single hook |
| 15 | numItersPLR==0 + CMS branch (`elif`) | `:5019`–`:5049` | Half LRs + (optionally) half Packs emitted directly into `module`. | `numItersPLR==0 and UseCustomMainLoopSchedule` | hook |
| 16 | `closeSumAtLeastUnroll(prefetch=True, ...)` | `:5051` | label/end of prefetch block | `PGR` | n/a |
| 17 | `openLoop` | `:5067` | mainloop entry label — END OF PROLOGUE | always | n/a |

**Specifically locating the three buckets the bead asks about:**

- **Initial GR loads** — three sources, all emitted into `module`:
  - First round: `globalReadDo(kernel, 0, ...)` calls inside `setupNewTile`'s
    PGR branch at `KernelWriter.py:3003,3007,3011,3015` (called from `kernelBody`
    via `setupNewTile` at `:4739`).
  - Second-and-later rounds (PGR>=2): `globalReadDo(kernel, 0, ...)` at
    `:4843`/`:4850`/`:4857`/`:4864`.
  - These are the *only* GRs that emit before `openLoop`. The mainloop's
    `globalReadDo(kernel, 1, ...)` calls at `:3180,3194,3816,3838` go into
    `self.codes.globalReadA`/`B` (NOT into `module`) and are scheduled by SIA3 /
    CMS via the `_makeSubIterSchedule` path — they belong to the steady-state
    captures already covered by `FourPartCapture`.

- **Prefetch LRs** — `localReadDo` calls at `:4932,4952,4958,4964,4971` for A,
  MXSA, MXSB, metadata, B respectively, gated by `if self.states.numItersPLR:`
  at `:4902`. `localReadInc` calls at `:4990,4994,4998,5002,5005` advance the LR
  pointer between subiters.

- **usePLRPack Pack chain** — same prefetch-local block. `usePLRPack` resolves
  to `self.states.doFullPackCodePrefetch or (kernel["UseCustomMainLoopSchedule"]
  and kernel["UsePLRPack"])` at `:4910`. When set, the per-subiter `packCodeA/B`
  returned by `localReadDo` is added to `packPrePrefetchA/B` (`:4944,4983`)
  rather than to `pack[plrIdx]`; it gets interleaved at `:5008` and appended to
  `module` at `:5012`. The kernel-config gates are therefore
  `numItersPLR>0 AND (doFullPackCodePrefetch OR (UseCustomMainLoopSchedule AND
  UsePLRPack))`.

**Natural capture hook.** The cleanest single insertion point is *after* line
`:5051` (`closeSumAtLeastUnroll(prefetch=True)`) and *before* line `:5067`
(`openLoop`). At that point the prologue is wholly inside `module` (under the
`if kernel["PrefetchGlobalRead"]:` umbrella at `:4748`). The capture would walk
the items appended to `module` between two recorded length checkpoints — the
length right after `setupNewTile` returns at `:4739`, and the length right
before `openLoop` at `:5067` — exactly mirroring how `LoopBodyCaptureBuilder`
already collects per-subiter slices for the mainloop.

The categorization machinery already exists: `build_idmap` /
`invert_idmap_to_id_to_category` (used at `:4436` and `:4537`) take the same
`globalReadA`/`globalReadB`/`localWriteA`/`localWriteB` modules that feed the
prologue; the prefetch LR pack snapshots `_capture_context.prefetch_pack_a/b`
populated at `:4917`–`:4918`/`:4939`/`:4948`/`:4978`/`:4987` already preserve
leaf identity for the prefetch Pack chain. All the infrastructure to tag
prologue leaves by category exists; the missing piece is the
checkpoint/capture wrapper itself.

**GSU `setupNewTile` overrides (out of scope per bead).** For survey
completeness: GSU defines its own `setupNewTile` overrides at
`Tensile/Components/GSU.py:139` (base), `:251` ("GSU Off"), and `:636`
("GSU On"). These replace step (6) above when GSU is active and would emit a
GSU-shaped prologue. Per the bead, Stream-K / GSU / grouped-GEMM are out of
scope for Phase 1, but any future prologue-capture work that broadens beyond
the default codepath will need to handle these overrides explicitly.

**Non-PGR kernels.** When `kernel["PrefetchGlobalRead"]==0`, the entire
`if kernel["PrefetchGlobalRead"]:` block (`:4748`–`:5051`) is skipped. There
is no prefetch GR/LR/Pack at all — the first GR/LR/Pack happens inside
`_loopBody`/`_makeSubIterSchedule`. For PGR=0 kernels the prologue *is* empty
of dataflow-bearing instructions and the prologue capture would be empty.
Today the entire CMS path effectively requires PGR — `customMainLoopSchedule`
asserts `opt1.numMfma == len(mfmaCode)` and the schedule registry only ships
schedules for PGR>=1 (see `RegisterSchedule.prefetch_global_read` at
`CustomSchedule.py:695,883`). So in practice the prologue is non-empty for
every CMS-validated kernel.

---

## Q2: What shape would capture take?

### CMS-vs-default question first

Before deciding capture shape, the bead's load-bearing sub-question:
**does the CMS path even have a parallel prologue, or is the prologue identical
between CMS=0 and CMS=1?**

Verdict from the code: CMS does not directly emit prologue instructions, but
it **does shape prologue structure indirectly** via the `UsePLRPack`
kernel-config flag that ~14 CMS schedule functions set to `True`. Specifically:

- The prologue (Q1 walk above, `:4721`–`:5051`) sits at the `kernelBody` top
  level, **above** the `_loopBody` invocation that contains the
  `if kernel["UseCustomMainLoopSchedule"]:` branch. CMS does not redirect any
  prologue emission **at the call-site level**.
- `customMainLoopSchedule` (`Components/CustomSchedule.py:284`–`553`) is the
  whole CMS code surface. A `grep` over that file shows zero references to
  `preLoop`, `setupNewTile`, `prefetch` (other than the `prefetch_global_read`
  schedule-knob field), `globalReadDo`, or `localReadDo`. CMS only consumes
  `kernel["PrefetchGlobalRead"]`/`PrefetchLocalRead` as kernel-config knobs that
  shape the *mainloop* schedule recipe; it does not emit prologue instructions.
- The CMS branch at `:4486`–`:4587` runs *inside* `_loopBody` and only replaces
  the per-subiter scheduling of the mainloop itself.
- **However**, ~14 CMS schedule registrations in
  `Components/CustomSchedule.py` set `kernel["UsePLRPack"] = True` (lines
  `1449, 1493, 3494, 3559, 3692, 3888, 3974, 4168, 4288, 4375, 4535, 4718,
  4850, 4869`), in addition to the `RegisterSchedule` default
  `"UsePLRPack": False` at `:789`. The prologue's `usePLRPack` gate at
  `KernelWriter.py:4910` reads exactly this flag; when `True` the per-subiter
  `packCodeA/B` returned by prefetch `localReadDo` is appended to
  `packPrePrefetchA` (`KernelWriter.py:4944`) instead of `pack[plrIdx]`
  (`:4946`). That branch determines whether the Pack chain lives in the
  prologue at all. So CMS schedules **do** shape the prologue's pack-chain
  branch — just via kernel-config, not via direct emission.

This is structurally important: there is one prologue *emission path*, but its
internal structure (Pack chain in prologue vs in mainloop subiter) is shaped by
a CMS-set flag. The "divergence to detect" the bead postulates is therefore
**not hypothetical** — it already exists structurally between schedules that
flip `UsePLRPack` and the `RegisterSchedule` default. Two builds of the same
problem with different CMS schedules can produce structurally different
prologues today; what's missing is a validator that sees them.

The comparison shape question is consequently sharper, not collapsed: a CMS-vs-
CMS-shadow prologue capture with `UsePLRPack` held constant would be trivially
equal today (same `module.items()` slice, modulo SIA3's `structural_clone` at
`:4467`–`:4477`), but a CMS-vs-default capture (or two CMS captures with
`UsePLRPack` differing) would diverge by construction. Q4 develops this as the
load-bearing test fixture for Phase 3.

### So why capture it at all?

Three reasons the gap is still worth thinking about, in declining order of
importance:

1. **First-iteration cross-boundary edges.** The prologue's last
   `localReadDo`/Pack writes the registers that the mainloop's *first* MFMA
   reads. Today `compare_graphs` builds graphs whose earliest body is `ML-1`
   (`BODY_LABEL_ML_PREV` at `ScheduleCapture.py:348`). MFMAs in `ML-1` that
   read prefetch-Pack outputs have **no producer in the graph** — the validator
   silently treats them as having an external dataflow source. If a future CMS
   schedule (or a refactor of `_loopBody`'s capture-id-map building at
   `:4451`–`:4477`) staged a different LR/Pack ordering in the prologue, the
   first MFMA's input registers would change without `compare_graphs` flagging
   it.
2. **Forcing the constraint into the type system.** Today the validator's
   contract document says "equivalent kernel"; the implementation says
   "equivalent steady-state mainloop". Making the prologue first-class in the
   capture object closes that gap by construction — even if no current schedule
   diverges, future schedules cannot regress silently.
3. **Sharing infrastructure with the symbolic-register epic
   (`rocm-libraries-d0xd`).** The prologue is a stable place to test
   symbolic→numeric resolution: `setupNewTile` allocates a lot of vgprs in
   well-known patterns, so a prologue capture is the cleanest place to grow
   the register-resolution machinery if/when d0xd lands.

### Capture object shape

Given that (a) there's only one prologue per kernel — not one per codepath —
and (b) the prologue is shared between CMS=0 and CMS=1, the right shape is
**not** to add a fifth dict to `FourPartCapture`. The existing dicts
(`main_loop`, `main_loop_prev`, `n_gl`, `n_ll`) are keyed by codepath, and the
prologue has no codepath structure. Two clean options:

- **Option A — `FourPartCapture` gains a single `prologue: Optional[LoopBodyCapture]`
  field (not a dict).** Default `None` (kernel had no PGR or prologue capture
  off). Pros: single object, single comparison call,
  `build_dataflow_graph` already supports adding a body to its `captures`
  dict; the absent-key path at `CMSValidator.py:957` would extend cleanly to
  `BODY_LABEL_PROLOGUE`. Cons: muddies "FourPart" naming; renaming to
  `KernelCapture` carries a churn cost across the validator.

- **Option B — separate `PrologueCapture` object held alongside `FourPartCapture`
  on `CaptureContext` (e.g. `CaptureContext.default_prologue` /
  `CaptureContext.cms_prologue`).** Pros: zero churn on existing validator
  surface; tests for prologue can be added in isolation; the validator would
  call a new `compare_prologue(default_prologue, cms_prologue)` rather than
  reusing `compare_graphs`. Cons: cross-graph edges from prologue to first
  mainloop iteration become awkward — they span two graphs.

**Recommendation (subject to Phase 2 review):** Option A with a clear "prologue
is not codepath-keyed" comment. The cross-boundary-edge use case
(reason 1 above) is the primary motivator, and that needs prologue + main_loop
in the **same** `DataflowGraph`. Option B forfeits that.

### Granularity

- One body capture, **not** one per codepath. The prologue has no codepath
  structure today (`numCodePath` only varies the mainloop scheduling order,
  which is well downstream of prologue emission).
- The prologue body should be a single `LoopBodyCapture` keyed under a new
  `BODY_LABEL_PROLOGUE = "PROLOGUE"` with `BODY_LABEL_TO_LOOP_INDEX` value
  `-1` (so it sorts before `ML-1=0`).
- Categories already exist in `build_idmap`: `GRA`/`GRB` for the prologue's
  initial GRs, `LRA0`/`LRB0` for the first PLR LRs (`plrIdx=0`), `PackA0`/
  `PackB0` for usePLRPack outputs; `LWA`/`LWB` for `preLoopLocalWriteDo`. No
  new categories needed.

---

## Q3: Cross-boundary edges

The bead asks: should the prologue be a single concatenated graph with the
mainloop bodies, or two graphs with a documented boundary contract?

**Answer: single concatenated graph.** Justification from the existing code:

- `DataflowGraph` is *already* a single graph spanning all four bodies
  (`CMSValidator.py:443`–`469` docstring: "Single graph (not one per body) so
  cross-body edges (e.g. DTL+LdsBuf previous-iteration LR0 -> current GR) are
  represented natively as edges between nodes whose body_labels differ").
- The graph builder iterates `_BODY_BUILD_ORDER` (`build_dataflow_graph` at
  `:980`) and seeds `nodes_by_identity` across all bodies before edge
  formation in Phase 2. Adding a `BODY_LABEL_PROLOGUE` to that order
  list — at index `-1`, before `ML-1` — is a one-line conceptual change; the
  rest of the builder already supports cross-body resource resolution.
- Stream order is encoded in `SchedulePosition` (`(loop_index, vmfma_index,
  sub_index)`); placing the prologue at `loop_index=-1` makes prologue writes
  precede every `ML-1` read in resource resolution, which is what we want.

**Two-graph alternative is strictly worse.** A separate `PrologueGraph` would
require either:
- **A boundary contract** (e.g. "prologue's writes to `ValuA0/ValuB0` must
  appear in mainloop's read set at iter 0"), which is a documented invariant
  rather than a structural check — a regression to the current validator's
  weak contract.
- **A handoff-tagging scheme** that propagates "exit-state" registers from
  prologue graph into a synthetic node in the main graph. That's just
  re-implementing cross-body edges with an extra layer of indirection.

**`validate_edge_wait_coverage`** (`CMSValidator.py:2751`) already walks
"across bodies in execution order so cross-body queue state survives boundary
crossings" (line 1455 docstring). It's per-graph, not per-body, so it would
extend cleanly: the prologue's `s_waitcnt` at `:4762` would simply become
the first wait in the unified stream, and any prologue-to-mainloop edge
would be checked against waits from both bodies in stream order. No new
boundary-spanning helper is required.

**`compare_graphs`** would also extend cleanly. The identity-coverage gate at
`CMSValidator.py:2371` operates on `_DATA_FLOW_KINDS = ("LR", "LW", "GR",
"MFMA")` regardless of body; prologue LR/LW/GR identities would fall under
the same gate. The edge-key set comparison at `:2401` is already body-agnostic.

The only real implementation cost is the **builder/checkpoint plumbing** in
`kernelBody` (insert capture call between `:5051` and `:5067`), the new
`BODY_LABEL_PROLOGUE` constant, and the prologue-side variant of the
`build_idmap`/`invert_idmap_to_id_to_category` call (which has effectively
already been written twice at `:4436` and `:4537`).

---

## Q4: What real divergences would this catch?

### Survey of `Tensile/Components/CustomSchedule.py`

A grep for any prologue-touching call in the entire CMS file:

```
grep -n "preLoop\|prologue\|setupNewTile\|globalReadDo\|localReadDo" \
    Tensile/Components/CustomSchedule.py
```

returns only `prefetch_global_read` / `prefetch_local_read` as schedule-recipe
fields on `RegisterSchedule` (`CustomSchedule.py:695,696,764,765,883,884`).
**No CMS schedule directly emits any prologue instruction.** All ~50 schedule
functions (`_get_schedule_*` from `:908` to `:5681`) populate the
`optSchedule` / `optSyncSchedule` / `mfmaReorder` index lists that drive
`customMainLoopSchedule`'s macro emission — i.e., they describe the
steady-state mainloop only.

`customMainLoopSchedule` itself produces a single `Macro("MAINLOOP", ...)` plus
the `_pending_cms_capture_inputs` stash. The macro is invoked from the
mainloop dispatcher (via `simdSpecDispatch`) inside `_loopBody`; it does not
re-emit anything in the prologue.

**However**, as Q2 establishes, ~14 CMS schedules set
`kernel["UsePLRPack"] = True` (vs the `RegisterSchedule` default `False` at
`CustomSchedule.py:789`), and that flag is read by the prologue's pack-chain
gate at `KernelWriter.py:4910`. So the prologue's *structure* differs across
CMS schedules today — specifically, schedules that flip `UsePLRPack` move the
Pack chain into the prologue while schedules that leave it default keep the
Pack chain in the mainloop subiter.

**Conclusion:** no CMS schedule directly emits prologue instructions, but
existing CMS schedules already produce structurally different prologues via
`UsePLRPack`. The gap the bead describes is **structural and present today**,
not purely latent — what's missing is a validator that observes it. This still
matches the bead's "not a known active bug" framing (no kernel is *miscompiled*
because of this) and its `P2` priority, but the divergence-catching test
fixture in Q2's reasoning and Phase 2 Q4 below has a concrete, in-tree
schedule shape to exercise rather than a hypothetical future schedule.

### Concrete (in-tree) divergence: `UsePLRPack`

The primary in-tree divergence — and the one most worth a unit-test fixture
in Phase 3 — is the `UsePLRPack` flip described above. CMS lets
`usePLRPack=True` move the Pack chain into the prologue
(`packPrePrefetchA.add(packCodeA)` at `KernelWriter.py:4944`) where
`usePLRPack=False` would have appended to `pack[plrIdx]` instead (`:4946`).
Since `UsePLRPack` is a CMS-shaped kernel-config flag (see `:4910`), comparing
a CMS build (with `UsePLRPack=1`) against the same problem built default-style
(which would not enable `UsePLRPack`) would show **structurally different
prologues** by construction.

That's not a bug — it's the intent of `UsePLRPack` — but it does mean a
prologue-capture validator must compare **CMS-vs-CMS-shadow** with
`UsePLRPack` held constant, not CMS-vs-non-CMS. This matches today's
`_captureDefaultSchedule` flow (`:4493,4520`), which already runs the default
capture inside the same `kernelBody` invocation, so both captures see the same
`kernel["UsePLRPack"]` value.

### Hypothesized future divergence

A second, hypothetical divergence: a future CMS schedule could request a
different `usePLRPack` interleave ordering. Today the prologue's
`_interleavePackAB` at `KernelWriter.py:5008` lays out `packPrePrefetchA` then
`packPrePrefetchB` according to `_packItemsConditional` rules (`:676`–`706`).
If a future CMS schedule wanted to force B-then-A interleave at prologue time
(paralleling the A/B order swap available in the steady-state via
`switch_A_B_schedule` at `CustomSchedule.py:204`), the prologue's
first-iteration MFMA inputs would come from a different physical register
sequence than the default prologue produced. `compare_graphs` would NOT flag
this today — both captures' identity sets begin at `ML-1`'s MFMAs, and the
producers of those MFMAs' read registers are off-graph.

### Symbolic-register overlap (`rocm-libraries-d0xd`)

The prologue uses `vgprValuA0/vgprValuB0/vgprG2LA/vgprG2LB` and other
allocator-named vgprs heavily. As long as both captures observe the same
`writer.vgprPool` snapshot (which they do — both run inside the same
`kernelBody` call), the symbolic register names will resolve the same way on
both sides. The d0xd epic's symbolic-resolution work is **not** a prerequisite
for prologue capture; it is a parallel concern. Prologue capture would
*benefit* from d0xd if/when a future feature emits symbolic registers in CMS
that the default path numerizes (or vice versa), but no such case exists
today.

---

## Phase 2 decisions — RESOLVED

User decisions recorded 2026-05-11.

1. **Implement now → YES.** Worth implementing immediately. Two motivations:
   (a) catch current CMS divergences in the preloop (defense-in-depth);
   (b) the change is a precursor to leveraging the validator-on-arbitrary-
   timeline work — once the validator accepts arbitrary timelines, this
   capture lets us test different flags that impact the preloop (e.g. the
   conditions under which `usePLRPack` is initially introduced).

2. **Capture shape → Option A.** Extend `FourPartCapture` with
   `prologue: Optional[LoopBodyCapture]`. Don't worry about the
   `FourPartCapture` naming becoming a misnomer or the resulting code
   churn — everything is on a development branch. Take the change all
   the way; rename / clean up downstream as needed.

3. **Comparison shape → ONE graph for the whole thing.** Single
   concatenated graph (Q3's verdict). Don't preserve a prologue-vs-
   mainloop boundary in the graph structure; the prologue is just more
   nodes/edges in the same `DataflowGraph`.

4. **`UsePLRPack`-vs-non-`UsePLRPack` divergence-catching tests → YES,
   plus a parallel whole-kernel test.** Phase 3 builds two parallel
   tests:

   a. **Preloop-only divergence test.** Construct two captures from
      the same kernel module — one with `UsePLRPack=1`, one with
      `UsePLRPack=0`. When comparing JUST the preloop (or the
      preloop's contribution to the merged graph), `compare_graphs`
      MUST flag the prologue structural difference.

   b. **Whole-kernel `UsePLRPack` CMS test.** Construct a CMS kernel
      that uses `UsePLRPack`. Compare it against BOTH:
        - default with `UsePLRPack=True`, AND
        - default with `UsePLRPack=False`.
      BOTH comparisons MUST come out as PASSING. (The CMS schedule
      compensates for the flag's prologue effect during the mainloop;
      the whole-kernel comparison should be insensitive to the
      `UsePLRPack` setting on the default side.)

   These two tests together prove: the validator is structurally
   sensitive to prologue divergences (test a) AND CMS schedules can
   correctly absorb prologue-flag differences when the whole kernel
   is considered (test b).

   **4(b)' — supersedes 4(b) per the 2lzd shadow-decision (2026-05-12).**
   The original 4(b) test architecture relied on the shadow capture
   being the default-side reference (per `_captureDefaultSchedule` plus
   the inherited-prologue plumbing observed at
   `Tests/unit/test_prologue_capture.py:342`:
   `assert cap_with_cms.prologue is cap_with_default.prologue`). With
   `2LZD_INVESTIGATION.md §6` rejecting shadow-as-reference, that
   architecture is invalidated — there is no shadow-side prologue for
   the CMS side to inherit, and the inherited-prologue trick masked
   exactly the body-label-sensitivity issue documented in §"Phase 3
   blocker" §3 below.

   **Restated test intent in shadow-free terms.** Build the CMS kernel
   normally; build TWO real non-CMS kernels (one with `UsePLRPack=True`,
   one with `UsePLRPack=False`); compare CMS-vs-each. BOTH comparisons
   should pass IF the validator's cross-body comparison handles
   `UsePLRPack` pipelining correctly. Today's `compare_graphs` does
   NOT handle that (per §2 below — `loop_index` in the identity tuple
   makes pipelined producers fail to match across body boundaries),
   which is why oram.1's body_label-sensitivity blocker becomes
   critical-path for this restated test.

   The CMS-vs-non-CMS comparison shape is the live question under
   `2LZD_INVESTIGATION.md §6`'s Approach A; under Approach H both
   sides become real CMS builds (one synthesized via
   `cms_from_default`) and the body-shape match is by construction;
   under Approach D no comparison happens at all and 4(b)' becomes
   moot. Implementation choice is open. Cross-link:
   `2LZD_INVESTIGATION.md §6`, `rocm-libraries-oram.1`.

5. **Default for absent prologue → optional.** PGR=0 kernels emit no
   prologue; in those cases `prologue: Optional[LoopBodyCapture]` is
   `None`. Option A's typing already supports this. Update
   `BODY_LABEL_PROLOGUE` consumers to handle the `None` case cleanly —
   no changes to the absent-key skip path needed beyond what naturally
   falls out of the `Optional` shape.

### Phase 2 implementation directives (derived)

- **Capture surface**: `FourPartCapture.prologue: Optional[LoopBodyCapture]`
  (decision 2). Rename the dataclass if naming clarity warrants — code
  churn acceptable per decision 2.
- **Graph integration**: the prologue's nodes/edges land in the same
  `DataflowGraph` as the mainloop (decision 3). Implement by extending
  `build_dataflow_graph`'s body-walk to start at the prologue when
  present.
- **Default-side prologue capture site**: per Phase 1 §Q1, the prologue
  ends at a clear checkpoint in `kernelBody`. Plumb a
  `LoopBodyCaptureBuilder` for the prologue body labelled
  `BODY_LABEL_PROLOGUE` and finalize at that checkpoint.
- **Test fixtures (decision 4)**:
  - `test_preloop_divergence_catches_useplrpack_change` (or similar):
    same kernel module, two captures (UsePLRPack=1 vs =0), assert
    `compare_graphs` reports prologue structural diff.
  - `test_whole_kernel_useplrpack_cms_matches_both_defaults` (or
    similar): one CMS kernel using `UsePLRPack`, compared against
    `UsePLRPack=True` default and `UsePLRPack=False` default; assert
    both comparisons pass.
- **None-handling (decision 5)**: `Optional` semantics; PGR=0 kernels
  have `capture.prologue is None`; downstream consumers branch
  cleanly on the None case (no `BODY_LABEL_PROLOGUE` skip-path
  changes needed).

### Sequencing

- This bead is **independent** — does not block on anything currently
  in flight.
- Soft-aligned with the broader 5gd Timeline-generalization work (3g4
  scaffold + xe5 bridge already in vlt). Once the validator accepts
  arbitrary timelines, the prologue-aware capture is one of the
  flag-toggle test scenarios that timeline shape enables (per decision 1's
  motivation b).

---

## Phase 3 blocker: `compare_graphs` identity is body-label-sensitive

Investigator: `preloop-compare-investigation` (Claude Opus 4.7, May 2026).
Working tree: `/home/alvasile/rocm-libraries/.worktrees/preloop-compare-investigation/`.
Worktree fork point: vlt at `0be21515067b` (rebased onto vlt's o0ei
landing; original investigator commit was off `83fd43ce4fa1`, no
interaction with o0ei's `QUAD_CYCLE_DISPATCH_AUDIT.md` rewrite).
Sub-bead tracking the open question: see §"Sub-bead reference" at end.

This section was added after the dispatching agent (me) misframed Decision
4(b) as "only meaningful under contract (b)" of bead `rocm-libraries-2lzd`.
The user pushed back: `UsePLRPack` is **pipelining-only** at the
symbolic-register level — first iteration's packs land in the prologue,
iteration N's packs land at the END of iteration N-1, total dataflow
unchanged. Under that model 4(b)'s test SHOULD be meaningful regardless
of the contract chosen for 2lzd. The structural reason it would still
fail today is the body-label-sensitivity of `compare_graphs`'s identity
construction. This section documents both halves: the verified
pipelining model, and the body-label-sensitivity blocker.

### §1 — `UsePLRPack` is pipelining (verified)

Walked `KernelWriter.py` end-to-end. Cited line numbers below are tip
`83fd43ce4fa1` (worktree vlt fork-point). Per-line mismatches with
earlier sections of THIS memo — earlier §Q1 cited `:4910` for the
`usePLRPack = ...` gate, current line is `:4961`; earlier §Q1 cited
`:4944, 4983` for `packPrePrefetchA/B.add(packCodeA/B)`, current lines
are `:5009, 5048`. The shapes are unchanged; new prologue-snapshot
plumbing inserted upstream shifted the line numbers by ~50.

#### 1.1 The branching gate

`KernelWriter.py:4961`:
```python
usePLRPack = self.states.doFullPackCodePrefetch or
             (kernel["UseCustomMainLoopSchedule"] and kernel["UsePLRPack"])
```

`doFullPackCodePrefetch` is set ONLY at `:8023`:
```python
self.states.doFullPackCodePrefetch = kernel["UsePLRPack"] and
                                     not kernel["UseCustomMainLoopSchedule"]
```

So the two routes that lead to `usePLRPack=True` at `:4961` are:
- **Default + `UsePLRPack=True`** (via `doFullPackCodePrefetch=True`).
- **CMS + `UsePLRPack=True`** (via the `or` clause).

When `usePLRPack=True`, the per-subiter `packCodeA/B` returned by
prefetch `localReadDo` is appended to `packPrePrefetchA/B`
(`:5009, 5048`); when `usePLRPack=False`, it lands in `pack[plrIdx]`
(`:5011, 5050`).

#### 1.2 Prologue-side emission (when `usePLRPack=True`)

`packPrePrefetchA/B` populated in the per-`plrIdx` loop
(`:4985-5070`), then interleaved at `:5073`
(`self._interleavePackAB(...)`), then **appended directly to `module`**
at `:5088` via `module.addItems(packPrePrefetchItems)`. Result: the
pack chain physically lands in the kernel BEFORE `openLoop`
(`:5159`) — i.e., in the prologue.

#### 1.3 Mainloop-side emission (when `usePLRPack=False`)

`pack[plrIdx]` modules are passed into `_loopBody` and consumed by
`_makeSubIterSchedule` for every steady-state iteration. The
producer-consumer rotation is at `KernelWriter.py:3289-3301`:
```python
plrIdx     = (u + pflr) % self.states.numVgprBuffer
packPreIdx = (u + pflr) % self.states.numPackBuffer  # store/preread
packIdx    = u            % self.states.numPackBuffer  # read
packStoreIdx = (uNext + pflr) % self.states.numPackBuffer
```

**Producer-consumer offset of `pflr` = "pack stores N iterations
ahead of where MFMA reads".** In a steady-state iteration `u`, the
iteration emits LRs/Packs that target `pack[packStoreIdx]` (where
`packStoreIdx` is offset by `pflr` from `u`), while the MFMA reads
`pack[packIdx]` (using `u` directly, no offset). Modulo `numPackBuffer`
arithmetic, that's a `pflr`-iteration pipeline.

#### 1.4 First-iteration boundary

Iteration `u=0` reads `pack[0]` for its MFMA. If `usePLRPack=True`,
`pack[0]` is empty in the mainloop — the producer ran in the
prologue's `packPrePrefetchA/B` chain (§1.2). If `usePLRPack=False`,
the prologue still runs the prefetch `localReadDo` to fill `pack[0]`
(`:5011`), but the pack code lives in the mainloop's per-iter pack
module rather than appended to `module`. Either way, **iteration 0
of the mainloop reads pack contents that were physically produced
before `openLoop`** — the prologue's pack chain (in the
`usePLRPack=True` route) or the prologue's `pack[0]` accumulation
that gets consumed once mainloop iter 0 dispatches it (in the
`usePLRPack=False` route).

#### 1.5 Last-iteration boundary

For `usePLRPack=True`: the loopBody walk at `:3289` uses
`packStoreIdx = (uNext + pflr) % numPackBuffer`, so the last
mainloop iteration's pack writes wrap around to a `pack[]` slot
that the next-loop iteration would consume. In `noLoadLoop` /
`closeSumAtLeastUnroll` (`:3659-3791, 5127`), the tail iterations
consume those wrapped packs as their MFMA inputs. No pack
producer is duplicated; no pack producer goes unconsumed. The
volume invariants are:
- **Per-iteration pack count is invariant** under the
  `usePLRPack` flag flip (CMS-side pack count per mainloop
  iteration matches default-side per-iteration pack count;
  the difference is WHICH iteration produces them).
- **Total prologue + mainloop pack count is invariant** modulo
  the boundary trick: the prologue pack chain in
  `usePLRPack=True` displaces what would have been
  iteration-0's mainloop packs in `usePLRPack=False`.

#### 1.6 Empirical confirmation

Empirical evidence the user supplied earlier (from `kernel_cms.s`
vs `kernel_default.s` at the same problem):
- `v_cvt_pk_bf16_f32`: 128 (cms) vs 320 (default), ratio 2.5×.
- `v_mfma_f32_4x4x4_16b_bf16`: 32 vs 80, same 2.5× ratio.
- `v_mfma_f32_16x16x32_bf16`: 96 vs 240, same 2.5× ratio.

Uniform 2.5× across all op classes, NOT just packs. The
`kernel_default.s` path emits the mainloop inline (no
`.macro MAINLOOP`), so its iteration count is multiplied by ~2.5×
relative to `kernel_cms.s` which emits a `.macro MAINLOOP` (line
~1846 of `kernel_cms.s`) and invokes it 2.5× from the unrolled
caller. Per-iteration pack count is invariant; the 2.5× ratio is
just the macro-vs-inline expansion factor.

The user also verified at `kernel_cms.s:1881` (wide MFMA at
`mfmaIndex:3` reads `vgprValuA_X0_I0+8` from a prologue
prefetch, then immediately following packs write
`vgprValuA_X0_I0+16/17` for the NEXT iteration). Pure pipelining;
no duplication, no missing producer.

#### 1.7 Conclusion of §1

**The user's pipelining model is correct.** Under contract (a) of
2lzd ("CMS preserves what a real non-CMS Tensile build would emit
for this problem"), the dataflow IS equivalent regardless of
`UsePLRPack` value. Decision 4(b)'s test should be meaningful
under contract (a). The dispatching agent's earlier dismissal
("only meaningful under contract (b)") was wrong.

### §2 — Body-label-sensitivity in `compare_graphs` (the actual blocker)

The reason Decision 4(b)'s test would fail today even though the
dataflow IS equivalent is that `compare_graphs` builds identity
tuples that bake `body_label` into the identity. Two graphs with
the same canonical pack instruction emitted from different bodies
will not match.

#### 2.1 Where body_label enters identity

`CMSValidator.py:1594` (in `_make_node`):
```python
identity = tagged_inst.identity_for(body_label)
```

`ScheduleCapture.py:507-536` (the `identity_for` method):
```python
def identity_for(self, body_label: str) -> tuple:
    ...
    loop_idx = BODY_LABEL_TO_LOOP_INDEX[body_label]
    cls_tag = WrappedInstruction.class_tag_for_category(self.category, inst)
    return (cls_tag, loop_idx, WrappedInstruction.canonical_str(inst))
```

`BODY_LABEL_TO_LOOP_INDEX` (`ScheduleCapture.py:702-708`):
```python
BODY_LABEL_TO_LOOP_INDEX = {
    BODY_LABEL_PROLOGUE: -1,
    BODY_LABEL_ML_PREV:   0,
    BODY_LABEL_ML:        1,
    BODY_LABEL_NGL:       2,
    BODY_LABEL_NLL:       3,
}
```

The identity tuple is `(class_tag, loop_index, canonical_render)`.
Same `class_tag` + same `canonical_render` in different bodies
produces a **different identity**.

#### 2.2 Where body-label sensitivity propagates into edge keys

`CMSValidator.py:1238` (`DataflowGraph.edge_keys`):
```python
return {(e.producer.identity[0], e.producer.position, e.src_operand_slot,
         e.consumer.identity[0], e.consumer.position, e.sink_operand_slot,
         e.edge_kind, e.intra_operand_byte_offset)
        for e in self.edges}
```

Here `e.producer.position` is a `SchedulePosition` carrying
`(loop_index, stream_index)` (`ScheduleCapture.py:713-741`). So even
though edge_keys uses `identity[0]` (just `class_tag`, dropping
`loop_index` from identity), it ALSO carries `producer.position`
which encodes `loop_index` directly. **Body sensitivity is in BOTH
the identity-coverage gate and the edge-key tuple.** Two source
mechanisms, same effect: a pack edge from PRO-body to ML-1-body
cannot match an edge from ML-1-body to ML-1-body even when their
canonical render and class_tag agree.

#### 2.3 The data-flow node identity-set gate

`CMSValidator.py:3306-3336` (the entry-time check):
```python
ref_ids = _data_flow_ids(reference)
subj_ids = _data_flow_ids(subject)
if ref_ids != subj_ids:
    ...
    raise CaptureConsistencyError(
        "compare_graphs: data-flow node identity sets differ. " ...)
```

Here `_data_flow_ids(graph)` returns `{k for k in graph.nodes.keys()
if k and k[0] in ("LR", "LW", "GR", "MFMA")}`. Pack identities (class
tag `"PACK"`) are NOT in this set, so **the gate does not fire on
pack-identity body-label drift** — it fires on LR/LW/GR/MFMA drift.
This is what the existing `test_preloop_divergence_catches_useplrpack_change`
test (`Tensile/Tests/unit/test_prologue_capture.py:165-274`) leans
on: comparing two graphs where one has prologue PACK producers and
the other doesn't, the gate doesn't catch it directly — instead
`diagnose_missing_edge` Phase 0 raises when a missing edge points
at a PACK producer absent from the subject.

The blocker for Decision 4(b) is downstream: if both sides have the
same set of pack producers but in different bodies (PRO on the
CMS+UsePLRPack=True side, ML or ML-1 on the default+UsePLRPack=False
side), the edge_keys differ at `producer.position.loop_index`, and
`compare_graphs` returns "missing edge" failures. Phase 1's
order-check (`:3434`) only runs `if p_node.body_label ==
c_node.body_label` and skips cross-body legitimate-reorder.
`diagnose_missing_edge`'s legitimate-CMS-reorder branch
(`:3417-3424`) requires identical `(producer.identity, consumer.identity)`
on both sides — which fails when `loop_index` differs because
identity carries `loop_index` (§2.1).

#### 2.4 Concrete failure shape

Let "P" = the canonical pack instruction `v_cvt_pk_bf16_f32 ...
v[vgprValuA_X0_I0+0:...], ...`. Let "M" = the consuming
`v_mfma_f32_16x16x32_bf16 ... v[vgprValuA_X0_I0+0:...] ...`.

- CMS+UsePLRPack=True side: P is in PRO body (loop_index=-1). M
  is in ML-1 body (loop_index=0). Identity for P is
  `("PACK", -1, "<P-canonical>")`. Edge P → M has
  `producer.position = (-1, K_pro)`,
  `consumer.position = (0, K_mlp)`.
- Default+UsePLRPack=False side: P is in ML or ML-1 body
  (loop_index=0 or 1, depending on rotation). M is in same body
  as before. Identity for P is `("PACK", 0, "<P-canonical>")` or
  `("PACK", 1, ...)`. Edge has `producer.position = (0, K_ml)`
  or `(1, K_ml)`, **different from the CMS side**.

Per §2.2, the edge_keys differ on `producer.position`. Per §2.3,
the data-flow identity-set gate doesn't catch it (PACK is not in
the gated kinds). Per `compare_graphs:3340`, the missing-edge
diagnosis fires. Per `diagnose_missing_edge:3417`, the
legitimate-CMS-reorder branch requires identity equality on both
endpoints — fails because `loop_index` is in identity (§2.1). The
result: spurious `OrderInvertedFailure` or unexplained-missing-edge
raises.

This is what the user means by "compare_graphs would see the
dataflow-equivalent edges as identity-set differences and fire
spurious failures." The identity-set gate doesn't fire directly
(PACK is not data-flow), but the downstream missing-edge classifier
does, and the cross-body reorder branch can't absorb the false
positive because identity itself encodes `loop_index`.

### §3 — Why the existing UsePLRPack tests pass anyway

`test_whole_kernel_useplrpack_cms_matches_both_defaults`
(`Tensile/Tests/unit/test_prologue_capture.py:277-350`) currently
passes. It does not catch the body-label-sensitivity blocker
because of an artifact of the test setup:

The test asserts at `:342`:
```python
assert cap_with_cms.prologue is cap_with_default.prologue
```

The CMS-side capture **inherits the default-side prologue verbatim**
via `build_cms_four_part_capture(prologue=default_capture.prologue,
...)` (cited as Phase 2 decision 3 implementation). So the two
captures being compared have IDENTICAL prologue bodies — there is
no body-label drift to detect, because the prologue is shared by
construction. The test is really pinning that the prologue
propagates symmetrically AND that the mainloop passes
`compare_graphs` independently.

When the test setup is changed to have the CMS side and default
side derive prologues independently — which is what would be
needed to prove decision 4(b)'s "BOTH MUST PASS" claim under
contract (a) — the body-label-sensitivity issue surfaces.
Specifically: build CMS+UsePLRPack=True with its OWN prologue
capture, and compare against default+UsePLRPack=False with its
OWN prologue capture. Today's `compare_graphs` would fail on the
body-label-mismatched pack edges per §2.4.

### §4 — Approaches to fix (sketched; full catalog in sub-bead)

The sub-bead filed alongside this memo will develop a full
approach catalog. Five distinct directions are sketched here so
the user can react to the framing before the sub-bead's
investigation lands. **Names below are short, memorable labels;
mechanism + cost + risk capsules.**

#### Approach A — "Drop body_label from identity"

Mechanism: change `TaggedInstruction.identity_for` to return
`(class_tag, canonical_render)` (drop `loop_idx`). Body
information stays on `GraphNode.body_label` and on
`SchedulePosition.loop_index`; identity becomes
allocation-and-body-invariant.

What it does to identity: drops the body element. Two pack
instructions with the same canonical text in different bodies
collapse to one identity tuple.

Cost: ~5 LoC change in `ScheduleCapture.py`. Downstream: the
identity-coverage gate at `:3306` becomes weaker (identity
collisions across bodies are now tolerated); the edge-key tuple
at `:1238` still carries `producer.position` (which has
`loop_index`), so cross-body register reuse remains
distinguishable in edge comparison.

Risks: cross-body identity collisions — if two pack instructions
in different bodies legitimately produce different downstream
behavior (e.g., one feeds ML iter 0, one feeds NLL), the
identity-coverage gate would no longer catch the case where one
side dropped one of them. Phase 1 order check (`:3434`) becomes
ambiguous when `p_node.body_label != c_node.body_label` is the
ONLY discriminator. Probably safe — the order check already
skips cross-body — but a full audit is needed.

#### Approach B — "Body-label-equivalence layer"

Mechanism: keep `identity_for` as-is, but add a normalizer that
maps `{PROLOGUE, ML-1, ML, NGL, NLL}` into "iteration buckets"
(e.g. `{PROLOGUE+ML-1+ML} -> "STEADY", {NGL+NLL} -> "TAIL"`)
before identity construction. Or: only collapse PRO into ML-1
(both at the "first-iteration" boundary).

What it does to identity: remaps `loop_idx`. PRO and ML-1 share
the same `loop_idx` in the comparison, so a pack that landed in
PRO on one side and ML-1 on the other still matches.

Cost: ~20 LoC. Adds a `comparison_loop_idx_for(body_label)`
helper in `ScheduleCapture.py` and wires it into both
`identity_for` and `make_position`. Tests for
`build_dataflow_graph` need updating to assert the new bucket
behavior.

Risks: arbitrary bucket boundaries. The "where does the iteration
boundary live" question is where the design has to be principled
— if PRO+ML-1 collapse but ML doesn't, then a pack that drifted
from ML-1 to ML still fails comparison. Likely insufficient for
the full pipelining model where producers can drift across
arbitrary iteration boundaries.

#### Approach C — "Cross-body pipeline-aware comparison"

Mechanism: leave identity as-is. Add a new comparison phase to
`compare_graphs` (or a new `diagnose_missing_edge` branch) that
recognizes cross-body pipelined drift: for a missing edge
P → M in reference, search the subject for an edge P' → M
where P' has the same `(class_tag, canonical_render_modulo_register_rotation)`
as P but a different body. If found, treat as a legitimate
pipeline.

What it does to identity: keeps it. Adds a pipeline-aware
fallback to the missing-edge classifier.

Cost: ~80-150 LoC. The fallback search is a new branch in
`diagnose_missing_edge` (`:3361+`). Needs a "register rotation"
canonicalizer (since iteration N+1's pack writes vgpr+8 where
iteration N's wrote vgpr+0).

Risks: register-rotation canonicalization is itself non-trivial
and may have its own false-positive surface. The classifier
becomes more complex and harder to reason about.

#### Approach D — "Symbolic-register rotation canonicalization"

Mechanism: rotate the operand vgpr indices in the canonical
render text BEFORE identity computation. iteration N+1's pack
writing `+16` and iteration N+1's iter-bumped pack writing
`+0` look the same after rotation. Effectively: bake the
pipelining into the canonical renderer.

What it does to identity: keeps the body element but changes
canonical_render so that pipeline-rotated insts produce
identical renders.

Cost: ~50-100 LoC. Touches `WrappedInstruction.canonical_str`
in `ScheduleCapture.py` plus needs an iteration-context input
(which iteration is this instruction part of?) which may not be
available at canonical-str time.

Risks: deep entanglement with the symbolic-register epic
(`rocm-libraries-d0xd`). May require `d0xd`-style numerization
work as a prerequisite. Iteration-context is typically known
by the body but not by the bare instruction; threading it
through requires API surface changes.

#### Approach E — "Compare at edge-level with structural identity"

Mechanism: switch `edge_keys` from
`(producer.identity[0], producer.position, ...)` to
`(producer-resource-write-key, consumer-resource-read-key,
edge_kind)`. The producer-resource-write-key is the byte-key of
the register the producer writes; the consumer-resource-read-key
is the byte-key of what the consumer reads. Body becomes
irrelevant because the comparison is on what-resource-flowed-where,
not on identity-of-instruction-that-emitted-it.

What it does to identity: keeps identity for diagnosis but
DROPS it from edge-key matching. The matching layer becomes
purely structural.

Cost: ~150-300 LoC. Major refactor of `compare_graphs` and
`diagnose_missing_edge`. The "missing edge" diagnosis then
needs a different reverse-lookup path (find the producer by
its write-key + position, not by identity).

Risks: large scope. The structural reverse-lookup may have
false positives when two distinct producers happen to write
the same byte-key in different bodies (which is exactly what
happens during register rotation — but here the false positive
is a feature, not a bug). However, the resource-key
construction is allocation-sensitive; if symbolic-vs-numeric
naming differs across captures, the keys might not match. Needs
the d0xd resolution layer.

#### Hybrid candidates

- **A + C**: drop body from identity; add a cross-body pipeline-
  aware fallback for the few cases where dropping body causes
  identity collisions to lose information.
- **B + E**: bucket loop indices for identity AND switch
  edge_key matching to resource-key based.
- **D + E**: combine canonicalization at render time with
  resource-key matching; the most thorough fix and likely the
  most expensive.

**Approach H — `cms_from_default`-driven same-direction synthesis
(borrowed from 2lzd memo, `2LZD_INVESTIGATION.md:593`).** Bypasses
the body-label issue entirely: build BOTH kernels from the same
synthesized CMS schedule (one real CMS, one default-converted-to-CMS
via `Tensile/Components/CustomSchedule/cms_from_default.py`,
`default_schedule_to_cms` API). Both sides have identical body shape
and identical pipelining choices, so loop_index-in-identity never
becomes a source of false positives. **Cost:** zero changes to
`compare_graphs`; only the comparison harness shifts. **Risk:** changes
what's being validated — "two CMS-emit paths agree" rather than
"CMS-emit matches default Tensile." Coupled to 2lzd's contract
decision; if user picks contract (a), H is the cheapest joint
resolution for both 2lzd and oram.1 in one move.

The sub-bead's full catalog will rank these against (1) the
existing `test_preloop_divergence_catches_useplrpack_change`
test (which depends on Pack-identity-set divergence being
caught by `diagnose_missing_edge` Phase 0), (2) the
cross-body legitimate-reorder branch
(`diagnose_missing_edge:3417`), and (3) the `validate_edge_wait_coverage`
walk (`:2751`) which is per-graph, not per-body, and unchanged
by any of these approaches.

### §5 — Why this blocks Decision 4(b) under contract (a)

Decision 4(b) states (excerpt from §"Phase 2 decisions —
RESOLVED"):

> Whole-kernel `UsePLRPack` CMS test. Construct a CMS kernel
> that uses `UsePLRPack`. Compare it against BOTH:
>   - default with `UsePLRPack=True`, AND
>   - default with `UsePLRPack=False`.
> BOTH comparisons MUST come out as PASSING.

Under contract (a) of `rocm-libraries-2lzd` ("CMS preserves what
a real non-CMS Tensile build would emit for this problem"), the
default-with-`UsePLRPack=False` reference is the canonical one —
the version that matches `kernel_default.s` emission. The CMS
build IS the real CMS emission. Both should match because
`UsePLRPack` is pipelining-only (§1), but today's
`compare_graphs` would fail because:

- The CMS+UsePLRPack=True kernel emits its prologue Pack chain
  (per §1.2) into the PRO body.
- The default+UsePLRPack=False kernel emits no prologue Pack
  chain; instead the equivalent pack producers run in ML-1 / ML
  bodies during the mainloop (per §1.3).
- The two captures' edge_keys differ at the
  `producer.position.loop_index` field (§2.4), and the
  legitimate-reorder fallback can't rescue the comparison
  because identity itself carries `loop_index` (§2.1).

The result: a real, dataflow-equivalent kernel pair fails
`compare_graphs` for purely structural-comparator reasons.
Decision 4(b)'s "BOTH MUST PASS" semantic is unimplementable on
top of today's body-label-sensitive comparator. **Phase 3 must
either fix `compare_graphs` (one of §4's approaches) or weaken
4(b)'s test to compare CMS+UsePLRPack=True against
default+UsePLRPack=True only** (the path the existing
`test_whole_kernel_useplrpack_cms_matches_both_defaults` test
takes today, by inheriting the prologue across both sides per
§3).

The user's choice: the validator's "equivalent kernel" contract
is stronger than the comparator's "equivalent identity-keyed
graph" implementation. The body-label-sensitivity is the
implementation gap. Closing it is a Phase 3 prerequisite for
the prologue-capture work to be load-bearing on real
CMS-vs-default validation.

### §6 — Interaction with `rocm-libraries-2lzd`

**UPDATED 2026-05-12 (2lzd decision).** 2lzd has decided the
shadow capture is rejected as the validator's reference (see
`2LZD_INVESTIGATION.md §6`: "We should not be comparing against
the shadow. The shadow is a kernel that cannot be emitted and
does not get run."). The joint-resolution discussion that
previously enumerated contracts (a), (b), and (c) for 2lzd
collapses: contract (b)'s "hold `UsePLRPack` constant on both
sides via shadow inheritance" is no longer viable because the
shadow itself is going away. The live 2lzd direction is one of
{Approach A, Approach D, Approach H} from `2LZD_INVESTIGATION.md
§3` — implementation choice still open as of 2026-05-12.

**What this means for the body-label-sensitivity blocker:**

- Under 2lzd Approach A (real non-CMS reference build), the
  body-label-sensitivity becomes the **immediate** false-positive
  source on every cross-build comparison. Pipelined producers will
  drift across body boundaries (PRO ↔ ML iter i ↔ ML iter i+1)
  under `UsePLRPack` flips; today's identity construction at
  `ScheduleCapture.py:507-536` bakes `loop_index` into identity, so
  drifted producers fail to match. **Fixing oram.1 is a
  prerequisite for A.**
- Under 2lzd Approach D (drop comparison entirely; validate
  CMS-emit against the schedule's slot table), oram.1 becomes
  **moot** — no cross-graph comparison happens, so identity
  construction never runs across two captures. D is the only
  approach that obsoletes oram.1 outright.
- Under 2lzd Approach H (`cms_from_default`-driven synthesis;
  both sides are real CMS builds), the body-label issue
  **sidesteps itself by construction**. Both sides emit through
  the CMS macro path, so the prologue routing and body shape
  match. Producers don't drift across body boundaries because
  there are no flag-driven prologue-routing differences between
  the two captures. H is the cheapest joint resolution that
  doesn't require fixing oram.1.

The cross-body-tagging design (cause-(B)/(B-2) in 2lzd's memo)
is now historical context only — those artifacts were specific to
the shadow's idmap walks and disappear with the shadow. The
sub-bead's catalog in §4 above remains relevant as the
implementation surface for any A-style or H-style fix that
encounters body-shape mismatches between two real captures.

### §7 — Updated context post-2lzd-decision (2026-05-12)

The 2lzd shadow rejection (`2LZD_INVESTIGATION.md §6`) elevates
the body-label-sensitivity blocker from a downstream concern
("this would matter if Decision 4(b) were to be exercised under
contract (a)") to a **critical-path prerequisite** for any
cross-build comparison the validator might still want to perform.
Specifically:

- Under any of the live 2lzd approaches that require comparing
  two real captures (Approach A, Approach H), the
  body-label-sensitivity in `compare_graphs` (this memo §2) is
  the proximate cause of false positives that today's existing
  shadow-shared-prologue trick (`Tests/unit/test_prologue_capture.py:342`,
  `cap_with_cms.prologue is cap_with_default.prologue`) was
  masking. Removing the shadow removes the mask.
- **Approach H is especially attractive post-decision.** Both
  sides are CMS builds with identical body shape, so the
  body-label issue does not surface — the validator never has to
  match a producer in PRO body against a producer in ML iter i body.
  This sidesteps the §4 catalog of approaches (A through E) for
  oram.1 entirely: H implementations need zero changes to
  `compare_graphs`. If 2lzd lands H, oram.1 may close as
  "obsoleted by H" rather than requiring its own implementation
  cycle.
- **Approach D obsoletes oram.1 outright.** No comparison
  happens, so identity construction across two captures is moot.
  Same close-as-obsoleted disposition.
- **Approach A is the only live 2lzd direction that requires
  oram.1 to be solved on its own merits.** A's reference is a
  real non-CMS kernel build, which has a structurally different
  prologue from the CMS kernel by construction (the
  `UsePLRPack`-driven prologue Pack chain present in the CMS
  kernel and absent in the default). Without an oram.1 fix, A
  cannot land.

The live 2lzd approach set is {A, D, H}. D would obsolete oram.1
entirely (no comparison happens, no body_label problem). H would
sidestep it (identical body shapes on both sides). A would
require oram.1 to be fixed first. **The user's choice on 2lzd
implementation determines whether oram.1's §4 approach catalog
needs to be implemented or can be abandoned.**

### Sub-bead reference

A child bead has been filed under `rocm-libraries-oram` to
track the resolution of this body-label-sensitivity issue and
the catalog of approaches in §4. See bead listed in the commit
that added this section. Cross-link comment also added to
`rocm-libraries-2lzd` noting the coupling described in §6.

---

## Hard rules compliance

- Phase 1 is research only. **No code changes.** This memo is the only
  deliverable.
- Reference template `c1054a019c` (dj1g) consulted: the deferred-expansion
  pattern (default-side capture is the single source of truth, CMS-side
  expansion mirrors its body shape) would extend naturally to prologue —
  default-side prologue capture would be built in `kernelBody` at the
  prologue-end checkpoint, then any future "CMS-side prologue rewriter"
  (currently nonexistent) would expand a stashed prologue macro driven by
  the default-side prologue's body shape, just as
  `build_cms_four_part_capture` consumes `default_capture` today.
- d0xd overlap: noted in Q4. Not a prerequisite.
- Bead not closed; nothing pushed; nothing squashed.
