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

## Phase 2 questions to stage

Per the bead's Phase 2 directive, these are the questions to ask the user
before any implementation:

1. **Worth closing now?** Q4 confirms the gap is latent — no current CMS
   schedule diverges. Is it worth implementing now (defense-in-depth, makes
   the validator's stated contract true by construction), or document and
   defer (P2 stays open as a watching brief, re-evaluate when next CMS
   schedule lands that touches PLR/Pack)?
2. **Capture shape:** Option A (extend `FourPartCapture` with
   `prologue: Optional[LoopBodyCapture]`, accept `FourPartCapture` naming
   becomes a misnomer) or Option B (separate `PrologueCapture` object on
   `CaptureContext`, lose cross-graph edges)? Recommendation above is A; user
   sign-off needed.
3. **Comparison shape:** Confirm single concatenated graph (Q3 verdict). Is
   there any reason to prefer the boundary-contract approach (e.g. plans for
   the validator to compare prologue-only across kernel variants)?
4. **`UsePLRPack`-vs-non-`UsePLRPack` as an explicit divergence-catching
   shape:** should Phase 3 build a unit-test fixture that constructs *two*
   captures from the same kernel module — one with `UsePLRPack=1`, one with
   `UsePLRPack=0` — and asserts that `compare_graphs` flags the prologue
   structural difference? This would prove the validator is sensitive to
   prologue divergences, separate from any production schedule actually
   exhibiting one.
5. **Default for absent prologue:** PGR=0 kernels emit no prologue dataflow.
   Should `prologue` be `None` (matches Option A's `Optional` type), or
   `LoopBodyCapture(instructions=[])` (would trip the empty-body guard at
   `:961` — would need to add `BODY_LABEL_PROLOGUE` to the absent-key skip
   path at `:957`)?

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
