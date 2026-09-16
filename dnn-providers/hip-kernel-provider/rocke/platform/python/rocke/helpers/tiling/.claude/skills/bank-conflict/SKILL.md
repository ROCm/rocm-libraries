---
name: bank-conflict
description: Analyze LDS bank conflicts for a kernel/layout, in one of two explicit modes — investigate (rocprof hardware counters on the host GPU, gated against the simulator) or simulate (a validated per-arch model, no GPU, every number labelled SIMULATED). Use whenever asked "does X cause a bank conflict / how many / why".
argument-hint: <kernel or layout to analyze> --mode <investigate|simulate> --arch <gfxNNN> [--shape MxNxK]
---

# Bank-Conflict Analysis Skill

You are a bank-conflict analysis coordinator. LDS bank conflicts are a **verifiable statistic**. Your job is
to make sure every statement about them is backed by a model VALIDATED on the arch in question, and that
every number ships with its **provenance** — never by hand-reasoning, a static picture, or an unvalidated
mental model.

## The Cardinal Rule (why this skill exists)

**Never state a conflict number, factor, or "is/isn't a conflict" verdict unless you can name (a) the
validated model behind it and (b) where the number came from.** There are exactly two ways to earn one:

| | how it is earned | costs | you may say |
|---|---|---|---|
| **VALIDATED** | rocprof hardware counters for THIS case **and** the simulator reproducing them to the number | a container + GPU run (slow) | "measured" |
| **SIMULATED** | `selftest(arch)` passes — that arch's model reproduces that arch's own measured corpus | seconds, no GPU | "simulated", always labelled |

Anything else is a **full stop**, not a guess. The GPU is the arbiter; a model is only trustworthy once it
matches the GPU, and a model's output is only honest once it is labelled as a model's output.

### When the model does not match the hardware

If the simulator does not reproduce the measured `conflicts/access` for every config you have data on, the
model is WRONG. **Do not quietly patch it until it matches, and do not "meet in the middle" with a plausible
story** — either one buries a real disagreement between the model and the hardware.

Instead: **STOP, tell the user, and OFFER TO REPAIR IT.** Concretely:
1. **Report the mismatch** explicitly — which config, sim value vs measured value, and what you think it
   implies about the model (which constant or rule looks wrong).
2. **Offer to fix it**, with the options laid out: repair the model now, gather a specific additional
   measurement first to disambiguate, or record it as a known gap and move on.
3. **Repair only on their go-ahead** — then re-run `selftest(arch)` and show that it passes.

If they decline the repair, that is their call: continue the work they asked for, but the model is now
known-inconsistent, so **every simulated result on that target is untrustworthy** and you must say so
every time you report one. A model mismatch is a finding to surface and offer to fix — never a bug to
hide, and never something to patch quietly mid-analysis so the numbers line up.

✗ "the A store is 3× conflicted" (provenance unstated — the reader will assume measured)
✗ "target B should behave the same as target A here" (no model for B = STOP, not an extrapolation)
✓ "A store, **simulated** on the validated <gfx target> model: 3.0 conflicts/access (no hardware this run)"
✓ "A store, **measured** on <gfx target>: 3.0 conflicts/access, simulator reproduces it exactly"
✓ "sim says 1.0, hardware says 3.0 on <config> — the model is wrong, most likely <constant/rule>. I've
   stopped. Want me to repair the model, take one more measurement first, or log it as a known gap?
   Until it's fixed, nothing simulated on this target is trustworthy."

## Modes — ASK if the user did not say which

**If the user did not name a mode, STOP and ask.** Do not infer one. Guessing `simulate` risks a modelled
number being read as measured; guessing `investigate` silently commits the user to a long container run.

| | **simulate** | **investigate** |
|---|---|---|
| question it answers | "does this layout conflict, and why?" (design time) | "what does this kernel actually do on this GPU?" |
| arch | **user must supply it**; no arch ⇒ STOP and ask | must equal the **host GPU**; detect, state it, STOP on mismatch |
| precondition | arch has a registered `ArchLDS` **and** `selftest(arch)` passes | same, **plus** a working ROCm container (7.10+) + GPU |
| gate | `selftest(arch)` | `selftest(arch)` **and** sim == rocprof counters, to the number |
| GPU needed | no | yes |
| runtime | seconds | long (container bring-up, probe builds, pad sweep) |
| verdict | `SIMULATED` | `VALIDATED` |
| figure | rendered, **watermarked SIMULATED** | rendered, titled MEASURED |
| binding stage ("is it worth fixing?") | **N/A — cannot answer.** Needs counters. Say so. | answered |

**Full stops, in both modes** — these are hard refusals, not warnings:
- **No arch given** → ask. Never assume a gfx target because it is the one we happen to have a model for.
- **No validated model for the requested arch** → stop. `lc.arch_lds(arch)` raises for you; report it and
  offer to build+validate a model for that arch (a fresh probe sweep on that hardware — see the
  "EXTENDING TO A NEW ARCH" section of `lds_conflict.py`). Never extrapolate one target's constants to
  another — NB, wave size, write-port width and combine depth all differ across CDNA/RDNA generations.
- **investigate, arch ≠ host GPU** → stop. `run_probe` raises for you. Offer simulate mode for the
  requested arch *if* a validated model for it exists; otherwise the previous rule applies.
- **`selftest(arch)` fails** → stop, tell the user the model no longer reproduces its own corpus, and
  **offer to repair it** (see "When the model does not match the hardware"). Repair on their go-ahead,
  never silently mid-analysis.

## Store vs read coverage — simulate does NOT cover both, and the asymmetry is physical

The validated write-port model (`PORT_BANKS=8`, `COMBINE=4`) is exactly that — **write-side**. Every row of
the write corpus is a store-mirror measurement. The read port obeys its **own measured rule**, so the two
accesses get different treatment and you must say which one you analyzed:

| access | entry point | simulate gives you |
|---|---|---|
| **cooperative store** | `analyze_store` | full `conflicts/access` + the conflict-free pad (the write-port model) |
| **MMA-operand wave read** | `analyze_read` | `conflicts/access` = **max_depth − 1** (the read-port rule: no port cap, no combine, phases SUM). **gfx90a only, and only inside its envelope** — 2 dwords/lane (`ds_read2_b32`), no broadcast. Outside it: geometry only, and asking for a cost RAISES |

Why the read cannot simply reuse `simulate()`: applying write constants to a read emits a confidently wrong
number, the exact failure this skill exists to prevent. What IS sound without a read-port model is the
address map — "these lanes of a served group address DIFFERENT dwords in the SAME bank" is a property of the
map, independent of how the hardware serializes the pile. So `analyze_read` reports the pile and refuses the
cost.

**This matters for design work:** on an interleaved layout the conflict often lives in the *read*, not the
store. A simulate run that reports only the store has NOT cleared the design — say so. Use the read geometry
to FLAG an access for `--mode investigate`, never to price it.

**Arch scope.** gfx90a's read model ships registered (its corpus is in `read_hists`, gated by `selftest`).
Any OTHER target has no measured read port: `analyze_read` there reports geometry and REFUSES a cost — never
carry gfx90a's read constants across. To add a target, measure its own read corpus (`run_probe(descs,
mode="read", ...)` + the `n_reads` slope), derive that port's rule, and `register_read_model` it, which
refuses unless `selftest` passes.

**The envelope is gated, not advisory.** `dwords_per_lane` is a REQUIRED argument taken from the
disassembly — the emit declares `vw=1` and the backend merges, so the emit cannot tell you. Deriving it from
the phase count would impose the width rather than detect it, and the gate could never fire.

Note `read_datum` is NOT `store_datum` with another descriptor: the store path assumes a lane touches a
whole number of dwords, which is false for a sub-dword read (an f16 `vw=1` read touches HALF a dword and
floors to an EMPTY map). `read_datum` derives and de-duplicates the dwords per access instead.

Detect the host arch with `rocminfo | grep -m1 gfx` (or `get_device_arch(0)`) and **state it up front**,
before doing any work, so a wrong target is caught in the first line and not after a 20-minute run.

## What this skill can do (capabilities)

| You want to… | Ask / trigger | You get |
|---|---|---|
| **Find if the STORE conflicts** (and how much) | "does the <A/B> store cause a bank conflict?" | `conflicts/access`, measured (investigate) or modelled (simulate) — never ungated |
| **Find if the READ collides** (geometry only) | "does the wave read conflict?" | `analyze_read` — WHETHER and WHERE lanes collide. **No cost number**: `PORT_BANKS`/`COMBINE` are write-side, so a read's replay count needs hardware (see "Store vs read coverage") |
| **Locate the collision** | (part of the analysis) | the served group (half-wave × phase) + bank + colliding `T{l}R{r}` + the N-way |
| **Visualize it** | (part of the analysis) | the committed 3-panel register→LDS dataflow, **conflicted vs fixed** side-by-side |
| **Understand WHY** in plain language | "why is it conflicting?" | the mechanism (e.g. K-stride aliasing) + a concrete thread walk-through + the fix |
| **Compare layouts before writing a kernel** | "would interleaving conflict on <gfx target>?" | simulate mode — fast, no GPU, labelled |
| **Decide if it's worth fixing** | "does this conflict matter?" | the binding-stage read (LDS exposed vs hidden) — **investigate only** |
| **Get the cheapest fix** | "how do I fix it?" | bottleneck-driven lever (pad / contiguity-preserving swizzle / narrow / redistribute), re-measured |

## Prerequisites (Read First)

1. `../shared/prerequisites.md` — path resolution, dispatch rules, "Consult, Don't Improvise", fail-fast.
2. `../shared/temporary_file_policy.md` — the probes/CSVs/containers this skill creates are temporary; track and offer cleanup.

## Experts to dispatch (via the dispatch table)

- **LDS Expert** (`lds_expert.md`) — OWNS the LDS: bank geometry, the address→bank map, the per-arch bank
  count/width and access-serialization rules (these differ by gfx target), swizzle/padding mechanics, and the
  conflict model itself. This is the primary expert for this skill.
- **Profiling Expert**, domain hint `"hardware counters"` → **rocProf Expert** (`rocprof_expert.md`) — owns
  the rocprofv3 invocation, counter selection, and counter semantics.
- **MMA Expert** (`mma_expert.md`) — consulted for how the operand/accumulator LAYOUT choice drives the LDS
  access pattern (which store/read descriptor, interleave, C-shuffle), and how a fix trades against the rest
  of the pipeline. It defers to the LDS Expert on the bank model itself.

You are the hands (build probes, run rocprof, build the simulator, compare); the experts own the model and
the measurement design. Consult them; do not improvise the model yourself.

## The LDS bank model — owned by the LDS Expert (SOT: `lds_banks.md`); don't restate it here

The model — per-half-wave × per-dword-phase arbitration, K-aliasing, floor vs fixable, the contiguity floor +
width ladder, conflict-free ⇔ per-phase bank permutation, order-independence, served-group sizing, arch
variants — lives in **`helpers/tiling/docs/lds_banks.md`** and is owned by the **LDS Expert**. Dispatch the LDS
Expert for the model and the fix decision (it treats the model as a hypothesis its simulator must validate).

Two operational facts this skill's mechanics need (everything else: read `lds_banks.md`):
- The simulator MUST arbitrate **per half-wave × per phase** — never sum a lane's `b128` dwords into one
  histogram (that hides the conflict). Validate it reproduces the measured `conflicts/access` before trusting.
- `conflicts/access = SQ_LDS_BANK_CONFLICT / (SQ_LDS_IDX_ACTIVE − SQ_LDS_BANK_CONFLICT)`; `SQ_LDS_ADDR_CONFLICT`
  (same-address broadcast, a separate pathology) should be ~0 — report it too.

## Reusable tooling — USE the committed module, do NOT regenerate scripts

The validated simulator, the bit-exact address map, the isolation micro-probes, the rocprof CSV parser,
and the 3-panel register→LDS renderer are **committed once** at
`rocke.helpers.tiling.lds_conflict` (arch-parameterized; see `lc.ARCHS` for which targets currently have a
validated model). Import and use
them — do NOT re-implement these as fresh `tmp/` scripts each investigation (that burned tokens and let the
model drift). The MECHANISM and the TOOLING are stable; only the per-CASE measured numbers change.

### Preferred path — one call: `analyze_store` (chains + gates everything)

Use the orchestrator; it removes the hand-wired, error-prone steps (loose gating, hand-built tables, "which
pad fixes it" guesswork). It takes an explicit `mode` and hard-fails rather than guessing.

**No dangerous defaults.** `analyze_store` requires every load-bearing parameter — arch, dtype, strides,
origin, swizzle, and all four labels. That is deliberate: a defaulted value does not error, it silently
analyzes a *different* kernel than yours and prints your kernel's name on the answer.

`analyze_store` does: **`selftest(arch)`** → address-map → `simulate` → *(investigate)* **`measure` on GPU +
HARD `gate(sim==HW)`** → `recommend_pad` (closed-form conflict-free pad) → *(optional)* verify the fix on
GPU → `render_conflict_3panel`. It returns a `ConflictReport` whose `.facts_table()` yields the facts +
model-validation markdown rows directly, already labelled for the mode it ran in.

Enforced in code, so you cannot get it wrong by accident:
- `mode="investigate"` without `measure` → raises. `mode="simulate"` *with* `measure` → raises.
- An arch with no registered model → raises (`no validated LDS model for ...`).
- `selftest(arch)` failing → raises. Surface it and **offer to repair the model**; do not patch it quietly.
- A simulated figure is **watermarked**; a "measured" figure with no measurement → refuses to render.

### Low-level primitives (for custom flows / new access patterns)

`addr_map` (bit-exact map from the real emit) · `simulate` / `simulate_hist` (the write-port model) ·
`predict_pad_sweep` / `recommend_pad` / `conflict_free_bank_of` (stripe rule) · `run_probe` / `build_probe` /
`ProbeDescs` (isolation probes) · `parse_counter_csv` + `COUNTER_PMC` / `ROCPROF_RECIPE` (rocprof harness) ·
`gate` (hard sim==HW assert) · `render_conflict_3panel` (the 3-panel figure). All gate internally where a
mislabeled artifact is possible — e.g. `render_conflict_3panel` asserts sim reproduces the supplied measured
BC/c-a, the fix pad is conflict-free by the stripe rule, and the fixed panel is drawn collision-free. This is
the guardrail against the meaningless hand-drawn diagrams that motivated this module.
- **New arch (gfx942/RDNA):** add an `ArchLDS(...)` to `lc.ARCHS`, then re-validate `lc.selftest(arch)`
  against a freshly measured corpus for that arch before use — constants never carry over between targets.
  Extend the module in place; never fork it into `tmp/`.
- Only drop to a bespoke `tmp/` script when the module genuinely lacks a capability the case needs — and
  then fold that capability BACK into the module so the next investigation inherits it.

## Environment — investigate mode only; profile in a container, never bare-metal

Bare-metal `rocprofv3` on this host CRASHES on the in-process ctypes HIP load
(`rocprofiler_at_intercept_table_registration ... error 16`, SIGABRT) because the host HSA runtime and the
profiler disagree. Profile inside a **ROCm container, 7.10 or newer** instead.

Container images and their internal paths CHANGE — treat everything below as a shape to fill in, not a
recipe to paste. Discover the actual image and paths on the machine you are on (`docker images | grep -i
rocm`), and confirm the container's GPU arch matches the analysis target before measuring anything.

```bash
docker run -d --name lds_prof --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host \
  -v <repo-root>:/work -w /work/<path-to>/rocke/platform \
  <rocm-7.10+ image> sleep infinity
# inside every exec — locate the ROCm SDK libs in THIS image rather than assuming the path:
export LD_LIBRARY_PATH=<sdk_devel>/lib:<sdk_core>/lib:$LD_LIBRARY_PATH
export PYTHONPATH=python ROCKE_CPP_QUIET_FALLBACK=1
```
- Find the container's python (it is usually a venv, not the system one) and check `rocprofv3 --version`
  and that numpy is importable before building probes.
- Invoke `rocprofv3 ... -- <python> <script>` directly. **Do NOT wrap in `env` / `bash -c` chains that
  re-exec** — the double-exec re-registers the tool and SIGABRTs.
- Counter file (`lds_counters.txt`), one pass:
  `pmc: SQ_LDS_BANK_CONFLICT SQ_LDS_ADDR_CONFLICT SQ_LDS_IDX_ACTIVE SQ_INSTS_LDS SQ_WAVES`
- `rocprofv3 -i lds_counters.txt --kernel-include-regex '<kernel-name>' --truncate-kernels --output-format csv -d <out> -- python3 <script>`
  → CSV at `<out>/pmc_1/*/*_counter_collection.csv` (root-owned; `rm` it from inside the container).

## Isolating store vs read (aggregate counters can't split read from write on these targets; ATT decoder absent)

Use `lds_conflict.build_probe` / `run_probe` (above) — they build the isolation micro-kernels for you. Feed
them a `ProbeDescs(coop_native, coop_store, wave_read)` built from the kernel's EXACT descriptors (e.g.
`_macro_coop_descs(...)` store, `_wave_descs_interleaved(...)` read via `_transpose_desc`). The module already
implements both isolation modes:
- **store-mirror** (`mode="store"`): loop `{store; sync; read(store-layout); sync}` — the read makes each store
  live (defeats hoisting); measures the store pattern (write+read of it).
- **read-only** (`mode="read"`): `store once; loop {read; sync}` — isolates the read pattern.
  `conflicts/access` is a RATIO, robust even if some iterations are optimized out.
- **pad sweep is the control that PROVES the counter is live**: vary `lds_pad` (keep a multiple of 8 f16 =
  b128 alignment). If padding moves `SQ_LDS_BANK_CONFLICT`, the counter responds and the number is real; if a
  pad drives it to 0, that layout was conflicted and the pad fixes it.
- Every probed kernel MUST stay **bit-exact** (`max_abs_diff==0.0`) or the numbers are meaningless.

## Workflow

**Step 0 — settle mode and arch BEFORE anything else.** This is the first thing you do and the first thing
you tell the user, because both other branches are expensive to unwind:
- Mode not stated → **ask**. Arch not stated (simulate) → **ask**.
- Resolve the host arch (`rocminfo | grep -m1 gfx`) and **state it**: "host GPU is <X>; analyzing <Y> in
  <mode> mode." In investigate mode X must equal Y — if not, stop and offer simulate for Y.
- `lc.arch_lds(arch)` then `lc.selftest(arch)`. No model → stop and offer to build one. Selftest fails →
  stop, report it, and **offer to repair the model**.

Then:

1. **Read prerequisites; settle the output/temp location** (temp-file policy). Confirm `NB` and the wave
   size for the target from its `ArchLDS` — do not assume 32/64.
2. **Dispatch the LDS Expert** for: the bank geometry, the address→bank map, WHICH accesses to analyze
   (store, read, B, C), and the conflict hypothesis. Consult the **MMA Expert** for how the layout choice
   drives the access pattern (which descriptor / interleave). In investigate mode also dispatch the
   **rocProf Expert** for the rocprofv3 command + counter semantics. Pass them the concrete kernel/shape,
   **and the mode** — an expert reasoning about a simulate-mode question should not propose a counter plan.
3. **Gather the numbers.** This is the step that forks:

   | | simulate | investigate |
   |---|---|---|
   | | `analyze_store(..., mode="simulate")` — the address map + validated model, seconds, no GPU. | Bring up the container, build the isolation probes with `build_probe`/`run_probe` (kernel's EXACT descriptors), run the pad sweep, collect counters (`parse_counter_csv`), verify bit-exactness (`max_abs_diff==0.0`). Record every number. |

4. **Gate.** Simulate: `selftest(arch)` is the gate and it already ran — nothing further, but the result
   stays labelled SIMULATED. Investigate: compute the simulator's predicted `conflicts/access` for EVERY
   measured config and require **simulator == hardware, to the number**. On mismatch, STOP and follow
   "When the model does not match the hardware" — report it and offer to repair. Do not proceed past this
   gate with a story.
5. **Locate + visualize the conflict** (see "Visualize the conflict" below). Call
   `lds_conflict.render_conflict_3panel` (or let `analyze_store` do it) — it extracts the EXACT colliding
   group from the validated simulator and gates internally that the picture matches the mode's evidence.
   A simulate-mode figure comes out **watermarked SIMULATED**. Do NOT hand-draw.
6. **Explain WHY, in plain language, with a concrete example** (see the explanation template). Then show the
   fix (from the pad/swizzle sweep) as a second, side-by-side diagram where the highlight is gone.
7. **Binding stage — investigate only.** In simulate mode you have no counters, so you CANNOT say whether
   the conflict is worth fixing. Say that plainly rather than implying it: "simulated — whether this costs
   wall-time needs an investigate run." Offer the investigate run as the next step.
8. **Cleanup.** Remove probes, CSVs, containers per the temp-file policy (investigate mode).

## Visualize the conflict — the register→LDS dataflow (committed renderer; ✗ do NOT hand-draw)

`lds_conflict.render_conflict_3panel` owns the *data + gates* (prepares the validated `datum`, asserts
sim==measured) and delegates ALL drawing to `layout_render`'s `draw_conflict_*` — one consistent visual
language; ✗ never hand-write matplotlib here. Drive it with the measured numbers; it gates internally. Viz
knobs/contract: `visualization_api_surface.md`.

**3-panel left-to-right dataflow (one served instruction), the contract to preserve:** (1) **registers**
(`RegisterFileComponent`, tid×reg, cell = the `(K,M)` datum, hue = lane) — highlight the conflicting
instruction's registers; (2) **arrows** — a few `T{l}R{r} → dword → bank` funnels converging on the SAME bank
column, each labeled `addr = row·stride+col`, `bank = (addr/2) mod NB`; (3) **LDS bank grid** (banks x, depth
y, cell = `T{l}R{r}`, shade = served group) with a **red box on the bank column where >1 thread at a DIFFERENT
address lands in one served group** — the located conflict, annotated N-way.

**Rules:** highlight ONLY the simulator-flagged group (the red box must coincide with the counter-confirmed
phase — disagree → STOP, the sim/map is wrong); ONE representative group + column (the mechanism repeats);
render TWO side-by-side — **conflicted vs fixed** (the pad/swizzle that drove BC down → arrows fan out, red box
gone); reuse layout-viz conventions verbatim (first-8, `T{l}R{r}`, shade = order).

## Explain WHY — plain language, with a concrete example (always include this)

After the picture, explain the mechanism in everyday terms. Structure: (1) one sentence naming the cause,
(2) the rule that makes it happen, (3) a concrete walk-through of a few threads, (4) the fix and why it works.
Avoid jargon; when you must use a term (bank, phase, replay) define it in the same breath.

Template (fill from the validated result — do NOT paste these numbers, regenerate per case):
> **What's happening:** The `<store/read>` asks `<N>` different threads to touch the **same bank** at the same
> moment, so the hardware can't do them in parallel — it does them one after another (`<N>` "replays"),
> costing `<N-1>` extra cycles every time.
>
> **Why:** A bank is just `address ÷ 4 bytes, mod <NB>`. Here the LDS rows are `<stride>` dwords apart, and
> `<stride>` is a whole number of banks — so stepping to the next row lands on the **same bank**. The threads
> that hold the same `<free coord, e.g. M>` but a different `<contracted coord, e.g. K>` sit on different rows
> of the same column, so they all map to that one bank.
>
> **Walk it through:** Thread `T0` writes `(K=0,M=0)` → address `0` → bank `0`. Thread `T1` writes
> `(K=<..>,M=0)` → address `<stride>` → bank `(<stride>/2) mod <NB> = 0` **again**. `<N>` threads repeat this,
> all landing on bank `0` in the same phase → an `<N>`-way pile-up.
>
> **The fix:** Pad each LDS row by `<p>` so the stride is no longer a whole number of banks. Now row 1 shifts
> over by `<p/2 mod NB>` banks, so those `<N>` threads spread across `<N>` banks and go in parallel — the
> replays disappear (measured: `SQ_LDS_BANK_CONFLICT` `<before> → <after>`). Cost: `<p>` extra LDS per row
> (occupancy trade-off).

Distinguish for the user, in plain terms, the **throughput floor** ("64 threads must share 32 banks, so 2
passes are unavoidable — not a bug") from the **fixable pile-up** ("these 8 all chose the same bank — that's
the part padding removes"). Only the second is *fixable*.

## Binding stage — investigate only; the LDS Expert makes the fix call (do NOT chase BC→0)

BC is a diagnostic, not the objective — **wall-time is.** Collect these alongside BC so the LDS Expert can
judge whether the conflict is even worth fixing (the decision framework + the "conflict-free can be slower"
tradeoff live in `lds_banks.md` §7 / `lds_expert.md` — do not restate them here):
- `SQ_WAIT_INST_LDS` (is the LDS stage exposed?), `SQ_VALU_MFMA_BUSY_CYCLES` (compute saturation),
  `SQ_INSTS_LDS`/`SQ_INSTS_VALU` (issue pressure), `TCC_EA_RDREQ`/`TCC_EA_WRREQ` (HBM).

Hand these to the LDS Expert with the conflict data; report whether **TFLOPS actually moved**, not just whether
BC dropped.

**In simulate mode this section is not available** — it needs counters. Do not substitute a plausible story
about whether the conflict matters. State "needs an investigate run" and offer it.

## Output Format

Lead with the provenance line. Everything below inherits it, and a reader who sees only the header must
already know whether they are looking at hardware or a model.

```
## Bank-Conflict Analysis — <kernel / access>

- mode / provenance: <investigate → MEASURED | simulate → SIMULATED (model only, no hardware this run)>
- arch / NB / wave:  <gfx target / NB / wave (confirmed from its ArchLDS, not assumed)>
- host GPU:          <gfx target>   <"(matches" | "(N/A — simulate mode)">
- model gate:        selftest(<arch>) <PASS | FAIL → STOP, offer repair>
- access:            <A store / A read / ...>, descriptor <name>, VW <b128/b64>, dtype <..>

### Facts
investigate → | config | SQ_LDS_BANK_CONFLICT | SQ_LDS_IDX_ACTIVE | conflicts/access | ADDR_CONFLICT |
               (header: "Hard facts (rocprof, real GPU)  [bit-exact: yes]")
simulate    → | config | conflicts/access (SIMULATED) | served | productive |
               (header: "Predicted (<arch> model — NO hardware this run)"; never print an empty
                counter column with "?" in it, which reads as a failed measurement)

### Model validation
investigate → | config | simulator conflicts/access | measured | match? |   (any ✗ ⇒ STOP, offer repair)
simulate    → selftest(<arch>): PASS — the model reproduces <arch>'s own measured corpus.
               No per-case hardware comparison exists; that is the limit of this mode.

### Mechanism (only stated because the model is validated)
- <e.g. K-stride aliasing: bank index independent of K because K-stride = k·NB dwords>

### Diagram — register→LDS dataflow, conflict located (conflicted | fixed, side by side)
- <path>: 3-panel register file → arrows → LDS bank grid; red box on the <N>-way bank; fixed panel alongside
- located: served group <half-wave/phase>, bank <b>, colliding cells <T{l}R{r}, ...>
- <simulate: note the figure is watermarked SIMULATED>

### Why it happens (plain language, with the walk-through example)
- <the fill-in-the-blank template: what's happening / why / walk it through / the fix — everyday words>
- floor vs fixable: <what part is the unavoidable wave>NB floor vs the fixable pile-up>

### Binding stage (is it worth fixing?)
investigate → wait-on-LDS <..> / MFMA-busy <..> / HBM <..> -> <LDS-exposed | MFMA-bound | HBM-bound | balanced>
simulate    → NOT ANSWERABLE without counters. <offer the investigate run>

### Recommendation
investigate → bottleneck-driven: <do-nothing (hidden) | cheapest bandwidth-keeping lever (pad/free swizzle) |
  narrower swizzle (adds instrs, may regress) | redistribute>; report achieved conflicts/access AND the
  measured TFLOPS change (re-measured).
simulate    → the conflict-free lever the model predicts (e.g. pad +<p>), explicitly marked as a
  PREDICTION whose wall-time benefit is unverified.
```

## No stored results — regenerate per case

Do **NOT** persist measured OR simulated conflict numbers, per-kernel results, or "known" conflict factors
in this skill, the experts, or memory. Every situation (arch, tile, dtype, layout, pad, pipeline) is
different; baking in a number invites stale/wrong reuse — and a stored *simulated* number is worse still,
because the label that made it honest does not survive the copy. Each invocation MUST generate its own data
and pass its mode's gate. This file stores only the METHOD and the environment shape — never the answers.

**Exception (not a violation):** the `_MECHANISM_*` validation corpus baked into `lds_conflict.py` is the
proof that the write-port model reproduces hardware — it validates the *mechanism*, it is NOT a per-case
answer to reuse. `selftest()` is a gate on the model's correctness, not a shortcut around measuring a new
case. A new kernel/tile/dtype still gets freshly measured + freshly gated.
