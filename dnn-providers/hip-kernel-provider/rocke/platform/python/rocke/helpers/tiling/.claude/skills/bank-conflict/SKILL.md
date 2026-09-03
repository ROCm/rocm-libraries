---
name: bank-conflict
description: Rigorously analyze LDS bank conflicts for a kernel/layout. ENFORCES empirical validation — no conflict claim ships without rocprof hardware counters AND a simulator that reproduces them to the number. Use whenever asked "does X cause a bank conflict / how many / why".
argument-hint: <kernel or layout to analyze> [--shape MxNxK]
---

# Bank-Conflict Analysis Skill

You are a bank-conflict analysis coordinator. LDS bank conflicts are an **empirically verifiable
statistic**. Your job is to make sure every statement about them is backed by hardware measurement AND a
model that reproduces that measurement — never by hand-reasoning, a static picture, or an unvalidated mental
model.

## What this skill can do (capabilities)

Every output is backed by rocprof counters AND a simulator that reproduces them to the number (the Cardinal
Rule) — no measurement, no verdict.

| You want to… | Ask / trigger | You get |
|---|---|---|
| **Measure if an access conflicts** (and how much) | "does <A/B store/read> cause a bank conflict?" | `conflicts/access` from rocprof + a simulator gated to reproduce it (else it REFUSES to answer) |
| **Locate the collision** | (part of the analysis) | the served group (half-wave × phase) + bank + colliding `T{l}R{r}` + the N-way |
| **Visualize it** | (part of the analysis) | the committed 3-panel register→LDS dataflow, **conflicted vs fixed** side-by-side |
| **Understand WHY** in plain language | "why is it conflicting?" | the mechanism (e.g. K-stride aliasing) + a concrete thread walk-through + the fix |
| **Decide if it's worth fixing** | "does this conflict matter?" | the binding-stage read (LDS exposed vs hidden) — BC is subordinate to wall-time |
| **Get the cheapest fix** | "how do I fix it?" | bottleneck-driven lever (pad / contiguity-preserving swizzle / narrow / redistribute), re-measured |

## The Cardinal Rule (why this skill exists)

**Do NOT state a conflict number, factor, or "is/isn't a conflict" verdict until BOTH are true:**
1. You have **rocprof hardware counters** for it (real GPU), and
2. You have a **simulator/model that predicts those exact counters** from the address maps.

If the model does not reproduce the measured `conflicts/access` for every config you have data on, the
model is WRONG — keep fixing it silently until it matches, or gather the specific new measurement you need.
Never "meet in the middle" with a plausible story. This rule is here because hand-reasoned/static-model
claims about this kernel flip-flopped (8× → "no conflict" → 7×) and wasted the user's time. The GPU is the
arbiter; a model is only trustworthy once it matches the GPU.

## Prerequisites (Read First)

1. `../shared/prerequisites.md` — path resolution, dispatch rules, "Consult, Don't Improvise", fail-fast.
2. `../shared/temporary_file_policy.md` — the probes/CSVs/containers this skill creates are temporary; track and offer cleanup.

## Experts to dispatch (via the dispatch table)

- **LDS Expert** (`lds_expert.md`) — OWNS the LDS: bank geometry, the address→bank map, the per-arch bank
  count/width and access-serialization rules (gfx90a/gfx942/RDNA differ), swizzle/padding mechanics, and the
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
`rocke.helpers.tiling.lds_conflict` (arch-parameterized; **gfx90a validated**). Import and use
them — do NOT re-implement these as fresh `tmp/` scripts each investigation (that burned tokens and let the
model drift). The MECHANISM and the TOOLING are stable; only the per-CASE measured numbers change.

### Preferred path — one call: `analyze_store` (chains + gates everything)

Use the orchestrator; it removes the hand-wired, error-prone steps (loose gating, hand-built tables, "which
pad fixes it" guesswork). You supply only a `measure` callable that runs rocprof in the container and returns
the counters — the module does the rest and HARD-FAILS if the sim doesn't match hardware.

```python
from rocke.helpers.tiling import lds_conflict as lc

assert lc.selftest(lc.GFX90A)                        # gate the model on this arch first

descs = lc.ProbeDescs.from_coop(coop_native, wave_native, transpose=_transpose_desc)  # both transposes, correct

def measure(pad, mode="store"):
    """Run ONE probe under rocprof in the container; return its counters. This is the only host/
    container-specific glue — everything else is in the module."""
    r = lc.run_probe(descs, mode, tile_free=TF, lds_pad=pad, arch="gfx90a")   # bit-exact round-trip
    assert r["max_abs_diff"] == 0.0
    # ... docker exec rocprofv3 (lc.COUNTER_PMC / lc.ROCPROF_RECIPE) on a runner that calls run_probe ...
    hw = lc.parse_counter_csv(outdir)                 # {BC, IDX, conflicts_per_access, ADDR, ...}
    hw["max_abs_diff"] = r["max_abs_diff"]
    return hw

rep = lc.analyze_store(descs, tile_free=TF, wtag="b64", operand_label="A", dims_label="M",
                       measure=measure, verify_fix=True, render_to=out_png)
# rep.verdict / rep.conflicts_per_access / rep.fix_pad / rep.located / rep.png / rep.facts_table()
```

`analyze_store` does: address-map → `simulate` → **`measure` on GPU + HARD `gate(sim==HW)`** →
`recommend_pad` (closed-form conflict-free pad) → *(optional)* verify the fix on GPU → `render_conflict_3panel`.
It returns a `ConflictReport` whose `.facts_table()` yields the "Hard facts" + "Model validation" markdown
rows directly. **No `measure` ⇒ the report is `UNVALIDATED` and rendering is REFUSED** — a number without
hardware is never presented (the cardinal rule, enforced in code).

### Low-level primitives (for custom flows / new access patterns)

`addr_map` (bit-exact map from the real emit) · `simulate` / `simulate_hist` (the write-port model) ·
`predict_pad_sweep` / `recommend_pad` / `conflict_free_bank_of` (stripe rule) · `run_probe` / `build_probe` /
`ProbeDescs` (isolation probes) · `parse_counter_csv` + `COUNTER_PMC` / `ROCPROF_RECIPE` (rocprof harness) ·
`gate` (hard sim==HW assert) · `render_conflict_3panel` (the 3-panel figure). All gate internally where a
mislabeled artifact is possible — e.g. `render_conflict_3panel` asserts sim reproduces the supplied measured
BC/c-a, the fix pad is conflict-free by the stripe rule, and the fixed panel is drawn collision-free. This is
the guardrail against the meaningless hand-drawn diagrams that motivated this module.
- **New arch (gfx942/RDNA):** add an `ArchLDS(...)` to `lc.ARCHS`, then re-validate `lc.selftest(arch)`
  against a freshly measured corpus for that arch before use — do NOT assume gfx90a constants carry over.
  Extend the module in place; never fork it into `tmp/`.
- Only drop to a bespoke `tmp/` script when the module genuinely lacks a capability the case needs — and
  then fold that capability BACK into the module so the next investigation inherits it.

## Environment — the ONLY way to profile these kernels (validated recipe)

Bare-metal `rocprofv3` on this host is **HSA 8.19** and CRASHES on the in-process ctypes HIP load
(`rocprofiler_at_intercept_table_registration ... error 16`, SIGABRT). Profile inside a **ROCm 7.14
container** instead:

```bash
docker run -d --name lds_prof --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host \
  -v <repo-root>:/work -w /work/dnn-providers/hip-kernel-provider/rocke/platform \
  fmha-build-a:rocm714 sleep infinity
# inside every exec:
export LD_LIBRARY_PATH=/opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel/lib:/opt/venv/lib/python3.14/site-packages/_rocm_sdk_core/lib:$LD_LIBRARY_PATH
export PYTHONPATH=python ROCKE_CPP_QUIET_FALLBACK=1
```
- Container python is `/opt/venv/bin/python3` (rocprofv3 1.3.2, numpy present). gfx90a = MI210.
- Invoke `rocprofv3 ... -- python3 <script>` directly. **Do NOT wrap in `env` / `bash -c` chains that
  re-exec** — the double-exec re-registers the tool and SIGABRTs.
- Counter file (`lds_counters.txt`), one pass:
  `pmc: SQ_LDS_BANK_CONFLICT SQ_LDS_ADDR_CONFLICT SQ_LDS_IDX_ACTIVE SQ_INSTS_LDS SQ_WAVES`
- `rocprofv3 -i lds_counters.txt --kernel-include-regex '<kernel-name>' --truncate-kernels --output-format csv -d <out> -- python3 <script>`
  → CSV at `<out>/pmc_1/*/*_counter_collection.csv` (root-owned; `rm` it from inside the container).

## Isolating store vs read (aggregate counters can't; gfx90a has no read/write-split LDS counter; ATT decoder absent)

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

1. **Read prerequisites; settle the output/temp location** (temp-file policy). Confirm the target arch and
   `NB` (do not assume 32).
2. **Dispatch the LDS Expert** for: the bank geometry, the address→bank map, WHICH accesses to analyze
   (store, read, B, C), and the conflict hypothesis. Dispatch the **rocProf Expert** for the rocprofv3
   command + counter semantics. Consult the **MMA Expert** for how the layout choice drives the access
   pattern (which descriptor / interleave). Pass them the concrete kernel/shape.
3. **Measure (hard facts first).** Bring up the container, build the isolation probes with
   `lds_conflict.build_probe`/`run_probe` (kernel's EXACT descriptors), run the pad sweep, collect the
   counters (`lds_conflict.parse_counter_csv`). Verify bit-exactness (`max_abs_diff==0.0`). Record every number.
4. **Model + VALIDATE.** The simulator already exists — `lds_conflict.simulate` (+ `simulate_hist`). First
   run `lds_conflict.selftest(arch)` to confirm the model still reproduces the validation corpus, then compute
   its predicted `conflicts/access` for EVERY measured config. **Gate:** simulator == hardware (to the number)
   across all configs, or the model is wrong — fix it *in the module* and re-check. Do not proceed past this gate.
5. **Locate + visualize the conflict** (see "Visualize the conflict" below). Call
   `lds_conflict.render_conflict_3panel` — it extracts the EXACT colliding group from the validated simulator
   and gates internally that the picture shows the SAME collision the counters proved. Do NOT hand-draw.
6. **Explain WHY, in plain language, with a concrete example** (see the explanation template). Then show the
   fix (from the pad/swizzle sweep) as a second, side-by-side diagram where the highlight is gone.
7. **Cleanup.** Remove probes, CSVs, containers per the temp-file policy.

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

## Binding stage — collect these; the LDS Expert makes the fix call (do NOT chase BC→0)

BC is a diagnostic, not the objective — **wall-time is.** Collect these alongside BC so the LDS Expert can
judge whether the conflict is even worth fixing (the decision framework + the "conflict-free can be slower"
tradeoff live in `lds_banks.md` §7 / `lds_expert.md` — do not restate them here):
- `SQ_WAIT_INST_LDS` (is the LDS stage exposed?), `SQ_VALU_MFMA_BUSY_CYCLES` (compute saturation),
  `SQ_INSTS_LDS`/`SQ_INSTS_VALU` (issue pressure), `TCC_EA_RDREQ`/`TCC_EA_WRREQ` (HBM).

Hand these to the LDS Expert with the conflict data; report whether **TFLOPS actually moved**, not just whether
BC dropped.

## Output Format

```
## Bank-Conflict Analysis — <kernel / access>

- arch / NB:     <gfx90a / 32 (confirmed)>
- access:        <A store / A read / ...>, descriptor <name>, VW <b128/b64>

### Hard facts (rocprof, real GPU)   [bit-exact: yes]
| config | SQ_LDS_BANK_CONFLICT | SQ_LDS_IDX_ACTIVE | conflicts/access | ADDR_CONFLICT |
| ...    | ...                  | ...               | ...              | 0             |

### Model validation (simulator vs hardware)
| config | simulator conflicts/access | measured | match? |
| ...    | ...                        | ...      | ✓      |
(If any row is ✗, the analysis is NOT done — the model is wrong.)

### Mechanism (only stated because the model matches)
- <e.g. K-stride aliasing: bank index independent of K because K-stride = k·NB dwords>

### Diagram — register→LDS dataflow, conflict located (conflicted | fixed, side by side)
- <path>: 3-panel register file → arrows → LDS bank grid; red box on the <N>-way bank; fixed panel alongside
- located: served group <half-wave/phase>, bank <b>, colliding cells <T{l}R{r}, ...>

### Why it happens (plain language, with the walk-through example)
- <the fill-in-the-blank template: what's happening / why / walk it through / the fix — everyday words>
- floor vs fixable: <what part is the unavoidable 64>NB floor vs the fixable pile-up>

### Binding stage (is it worth fixing?)
- wait-on-LDS <..> / MFMA-busy <..> / HBM <..> -> <LDS-exposed | MFMA-bound | HBM-bound | balanced>

### Recommendation (bottleneck-driven; conflicts/access AND TFLOPS)
- <do-nothing (hidden) | cheapest bandwidth-keeping lever (pad/free swizzle) | narrower swizzle (adds instrs,
  may regress) | redistribute>; report achieved conflicts/access AND the measured TFLOPS change (re-measured)
```

## No stored results — regenerate per case

Do **NOT** persist measured conflict numbers, per-kernel results, or "known" conflict factors in this skill,
the experts, or memory. Every situation (arch, tile, dtype, layout, pad, pipeline) is different; baking in a
number invites stale/wrong reuse. Each invocation MUST generate its own hard data on the real GPU and gate
against the simulator. This file stores only the METHOD and the environment recipe — never the answers.

**Exception (not a violation):** the `_MECHANISM_*` validation corpus baked into `lds_conflict.py` is the
proof that the write-port model reproduces hardware — it validates the *mechanism*, it is NOT a per-case
answer to reuse. `selftest()` is a gate on the model's correctness, not a shortcut around measuring a new
case. A new kernel/tile/dtype still gets freshly measured + freshly gated.
