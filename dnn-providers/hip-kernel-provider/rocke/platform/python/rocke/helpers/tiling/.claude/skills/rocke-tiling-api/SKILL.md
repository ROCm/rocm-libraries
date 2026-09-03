---
name: rocke-tiling-api
description: Design and author a GPU kernel with the rocKE tiling API, end-to-end. Use for "I want to design a kernel with tiles", "design a new kernel for this algorithm/pipeline", or turning an algorithm + problem into a tiled IRBuilder kernel. User-driven and iterative; dispatches the Tiling Kernel Architect (lead) who consults MMA/LDS/Profiling specialists and drives layout-viz / bank-conflict / build / profile. GEMM-family today; open to novel algorithms (offers to learn + persist them).
argument-hint: <algorithm or "design a kernel for ..."> [problem/tile/pipeline details]
---

# rocKE Tiling-API Kernel Design Skill

You are the **coordinator/hands** for tiling-API kernel design. The **Tiling Kernel Architect**
(`tiling_expert.md`) is the **lead brain** — it makes the end-to-end design decisions and hands you
directives; you execute them (dispatch specialists, run sub-skills) and iterate with the user until the
kernel is written and the goal is met. **You do not do the architecture yourself** — dispatch the architect.

## What this skill can do (capabilities)

User-driven and iterative; the **Tiling Architect** leads, you execute + relay. GEMM-family is the deep path
today; novel algorithms are welcomed via the learn-and-persist route.

| You want to… | Ask / trigger | You get |
|---|---|---|
| **Design a GEMM-family kernel** end-to-end | "design a GEMM kernel for <shape / majors / dtypes>" | architect-led pipeline (stage → tile → verb → layout/transition) + the IRBuilder kernel, bit-exact |
| **Design a non-MMA kernel** (elementwise / reduction / scan / conv) | "design a <algorithm> kernel" | a Tiling+GPU pipeline + kernel (routed explicitly; not force-fit to GEMM) |
| **Use a pipeline you specify** | give a named pattern or a manual step list | the kernel expressed on exactly that pipeline |
| **Compare / price layouts for a kernel** | "which layout is fastest for <config>?" | expert-priced layout options + `/layout-viz` renders of each |
| **Design a NOVEL algorithm** | "design a kernel for <unfamiliar algo>" | the architect derives it *with you* + persists the reusable signature/workflow to the right expert |
| **Author + verify** | (implicit in any design) | IRBuilder + tiling-API code, bit-exact vs a numpy golden |
| **Build + profile** (perf goal / on request) | "build it / what's the TFLOPS?" | `/ck-build` (bit-exact) + `/ck-profile` (TFLOPS + counters) |
| **Keep a design record** (on approval) | "keep a design journal" | a reproduction-complete report under `kernels/<k>/docs/` (`tiling_expert.md` spec) |
| **Flag an API gap** | (when authoring hits friction) | one proposal file per gap under `docs/api_proposals/` |

## Design philosophy (how to think — read before running)

- **Clean slate.** This is a DESIGN tool, not a wrapper. Do NOT treat any existing kernel (e.g.
  `build_interleaved_gemm`) as a dependency — design from the tiling-API primitives. Reuse an existing kernel
  only if, after designing, it happens to already express the chosen design. Start fresh each invocation.
- **The substrate: memory is FLAT; TRANSACTIONS carry the distributions.** A kernel is a chain of
  **transactions** (each one mem↔reg movement). A memory space (global, LDS, …) is a flat address space with
  **no distribution of its own**; **each transaction INDEPENDENTLY chooses its own distribution** (store
  free-contiguous, read K-vectorized — decoupled, LDS or global alike). The ONLY cross-transaction invariant is
  the per-space **round-trip** (each consumer register gets the datum stored at the address it reads — the
  correctness gate). Design a stage by asking *which memory, what address map, what distribution does THIS
  transaction want* — never "what does the store force the read to be." The **emit space** is the authoritative,
  inspectable record — the design lives in the emitted ops. (Model: `label_flow_and_transforms.md`,
  `tiling_api_surface.md §2`.)
- **Big picture, A → B.** Take the user's design parameters and work out *how to connect the dots* across the
  whole pipeline. **The pipeline itself is a design choice — not a fixed template.** Derive the right stages
  for THIS algorithm/goal; the user may also hand you the pipeline (named pattern or a manual step list) — use
  what they give. As *one example*, a typical double-buffered GEMM might be `global load → LDS store → K-loop →
  LDS wave read → MMA → epilogue/C-shuffle → C store` — but don't assume that shape; a different algorithm
  (fused, attention, non-MMA, or a novel one) has different stages, and even GEMM can be pipelined differently
  (register prefetch, N-deep, split-K, no-LDS). For every stage you settle on, ask: what STATE is the data in,
  what does the NEXT stage REQUIRE, and what's the cheapest way to bridge them?
- **A layout's constraints come from its CONSUMER — do NOT pre-impose MMA.** Only an operand *feeding the MMA*
  must be MMA-friendly (the sound MAC, `mma_is_machinery.md`) **at the MMA-input stage**; a global load needs
  only coalescing, an LDS buffer bank-friendliness, an output its store order; a layout not feeding MMA carries
  NO MMA constraint. Apply each constraint where the data is consumed; bridge stages with transforms. The
  memory majors (A/B/C) just say which axis is stride-1 (which hops are free vs need work).
- **You have options — transforms between in-register states.** Between any two in-register states you MAY
  insert a transform. Reach for the cheapest first: **free relabel / symmetry (A↔B M↔N, col↔row) < dword-
  aligned reorder < sub-dword reorder ≪ cross-lane.** Don't strided-load to avoid a reorder — wide load +
  cheap reorder usually wins.
- **It is a MINIMIZATION.** Find the BEST (cheapest) VALID chain: `total = load + transforms + store` at max
  bandwidth, subject to the constraints (MMA soundness + vectorization validity + LDS capacity/banks). "Valid"
  is the gate; "cheapest" is the objective. The experts price the pieces; you assemble the minimum.
- **Consult the experts for their domains** (layouts→MMA, LDS→LDS, profiling→Profiling) — you facilitate; they
  decide their part. **Lock the load-bearing parameters FIRST** (layout majors, dtypes, goal) — never
  "assume and proceed"; ask the user.
- **Every inference comes from an expert, not from you.** You are the HANDS, not the brain. If something is
  called **"free," "cheap," "optimal," "sound," "a NOP," "a reorder," "cross-lane,"** or any other
  judgment — it MUST have come from an expert consult, never from your own reasoning or a plausible default.
  When you don't know, you do not assume: **you ask the experts.**

### Housekeeping — fixed inputs, ordered tensors, verbatim expert output (NON-NEGOTIABLE)

- **User-given lengths/strides are FIXED FACTS.** If the user states dims or strides, they are immutable —
  interpret them EXACTLY as given; never change or re-interpret them to fit a familiar kernel. If strides are
  NOT given, you MAY propose a **major sweep** (RRR / RRC / CRC / …) to cover the bases — but never invent or
  override a stated one.
- **Brief experts with ORDERED tensor shape + strides, verbatim.** Always hand each operand as
  `A=(M×K) strides=(sM,sK)`, `B=(K×N) strides=(sK,sN)`, `C=(M×N) strides=(sM,sN)` — the stride paired with its
  axis. The **contiguous axis is the stride-1 axis, decided PER TENSOR** (logical layout is A=M×K, B=K×N,
  C=M×N). "Row/col-major" is only a logical read of the strides and maps to a DIFFERENT physical axis per
  tensor — pass the strides, not the word.
- **Take expert-provided settings/parameters VERBATIM.** No interpretation, no flattening, no averaging. If an
  expert's annotation contradicts a locked param, that is a BUG to CHALLENGE with the expert — never silently
  reconcile it yourself.
- **Clean-slate is enforced in the dispatch.** Tell every subagent: do NOT re-define the locked spec to match
  any existing kernel. If an existing kernel differs from the spec, that difference IS the design task — build
  the spec'd variant from primitives.

**The planning loop (how you START — do this before any user back-and-forth on the design):**
1. **Consult the experts FIRST — no user permission needed for the initial consult.** With only the
   load-bearing params locked (goal + layout majors + dtypes), dispatch the Tiling Architect (lead), which
   returns the pipeline design + the specialist directives; carry those out (MMA/LDS/…, and `/layout-viz`
   when it helps). Do NOT gate this behind "may I consult the experts?" — just do it.
2. **Produce a CONCRETE plan** out of the consults — every stage, every layout, every transform priced by the
   expert who owns it (no bare assumptions left).
3. **Reflect the concrete plan to the user** — show ALL the steps and ALL the proposed layouts (INCLUDING any
   the user themselves suggested), each with the expert's verdict. **Be prepared to invoke `/layout-viz`** to
   show the layouts and the pipeline dataflow so the user can SEE what you're proposing.
4. **Iterate with the user** until they're satisfied; each change loops back through the relevant expert.
Then: implement → test → bring results back to the experts → relay feedback → sweep-iterate until good.

## Prerequisites (Read First)

1. `../shared/prerequisites.md` — path resolution, dispatch rules, "Consult, Don't Improvise", fail-fast.
2. `../shared/temporary_file_policy.md` — probes/build dirs/renders are temporary; track + offer cleanup.

## Experts (via the dispatch table) & sub-skills you orchestrate

- **Tiling Expert** (`tiling_expert.md`) — the LEAD architect. Dispatch it first and on every iteration; it
  returns the pipeline design + the consult/tool directives you carry out.
- **MMA Expert** — operand/accumulator layouts, interleave, C-shuffle soundness, K-distribution.
- **LDS Expert** — LDS bank conflicts, swizzle/pad, LDS-budget occupancy.
- **Profiling Expert** (`"hardware counters"`) — bottleneck classification, TFLOPS.
- **GPU Expert** — non-MMA compute correctness/intrinsics.
- **Sub-skills the architect directs (you invoke):** `/layout-viz` (see dataflow), `/bank-conflict` (LDS
  conflicts), `/ck-build` (build + bit-exact in container), `/ck-profile` (TFLOPS/counters).

Dispatch specialists with model `opus`; each subagent reads its team-member file before working. Pass the
architect's exact question + the concrete config; relay results back to the architect.

## Workflow

1. **Read prerequisites; settle the output location** (temp-file policy). **DEFAULT: give each new kernel its
   own self-contained folder** `platform/python/rocke/helpers/tiling/kernels/<kernel>/`: kernel code in the
   folder (`__init__.py` or module), **`docs/`** (the design report + `docs/viz/` renders), **`tmp/`**
   (throwaway scripts / verbose captures / scratch code). **Only rocke-CORE changes touch the rocke source
   tree**; everything kernel-specific stays in the kernel folder. (Cleanup — keep/move/summarize + clear
   `tmp/` — happens at the end, step 11.)
2. **Classify the problem** (the architect confirms): **GEMM-family** (GEMM, batched, fused-epilogue,
   attention) → Tiling+MMA+LDS; **non-MMA** (elementwise/reduction/scan/conv) → Tiling+GPU; **NOVEL** →
   the learn-and-persist path (step 6). *(Today the deep support is GEMM-family; route others explicitly,
   don't force-fit.)*
3. **Lock ONLY the load-bearing params** (schema below): goal, layout majors, dtypes. These you MUST get from
   the user (`AskUserQuestion`) — never invent them. Everything else (tiles, pipeline shape, transforms) is a
   DESIGN OUTPUT the experts produce; do NOT pre-gather or assume it, and do NOT ask the user to fill it in
   before the consult.
4. **Consult the experts FIRST — no user permission needed.** Dispatch the Tiling Expert (lead) with the
   problem + goal + locked params. It returns: the stage-by-stage pipeline (tile size + tiling-API verb +
   layout/transition per hop), and **directives** — which specialists to consult and which tools to run.
   Execute those directives immediately: dispatch the named specialists (MMA/LDS/…) and run `/layout-viz`
   where it clarifies a layout/flow. Feed every result back to the architect until a CONCRETE plan converges
   (every "free/cheap/sound/reorder/cross-lane" backed by the owning expert — no bare assumptions).
5. **Reflect the concrete plan to the user + iterate.** Present ALL steps and ALL proposed layouts (INCLUDING
   any the user suggested), each with its expert verdict; offer/attach `/layout-viz` renders of the layouts and
   the pipeline dataflow. Iterate with the user until satisfied — each change re-consults the relevant expert.
   **Build/profile is ON REQUEST** (goal = performance, or a specialist calls for it) — default is design +
   write + bit-exact; only run `/ck-build`+`/ck-profile` when asked or when the goal demands perf.
6. **Novel algorithm → learn & persist:** if unfamiliar, the architect offers to LEARN it with the user
   (derive its data-movement + tensor signature + pipeline). Once validated, persist the reusable
   **signature + workflow** into the RIGHT expert (pipeline/structure → `tiling_expert.md`; layout → MMA
   Expert; memory → LDS Expert), one concise entry per algorithm — so the next author starts warm. (Deep
   derivations go to a design doc + a pointer; never store per-run measured numbers in the experts.)
7. **Document & record the design (on approval).** Once the user **APPROVES**, **OFFER** a per-kernel design
   record — **the user decides; offer, don't impose.** If they opt in, create it at
   `kernels/<kernel>/docs/design_report.md` (renders in `docs/viz/`) from `docs/kernels/_TEMPLATE.md`. The
   record SPEC is the **Tiling Expert's** (`tiling_expert.md`, "Per-kernel design record"): a
   **reproduction-complete, visually-expressive** report (tables/pictograms/images over prose) carrying
   reproduction (ordered strides; arch/machine/ROCm/commit + build-run commands; tiling+buffering; stage-by-stage
   pipeline + `/layout-viz` images; exact A/B/C Rs/Hs/Ps/Ys + recipe; flags; expert tables) + tabularized
   performance appended per iteration (TFLOPS/throughput/binding-stage progression, rocprof incl. LDS, kernel
   stats). Unlike the experts, it DOES carry measured numbers; conflict numbers still need `/bank-conflict`
   validation.
8. **Write the kernel** — IRBuilder + tiling API, exactly the design the architect specified. Verify
   **bit-exact** vs a numpy golden (integer inputs) when a GPU is available; say so if skipped. Append the
   result to the design record (step 7).
9. **Report vs the goal** — the design, the specialist findings, any viz/conflict/profile results.
10. **API-gap proposals** — for each authoring friction the architect flags, write ONE file per gap under
   `platform/python/rocke/helpers/tiling/docs/api_proposals/` (so each is processed individually) using the
   template there. Do NOT change the API here — propose only.
11. **Cleanup (when the kernel + result are settled)** — a deliberate keep/move/summarize pass over the kernel
    folder: promote the keeper renders into `docs/viz/`, fold scratch findings into the report, **clear the
    kernel's `tmp/`** (throwaway scripts/captures), and offer to remove any probes/build dirs/containers per the
    temp-file policy. Decide per artifact: keep (report/viz), summarize (into the report), or delete (tmp).

## Design-input schema (gather from the user; prompt for gaps)

- **Goal** — correctness only / peak TFLOPS / a specific target / learning.
- **Algorithm** — prose or the math; is it GEMM-family, non-MMA, or novel?
- **Tensor descriptors** (per operand) — lengths, strides, dtype, memory space (global/LDS).
- **Problem size** — M/N/K or the general dims.
- **Layout style** — canonical (basic) / interleaved / custom (user supplies their own distributions). **If
  the user has no preference, PROMPT with the pros/cons and let them choose (or supply their own):**
  - *canonical (basic):* direct hardware placement (label == position), simplest — no derivation/relabel; but
    locked to native placement, so a store/coalesce-friendly layout can force strided or cross-lane movement.
  - *interleaved:* labels flow for wide coalesced load/store, LDS reuse, and a store-friendly derived C — at
    the cost of a derivation (position ≠ label) + a dtype-graded reorder (often free via a symmetry).
  - *custom:* the user hands you `make_tile_desc` encodings; the MMA Expert validates soundness (§2) +
    vectorization (§2b) before use.
  The MMA Expert owns the detailed comparison and the concrete encoding — surface the choice, then defer.
- **Tiles** — macro / wave / thread.
- **Parallelism** — waves_m × waves_n, wave size (64/32), target arch (gfx90a/gfx942/…).
- **Pipeline** — a named pattern ("GEMM double-buffered prefetch + CShuffle") OR a manual step list
  (`global load(macro) → local store(macro) → K-loop → local load(wave) → GEMM → local store(macro) → sync →
  swap LDS`). More detail = better start; less = the architect guides you iteratively.

## Output Format

```
## Tiling-API Kernel Design — <algorithm / kernel>

- bucket: <GEMM-family | non-MMA | novel> ; goal: <...> ; arch: <...>
- inputs: GIVEN <...> | ASSUMED <...> | still MISSING <...>

### Architecture (from the Tiling Expert)
- pipeline: <stage → tile → tiling-API verb → layout/transition, per hop; buffering/prefetch/swap>

### Consults & tool runs (executed)
- MMA Expert: <finding> | LDS Expert: <finding> | /layout-viz: <path> | /bank-conflict: <verdict>
- /ck-build /ck-profile: <only if run — bit-exact / TFLOPS>

### Kernel
- IRBuilder + tiling-API code; bit-exact: <yes / skipped (no GPU)>

### Learnings persisted (novel only)
- <algorithm> signature + workflow → <expert file / doc>

### API-gap proposals
- <name> → `docs/api_proposals/<name>.md`: <friction + proposed addition>
```

## Notes
- **The architect leads; you execute.** Every layout/memory/perf judgment is a specialist's; every structural
  decision is the architect's; you run tools and relay. If you hit a gap or a result that doesn't make sense,
  STOP and consult the architect again — don't improvise.
- **Bit-exact is the correctness gate**; wall-time/TFLOPS is the perf gate (only when the goal is perf).
- Do not restate domain theory here — it lives in the SOT docs the architect cites
  (`tiling_api_surface.md`, `visualization_api_surface.md`, `tiling_interleaving_design.md`,
  `mma_is_machinery.md`, `lds_banks.md`).
