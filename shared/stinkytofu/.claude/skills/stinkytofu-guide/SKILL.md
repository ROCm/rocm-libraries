---
name: stinkytofu-guide
description: >-
  Locate where a feature lives in the StinkyTofu codebase, and advise where and
  how to add a new one, grounded in the actual source. Use when the user asks
  "where is X", "where should this new feature go", "how do I add a pass /
  instruction / architecture / scheduling constraint", or types
  /stinkytofu-guide <topic>.
allowed-tools: Bash, Read, Grep, Glob
---
# stinkytofu-guide

Answer two kinds of question about the StinkyTofu tree, and nothing else:

1. **Where does an existing feature live?** — find the files/symbols, cite `file:line`.
2. **Where and how should a NEW feature go?** — recommend the location and the pattern
   to follow, grounded in how the codebase already does it.

This skill is **read-only and advisory**: locate, explain, recommend. Do not implement
unless the user separately asks. There is no external wiki — the source tree in this
directory is the single source of truth.

## 0. Verify against CODE, not docs (docs may be stale)

Docs under `docs/` and the map in this file are starting points, **not authority**.
Before you assert where something is or how it works, open the actual `.cpp`/`.hpp`
and confirm. Cite `file:line` and name stable symbols (pass classes, functions,
structs) — line numbers drift. If code and docs disagree, trust the code and tell the
user the doc is stale.

## 1. StinkyTofu is a SCHEDULER, not a code generator (the foundational rule)

StinkyTofu reorders and annotates instructions TensileLite already produced; it does
**not invent computation from nowhere**. This frames every "where should my feature go?"
answer. Three consequences to apply, and to raise with the user:

1. **Correctness must not depend on StinkyTofu.** With `ScheduleIterAlg=0` (SIA0) in
   TensileLite, the generated kernel must already be **correct without any StinkyTofu
   scheduling passes** (peephole, DAG scheduler, etc.). StinkyTofu makes correct code
   *faster*, it does not make incorrect code correct. If a proposed feature would make
   the kernel wrong unless a StinkyTofu pass runs, the design is inverted — the
   correctness piece belongs in TensileLite code gen, not in a scheduler pass.

2. **Instruction order chosen in TensileLite code gen is not authoritative — the DAG
   scheduler reorders it.** So "fixing" behavior by carefully ordering instructions in
   the TensileLite generator is meaningless for anything the DAG scheduler will touch.
   **Anchor invariant:** the DAG scheduler uses **WMMA as anchors — the relative order
   among WMMA instructions is preserved.** You can reason about ordering *relative to
   WMMA*, but not about the absolute order of surrounding instructions.
   **Caution:** this invariant breaks if a pass that reorders WMMA (e.g. a WMMA-reorder
   pass such as `StinkyWmmaVgprReorderPass`) runs *before* the DAG scheduler — then the
   anchor order itself changed. Check the pass ordering before relying on WMMA order.

3. **Ask: does this belong in code gen (TensileLite), the scheduler (StinkyTofu), or
   both — split across the two?** The analogy is C++ vs. the compiler. The compiler
   schedules; it does not decide you wanted prefetching. If you need prefetch, either
   **write the prefetch instructions in TensileLite code gen**, or **build an explicit
   signal** (a pragma / intrinsic / annotation) that tells StinkyTofu what to do.
   StinkyTofu will not synthesize a feature the input never expressed. Classify the
   feature, and expect it may **split into two halves**:
   - The **code part** (new instructions / new computation / a new signal to emit) ⇒
     TensileLite code gen, or a lowering / intrinsic-expansion step in StinkyTofu.
   - The **scheduling part** (how the scheduler should place, order, or hazard-guard
     those instructions) ⇒ the DAG scheduler / cost model.
   Many real features need **both**: e.g. prefetch emits the load instructions (code
   part) *and* teaches the scheduler how far ahead to hoist them (scheduling part). When
   that's the case, say so explicitly and scope each half to its correct home — don't try
   to cram the whole thing into one side because it's convenient.

## 2. Multi-instruction features — don't make the scheduler learn a whole sequence

Some features are not one instruction but a **multi-line sequence** with internal
structure. Exposing that raw sequence to the DAG scheduler (or other passes) forces them
to understand relationships they were never designed for, and can **break the DAG or
other features**. This is not a yes/no rule — it's a consideration to raise before adding
such a feature. Two established patterns to offer:

- **Collapse to a pseudo-instruction, expand after scheduling.** If the sequence doesn't
  affect performance much, represent it during scheduling as a **single pseudo
  instruction carrying the summed cycles/latency**, let the scheduler treat it as one
  opaque node, then **expand it into the real instructions in a post-scheduling pass**.
  This keeps the scheduler's model simple and correct. Example: the **cluster barrier**
  is several instructions, not one — see `InsertClusterBarrierPass.cpp` and
  `docs/developer/cluster-barrier.md`; pseudo instructions are recognized via
  `isPseudoInst(...)` throughout the passes.

- **Do it as a post-scheduling transform when it's just a mechanical copy/edit.** If the
  feature is essentially duplicating or lightly editing already-scheduled instructions,
  do it *after* scheduling rather than in TensileLite code gen — so the scheduler never
  has to deal with the extra, complicated cases. Example: the **InitC** feature clones
  the iter=0 region and rewrites the WMMA `src C` to zero on the chain heads; it lives in
  `RegionClonePass` (`initCIterWmma_zeroChainHeads`, `src/transforms/asm/RegionClonePass.cpp`;
  spec in `include/stinkytofu/pipeline/CloneSpec.hpp`). Implementing that in code gen
  would just hand the scheduler more complex input for no benefit.

Weigh this against performance: if the sequence's internal scheduling genuinely matters
for perf, then hiding it behind a pseudo may cost too much and it should be modeled
properly. Raise the tradeoff; don't decide silently.

## 3. Don't default to a new pass because it's easier — check the scheduler question

**The most common design mistake here:** a new feature interacts with instruction
scheduling, but the author adds a standalone post-scheduling pass anyway — simply
because adding a pass is far easier than touching the DAG scheduler model
(`src/transforms/asm/dag/CDNA5.hpp` and friends). "Easier" is not a design reason.

A post-fix pass is **not automatically wrong** — LLVM itself implements plenty of
features as post-schedule fixups, and StinkyTofu does too. The goal is to make this a
*conscious decision*, not a default driven by effort. So before recommending where a
feature goes, surface the scheduling question with the user:

- "Does this feature interact with instruction scheduling — ordering, latencies,
  co-issue, hazards, or resource/slot usage?"
- If yes or unsure: "Does the scheduler need to *know* about it while placing
  instructions (so it should be a dependency/hazard/priority in the DAG), or can it be
  correctly and cheaply applied as a fix-up after scheduling?"

Then weigh it honestly with them:

- **Scheduler needs to know** — if a correct result requires the scheduler to account
  for this while choosing an order (it changes what a good/legal schedule is), encode it
  in the scheduler: the ready-queue / cost model in `src/transforms/asm/dag/CDNA5.hpp`,
  the DAG build in `StinkyDAGSchedulerPass`
  (`src/transforms/asm/StinkyDAGSchedulerPass.cpp`), or dependency construction
  (`StinkyBuildImplicitDependencyPass`, `BuildDefUseChain.cpp`). A later pass can't
  recover a scheduling decision the scheduler already made wrong.
- **A post-fix is genuinely fine** — if the transform is local, doesn't change what a
  good schedule looks like, and can be applied correctly after scheduling (or is a
  lowering / standalone analysis / declarative rewrite), then a separate pass or a
  peephole pattern is the right, simpler choice.

Whichever way it goes, state the tradeoff explicitly so the choice is deliberate — never
"a pass is easier, so a pass."

## 4. The map — where things live (verify before citing)

Paths are relative to this `shared/stinkytofu/` directory.

- **Scheduler (read before advising on ordering/latency/hazards/co-issue)**
  - Per-arch model + ready queue: `src/transforms/asm/dag/CDNA5.hpp` (CDNA5/Gfx1250:
    WMMA–VALU co-issue timeline, latencies, slot fill), plus `InFlightQueue.hpp`,
    `ReadyQueue.hpp` in the same dir.
  - The pass: `StinkyDAGSchedulerPass` — `src/transforms/asm/StinkyDAGSchedulerPass.cpp`,
    header `include/stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp`. Splits blocks
    into regions, builds a per-region DAG from physical registers, drains ready nodes.
  - Dependencies feeding the DAG: `StinkyBuildImplicitDependencyPass.cpp`,
    `BuildDefUseChain.cpp` (inserts pseudo-PHIs), `PhiPlacement.cpp`.
- **IR definitions**
  - Logical IR (architecture-agnostic): `include/stinkytofu/ir/logical/`, `src/ir/logical/`
  - Asm IR (arch-specific: `StinkyInstruction`, `Function`, `BasicBlock`):
    `include/stinkytofu/ir/asm/`, `src/ir/asm/`
- **Passes** — Asm: `src/transforms/asm/`; Logical: `src/transforms/logical/`.
  Registry of names usable from `stinkytofu-opt`: `availablePasses` in
  `tools/stinkytofu-opt/stinkytofu-opt.hpp`.
- **Hardware / architectures** — `hardware/GfxXXX/` `.def` files feed TableGen; a new
  arch needs no C++. See `docs/developer/adding-architecture.md` (verify against tree).
- **Peephole / intrinsic patterns** — `src/transforms/asm/PeepholePatterns.pattern`,
  `src/transforms/logical/LogicalIRPatterns.pattern`; grammar in
  `docs/developer/pattern-grammar.md`.
- **Wait counts** — `StinkyWaitCntInsertionPass.cpp` and `src/transforms/asm/waitcnt/`.
- **Tools** — `tools/stinkytofu-opt/` (standalone pass driver: parses a `.stir`/`.s`
  asm/IR file and runs passes on it, `stinkytofu-opt [options] <ir_file> [--PassName...]`;
  passes run in command-line order — see `tools/stinkytofu-opt/README.md`),
  `tools/stinkytofu-check/` (FileCheck harness over stinkytofu-opt output), plus
  `stinkytofu-cfg`, `waitcnt-check`, `tablegen`, `intrinsic-compiler`.
- **Tests** — GTest in `tests/`, FileCheck `.stir` in `tests/filecheck/`.
- **How-to-extend guides** — `docs/developer/adding-instructions.md`,
  `adding-architecture.md`, `adding-intrinsics.md`, `adding-peephole-patterns.md`,
  `architecture.md`. Treat as guidance; confirm against code.

## 5. Design constraints — apply these when advising

Load-bearing invariants. Advice that violates one is wrong.

- **Design for future-proofing — features stack, so no hardcoded if-else ladders.**
  Assume an unbounded number of features will be layered on top of each other over time.
  A design that special-cases the current feature with a hardcoded `if (thisFeature)` /
  `switch` branch does not compose: the next feature adds another branch, and the
  combinatorics explode. Prefer extensible mechanisms — a registry/table, per-feature
  config carried through `PassFeatureConfig`, a pass that composes with others, a pattern
  entry — over a one-off conditional wired into shared code. When you review or recommend
  a design, explicitly check "what happens when the 2nd, 5th, 10th feature does this
  too?" and steer away from anything that only works because it's the only one.
- **Make the scheduler-vs-post-fix choice deliberate** (see §3); don't pick a pass just
  because it's less work.
- **New architecture ⇒ no C++.** Add a `hardware/GfxXXX/` `.def` set; TableGen generates
  instruction tables, latencies, opcode enums. Don't steer toward editing generated
  `.inc` or per-arch C++ when a `.def` suffices.
- **New pass ⇒ register it** in `availablePasses`
  (`tools/stinkytofu-opt/stinkytofu-opt.hpp`), else it can't be FileCheck-tested.
- **A new switch/option must be reachable from BOTH entry points.** A feature gated by a
  flag/option is only usable if it is plumbed through:
  1. the **Python API** — so TensileLite can turn it on. Module-level options live in the
     `MODULE_OPTIONS_LIST` X-macro (`include/stinkytofu/bindings/python/Module.hpp`),
     which generates both the `ModuleOptions` struct and its Python binding; pass-tuning
     flags live in `PassFeatureConfig` (`include/stinkytofu/core/Types.hpp`) carried via
     `PassContext`. Verify the option is actually exposed, not just defined in C++.
  2. **stinkytofu-opt** — so it is runnable/testable in isolation. stinkytofu-opt runs
     passes directly on a `.stir`/`.s` file, so the option must be selectable from its
     command line (parsed in `tools/stinkytofu-opt/stinkytofu-opt.cpp` — the
     `--PassName=arg` / `--flag` handling) to be exercised standalone and FileCheck-tested.
     An option wired to only one side is a half-feature: either TensileLite can't use it,
     or it can't be run/tested on an asm file without the full Python/TensileLite path.
     Advise wiring both.
- **Prefer a peephole/intrinsic pattern over a new pass** for local, declarative rewrites
  (`.pattern` files, not C++).
- **Pseudo-PHI nodes are never emitted.** Code that snapshots, counts, or linearizes
  instructions MUST skip `GFX::PHI` (see `linearizeProgramOrderStinky()` in
  `src/transforms/asm/PassOrderSnapshotJson.cpp`). Flag this for any stream-walking feature.
- **rocisa vs StinkyTofu split is by arch:** gfx1250+ uses StinkyTofu, older uses rocisa;
  cross-boundary work goes through `src/conversion/rocisa/`.
- **Two IR levels are distinct:** arch-agnostic logic in Logical IR; concrete/arch-specific
  in Asm IR.

## 6. Point to the author when the code can't answer the design question

Not every "why is it designed this way?" is answerable from the source. When the design
intent is unclear, or a feature's rationale isn't captured in code/comments, it is
genuinely helpful to name **who to ask** rather than guess. Use git history as the
source of truth for authorship:

- Who wrote / owns a feature: `git log --format='%an' -- <path>` (or `git shortlog -sne
  -- <path>` for the ranked contributor list), and `git log -p -- <path>` /
  `git log -S'<symbol>'` for the commit that introduced a specific behavior.
- For a specific line's origin: `git blame -L <start>,<end> <file>`.

You don't need to do this every time — only when it adds value: an unclear design, a
non-obvious tradeoff, or when the user is about to change something whose rationale
matters. Then it's fine to say, e.g., *"the cluster-barrier adjustment was authored by
`hcman2` — the code here doesn't explain the timing choice, so ask them for the design
intent; what's in the tree isn't enough to answer that."* When multiple people touched
it, list the co-authors so the user knows the full set to consult. Always ground the
name in `git log`/`git blame` output, never invent one.

## 7. Shape of a good answer

**Lead with a TL;DR, or end with one — never make the user mine a wall of text for the
point.** You may explain in depth (this domain needs it), but the bottom line must be
scannable in one place: a 1–3 line summary at the top, or a "**Bottom line:**" at the
end. Detail supports the summary; it does not replace it. Don't defocus.

**Prefer a few lines of pseudo-code over 2–3 sentences of prose.** When describing a
mechanism, control flow, data shape, or where code should hook in, show it as a small
code/pseudo-code block. It's faster to read and less ambiguous than sentences. E.g.
instead of "the scheduler drains ready nodes and picks the one with the highest priority
that fits the co-issue window," write:

```
while ready.notEmpty():
    n = ready.pickHighestPriority(fitsCoIssueWindow)
    schedule(n); updateReady(n)
```

- **"Where is X?"** → the file(s) + key symbol(s) + one line each, with `file:line`.
- **"Where should my new feature go?"** → first surface the §3 scheduling question with
  the user, then give: target directory + an existing file to model it on + any
  registration / skip-PHI / IR-level / scheduler constraint that applies + a concrete
  analogue already in the tree. Cite what you actually read; if the tree contradicts
  this skill, trust the tree and say so.
- **"Who made this / why is it like this?"** → when code doesn't explain the design,
  name the author(s) from `git log`/`git blame` and say the source is insufficient —
  point the user to the right person instead of guessing (see §6).
