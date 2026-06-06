# TensileLite codegen coverage — plan (coverage efficiency → coverage expansion to ≥80%)

**North star:** raise coverage of the TensileLite codegen / assembly-generation code to
**≥80%** (or a documented, honest ceiling), **CPU-only** and **add-only**, in two stages:

1. **Coverage efficiency first** — build a *fast* harness that reproduces the **maximum
   coverage reachable from existing inputs** using the **smallest set of inputs**
   (set-cover), and record **goldens** for that set. This is the accelerator, not the goal.
2. **Then coverage expansion** — iteratively close coverage gaps by adding characterization
   tests that target still-uncovered code, re-measuring after each round, until ≥80% or a
   documented ceiling.

> **Framing (important).** A coverage % is a **selection signal**, not a number to inflate
> by adding tests indiscriminately. Stage 1 does **not** try to maximize coverage; it finds
> the *smallest* inputs that reproduce the *current* maximum, so every later "add a test →
> re-measure" cycle is fast and each test ships with a real input/output golden. **Adding
> tests to raise coverage is Stage 2**, and it is gap-driven — close specific uncovered code
> with the cheapest input that reaches it.

> **Decisions (resolved 2026-06-04):**
> - **Isolation.** One shared `tl-char` container; each generation runs in its **own child
>   process** (`docker exec … pytest`) on a **disjoint** input with its own
>   `COVERAGE_FILE` (a fresh process ⇒ clean rocisa state). Agents are **producers only**;
>   a single serialized **Assemble** step does `coverage combine`, the gate, and all
>   commits. *(Escalation: per-agent worktree + container if collisions appear.)*
> - **Measurement.** **Total coverage is the merge of those isolated runs** (`coverage
>   combine` reconstructs the true union). It both scores each input's contribution for
>   set-cover and reports the union of a chosen set — measurement, not inflation. Validated
>   in P0.
>
> **Decisions (resolved 2026-06-05, via `/orchestration-plan` grill):**
> - **Stage-1 deliverable is a *designed* minimal set, not a selection of existing files.**
>   We author a **minimal, custom, parameter-varied input set across *all* archs**, designed
>   for the **fewest kernels that cover the most whole-project lines**, kept by a
>   **marginal-yield cutoff** (add a parameter-varied kernel only while it adds ≥ a threshold
>   of new whole-project lines). The existing corpus (44 curated + a corpus sample) and
>   `ValidParameters` are **discovery inputs** that *attribute* lines↔parameters — they are
>   not the shipped set.
> - **Two input channels, both built.** (1) **Logic YAML** via `parseLibraryLogicFile`
>   (existing harness). (2) **BenchmarkProblems config** via `BenchmarkProcess` →
>   `constructForkPermutations` → `_generateForkedSolutions` (`Tensile/BenchmarkProblems.py`,
>   `Tensile/BenchmarkStructs.py`) — a **new add-only harness path**. The config's
>   **`ForkParameters`** key emits one Solution per parameter combination and is the **primary
>   parameter-variation lever** for designing minimal-kernel sweeps. Cheapest-first still
>   governs which channel reaches a given target (§3).
> - **Metric = whole-project line-set.** Set-cover / marginal-yield optimize the
>   **whole-project TOTAL** (`coverage report | tail -1`); the two codegen files
>   (`KernelWriterAssembly.py`, `KernelWriter.py`) remain a *named secondary* readout, not the
>   objective.
> - **Two stages kept.** Stage 1 = minimal custom **seed set** + recorded ceiling + goldens
>   (the fast harness). Stage 2 = gap-targeted custom inputs to ≥80% or documented ceiling.
> - **Scope of this run.** `/orchestration-plan` emitted ready-to-run Dynamic-Workflow plans
>   for **all remaining phases (now P1–P6)** in `WORKFLOW-SPECS.md`; each phase = one workflow
>   run, gated by a machine checkpoint (verification receipt + baseline commit + progress-log
>   update), with **no human approval between phases** once launched.
> - **GPU boundary is narrow; one prerequisite PR (P0.5), not an ADD-ONLY relax.** The codegen
>   emit, **solution derivation** (`_generateForkedSolutions` — CPU-only, like
>   `parseLibraryLogicFile`), **and TensileCreateLibrary** (a *cross-compiler*: host-side
>   `amdclang++` for the target ISA, **no device**) are all CPU-reachable today. The **only**
>   true GPU dependency is the **client perf-run** — the client app that launches GEMM kernels
>   on a device to collect perf metrics (`ClientWriter.runClient`/`getClientExecutablePath`),
>   plus the system probes `amd-smi` / `rocm_agent_enumerator` (already has `--no-enumerate`;
>   `ProblemSizesMock*` precedent exists). Decision: land a **small standalone prerequisite PR
>   (P0.5)** adding a `--cpu-only`/`--mock-gpu` switch that (a) returns **synthetic perf
>   metrics** in place of the client run and (b) **spoofs `amd-smi`/`rocm_agent_enumerator`
>   output per target architecture** — so the run-orchestration code is exercisable CPU-only
>   on any host. The characterization campaign stays **strictly ADD-ONLY** and merely
>   *consumes* the switch. **P0.5 gates only the P4 expansion rounds that target the client/run
>   path**; P1–P3 and all codegen/CreateLibrary coverage proceed without it.

This file is the single source of truth + checkpoint. Resume: read top-to-bottom, take the
last `coverage/master-baseline-*.txt` as the current BEFORE, continue at the first unchecked
item in §8. Exact, reproducible coverage commands live in `coverage-methodology.md` /
`coverage-methodology.html` — cite them, don't reinvent.

> **⚠️ CURRENT STATE (2026-06-06) — authoritative; read before acting.** **P0, P1, P2, P3 are
> DONE and committed**, and the **`--cpu-only` GPU-mock switch is integrated** (8 `gpu-mocks`
> commits cherry-picked onto this branch; HEAD `6f1e20b1a7f`; `test_cpu_only_switch.py` 17/17).
> Stage 1 shipped a **widened 15-seed set** (gfx942 / gfx950 / gfx90a; seed-subset ceiling
> 35.89%, methodology B). **The true current whole-project coverage is 68.85%** (HEAD, full
> `-m unit`, methodology A = tox coverage-unit, `--cov=Tensile --cov=rocisa`; develop baseline
> 22.47% → **+46.38 pts**; receipt `coverage/head-unit-baseline.txt`). **Gap to ≥80% ≈ 11 pts.**
> Because the switch is present, **P4 runs with `haveSwitch=true`** — no targets are skipped as
> switch-gated; the branch is therefore **no longer strictly add-only** (carries switch source by
> decision). **NEXT = P4** (Stage-2 expansion, rank term-missing on the 68.85% baseline).
> Authoritative status: **§8 + §11 + `BASELINE-AND-PROGRESS.md`**. Sections **§4/§5 below are the
> ORIGINAL narrative (historical framing)** — `PHASE 0/1/2/3` predates the §8 `P0–P6` renumber;
> do **not** re-run P0–P3 from them; start at the first unchecked §8 item (P4).

---

## 1. Hard rules (carried from PLAN-80 — do not relax)

1. **ADD-ONLY.** New files only, under `Tensile/Tests/unit/characterization/` (tests,
   goldens, custom test logic). Never modify/delete any existing file — including
   `pyproject.toml`. Config needs (e.g. parallel coverage) go in a **new** file via
   `--cov-config=<newfile>` / `COVERAGE_FILE`, or are recorded in `resistance.md`.
2. **NEVER push / no PR.** Local atomic commits only, made by the Assemble step.
3. `--cov` takes the **path** `Tensile` — never a dotted module (rocisa SIGABRT).
   Scope to `Tensile/Tests/unit`.
4. Goldens generated **in-container** (`--snapshot-update`, root-owned). Stored
   order-invariant (`{basename, err}`) and run through `canonicalize_asm` (strip
   register/addr/temp-label numbering) — assembly text is order-coupled via rocisa
   scheduler state (PLAN-80 P2).
5. **Pin, don't fix, latent bugs.** Snapshot actual current behavior; log in
   `DECISIONS.md`. Source changes are forbidden — **with one controlled exception: P6 mutation
   testing** (§5 / WORKFLOW-SPECS P6). Mutants are temporary source edits that live **only inside
   throwaway per-mutant git worktrees**, are **always reverted**, are **never committed**, and
   never touch the campaign worktree. After P6 the campaign worktree must be byte-for-byte
   add-only (`git status --porcelain` shows nothing from mutation). P0.5 (the GPU-mock switch) is
   *also* source-changing but is a **separate PR/branch**, not part of this add-only campaign.

---

## 2. Orchestration model (every phase = one workflow run)

**One workflow per phase / per expansion round** (separate runs). Workflows take no mid-run
input, so each ends at its barrier and a **machine checkpoint** (full `-m unit` + baseline
receipt + scope/clean-tree check + explicit-path commit + §8/§11 update) gates the next. No
human approval is required between phases. After a phase passes its machine checkpoint, the
driver/lead immediately launches the next unchecked phase or expansion round.

If a phase cannot pass its machine checkpoint, it must write the named failure/deferred artifact
(`resistance.md`, `round-<N>-deferred.txt`, `CEILING-FINDINGS.md`, mutation leak report, etc.),
update §11, commit only safe report artifacts, and return a structured failure/deferred result
instead of asking for human input.

**Common 4-stage shape:**

```text
A. Enumerate  — derive this run's input list (existing logic files / test configs, a
                set-cover candidate batch, or the uncovered-code targets for an expansion
                round). Parameterizable via `args` to run a small subset first.
B. Generate (fan-out, <=16 parallel) — one agent per input, each an ISOLATED run:
                `docker exec … COVERAGE_FILE=.coverage.<id> pytest … --cov=Tensile …`
                (add `--snapshot-update` only when recording goldens). PRODUCER ONLY:
                writes its test/golden/data file, returns a result schema.
C. Verify (pipeline, skeptical) — re-run without --snapshot-update; confirm goldens are
                stable across two runs, 0 failures; err!=0 outcomes are valid covered
                rejections. Reject flaky/empty.
D. Assemble (serialized, ONLY committer) — `coverage combine` into an EXPLICIT
                COVERAGE_FILE; compute the delta; write resistance.md for unreachable
                code; atomic commits (explicit-path `git add`, never -A); update §8/§11.
```

**Authoring cautions:** 16-concurrent cap (batch wider work-lists); pipeline-by-default,
barrier only where a stage needs all prior results (Assemble); **never have one agent emit a
large artifact in a single turn** (split into section-writers + a cheap assemble step);
**self-contained prompts** (agents see nothing from siblings — pass paths, scope, schema
intent every time).

**Reuse:** the existing harness at `Tensile/Tests/unit/characterization/_codegen/`
(`codegen_harness.py`, `matrix.py`, `canonicalize_asm`, the isaInfoMap conftest). Build on it.

---

## 3. Inputs & the cheapest-first rule

Three input sources, used in this **cost order** (cheapest that reaches the target wins):

1. **Test configs** — `Tests/common/**` BenchmarkProblems specs. These are **not** consumable
   by the logic-file harness (different schema); the **new BenchmarkProblems→Solutions path**
   (built in P1: `BenchmarkProcess` → `constructForkPermutations` → `_generateForkedSolutions`)
   is what makes them an input channel.
2. **Logic files** — shipped tuning YAMLs (existing `parseLibraryLogicFile` harness).
3. **Custom test logic (designed)** — the **primary Stage-1 mechanism** (resolved 2026-06-05):
   a hand-authored **BenchmarkProblems config whose `ForkParameters` sweep** emits the
   *smallest number of kernels* that toggle the *most* whole-project lines, varying only the
   parameters the P1 attribution table shows move coverage. Add-only.

The only real toolchain dependency (from P0's inventory) is the capability probe
`rocisa…init(isa, cxx)` in `Common/Capabilities.py::makeIsaInfoMap` — the **toolchain ↔
snapshot coupling**. **Pin that capability info once → the whole codegen path is
deterministic and snapshot-able.** That single point (plus the subprocess build/run helpers,
which are outside codegen) is all that ever needs mocking.

---

## 4. STAGE 1 — Coverage efficiency (fewest inputs + goldens)

### PHASE 0 — Validate the method; map the toolchain coupling  ✅ DONE (2026-06-04)
- Made golden generation runnable in **isolated processes**; confirmed `coverage combine`
  reconstructs the union of separate-process runs correctly (merged > any single run);
  confirmed the **per-process rocisa footprint limitation** is per-process (a fresh process
  per run avoids it).
- Wrote `toolchain-inventory.md` — key finding: codegen does **not** invoke a compiler; the
  only dependency is `makeIsaInfoMap`. Committed `d6ae0b113d9` (inventory only).
- Result on the de-risk subset: merged 59.18% / 65.93% across the two core codegen files
  (`KernelWriterAssembly.py` / `KernelWriter.py`) from 4 existing tests.

### PHASE 1 — Coverage-efficiency set (set-cover over existing inputs)  ✅ DONE (historical framing — see §8 P1; superseded by the widened seed set)
- **Goal:** the **smallest set of existing inputs that reproduces the maximum coverage
  reachable from (1) test configs + (2) logic files** — per arch. This is the empirical
  **current ceiling** and the **fast harness** for Stage 2.
- **Enumerate:** all candidate inputs (test configs + in-tree logic; a per-arch sample of
  the tuning tree). **Generate/measure:** one agent per input, an isolated run with its own
  coverage data file. **Assemble:** run **set-cover** over the per-input coverage to pick
  the smallest subset whose merge equals the full-set total; record the merged % (the
  ceiling) and the dropped redundancy.
- **Done-criteria (provable in the report):** committed minimal `test-logic/` set; per-input
  coverage data retained; the **empirical current-max coverage** recorded as the
  whole-project TOTAL (and, if a file-scoped figure is quoted, named to its file(s)); a
  one-line set-cover justification per kept input; full `-m unit` 0 failed.
- **This phase does NOT add tests to raise coverage** — it only selects and measures.

### PHASE 2 — Generate goldens for the efficiency set  ✅ DONE as §8 P3 ({basename,err} goldens, 15 seeds, reproduces 35.89%)
- **Goal:** record real **inputs/outputs (goldens)** for every input kept in Phase 1, so the
  set becomes a committed, re-runnable characterization suite.
- **Generate:** one agent per kept input → `--snapshot-update` in-container → snapshot the
  canonicalized, order-invariant assembly + reject outcome. **Verify:** two-run stability.
  **Assemble:** commit tests + `__snapshots__/` per arch.
- **Done-criteria:** every kept input has a stable golden; re-running the set reproduces the
  Phase-1 ceiling (now *fast*); baseline committed.
- **Open:** golden representation — compact digest (today) vs full assembly vs a chunked
  random-sample of substrings + positions. Decide once Phase 1 reveals the worst-case
  assembly size.

---

## 5. STAGE 2 — Coverage expansion (iteratively close gaps to ≥80%)

### PHASE 3 — Iteratively close coverage gaps  ⏳ = §8 **P4 (NEXT)** (repeats until target/ceiling)
Each **round** is one workflow run:
- **Enumerate (rank gaps):** run `--cov-report=term-missing` on the current suite; rank
  uncovered code by missing-statement yield.
- **Pick the cheapest input** that reaches the top targets, per §3 order: config tweak →
  custom test logic (min-parameter variation for that arch) → new logic file.
- **Generate + golden** the new targeted inputs (isolated runs); **Verify** stability;
  **Assemble** `coverage combine` (efficiency set + new inputs) → **measure the delta** →
  commit.
- **Repeat** rounds until **≥80%** on the canonical gate, or a round adds ~nothing and the
  remainder is provably GPU-only/unreachable (→ `resistance.md`).
- The Phase-1 efficiency set is what makes each round's re-measure fast.

> Rounds are gap-driven, not a fixed list. Expect several; each round machine-checkpoints and
> auto-continues while it is still making progress. Independent work
> can proceed in parallel across TensileLite's components — the **codegen modules**,
> **TensileCreateLibrary** (the build system), and the **Tensile(Lite)** tuning workflow —
> since they touch different code.

### PHASE 4 — Gate + golden governance  ⏳ (the ≥80% decision point)
- `coverage combine` across all data → the **whole-project gate**. Assert **≥80%**, or
  record the honest ceiling + evidence in `CEILING-FINDINGS.md`.
- Golden governance: key each golden by **(architecture, compiler version)**; **stable arch
  → one golden** (a change = suspected compiler regression); **evolving arch → keep N
  compiler generations**, retire the oldest as releases settle (findings §8).
- **Done:** combined `master-baseline-<N>.txt`; `golden-governance.md`; target met or ceiling
  documented; `recommendations.md`.

### PHASE 5 — Mutation validation  ⏳
- One agent per mutant, run only against the coverage-selected subset it touches; classify
  killed/survived; **surviving mutants → new targeted tests** fed back into Phase 3. No
  source left mutated; report committed.

---

## 6. Target & trajectory

- **Target = ≥80%** on the whole-project gate, reached by Stage 2.
- **Stage 1 establishes the starting ceiling** (max from existing inputs) — the gap between
  that and 80% is exactly what Stage 2 closes by adding targeted inputs.
- Only measured numbers are trustworthy; the findings' ~50/62/67/69% were projections.
  Record each round's real delta in §11. If the rounds stop adding coverage below 80%, that
  becomes the documented ceiling (the honest outcome), not a failure to hide.

---

## 7. Verification & trust (the Verify stage contract)

- **Golden stability:** re-run without `--snapshot-update`; identical → trustworthy. Churn →
  fix via `canonicalize_asm` / `{basename,err}`, not by re-recording.
- **Coverage repeatability:** when a number is load-bearing, measure the same input twice
  *with* `--cov` into two files and diff (see `coverage-methodology.md`) — pass-count
  stability is not coverage-number stability.
- **No regression:** full `-m unit` 0 failed, pass-count only grows; 201 skipped unchanged.
- **Provenance:** every reported % saved with its exact command + scope to a `coverage-*.txt`
  receipt.
- **Separation:** the agent that generates an input never verifies it; the verifier defaults
  to skepticism and can reject.

---

## 8. Checklist (resume index)

Renumbered 2026-06-05 for the custom-minimal-design reframe (old P1 split into discovery
P1 + design P2; everything after shifts by one). Ready-to-run workflow plans: `WORKFLOW-SPECS.md`.

Prerequisite (separate source PR, NOT part of the ADD-ONLY campaign):
- [ ] **P0.5** land the `--cpu-only`/`--mock-gpu` switch PR — full brief in **`GPU-MOCK-PR.md`**
  (synthetic client perf metrics + per-arch `amd-smi`/`rocm_agent_enumerator` spoof). Gates
  **only** the P4 client/run-path rounds. Single-agent/goal-file work, not a fan-out workflow.

Stage 1 — coverage efficiency (minimal custom seed set + goldens):
- [x] **P0** validate method + inventory; commit `d6ae0b113d9`.
- [x] **P1 (first slice, 2026-06-05)** config harness path built + smoked (`_codegen/config_harness.py`, `emit_kernels_from_config`, CPU-only — independently re-smoked on `Tests/common/gemm/fp16_tn.yaml`); 29 curated inputs (gfx942/gfx950/gfx90a) measured isolated → `attribution-<arch>.json` (greedy set-cover marginal ranking). Commits `0d818440d0c`, `97a331c434f`. **Remaining P1b (widen):** corpus sample + single-parameter `ForkParameters` probes for parameter-level attribution.
- [x] **P2 (done, widened — option b, 2026-06-06)** minimal custom seed set, **multi-ProblemType per arch** (single-config-per-arch undershot at 30.53%; widened to a per-arch seed *set* spanning distinct high-yield families). 15 seed tests (gfx942/gfx950/gfx90a), each isolated-measured; union **whole-project ceiling = 35.89%** (21156/54491), recorded in `coverage/p2/ceiling-widened.txt`. Designed seed YAMLs live under `_codegen/data/test_data/_designed/**` (the `test_data` path keeps them out of `findConfigs` GPU auto-discovery — no `config_helpers.py` change). Full `-m unit` 0 failed (2528 passed/201 skipped). Commit `ec7524bd1be`.
- [x] **P3 (done, 2026-06-06)** goldens for the seed set; fast suite reproduces the P2 ceiling; baseline.

Stage 2 — coverage expansion:
- [x] **P4 round 1 (done, 2026-06-06)** cheap standalone/library-mgmt modules. Dynamic workflow `wf_f333c274-b78` (8 Haiku authors) → driver-run **deterministic methodology-A gate** (Part A bulk `-n4` + Part B `cpu_only_switch` isolated + `coverage combine` — avoids a pre-existing `problemTypeToEnum` xdist flake; see `coverage/p4/RANKING-AND-METHODOLOGY.md`). **68.85% → 69.21%** (+0.36 pts, 15723→15492 miss), 6 tests kept (verify_stinky/mergeLogic/benchclient/updatelib/gensummations/retunelib), 2 dropped as 0-marginal (BenchmarkProblems/LibraryLogic). Full `-m unit` 2620 passed / 0 failed / 201 skipped. Receipt `coverage/p4/master-baseline-R1.txt`.
- [x] **P4 round 2 (done, 2026-06-06)** codegen emit widening. Dynamic workflow `wf_ee11a589-d3b` (12 Sonnet designers, ForkParameters sweeps) → driver deterministic gate. **69.21% → 72.53%** (+3.32 pts, 15492→13470 miss, 2022 lines). 11 tests kept (per-target miss: WorkGroupMapping 364→32, GlobalWriteBatch 787→518, KWA 3987→3558, KW 1879→1664, StreamK 883→685, Subtile 313→155, ShiftVector 293→188, MAC/Activation/LRA/AsmStoreState + gsu/solution/kwconv add 62 via KWA/KW per leave-3-out). 1 dropped (localread, <cutoff). Fixed in-flight: 4 agents committed golden tests without seeding `.ambr` (seeded via `--snapshot-update`). Full `-m unit` 2631 passed / 0 failed / 201 skipped. Receipt `coverage/p4/master-baseline-R2.txt`.
- [x] **P4 round 3 (done, 2026-06-06)** deeper codegen + client/run path (switch). Dynamic workflow `wf_85433c15-e08` (14 Sonnet designers, all kept/stable) → driver deterministic gate (4-process: bulk `-n4` + cpu_only + ClientPath + TensileCreateLibraryRun isolated + combine). **72.53% → 75.35%** (+2.82 pts, 13470→11899 miss, 1571 lines). Standouts: LibraryLogic 535→142, LraTileAssignment 279→46, GlobalWriteBatch 518→362, StreamK 685→548, KWA 3558→3383, KW 1664→1555, TensileCreateLibrary/Run 275→197, ClientWriter 221→158. Full `-m unit` 2836 passed / 0 failed / 201 skipped. Receipt `coverage/p4/master-baseline-R3.txt`.
- [ ] **P4** expansion rounds 4..n (repeat until >=80% or no further gain). **NEXT = round 4** (remaining KWA 3383 / KW 1555 / Solution 1292 / LocalRead 497 / StreamK 548 / GlobalWriteBatch 362 / Activation 279 depth; gap to 80% ~4.65 pts ≈ ~2550 lines). Diminishing returns watch: round deltas +0.36 → +3.32 → +2.82.
- [ ] **P5** whole-project gate: >=80% or documented ceiling; `golden-governance.md`; `recommendations.md`.
- [ ] **P6** mutation validation; survivors → P4 backlog; tree clean.

---

## 9. Risk register (honest)

- **R1 — the existing-input ceiling is well below 80%.** Likely; that's *why* Stage 2 exists.
  Mitigation: custom test logic (min-parameter variation) to reach isolated branches; if a
  region is provably GPU-only/unreachable → `resistance.md` and the ceiling is documented,
  not forced.
- **R2 — an isolated run generates too many kernels and trips the rocisa limit.** Keep each
  run bounded; the limit is per-process accumulation. Split a too-large input.
- **R3 — concurrent `docker exec` contention.** Bounded by the 16-concurrent cap; lower the
  batch width or escalate to per-agent containers if it thrashes.
- **R4 — golden churn / nondeterminism.** `canonicalize_asm` + `{basename,err}`; Verify gates
  on two-run stability.
- **R5 — ADD-ONLY vs parallel coverage config.** Per-run `COVERAGE_FILE` + a new
  `--cov-config` file; never edit `pyproject.toml`.
- **R6 — combine writes to bare `.coverage`.** Always combine into an explicit
  `COVERAGE_FILE` (P0's Assemble clobbered a stray; fixed in `coverage-methodology.md`).
- **R7 — mistaking Stage 1 for the goal.** Coverage efficiency is the accelerator; the target
  is Stage 2's ≥80% (or a documented ceiling). Don't stop at the efficiency set.

---

## 10. Preconditions & provenance

- **Dynamic Workflows:** research preview, Claude Code v2.1.154+. Pre-allowlist `docker`,
  `git`, `coverage`, `pytest`. Run from the `tensilelite-coverage` worktree.
- **Container `tl-char`** (worktree at `/work`, rocisa baked); recreate per PLAN-80 §6 if
  cold; `invoke rocisa`; install `syrupy` if fresh.
- **Authored via** the `orchestration-plan` skill (template `reference.md` §5).
- **Companion docs:** `codegen-coverage-findings.html` (basis), `coverage-methodology.md` /
  `.html` (exact reproducible commands), `codegen-campaign-roadmap.html` (one-page status),
  `GLOSSARY-WIP.md` (terminology), `PLAN-80.md` / `CEILING-FINDINGS.md` / `DECISIONS.md`
  (prior campaign), `toolchain-inventory.md` (the mock point).

---

## 11. Progress log
(one line per completed phase/round: `<item> — before% → after% (N inputs/tests), combined baseline <file>, commit <sha>`)

- 2026-06-04 — Plan authored (Dynamic Workflows; isolated-runs + merge-as-measurement).
- 2026-06-04 — **P0 done.** Isolated-run harness + merge-as-measurement validated (merged
  59.18% / 65.93% across the two core codegen files from the 4-test de-risk subset > best
  single 53.88%); per-process rocisa footprint limitation confirmed; `toolchain-inventory.md`
  committed `d6ae0b113d9`. GO for the efficiency → expansion arc.
- 2026-06-04 — Plan re-framed to the two-stage arc (coverage efficiency for a fast baseline +
  goldens, then gap-driven coverage expansion to ≥80%): coverage is a selection signal; tests
  are added in Stage 2, gap-driven.
- 2026-06-05 — Terminology aligned to `GLOSSARY-WIP.md` (coverage efficiency/expansion,
  isolated runs, per-process rocisa footprint limitation, toolchain↔snapshot coupling, custom
  test logic, real component names).
- 2026-06-05 — **P1 first slice done** (dynamic workflow `wf_2b1ee1b2-270`, 35 agents). Built
  the BenchmarkProblems→Solutions CPU path (`config_harness.py`) + 29-input per-arch attribution.
  Per-arch union covered-lines: gfx942 21900, gfx950 22121, gfx90a 18587. **Set-cover signal:**
  one rich config dominates each arch (gfx942 BBS_BH_Bias_Act 19219 marginal; gfx950 StreamK_B8F8
  19547; gfx90a BBS 17875), small residue after — exactly the minimal-seed shape P2 wants. No
  regression: in-scope suite 2513 passed / 201 skipped / 0 failed (full `-m unit` blocked only by
  the pre-existing uncommitted `M Tests/common/config_helpers.py` collection error — not P1).
  Commits `--no-verify` (hipBLASLt is hook-opted-out; host black needs py≥3.10). Commits
  `0d818440d0c`, `97a331c434f`.
- 2026-06-06 — **P2 done (widened, option b).** First pass used one dominant config per arch;
  because the config harness reads only `BenchmarkProblems[0]` (one ProblemType/DataType per
  config), the combined ceiling was only **30.53%**. Widened to a per-arch seed *set* spanning the
  distinct high-yield families from P1 attribution (gfx942: dominant/F8N/GG/Grad/DB/HSS;
  gfx950: dominant/BBS/HHS/I8_GSU/HSS/DTL; gfx90a: dominant/DB/HHS — 15 seeds, cheapest-first:
  most reference shipped `Tests/common/**` configs). Union **whole-project ceiling 30.53% → 35.89%**
  (21156/54491, +2773 stmts), `coverage/p2/ceiling-widened.txt`. Blocker fixed add-only: designed
  seed YAMLs relocated under `_codegen/data/test_data/_designed/**` so `findConfigs` (which skips
  `test_data` paths) stops auto-running them through the GPU `Tensile.Tensile()` path — no
  `config_helpers.py` change. Full `-m unit` **2528 passed / 201 skipped / 0 failed**. Commit `ec7524bd1be`.
- 2026-06-06 — **P4 round 3 — 72.53% → 75.35% (+2.82 pts, 14 candidates / 15 test files), `coverage/p4/master-baseline-R3.txt`.**
  Deeper codegen + client/run path (switch-enabled) via dynamic workflow `wf_85433c15-e08` (14 Sonnet
  designers, all kept + two-run-stable; agents seeded their own `.ambr` this round). 1571 lines
  (13470→11899 miss). Standouts: LibraryLogic 535→142 (the R1 0-gain target, now driven via real
  analysis), LraTileAssignment 279→46, GlobalWriteBatch 518→362, StreamK 685→548, KWA 3558→3383,
  KW 1664→1555, TensileCreateLibrary/Run 275→197, ClientWriter 221→158, TensorDataMover 158→108.
  GSU.py stayed a dead-end on its named target (256, −1; the reduction arms 442-588 need a multi-kernel
  GSU path the single-config harness can't trigger — a P5 ceiling candidate) but its emit feeds KWA/KW.
  Gate extended to 4 processes (bulk + cpu_only + ClientPath + TensileCreateLibraryRun isolated) since the
  two new client-path suites also run full Tensile flows (problemTypeToEnum). 2836 passed / 0 failed / 201 skipped.
- 2026-06-06 — **P4 round 2 — 69.21% → 72.53% (+3.32 pts, 11 tests), `coverage/p4/master-baseline-R2.txt`.**
  Codegen emit widening via dynamic workflow `wf_ee11a589-d3b` (12 Sonnet ForkParameters designers;
  driver gate+commit). 2022 lines covered (15492→13470 miss). Biggest per-target wins: WorkGroupMapping
  364→32, GlobalWriteBatch 787→518, KernelWriterAssembly 3987→3558, KernelWriter 1879→1664, StreamK
  883→685, Subtile/GREmit 313→155, ShiftVector 293→188. The gsu/solution/kwconv sweeps hit ~0 of their
  NAMED target but add 62 whole-project lines via KWA/KW (proven by a leave-3-out gate 72.39% vs 72.53%),
  so kept. Dropped localread (<cutoff). Fixed: 4 designers committed golden tests without an `.ambr`
  (the codegen emit was byte-stable per two-run verify; seeded once with `--snapshot-update`). Deterministic
  2-process gate (bulk -n4 + cpu_only isolated). Full `-m unit` 2631 passed / 0 failed / 201 skipped.
- 2026-06-06 — **P4 round 1 — 68.85% → 69.21% (+0.36 pts, 6 tests), `coverage/p4/master-baseline-R1.txt`.**
  Cheap standalone/library-mgmt modules via dynamic workflow `wf_f333c274-b78` (8 Haiku authors;
  driver did the gate+commit, not an Assemble agent — the agent variant returned prematurely on a
  Monitor). Kept (per-file miss): verify_stinky 101→24, mergeLogic 133→89, benchclient 92→50,
  updatelib 97→63, gensummations 107→83, retunelib 93→81 (=231 lines). Dropped 2 as 0 whole-project
  marginal (BenchmarkProblems 111→111, LibraryLogic 535→535 — already full-suite-covered; remain R2/R3
  targets). Fixed in-flight: an ADD-ONLY violation (mergeLogic tests appended to an existing tracked
  file → relocated to a new `test_merge_logic_char.py`) and a stray duplicate `test_retunelib_char.py`
  at `Tests/unit/` root (import-mismatch). **Gate made deterministic** (Part A bulk `-n4` + Part B
  `cpu_only_switch` isolated + combine) to dodge a **pre-existing latent product flake**:
  `SolutionStructs/Problem.py:711 problemTypeToEnum()` mutates a ProblemType dict in place
  (DataType→int), intermittently breaking `cpu_only_end_to_end`'s `F32XdlMathOp.isSingle()` when xdist
  co-schedules them. Full `-m unit` 2620 passed / 0 failed / 201 skipped.
- 2026-06-06 — **P3 done.** Recorded order-invariant `{basename,err}` digest goldens (syrupy
  `__snapshots__/*.ambr`) for all 15 seed tests; each golden verified stable across two no-update
  runs. Seed suite reproduces the P2 ceiling exactly: union of the 15 `err==0`
  `.coverage.seedw_*` shards → whole-project TOTAL **35.89%** (21156/54491, **+0 vs P2**).
  Master-baseline `coverage/p3/master-baseline-p3.txt`; full `-m unit` **2543 passed / 201
  skipped / 0 failed** (728 snapshots passed), `coverage/p3/unit-gate.txt`. Commit `1d66d19`.
