# TensileLite codegen coverage — ready-to-run Dynamic-Workflow specs (P1–P6)

Companion to `PLAN-CODEGEN-WORKFLOW.md` (the narrative source of truth). One workflow
run per phase; each ends at its barrier + a **human checkpoint** (full `-m unit` +
baseline commit) that gates the next. Authored via `/orchestration-plan` (`reference.md`
§5). Decisions resolved 2026-06-05 are baked in: **custom-designed minimal logic**
(fewest kernels / most whole-project lines, marginal-yield cutoff, all archs); **two input
channels** (logic YAML + new BenchmarkProblems `ForkParameters` path); **whole-project
line-set** as the metric; **two stages kept**.

Each phase below has **Part A** (the task description you type to trigger the workflow —
Claude authors the script and you approve the card) and **Part B** (the script shape to
expect/review). Approve the card only if it matches the Part-B review checklist.

---

## Shared preconditions, env, and gotchas (apply to EVERY phase)

> Front-load these into **every** agent prompt — workflow subagents start fresh and see
> nothing from siblings (`reference.md` §5.2).

**Preconditions**
- Claude Code **2.1.162** (≥ 2.1.154 workflow floor ✓). Run from the `tensilelite-coverage`
  worktree.
- **Pre-allowlist before launch** (else a long run stalls on a permission prompt):
  `docker`, `git`, `coverage`, `pytest`. File edits already run in `acceptEdits`.
- Container **`tl-char`** is up (`docker ps --filter name=tl-char`); rocisa baked. If cold,
  recreate per `PLAN-80.md` §6; `invoke rocisa`; `pip install syrupy` if fresh.

**Canonical exec prefix** (paths are *inside* the container):
```bash
CON=tl-char
PROJ=/work/projects/hipblaslt/tensilelite
docker exec -e PYTHONPATH=$PROJ -e COVERAGE_FILE=$PROJ/.coverage.<ID> -w $PROJ $CON \
  pytest -p no:cacheprovider -m unit --cov=Tensile --cov-config=pyproject.toml -q <SUITE_OR_NODE>
```

**Hard rules / poka-yoke (carry verbatim into prompts):**
1. **ADD-ONLY.** New files only, under `Tensile/Tests/unit/characterization/`. Never modify
   or delete any existing file, incl. `pyproject.toml`. Parallel-coverage config goes in a
   **new** file via `--cov-config=<newfile>` / `COVERAGE_FILE`.
2. **`--cov=Tensile` is a PATH, never a dotted module** (`Tensile.x` → rocisa SIGABRT).
3. **One `COVERAGE_FILE` per shard**, changed together with the suite name (reuse silently
   overwrites). **`coverage combine` must target an explicit `COVERAGE_FILE`** (else it
   clobbers a bare `.coverage`). Use `--keep`.
4. **No `--snapshot-update`** except in the dedicated golden-recording stage.
5. **rocisa footprint is per-process** — one fresh `docker exec` process per input; keep each
   input bounded (cap kernels with `emit_kernels_from_logic(..., limit=N)` or a tight
   `ForkParameters`).
6. **NEVER push / no PR.** Only the serialized **Assemble** agent commits, with explicit-path
   `git add` (never `-A`).
7. Goldens are **root-owned** (generated in-container), stored order-invariant
   (`{basename, err}`) and `canonicalize_asm`'d.

**Reuse (don't reinvent):** `Tensile/Tests/unit/characterization/_codegen/`
(`codegen_harness.py` — `emit_kernels_from_logic`, `emit_helpers_from_logic`,
`canonicalize_asm`, cached `_toolchain()`; `matrix.py` — `digests_for_dir`; the
`characterization/conftest.py` `sys.path`+fixtures). Exact coverage commands:
`coverage-methodology.md`.

**Cost routing (every phase):** mechanical producer/measure agents → **Haiku**; design,
attribution-synthesis, verify (skeptical), and Assemble (set-cover / commit logic) → keep on
the **session model** (Opus/Sonnet). Check `/model` before a large run.

**Concurrency:** 16 concurrent / 1000 total per run. **Pipeline by default; barrier only**
where a stage needs all prior results (Assemble). **Never** have one agent emit a large
artifact in a single turn — split section-writers → cheap assemble.

**Slice-first (`args`):** every work-list is parameterizable so you can run **one arch**
(or one dir) first to price the run before the full fan-out.

---

## P0.5 — Prerequisite: the `--cpu-only` / `--mock-gpu` switch PR

> **Moved to its own document: [`GPU-MOCK-PR.md`](./GPU-MOCK-PR.md).** It is a separate,
> source-changing PR (single-agent / goal-file work, not a workflow), and the coverage campaign
> stays strictly ADD-ONLY. It **gates only the P4 expansion rounds that target the client /
> run path**; P1–P3 and all codegen + TensileCreateLibrary coverage do not depend on it.

---

## P1 — Discovery: parameter→coverage attribution + the config harness path

**Goal (one sentence):** stand up the add-only BenchmarkProblems→Solutions emit path and
produce a per-arch **attribution table** mapping each logic/`ForkParameters` value to the
**whole-project lines it uniquely toggles**, so P2 can design minimal-kernel sweeps from real
data.

**Unit of work:** one *(arch, parameter-probe)* — a single input (curated YAML, sampled
corpus YAML, or a one-parameter-varied config) emitted in an isolated process, its executed
whole-project line-set captured via `coverage json`.

**Done-criteria (provable in the workflow `return`):**
- The new harness path file (e.g. `_codegen/config_harness.py`) exists and a smoke test
  emits ≥1 kernel from a `Tests/common/**` config via
  `BenchmarkProcess`→`constructForkPermutations`→`_generateForkedSolutions`, `err==0`.
- An `attribution-<arch>.json` per arch: `{param, value} → {marginal_lines, total_lines}`
  over the **whole-project** line-set, ranked by marginal yield.
- Full `-m unit` shows 0 failed (pass-count only grows; 201 skipped unchanged).
- Receipts: every number saved with its exact command to a `coverage-*.txt`.

**Verification command (lands the signal in the transcript):**
```bash
docker exec -e COVERAGE_FILE=$PROJ/.coverage.combined -w $PROJ $CON coverage report | tail -1
docker exec -e PYTHONPATH=$PROJ -w $PROJ $CON pytest -p no:cacheprovider -m unit -q | tail -3
```

### Part A — task description
```text
ultracode: Build the add-only BenchmarkProblems->Solutions CPU emit path and attribute
whole-project line coverage to individual logic / ForkParameters values, per arch.

UNIT OF WORK: one (arch, parameter-probe) input emitted in its OWN docker exec process with
its OWN COVERAGE_FILE; capture its executed whole-project line-set via `coverage json`.
WORK-LIST (derive in a Boundaries phase, parameterizable via args = [arch,...]):
  - the 44 curated data/<arch>/*.yaml,
  - a small structured sample of the 2518 asm_full tuning YAMLs (a few per arch, diverse
    dtype/schedule/op), and
  - single-parameter probes enumerated from Tensile/Common/ValidParameters.py for the knobs
    most likely to gate codegen branches.

PHASES:
  1. Boundaries — enumerate the work-list; pick the arch slice from args (default: one arch).
  2. Harness — ONE agent writes _codegen/config_harness.py: load a BenchmarkProblems config
     -> BenchmarkProcess -> constructForkPermutations -> _generateForkedSolutions -> reuse
     generateKernelObjectsFromSolutions + the existing emit. Add a smoke test. ADD-ONLY.
  3. Measure (fan-out, <=16, Haiku) — one agent per input: isolated `docker exec ... pytest
     --cov=Tensile ... COVERAGE_FILE=.coverage.<id>`; then `coverage json` for that file;
     return the file path + the executed whole-project line count + per-file lines.
  4. Attribute (pipeline) — diff each probe's line-set against the baseline to get MARGINAL
     whole-project lines; rank per arch.
  5. Assemble (serialized, ONLY committer) — write attribution-<arch>.json + receipts;
     run full -m unit; atomic commits (explicit-path git add). Never push.

OUTPUT SCHEMA: every measure agent returns {input, channel, arch, cov_file, err,
  whole_project_lines, per_file_lines}. Attribution returns {arch, ranked:[{param,value,
  marginal_lines}]}.
DONE WHEN: config_harness smoke passes; attribution-<arch>.json committed for the run's
  archs; full -m unit 0 failed; receipts saved.
COST: run the slice args=[gfx942,gfx950,gfx90a] first. Measure agents -> Haiku;
  harness/attribute/assemble -> session model.
PERMISSIONS: docker, git, coverage, pytest pre-allowlisted.
```

### Part B — script shape to expect
```javascript
export const meta = {
  name: 'codegen-p1-attribution',
  description: 'Build config harness path + attribute whole-project lines to params, per arch',
  phases: [{title:'Boundaries'},{title:'Harness'},{title:'Measure'},{title:'Attribute'},{title:'Assemble'}],
}
const MEASURE = { type:'object', additionalProperties:false,
  required:['input','channel','arch','cov_file','err','whole_project_lines'],
  properties:{ input:{type:'string'}, channel:{enum:['logic','config','probe']},
    arch:{type:'string'}, cov_file:{type:'string'}, err:{type:'integer'},
    whole_project_lines:{type:'integer'},
    per_file_lines:{type:'object', additionalProperties:{type:'integer'}} } }
const ATTR = { type:'object', additionalProperties:false, required:['arch','ranked'],
  properties:{ arch:{type:'string'}, ranked:{type:'array', items:{ type:'object',
    additionalProperties:false, required:['param','value','marginal_lines'],
    properties:{ param:{type:'string'}, value:{type:'string'}, marginal_lines:{type:'integer'} } } } } }

phase('Boundaries')
const archs = args ?? ['gfx942','gfx950','gfx90a']   // slice: the three MI archs by default
const worklist = /* enumerate curated + sample + ValidParameters probes for `archs` */ []

phase('Harness')
await agent(`Write _codegen/config_harness.py (ADD-ONLY) driving a BenchmarkProblems config
through BenchmarkProcess -> constructForkPermutations -> _generateForkedSolutions -> emit,
plus a smoke test. Cite Tensile/BenchmarkProblems.py:185,447 and BenchmarkStructs.py.
Self-contained: <paste shared env + hard rules>.`, {label:'harness', phase:'Harness'})

phase('Measure')                                  // pipeline; per-input, no barrier
const measured = await pipeline(worklist,
  (u) => agent(`Isolated run for ${u.id}: <exec prefix, COVERAGE_FILE=.coverage.${u.id}>,
    then 'coverage json'. Return MEASURE schema. <hard rules>`,
    {label:`meas:${u.id}`, phase:'Measure', schema:MEASURE, model:'haiku'}),
)
phase('Attribute')
const attribution = await parallel(archs.map(a => () =>
  agent(`Compute MARGINAL whole-project lines per probe for ${a} from these line-sets;
    rank desc.\n`+JSON.stringify(measured.filter(m=>m&&m.arch===a)), {schema:ATTR, phase:'Attribute'})))

phase('Assemble')                                 // serialized, only committer
const report = await agent(`Write attribution-<arch>.json + coverage receipts; run full
  '-m unit'; atomic explicit-path commits; NEVER push.\n`+JSON.stringify(attribution.filter(Boolean)),
  {label:'assemble', phase:'Assemble'})
return { report, archs, inputs: measured.filter(Boolean).length }
```

---

## P2 — Design the minimal custom seed set (all archs)

**Goal (one sentence):** using P1's attribution, author a **minimal custom set of
parameter-varied inputs** (primarily BenchmarkProblems `ForkParameters` sweeps) that covers
the **most whole-project lines with the fewest kernels**, across **all archs**, and record
the empirical **ceiling**.

**Unit of work:** one *(arch)* design candidate — a custom config whose `ForkParameters`
encodes the high-yield values from P1; measured isolated; pruned by marginal-yield.

**Done-criteria (provable in `return`):**
- Committed minimal `test-logic/` (or `data/_designed/<arch>/`) seed set, all archs.
- Recorded **whole-project ceiling** (combined TOTAL) + each kept kernel's **marginal
  whole-project line yield** (one-line justification; every kept kernel ≥ the cutoff).
- A dropped-redundancy list (set-cover prune: kernels whose lines are already covered).
- Full `-m unit` 0 failed; receipts saved.

**Verification command:**
```bash
docker exec -e COVERAGE_FILE=$PROJ/.coverage.seed -w $PROJ $CON coverage report | tail -1   # the ceiling
```

### Part A — task description
```text
ultracode: Design the minimal custom seed set (parameter-varied ForkParameters sweeps, all
archs) that maximizes whole-project line coverage at minimum kernel count; record the ceiling.

UNIT OF WORK: one arch design candidate. INPUT: P1 attribution-<arch>.json (paste into each
prompt). RULE: marginal-yield cutoff — keep a parameter-varied kernel only while it adds
>= THRESHOLD new whole-project lines (state the threshold; default 5).
WORK-LIST (args = [arch,...]; default ALL archs): one per supported arch.

PHASES:
  1. Design (fan-out per arch) — author a custom BenchmarkProblems config (ForkParameters
     over the top-ranked P1 values) + the test that emits it; keep kernels minimal. ADD-ONLY.
  2. Measure (fan-out, Haiku) — isolated run per candidate; `coverage json`; whole-project lines.
  3. Prune+Combine (BARRIER->Assemble) — set-cover across all candidates' line-sets: drop
     kernels whose lines are already covered; `coverage combine` the kept set into
     .coverage.seed; report TOTAL = ceiling.
  4. Assemble (serialized, ONLY committer) — commit the seed set + ceiling receipt +
     per-kernel yield table; full -m unit; never push.

OUTPUT SCHEMA: design -> {arch, config_path, test_path, kernels:[basename]};
  measure -> {arch, cov_file, whole_project_lines, per_kernel_lines}.
DONE WHEN: seed set committed for ALL archs; ceiling recorded; every kept kernel >= cutoff;
  -m unit 0 failed.
COST: run ONE arch first. Measure -> Haiku; design/prune/assemble -> session model.
PERMISSIONS: docker, git, coverage, pytest.
```

### Part B — script shape
```javascript
export const meta = {
  name: 'codegen-p2-seed-design',
  description: 'Design minimal custom ForkParameters seed set (all archs), record the ceiling',
  phases: [{title:'Design'},{title:'Measure'},{title:'Assemble'}],
}
const DESIGN = { type:'object', additionalProperties:false,
  required:['arch','config_path','test_path','kernels'],
  properties:{ arch:{type:'string'}, config_path:{type:'string'}, test_path:{type:'string'},
    kernels:{type:'array', items:{type:'string'}} } }
const MEAS = { type:'object', additionalProperties:false,
  required:['arch','cov_file','whole_project_lines'],
  properties:{ arch:{type:'string'}, cov_file:{type:'string'},
    whole_project_lines:{type:'integer'},
    per_kernel_lines:{type:'object', additionalProperties:{type:'integer'}} } }

phase('Design')
const archs = args ?? ['gfx900','gfx906','gfx908','gfx90a','gfx942','gfx950','gfx1100','gfx1201','gfx1250']
const designed = await parallel(archs.map(a => () =>
  agent(`Design a minimal custom BenchmarkProblems config (ForkParameters over the top P1
    values) + emit test for ${a}. Marginal-yield cutoff THRESHOLD=5. ADD-ONLY. Paste
    attribution-${a}.json: <...>. <shared env + hard rules>`, {schema:DESIGN, phase:'Design'})))

phase('Measure')
const measured = await pipeline(designed.filter(Boolean),
  (d) => agent(`Isolated run of ${d.test_path}; 'coverage json'; return MEAS. <hard rules>`,
    {label:`meas:${d.arch}`, phase:'Measure', schema:MEAS, model:'haiku'}))

phase('Assemble')                                 // barrier: needs ALL line-sets for set-cover
const report = await agent(`Set-cover prune across these line-sets (drop kernels already
  covered); 'coverage combine' kept set into .coverage.seed; report TOTAL ceiling; write the
  per-kernel marginal-yield table + dropped-redundancy list; commit seed set (explicit-path);
  full -m unit; NEVER push.\n`+JSON.stringify(measured.filter(Boolean)),
  {label:'assemble', phase:'Assemble'})
return { report, archs, ceiling_from: '.coverage.seed' }
```

---

## P3 — Goldens for the seed set (the fast harness)

**Goal:** record stable goldens for every kept seed kernel so the seed set becomes a
committed, re-runnable suite that **reproduces the P2 ceiling fast**.

**Unit of work:** one kept seed *(arch, config)*.

**Done-criteria (provable in `return`):**
- Every kept input has a golden under `__snapshots__/`; a **second run without
  `--snapshot-update`** is byte-identical (two-run stability).
- Re-running only the seed suite reproduces the P2 whole-project ceiling (±0 lines).
- Full `-m unit` 0 failed; baseline `master-baseline-<N>.txt` committed.

**Verification command:**
```bash
docker exec -e PYTHONPATH=$PROJ -w $PROJ $CON pytest -p no:cacheprovider -m unit -q \
  Tensile/Tests/unit/characterization/_codegen | tail -3   # run twice, expect identical, 0 failed
```

### Part A — task description
```text
ultracode: Record + verify goldens for the P2 seed set so it reproduces the ceiling fast.

UNIT OF WORK: one seed (arch, config). PHASES:
  1. Record (fan-out, Haiku) — `--snapshot-update` in-container per seed; canonicalize_asm;
     order-invariant {basename, err}.
  2. Verify (pipeline, SKEPTICAL, separate agent) — re-run the SAME seed WITHOUT
     --snapshot-update twice; reject if goldens churn or any run differs; err!=0 is a valid
     pinned rejection, not a failure.
  3. Assemble (serialized) — confirm seed-only run == P2 ceiling; commit __snapshots__ +
     master-baseline-<N>.txt; full -m unit; never push.

OUTPUT SCHEMA: record -> {seed, snapshot_path}; verify -> {seed, stable:bool, reason}.
DONE WHEN: all seeds stable; seed-only coverage == P2 ceiling; -m unit 0 failed.
COST: record/verify -> Haiku/Sonnet; assemble -> session model. Run one arch first.
PERMISSIONS: docker, git, coverage, pytest.
```

### Part B — script shape
```javascript
export const meta = { name:'codegen-p3-goldens',
  description:'Record + two-run-verify goldens for the seed set; reproduce the ceiling',
  phases:[{title:'Record'},{title:'Verify'},{title:'Assemble'}] }
const REC = { type:'object', additionalProperties:false, required:['seed','snapshot_path'],
  properties:{ seed:{type:'string'}, snapshot_path:{type:'string'} } }
const VER = { type:'object', additionalProperties:false, required:['seed','stable','reason'],
  properties:{ seed:{type:'string'}, stable:{type:'boolean'}, reason:{type:'string'} } }

phase('Record'); phase('Verify')
const results = await pipeline(args ?? /* seeds from P2 */ [],
  (s) => agent(`Record golden for ${s} with --snapshot-update in-container. <rules>`,
    {label:`rec:${s}`, phase:'Record', schema:REC, model:'haiku'}),
  (rec, s) => agent(`Adversarially verify ${s}: re-run WITHOUT --snapshot-update TWICE;
    stable only if byte-identical both runs. Default stable=false on any doubt.`,
    {label:`ver:${s}`, phase:'Verify', schema:VER}))

phase('Assemble')
const report = await agent(`Confirm seed-only coverage == P2 ceiling; commit __snapshots__ +
  master-baseline; full -m unit; NEVER push.\n`+JSON.stringify(results.filter(Boolean)),
  {label:'assemble', phase:'Assemble'})
return { report, stable: results.filter(r=>r&&r.stable).length }
```

---

## P4 — Stage-2 expansion round (repeat until ≥80% or ceiling)

**One workflow run per round.** Gap-driven, not a fixed list. Independent work can proceed in
parallel across components (codegen modules / TensileCreateLibrary / Tensile(Lite) tuning) —
they touch different code. **Codegen + TensileCreateLibrary (cross-compile) rounds need no
mock** and start as soon as P3 lands. **Rounds whose gap-targets sit in the client / perf-run
path are gated on the P0.5 switch** — the Rank phase must tag each target
`needs_cpu_only_switch: true|false` and skip the gated ones until P0.5 is merged.

**Goal:** rank uncovered whole-project code on the current suite; close the top targets with
the **cheapest input** (config tweak → custom `ForkParameters` variation → new logic);
re-measure the **whole-project delta**; keep only kernels clearing the marginal-yield cutoff.

**Unit of work:** one *(gap-target)* → one cheapest-input candidate.

**Done-criteria (provable in `return`):**
- This round's combined whole-project TOTAL **> previous baseline** (or a round that adds
  ~nothing → the remainder is provably GPU-only/unreachable → `resistance.md`).
- New inputs goldened + two-run stable; full `-m unit` 0 failed; new
  `master-baseline-<N>.txt` committed; §11 progress log updated with the real delta.

**Measurement contract (MANDATORY — Assemble rebuilds the combined data from scratch; no stale
file may satisfy the gate).** The round's TOTAL is only trustworthy if `.coverage.combined` is
freshly rebuilt from the *exact* seed shards + every kept-input shard, then compared to the
**previous baseline file named in §11** (not an in-memory guess). Persist every kept input's
coverage data under a stable name (`.coverage.kept_<round>_<target>`) so each round's combine is
cumulative and reproducible.

**Verification command (the literal Assemble sequence — run, don't paraphrase):**
```bash
# 0. Identify the prior baseline FILE from PLAN §11 (the last 'master-baseline-*' it names).
PREV=work/tensilelite-characterization/coverage/p4/<prior-baseline>.txt   # read its TOTAL line
# 1. Rebuild combined data from scratch — never trust a pre-existing .coverage.combined.
docker exec -w $PROJ $CON rm -f $PROJ/.coverage.combined
# 2. Combine the EXACT seed shards + ALL kept-input shards (this round + prior rounds), --keep.
docker exec -e COVERAGE_FILE=$PROJ/.coverage.combined -w $PROJ $CON \
  coverage combine --keep $PROJ/.coverage.seedw_* $PROJ/.coverage.kept_*
# 3. Write the new baseline receipt and read its TOTAL.
docker exec -e COVERAGE_FILE=$PROJ/.coverage.combined -w $PROJ $CON coverage report \
  | tee work/tensilelite-characterization/coverage/p4/master-baseline-<N>.txt | tail -1
# 4. GATE: new TOTAL must STRICTLY EXCEED PREV's TOTAL. A non-increase => no-gain round:
#    record the unreachable/low-yield remainder in resistance.md; do NOT fake a gain.
```

### Part A — task description
```text
ultracode: Stage-2 expansion round N — close the highest-yield whole-project gaps with the
cheapest input, golden them, re-measure the delta against the PRIOR BASELINE FILE.

UNIT OF WORK: one gap-target -> one cheapest-input candidate (config tweak < custom
ForkParameters variation < new logic file). PHASES:
  1. Rank — run coverage term-missing on the CURRENT combined suite; SAVE the term-missing
     receipt to coverage/p4/term-missing-<N>.txt; rank uncovered code by missing-statement
     yield; pick the top K (args = K, default 8). For each target return the FULL schema below
     (file path + missing line ranges + est. yield + component + cheapest-channel rationale +
     needs_cpu_only_switch), and return the measured baseline_total + baseline_file it ranked
     against (so downstream agents and the gate share one provenance).
  2. GATE-EMPTY guard — if, after dropping needs_cpu_only_switch targets (args.haveSwitch
     false), the target list is EMPTY: write coverage/p4/round-<N>-deferred.txt
     ("all top-K gaps are P0.5/client-path gated; round DEFERRED, not a no-gain round"),
     update §11 with 'DEFERRED pending P0.5', and STOP. Do NOT run Assemble or claim a no-gain.
  3. Cheapest-input (fan-out, Haiku) — per target, author the cheapest add-only input that
     reaches it (per the §3 cost order); isolated measure into .coverage.kept_<N>_<target>;
     keep only if MEASURED marginal whole-project lines >= cutoff (report the receipt).
  4. Golden+Verify (pipeline, skeptical) — record + two-run-verify the kept inputs.
  5. Assemble (serialized) — run the MANDATORY measurement contract above: rm .coverage.combined,
     combine .coverage.seedw_* + .coverage.kept_* with --keep, write master-baseline-<N>.txt,
     compare TOTAL vs the prior baseline FILE; commit; update PLAN §8/§11; resistance.md for
     unreachable arms; never push.

OUTPUT SCHEMA: target -> {region, file_path, missing_ranges, est_missing_stmts, component,
  channel_hint:'config'|'fork'|'logic'|'other', channel_rationale, needs_cpu_only_switch};
  candidate -> {target, channel, input_path, cov_file, measured_marginal, receipt, kept:bool};
  golden -> {input_path, stable:bool}.
DONE WHEN: combined TOTAL (freshly rebuilt) > prior baseline FILE's TOTAL (or documented no-gain
  -> resistance.md; or DEFERRED via the empty-guard); goldens stable; -m unit 0 failed;
  master-baseline-<N>.txt + §11 updated.
COST: run on the top-K slice first. Candidate/golden -> Haiku/Sonnet; rank/assemble -> session model.
PERMISSIONS: docker, git, coverage, pytest.
```

### Part B — script shape
```javascript
export const meta = { name:'codegen-p4-expansion-round',
  description:'Rank whole-project gaps -> cheapest input -> golden -> re-measure delta',
  phases:[{title:'Rank'},{title:'Cheapest-input'},{title:'Golden'},{title:'Assemble'}] }
// Rank targets carry everything a cheapest-input agent needs to choose a channel safely.
const TARGET = { type:'object', additionalProperties:false,
  required:['region','file_path','missing_ranges','est_missing_stmts','component',
            'channel_hint','channel_rationale','needs_cpu_only_switch'],
  properties:{ region:{type:'string'}, file_path:{type:'string'},
    missing_ranges:{type:'string'},          // e.g. "412-455,610-633" from term-missing
    est_missing_stmts:{type:'integer'}, component:{enum:['codegen','createlibrary','tuning','io','other']},
    channel_hint:{enum:['config','fork','logic','other']}, channel_rationale:{type:'string'},
    needs_cpu_only_switch:{type:'boolean'} } }
const RANK = { type:'object', additionalProperties:false,
  required:['baseline_total','baseline_file','receipt_path','targets'],
  properties:{ baseline_total:{type:'number'}, baseline_file:{type:'string'},
    receipt_path:{type:'string'}, targets:{type:'array', items:TARGET} } }
const CAND = { type:'object', additionalProperties:false,
  required:['target','channel','input_path','cov_file','measured_marginal','receipt','kept'],
  properties:{ target:{type:'string'}, channel:{enum:['config','fork','logic']},
    input_path:{type:'string'}, cov_file:{type:'string'},   // .coverage.kept_<N>_<target>
    measured_marginal:{type:'integer'}, receipt:{type:'string'}, kept:{type:'boolean'} } }
const GOLD = { type:'object', additionalProperties:false, required:['input_path','stable'],
  properties:{ input_path:{type:'string'}, stable:{type:'boolean'} } }

phase('Rank')
const N = args?.round ?? 1
const K = args?.K ?? 8
const ranked = await agent(`Rebuild the current combined suite, run coverage term-missing, and
  SAVE the receipt to coverage/p4/term-missing-${N}.txt. Return RANK: the measured baseline_total
  and the baseline_file you compared to (the last master-baseline named in PLAN §11), the
  receipt_path, and the top ${K} uncovered regions by missing-statement yield — each with
  file_path, missing_ranges, est_missing_stmts, component, channel_hint + channel_rationale
  (why config<fork<logic is cheapest for THIS region), and needs_cpu_only_switch (true iff it
  lives in the client/perf-run or device/ISA-probe path). <env>`,
  {phase:'Rank', schema:RANK})

// Drop switch-gated targets until P0.5 is merged (set args.haveSwitch=true once it is).
const targets = ranked.targets.filter(t => args?.haveSwitch || !t.needs_cpu_only_switch)

// EMPTY-GUARD: if every top-K gap is gated, this is NOT a no-gain round — it is DEFERRED.
if (targets.length === 0) {
  const note = await agent(`All top-${K} gaps are P0.5/client-path gated (haveSwitch=false).
    Write coverage/p4/round-${N}-deferred.txt explaining the deferral with the gated regions
    from this RANK, append a PLAN §11 line 'P4 round ${N} DEFERRED pending P0.5', commit those
    two files only (explicit-path, --no-verify), NEVER push. Do NOT touch coverage data or claim
    a coverage attempt.\n`+JSON.stringify(ranked), {label:'defer', phase:'Rank'})
  return { deferred:true, round:N, note, gated:ranked.targets.length }
}

phase('Cheapest-input'); phase('Golden')
const worked = await pipeline(targets,
  (t) => agent(`Author the CHEAPEST add-only input reaching ${t.region} (${t.file_path}
    ${t.missing_ranges}; suggested channel ${t.channel_hint} — ${t.channel_rationale}); follow
    config<fork<logic cost order. Isolated-measure into COVERAGE_FILE=.coverage.kept_${N}_<target>;
    keep ONLY if MEASURED marginal whole-project lines >= cutoff (save the receipt). Report the
    real measured_marginal, never an estimate. <rules>`,
    {label:`cand:${t.region}`, phase:'Cheapest-input', schema:CAND, model:'haiku'}),
  (c) => c && c.kept
    ? agent(`Record + two-run verify golden for ${c.input_path}. Default stable=false on doubt.`,
        {label:`gold`, phase:'Golden', schema:GOLD})
    : null)

phase('Assemble')                                 // barrier
const report = await agent(`Run the MANDATORY measurement contract (stale-proof):
  1) rm .coverage.combined; 2) 'coverage combine --keep' .coverage.seedw_* + .coverage.kept_*
  (seeds + ALL rounds' kept shards); 3) write coverage/p4/master-baseline-${N}.txt; 4) GATE: its
  TOTAL must STRICTLY EXCEED the prior baseline FILE ${ranked.baseline_file} (TOTAL
  ${ranked.baseline_total}) — a non-increase is a no-gain round (resistance.md, with file:line
  evidence for unreachable arms), NOT a faked gain. Commit kept inputs+goldens+baseline+receipts
  (explicit-path, --no-verify); update PLAN §8/§11 with the REAL delta; NEVER push.\n`
  +JSON.stringify({baseline:ranked.baseline_file, prev_total:ranked.baseline_total,
                   kept:worked.filter(Boolean)}), {label:'assemble', phase:'Assemble'})
return { report, round:N, kept: worked.filter(Boolean).length, prev_total: ranked.baseline_total }
```

---

## P5 — Whole-project gate + golden governance + recommendations

**Goal:** the **≥80% decision point** — `coverage combine` across all data → assert
whole-project ≥80%, else record the honest ceiling + evidence.

**Unit of work:** the whole combined suite (single deterministic gate) + governance docs.

**Done-criteria (provable in `return`):**
- Combined `master-baseline-<N>.txt` whole-project TOTAL ≥ 80% **or** `CEILING-FINDINGS.md`
  with per-region evidence that the remainder is GPU-only/unreachable.
- `golden-governance.md`: goldens keyed by **(architecture, compiler version)**; stable arch →
  one golden; evolving arch → keep N compiler generations, retire oldest.
- `recommendations.md`; full `-m unit` 0 failed.

**Verification command:**
```bash
docker exec -e COVERAGE_FILE=$PROJ/.coverage.combined -w $PROJ $CON coverage report | tail -1
```

### Part A — task description
```text
ultracode: Whole-project gate — combine all coverage data, assert >=80% or document the
honest ceiling; write golden governance + recommendations.

UNIT OF WORK: the combined suite + governance docs. PHASES:
  1. Gate — `coverage combine` ALL data into .coverage.combined; report whole-project TOTAL;
     if <80%, run term-missing and classify each remaining region reachable/GPU-only/unreachable.
  2. Govern (parallel section-writers -> cheap assemble) — golden-governance.md (key by
     arch+compiler version) and recommendations.md, each authored by its own agent into its
     own file, then concatenated. (Never one agent emitting a big artifact.)
  3. Assemble (serialized) — commit master-baseline-<N>.txt + CEILING-FINDINGS.md (if <80%)
     + governance docs; full -m unit; never push.

OUTPUT SCHEMA: gate -> {total_pct, met:bool, remaining:[{region,classification}]}.
DONE WHEN: TOTAL >=80% committed OR CEILING-FINDINGS.md with per-region evidence; governance
  + recommendations committed; -m unit 0 failed.
COST: gate/assemble -> session model; section-writers -> Sonnet.
PERMISSIONS: docker, git, coverage, pytest.
```

### Part B — script shape
```javascript
export const meta = { name:'codegen-p5-gate',
  description:'Whole-project >=80% gate or documented ceiling; governance + recommendations',
  phases:[{title:'Gate'},{title:'Govern'},{title:'Assemble'}] }
const GATE = { type:'object', additionalProperties:false, required:['total_pct','met'],
  properties:{ total_pct:{type:'number'}, met:{type:'boolean'},
    remaining:{type:'array', items:{ type:'object', additionalProperties:false,
      required:['region','classification'],
      properties:{ region:{type:'string'},
        classification:{enum:['reachable','gpu-only','unreachable']} } } } } }

phase('Gate')
const gate = await agent(`coverage combine ALL data -> .coverage.combined; report whole-project
  TOTAL; if <80% run term-missing and classify each remaining region. <env>`,
  {phase:'Gate', schema:GATE})
phase('Govern')                                   // section-writers -> own files
await parallel([
  () => agent(`Write golden-governance.md (key goldens by arch+compiler version; stable arch=1
    golden; evolving arch=keep N generations).`, {label:'gov', phase:'Govern'}),
  () => agent(`Write recommendations.md from the campaign findings.`, {label:'rec', phase:'Govern'}),
])
phase('Assemble')
const report = await agent(`Commit master-baseline + (CEILING-FINDINGS.md if not met) +
  governance docs; full -m unit; NEVER push.\n`+JSON.stringify(gate),
  {label:'assemble', phase:'Assemble'})
return { report, total_pct: gate.total_pct, met: gate.met }
```

---

## P6 — Mutation validation

**Goal:** confirm the suite actually *catches* regressions — mutate codegen source, run only
the coverage-selected subset each mutant touches, classify killed/survived; **surviving
mutants → new P4 targets**. No source left mutated.

**Unit of work:** one *(mutant patch)*.

> **CONTROLLED ADD-ONLY EXCEPTION (mutation is the one allowed source edit).** P6 *does* edit
> source — but only as **temporary, reverted, never-committed** mutants. This is reconciled with
> the campaign's "source changes forbidden" rule (PLAN §1.5) as follows:
> - **Isolation primitive:** `agent(..., {isolation:'worktree'})` **is** a supported workflow
>   option — it runs that agent in a **fresh temporary git worktree** (auto-removed afterward if
>   unchanged), so parallel mutants cannot collide and the campaign worktree is never touched.
> - **Belt-and-suspenders (do not rely on auto-cleanup alone):** each mutant agent must (a)
>   record its `worktree_path` and `base_commit`, (b) **revert** its own patch (`git checkout --`
>   / `git stash`/restore) after classifying, and (c) report `reverted` + the post-run
>   `git status --porcelain` of its worktree.
> - **Serialized Cleanup/Verify phase** runs before Assemble: it asserts every mutant worktree is
>   clean/removed AND that the **campaign worktree** (`tensilelite-coverage`) `git status
>   --porcelain` shows **no** source mutation. If any leak is found it STOPS (no commit).
> - Assemble commits **only the report + p4-backlog.md** (add-only), never any source.

**Done-criteria (provable in `return`):**
- Per-mutant killed/survived table; each row carries `worktree_path`, `base_commit`, `reverted`,
  `post_status`; survivors written to a P4 backlog file.
- The serialized Cleanup/Verify phase confirms the **campaign worktree** `git status --porcelain`
  prints nothing source-related (no mutation leaked); full `-m unit` 0 failed.

**Verification command:**
```bash
git -C /home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-coverage status --porcelain | head
```

### Part A — task description
```text
ultracode: Mutation-validate the codegen suite — one mutant per agent, run only the
coverage-selected subset it touches, classify, feed survivors back to P4.

UNIT OF WORK: one mutant patch (apply -> run selected subset -> classify -> REVERT). Each agent
runs in its OWN temporary git worktree via agent(...,{isolation:'worktree'}) so parallel mutants
never collide and the campaign worktree is never touched.
PHASES:
  1. Enumerate — generate mutant patches over the codegen surface (boundary/operator/return
     mutations); args = max mutants (default 30).
  2. Run (fan-out, isolation:'worktree', Haiku) — in its OWN worktree: record worktree_path +
     base_commit; apply mutant; run ONLY the coverage-selected subset touching the mutated lines;
     classify killed (a test failed) / survived; REVERT the patch; report reverted + post_status
     (its worktree 'git status --porcelain').
  3. Cleanup/Verify (SERIALIZED, before Assemble) — assert every mutant reverted==true and its
     post_status is empty; assert the CAMPAIGN worktree 'git status --porcelain' shows no source
     mutation. If any leak: STOP, report, DO NOT commit.
  4. Assemble (serialized) — table + survivors->p4-backlog.md; commit the REPORT ONLY (add-only,
     explicit-path); never commit source; never push.

OUTPUT SCHEMA: run -> {mutant, file, line, status:'killed'|'survived', subset_size,
  worktree_path, base_commit, reverted:bool, post_status}.
DONE WHEN: every mutant classified + reverted; Cleanup/Verify confirms the campaign worktree
  clean of mutation; survivors in p4-backlog.md; -m unit 0 failed.
COST: run -> Haiku; enumerate/cleanup/assemble -> session model. Slice: args caps mutant count.
PERMISSIONS: docker, git, coverage, pytest.
```

### Part B — script shape
```javascript
export const meta = { name:'codegen-p6-mutation',
  description:'One mutant per worktree-isolated agent; classify; cleanup-verify; survivors -> P4 backlog',
  phases:[{title:'Enumerate'},{title:'Run'},{title:'Cleanup'},{title:'Assemble'}] }
const RUN = { type:'object', additionalProperties:false,
  required:['mutant','file','line','status','subset_size','worktree_path','base_commit','reverted','post_status'],
  properties:{ mutant:{type:'string'}, file:{type:'string'}, line:{type:'integer'},
    status:{enum:['killed','survived']}, subset_size:{type:'integer'},
    worktree_path:{type:'string'}, base_commit:{type:'string'},
    reverted:{type:'boolean'}, post_status:{type:'string'} } }   // its worktree git status --porcelain
const CLEAN = { type:'object', additionalProperties:false, required:['campaign_clean','leaks'],
  properties:{ campaign_clean:{type:'boolean'}, leaks:{type:'array', items:{type:'string'}} } }

phase('Enumerate')
const N = args ?? 30
const mutants = await agent(`Generate up to ${N} mutant patches over the codegen surface
  (KernelWriterAssembly/KernelWriter/Components/Asm*). Return [{id,file,line,patch}].`,
  {phase:'Enumerate', schema:{type:'object', required:['mutants'], additionalProperties:false,
    properties:{mutants:{type:'array', items:{type:'object'}}}}})

phase('Run')                                      // isolation:'worktree' = fresh temp worktree per mutant
const ran = await parallel(mutants.mutants.map(m => () =>
  agent(`In THIS (isolated) worktree: record worktree_path + base_commit (git rev-parse HEAD);
    apply mutant ${m.id} at ${m.file}:${m.line}; run ONLY the coverage-selected subset touching
    those lines; classify killed/survived; then REVERT the patch and report reverted + post_status
    ('git status --porcelain' of this worktree, expected empty). <rules>`,
    {label:`mut:${m.id}`, phase:'Run', schema:RUN, model:'haiku', isolation:'worktree'})))

phase('Cleanup')                                  // SERIALIZED guard before any commit
const clean = await agent(`Verify no mutation leaked. For the runs below assert every reverted==true
  and post_status empty. Then assert the CAMPAIGN worktree
  (/home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-coverage) 'git status
  --porcelain' shows NO source (Tensile/*.py) modification. Return CLEAN{campaign_clean,leaks}.
  If campaign_clean is false, the caller STOPS.\n`+JSON.stringify(ran.filter(Boolean)),
  {phase:'Cleanup', schema:CLEAN})
if (!clean.campaign_clean) return { stopped:'mutation leaked into campaign worktree', leaks: clean.leaks }

phase('Assemble')
const report = await agent(`Write the killed/survived table + survivors->p4-backlog.md; commit the
  REPORT ONLY (add-only, explicit-path, --no-verify); commit NO source; NEVER push.\n`
  +JSON.stringify(ran.filter(Boolean)), {label:'assemble', phase:'Assemble'})
return { report, survived: ran.filter(r=>r&&r.status==='survived').length }
```

---

## Per-card review checklist (apply before approving ANY phase's card)

- [ ] Phases named & meaningful; match `meta.phases`.
- [ ] **Every consumed stage has a `schema`** (no silent garbage downstream).
- [ ] Prompts **self-contained** — shared env + hard rules + paths pasted in (agents see
      nothing from siblings).
- [ ] An **independent verify** stage exists where trust matters (P3/P4 golden two-run;
      P1/P2 measurement separated from design).
- [ ] **Assemble is serialized and the only committer**; explicit-path `git add`; **no push**.
- [ ] Cheap/mechanical stages routed to **Haiku**; design/verify/assemble on the session model.
- [ ] Work-list **parameterized via `args`** so you can run one arch / top-K first.
- [ ] `docker`, `git`, `coverage`, `pytest` **pre-allowlisted**.
- [ ] No single agent emits a large artifact in one turn (P5 governance = section-writers).
- [ ] Poka-yoke present: `--cov=Tensile` is a PATH; one `COVERAGE_FILE` per shard; combine
      targets an explicit `COVERAGE_FILE` with `--keep`; no stray `--snapshot-update`.
- [ ] **Script is copy-safe JS:** NO inner backticks inside `agent(\`...\`)` template literals —
      write command names as plain quotes (`'coverage json'`, `'-m unit'`, `'git status
      --porcelain'`), never `` `coverage json` `` (an inner backtick closes the template literal
      and breaks the script). The Part-B examples here follow this; keep it when authoring.
- [ ] **Measurement is stale-proof (P4/P5):** Assemble rebuilds `.coverage.combined` from scratch
      (`rm` then `coverage combine --keep` of the exact seed + kept shards), writes
      `master-baseline-<N>.txt`, and compares its TOTAL to the **prior baseline FILE named in
      §11** — never to an in-memory or pre-existing combined file.
- [ ] **Empty-target guard (P4):** if all top-K targets are switch-gated, the round writes a
      `round-<N>-deferred.txt` checkpoint and returns `deferred` — it is NOT run as / logged as a
      no-gain coverage attempt.
- [ ] **Mutation is contained (P6):** mutants run under `isolation:'worktree'`, self-revert and
      report `reverted`+`post_status`; a serialized Cleanup phase asserts the campaign worktree
      has no source mutation before Assemble commits (report-only).
