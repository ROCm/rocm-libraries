# Production Mutation Testing for TensileLite Python — Dynamic-Workflow Plan

Converts the current **bounded 6-mutant spot-check** (`mutants.md`) into a
reproducible, **mutmut-driven** mutation-testing pipeline that, *per module
slice*, mutates covered lines, **triages every survivor** into actionable
buckets, and **sharpens the tests** for the survivors that matter — orchestrated
as a Claude Code **dynamic workflow** (research preview; `ultracode`).

Grounded in the survey `mutants.pdf` (primary engine = mutmut; Cosmic Ray only
for distributed escalation; "good enough to adopt, not to automate blindly";
Google-style incremental CI, not a global threshold; AI only inside a verified
loop). Revised against the review in `mutants-findings.md` (all 7 findings; see
the **Review findings addressed** table at the end).

ASSUMES: the `tl-char` container + worktree env is built and validated
(`env/README.md`); the `unit` suite is green (~2072 passed); `rocisa` is built
once in-container; the serial apply→run→revert harness pattern exists
(`wf/p6-mutation.sh`, `wf/p7-survivor-kill.sh`) and is **generalized** by this
plan into a manifest-driven runner (`wf/mutmut-verify.sh`, Phase 0).

> **Tool layer:** Dynamic Workflow (reference.md §5). Precondition: Claude Code
> **v2.1.154+**, trigger `ultracode`. Research preview — script grammar may drift.

---

## THE concurrency rule (load-bearing — every phase obeys it)

> **A stage may fan out (parallel/pipeline) ONLY if it is read-only on tracked
> files, or writes NEW files exclusively. Any stage that mutates tracked source
> — running mutmut, applying a mutant, or editing a source line for a pragma —
> must be a SINGLE SERIAL ACTOR.** A prompt that says "serial" is not a lock;
> serialization must be structural (one agent / one script loop).

This rule is what the review's findings #1, #2, #7 all reduce to. The pipeline
below is built around it: only **triage** and **equivalence reasoning** fan out;
**mutmut execution**, **kill-proof application**, and **pragma application** are
each one serial actor.

All host `git` commands run with `git -C "$SRC"` (or `cd "$SRC"` first), where
`SRC=projects/hipblaslt/tensilelite` — never bare from the worktree root, or
pathspecs like `Tensile/*.py` match nothing and report **false-clean** (finding #3,
verified).

---

## Locked decisions

| # | Decision | Resolution |
|---|----------|------------|
| D1 | **Done-criteria** | **GATE + TRIAGE**, gate is **incremental (Google-style), not a global %.** Per slice: every survivor triaged into 5 buckets **and** every test-fixable survivor killed (proven FAIL@mutant / PASS@clean). Score gate is report-only on the pilot → fail-on-regression → small per-slice floor; never a day-one global threshold. |
| D2 | **Engine** | **`mutmut`** (report primary). Cosmic Ray = documented escalation only. |
| D3 | **Isolation / parallelism** | mutmut owns per-mutant isolation; **single instance, serial** execution in the one `tl-char` container. Fan-out is analysis-only (triage + equivalence reasoning). See THE concurrency rule. |
| D4 | **First slice** | **Cheap pure-Python first** — `Common/Utilities.py` + `TensileLogic` validators (3–8 deterministic modules). |

### TBD slots (one numeric value left)

- **`SLICE_FLOOR`** — per-slice mutation-score floor introduced at **CI phase 3**
  (after noise is stable). NOT applied to the pilot. Derive from the pilot's
  observed actionable-survivor rate, not a round number.

### Escalation trigger (mutmut → Cosmic Ray)

Switch only if (a) a slice's population pass exceeds the wall-clock budget and
mutmut's incremental cache can't amortize it, or (b) you need distributed
workers. Cosmic Ray then reuses `coverage.db` + `mutant_select.py` (its worker
model needs a fast per-mutant test command); under mutmut those are **unused**.

---

## GOAL

A re-runnable dynamic workflow that takes a **module slice** and produces a
**fully-triaged, test-sharpened mutation result**: covered-line mutants run by a
single mutmut pass, every survivor classified into an actionable bucket, every
test-fixable survivor killed by an add-only test proven FAIL@mutant / PASS@clean,
proposed pragmas applied in one serial pass — with score + noise metrics reported
(gated incrementally, never globally).

**Achieved when** (per slice — each item provable by output an agent surfaces):

- `mutmut results` shows the full slice run (N mutants: killed / survived /
  skipped / suspicious);
- the survivor ledger shows **0 untriaged** survivors, each with a bucket
  ∈ {missing-assertion-strength, wrong-granularity, equivalent, intentionally-unhelpful, design-smell};
- every **missing-assertion-strength** / **wrong-granularity** survivor has an
  add-only test whose verify row reads `base_rc=0 mut_rc!=0 revert=ok` (KILLED),
  from the **single serial** `wf/mutmut-verify.sh` run;
- every **intentionally-unhelpful** survivor has a justified `# pragma: no mutate`
  applied in the **single serial** pragma stage; every **design-smell** survivor
  has a refactor note;
- the slice suite is green after the new tests + pragmas:
  `docker exec -w $PROJ tl-char pytest -m unit <slice test dirs>` → `0 failed`;
- `git -C "$SRC" status --porcelain -- 'Tensile/*.py' 'Tensile/**/*.py' | grep -v config_helpers.py`
  shows only the add-only test files + the justified pragma lines (no leak);
- `mutation-report.json` (schema-valid: score + noise metrics) + `survivor-ledger.md`
  + `recommendations.md` exist (each proven by an agent `cat`-ing a header line);
- **pilot (slice-1) is report-only** — no score gate fails it.

---

## UNIT OF WORK + work-list

- **Analysis fan-out unit** = **one survivor mutant** (read-only / new-file-only).
- **Engine input unit** = **one module slice** (3–8 disjoint deterministic files),
  run as ONE mutmut pass (not one-per-module — finding #5).

### slice-1 (pinned — the de-risking pilot, report-only)

| Module | LOC | Char test dir | Notes |
|--------|-----|---------------|-------|
| `Tensile/Common/Utilities.py` | 415 | `…/characterization/CommonUtilities` | small, deterministic, covered |
| `Tensile/TensileLogic/ValidChipId.py` | 207 | `…/characterization/TensileLogic` | chip-id table/boundary logic |
| `Tensile/TensileLogic/ValidMatrixInstruction.py` | 72 | `…/characterization/TensileLogic` | validator predicates |
| `Tensile/TensileLogic/ValidWorkGroup.py` | 47 | `…/characterization/TensileLogic` | tiny — fastest first |
| `Tensile/TensileLogic/ValidWorkGroupMappingXCC.py` | 90 | `…/characterization/TensileLogic` | validator predicates |

> **SKIP / DEFER:** `SolutionStructs/Solution.py` (5230 LOC); the codegen/KWA
> region (coarse `{basename,err}` goldens under multiprocessing → operand mutants
> slip, `golden-governance.md`); client-build / GPU / broader `common` tests.

### Scaling slices (smallest-first, after slice-1 green)

`Common/ValidParameters.py` → `CustomYamlLoader.py` → `BenchmarkSplitter.py` →
`Configuration.py` → `LibraryIO.py` → `SolutionStructs/{Utilities,Naming,LdsPadding,Problem}.py`
→ (last) `Solution.py`. **One slice per workflow invocation** (slice = `args`).

---

## PHASES (prep → execute → triage → verify → synthesize)

Pipeline-by-default; barrier only where a stage needs ALL prior results.
Concurrency capped at **16**; never emit a large artifact in one agent turn
(split section-writers + cheap assemble). **THE concurrency rule governs which
stages may fan out.**

### Phase 0 — Prep (serial; one-time per slice)

1. **Baseline green**: `docker exec -w $PROJ tl-char pytest -m unit <slice test dirs>`.
2. **mutmut present**: `pip show mutmut`; install if absent.
3. **Coverage data** for covered-line filtering: a standard `coverage run` over
   the slice's tests (NOT the custom per-test `coverage.db`).
   - **Gotcha:** `--cov` takes a directory **PATH**, never a dotted module
     (rocisa double-import → nanobind dup-key → SIGABRT).
4. **mutmut config + tox env** (committed deliverable):
   - `[mutmut]` in `setup.cfg`/`pyproject.toml`: `paths_to_mutate` = the **whole
     slice** (all modules, one pass); `tests_dir` = slice char dirs;
     **`max_stack_depth`** set (so a broad char test isn't "the relevant test"
     for everything); covered-lines-only; type-check filter **deferred**.
   - a `tox` **`mutation-unit`** env next to `coverage-unit`.
5. **Generic verifier** `wf/mutmut-verify.sh` (committed deliverable — finding #4):
   a **manifest-driven, serial** runner generic over any mutmut survivor. Per
   manifest row it: asserts clean → runs the test node on clean source (expect
   pass) → **materializes the mutant via `mutmut apply <mutant_id>`** (NOT
   hardcoded search/replace) → runs the test node (expect fail) →
   `git -C "$SRC" checkout -- <file>` → asserts clean. Trap-reverts every target
   on exit. Emits one kill-matrix row per manifest row.
   - Manifest row (TSV): `mutant_id | file | apply_method(mutmut_apply|diff:<path>) | test_node | expect_clean_rc | expect_mutant_rc | revert_assert`.

### Phase 1 — Execute (SERIAL; ONE agent — fixes #1, #5)

A single agent runs **one slice-level `mutmut run`** (config from Phase 0; covered
lines; stack-depth limited), then `mutmut results`, then `mutmut show <id>` per
survivor. It **groups survivors by module** by partitioning on the file path.
One instance, one cache, serial — no `parallel`, no per-module agents. mutmut
streams per-mutant progress, so the long single command keeps the turn alive.
Returns `{per-module counts, survivors[]}` (each survivor carries its diff).

### Phase 2 — Triage (PARALLEL; one agent per survivor — PROPOSAL-ONLY)

Read-only on source; writes **only NEW files**. No source mutation here
(finding #7). Each agent classifies into the report's 5 buckets and emits a
**proposal**, never an in-place edit:

| Bucket | Meaning | Proposal it emits |
|--------|---------|-------------------|
| **missing-assertion-strength** | reaches code, doesn't pin boundary/output | write a DISTINCT NEW test file + a VERIFY manifest row |
| **wrong-granularity** | broad char test hits it indirectly | write a focused DISTINCT NEW test file + a VERIFY manifest row |
| **equivalent / near-equivalent** | mutant not worth fighting | emit an equivalence claim (audited in 3a) |
| **intentionally-unhelpful** | logging/telemetry/format noise | emit a **proposed pragma edit** (file, line, exact original line) — DO NOT edit source |
| **design-smell** | killer test harder than refactor | emit a refactor note |

New test files (`test_mut_<mutant_id>_char.py`) are collision-free (distinct
names). This is the report's verified helper loop: original + mutant + tests +
the invariant that should hold → one test that fails on the mutant, passes clean.

### Phase 3 — Verify (concurrency-correct split — fixes #2, #4, #7)

- **3a Equivalence audit (PARALLEL; read-only reasoning):** a skeptical second
  agent re-derives each equivalent / intentionally-unhelpful claim; **downgrade**
  to "needs-test" unless airtight (routes back to authoring a test row). No tree
  writes. The model never *decides* equivalence — it argues, and unconfirmed
  claims are downgraded.
- **3b Kill-proof runner (SERIAL; ONE agent):** feeds the **full VERIFY manifest**
  (all authored-test rows) to `wf/mutmut-verify.sh` in one invocation. The script
  applies mutants **one at a time** with guaranteed revert and returns the whole
  kill matrix. This is the only place mutants are applied — structurally serial.
- **3c Pragma applier (SERIAL; ONE agent):** applies all **approved** pragma
  proposals in one serialized source-edit pass, then re-runs the slice suite to
  confirm green. (Pragmas are source edits → must not fan out — finding #7.)

### Phase 4 — Synthesize (PARALLEL section-writers + validating assemble)

Section-writers (one artifact each, new files), then a cheap assemble agent that
**validates required fields/headers** (finding #6):

- `mutation-report.json` (**schema-validated**, required fields): population
  (killed/survived/skipped/suspicious), per-module score (**reported**),
  **median runtime per mutant batch**, **% survivors actionable**,
  **# pragmas added**, **# focused tests added**, gate verdict (pilot:
  report-only).
- `survivor-ledger.md` — one row per survivor: bucket, rationale, action, test
  path + verify row OR pragma/refactor justification (assemble checks the header
  row + that every survivor id appears).
- `recommendations.md` — assertion-strength gaps, equivalent-mutant exclusion
  patterns, design-smell refactor candidates, next-slice notes, bug-prone
  correlation.

---

## Output schemas (every CONSUMED stage — finding #6)

```jsonc
// Phase 1 -> Phase 2
SURVIVOR = { module: str, mutant_id: str, file: str, line: int, diff: str }

// Phase 2 -> Phase 3a/3b/3c. Exactly one of the *_proposal fields is set per `action`.
TRIAGE = { mutant_id: str,
           bucket: "missing-assertion-strength"|"wrong-granularity"|"equivalent"|"intentionally-unhelpful"|"design-smell",
           rationale: str, evidence: [str],
           action: "add-test"|"pragma"|"refactor-note"|"accept-equivalent",
           // add-test -> verify_row; pragma -> pragma_proposal; else null
           verify_row: VERIFY_ROW | null,
           pragma_proposal: PRAGMA | null }

// consumed by 3b (the serial kill-proof runner / wf/mutmut-verify.sh)
VERIFY_ROW = { mutant_id: str, file: str,
               apply_method: "mutmut_apply" | str,   // "diff:<path>" allowed
               test_node: str, expect_clean_rc: int, expect_mutant_rc_nonzero: bool,
               revert_assert: bool }

// consumed by 3c (the serial pragma applier)
PRAGMA = { mutant_id: str, file: str, line: int, original_line: str, justification: str }

// 3b -> Phase 4. Evidence fields REQUIRED (were optional before — finding #6).
VERDICT = { mutant_id: str, bucket: str,
            kill_proven: bool, base_rc: int, mut_rc: int,
            revert: "ok"|"LEAK", equivalence_confirmed: bool, notes: str }

// Phase 4 mutation-report.json (schema-validated by the assemble step)
REPORT = { slice: [str], total: int, killed: int, survived: int, skipped: int,
           score_by_module: object, median_runtime_ms_per_mutant: number,
           pct_survivors_actionable: number, pragmas_added: int, tests_added: int,
           gate: "report-only"|"fail-on-regression"|"floor", gate_pass: bool }
```

---

## CONSTRAINTS (hard)

- **THE concurrency rule** (top of doc) governs every stage. Tree-mutating stages
  are single serial actors; only triage + equivalence reasoning fan out.
- **Engine = mutmut**; Cosmic Ray only on the escalation trigger.
- **ADD-ONLY** test files; pragmas **sparingly + justified**, applied only in the
  serial 3c stage; **scope: stay within `projects/hipblaslt/tensilelite`**.
- **All host git via `git -C "$SRC"`** (never bare from worktree root — finding #3).
- **Never `--snapshot-update` to mask; never skew goldens/tests** — pin ACTUAL
  behavior only.
- **Generic verifier**: `wf/mutmut-verify.sh` is manifest-driven and materializes
  mutants via `mutmut apply` — never the hardcoded p7 search/replace (finding #4).
- **No global threshold** — gate is incremental per slice; pilot is report-only.
- **Commit ATOMICALLY** (one slice: config + verifier + tests + pragmas + reports);
  **NEVER push**.
- **Slice-first** — one slice per workflow invocation via `args`.

---

## ENV (reuse — already built & validated)

- In-container: `docker exec -e PYTHONPATH=$PROJ -w $PROJ tl-char …`,
  `PROJ=/work/projects/hipblaslt/tensilelite`.
- Host paths: `SRC=projects/hipblaslt/tensilelite` (worktree-relative) for git.
- `rocisa` built once (`invoke rocisa`); `LD_LIBRARY_PATH=/opt/rocm/lib` baked in.
- **`--cov` takes a directory PATH, never a dotted module** (SIGABRT).
- mutmut legacy-suite controls: covered-lines-only, **`max_stack_depth`**,
  whole-slice `paths_to_mutate`, `pragma: no mutate` sparingly, type-check
  deferred.
- Verifier primitive to generalize: `wf/p7-survivor-kill.sh` (serial apply/run/
  revert with trap) → rewrite as the manifest-driven `wf/mutmut-verify.sh`.

---

## CI rollout (Google-style incremental ladder)

1. **report-only**: mutmut on the pilot allowlist (slice-1); publish report; no fails.
2. **fail-on-regression**: fail `mutation-unit` only when a previously-killed
   pilot mutant now survives.
3. **small floor**: add `SLICE_FLOOR` per stabilized slice (from observed
   actionable-survivor rate), only after runtime + noise are stable.
4. **ratchet**: extend the allowlist slice-by-slice; never a repo-wide global gate.

---

## COST & SCALE controls

- **Slice-first**; `/workflows` shows live tokens — stop without losing work.
- **16 concurrent cap**; size the triage / equivalence fan-out to it.
- **Model routing**: triage drafting + equivalence reasoning → smaller model;
  the single serial verify runner + synthesis → strongest model; Phase 1 mutmut
  shell-out → smaller model (it just drives the CLI).
- **mutmut pop run is the dominant cost** — one slice-level pass; rely on the
  incremental cache across re-runs; escalate to Cosmic Ray if it stops amortizing.
- **Optional survivor-ranking agent**: rank survivors so a bounded run spends
  budget on the highest-value ones first.

## PERMISSIONS (pre-allowlist)

`docker exec` (tl-char), `pytest`, `coverage`, `mutmut` (incl. `mutmut apply`),
`git -C … checkout/--/diff/status`, `pip`. File edits auto-approve in
`acceptEdits`; pre-allowlist shell commands so a long run never stalls on a prompt.

---

## Part A — task description to type (Claude authors the script; approve the card)

```text
ultracode: production mutation testing for one TensileLite Python module slice with
mutmut. CONCURRENCY RULE: only read-only / new-file-only stages may fan out; running
mutmut, applying a mutant, and editing source for a pragma are each ONE serial actor.
Engine = mutmut (Cosmic Ray only on escalation). Gate is incremental (report-only on
the pilot), never a global threshold. All host git via `git -C projects/hipblaslt/tensilelite`.

UNIT OF WORK: one survivor mutant (analysis fan-out); engine input = one module slice
run as ONE mutmut pass. WORK-LIST = args (default slice-1: Tensile/Common/Utilities.py +
Tensile/TensileLogic/{ValidChipId,ValidMatrixInstruction,ValidWorkGroup,ValidWorkGroupMappingXCC}.py).

PHASES:
 0 Prep (serial): baseline green; ensure mutmut; coverage run for covered-line filtering
   (--cov = dir PATH); write [mutmut] config (whole-slice paths_to_mutate, covered-lines,
   max_stack_depth) + tox mutation-unit env + the GENERIC serial verifier wf/mutmut-verify.sh
   (manifest-driven; materializes mutants via `mutmut apply`, not search/replace).
 1 Execute (SERIAL, ONE agent): one slice-level `mutmut run`; `mutmut results`; group
   survivors by module; return SURVIVOR{module,mutant_id,file,line,diff} via `mutmut show`.
 2 Triage (PARALLEL, one agent per survivor, PROPOSAL-ONLY): classify into 5 buckets; for
   add-test buckets write a DISTINCT NEW test file + emit a VERIFY_ROW; for intentionally-
   unhelpful emit a PROPOSED pragma (do NOT edit source); for design-smell a refactor note;
   for equivalent an equivalence claim. No source mutation in this phase.
 3 Verify: 3a equivalence audit (PARALLEL, read-only, skeptical, downgrade unconfirmed);
   3b kill-proof (SERIAL, ONE agent) feeds the full manifest to wf/mutmut-verify.sh (applies
   mutants one at a time, guaranteed revert) -> kill matrix; 3c pragma applier (SERIAL, ONE
   agent) applies approved pragmas in one pass, re-runs slice suite green.
 4 Synthesize (PARALLEL section-writers + validating assemble): mutation-report.json (schema-
   validated: score + median runtime/mutant + % actionable + #pragmas + #tests; gate=report-only
   for pilot) + survivor-ledger.md + recommendations.md; assemble validates required fields/headers.

OUTPUT SCHEMA: SURVIVOR -> TRIAGE(+VERIFY_ROW/PRAGMA) -> VERDICT -> REPORT (see plan; VERDICT
requires base_rc/mut_rc/revert/notes).
DONE WHEN: full slice mutmut run; 0 untriaged survivors; every test-fixable survivor KILLED via
the serial runner; approved pragmas applied serially; design-smell noted; slice suite green;
`git -C projects/hipblaslt/tensilelite status` shows no leak beyond new tests + justified
pragmas; reports emitted & validated. Pilot is report-only.
COST: slice-1 first; triage/equivalence on a smaller model, serial verify + synthesis on the
strongest; optional survivor-ranking agent.
PERMISSIONS: docker exec tl-char, pytest, coverage, mutmut (incl. apply), git -C, pip.
```

## Part B — script shape to expect / review (verified grammar; concurrency-correct)

```javascript
export const meta = {
  name: 'tl-mutmut-slice',
  description: 'mutmut mutation run for one TensileLite Python slice: serial execute -> parallel triage (proposal-only) -> parallel equivalence audit + SERIAL kill-proof + SERIAL pragma apply -> validated report. Incremental gate.',
  phases: [{ title: 'Prep' }, { title: 'Execute' }, { title: 'Triage' }, { title: 'Verify' }, { title: 'Synthesize' }],
}

const SURVIVOR = { type:'object', additionalProperties:false,
  required:['module','mutant_id','file','line','diff'],
  properties:{ module:{type:'string'}, mutant_id:{type:'string'},
    file:{type:'string'}, line:{type:'integer'}, diff:{type:'string'} } }

const VERIFY_ROW = { type:['object','null'], additionalProperties:false,
  required:['mutant_id','file','apply_method','test_node','expect_clean_rc','expect_mutant_rc_nonzero','revert_assert'],
  properties:{ mutant_id:{type:'string'}, file:{type:'string'}, apply_method:{type:'string'},
    test_node:{type:'string'}, expect_clean_rc:{type:'integer'},
    expect_mutant_rc_nonzero:{type:'boolean'}, revert_assert:{type:'boolean'} } }

const PRAGMA = { type:['object','null'], additionalProperties:false,
  required:['mutant_id','file','line','original_line','justification'],
  properties:{ mutant_id:{type:'string'}, file:{type:'string'}, line:{type:'integer'},
    original_line:{type:'string'}, justification:{type:'string'} } }

const TRIAGE = { type:'object', additionalProperties:false,
  required:['mutant_id','bucket','rationale','evidence','action','verify_row','pragma_proposal'],
  properties:{ mutant_id:{type:'string'},
    bucket:{enum:['missing-assertion-strength','wrong-granularity','equivalent','intentionally-unhelpful','design-smell']},
    rationale:{type:'string'}, evidence:{type:'array',items:{type:'string'}},
    action:{enum:['add-test','pragma','refactor-note','accept-equivalent']},
    verify_row: VERIFY_ROW, pragma_proposal: PRAGMA } }

const VERDICT = { type:'object', additionalProperties:false,
  required:['mutant_id','bucket','kill_proven','base_rc','mut_rc','revert','equivalence_confirmed','notes'],
  properties:{ mutant_id:{type:'string'}, bucket:{type:'string'}, kill_proven:{type:'boolean'},
    base_rc:{type:'integer'}, mut_rc:{type:'integer'}, revert:{enum:['ok','LEAK']},
    equivalence_confirmed:{type:'boolean'}, notes:{type:'string'} } }

const slice = args ?? [
  'Tensile/Common/Utilities.py',
  'Tensile/TensileLogic/ValidChipId.py',
  'Tensile/TensileLogic/ValidMatrixInstruction.py',
  'Tensile/TensileLogic/ValidWorkGroup.py',
  'Tensile/TensileLogic/ValidWorkGroupMappingXCC.py',
]

// Phase 0 — Prep (SERIAL): baseline + mutmut + coverage + [mutmut] config (whole-slice)
// + tox mutation-unit env + wf/mutmut-verify.sh (generic, mutmut-apply-based). Host git via -C.
phase('Prep')
await agent(`Prep mutmut for slice ${JSON.stringify(slice)} in tl-char: baseline pytest green; `+
  `ensure mutmut; coverage run (--cov=dir PATH); write [mutmut] config with paths_to_mutate = the `+
  `WHOLE slice, covered-lines-only, max_stack_depth set; add tox mutation-unit env; write/confirm `+
  `wf/mutmut-verify.sh (manifest-driven serial runner that materializes mutants via \`mutmut apply\`, `+
  `runs the test node clean then mutated, reverts via \`git -C projects/hipblaslt/tensilelite checkout --\`).`,
  { label:'prep', phase:'Prep' })

// Phase 1 — Execute (SERIAL, ONE agent; fixes #1,#5). One slice-level mutmut pass; group by module.
phase('Execute')
const pop = await agent(
  `Run ONE slice-level \`mutmut run\` in tl-char (config from prep; covered lines; stack-depth set). `+
  `Then \`mutmut results\`; for each survivor capture \`mutmut show <id>\`. Group survivors by module `+
  `(partition on file path). Do NOT launch more than one mutmut process.`,
  { label:'mutmut:slice', phase:'Execute', schema:{type:'object', additionalProperties:false,
      required:['total','killed','survived','survivors'],
      properties:{ total:{type:'integer'}, killed:{type:'integer'}, survived:{type:'integer'},
        skipped:{type:'integer'}, suspicious:{type:'integer'},
        survivors:{type:'array', items: SURVIVOR } } } })
const survivors = (pop?.survivors ?? [])

// Phase 2 — Triage (PARALLEL, proposal-only; read-only + NEW files only; fixes #7).
phase('Triage')
const triage = (await parallel(survivors.map((s) => () => agent(
  `Triage survivor ${s.mutant_id} in ${s.file}:${s.line}.\nDIFF:\n${s.diff}\n`+
  `Classify into one of {missing-assertion-strength|wrong-granularity|equivalent|intentionally-unhelpful|`+
  `design-smell} + evidence. For missing-assertion-strength/wrong-granularity: write the SMALLEST add-only `+
  `test to a DISTINCT NEW file test_mut_${s.mutant_id}_char.py and emit verify_row (apply_method="mutmut_apply", `+
  `test_node = the new test). For intentionally-unhelpful: emit pragma_proposal ONLY (do NOT edit source). `+
  `For design-smell: refactor note. For equivalent: equivalence claim. Do not mutate any tracked source file.`,
  { label:`triage:${s.mutant_id}`, phase:'Triage', schema: TRIAGE })))).filter(Boolean)

// Phase 3a — Equivalence audit (PARALLEL, read-only reasoning; downgrade unconfirmed).
phase('Verify')
const equivClaims = triage.filter(t => t.action === 'accept-equivalent' || t.bucket === 'intentionally-unhelpful')
const audited = (await parallel(equivClaims.map((t) => () => agent(
  `Skeptically re-derive the ${t.bucket} claim for ${t.mutant_id}: ${t.rationale}. `+
  `DOWNGRADE to "needs-test" unless the equivalence/no-value argument is airtight. Read-only; no edits.`,
  { label:`equiv:${t.mutant_id}`, phase:'Verify',
    schema:{type:'object', additionalProperties:false, required:['mutant_id','equivalence_confirmed','notes'],
      properties:{ mutant_id:{type:'string'}, equivalence_confirmed:{type:'boolean'}, notes:{type:'string'} } } })))).filter(Boolean)

// Phase 3b — Kill-proof (SERIAL, ONE agent; fixes #2,#4). Full manifest -> wf/mutmut-verify.sh, one apply at a time.
const manifest = triage.filter(t => t.action === 'add-test' && t.verify_row).map(t => t.verify_row)
const killMatrix = await agent(
  `Write this VERIFY manifest to a TSV and run wf/mutmut-verify.sh over it ONCE (serial; applies mutants one at a `+
  `time via \`mutmut apply\`, reverts each). Return one VERDICT per row with base_rc/mut_rc/revert/notes.\n`+
  JSON.stringify(manifest, null, 2),
  { label:'killproof', phase:'Verify',
    schema:{type:'object', additionalProperties:false, required:['verdicts'],
      properties:{ verdicts:{type:'array', items: VERDICT } } } })

// Phase 3c — Pragma applier (SERIAL, ONE agent; fixes #7). Approved pragmas in one source-edit pass.
const pragmas = triage.filter(t => t.action === 'pragma' && t.pragma_proposal).map(t => t.pragma_proposal)
const pragmaResult = await agent(
  `Apply these approved \`# pragma: no mutate\` edits in ONE serial pass, then re-run the slice suite; confirm green. `+
  `Use \`git -C projects/hipblaslt/tensilelite\` for any status/diff.\n${JSON.stringify(pragmas, null, 2)}`,
  { label:'pragma-apply', phase:'Verify',
    schema:{type:'object', additionalProperties:false, required:['applied','suite_green'],
      properties:{ applied:{type:'integer'}, suite_green:{type:'boolean'}, notes:{type:'string'} } } })

// Phase 4 — Synthesize (PARALLEL section-writers + validating assemble; fixes #6).
phase('Synthesize')
const ctx = JSON.stringify({ slice, pop, verdicts: killMatrix?.verdicts ?? [], audited, pragmaResult })
await parallel([
  () => agent(`Write mutation-report.json for ${ctx}. Emit the REPORT schema fields: score_by_module (REPORTED), `+
    `median_runtime_ms_per_mutant, pct_survivors_actionable, pragmas_added, tests_added; gate="report-only", gate_pass=true (pilot).`,
    { label:'report', phase:'Synthesize',
      schema:{type:'object', additionalProperties:true, required:['total','killed','survived','score_by_module',
        'median_runtime_ms_per_mutant','pct_survivors_actionable','pragmas_added','tests_added','gate','gate_pass'],
        properties:{ total:{type:'integer'}, killed:{type:'integer'}, survived:{type:'integer'},
          score_by_module:{type:'object'}, median_runtime_ms_per_mutant:{type:'number'},
          pct_survivors_actionable:{type:'number'}, pragmas_added:{type:'integer'}, tests_added:{type:'integer'},
          gate:{type:'string'}, gate_pass:{type:'boolean'} } } }),
  () => agent(`Write survivor-ledger.md from ${ctx}: one row per survivor (bucket, rationale, action, test path + verify row OR pragma/refactor justification).`,
    { label:'ledger', phase:'Synthesize' }),
  () => agent(`Write recommendations.md from ${ctx}: assertion-strength gaps, equivalent exclusion patterns, design-smell refactor candidates, next-slice notes, bug-prone correlation.`,
    { label:'recs', phase:'Synthesize' }),
])
// cheap validating assemble: confirm report keys + ledger header + every survivor id present.
const check = await agent(`Validate: mutation-report.json has all REPORT required keys; survivor-ledger.md has the header `+
  `row and a line for every survivor id in ${JSON.stringify(survivors.map(s=>s.mutant_id))}. Report PASS/FAIL + missing items.`,
  { label:'assemble-check', phase:'Synthesize' })

return { slice, survivors: survivors.length,
         killed: (killMatrix?.verdicts ?? []).filter(v => v.kill_proven).length,
         equivalent: audited.filter(v => v.equivalence_confirmed).length,
         pragmas: pragmaResult?.applied ?? 0, validation: check }
```

**Review checklist before approving the card:** THE concurrency rule holds —
Execute / kill-proof / pragma-apply are each ONE serial agent, only triage +
equivalence reasoning use `parallel` · every consumed stage has a `schema`
(SURVIVOR, TRIAGE, VERIFY_ROW, PRAGMA, VERDICT-with-evidence, REPORT) · host git
uses `-C "$SRC"` · `wf/mutmut-verify.sh` is generic (mutmut apply, not p7
search/replace) · triage writes only NEW files · pragmas applied only in 3c ·
synthesis split + a validating assemble step · slice via `args` · `SLICE_FLOOR`
deferred to CI phase 3.

---

## Review findings addressed (`mutants-findings.md`)

| # | Sev | Finding | Fix in this revision |
|---|-----|---------|----------------------|
| 1 | High | Part B ran `parallel(... mutmut ...)` — multiple mutmut instances on one tree/cache | Phase 1 is **one serial agent, one slice-level mutmut pass**; THE concurrency rule stated once and enforced structurally |
| 2 | High | `pipeline` verified survivors concurrently; "serial" prompt ≠ lock | Verify split: parallel **equivalence reasoning** only; **kill-proof is one serial agent** driving the manifest runner |
| 3 | High | Leak check cwd-sensitive → false-clean from root (verified) | All host git via **`git -C "$SRC"`**; done-criterion + constraints + ENV updated |
| 4 | High | `wf/p7` hardcoded to 2 mutants/files + search/replace, not generic | New **`wf/mutmut-verify.sh`** manifest-driven runner that materializes mutants via **`mutmut apply`**; interface fields defined |
| 5 | Med | mutmut scoping ambiguous (per-module vs whole slice) | **One slice-level pass**, `paths_to_mutate` = whole slice, results grouped by module |
| 6 | Med | Synthesis had no schemas; VERDICT evidence fields optional | **REPORT schema** + validating assemble step; **VERDICT now requires** base_rc/mut_rc/revert/notes; added VERIFY_ROW + PRAGMA schemas |
| 7 | Med | Concurrent triage could edit shared source (pragmas) | Triage is **proposal-only** (new files only); pragmas applied in the **single serial 3c stage** |

## Remaining open item

`SLICE_FLOOR` — set per stabilized slice at CI phase 3 from the pilot's observed
actionable-survivor rate. Everything else is pinned; slice-1 is runnable
end-to-end once Phase 0 writes the `[mutmut]` config + `wf/mutmut-verify.sh`.
