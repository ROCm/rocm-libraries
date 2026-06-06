export const meta = {
  name: 'codegen-p4-round1',
  description: 'P4 Stage-2 expansion round 1: characterize cheap standalone/library-mgmt modules; methodology-A gate',
  phases: [{ title: 'Cheapest-input' }, { title: 'Golden' }, { title: 'Assemble' }],
}

// ---------------------------------------------------------------------------
// SHARED ENV + HARD RULES — pasted into EVERY agent prompt (subagents see nothing
// from siblings). Source of truth: PLAN-CODEGEN-WORKFLOW.md, WORKFLOW-SPECS.md,
// BASELINE-AND-PROGRESS.md, coverage/p4/RANKING-AND-METHODOLOGY.md.
// ---------------------------------------------------------------------------
const SHARED = [
  'ENV (paths are INSIDE the container):',
  '  CON=tl-char ; PROJ=/work/projects/hipblaslt/tensilelite',
  '  Host worktree root: /home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-coverage',
  '  Host project: <root>/projects/hipblaslt/tensilelite (== $PROJ inside the container).',
  '  Edit files on the HOST path; RUN pytest/coverage INSIDE the container via docker exec.',
  '  The container has cp312 pytest/coverage entrypoints (python3.11 has neither).',
  '',
  'ISOLATED MEASURE PREFIX (one fresh process per input, its own COVERAGE_FILE):',
  '  docker exec -e PYTHONPATH=$PROJ -e COVERAGE_FILE=$PROJ/.coverage.<ID> -w $PROJ $CON \\',
  '    pytest -p no:cacheprovider -m unit --cov=Tensile --cov-config=pyproject.toml -q <TEST_NODE>',
  '  then per-file JSON:  docker exec -e COVERAGE_FILE=$PROJ/.coverage.<ID> -w $PROJ $CON \\',
  '    coverage json -o /tmp/<ID>.json   (executed_lines live under files["Tensile/<f>.py"].executed_lines)',
  '',
  'HARD RULES (poka-yoke — violate none):',
  '  1. ADD-ONLY. NEW files only, under Tensile/Tests/unit/characterization/. NEVER modify or',
  '     delete any existing file (incl. pyproject.toml, existing tests, source). New fixtures go',
  '     under characterization/ too.',
  '  2. --cov=Tensile is a PATH, never a dotted module (Tensile.x => rocisa SIGABRT).',
  '  3. One COVERAGE_FILE per shard; change it together with the test node. Never write bare .coverage.',
  '  4. Every new test module MUST set: import pytest ; pytestmark = pytest.mark.unit  (so it joins',
  '     the -m unit gate suite). Tests MUST be CPU-only and pass with no GPU.',
  '  5. rocisa footprint is per-process: keep each input bounded (limit kernels; tiny fixtures).',
  '  6. NEVER push, NEVER commit (only the serialized Assemble agent commits).',
  '',
  'HARNESS / PATTERNS to reuse (do NOT reinvent):',
  '  - characterization/conftest.py sets sys.path + fixtures; mirror an EXISTING neighbour test in',
  '    the target subdir for import style. Many subdirs already exist (LibraryLogic,',
  '    TensileMergeLibrary, TensileRetuneLibrary, TensileBenchmarkLibraryClient, BenchmarkStructs,',
  '    TensileCreateLibraryRun, etc.). Create a new subdir only if none fits.',
  '  - _codegen/config_harness.py: emit_kernels_from_config(config_path, limit=8, arch, ...),',
  '    solutions_from_config(...). _codegen/codegen_harness.py: emit_kernels_from_logic(...),',
  '    solutions_from_logic(...). Use these for any codegen/Solution-derivation reach.',
  '  - Real shipped configs/logic live under Tensile/Tests/common/** and the tuning tree; designed',
  '    seed YAMLs under _codegen/data/test_data/_designed/<arch>/.',
  '',
  'GOAL-FILE HINTS: work/tensilelite-characterization/next-goal-*.md hold prior analysis of several',
  '  of these modules (libraryio, tensilelogic, solution-*, validparameters, etc.) — consult the',
  '  matching one for the cheapest reach if present.',
].join('\n')

// ---------------------------------------------------------------------------
// Round-1 targets — pre-ranked from the methodology-A term-missing receipt
// (coverage/head-unit-coverage.log -> coverage/p4/gap-by-miss.tsv). These are the
// low-/zero-coverage standalone + library-management modules: highest miss-per-effort,
// CPU-reachable by import + small-fixture invocation.
// ---------------------------------------------------------------------------
const TARGETS = [
  { id: 'gensummations', file: 'Tensile/GenerateSummations.py', miss: 107, ranges: '25-188',
    hint: 'Module is 0% — not even imported past its header. Import the module and invoke its entry/main with mock argv (monkeypatch sys.argv) + a tiny library-logic fixture or a stubbed config so the body runs. Pin actual behavior (return code / printed output / written file).' },
  { id: 'updatelib', file: 'Tensile/TensileUpdateLibrary.py', miss: 97, ranges: '25-165',
    hint: 'Module is 0%. CLI script: import + invoke main() with monkeypatched argv pointing at a tiny in-tree library/logic fixture; characterize the actual outcome. Mirror TensileRetuneLibrary/ or TensileMergeLibrary/ neighbour tests if present.' },
  { id: 'retunelib', file: 'Tensile/TensileRetuneLibrary.py', miss: 93, ranges: '71-95,101-140,144-232,236',
    hint: '25% covered. Drive the un-run code paths (argv variants / a tiny logic input). A TensileRetuneLibrary/ subdir already exists — add a NEW test there.' },
  { id: 'mergelib', file: 'Tensile/TensileMergeLibrary.py', miss: 133, ranges: '67,96-104,164-165,181-197,223-280,283-349,352-373',
    hint: '49% covered. Merge two tiny logic YAMLs (the big 223-373 block is the core merge loop). A TensileMergeLibrary/ subdir exists — add a NEW test exercising the merge/dedup path.' },
  { id: 'verifystinky', file: 'Tensile/verify_stinky_comment_vs_elf_text.py', miss: 101, ranges: '53-77,82-105,110-133,138-141,149-188,193-209',
    hint: '9.88% covered. Import + invoke the verify routine on a small synthetic fixture (comment vs elf-text). Pin actual pass/mismatch behavior. See work/tensilelite-characterization/*stinky* or TensileCreateLibraryRun neighbours.' },
  { id: 'benchproblems', file: 'Tensile/BenchmarkProblems.py', miss: 111, ranges: '182-222,259-282,297-325,431-452,543-555,587-594,635-642,676-680,780-782',
    hint: '64.69% covered. Use _codegen/config_harness (BenchmarkProcess -> constructForkPermutations -> _generateForkedSolutions) with a config whose ForkParameters breadth drives the un-run derivation arms (182-325, 431-452). Cheapest channel = a designed config under _designed/.' },
  { id: 'librarylogic', file: 'Tensile/LibraryLogic.py', miss: 535, ranges: '112-169,419-455,477-538,552-632,671-758,782-1017,1024-1141,1215-1424',
    hint: '39.5% covered, LARGE gap. Parse + analyze a tiny logic set so the selection-library/analysis arms run (782-1017 and 1024-1141 are the biggest blocks). Cheapest = drive parseLibraryLogicFile + the analysis entry on a small in-tree logic fixture. A LibraryLogic/ subdir exists.' },
  { id: 'benchclient', file: 'Tensile/TensileBenchmarkLibraryClient.py', miss: 92, ranges: '34-73,80-100,107-159,180',
    hint: '19% covered. The --cpu-only switch is present on this branch (test_cpu_only_switch.py), so the client driver path is CPU-reachable now. Drive the client-library benchmark entry with the switch on + a tiny fixture. A TensileBenchmarkLibraryClient/ subdir exists.' },
]

const CAND = {
  type: 'object', additionalProperties: false,
  required: ['target', 'input_path', 'cov_file', 'measured_marginal', 'err', 'kept', 'note'],
  properties: {
    target: { type: 'string' },
    input_path: { type: 'string' },   // host path of the NEW test file (+ fixtures noted in `note`)
    cov_file: { type: 'string' },      // .coverage.kept_1_<id>
    measured_marginal: { type: 'integer' }, // count of THIS target's previously-missing lines now executed
    err: { type: 'integer' },          // pytest exit code of the isolated run (0 = passed)
    kept: { type: 'boolean' },         // true iff err==0 AND measured_marginal >= 10
    note: { type: 'string' },          // fixtures created; or why dropped/deferred
  },
}
const GOLD = {
  type: 'object', additionalProperties: false, required: ['input_path', 'stable', 'reason'],
  properties: { input_path: { type: 'string' }, stable: { type: 'boolean' }, reason: { type: 'string' } },
}

phase('Cheapest-input'); phase('Golden')
const worked = await pipeline(TARGETS,
  (t) => agent(
    'ultracode: Author the CHEAPEST add-only, CPU-only characterization test that EXECUTES the\n' +
    'currently-uncovered lines of ' + t.file + ' (methodology-A missing ranges: ' + t.ranges + ').\n' +
    'Approach hint: ' + t.hint + '\n\n' +
    'STEPS:\n' +
    '  1. Read the target source ' + t.file + ' and an existing neighbour test in the matching\n' +
    '     characterization/ subdir to copy import/fixture style. Place the NEW test in that subdir\n' +
    '     (create one only if none fits). Name it test_<area>_char.py. Set pytestmark = pytest.mark.unit.\n' +
    '  2. Write tests that PIN ACTUAL current behavior (return values / printed output / written\n' +
    '     artifacts / raised errors). Do NOT change source to make output prettier. err!=0 inside the\n' +
    '     code-under-test that you ASSERT on is a valid pinned rejection; a pytest FAILURE is not.\n' +
    '  3. ISOLATED MEASURE: run only your new test node with COVERAGE_FILE=$PROJ/.coverage.kept_1_' + t.id + ' ;\n' +
    '     then coverage json -o /tmp/' + t.id + '.json. Compute measured_marginal = how many line\n' +
    '     numbers inside the ranges [' + t.ranges + '] of ' + t.file + ' now appear in\n' +
    '     files["' + t.file + '"].executed_lines. (A short docker exec python -c reading the json and\n' +
    '     intersecting with the parsed ranges is the reliable way — report the real count.)\n' +
    '  4. KEEP only if pytest err==0 AND measured_marginal >= 10. If the cheapest reach cannot clear\n' +
    '     that (too much GPU/IO scaffolding), set kept=false and explain in note (this becomes ceiling\n' +
    '     evidence, not a failure). Do NOT commit. Return the CAND schema with REAL measured numbers.\n\n' +
    SHARED,
    { label: 'cand:' + t.id, phase: 'Cheapest-input', schema: CAND, model: 'haiku' }),
  (c) => c && c.kept
    ? agent(
        'ultracode: Adversarially verify the kept characterization test at ' + c.input_path + '.\n' +
        'Re-run that test node INSIDE the container TWICE (no --snapshot-update), each with its own\n' +
        'COVERAGE_FILE (e.g. .coverage.ver_a / .coverage.ver_b). It is stable ONLY IF both runs pass\n' +
        'with identical outcome (and any syrupy snapshot is byte-identical both runs). Default\n' +
        'stable=false on ANY doubt, churn, or order-dependence. Return GOLD{input_path,stable,reason}.\n\n' +
        SHARED,
        { label: 'gold:' + c.target, phase: 'Golden', schema: GOLD })
    : null)

phase('Assemble')
const kept = worked.filter(Boolean)
const report = await agent(
  'ultracode: SERIALIZED Assemble for P4 Round 1 — you are the ONLY committer. Run the\n' +
  'methodology-A whole-project gate, compare to the prior baseline FILE, and commit add-only.\n\n' +
  'INPUTS (kept-input + verify results from this round):\n' + JSON.stringify(kept) + '\n\n' +
  'PRIOR BASELINE FILE (the BEFORE): work/tensilelite-characterization/coverage/head-unit-baseline.txt\n' +
  '  TOTAL = 68.85% (54867 stmts, 15723 miss), commit 6f1e20b1a7f. Target gate >= 80.00%.\n\n' +
  'STEPS (run, do not paraphrase):\n' +
  '  1. Drop any candidate with kept=false or whose Golden verify returned stable=false. Confirm the\n' +
  '     kept test FILES exist on disk (host project path). If ZERO survive: write\n' +
  '     work/tensilelite-characterization/coverage/p4/round-1-deferred.txt explaining why (with the\n' +
  '     per-target notes as ceiling evidence), append a PLAN-CODEGEN-WORKFLOW.md §11 line\n' +
  '     "P4 round 1 NO-GAIN/DEFERRED" , commit ONLY that file (explicit-path, --no-verify), return.\n' +
  '  2. METHODOLOGY-A GATE (identical command to head-unit-baseline so the delta is valid; ~12 min):\n' +
  '       docker exec -e PYTHONPATH=$PROJ -e COVERAGE_FILE=$PROJ/.coverage.mA1 -w $PROJ $CON \\\n' +
  '         pytest -p no:cacheprovider -m unit --cov=Tensile --cov=rocisa --cov-config=pyproject.toml \\\n' +
  '         -n 4 -q Tensile/Tests/unit\n' +
  '     (pytest-cov auto-combines the -n 4 xdist worker data into .coverage.mA1.) Assert 0 failed;\n' +
  '     pass-count must only grow vs 2560; 201 skipped unchanged.\n' +
  '  3. Write the receipt + read TOTAL:\n' +
  '       docker exec -e COVERAGE_FILE=$PROJ/.coverage.mA1 -w $PROJ $CON coverage report --show-missing \\\n' +
  '         | tee work/tensilelite-characterization/coverage/p4/master-baseline-R1.txt | tail -1\n' +
  '  4. GATE CHECK: new TOTAL must STRICTLY EXCEED 68.85%. If it does NOT increase, this is a no-gain\n' +
  '     round: write coverage/p4/resistance-r1.md with file:line evidence for why the kept tests did\n' +
  '     not move the whole-project number (e.g. lines already covered by the full suite), and do NOT\n' +
  '     fake a gain. Either way keep going to step 5/6 to commit the (passing, add-only) tests +\n' +
  '     receipt honestly with the REAL delta.\n' +
  '  5. COMMIT (explicit-path git add of: each kept test file + any new fixtures + the\n' +
  '     coverage/p4/master-baseline-R1.txt receipt + RANKING-AND-METHODOLOGY.md + gap-by-miss.tsv +\n' +
  '     head-term-missing-raw.txt + resistance-r1.md if written). Use git commit --no-verify\n' +
  '     (hipBLASLt host hooks need py>=3.10). NEVER git add -A. NEVER push.\n' +
  '  6. Update PLAN-CODEGEN-WORKFLOW.md §8 (check the P4 round-1 box if it gained) and §11 (one line:\n' +
  '     "P4 round 1 — 68.85% -> <new>% (+<delta> pts, N tests), master-baseline-R1.txt, commit <sha>")\n' +
  '     and BASELINE-AND-PROGRESS.md §4 with the new number. Commit those doc edits too (explicit-path).\n' +
  '  Return a short text summary: new TOTAL %, delta, # tests kept, commit sha(s), and any deferrals.\n\n' +
  SHARED,
  { label: 'assemble', phase: 'Assemble' })

return { report, kept: kept.length, targets: TARGETS.length }
