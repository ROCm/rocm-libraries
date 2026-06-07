export const meta = {
  name: 'tl-mutmut-triage-slice1',
  description:
    'Triage + kill the 131 slice-1 mutmut survivors. Parallel per-function triage authors add-only characterization tests (proposal-only, no source edits); a SINGLE serial agent kill-proofs them via wf/mutmut-verify.sh; one bounded repair round; a SINGLE serial agent applies any pragmas; then synthesis. Concurrency rule: only triage fans out — apply/verify/pragma are single serial actors.',
  phases: [
    { title: 'Triage' },
    { title: 'Verify' },
    { title: 'Repair' },
    { title: 'Pragma' },
    { title: 'Synthesize' },
  ],
}

const CON = 'tl-mut'
const WT = '/home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-mutation'
const SRC_REL = 'projects/hipblaslt/tensilelite'
const PROJ = '/work/' + SRC_REL
const OUT = 'work/tensilelite-characterization/coverage/mutprod/workflow'

const groups = args
const srcByKey = {}
for (const g of groups) srcByKey[g.module + '::' + g.function] = g.source_file

const TRIAGE_ITEM = {
  type: 'object', additionalProperties: false,
  required: ['mutant_id', 'bucket', 'action', 'test_node', 'pragma_line', 'note'],
  properties: {
    mutant_id: { type: 'string' },
    bucket: { enum: ['missing-assertion-strength', 'wrong-granularity', 'equivalent', 'intentionally-unhelpful', 'design-smell'] },
    action: { enum: ['add-test', 'none', 'pragma', 'refactor-note'] },
    test_node: { type: ['string', 'null'] },
    pragma_line: { type: ['integer', 'null'] },
    note: { type: 'string' },
  },
}
const GROUP_RESULT = {
  type: 'object', additionalProperties: false,
  required: ['function', 'module', 'test_file', 'test_written', 'triage'],
  properties: {
    function: { type: 'string' }, module: { type: 'string' },
    test_file: { type: 'string' }, test_written: { type: 'boolean' },
    triage: { type: 'array', items: TRIAGE_ITEM },
  },
}
const VERDICT = {
  type: 'object', additionalProperties: false,
  required: ['mutant_id', 'verdict', 'base_rc', 'mut_rc', 'revert'],
  properties: {
    mutant_id: { type: 'string' }, verdict: { type: 'string' },
    base_rc: { type: 'integer' }, mut_rc: { type: 'integer' }, revert: { type: 'string' },
  },
}
const VERIFY_RESULT = {
  type: 'object', additionalProperties: false, required: ['verdicts'],
  properties: { verdicts: { type: 'array', items: VERDICT } },
}

// ---------------------------------------------------------------- Phase: Triage
phase('Triage')
const triaged = (await parallel(groups.map((g) => () => agent(
  `Triage mutmut survivors for ONE Python function and author ADD-ONLY characterization tests that KILL the test-fixable ones. Fresh context — everything you need is below; use absolute paths.\n\n` +
  `CONTEXT:\n` +
  `- Worktree root: ${WT}\n` +
  `- Source file: ${WT}/${g.source_file}  (function: ${g.function}; import module: ${g.module})\n` +
  `- Existing char tests (read 1-2 for import style + markers): ${WT}/${g.char_dir}\n` +
  `- WRITE your NEW test file here (add-only; do NOT touch any other file): ${WT}/${g.test_file}\n` +
  `- Container: ${CON}; in-container project dir: ${PROJ}\n` +
  `- Survivor mutant ids (${g.survivors.length}): ${JSON.stringify(g.survivors)}\n\n` +
  `COMMANDS:\n` +
  `- See a mutant diff:  docker exec -w ${PROJ} ${CON} mutmut show <mutant_id>\n` +
  `- Run a test on CLEAN source (must PASS):  docker exec -e PYTHONPATH=${PROJ} -w ${PROJ} ${CON} pytest -p no:cacheprovider -m unit -q "${g.test_file}::<test_fn>"\n` +
  `- DO NOT run \`mutmut run\` or \`mutmut apply\`. DO NOT edit source or existing tests. (Mutant application is a later serial phase.)\n\n` +
  `STEPS:\n` +
  `1. Read the source function. For EACH survivor, run \`mutmut show\` and classify into exactly one bucket:\n` +
  `   - missing-assertion-strength: code is reached but the changed behavior is unpinned -> add-test.\n` +
  `   - wrong-granularity: needs a smaller direct test -> add-test.\n` +
  `   - equivalent: cannot change observable behavior for any valid input -> action none (justify in note).\n` +
  `   - intentionally-unhelpful: pure logging/spinner/format noise with no behavioral contract worth pinning -> action pragma; set pragma_line to the exact source line number from the diff (do NOT edit source).\n` +
  `   - design-smell: killing needs a refactor not a test -> action refactor-note.\n` +
  `2. For every add-test survivor, write a distinct @pytest.mark.unit test function in ${WT}/${g.test_file} that:\n` +
  `   - imports the callable directly from ${g.module},\n` +
  `   - calls it with input(s) that DISTINGUISH original from mutant (read the diff to find the distinguishing input),\n` +
  `   - asserts the ORIGINAL (current) behavior, so it PASSES on clean source and will FAIL once the mutant is applied.\n` +
  `   Map each add-test survivor to the test_node (${g.test_file}::test_fn) that kills it. One test_fn may cover several survivors of the same logic; set each survivor's test_node to that node.\n` +
  `3. RUN your tests on clean source; they MUST pass (fix until green). If you cannot write a passing test that genuinely targets a survivor, reclassify it honestly (equivalent/design-smell) — never invent a bogus or always-true assertion.\n\n` +
  `HARD RULES: add-only; never edit source (pragmas are proposed via pragma_line, not applied); pin ACTUAL current behavior (characterization), never an idealized version; no assertion you have not confirmed green on clean source.\n\n` +
  `RETURN: function=${JSON.stringify(g.function)}, module=${JSON.stringify(g.module)}, test_file=${JSON.stringify(g.test_file)}, test_written, and triage[] with one entry per survivor {mutant_id, bucket, action, test_node|null, pragma_line|null, note}.`,
  { label: `triage:${g.function}`, phase: 'Triage', schema: GROUP_RESULT }
)))).filter(Boolean)

// Collect add-test verify rows (file from the deterministic group map, not agent echo).
const rows = triaged.flatMap((r) =>
  (r.triage || []).filter((t) => t.action === 'add-test' && t.test_node).map((t) => ({
    mutant_id: t.mutant_id,
    file: srcByKey[r.module + '::' + r.function],
    test_node: t.test_node,
  }))
)

// ---------------------------------------------------------------- Phase: Verify (SERIAL)
phase('Verify')
const verify = rows.length
  ? await agent(
      `SERIAL kill-proof. Build ONE manifest TSV and run the proven verifier ONCE. Bash + Read only; do not edit tests or source.\n\n` +
      `1. Write ${WT}/${OUT}/manifest.tsv. First line EXACTLY (tab-separated):\n` +
      `mutant_id\\tfile\\tapply_method\\ttest_node\\texpect_clean_rc\\texpect_mutant_rc_nonzero\\trevert_assert\n` +
      `Then one row per item (apply_method=mutmut_apply, expect_clean_rc=0, expect_mutant_rc_nonzero=true, revert_assert=true; file is relative to ${SRC_REL}):\n` +
      `${JSON.stringify(rows)}\n\n` +
      `2. Run:\n   cd ${WT} && bash work/tensilelite-characterization/wf/mutmut-verify.sh --container ${CON} --manifest ${OUT}/manifest.tsv --out ${OUT}/verify --src ${SRC_REL}\n` +
      `   (It may exit non-zero if any row is not KILLED — expected. Read the matrix regardless.)\n\n` +
      `3. Read ${WT}/${OUT}/verify/kill_matrix.tsv and return verdicts[] = {mutant_id, verdict, base_rc, mut_rc, revert} for every row.`,
      { label: 'killproof', phase: 'Verify', schema: VERIFY_RESULT }
    )
  : { verdicts: [] }

// ---------------------------------------------------------------- Phase: Repair (SERIAL, bounded 1 round)
phase('Repair')
const byId = {}
for (const r of triaged) for (const t of (r.triage || [])) byId[t.mutant_id] = { ...t, module: r.module, function: r.function, test_file: r.test_file, file: srcByKey[r.module + '::' + r.function] }
const bad = (verify.verdicts || []).filter((v) => v.verdict !== 'KILLED').map((v) => ({ ...v, ...byId[v.mutant_id] }))
const repaired = bad.length
  ? await agent(
      `SERIAL repair (ONE round). These authored tests did NOT prove a kill. For each: read the mutant diff (docker exec -w ${PROJ} ${CON} mutmut show <id>), the test, and the source, then EITHER (a) fix the test in its test_file so it passes clean and fails mutated, OR (b) if the mutant is genuinely equivalent (no distinguishing input), REMOVE the bogus test function and mark it reclassified-equivalent.\n` +
      `Items: ${JSON.stringify(bad.map((b) => ({ mutant_id: b.mutant_id, file: b.file, test_file: b.test_file, test_node: b.test_node, prev: { base_rc: b.base_rc, mut_rc: b.mut_rc } })))}\n\n` +
      `Then re-verify ONLY these: write ${WT}/${OUT}/manifest2.tsv (same header/columns as before, only fixed add-test rows) and run\n` +
      `   cd ${WT} && bash work/tensilelite-characterization/wf/mutmut-verify.sh --container ${CON} --manifest ${OUT}/manifest2.tsv --out ${OUT}/verify2 --src ${SRC_REL}\n` +
      `Read ${WT}/${OUT}/verify2/kill_matrix.tsv. Return verdicts[] {mutant_id, verdict, base_rc, mut_rc, revert} for every item (use verdict "EQUIVALENT" for those you reclassified and removed the test for).`,
      { label: 'repair', phase: 'Repair', schema: VERIFY_RESULT }
    )
  : { verdicts: [] }

// Merge: repaired overrides initial.
const finalVerdict = {}
for (const v of (verify.verdicts || [])) finalVerdict[v.mutant_id] = v
for (const v of (repaired.verdicts || [])) finalVerdict[v.mutant_id] = v
const finalVerdicts = Object.values(finalVerdict)

// ---------------------------------------------------------------- Phase: Pragma (SERIAL)
phase('Pragma')
const pragmas = triaged.flatMap((r) =>
  (r.triage || []).filter((t) => t.action === 'pragma' && t.pragma_line).map((t) => ({
    mutant_id: t.mutant_id, file: srcByKey[r.module + '::' + r.function], pragma_line: t.pragma_line,
  }))
)
const pragmaRes = pragmas.length
  ? await agent(
      `SERIAL pragma apply. Add \`# pragma: no mutate\` to the given source lines (one pass), then confirm the slice suite is green.\n` +
      `Items: ${JSON.stringify(pragmas)}\n` +
      `For each: open ${WT}/<file>, and to the END of the given 1-based line append \`  # pragma: no mutate\` (only if not already present). Do NOT change any code on the line. Dedup if multiple items target the same line.\n` +
      `Then run: docker exec -e PYTHONPATH=${PROJ} -w ${PROJ} ${CON} pytest -p no:cacheprovider -m unit -q Tensile/Tests/unit/characterization/CommonUtilities Tensile/Tests/unit/characterization/TensileLogic\n` +
      `If NOT 0 failed, revert your edits (git -C ${WT}/${SRC_REL} checkout -- <files>) and report suite_green=false. Return {applied:int, suite_green:bool, notes}.`,
      { label: 'pragma', phase: 'Pragma',
        schema: { type: 'object', additionalProperties: false, required: ['applied', 'suite_green', 'notes'], properties: { applied: { type: 'integer' }, suite_green: { type: 'boolean' }, notes: { type: 'string' } } } }
    )
  : { applied: 0, suite_green: true, notes: 'no pragmas proposed' }

// ---------------------------------------------------------------- Phase: Synthesize
phase('Synthesize')
const ctx = JSON.stringify({
  total_survivors: groups.reduce((a, g) => a + g.survivors.length, 0),
  triaged, finalVerdicts, pragmaRes,
})
await parallel([
  () => agent(
    `Write ${WT}/${OUT}/survivor-ledger.md: a markdown table, one row per survivor across all groups — columns: mutant_id, function, bucket, action, verdict (KILLED/BAD/EQUIVALENT/—), test_node-or-justification. Group by function. End with per-bucket counts. Data:\n${ctx}`,
    { label: 'ledger', phase: 'Synthesize' }),
  () => agent(
    `Write ${WT}/${OUT}/mutation-report.json with keys: total_survivors, killed (verdict==KILLED), still_surviving (BAD), equivalent (EQUIVALENT or bucket equivalent/none), pragmas_applied, tests_added (count of distinct test files written), by_bucket (object). Numbers must match the data exactly — do not inflate. Data:\n${ctx}`,
    { label: 'report', phase: 'Synthesize' }),
  () => agent(
    `Write ${WT}/${OUT}/recommendations.md: which functions still have surviving mutants and why, any equivalent-mutant patterns worth a do_not_mutate rule, design-smell refactor candidates, and next-slice notes. Concise. Data:\n${ctx}`,
    { label: 'recs', phase: 'Synthesize' }),
])
const check = await agent(
  `Validate the synthesis. Read ${WT}/${OUT}/mutation-report.json and ${WT}/${OUT}/survivor-ledger.md. Confirm: report has all required keys; killed+still_surviving+equivalent accounts for every add-test/equivalent survivor; the ledger lists every one of the ${groups.reduce((a, g) => a + g.survivors.length, 0)} survivors exactly once. Report PASS or FAIL with specifics. Read only.`,
  { label: 'assemble-check', phase: 'Synthesize' }
)

return {
  groups: groups.length,
  survivors: groups.reduce((a, g) => a + g.survivors.length, 0),
  tests_authored_rows: rows.length,
  killed: finalVerdicts.filter((v) => v.verdict === 'KILLED').length,
  still_bad: finalVerdicts.filter((v) => v.verdict === 'BAD').length,
  equivalent: finalVerdicts.filter((v) => v.verdict === 'EQUIVALENT').length,
  pragmas_applied: pragmaRes.applied,
  validation: check,
}
