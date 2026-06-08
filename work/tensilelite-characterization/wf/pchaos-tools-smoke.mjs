export const meta = {
  name: 'pchaos-tools-smoke',
  description: 'Prove the deferred deeper-layer tools (PICT, CodeQL, Atheris) work in the tl-pchaos-tools image: classify availability, run ONE real-input smoke per tool, cross-check each against its stdlib fallback, and write committable receipts under parametric-chaos/_tooling/<tool>/. Add-only; the DRIVER runs the gate + commits.',
  phases: [
    { title: 'Preflight', detail: 'classify codeql/pict/atheris in tl-pchaos-tools; capture --version receipts -> optional_tools' },
    { title: 'Smoke', detail: 'per available tool: ONE real-input smoke + cross-check vs stdlib fallback (PICT⊇stdlib pairs; CodeQL⊇ast slice; Atheris witness re-checked)' },
    { title: 'Assemble', detail: 'tooling_summary.json + per-tool receipts; verify non-empty; scorecard' },
  ],
}

// ---------------------------------------------------------------------------
// Paths (container; /work == worktree root).
// ---------------------------------------------------------------------------
const PROJ = '/work/projects/hipblaslt/tensilelite'
const WF = '/work/work/tensilelite-characterization/wf/parametric-chaos'
const TOOLS_CON = args?.toolsContainer ?? 'tl-pchaos-tools'
// Real pchaos surface whose deliverables feed the cross-checks (Run-1 reproducible).
const SURFACE = args?.surface ?? 'PublicInputSurface'
const OUT = '/work/work/tensilelite-characterization/parametric-chaos/' + SURFACE
const TOOLDIR = '/work/work/tensilelite-characterization/parametric-chaos/_tooling'
const HTOOLDIR = 'work/tensilelite-characterization/parametric-chaos/_tooling'
const HOUT = 'work/tensilelite-characterization/parametric-chaos/' + SURFACE

const SHARED = [
  'ENVIRONMENT (paths are INSIDE container ' + TOOLS_CON + ' unless prefixed "host:"):',
  '  TOOLS_CON=' + TOOLS_CON + ' (the deferred-tools image; mounts /work == worktree root)',
  '  PROJ=' + PROJ + '  (real Tensile source; pass -w $PROJ to docker exec)',
  '  WF=' + WF + '  (committed stdlib helpers: branch_extractor.py, covering_array.py, harvest_constraints.py)',
  '  OUT=' + OUT + '  (real pchaos deliverables for surface ' + SURFACE + ': domain_model.json, covering_array/, branch_census.jsonl)',
  '  TOOLDIR=' + TOOLDIR + '  (write receipts here; host path ' + HTOOLDIR + ')',
  '  host: /work == worktree root. You MAY write receipt JSON/text with the Write tool to the host path',
  '        (the mount makes it visible in-container); EXECUTION (tools, python, pytest) MUST run in-container',
  '        via `docker exec -w $PROJ ' + TOOLS_CON + ' ...`.',
  '',
  'HARD RULES (memory: rigor-no-skewing-goldens, no-push-local-proof-first, measure-dont-inflate):',
  '  1. ADD-ONLY. Never modify/delete an existing repo file. New files only under TOOLDIR/.',
  '  2. Measure, do not inflate. If a smoke or cross-check FAILS, report it honestly (passed=false +',
  '     detail). NEVER fabricate tool output or a passing cross-check. A real negative is a valid result.',
  '  3. The stdlib fallback is the ORACLE, never replaced: the real tool must SUPERSET / agree with it.',
  '  4. NEVER push, NEVER commit, NEVER start a Monitor. The driver owns the gate + commit.',
  '  5. Keep each unit bounded (<~180s). Scope CodeQL DB create to a SMALL source subset for the smoke.',
].join('\n')

const PREFLIGHT_TOOLS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['container', 'optional_tools', 'receipt_path'],
  properties: {
    container: { type: 'string' },
    optional_tools: {
      type: 'array', items: {
        type: 'object', additionalProperties: false,
        required: ['tool', 'available', 'version'],
        properties: { tool: { type: 'string' }, available: { type: 'boolean' }, version: { type: 'string' } },
      },
    },
    receipt_path: { type: 'string' },
  },
}

const SMOKE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['tool', 'available', 'version', 'smoke_ran', 'real_input', 'output_path', 'crosscheck', 'receipt_path'],
  properties: {
    tool: { type: 'string' },
    available: { type: 'boolean' },
    version: { type: 'string' },
    smoke_ran: { type: 'boolean' },
    real_input: { type: 'string' },
    output_path: { type: 'string' },
    crosscheck: {
      type: 'object', additionalProperties: false,
      required: ['kind', 'fallback_ref', 'passed', 'detail'],
      properties: {
        kind: { type: 'string' },
        fallback_ref: { type: 'string' },
        passed: { type: 'boolean' },
        detail: { type: 'string' },
      },
    },
    receipt_path: { type: 'string' },
  },
}

const ASSEMBLE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['complete', 'present', 'missing', 'summary_path'],
  properties: {
    complete: { type: 'boolean' },
    present: { type: 'array', items: { type: 'string' } },
    missing: { type: 'array', items: { type: 'string' } },
    summary_path: { type: 'string' },
  },
}

// ===========================================================================
// PHASE 0 — Preflight: classify tool availability in the tools image
// ===========================================================================
phase('Preflight')
const pre = await agent(
  'Classify the deferred analysis tools inside container ' + TOOLS_CON + ' and capture --version receipts.\n' +
  'For EACH of: codeql, pict, atheris — determine availability and version via docker exec:\n' +
  '  - codeql:  docker exec ' + TOOLS_CON + ' codeql --version    (also: codeql resolve packs | head)\n' +
  '  - pict:    docker exec ' + TOOLS_CON + ' pict /usr/local/bin/pict  (PICT prints usage/version to stderr;\n' +
  '             availability = the binary runs; record the first version-ish line, else "present").\n' +
  '  - atheris: docker exec ' + TOOLS_CON + ' python3 -c "import atheris; print(getattr(atheris,\\"__version__\\",\\"present\\"))"\n' +
  'Write the combined receipt (raw stdout/stderr per tool) to host ' + HTOOLDIR + '/preflight_tools.json\n' +
  '(use the Write tool; create dirs as needed) and return PREFLIGHT_TOOLS_SCHEMA. available=true ONLY if the\n' +
  'command actually succeeded. Do not fail the phase if a tool is missing — just record available=false.\n\n' + SHARED,
  { phase: 'Preflight', schema: PREFLIGHT_TOOLS_SCHEMA, model: 'sonnet' })

const avail = Object.fromEntries((pre.optional_tools || []).map(t => [t.tool, t.available]))
log('Tools in ' + pre.container + ': ' + (pre.optional_tools || []).map(t => t.tool + '=' + t.available + '(' + t.version + ')').join(' '))

// ===========================================================================
// PHASE 1 — Smoke + cross-check, one agent per AVAILABLE tool (parallel)
// ===========================================================================
phase('Smoke')

const pictThunk = () => agent(
  'PICT SMOKE + cross-check (combinatorial covering arrays).\n' +
  'REAL INPUT: the stdlib covering array for surface ' + SURFACE + ':\n' +
  '  model:  ' + OUT + '/covering_array/model.json   (has parameters: {param:[values...]})\n' +
  '  cases:  ' + OUT + '/covering_array/cases.csv     (stdlib 2-way pairwise rows)\n\n' +
  'STEPS (all in-container via docker exec -w $PROJ ' + TOOLS_CON + '):\n' +
  '  1. Read model.json parameters. Emit a PICT model file ' + TOOLDIR + '/pict/model.pict where each line is\n' +
  '     "ParamName: v1, v2, ..." using the SAME params+values as the stdlib model (stringify booleans/null\n' +
  '     deterministically; keep a value<->token map so you can compare back).\n' +
  '  2. Run: pict ' + TOOLDIR + '/pict/model.pict > ' + TOOLDIR + '/pict/cases.tsv   (PICT emits TAB-separated,\n' +
  '     header row = param names). CAPTURE it.\n' +
  '  3. CROSS-CHECK (the deliverable): compute the set of all 2-way pairs {(parX=valA, parY=valB)} COVERED BY\n' +
  '     the stdlib cases.csv. Confirm EVERY such pair is ALSO covered by some PICT row (PICT ⊇ stdlib pairs).\n' +
  '     Report the pair counts and any pair the PICT output misses. passed=true iff no stdlib pair is missing.\n' +
  '  Write a small comparison script under ' + TOOLDIR + '/pict/ and run it in-container; do NOT hand-compute.\n\n' +
  'Write the receipt to host ' + HTOOLDIR + '/pict/receipt.json and return SMOKE_SCHEMA (tool="pict",\n' +
  'real_input=the model.json path, output_path=cases.tsv path, crosscheck.kind="pict_superset_of_stdlib_pairs",\n' +
  'fallback_ref=the cases.csv path).\n\n' + SHARED,
  { label: 'smoke:pict', phase: 'Smoke', schema: SMOKE_SCHEMA, model: 'sonnet' })

const codeqlThunk = () => agent(
  'CodeQL SMOKE + cross-check (interprocedural backward Python slice vs stdlib ast def-use).\n' +
  'REAL INPUT: the actual Tensile source + the stdlib def-use seed from branch_census.jsonl (surface ' + SURFACE + ').\n' +
  '  census: ' + OUT + '/branch_census.jsonl  (each line has id,file,function,location,referenced_symbols,derived_symbols)\n\n' +
  'Pick ONE concrete unit to slice — prefer a Tensile/Tensile.py branch (e.g. the alt_format / configPaths\n' +
  'predicate around L526/L529 if present in the census; otherwise the highest-rank Tensile.py unit). Note its\n' +
  'file, line, and the stdlib referenced_symbols+derived_symbols (this is the FALLBACK slice = the oracle).\n\n' +
  'STEPS (in-container via docker exec -w $PROJ ' + TOOLS_CON + '; KEEP IT BOUNDED <180s):\n' +
  '  1. Stage a SMALL source subset to a temp dir to keep DB-create fast: copy just the chosen file (and, if the\n' +
  '     slice obviously crosses into it, Tensile/Configuration.py and/or Tensile/Common/GlobalParameters.py) into\n' +
  '     e.g. /tmp/cqsrc preserving package layout. DO NOT build a DB over the whole tree for the smoke.\n' +
  '  2. codeql database create /tmp/cqdb --language=python --build-mode=none --source-root=/tmp/cqsrc --overwrite\n' +
  '  3. Slice query (write real QL under ' + TOOLDIR + '/codeql/slice.ql with a qlpack.yml depending on\n' +
  '     codeql/python-all). Achievable recipe: `import python`; locate the enclosing Function and the predicate\n' +
  '     at the chosen line; collect every `Name` it reads, then follow def-use via `SsaVariable`/`getDefinition()`\n' +
  '     and the variables those definitions read (transitively, bounded depth) to get the dependency symbol set —\n' +
  '     this reaches BEYOND a single intra-function statement, which is the interprocedural/cross-statement win\n' +
  '     over the stdlib ast seed. Run with `codeql query run --database=/tmp/cqdb` (or database analyze) and\n' +
  '     capture the symbol set to ' + TOOLDIR + '/codeql/slice.json.\n' +
  '  4. CROSS-CHECK (the deliverable): assert the CodeQL symbol set ⊇ the stdlib referenced_symbols set for that\n' +
  '     unit (CodeQL should resolve at least what the intra-function ast def-use found; ideally MORE, across\n' +
  '     functions). Report any stdlib symbol CodeQL missed (would be a real finding) and any extra symbol only\n' +
  '     CodeQL found (the interprocedural win). passed=true iff CodeQL set ⊇ stdlib set.\n' +
  '  If CodeQL DB-create or query genuinely cannot complete in budget, report smoke_ran=false with the exact\n' +
  '  error in crosscheck.detail and passed=false — do NOT fabricate a slice.\n\n' +
  'Write the receipt to host ' + HTOOLDIR + '/codeql/receipt.json and return SMOKE_SCHEMA (tool="codeql",\n' +
  'crosscheck.kind="codeql_superset_of_ast_defuse", fallback_ref=the branch_census.jsonl path + chosen branch_id).\n\n' + SHARED,
  { label: 'smoke:codeql', phase: 'Smoke', schema: SMOKE_SCHEMA })

const atherisThunk = () => agent(
  'Atheris SMOKE + cross-check (coverage-guided fuzz of an extracted PURE helper).\n' +
  'REAL INPUT: the canonical pchaos pure helper for the alt_format/configPaths branch:\n' +
  '  def alt_format_rejected(alt_format: bool, n_config_files: int) -> bool:\n' +
  '      return alt_format and n_config_files > 2\n' +
  '(If a Solve fragment under ' + OUT + '/_frags/Solve/ contains a pure_helper, prefer that real one and say which.)\n\n' +
  'STEPS (in-container via docker exec -w $PROJ ' + TOOLS_CON + '):\n' +
  '  1. Write an atheris harness under ' + TOOLDIR + '/atheris/fuzz_helper.py that feeds fuzzed (bool,int) inputs\n' +
  '     into the helper and asserts a KNOWN invariant you expect to hold, chosen so atheris can find the boundary\n' +
  '     witness (e.g. assert that result implies n_config_files>2; or search for an input where result==True).\n' +
  '  2. Run TIME-BOXED: docker exec -w $PROJ ' + TOOLS_CON + ' python3 ' + TOOLDIR + '/atheris/fuzz_helper.py \\\n' +
  '       -max_total_time=30 -runs=200000   (the time box IS the bound). Capture the run + any crash/witness file.\n' +
  '  3. CROSS-CHECK: take any witness atheris produces (e.g. the (True,3) boundary) and RE-EXECUTE it directly\n' +
  '     against the plain helper to confirm the behavior reproduces (same Verify discipline as SAT witnesses).\n' +
  '     passed=true iff atheris ran AND the witness reproduces against the real predicate.\n\n' +
  'If atheris is NOT importable in this image (3.12 build may have failed), return available=false, smoke_ran=false,\n' +
  'crosscheck.passed=false with detail "atheris unavailable; Validate falls back to Hypothesis" — this is an\n' +
  'ACCEPTED outcome per the plan, NOT a failure to hide.\n\n' +
  'Write the receipt to host ' + HTOOLDIR + '/atheris/receipt.json and return SMOKE_SCHEMA (tool="atheris",\n' +
  'crosscheck.kind="atheris_witness_reproduces", fallback_ref="Hypothesis").\n\n' + SHARED,
  { label: 'smoke:atheris', phase: 'Smoke', schema: SMOKE_SCHEMA, model: 'sonnet' })

const thunkByTool = { pict: pictThunk, codeql: codeqlThunk, atheris: atherisThunk }
// Always run pict + codeql smokes (core deliverable); run atheris only if importable,
// else synthesize the accepted-unavailable receipt without burning an agent.
const toRun = ['pict', 'codeql'].filter(t => avail[t]).map(t => thunkByTool[t])
if (avail.atheris) toRun.push(atherisThunk)
// If a core tool is unavailable, still spawn its agent so the receipt records WHY.
for (const t of ['pict', 'codeql']) if (!avail[t]) toRun.push(thunkByTool[t])

const smokes = (await parallel(toRun)).filter(Boolean)
for (const s of smokes) log('smoke ' + s.tool + ': ran=' + s.smoke_ran + ' crosscheck(' + s.crosscheck.kind + ')=' + s.crosscheck.passed)

// ===========================================================================
// PHASE 2 — Assemble tooling summary
// ===========================================================================
phase('Assemble')
const asm = await agent(
  'ASSEMBLE the tooling receipt bundle. Read host ' + HTOOLDIR + '/preflight_tools.json and each tool receipt\n' +
  '(' + HTOOLDIR + '/{pict,codeql,atheris}/receipt.json where present).\n\n' +
  'Write host ' + HTOOLDIR + '/tooling_summary.json with shape:\n' +
  '  {generated_for_surface, container, tools:[{tool,available,version,smoke_ran,crosscheck_kind,crosscheck_passed,\n' +
  '   receipt_path}], notes}. notes MUST state honestly: which tools are real-output-verified, which are\n' +
  '   accepted-unavailable (atheris fallback=Hypothesis), and that the stdlib fallbacks remain the oracle/floor.\n' +
  'Also write host ' + HTOOLDIR + '/README-tooling.md : a short human-facing summary (what each receipt proves,\n' +
  'the cross-check result per tool, how to rebuild the image: `docker build -f env/Dockerfile.tools -t\n' +
  'tl-pchaos-tools work/tensilelite-characterization/env`).\n\n' +
  'THEN verify these exist + non-empty under host ' + HTOOLDIR + ': preflight_tools.json, tooling_summary.json,\n' +
  'README-tooling.md, and a receipt.json for every tool reported available=true in preflight.\n' +
  'Return ASSEMBLE_SCHEMA (summary_path=' + HTOOLDIR + '/tooling_summary.json).\n\n' + SHARED,
  { phase: 'Assemble', schema: ASSEMBLE_SCHEMA, model: 'sonnet' })

return {
  ok: asm.complete,
  container: pre.container,
  optionalTools: pre.optional_tools,
  smokes: smokes.map(s => ({ tool: s.tool, available: s.available, smoke_ran: s.smoke_ran, crosscheck_passed: s.crosscheck.passed })),
  toolingSummary: asm.summary_path,
  deliverablesMissing: asm.missing,
}
