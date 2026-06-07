export const meta = {
  name: 'parametric-chaos-characterize',
  description: 'v2 Tensile pipeline (public-input surface): ast census + Configuration.py constraint harvest -> backward-slice -> domains -> Z3/CrossHair witnesses -> adversarial verify -> stdlib pairwise covering array -> reify add-only char tests. Returns a scorecard; the DRIVER runs the gate + commits.',
  phases: [
    { title: 'Preflight', detail: 'tl-char up, /work mounted, inputs exist, modules import, pip layer, optional-tool classify, git baseline -> preflight.json' },
    { title: 'Inventory', detail: 'branch_extractor.py (ast census + def-use) + harvest_constraints.py -> file_inventory.csv, branch_census.jsonl, constraints_harvested.jsonl; top-N ranked units' },
    { title: 'Slice', detail: 'backward-slice each branch to public inputs, source-category tags -> _frags/Slice' },
    { title: 'Domain', detail: 'seed domains from literals/type-hints/yaml typing/env semantics -> _frags/Domain' },
    { title: 'Solve', detail: 'Z3/PySMT witness + UNSAT; CrossHair on pure helper; classify static/under-assumptions/runtime -> _frags/Solve' },
    { title: 'Verify', detail: 'adversarial re-exec of witness vs real fn; re-derive UNSAT; downgrade unconfirmed to UNKNOWN -> _frags/Verify' },
    { title: 'Reify', detail: 'one add-only pytest per confirmed SAT witness under PublicInputSurface/ -> _frags/Reify' },
    { title: 'Combinatorial', detail: 'covering_array.py: stdlib pairwise over reduced domains -> covering_array/' },
    { title: 'Assemble', detail: 'concat fragments -> 13 canonical deliverables; verify each non-empty; scorecard.json' },
  ],
}

// ---------------------------------------------------------------------------
// Paths (container; /work == worktree root). Source paths begin with Tensile/.
// ---------------------------------------------------------------------------
const PROJ = '/work/projects/hipblaslt/tensilelite'
const WF = '/work/work/tensilelite-characterization/wf/parametric-chaos'
// SURFACE names the deliverable/test bucket so distinct runs (public-input, deeper, codegen
// residue) never clobber each other. Default = Run-1's PublicInputSurface (Run-1 reproducible).
const SURFACE = args?.surface ?? 'PublicInputSurface'
const OUT = '/work/work/tensilelite-characterization/parametric-chaos/' + SURFACE
const FRAGS = OUT + '/_frags'
const TESTDIR = PROJ + '/Tensile/Tests/unit/characterization/' + SURFACE
// Host equivalents (Write tool writes to host; the mount makes them visible in-container):
const HOUT = 'work/tensilelite-characterization/parametric-chaos/' + SURFACE

const files = args?.files ?? [
  'Tensile/Tensile.py', 'Tensile/Configuration.py',
  'Tensile/Common/GlobalParameters.py', 'Tensile/CustomYamlLoader.py',
]
const maxUnits = args?.maxUnits ?? 20
// Constraint-harvest scan targets (modules with their own AST-constraint machinery).
const scan = args?.scan ?? ['Tensile/Configuration.py', 'Tensile/TensileBenchmarkCluster.py']

const SHARED = [
  'ENVIRONMENT (paths are INSIDE container tl-char unless prefixed "host:"):',
  '  CON=tl-char ; PROJ=' + PROJ + ' (cwd for every docker exec: pass -w $PROJ)',
  '  WF=' + WF + '  (the committed helper scripts: branch_extractor.py, harvest_constraints.py, covering_array.py)',
  '  OUT=' + OUT + '  (deliverable root) ; FRAGS=' + FRAGS + '  (per-phase fragments)',
  '  TESTDIR=' + TESTDIR + '  (reified add-only tests live here)',
  '  host: /work == worktree root. Output host dir = ' + HOUT + ' . You MAY write fragment JSON',
  '        with the Write tool to the host path (the mount makes it appear in-container), OR via',
  '        docker exec heredoc. EXECUTION (python/z3/pytest) MUST run in-container via docker exec.',
  '',
  'HARD RULES (memory: tensilelite-char-env, rigor-no-skewing-goldens, no-push-local-proof-first):',
  '  1. ADD-ONLY. Never modify/delete an existing repo file. New files only under OUT/ and TESTDIR/.',
  '  2. --cov=Tensile is a PATH, never a dotted module (rocisa nanobind SIGABRT). For a pass-check you',
  '     usually need NO coverage at all: pytest -p no:cacheprovider -m unit -q <node>.',
  '  3. Tests: pytestmark = pytest.mark.unit ; CPU-only ; UNIQUE basename ; deterministic.',
  '  4. NEVER push, NEVER commit, NEVER run the whole -m unit suite, NEVER start a Monitor. The driver',
  '     (main thread) owns the gate + commit. Keep each unit of work bounded (<~180s).',
  '  5. Measure, do not inflate. Pin ACTUAL behavior. If something cannot be solved/confirmed, say so',
  '     and downgrade — never fabricate a witness or a passing assertion.',
  '',
  'PYTHON-IN-CONTAINER PATTERN:',
  "  docker exec -w $PROJ tl-char python - <<'PY'   # heredoc; imports resolve from PROJ",
  '  ... your analysis ...',
  '  PY',
].join('\n')

// ---------------------------------------------------------------------------
// Concrete schemas (Run-1 Execution Contract).
// ---------------------------------------------------------------------------
const PREFLIGHT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['ok', 'checks', 'optional_tools', 'preflight_path'],
  properties: {
    ok: { type: 'boolean' },
    checks: {
      type: 'array', items: {
        type: 'object', additionalProperties: false,
        required: ['name', 'passed', 'detail'],
        properties: { name: { type: 'string' }, passed: { type: 'boolean' }, detail: { type: 'string' } },
      },
    },
    optional_tools: {
      type: 'array', items: {
        type: 'object', additionalProperties: false,
        required: ['tool', 'available'],
        properties: { tool: { type: 'string' }, available: { type: 'boolean' } },
      },
    },
    preflight_path: { type: 'string' },
  },
}

const UNIT = {
  type: 'object', additionalProperties: true,
  required: ['id', 'file', 'function', 'branch_kind', 'location', 'predicate_source', 'rank'],
  properties: {
    id: { type: 'string' }, file: { type: 'string' }, function: { type: 'string' },
    branch_kind: { type: 'string' },
    location: { type: 'object', properties: { line: { type: 'integer' }, col: { type: 'integer' } } },
    predicate_source: { type: 'string' },
    referenced_symbols: { type: 'array', items: { type: 'string' } },
    derived_symbols: { type: 'array' },
    rank: { type: 'integer' },
  },
}
const INVENTORY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['units', 'total_branches', 'constraints_harvested', 'op_surface_size', 'deliverables_written'],
  properties: {
    units: { type: 'array', items: UNIT },
    total_branches: { type: 'integer' },
    constraints_harvested: { type: 'integer' },
    op_surface_size: { type: 'integer' },
    deliverables_written: { type: 'array', items: { type: 'string' } },
  },
}

const HYPEREDGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['branch_id', 'predicate_normalized', 'public_inputs', 'derived_symbols', 'external_state', 'frag_path'],
  properties: {
    branch_id: { type: 'string' },
    predicate_normalized: { type: 'object' },
    public_inputs: {
      type: 'array', items: {
        type: 'object', additionalProperties: false, required: ['kind', 'name'],
        properties: {
          kind: { type: 'string', enum: ['cli', 'yaml', 'env', 'global-parameter', 'filesystem', 'os', 'gpu-probe', 'interactive', 'derived-local'] },
          name: { type: 'string' },
        },
      },
    },
    derived_symbols: {
      type: 'array', items: {
        type: 'object', additionalProperties: false, required: ['name', 'derived_from'],
        properties: { name: { type: 'string' }, derived_from: { type: 'string' } },
      },
    },
    external_state: { type: 'array', items: { type: 'string' } },
    frag_path: { type: 'string' },
  },
}

const DOMAIN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['branch_id', 'domains', 'frag_path'],
  properties: {
    branch_id: { type: 'string' },
    domains: {
      type: 'object',
      additionalProperties: {
        type: 'object', additionalProperties: false,
        properties: {
          type: { type: 'string', enum: ['bool', 'int', 'float', 'str', 'enum'] },
          min: { type: ['number', 'null'] }, max: { type: ['number', 'null'] },
          values: { type: 'array' },
        },
        required: ['type'],
      },
    },
    frag_path: { type: 'string' },
  },
}

const PREDICATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['branch_id', 'solver', 'solver_status', 'classification', 'true_examples', 'false_examples', 'frag_path'],
  properties: {
    branch_id: { type: 'string' },
    solver: { type: 'string', enum: ['z3', 'pysmt', 'crosshair', 'manual'] },
    solver_status: { type: 'string', enum: ['sat', 'sat-bounded', 'unsat', 'unknown'] },
    classification: { type: 'string', enum: ['fully-static', 'solver-backed-under-assumptions', 'runtime-dependent'] },
    true_examples: { type: 'array' },
    false_examples: { type: 'array' },
    pure_helper: { type: ['string', 'null'] },
    notes: { type: 'string' },
    frag_path: { type: 'string' },
  },
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['branch_id', 'confirmed', 'status', 'method', 'downgraded_to', 'frag_path'],
  properties: {
    branch_id: { type: 'string' },
    confirmed: { type: 'boolean' },
    status: { type: 'string', enum: ['SAT', 'UNSAT', 'UNKNOWN'] },
    method: { type: 'string' },
    downgraded_to: { type: ['string', 'null'], enum: ['unknown', null] },
    frag_path: { type: 'string' },
  },
}

const TEST_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['branch_id', 'reified', 'test_paths', 'passed', 'frag_path'],
  properties: {
    branch_id: { type: 'string' },
    reified: { type: 'boolean' },
    test_paths: { type: 'array', items: { type: 'string' } },
    kind: { type: 'string' },
    passed: { type: 'boolean' },
    reason: { type: 'string' },
    frag_path: { type: 'string' },
  },
}

const COMBO_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['parameters', 'cases', 'domain_fragments_read'],
  properties: {
    parameters: { type: 'integer' }, cases: { type: 'integer' },
    domain_fragments_read: { type: 'integer' }, param_names: { type: 'array', items: { type: 'string' } },
  },
}

const ASSEMBLE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['complete', 'present', 'missing'],
  properties: {
    complete: { type: 'boolean' },
    present: { type: 'array', items: { type: 'string' } },
    missing: { type: 'array', items: { type: 'string' } },
  },
}

// ===========================================================================
// PHASE 0 — Preflight (blocks the whole run on a mandatory-check failure)
// ===========================================================================
phase('Preflight')
const pre = await agent(
  'Run preflight for the parametric-chaos Run-1 pipeline. Execute these checks in container tl-char\n' +
  'and write the result to host path ' + HOUT + '/preflight.json (use the Write tool), then return it.\n\n' +
  'MANDATORY checks (all must pass; if any fails set ok=false):\n' +
  '  - container tl-char responds: `docker exec tl-char true`\n' +
  '  - /work mounted + cwd resolves: `docker exec -w ' + PROJ + ' tl-char pwd`\n' +
  '  - each input file exists: ' + files.join(', ') + '  (docker exec -w $PROJ tl-char test -f <f>)\n' +
  '  - each target module imports: docker exec -w $PROJ tl-char python -c "import ast" and for the\n' +
  '    source modules a syntax check `python -m py_compile <f>` (a full import of Tensile.Tensile is\n' +
  '    heavy/side-effecting — py_compile is the mandatory check; note import availability separately).\n' +
  '  - mandatory pip layer imports: docker exec tl-char python -c "import ast,z3,crosshair,hypothesis,pysmt"\n' +
  '  - the 3 helper scripts exist under ' + WF + ' and `python -m py_compile` each.\n' +
  '  - output dir writable: create ' + OUT + '/_frags/{Census,Slice,Domain,Solve,Verify,Reify} (mkdir -p).\n' +
  '  - git baseline captured: host `git -C /home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-coverage status --porcelain`\n' +
  '    -> store the line count in a check detail (this is the pre-write baseline).\n' +
  'OPTIONAL tools (classify available/unavailable, never fail on these): codeql, acts (java -jar), pict, daikon, atheris.\n\n' +
  'preflight.json shape: {ok, checks:[{name,passed,detail}], optional_tools:[{tool,available}], git_baseline_lines:N}.\n' +
  'Return PREFLIGHT_SCHEMA with preflight_path=' + HOUT + '/preflight.json.\n\n' + SHARED,
  { phase: 'Preflight', schema: PREFLIGHT_SCHEMA, model: 'sonnet' })

if (!pre || !pre.ok) {
  log('PREFLIGHT FAILED — aborting before any analysis agent spawns. See preflight.json.')
  throw new Error('preflight failed: ' + JSON.stringify(pre?.checks?.filter(c => !c.passed) ?? 'no result'))
}
log('Preflight OK. Optional tools: ' + (pre.optional_tools || []).map(t => t.tool + '=' + t.available).join(' '))

// ===========================================================================
// PHASE 1 — Inventory (deterministic helpers do the heavy lifting)
// ===========================================================================
phase('Inventory')
const census = await agent(
  'Run the Inventory phase by invoking the two committed helper scripts in tl-char (do NOT re-implement them):\n\n' +
  '1) Census + def-use:\n' +
  '   docker exec -w $PROJ tl-char python ' + WF + '/branch_extractor.py \\\n' +
  '     --root $PROJ --outdir ' + OUT + ' --max-units ' + maxUnits + ' ' + files.join(' ') + '\n' +
  '   -> writes ' + OUT + '/file_inventory.csv and ' + OUT + '/branch_census.jsonl ; prints JSON\n' +
  '      {units:[...], total_branches, files} on stdout. CAPTURE that stdout.\n\n' +
  '2) Constraint harvest:\n' +
  '   docker exec -w $PROJ tl-char python ' + WF + '/harvest_constraints.py \\\n' +
  '     --root $PROJ --outdir ' + OUT + ' --scan ' + scan.join(' ') + '\n' +
  '   -> writes ' + OUT + '/constraints_harvested.jsonl ; prints {constraints_harvested, op_surface_size,...}.\n\n' +
  'Then for EACH returned unit, write a census fragment to host ' + HOUT + '/_frags/Census/<branch_id>.json\n' +
  '(the unit object verbatim). Verify file_inventory.csv, branch_census.jsonl, constraints_harvested.jsonl\n' +
  'are all non-empty in-container.\n\n' +
  'Return INVENTORY_SCHEMA: units (the top ' + maxUnits + ' from script stdout, verbatim — preserve id/file/\n' +
  'function/branch_kind/location/predicate_source/referenced_symbols/derived_symbols/rank), total_branches,\n' +
  'constraints_harvested, op_surface_size, deliverables_written (the 3 filenames).\n\n' + SHARED,
  { phase: 'Inventory', schema: INVENTORY_SCHEMA, model: 'sonnet' })

log('Inventory: ' + census.units.length + ' units (of ' + census.total_branches + ' branches); ' +
    census.constraints_harvested + ' constraints harvested; op-surface ' + census.op_surface_size + ' node types.')

// ===========================================================================
// PHASES 2-7 — pipeline per unit: Slice -> Domain -> Solve -> Verify -> Reify
// (pipeline = no barrier; unit A can be in Verify while unit B is still in Slice)
// ===========================================================================
const results = await pipeline(census.units,

  // ---- Slice -----------------------------------------------------------
  (u) => agent(
    'SLICE unit ' + u.id + '  (' + u.file + ':' + u.location.line + ' [' + u.branch_kind + '])\n' +
    'Predicate: ' + u.predicate_source + '\n' +
    'Referenced symbols: ' + JSON.stringify(u.referenced_symbols || []) + '\n' +
    'Seed def-use (from the extractor): ' + JSON.stringify(u.derived_symbols || []) + '\n\n' +
    'Backward-slice each referenced symbol to its PUBLIC INPUT. Read ' + u.file + ' around line ' +
    u.location.line + ' (and the enclosing function ' + u.function + ') in-container to confirm derivations.\n' +
    'Tag every public input with a source-category: cli | yaml | env | global-parameter | filesystem | os |\n' +
    'gpu-probe | interactive | derived-local. Use the extractor def-use seed as a starting point; refine it.\n' +
    'List any external_state the predicate truly depends on (os.environ keys, filesystem probes, hw probes).\n\n' +
    'Write the HYPEREDGE record to host ' + HOUT + '/_frags/Slice/' + u.id + '.json and return HYPEREDGE_SCHEMA\n' +
    '(frag_path=' + HOUT + '/_frags/Slice/' + u.id + '.json). predicate_normalized: the normalized predicate AST\n' +
    '(reuse the extractor census record if convenient).\n\n' + SHARED,
    { label: 'slice:' + u.file.split('/').pop() + ':' + u.location.line, phase: 'Slice', schema: HYPEREDGE_SCHEMA, model: 'sonnet' }),

  // ---- Domain ----------------------------------------------------------
  (edge, u) => agent(
    'DOMAIN seeding for unit ' + u.id + '  (' + u.file + ':' + u.location.line + ')\n' +
    'Predicate: ' + u.predicate_source + '\n' +
    'Slice (public inputs + derivations): ' + JSON.stringify(edge) + '\n\n' +
    'Infer a small, concrete domain for EACH symbol that appears in the predicate, from: integer/str literals\n' +
    'in the comparison, type hints, YAML scalar typing (CustomYamlLoader -> bool|None|int|float|str),\n' +
    'enumerated config values, and known env-var semantics. Prefer the smallest domain that exercises both\n' +
    'sides of the branch (e.g. for `len(configPaths) > 2`, model ConfigFile_count:int min 0; for a bool flag,\n' +
    'type bool). Read the source if unsure. Keep value sets <= 4.\n\n' +
    'Write the DOMAIN record to host ' + HOUT + '/_frags/Domain/' + u.id + '.json and return DOMAIN_SCHEMA\n' +
    '(frag_path set). domains maps SYMBOL -> {type, min?, max?, values?}.\n\n' + SHARED,
    { label: 'domain:' + u.file.split('/').pop() + ':' + u.location.line, phase: 'Domain', schema: DOMAIN_SCHEMA, model: 'sonnet' }),

  // ---- Solve (session model: Z3/PySMT/CrossHair) -----------------------
  (dom, u) => agent(
    'SOLVE unit ' + u.id + '  (' + u.file + ':' + u.location.line + ' [' + u.branch_kind + '])\n' +
    'Predicate: ' + u.predicate_source + '\n' +
    'Domains: ' + JSON.stringify(dom.domains) + '\n' +
    '(slice fragment on disk: ' + HOUT + '/_frags/Slice/' + u.id + '.json)\n\n' +
    'GOAL: produce a TRUE-witness and a FALSE-witness (or prove UNSAT) for this branch predicate, then\n' +
    'CLASSIFY it. Steps, in-container (use z3; pysmt/crosshair if helpful):\n' +
    '  1. Translate the predicate into Z3 over the seeded domains. Harvested op-surface (boolean/compare/\n' +
    '     arith/bit/unary/conditional) is the encoder boundary — mirror Configuration.py ExpressionEvaluator.\n' +
    '  2. If the predicate is tractable (boolean/integer/comparison over cli/yaml/int/bool inputs): solve for\n' +
    '     a model where the predicate is True (true_examples) and one where it is False (false_examples).\n' +
    '     Report solver_status sat / sat-bounded (if you bounded an int range) / unsat.\n' +
    '  3. If it depends on runtime/external state (isinstance on live objects, nodeType dispatch over a parsed\n' +
    '     tree, os/filesystem/gpu probes): classification=runtime-dependent, solver_status=unknown, and give\n' +
    '     a representative true/false example by VALUE (not solver-proven) so downstream can still pin behavior.\n' +
    '  4. If you can extract a PURE helper (e.g. for Tensile.py:526 -> `def alt_format_rejected(alt_format: bool,\n' +
    '     n_config_files: int) -> bool: return alt_format and n_config_files > 2`), run CrossHair on it and put\n' +
    '     its source in pure_helper. CrossHair: absence of counterexample != proof — note that.\n' +
    '  classification in {fully-static, solver-backed-under-assumptions, runtime-dependent}.\n\n' +
    'Actually RUN z3 in-container and use the real model output — do not hand-wave. Write the PREDICATE record\n' +
    'to host ' + HOUT + '/_frags/Solve/' + u.id + '.json and return PREDICATE_SCHEMA (frag_path set).\n\n' + SHARED,
    { label: 'solve:' + u.file.split('/').pop() + ':' + u.location.line, phase: 'Solve', schema: PREDICATE_SCHEMA }),

  // ---- Verify (session model, adversarial, default-skeptical) ----------
  (pred, u) => agent(
    'VERIFY (ADVERSARIAL, default-skeptical) unit ' + u.id + '  (' + u.file + ':' + u.location.line + ')\n' +
    'Predicate: ' + u.predicate_source + '\n' +
    'Claimed solver result: ' + JSON.stringify(pred) + '\n\n' +
    'You are the CHECKER, not the doer. Independently re-establish the claim against the REAL code:\n' +
    '  - If solver_status is sat/sat-bounded: re-execute the claimed true_examples / false_examples against the\n' +
    '    ACTUAL predicate (extract the real expression or call the real function/helper in-container) and confirm\n' +
    '    each evaluates as claimed. If ANY example fails to reproduce, confirmed=false, downgrade status to UNKNOWN.\n' +
    '  - If solver_status is unsat: independently re-derive unsat (try to find any counter-model); only confirm if\n' +
    '    you also find none. status=UNSAT when confirmed.\n' +
    '  - If runtime-dependent/unknown: confirmed=false unless you can actually exhibit the behavior; status=UNKNOWN.\n' +
    'NEVER upgrade a weaker claim. When in doubt, downgrade to UNKNOWN and say why in method.\n' +
    'status in {SAT, UNSAT, UNKNOWN}; downgraded_to is "unknown" if you downgraded, else null.\n\n' +
    'Write the VERDICT record to host ' + HOUT + '/_frags/Verify/' + u.id + '.json and return VERDICT_SCHEMA.\n\n' + SHARED,
    { label: 'verify:' + u.file.split('/').pop() + ':' + u.location.line, phase: 'Verify', schema: VERDICT_SCHEMA }),

  // ---- Reify (only confirmed SAT witnesses; one agent -> one test file) -
  (verdict, u) => (verdict && verdict.confirmed && verdict.status === 'SAT')
    ? agent(
        'REIFY unit ' + u.id + '  (' + u.file + ':' + u.location.line + ' [' + u.branch_kind + '])\n' +
        'Confirmed verdict: ' + JSON.stringify(verdict) + '\n' +
        'Predicate: ' + u.predicate_source + '\n' +
        'Solve fragment (witness + pure_helper) on disk: ' + HOUT + '/_frags/Solve/' + u.id + '.json\n\n' +
        'Reify the CONFIRMED witness into add-only pytest char test(s) under TESTDIR (' + TESTDIR + ').\n' +
        'Create TESTDIR if needed (mkdir -p in-container) and add an __init__.py if the sibling characterization\n' +
        'dirs have one (check). Unique basename: test_pchaos_' + u.file.split('/').pop().replace('.py', '') +
        '_L' + u.location.line + '_char.py. pytestmark = pytest.mark.unit. CPU-only. Pin ACTUAL behavior.\n\n' +
        'CANONICAL CASE: if this is Tensile.py:526 or Tensile.py:529, reify BOTH (Run-1 contract):\n' +
        '  (1) pure-helper test: define/import alt_format_rejected(alt_format,n_config_files) and assert the\n' +
        '      witness e.g. (True,3)->True plus a False case;\n' +
        '  (2) real-entry pin: call Tensile.Tensile(userArgs) with a tmp output path and ConfigFile args that\n' +
        '      trigger L526/L529, monkeypatch/observe printExit, assert pytest.raises(SystemExit) (or the patched\n' +
        '      call). Read Tensile.py argparse first to build correct userArgs; do NOT reach deeper config parsing.\n' +
        'For non-canonical units, a focused pure-assert test on the extracted predicate/helper is fine.\n\n' +
        'CONFIRM it passes in-container (NO coverage needed for a pass-check):\n' +
        '  docker exec -w $PROJ tl-char pytest -p no:cacheprovider -m unit -q <relative test path>\n' +
        'passed=true ONLY if it actually passes (0 failed). If it cannot be made to pass honestly, reified=false\n' +
        'with the reason — do NOT weaken the assertion to force a pass.\n\n' +
        'Write the TEST record to host ' + HOUT + '/_frags/Reify/' + u.id + '.json and return TEST_SCHEMA.\n\n' + SHARED,
        { label: 'reify:' + u.file.split('/').pop() + ':' + u.location.line, phase: 'Reify', schema: TEST_SCHEMA, model: 'sonnet' })
    : null,
)

// ===========================================================================
// PHASE 8 — Combinatorial (covering array over the reduced domains)
// ===========================================================================
phase('Combinatorial')
const combo = await agent(
  'Run the Combinatorial phase by invoking the committed helper in tl-char:\n' +
  '  docker exec -w $PROJ tl-char python ' + WF + '/covering_array.py \\\n' +
  '    --fragdir ' + FRAGS + '/Domain --outdir ' + OUT + ' --max-params 12\n' +
  '-> writes ' + OUT + '/covering_array/model.json and ' + OUT + '/covering_array/cases.csv ; prints a JSON\n' +
  'summary. Verify both files are non-empty. Return COMBO_SCHEMA from the printed summary.\n\n' + SHARED,
  { phase: 'Combinatorial', schema: COMBO_SCHEMA, model: 'sonnet' })
log('Combinatorial: ' + combo.parameters + ' params, ' + combo.cases + ' pairwise cases.')

// ===========================================================================
// PHASE 9 — Assemble (concat fragments -> deliverables; verify; scorecard)
// ===========================================================================
const reified = results.filter(Boolean)

phase('Assemble')
const asm = await agent(
  'ASSEMBLE the final deliverables for the parametric-chaos Run-1 bundle. Read the per-phase fragments under\n' +
  FRAGS + '/{Census,Slice,Domain,Solve,Verify,Reify} and the already-written files in ' + OUT + '.\n\n' +
  'PRODUCE (host paths under ' + HOUT + ', use Write tool; run any aggregation python in-container):\n' +
  '  - branch_parameter_hypergraph.json : {nodes:[branch records], edges:[slice hyperedges]} built by joining\n' +
  '    branch_census.jsonl with the Slice fragments (one edge per branch_id -> its public_inputs).\n' +
  '  - domain_model.json : { <branch_id>: <domains> } merged from all Domain fragments.\n' +
  '  - characterization_catalog.jsonl : one line per unit = the FULL v2 branch-record (census + slice + domains\n' +
  '    + solver witness + verdict + reified-test path if any), joined by branch_id across the fragment dirs.\n' +
  '  - validation_report.md : per-unit table (branch_id, file:line, classification, solver_status, confirmed,\n' +
  '    reified?), plus counts of SAT/UNSAT/UNKNOWN and confirmed witnesses, and the validation methods used\n' +
  '    (z3, crosshair, pytest pass-check). State runtime-dependent branches explicitly (never silently asserted).\n' +
  '  - analyst_summary.md : human-facing — clustered branch families, prioritized hotspots, the canonical\n' +
  '    Tensile.py:526/529 worked example, and caveats/blind spots.\n' +
  '  - README-analysis.md : repro instructions (the exact helper-script commands + this workflow), tool versions\n' +
  '    (python, z3, crosshair, hypothesis, pysmt — get them in-container), and the static/solver/runtime split.\n' +
  '  - scorecard.json : {branchesInventoried, totalBranches, constraintsHarvested, publicInputsMapped,\n' +
  '    satCount, unsatCount, unknownCount, witnessesConfirmed, coveringArrayRows, testsReified} computed from\n' +
  '    the fragments (these counts are AUTHORITATIVE).\n\n' +
  'THEN VERIFY this exact canonical deliverable list exists and is NON-EMPTY (in ' + OUT + '):\n' +
  '  preflight.json, file_inventory.csv, branch_census.jsonl, constraints_harvested.jsonl,\n' +
  '  branch_parameter_hypergraph.json, domain_model.json, characterization_catalog.jsonl,\n' +
  '  covering_array/model.json, covering_array/cases.csv, validation_report.md, analyst_summary.md,\n' +
  '  README-analysis.md, scorecard.json\n' +
  'Return ASSEMBLE_SCHEMA: complete (true iff none missing), present[], missing[].\n\n' + SHARED,
  { phase: 'Assemble', schema: ASSEMBLE_SCHEMA, model: 'sonnet' })

// ---------------------------------------------------------------------------
// Return scorecard (the workflow STOPS here; the DRIVER runs the gate + commits).
// ---------------------------------------------------------------------------
return {
  ok: pre.ok && asm.complete,
  files,
  maxUnits,
  branchesInventoried: census.units.length,
  totalBranches: census.total_branches,
  constraintsHarvested: census.constraints_harvested,
  opSurfaceSize: census.op_surface_size,
  coveringArrayParameters: combo.parameters,
  coveringArrayRows: combo.cases,
  testsReifiedCount: reified.filter(r => r.reified && r.passed).length,
  testsReifiedPaths: reified.filter(r => r.reified && r.passed).flatMap(r => r.test_paths || []),
  deliverablesComplete: asm.complete,
  deliverablesMissing: asm.missing,
  optionalTools: pre.optional_tools,
}
