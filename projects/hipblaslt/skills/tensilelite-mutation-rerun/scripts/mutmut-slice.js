// mutmut-slice.js — Issue 6: per-slice mutation orchestrator (Phases 0-5).
//
// Ties together the slice infrastructure:
//   - slice-preflight.sh        (env/base-pin capture)          Phase 0
//   - pyproject-mutmut.sh       (backup/set/restore/assert)     Phase 0 + 5
//   - mutmut run/results/show   (single serial mutation run)    Phase 1
//   - mutmut-results-adapter.py (survivors -> triage groups)    Phase 1
//   - triage-workflow.js        (parallel triage + kill-proof)  Phase 2/3
//   - mutmut-verify.sh          (strict serial kill-proof)      Phase 3
//
// DESIGN — pure plan + thin executor:
//   buildPlan(config) is a PURE function returning the ordered phase/step plan
//   (no spawn, no FS writes). `--dry-run` (the reviewable path) validates the
//   config and PRINTS that plan; it never spawns a subprocess and never touches
//   pyproject.toml. `--execute` runs the same plan for real. This split is
//   deliberate: the review sandbox blocks child_process spawning, so the
//   reviewed dry-run path must be subprocess-free.
//
// CONCURRENCY (enforced by phase ordering + serial flags):
//   - `mutmut run` is ONE serial actor (Phase 1).
//   - `mutmut apply` / mutmut-verify.sh is ONE serial actor (Phase 3).
//   - pyproject edits are ONE serial actor (Phase 0 set / Phase 5 restore).
//   - only triage/equivalence reasoning fans out, and only inside triage-workflow.js.
//
// FAILURE / RESTORE STRATEGY:
//   pyproject.toml is a tracked file mutated in Phase 0 (`set`). In --execute mode
//   the whole run is wrapped so Phase 5 `restore` is ATTEMPTED on any failure
//   after a backup was taken (best-effort; logged if it fails). --dry-run never
//   mutates pyproject, so there is nothing to restore.
//
// TRUST (--execute only): config field values are interpolated into shell command
//   strings that --execute runs. Supply a TRUSTED, operator-authored config; do
//   not feed untrusted input to --execute. --dry-run only prints these strings and
//   spawns nothing, so it is unaffected.
//
// USAGE:
//   node mutmut-slice.js --dry-run  --config <slice.json>   # review path (default)
//   node mutmut-slice.js --plan-only --config <slice.json>  # alias of --dry-run
//   node mutmut-slice.js --execute  --config <slice.json>   # real run (real runtime only)
//   node mutmut-slice.js --help

import { execSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'

const WF = 'projects/hipblaslt/tensilelite/Tensile/Tests/unit/mutation'
const TOOLS = {
  preflight: `${WF}/slice-preflight.sh`,
  pyproject: `${WF}/pyproject-mutmut.sh`,
  adapter: `${WF}/mutmut-results-adapter.py`,
  triage: `${WF}/triage-workflow.js`,
  verify: `${WF}/mutmut-verify.sh`,
}
const DEFAULTS = {
  src_rel: 'projects/hipblaslt/tensilelite',
  container: 'tl-mut',
  group_by: 'module_function',
  test_file_owner: 'one_file_per_function',
  max_children: 32,
}

function slugFor(mod) {
  const base = String(mod).split('/').pop().replace(/\.py$/, '')
  return base.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
}

// ------------------------------------------------------------- config
export function validateConfig(cfg) {
  if (!cfg || typeof cfg !== 'object' || Array.isArray(cfg)) throw new Error('config must be a JSON object')
  const errs = []
  if (cfg.slice_id === undefined || cfg.slice_id === null || String(cfg.slice_id) === '') errs.push('slice_id is required')
  if (!Array.isArray(cfg.only_mutate) || cfg.only_mutate.length === 0) errs.push('only_mutate must be a non-empty array')
  if (!Array.isArray(cfg.test_selection) || cfg.test_selection.length === 0) errs.push('test_selection must be a non-empty array')
  if (errs.length) throw new Error('invalid config: ' + errs.join('; '))
  const slice_id = cfg.slice_id
  const slug = slugFor(cfg.only_mutate[0])
  const src_rel = cfg.src_rel || DEFAULTS.src_rel
  return {
    slice_id,
    slug,
    src_commit: cfg.src_commit || null,
    only_mutate: cfg.only_mutate.slice(),
    test_selection: cfg.test_selection.slice(),
    out: cfg.out || `work/mutation/slices/${slice_id}-${slug}`,
    group_by: cfg.group_by || DEFAULTS.group_by,
    test_file_owner: cfg.test_file_owner || DEFAULTS.test_file_owner,
    src_rel,
    container: cfg.container || DEFAULTS.container,
    max_children: cfg.max_children || DEFAULTS.max_children,
  }
}

// ------------------------------------------------------------- plan
// step kinds: 'sh' (bash tool), 'python' (adapter), 'docker' (mutmut in container),
//             'workflow' (triage-workflow.js via the Workflow runner), 'placeholder'.
function step(kind, serial, cmd, note) {
  return { kind, serial, cmd, note }
}

export function buildPlan(cfg) {
  const c = validateConfig(cfg)
  const proj = '/work/' + c.src_rel
  const groups = `${c.out}/groups.json`
  const records = `${c.out}/survivors.json`
  const primaryMod = c.only_mutate[0]

  const phases = [
    {
      n: 0, name: 'prep',
      steps: [
        step('sh', true,
          `bash ${TOOLS.preflight} --slice ${c.slice_id} --module ${primaryMod} --out ${c.out} --container ${c.container} --src ${c.src_rel}`,
          'capture env/base-pin (env.json); fails if tracked source dirty or container missing'),
        step('sh', true, `bash ${TOOLS.pyproject} backup --src ${c.src_rel}`, 'snapshot [tool.mutmut] before edit'),
        step('sh', true,
          `bash ${TOOLS.pyproject} set --src ${c.src_rel} ` + c.only_mutate.map((m) => `--only-mutate ${m}`).join(' ') + ' ' + c.test_selection.map((t) => `--test-selection ${t}`).join(' '),
          'rewrite only_mutate + test-selection for this slice (tracked-file edit; restored in Phase 5)'),
      ],
    },
    {
      n: 1, name: 'execute',
      steps: [
        step('docker', true, `docker exec -w ${proj} ${c.container} mutmut run --max-children ${c.max_children}`, `SINGLE SERIAL mutation run (expensive; never during review). --max-children bounds worker self-contention so wall-clock timeouts and flaky-under-load kills stay out of the score (reproducibility control; see pyproject timeout_multiplier)`),
        step('docker', true, `docker exec -w ${proj} ${c.container} mutmut results > ${records}.raw`, 'capture survivor ids'),
        step('docker', true, `# for each survivor: docker exec -w ${proj} ${c.container} mutmut show <id>  -> build ${records}`, 'capture diffs/lines into SURVIVOR records JSON'),
        step('python', true, `python3 ${TOOLS.adapter} --fixture ${records} --src-root ${c.src_rel} > ${groups}`, 'survivors -> triage groups {module,function,source_file,char_dir,test_file,survivors[]}'),
      ],
    },
    {
      n: 2, name: 'triage',
      steps: [
        step('workflow', false,
          `Workflow runner: ${TOOLS.triage} with args={groups:<${groups}>, out:"${c.out}/workflow", test_selection:${JSON.stringify(c.test_selection)}, src_rel:"${c.src_rel}", con:"${c.container}"}`,
          'parallel per-function triage authors add-only tests (fan-out happens ONLY here)'),
      ],
    },
    {
      n: 3, name: 'verify',
      steps: [
        step('sh', true, `bash ${TOOLS.verify} --container ${c.container} --manifest ${c.out}/workflow/manifest.tsv --out ${c.out}/verify --src ${c.src_rel}`, 'SINGLE SERIAL strict kill-proof'),
        step('placeholder', true, `# equivalence audit (Phase 3a): review EQUIVALENT/BAD verdicts`, 'placeholder — serial equivalence reasoning'),
        step('placeholder', true, `# serial pragma/repair pass (via triage-workflow Pragma phase)`, 'placeholder — single serial actor'),
      ],
    },
    {
      n: 4, name: 'synthesize',
      steps: [
        step('placeholder', false, `# report hook: ${c.out}/workflow/{survivor-ledger.md,mutation-report.json,recommendations.md}`, 'placeholder — emitted by triage-workflow Synthesize phase'),
      ],
    },
    {
      n: 5, name: 'restore/certify',
      steps: [
        step('sh', true, `bash ${TOOLS.pyproject} restore --src ${c.src_rel}`, 'byte-restore pyproject.toml (ALWAYS attempted on failure in --execute)'),
        step('sh', true, `bash ${TOOLS.pyproject} assert-clean --src ${c.src_rel}`, 'gate: pyproject.toml == HEAD (unless deliberate allowlist)'),
        step('placeholder', false, `# certification hook: mark slice ${c.slice_id} complete in PLAN-MUTATION-COMPLETION.md`, 'placeholder'),
      ],
    },
  ]
  return { slice: c, phases }
}

// tools referenced by the plan (for the dry-run invocation map / selftest)
export function toolsInPlan(plan) {
  const found = new Set()
  const hay = JSON.stringify(plan)
  for (const [name, path] of Object.entries(TOOLS)) if (hay.includes(path)) found.add(name)
  return found
}

// ------------------------------------------------------------- dry-run print
function printPlan(plan) {
  const c = plan.slice
  console.log('mutmut-slice DRY-RUN (plan only; no mutmut run, no source edits)')
  console.log(`  slice_id        : ${c.slice_id}`)
  console.log(`  slug            : ${c.slug}`)
  console.log(`  src_commit      : ${c.src_commit || '(unpinned)'}`)
  console.log(`  src_rel         : ${c.src_rel}`)
  console.log(`  container       : ${c.container}`)
  console.log(`  only_mutate     : ${JSON.stringify(c.only_mutate)}`)
  console.log(`  test_selection  : ${JSON.stringify(c.test_selection)}`)
  console.log(`  out             : ${c.out}`)
  console.log(`  group_by        : ${c.group_by}`)
  console.log(`  test_file_owner : ${c.test_file_owner}`)
  console.log('')
  for (const ph of plan.phases) {
    console.log(`Phase ${ph.n} — ${ph.name}`)
    for (const s of ph.steps) {
      const tag = s.kind === 'placeholder' ? 'PLACEHOLDER' : (s.serial ? 'serial' : 'fan-out')
      console.log(`  [${tag}] ${s.cmd}`)
      console.log(`         ↳ ${s.note}`)
    }
    console.log('')
  }
  const tools = toolsInPlan(plan)
  console.log('Tool invocation map (all 5 must appear):')
  for (const name of ['preflight', 'pyproject', 'adapter', 'triage', 'verify']) {
    console.log(`  ${tools.has(name) ? '✓' : '✗'} ${name}  (${TOOLS[name]})`)
  }
  console.log('')
  console.log('DRY-RUN complete: no subprocess spawned, pyproject.toml untouched.')
}

// ------------------------------------------------------------- executor (real)
function execute(plan) {
  // Real run. Only reachable via --execute (never in review). Kept a thin,
  // honest skeleton: concrete tool steps run; expensive/destructive and
  // placeholder steps are guarded so a full campaign is never launched by
  // accident. pyproject restore is ALWAYS attempted after Phase 0 `set`.
  const c = plan.slice
  let backedUp = false
  const run = (cmd) => {
    console.log('+ ' + cmd)
    execSync(cmd, { stdio: 'inherit' })
  }
  try {
    for (const ph of plan.phases) {
      if (ph.n === 5) break // restore handled in finally
      for (const s of ph.steps) {
        if (s.kind === 'placeholder' || s.kind === 'workflow') { console.log(`(skip ${s.kind}) ${s.cmd}`); continue }
        if (s.cmd.includes('mutmut run')) throw new Error('refusing to launch `mutmut run` from this skeleton executor; wire deliberately before real campaigns')
        run(s.cmd)
        if (s.cmd.includes('pyproject-mutmut.sh backup')) backedUp = true
      }
    }
  } finally {
    if (backedUp) {
      try { execSync(`bash ${TOOLS.pyproject} restore --src ${c.src_rel}`, { stdio: 'inherit' }) }
      catch (e) { console.error('WARNING: pyproject restore failed: ' + e.message) }
    }
  }
}

// ------------------------------------------------------------- CLI
const HELP = `mutmut-slice.js — per-slice mutation orchestrator (Phases 0-5)

Usage:
  node mutmut-slice.js --dry-run  --config <slice.json>   review path (default); prints the phase plan, spawns nothing, does not touch pyproject.toml
  node mutmut-slice.js --plan-only --config <slice.json>  alias of --dry-run
  node mutmut-slice.js --execute  --config <slice.json>   real run (real runtime only)
  node mutmut-slice.js --help

Config (JSON): { slice_id, src_commit?, only_mutate[], test_selection[], out?, group_by?, test_file_owner?, src_rel?=projects/hipblaslt/tensilelite, container?=tl-mut }

Concurrency: mutmut run (Phase 1), verifier (Phase 3), and pyproject edits (Phase 0/5) are each a SINGLE serial actor; only triage-workflow.js fans out.
Failure/restore: in --execute, pyproject.toml restore is ALWAYS attempted after the Phase 0 backup, even on failure. --dry-run never mutates pyproject.
Trust: --execute runs shell commands built from config values; supply a trusted operator-authored config. --dry-run only prints them and spawns nothing.`

function parseArgs(argv) {
  const o = { mode: 'dry-run' }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--dry-run' || a === '--plan-only') o.mode = 'dry-run'
    else if (a === '--execute') o.mode = 'execute'
    else if (a === '--config') { if (i + 1 >= argv.length) throw new Error('--config requires a value'); o.config = argv[++i] }
    else if (a === '-h' || a === '--help') o.help = true
    else throw new Error('unknown argument: ' + a)
  }
  return o
}

function main() {
  let o
  try { o = parseArgs(process.argv.slice(2)) } catch (e) { process.stderr.write('mutmut-slice: ' + e.message + '\n'); process.exit(2) }
  if (o.help) { process.stdout.write(HELP + '\n'); process.exit(0) }
  if (!o.config) { process.stderr.write('mutmut-slice: --config <slice.json> is required (or --help)\n'); process.exit(2) }
  let cfg
  try { cfg = JSON.parse(readFileSync(o.config, 'utf8')) } catch (e) { process.stderr.write('mutmut-slice: cannot read config: ' + e.message + '\n'); process.exit(2) }
  let plan
  try { plan = buildPlan(cfg) } catch (e) { process.stderr.write('mutmut-slice: ' + e.message + '\n'); process.exit(1) }
  if (o.mode === 'execute') execute(plan)
  else printPlan(plan)
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  main()
}
