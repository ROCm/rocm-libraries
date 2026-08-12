// mutmut-slice-selftest.mjs — pure (no-spawn) checks of the orchestrator plan.
// Runs in the review sandbox: it only imports buildPlan/validateConfig/toolsInPlan
// and asserts the plan shape. It never spawns a subprocess and never touches
// pyproject.toml.
import { buildPlan, validateConfig, toolsInPlan } from '../mutmut-slice.js'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const HERE = dirname(fileURLToPath(import.meta.url))
const CFG = JSON.parse(readFileSync(join(HERE, 'fixtures', 'slice2-libraryio-config.json'), 'utf8'))

let fail = 0
const ok = (m) => console.log('ok   - ' + m)
const bad = (m) => { console.log('BAD  - ' + m); fail = 1 }

// --- config validation rejects missing required fields ---
const expectThrow = (fn, pat, label) => {
  try { fn() } catch (e) { return pat.test(e.message) ? ok(label) : bad(`${label}: wrong error ${e.message}`) }
  bad(`${label}: no error`)
}
expectThrow(() => validateConfig({}), /slice_id is required/, 'missing slice_id rejected')
expectThrow(() => validateConfig({ slice_id: 2, test_selection: ['x'] }), /only_mutate/, 'missing only_mutate rejected')
expectThrow(() => validateConfig({ slice_id: 2, only_mutate: ['x'] }), /test_selection/, 'missing test_selection rejected')

// --- defaults applied ---
const dc = validateConfig({ slice_id: 9, only_mutate: ['Tensile/Foo.py'], test_selection: ['t'] })
dc.src_rel === 'projects/hipblaslt/tensilelite' ? ok('src_rel defaults') : bad('src_rel default wrong')
dc.container === 'tl-mut' ? ok('container defaults to tl-mut') : bad('container default wrong')
dc.slug === 'foo' ? ok('slug derived from only_mutate[0]') : bad('slug wrong: ' + dc.slug)
dc.out.endsWith('slices/9-foo') ? ok('out derived when omitted') : bad('out default wrong: ' + dc.out)

// --- plan for the fixture ---
const plan = buildPlan(CFG)
const phaseNums = plan.phases.map((p) => p.n)
JSON.stringify(phaseNums) === JSON.stringify([0, 1, 2, 3, 4, 5]) ? ok('phases 0-5 present in order') : bad('phase ordering wrong: ' + phaseNums)

// all five tools referenced
const tools = toolsInPlan(plan)
;['preflight', 'pyproject', 'adapter', 'triage', 'verify'].forEach((t) =>
  tools.has(t) ? ok(`plan invokes ${t}`) : bad(`plan missing ${t}`))

// config values thread into commands
const allCmds = plan.phases.flatMap((p) => p.steps.map((s) => s.cmd)).join('\n')
allCmds.includes('Tensile/LibraryIO.py') ? ok('only_mutate LibraryIO.py appears in commands') : bad('LibraryIO.py missing from plan')
allCmds.includes('Tensile/Tests/unit/characterization/LibraryIO') ? ok('test_selection appears in commands') : bad('test_selection missing from plan')
allCmds.includes('--slice 2') ? ok('slice_id threads into preflight') : bad('slice_id not in preflight cmd')

// concurrency: mutmut run is serial; triage is the only fan-out
const mutRun = plan.phases[1].steps.find((s) => s.cmd.includes('mutmut run'))
mutRun && mutRun.serial ? ok('mutmut run is a serial actor') : bad('mutmut run not serial')
mutRun && /--max-children \d+/.test(mutRun.cmd) ? ok('mutmut run bounds --max-children (reproducibility control)') : bad('mutmut run missing --max-children cap')
const verifyStep = plan.phases[3].steps.find((s) => s.cmd.includes('mutmut-verify.sh'))
verifyStep && verifyStep.serial ? ok('verifier is a serial actor') : bad('verifier not serial')
const fanouts = plan.phases.flatMap((p) => p.steps).filter((s) => s.serial === false && s.kind !== 'placeholder')
fanouts.length === 1 && fanouts[0].kind === 'workflow' ? ok('only triage-workflow fans out') : bad('unexpected fan-out step(s)')

// restore present in Phase 5
const restore = plan.phases[5].steps.find((s) => s.cmd.includes('pyproject-mutmut.sh restore'))
restore ? ok('Phase 5 restores pyproject') : bad('no restore step in Phase 5')

// determinism
JSON.stringify(buildPlan(CFG)) === JSON.stringify(buildPlan(CFG)) ? ok('plan is deterministic') : bad('plan non-deterministic')

console.log('')
if (fail === 0) { console.log('ALL SELFTESTS PASSED'); process.exit(0) }
else { console.log('SELFTESTS FAILED'); process.exit(1) }
