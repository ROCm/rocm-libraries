// slice1-regression-dryrun.mjs — Issue 8: slice-1 regression dry-run gate.
//
// Proves the assembled workflow can PLAN the certified slice-1 regression path
// before any real (slice-2+) mutation run, and that the certified pilot numbers
// (131 -> 4 survivors, all remaining equivalent) are recorded for a future real
// regression comparison.
//
// SPAWN-FREE by design: the review sandbox blocks child_process. This harness
// imports the orchestrator's pure exports (buildPlan/validateConfig/toolsInPlan)
// and asserts the dry-run plan shape in-process. It never runs `mutmut run`,
// never runs `mutmut apply`, never spawns a subprocess, and never touches
// pyproject.toml. Because buildPlan is pure (no FS writes, no spawn), planning
// the slice cannot dirty the tracked pyproject.toml; the separate
// `git diff --quiet -- pyproject.toml` check in the handback confirms the on-disk
// invariant for the CLI `--dry-run` run.
import { buildPlan, toolsInPlan } from '../mutmut-slice.js'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const HERE = dirname(fileURLToPath(import.meta.url))
const SLICE = join(HERE, 'fixtures', 'slice-1-regression')
const CFG = JSON.parse(readFileSync(join(SLICE, 'config.json'), 'utf8'))
const EXP = JSON.parse(readFileSync(join(SLICE, 'expected.json'), 'utf8'))

const SLICE1_FILES = [
  'Tensile/Common/Utilities.py',
  'Tensile/TensileLogic/ValidChipId.py',
  'Tensile/TensileLogic/ValidMatrixInstruction.py',
  'Tensile/TensileLogic/ValidWorkGroup.py',
  'Tensile/TensileLogic/ValidWorkGroupMappingXCC.py',
]
const SLICE1_TESTDIRS = [
  'Tensile/Tests/unit/characterization/CommonUtilities',
  'Tensile/Tests/unit/characterization/TensileLogic',
]

let fail = 0
const ok = (m) => console.log('ok   - ' + m)
const bad = (m) => { console.log('BAD  - ' + m); fail = 1 }

// --- config is valid for mutmut-slice.js --dry-run ---
let plan
try { plan = buildPlan(CFG) } catch (e) { bad('slice-1 config rejected by buildPlan: ' + e.message) }
if (!plan) { console.log('\nSLICE-1 REGRESSION DRY-RUN FAILED'); process.exit(1) }
ok('slice-1 config is valid for mutmut-slice.js --dry-run')
plan.slice.slice_id === 1 ? ok('slice_id is 1') : bad('slice_id wrong: ' + plan.slice.slice_id)

// --- config selects EXACTLY the five slice-1 files (order + content), no more,
//     no fewer: guards against a dropped/extra/reordered/duplicated file ---
JSON.stringify(CFG.only_mutate) === JSON.stringify(SLICE1_FILES)
  ? ok('config only_mutate is exactly the five slice-1 files (order + content)')
  : bad('config only_mutate != required slice-1 files: ' + JSON.stringify(CFG.only_mutate))

// --- config selects EXACTLY the two slice-1 char dirs (order + content):
//     guards against a spurious extra/missing/reordered test dir ---
JSON.stringify(CFG.test_selection) === JSON.stringify(SLICE1_TESTDIRS)
  ? ok('config test_selection is exactly the two slice-1 char dirs (order + content)')
  : bad('config test_selection != required slice-1 dirs: ' + JSON.stringify(CFG.test_selection))

// --- all five slice-1 files actually thread into the dry-run plan commands ---
const allCmds = plan.phases.flatMap((p) => p.steps.map((s) => s.cmd)).join('\n')
for (const f of SLICE1_FILES) {
  allCmds.includes(f) ? ok('plan includes source file ' + f) : bad('plan MISSING source file ' + f)
}
plan.slice.only_mutate.length === 5 ? ok('exactly five slice-1 files in plan') : bad('only_mutate count != 5: ' + plan.slice.only_mutate.length)

// --- both slice-1 characterization dirs actually thread into the plan ---
for (const d of SLICE1_TESTDIRS) {
  allCmds.includes(d) ? ok('plan includes char dir ' + d) : bad('plan MISSING char dir ' + d)
}
plan.slice.test_selection.length === 2 ? ok('exactly two slice-1 char dirs in plan') : bad('test_selection count != 2: ' + plan.slice.test_selection.length)

// --- Phases 0-5 present, in order ---
const phaseNums = plan.phases.map((p) => p.n)
JSON.stringify(phaseNums) === JSON.stringify([0, 1, 2, 3, 4, 5])
  ? ok('phases 0-5 present in order') : bad('phase set/order wrong: ' + phaseNums)

// --- all five slice tools referenced ---
const tools = toolsInPlan(plan)
;['preflight', 'pyproject', 'adapter', 'triage', 'verify'].forEach((t) =>
  tools.has(t) ? ok('plan invokes ' + t) : bad('plan missing tool ' + t))

// --- pyproject.toml stays clean: dry-run is pure; set (Phase 0) is paired with
//     restore + assert-clean (Phase 5). No mutation of pyproject at plan time. ---
const p0 = plan.phases.find((p) => p.n === 0).steps.map((s) => s.cmd).join('\n')
const p5 = plan.phases.find((p) => p.n === 5).steps.map((s) => s.cmd).join('\n')
p0.includes('pyproject-mutmut.sh set') ? ok('Phase 0 sets pyproject [tool.mutmut]') : bad('Phase 0 missing pyproject set')
p5.includes('pyproject-mutmut.sh restore') ? ok('Phase 5 restores pyproject') : bad('Phase 5 missing pyproject restore')
p5.includes('pyproject-mutmut.sh assert-clean') ? ok('Phase 5 asserts pyproject clean') : bad('Phase 5 missing assert-clean')

// --- concurrency invariants (mutmut run + verifier serial; only triage fans out) ---
const mutRun = plan.phases[1].steps.find((s) => s.cmd.includes('mutmut run'))
mutRun && mutRun.serial ? ok('mutmut run is a serial actor') : bad('mutmut run not serial')
const fanouts = plan.phases.flatMap((p) => p.steps).filter((s) => s.serial === false && s.kind !== 'placeholder')
fanouts.length === 1 && fanouts[0].kind === 'workflow' ? ok('only triage-workflow fans out') : bad('unexpected fan-out step(s)')

// --- certified pilot numbers recorded in the machine-readable artifact ---
// Every field is pinned (not just the headline six): expected.json is the
// ground truth a future REAL regression run compares against, so an undetected
// edit to any number would silently corrupt that comparison.
const eqNum = (got, want, label) => got === want ? ok(`expected.json ${label} = ${want}`) : bad(`expected.json ${label} = ${got}, want ${want}`)
// headline pilot expectations named by the reviewer note
eqNum(EXP.certified.pre_triage_survivors, 131, 'certified.pre_triage_survivors')
eqNum(EXP.certified.remaining_survivors, 4, 'certified.remaining_survivors')
EXP.certified.remaining_all_equivalent === true ? ok('expected.json certified.remaining_all_equivalent = true') : bad('expected.json certified.remaining_all_equivalent not true')
eqNum(EXP.certified.no_test_count, 84, 'certified.no_test_count')
eqNum(EXP.scores_pct.covered_before, 77.5, 'scores_pct.covered_before')
eqNum(EXP.scores_pct.covered_after, 99.3, 'scores_pct.covered_after')
// full baseline block
eqNum(EXP.baseline.total_mutants, 665, 'baseline.total_mutants')
eqNum(EXP.baseline.killed, 450, 'baseline.killed')
eqNum(EXP.baseline.survived, 131, 'baseline.survived')
eqNum(EXP.baseline.no_covering_test, 84, 'baseline.no_covering_test')
eqNum(EXP.baseline.timeout, 0, 'baseline.timeout')
eqNum(EXP.baseline.suspicious, 0, 'baseline.suspicious')
// full certified block
eqNum(EXP.certified.total_mutants, 654, 'certified.total_mutants')
eqNum(EXP.certified.killed, 566, 'certified.killed')
eqNum(EXP.certified.killed_by_new_tests, 118, 'certified.killed_by_new_tests')
eqNum(EXP.certified.removed_by_pragma, 9, 'certified.removed_by_pragma')
eqNum(EXP.certified.pragmas_applied, 3, 'certified.pragmas_applied')
// raw scores + slice suite
eqNum(EXP.scores_pct.raw_before, 67.7, 'scores_pct.raw_before')
eqNum(EXP.scores_pct.raw_after, 86.5, 'scores_pct.raw_after')
eqNum(EXP.slice_suite.passed, 184, 'slice_suite.passed')
eqNum(EXP.slice_suite.snapshots, 70, 'slice_suite.snapshots')
eqNum(EXP.slice_suite.failed, 0, 'slice_suite.failed')
// internal-consistency invariants (accounting must close, independent of the literals above)
EXP.baseline.killed + EXP.baseline.survived + EXP.baseline.no_covering_test === EXP.baseline.total_mutants
  ? ok('baseline accounting closes (killed+survived+no_test == total)') : bad('baseline accounting does not close')
EXP.certified.killed_by_new_tests + EXP.certified.removed_by_pragma + EXP.certified.remaining_survivors === EXP.baseline.survived
  ? ok('survivor disposition closes (new-kills+pragma+remaining == 131)') : bad('survivor disposition does not close')
EXP.source && EXP.source.includes('PILOT-BASELINE.md') ? ok('expected.json cites PILOT-BASELINE.md') : bad('expected.json missing PILOT-BASELINE.md source')

// expected.json files/dirs agree with the config the plan was built from
JSON.stringify(EXP.only_mutate) === JSON.stringify(SLICE1_FILES) ? ok('expected.json only_mutate matches slice-1 files') : bad('expected.json only_mutate mismatch')
JSON.stringify(EXP.test_selection) === JSON.stringify(SLICE1_TESTDIRS) ? ok('expected.json test_selection matches slice-1 dirs') : bad('expected.json test_selection mismatch')

// --- determinism (same config -> byte-identical plan) ---
JSON.stringify(buildPlan(CFG)) === JSON.stringify(buildPlan(CFG)) ? ok('plan is deterministic') : bad('plan non-deterministic')

console.log('')
if (fail === 0) { console.log('SLICE-1 REGRESSION DRY-RUN PASSED'); process.exit(0) }
else { console.log('SLICE-1 REGRESSION DRY-RUN FAILED'); process.exit(1) }
