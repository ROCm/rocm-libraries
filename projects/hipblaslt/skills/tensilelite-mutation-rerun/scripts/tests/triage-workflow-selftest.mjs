// triage-workflow-selftest.mjs — no-docker, no-agent fixture harness for
// triage-workflow.js. It runs the REAL module code with stubbed workflow globals
// (agent/parallel/phase/log/args), capturing the prompts each agent() would
// receive, then asserts the generalization: per-slice OUT, parameterized
// pragma-stage test selection, char_dir string-or-list rendering, and legacy
// backward-compat when args is the bare groups array.
//
// The module uses the workflow runner's top-level `return` and `export const
// meta`; to execute it in plain node we mechanically (a) drop the `export` and
// (b) wrap the body in an async IIFE so top-level return/await are legal. No
// logic is altered — the arg-handling and prompt-building code runs verbatim.
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const HERE = dirname(fileURLToPath(import.meta.url))
const SRC = join(HERE, '..', 'triage-workflow.js')
const raw = readFileSync(SRC, 'utf8').replace('export const meta', 'const meta')

let fail = 0
const ok = (m) => console.log('ok   - ' + m)
const bad = (m) => { console.log('BAD  - ' + m); fail = 1 }

async function run(args) {
  const captured = { prompts: [], logs: [] }
  const phase = () => {}
  const log = (m) => captured.logs.push(String(m))
  const agent = (prompt, opts) => {
    captured.prompts.push({ prompt, label: opts && opts.label })
    const props = (opts && opts.schema && opts.schema.properties) || {}
    if ('triage' in props) {
      return Promise.resolve({
        function: 'f', module: 'm', test_file: 't', test_written: true,
        triage: [{ mutant_id: 'm1', bucket: 'intentionally-unhelpful', action: 'pragma', test_node: null, pragma_line: 5, note: 'n' }],
      })
    }
    if ('verdicts' in props) return Promise.resolve({ verdicts: [] })
    if ('applied' in props) return Promise.resolve({ applied: 1, suite_green: true, notes: '' })
    return Promise.resolve('ok')
  }
  const parallel = (thunks) => Promise.all(thunks.map((t) => t()))
  const fn = new Function('agent', 'parallel', 'phase', 'log', 'args',
    'return (async () => {\n' + raw + '\n})()')
  await fn(agent, parallel, phase, log, args)
  return captured
}

const WT = '/WT'
const group = (extra) => ({ module: 'm', function: 'f', source_file: 'src/f.py', test_file: 'tests/test_f.py', survivors: ['s1'], ...extra })

// ---- Case 1: object args, per-slice OUT + list test_selection + char_dir list ----
{
  const cap = await run({
    wt: WT, src_rel: 'SR', out: 'work/.../slices/9-libraryio/workflow',
    test_selection: ['sel/Alpha', 'sel/Beta'],
    groups: [group({ char_dir: ['charA', 'charB'] })],
  })
  const triage = cap.prompts.find((p) => p.label && p.label.startsWith('triage:'))
  const pragma = cap.prompts.find((p) => p.label === 'pragma')
  const synth = cap.prompts.filter((p) => ['ledger', 'report', 'recs', 'assemble-check'].includes(p.label))

  triage && triage.prompt.includes(`- ${WT}/SR/charA`) && triage.prompt.includes(`- ${WT}/SR/charB`)
    ? ok('char_dir list renders each dir on its own line (SRC_REL-prefixed)') : bad('char_dir list not rendered readably')
  triage && !triage.prompt.includes('charA,charB')
    ? ok('char_dir list not joined into a broken comma path') : bad('char_dir list produced a broken comma path')
  pragma && pragma.prompt.includes('sel/Alpha sel/Beta')
    ? ok('pragma stage uses parameterized test_selection') : bad('pragma stage missing parameterized selection')
  pragma && !pragma.prompt.includes('CommonUtilities') && !pragma.prompt.includes('TensileLogic')
    ? ok('pragma stage has no slice-1 literals when overridden') : bad('pragma stage still names slice-1 dirs')
  synth.some((p) => p.prompt.includes('9-libraryio/workflow'))
    ? ok('per-slice OUT threads into synthesis prompts') : bad('OUT override not used in synthesis')
  // recommendations language corrected
  const recs = cap.prompts.find((p) => p.label === 'recs')
  recs && recs.prompt.includes('do NOT recommend adding semantic equivalents')
    ? ok('recommendations forbid do_not_mutate for equivalents') : bad('recommendations language not corrected')
}

// ---- Case 2: legacy array args (backward compat) + char_dir string ----
{
  const cap = await run([group({ char_dir: 'charSingle', wt: undefined })])
  const triage = cap.prompts.find((p) => p.label && p.label.startsWith('triage:'))
  const pragma = cap.prompts.find((p) => p.label === 'pragma')
  // default WT is the legacy absolute path; assert char_dir string still renders as one line
  triage && /- \/.*\/charSingle/.test(triage.prompt)
    ? ok('char_dir string still renders (backward compat)') : bad('char_dir string broke under array args')
  pragma && pragma.prompt.includes('CommonUtilities') && pragma.prompt.includes('TensileLogic')
    ? ok('legacy array args reproduce slice-1 pragma selection') : bad('legacy default pragma selection lost')
  cap.logs.some((l) => l.includes('legacy default OUT'))
    ? ok('logs a warning when OUT defaulted') : bad('no default-OUT warning logged')
}

// ---- Case 3: empty test_selection must NOT produce an empty pytest argv ----
for (const empty of ['', []]) {
  const cap = await run({ wt: WT, out: 'o', test_selection: empty, groups: [group({ char_dir: 'c' })] })
  const pragma = cap.prompts.find((p) => p.label === 'pragma')
  const label = JSON.stringify(empty)
  pragma && pragma.prompt.includes('CommonUtilities') && pragma.prompt.includes('TensileLogic')
    ? ok(`empty test_selection ${label} falls back to default (no empty pytest argv)`)
    : bad(`empty test_selection ${label} produced an empty/whole-suite pytest selection`)
  cap.logs.some((l) => l.includes('legacy default pragma-stage selection'))
    ? ok(`empty test_selection ${label} logs the default warning`)
    : bad(`empty test_selection ${label} did not log the default warning`)
}

// ---- Case 4: degenerate args and missing char_dir must not throw or emit junk ----
for (const a of [undefined, null, [], {}]) {
  let threw = false
  try { await run(a) } catch (e) { threw = true }
  !threw ? ok(`args=${JSON.stringify(a)} runs without throwing`) : bad(`args=${JSON.stringify(a)} threw`)
}
{
  const cap = await run({ wt: WT, out: 'o', groups: [group({ char_dir: undefined })] })
  const triage = cap.prompts.find((p) => p.label && p.label.startsWith('triage:'))
  triage && !triage.prompt.includes(`${WT}/undefined`) && triage.prompt.includes('(none provided)')
    ? ok('missing char_dir renders "(none provided)" not a ${WT}/undefined junk path')
    : bad('missing char_dir produced a junk path')
}

console.log('')
if (fail === 0) { console.log('ALL SELFTESTS PASSED'); process.exit(0) }
else { console.log('SELFTESTS FAILED'); process.exit(1) }
