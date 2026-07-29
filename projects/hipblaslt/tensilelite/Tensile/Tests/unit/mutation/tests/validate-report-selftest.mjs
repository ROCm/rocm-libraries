// validate-report-selftest.mjs — checks the report validator against valid and
// invalid fixtures. Pure (no spawn); runs in the review sandbox.
import { validateReport } from '../validate-report.js'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const HERE = dirname(fileURLToPath(import.meta.url))
const FX = join(HERE, 'fixtures')
const load = (f) => JSON.parse(readFileSync(join(FX, f), 'utf8'))

let fail = 0
const ok = (m) => console.log('ok   - ' + m)
const bad = (m) => { console.log('BAD  - ' + m); fail = 1 }

// valid passes
{
  const r = validateReport(load('report-valid.json'))
  r.ok ? ok('valid report passes') : bad('valid report failed: ' + r.errors.join('; '))
}

// each invalid fails with a specific, expected error
const cases = [
  ['report-invalid-sum.json', /accounting mismatch/, 'counts not summing to total_mutants fails'],
  ['report-invalid-missing-shadow.json', /pragma_free_shadow_score/, 'missing pragma_free_shadow_score fails'],
  ['report-invalid-missing-inconclusive.json', /inconclusive/, 'missing inconclusive fails'],
  ['report-invalid-negative.json', /minimum/, 'negative count fails'],
  ['report-invalid-score.json', /maximum/, 'score above 1 fails'],
]
for (const [f, pat, label] of cases) {
  const r = validateReport(load(f))
  if (r.ok) { bad(`${label}: unexpectedly passed`); continue }
  pat.test(r.errors.join('\n')) ? ok(`${label} (${r.errors.find((e) => pat.test(e))})`) : bad(`${label}: wrong error(s): ${r.errors.join('; ')}`)
}

// extra: a non-object report is rejected cleanly
{
  const r = validateReport([1, 2, 3])
  r.ok === false && r.errors.some((e) => /must be a JSON object/.test(e)) ? ok('non-object report rejected') : bad('non-object report not rejected')
}

// extra: null / primitive reports are rejected cleanly (no throw)
for (const prim of [null, 5, 'x', true]) {
  let threw = false, r = null
  try { r = validateReport(prim) } catch (e) { threw = true }
  !threw && r && r.ok === false ? ok(`primitive report ${JSON.stringify(prim)} rejected cleanly (no throw)`) : bad(`primitive report ${JSON.stringify(prim)} threw or passed`)
}

// extra: multiple errors accumulate (do not stop at first)
{
  const r = validateReport({ slice_id: 2 })
  r.errors.length >= 5 ? ok(`accumulates multiple errors (${r.errors.length})`) : bad('did not accumulate errors: ' + r.errors.length)
}

console.log('')
if (fail === 0) { console.log('ALL SELFTESTS PASSED'); process.exit(0) }
else { console.log('SELFTESTS FAILED'); process.exit(1) }
