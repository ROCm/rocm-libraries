// validate-report.js — Issue 7: validate a per-slice mutation report against
// mutation-report.schema.json AND the accounting invariants, so false or
// incomplete mutation accounting fails fast before a slice is accepted.
//
// Zero external deps (no npm): a tiny built-in schema walker (required / type /
// minimum / maximum) plus semantic checks. Read-only.
//
// CLI:    node validate-report.js <report.json>        (exit 0 valid, 1 invalid, 2 usage)
// Module: import { validateReport } from './validate-report.js'
//         validateReport(reportObj) -> { ok: boolean, errors: string[] }
//
// Invariants enforced beyond structural typing:
//   - killed + survived + no_test + inconclusive + equivalent == total_mutants
//   - `inconclusive` present and distinct from `killed` (strict kill semantics)
//   - `pragma_free_shadow_score` present so pragmas cannot silently inflate score
//   - scores numeric in [0,1]; count fields non-negative integers

import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join, resolve } from 'node:path'

const SCHEMA_PATH = join(
  dirname(fileURLToPath(import.meta.url)),
  'schema', 'mutation-report.schema.json',
)

function jsType(v) {
  if (v === null) return 'null'
  if (Array.isArray(v)) return 'array'
  if (Number.isInteger(v)) return 'integer'
  return typeof v
}

function matchType(v, t) {
  switch (t) {
    case 'integer': return Number.isInteger(v)
    case 'number': return typeof v === 'number' && Number.isFinite(v)
    case 'string': return typeof v === 'string'
    case 'object': return v !== null && typeof v === 'object' && !Array.isArray(v)
    case 'array': return Array.isArray(v)
    case 'null': return v === null
    case 'boolean': return typeof v === 'boolean'
    default: return false
  }
}

function checkSchema(report, schema, errors) {
  if (report === null || typeof report !== 'object' || Array.isArray(report)) {
    errors.push('report must be a JSON object')
    return
  }
  for (const key of schema.required || []) {
    if (!(key in report)) errors.push(`missing required field: ${key}`)
  }
  for (const [key, spec] of Object.entries(schema.properties || {})) {
    if (!(key in report)) continue
    const v = report[key]
    const types = Array.isArray(spec.type) ? spec.type : [spec.type]
    if (!types.some((t) => matchType(v, t))) {
      errors.push(`field ${key}: expected ${types.join('|')}, got ${jsType(v)}`)
      continue
    }
    if (typeof v === 'number') {
      if (spec.minimum !== undefined && v < spec.minimum) errors.push(`field ${key}: ${v} < minimum ${spec.minimum}`)
      if (spec.maximum !== undefined && v > spec.maximum) errors.push(`field ${key}: ${v} > maximum ${spec.maximum}`)
    }
  }
}

function checkAccounting(report, errors) {
  if (report === null || typeof report !== 'object' || Array.isArray(report)) return // checkSchema already flagged it; avoid `in`/index on a non-object
  const parts = ['killed', 'survived', 'no_test', 'inconclusive', 'equivalent']
  if (parts.every((k) => Number.isInteger(report[k])) && Number.isInteger(report.total_mutants)) {
    const sum = parts.reduce((a, k) => a + report[k], 0)
    if (sum !== report.total_mutants) {
      errors.push(
        `accounting mismatch: killed+survived+no_test+inconclusive+equivalent = ${sum} != total_mutants ${report.total_mutants}`,
      )
    }
  }
  // strict kill semantics: inconclusive must exist as its own field (not folded into killed)
  if (!('inconclusive' in report)) errors.push('inconclusive is required and must be distinct from killed (strict kill semantics)')
  if (!('pragma_free_shadow_score' in report)) errors.push('pragma_free_shadow_score is required so pragmas cannot silently inflate the score')
}

let CACHED_SCHEMA = null
function loadSchema() {
  if (CACHED_SCHEMA) return CACHED_SCHEMA
  CACHED_SCHEMA = JSON.parse(readFileSync(SCHEMA_PATH, 'utf8'))
  return CACHED_SCHEMA
}

export function validateReport(report, schema) {
  const s = schema || loadSchema()
  const errors = []
  checkSchema(report, s, errors)
  checkAccounting(report, errors)
  return { ok: errors.length === 0, errors }
}

// ------------------------------------------------------------------- CLI
function main() {
  const args = process.argv.slice(2)
  if (args.length === 0 || args[0] === '-h' || args[0] === '--help') {
    process.stderr.write('usage: node validate-report.js <report.json>\n')
    process.exit(args[0] === '-h' || args[0] === '--help' ? 0 : 2)
  }
  let report
  try { report = JSON.parse(readFileSync(args[0], 'utf8')) } catch (e) {
    process.stderr.write('validate-report: cannot read report: ' + e.message + '\n')
    process.exit(2)
  }
  const { ok, errors } = validateReport(report)
  if (ok) {
    process.stdout.write(`OK: ${args[0]} is a valid mutation report\n`)
    process.exit(0)
  }
  process.stderr.write(`INVALID: ${args[0]}\n`)
  for (const e of errors) process.stderr.write('  - ' + e + '\n')
  process.exit(1)
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  main()
}
