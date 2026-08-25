# Schema (condensed)

Full required fields: `data/schemas.yaml`. Vocabulary: `data/tags.yaml`.

| Type | ID prefix | Purpose |
|---|---|---|
| process | `process-*` | Step 0, probes, routing, escape hatch |
| family | `family-*` | Operator × architecture tables |
| pattern | `pattern-*` | Symptom → techniques |
| technique | `technique-*` | Lever + rocke primitive + snippet |
| hardware | `hw-*` | gfx / MFMA / WMMA / LDS / gfx1250 features |
| migration | `migration-*` | Port gfx950/gfx1201 → gfx1250 |
| kernel | `kernel-*` | Rocke instance |
| project | `project-*` | Monorepo source tree |

`arch_specific: true` marks technique pages that apply to one architecture
family. `rocke_primitive` names the helper/spec field.

Confidence: `verified` (ISA/probe + numeric harness) > `source-reported` >
`inferred` > `experimental`.

Reproducibility for techniques/kernels should be at least `snippet`.

Software-achieved TFLOP/s, µs, GB/s are rejected by `scripts/validate.py`.
