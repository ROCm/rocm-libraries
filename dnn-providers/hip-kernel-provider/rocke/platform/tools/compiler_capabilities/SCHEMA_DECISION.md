# Compiler capability report schema decision

This document fixes the report contract for the standalone rocKE compiler
capability probe. The report is compiler evidence only: no rocKE lowering,
dispatch, build, or test path consumes it yet, and this work does not add CI
integration.

## Decision summary

1. `capabilities.json` is checkout-local evidence, not a compiler-version label
   and not yet a configuration input.
2. Provenance, measurement context, and observed behavior are separate sections.
3. Every capability has a stable behavior-oriented machine ID.
4. Candidate inputs are preserved separately from canonical compiler output.
5. Unsupported candidates are successful `known` observations; probe failures
   are `error` results.
6. The JSON schema and example report are normative for report shape. Human
   summaries and Markdown studies are derived views.
7. A report is cached only for the exact tool identity, processor context, and
   probe corpus that produced it.
8. Capability probes may be retired once their behavior is settled outside the
   supported compiler window and no active consumer needs the distinction.

## Report envelope

| Section | Contract |
| --- | --- |
| `probe` | Identifies the probe and schema source digests, target context, and complete cache fingerprint. |
| `compiler` | Records `clang` provenance. Paths and banners identify evidence; they are not capabilities. |
| `tools` | Records every supporting executable whose behavior contributes to the report. |
| `capabilities` | Contains only directly observed compiler behavior. |

The envelope deliberately does not contain a derived `llvm23`, `modern`, or
`compatible` label. Such labels collapse independent observations and become
incorrect as compiler behavior evolves.

## Capability naming

Capability IDs use a stable dotted namespace:

```text
<target-domain>.<layer>.<specific-contract>
```

Examples include `amdgpu.ir.data_layout` and
`amdgpu.intrinsic.named_barrier.pointer_address_space`. An ID names one
independently observable contract. It must not encode a release, LLVM major,
weekly candidate, or preferred rocKE policy.

A capability that varies independently gets its own ID even when current
compiler snapshots change it at the same time as another capability. A broad
umbrella boolean is not a substitute for the individual observations beneath
it.

## Result semantics

### `known`

The focused probe completed and produced reliable evidence. For candidate-set
probes, `known` contains both:

- `accepted`: submitted candidates LLVM recognized, together with canonical
  declarations or ABIs written back by LLVM;
- `rejected`: candidates LLVM examined and refused, with the first diagnostic.

A rejected candidate is data, not a probe failure. `known` also does not mean
that rocKE has selected that form or qualified a complete kernel.

For frontend, target, pipeline, option, and attribute probes, `known` contains a
list of stage-bearing observations. Each observation preserves the submitted
candidate, independent family, evidence stage, outcome, and compact canonical
output. `rejected`, `crash`, and `timeout` are reliable per-candidate outcomes;
they do not turn the whole capability into `error`.

### `error`

The probe did not obtain reliable evidence, for example because a tool was
missing, timed out, crashed, or returned malformed output. The diagnostic is
for a human; consumers must not parse it into a capability decision. An error is
never silently converted to unsupported, false, or a flavor-derived default.

The current schema attempts every required capability. The JSON schema pins
shape, fixed corpus identities, and buffer-specific canonical fields; the
checkout validator additionally proves cross-entry completeness, uniqueness,
and outcome-dependent layout evidence that JSON Schema cannot express directly.
If a future probe needs explicit `unmeasured` or `not_applicable` states, those
semantics must be added explicitly rather than inferred from absence.

## Submitted and canonical forms

Candidate corpora record the exact spelling, type, address space, operand count,
or layout submitted to the compiler. Accepted entries separately retain LLVM's
canonical output. This distinction is mandatory because LLVM can accept a
legacy input while upgrading it to a different intrinsic name or ABI.

The report does not choose among multiple accepted forms. Selection is a later
rocKE policy decision that requires complete-module and correctness
qualification.

## Schema evolution

The schema is source of truth in the rocKE checkout. A report is generated and
consumed only by code from that same checkout, so schema history is ordinary
source history. There is no parallel compatibility-version lifecycle: the
single schema, example, probe, consumers, and historical study change together.

The report records SHA-256 digests of the probe implementation and schema. They
identify exactly which source contract produced the evidence and automatically
invalidate cached results when either changes. No manual schema or probe-corpus
revision is maintained.

## Cache and reproducibility contract

The cache fingerprint includes the probe and schema digests, target triple,
processor, resolved tool paths, executable digests, and complete tool identity
text. A cache hit may skip behavior probes only after those inputs match.
Changing `clang`, `opt`, their reported compiler identity, the target context,
the probe, or the schema invalidates the cache. `--refresh` forces fresh compiler
invocations without changing report semantics.

A report records source and patch revisions when the compiler banner provides
them. Revisions identify the measured artifact but never substitute for a
behavior result.

## Adding a capability

A new capability must include:

1. a source-derived reason rocKE needs the distinction;
2. one stable machine ID;
3. a bounded candidate corpus or canonical observation;
4. deterministic success, rejection, and error handling;
5. schema and example-report updates;
6. focused tests for the probe and cache inputs;
7. a human summary that does not invent a selection policy; and
8. replay across the relevant compiler window before production integration.

## Retiring a capability

A capability can be removed from a future schema only when:

- its behavior is settled across the supported compiler window;
- no current lowering, qualification, or migration decision consumes it;
- historical reports retain enough provenance to explain the retired
  distinction; and
- removing it does not weaken a correctness or generated-code gate.

Retirement updates the single schema, consumers, and study in the same source
change.

## Front-door workflow

For a ROCm Python environment containing `rocm-sdk-core`:

```bash
python platform/tools/compiler_capabilities/probe.py \
  --python-env <environment> \
  --processor gfx942 \
  --validate
```

This writes `capabilities.json`, prints a compact human summary, and reuses an
exact matching cache. Use `--refresh` while collecting historical evidence.
Advanced users may select an LLVM tool directory with `--llvm-bin` or set
`ROCKE_PYTHON_ENV` / `ROCKE_LLVM_BIN`. `--quiet` leaves only the JSON artifact.

Schema validation is optional at runtime so the probe remains standalone; the
`--validate` flag requires the `jsonschema` Python package and validates against
the schema shipped beside the probe.

## Deferred integration

A later change may feed the report into the centralized LLVM profile, migrate a
contained lowering decision, and retain flavor inference as a temporary
fallback. That integration must update both Python and C++ emission and pass
byte-identity, complete-module, and correctness gates. It is intentionally
outside this schema-and-evidence change.
