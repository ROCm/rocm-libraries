# COMGR capability report schema decision

## Contents

- [Decision](#decision)
- [Why COMGR is the sole compiler route](#why-comgr-is-the-sole-compiler-route)
- [Envelope](#envelope)
- [Observation semantics](#observation-semantics)
- [Evidence and policy are separate](#evidence-and-policy-are-separate)
- [Cache and provenance](#cache-and-provenance)
- [Evolution](#evolution)
- [Deferred production integration](#deferred-production-integration)

## Decision

`capabilities.json` is checkout-local evidence about the exact COMGR library
that rocKE uses to compile production LLVM IR. It is not a compiler-version
label, an emission profile, a qualification claim, or a production
configuration input.

The schema has one compiler route: a complete LLVM-IR-to-HSACO COMGR pipeline.
Every capability contains an ordered, mechanically checked candidate corpus.

## Why COMGR is the sole compiler route

rocKE production does not invoke `clang` or `opt` to compile a kernel. Results
from those executables would require proving that they contain the same LLVM as
the dynamically loaded COMGR library, and would still not establish behavior of
COMGR's action pipeline.

Clang frontend ABI and `opt` canonicalization can be useful separate research,
but they do not decide what rocKE's production lowerer may emit. They are
therefore outside this report.

## Envelope

| Section | Contract |
| --- | --- |
| `probe` | Names the corpus implementation, schema, target context, and exact fingerprint. |
| `compiler` | Identifies and hashes the loaded COMGR library and records its embedded LLVM version evidence. |
| `capabilities` | Records only direct outcomes from the full COMGR route. |

The envelope contains no derived `llvm23`, `modern`, or `compatible` label.

## Observation semantics

Every observation names a `family`, exact `candidate`, and one outcome:

- `accepted`: COMGR completed compile-to-bitcode, code generation, and final
  link and returned a non-empty HSACO;
- `rejected`: COMGR returned a normal error;
- `crash`: the isolated worker terminated abnormally;
- `timeout`: the isolated worker exceeded the declared bound; or
- `error`: the probe infrastructure did not produce interpretable evidence.

Accepted observations carry the HSACO digest and size plus the completed-stage
list. Other outcomes carry a diagnostic. A crash and timeout are first-class
compiler behavior; an infrastructure error prevents a complete reusable report.

The dependency-free validator proves exact ordered corpus coverage and
outcome-dependent evidence. Optional JSON Schema validation checks the same
external document shape when `jsonschema` is available.

## Evidence and policy are separate

A compiler can accept several historical forms. Acceptance alone does not say
which form rocKE should emit. A later resolver must apply an explicit preference
and fallback policy, then qualify the selected form with complete kernels.

The intended layering is:

```text
COMGR candidate evidence
        ↓
rocKE selection policy
        ↓
normalized emission policy
      ↙     ↘
 Python     C++
```

The probe corpus does not import flavor tables from `rocke.core.llvm_profile`.
A test may compare the old profiles with the corpus to expose migration gaps,
but the old inference layer cannot define the measurement.

## Cache and provenance

The cache fingerprint includes:

- loaded COMGR path and executable digest;
- embedded LLVM version evidence and its query source;
- query-library provenance;
- target triple and requested processor (the report also records the effective
  concrete or generic processor selected by direct COMGR preflight);
- probe, worker, and support implementation digest; and
- schema digest.

Every worker is explicitly pinned to the selected library and disables COMGR's
on-disk cache. A matching report cache skips all workers. `--refresh` reruns
them without changing report semantics.

The embedded LLVM version explains which compiler lineage was observed. It does
not predict a capability and never substitutes for one.

The schema and probe are replayed against 43 weekly public snapshots, three
public release baselines, and five processor contexts. Historical evidence is
invalidated and regenerated when the probe or schema digest changes; the checker
pins exact candidate closure and published transition claims.

## Evolution

The schema is checkout-local. Its history is ordinary source history; there is
no parallel schema-version lifecycle. Probe, schema, example, validator, and
consumers change together.

Adding a capability requires:

1. a production COMGR contract rocKE needs to distinguish;
2. a stable behavior-oriented machine ID;
3. a bounded candidate corpus derived from current or historical emitted forms;
4. full-pipeline isolated observations;
5. explicit rejection, crash, timeout, and infrastructure-error handling;
6. schema/example/completeness updates; and
7. focused tests proving every candidate is attempted exactly once.

A capability can be retired when its behavior is settled across the supported
compiler window and no active migration or production consumer needs the
distinction.

## Deferred production integration

A later change may replace one flavor-derived production decision with COMGR
capability evidence. That change must solve report availability for the exact
runtime library, define selection policy separately from observations, mirror
the normalized policy in Python and C++, retain an explicit fallback during
migration, and pass byte-identity and relevant qualification gates.

This evidence change intentionally does not alter lowering or CI configuration.
