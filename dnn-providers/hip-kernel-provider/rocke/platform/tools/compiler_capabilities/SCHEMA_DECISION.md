# COMGR capability report schema decision

## Contents

- [Decision](#decision)
- [Why COMGR is the sole compiler route](#why-comgr-is-the-sole-compiler-route)
- [Envelope](#envelope)
- [Observation semantics](#observation-semantics)
- [Evidence and policy are separate](#evidence-and-policy-are-separate)
- [Cache and provenance](#cache-and-provenance)
- [Evolution](#evolution)
- [Deferred lowering integration](#deferred-lowering-integration)

## Decision

`capabilities.json` is installed-tool evidence about the exact COMGR library
that rocKE uses to compile production LLVM IR. It is not a compiler-version
label, an emission profile, or a qualification claim. The packaged resolver
turns it into the canonical `rocke.compiler_policy/v1` configuration input.

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
| `probe` | Names the semantic protocol, implementation, schema, target context, and exact fingerprint. |
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
which form rocKE should emit. The packaged resolver applies one explicit
preference order, preserves unsupported optional families as `null`, and binds
the normalized result to its evidence fingerprint with a policy digest.

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
- mapped shared LLVM dependency paths and digests;
- embedded LLVM version evidence and its query source;
- query-library provenance;
- target triple and requested processor (the report also records the effective
  concrete or generic processor selected by direct COMGR preflight);
- semantic protocol digest (exact requests plus verdict semantics);
- probe, worker, and support implementation digest; and
- schema digest.

Every worker is explicitly pinned to the selected library and disables COMGR's
on-disk cache. A matching report cache skips all workers only when the host can
enumerate behavior-contributing shared libraries; otherwise every query
remeasures. `--refresh` reruns workers without changing report semantics.

The embedded LLVM version explains which compiler lineage was observed. It does
not predict a capability and never substitutes for one.

Historical evidence covers nine weekly transition endpoints on gfx950 and
gfx1250, plus 7.0.0 and 7.2.0 release baselines on gfx950. The initial broad
survey found no structural change at 37 of 42 adjacent weekly boundaries, so
unchanged intermediate artifacts are not retained as a permanent replay tax.
Historical evidence is invalidated only when the semantic protocol or evidence
schema changes. File moves, comments, CLI changes, cache changes, and type-only
refactors change the local cache identity but do not require compiler replay.

## Evolution

The evidence schema and policy manifest ship in the rocKE wheel. Their history
is ordinary source history; there is no parallel release lifecycle. Probe,
schema, resolver, example, validator, and consumers change together.

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

## Deferred lowering integration

The installed package now supplies report availability for the exact COMGR
library and defines selection policy separately from observations. A later
change may replace one flavor-derived lowering decision with a field from the
canonical manifest. It must feed identical policy bytes to Python and C++, bind
the policy digest into compiled-kernel identity, retain explicit offline policy
fixtures during migration, and pass byte-identity and relevant qualification
gates.

This change intentionally does not alter lowering or CI selection.
