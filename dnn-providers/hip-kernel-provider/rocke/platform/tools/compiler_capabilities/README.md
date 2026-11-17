# Probing the COMGR contract rocKE uses

## Contents

- [Purpose](#purpose)
- [One compiler route](#one-compiler-route)
- [Observed contracts](#observed-contracts)
- [Isolation and verdicts](#isolation-and-verdicts)
- [Fast execution](#fast-execution)
- [Run the probe](#run-the-probe)
- [Report semantics](#report-semantics)
- [Historical validation](#historical-validation)
- [Relationship to LLVM flavors](#relationship-to-llvm-flavors)
- [Scope](#scope)

## Purpose

rocKE's production compilation path is:

```text
KernelDef -> LLVM IR text -> libamd_comgr -> HSACO
```

The compiler contract that matters is therefore the behavior of the selected
COMGR library. A nearby `clang` or `opt` executable is not evidence about that
library and is not required by this probe.

The probe asks COMGR focused behavioral questions instead of mapping its
embedded LLVM major to one broad emission profile. It records evidence only;
rocKE lowering does not consume the report in this change.

## One compiler route

Every candidate traverses the same raw COMGR actions used by production:

1. set the requested ISA and option list;
2. compile LLVM IR source to bitcode;
3. generate a relocatable code object;
4. link the relocatable to an executable HSACO.

A candidate is `accepted` only after all three compiler actions complete and an
HSACO is returned. This avoids the fictional-intrinsic failure mode where an
unknown `llvm.*` declaration survives parsing as an external function but fails
later.

The capability worker deliberately bypasses rocKE's current
major-derived IR/profile guard. Applying that guard would let the behavior under
study decide the answer before COMGR received the candidate. Normal compilation
continues to use the guard unchanged.

## Observed contracts

The bounded corpus covers the production decisions currently compressed into
LLVM flavors:

- all four historical complete data layouts;
- both buffer-resource spellings crossed with 32- and 64-bit counts;
- both packed operand policies across every affected homogeneous and mixed
  fp8/bf8 MFMA family;
- address spaces 3 and 15 across every pointer-taking named-barrier intrinsic;
- current and historical scaled-MFMA call policies at both known shapes; and
- requested and explicitly catalogued generic processor names.

A policy-wide candidate is one compiler job containing every operation or shape
that policy controls. It is accepted only when the complete family reaches an
executable HSACO. This keeps the report at 15–16 jobs per processor instead of
repeating one global choice for each member of its family.

Optimization levels, free-form backend options, function attributes, and HSACO
fingerprints are intentionally absent. Existing option tooling owns option-name
acceptance; attribute acceptance does not prove that an attribute is honored;
and generated-code changes belong to qualification, not configure-time feature
selection.

Candidate inputs are compiler archaeology owned by this probe. They are not
imported from `rocke.core.llvm_profile`, because the probe is intended to
supersede that module's assumptions rather than learn its answer from them.

## Isolation and verdicts

Each candidate runs in a disposable Python process pinned to one COMGR library
with `ROCKE_COMGR_LIB`. `AMD_COMGR_CACHE=0` prevents another process's code
object from answering the question.

Outcomes are:

| Outcome | Meaning |
| --- | --- |
| `accepted` | COMGR returned an executable HSACO after every required action. |
| `rejected` | COMGR returned a normal error. |
| `crash` | The isolated compiler process terminated abnormally. |
| `timeout` | The candidate exceeded the declared per-process bound. |
| `error` | The probe infrastructure did not produce interpretable evidence. |

A compiler crash or timeout is an observed candidate behavior, not a reason to
crash the controller or silently call the feature unsupported. Infrastructure
errors make the command fail and are not cached.

## Fast execution

The shortest reliable wall-time strategy is independent concurrent workers:

- every candidate is independently attributable;
- an abort loses only that candidate;
- COMGR loads and compiles in parallel up to the process's available CPU set;
- requested-processor probing is speculative: the common path runs every
  candidate in the first parallel wave, while a compiler that accepts only the
  catalogued generic name triggers one behavior-only replay with that ISA;
- `--jobs` allows a site to cap concurrency when compiler memory, rather than
  CPU, is the limiting resource; and
- exact report caching avoids all candidate compiles on a repeated toolchain.

Combining alternatives into one LLVM module is faster only when every candidate
succeeds. A rejected declaration makes the whole module fail and requires
recursive splitting; a fatal backend error loses the remainder. The current
corpus is small enough that CPU-wide independent workers provide low wall time
while preserving simple, trustworthy attribution. The report records no timing
measurements because timings are scheduling diagnostics, not capabilities.

## Run the probe

From `platform/`, using normal rocKE COMGR discovery:

```bash
PYTHONPATH=python python tools/compiler_capabilities/probe.py \
  --processor gfx950 \
  --validate
```

Pin a particular library when comparing compiler artifacts:

```bash
PYTHONPATH=python python tools/compiler_capabilities/probe.py \
  --comgr-lib <path-to-libamd_comgr> \
  --processor gfx942 \
  --refresh \
  --validate
```

Useful controls:

- `--jobs 0` uses the available CPU count; a positive value sets an explicit
  worker bound;
- `--timeout` sets the per-candidate process bound;
- `--cache-dir` changes the exact-evidence cache;
- `--refresh` ignores a matching cache; and
- `--quiet` writes only the JSON artifact.

No LLVM executables, GPU, Torch, or `jsonschema` package are required to produce
a report. `--validate` additionally requests full JSON Schema validation and
therefore requires `jsonschema`; report construction always runs the
checkout-local dependency-free completeness validator.

## Report semantics

The report separates:

- `probe`: implementation/schema digests and target context;
- `compiler`: the loaded COMGR path, binary digest, and embedded LLVM version
  evidence; and
- `capabilities`: ordered candidate observations from the full COMGR route.

The embedded LLVM version is provenance, not a capability and not a selection
key. Cache identity includes the COMGR digest, complete compiler evidence,
requested processor, target triple, probe implementation, and schema. The
report separately records the effective concrete or generic processor used for
behavior observations.

`accepted` does not select a rocKE policy. Several forms may compile. A later
resolver must choose among accepted forms, preserve unknown states, and qualify
the selected complete kernels through byte identity, generated-code/resource
checks, and numeric testing.

## Historical validation

The probe is replayed across nine selected weekly snapshots at the observed
transition endpoints, three public release baselines, and five processor
contexts. The initial broad survey showed that 37 of 42 adjacent weekly pairs
had no structural change; unchanged intermediate weeks are therefore not a
permanent replay obligation.
[`replay_historical.py`](replay_historical.py) uses HTTP range requests to
extract only COMGR and its required shared libraries from SDK wheels, downloads
only the smaller COMGR package for release baselines, reuses duplicate artifacts,
and checkpoints each completed row. The sanitized
machine evidence lives in
[`historical_observations.json`](historical_observations.json), with analysis in
[`historical_study.md`](historical_study.md) and
[`churn_report.md`](churn_report.md).

Run the dependency-free checker after any probe, schema, input, or study change:

```bash
python tools/compiler_capabilities/check_historical_study.py
```

The checker proves public artifact provenance, exact candidate closure and
ordering, report digests, processor coverage, duplicate-artifact repeatability,
absence of machine-local paths, and every published structural/churn boundary.

## Relationship to LLVM flavors

`rocke.core.llvm_profile` currently combines compiler-major classification,
structural emission choices, offline replay identities, and golden/cache
partitioning. This probe supersedes only its authority to infer production
compiler behavior.

A later vertical integration should:

1. resolve capabilities against the exact COMGR library that will compile;
2. convert evidence through an explicit selection policy into a normalized
   emission policy;
3. provide byte-identical policy data to the Python and C++ lowerers;
4. retain flavor modes temporarily for offline replay and fallback; and
5. remove ranked or umbrella flavor checks one independent capability at a
   time.

The report must not be imported directly as policy: compiler acceptance is
raw evidence, while choosing and qualifying an emitted form is a rocKE decision.

## Scope

This change does not:

- wire the probe into CI;
- modify TheRock configuration;
- change Python or C++ lowering;
- select an emitted ABI;
- remove LLVM flavors;
- claim compiler qualification; or
- replace complete-module, byte-identity, ISA/resource, numeric, or performance
  qualification.
