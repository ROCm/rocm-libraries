# Historical COMGR capability study

## Contents

- [Measured scope](#measured-scope)
- [Method](#method)
- [Structural transition boundaries](#structural-transition-boundaries)
- [Generated-code fingerprint churn](#generated-code-fingerprint-churn)
- [Interpretation](#interpretation)
- [Replay](#replay)

## Measured scope

This study replays the production COMGR LLVM-IR-to-HSACO route across 43 selected
weekly public ROCm SDK snapshots from W06 through W46 and public 7.0.0, 7.1.0,
and 7.2.0 release packages. Every snapshot is probed for gfx942, gfx950,
gfx1151, gfx1201, and gfx1250. The machine-readable evidence
is [`historical_observations.json`](historical_observations.json); the immutable
public artifact selection is
[`historical_inputs.json`](historical_inputs.json).

The study supersedes the earlier Clang/`opt` survey. Those executables do not
establish the behavior of the COMGR library rocKE loads in production.

## Method

For each unique wheel, the replay tool:

1. uses HTTP range requests to fetch only COMGR and its shared-library
   dependencies from SDK wheels, or extracts COMGR from the smaller public
   release package;
2. hashes and queries the extracted COMGR library;
3. runs the complete candidate corpus in isolated workers;
4. requires every accepted candidate to reach executable HSACO creation;
5. records accepted, rejected, crash, and failed-stage outcomes; and
6. checkpoints sanitized evidence with no machine-local paths.

Duplicate weekly selections reuse one report. Every processor report carries the
same probe/schema implementation digests and exact COMGR digest. The study
contains no timeout or infrastructure-error outcomes.

A structural vector compares candidate outcome and failed stage. Optimization
level output digests are retained separately as generated-code fingerprints;
they are not structural capability booleans.

## Structural transition boundaries

5 of the 42 adjacent weekly boundaries change the structural COMGR vector:

| Boundary | Changed cells | Capability families |
| --- | ---: | --- |
| W20 → W21 | 5 | optimization-level codegen (`-O0`) |
| W34 → W35 | 14 | buffer-resource ABI; named-barrier pointer ABI |
| W40 → W41 | 45 | named-barrier pointer ABI; optimization-level codegen (`-O0`) |
| W43 → W44 | 9 | named-barrier pointer ABI |
| W45 → W46 | 45 | named-barrier pointer ABI; optimization-level codegen (`-O0`) |

The cells are processor-specific. A transition can change normal rejection to a
compiler crash, move failure between parser and codegen stages, or exchange
which pointer-address-space candidate reaches code generation. Treating all of
those as one LLVM-major profile would discard information needed for safe
selection and diagnostics.

The buffer-resource transition demonstrates why acceptance and preference must
remain separate: a newer COMGR can accept more than one historical spelling or
count width. The report records all accepted forms; later rocKE policy chooses a
qualified form.

### Release baselines

| Release package | Embedded LLVM evidence | Purpose |
| --- | --- | --- |
| 7.0.0 | 20.0.0 | legacy profile baseline |
| 7.1.0 | 20.0.0 | second legacy release baseline |
| 7.2.0 | 22.0.0 | indexed-address-space-8 transition baseline |

The release rows use public COMGR packages directly and retain the same
five-processor candidate matrix as the weekly study. They are endpoints and are
not inserted into the adjacent weekly-boundary count.

## Generated-code fingerprint churn

The focused optimization-level reproducer changes HSACO digest at 31 of the 42
adjacent boundaries. This is intentionally reported separately from the five
structural boundaries. A code-object digest change is evidence that generated
code changed, not proof that a compatibility decision or runtime semantic
changed.

This separation keeps the structural manifest stable enough for configuration
while retaining a sentinel that can route selected snapshots to generated-code
and resource qualification.

## Interpretation

The COMGR study tightens the earlier conclusions:

- direct production-route probing avoids correlating unrelated Clang, `opt`, and
  COMGR identities;
- structural compatibility changes are substantially less frequent than
  generated-code fingerprint changes;
- independent family/processor/stage observations are necessary—especially for
  named barriers;
- compiler crash is a first-class result and must remain process-isolated;
- accepted candidates are evidence, not selection policy; and
- LLVM major remains provenance only.

The steady structural schema contains nine capability families and a bounded
47–48 candidates per processor in this source revision. Probes can retire when
their distinction is settled outside the supported compiler window and no
migration consumer needs it.

## Replay

From `platform/`:

```bash
PYTHONPATH=python python tools/compiler_capabilities/replay_historical.py
python tools/compiler_capabilities/check_historical_study.py
```

The replay downloads only the wheel ranges containing COMGR and required shared
libraries. It checkpoints after every weekly row and resumes a compatible
partial output. The checker validates public provenance, exact corpus closure,
digests, absence of local paths, and every published transition claim.
