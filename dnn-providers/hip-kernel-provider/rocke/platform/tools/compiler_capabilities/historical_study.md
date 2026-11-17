# Historical COMGR capability study

## Contents

- [Measured scope](#measured-scope)
- [Method](#method)
- [Structural transition boundaries](#structural-transition-boundaries)
- [Release baselines](#release-baselines)
- [Interpretation](#interpretation)
- [Replay](#replay)

## Measured scope

This study replays the production COMGR LLVM-IR-to-HSACO route across nine
selected weekly snapshots and public 7.0.0, 7.1.0, and 7.2.0 release packages.
The weekly set contains the first sampled artifact plus both sides of every
structural transition found by the initial 43-week survey. Every snapshot is
probed for gfx942, gfx950, gfx1151, gfx1201, and gfx1250.

The machine-readable evidence is
[`historical_observations.json`](historical_observations.json). The immutable
public artifact selection is
[`historical_inputs.json`](historical_inputs.json). Unchanged intermediate weeks
are not retained as a permanent replay obligation.

## Method

For each unique artifact, the replay tool:

1. fetches only COMGR and its shared-library dependencies;
2. hashes and queries the extracted COMGR library;
3. runs the complete 15–16-candidate corpus in isolated workers;
4. requires every accepted candidate to reach executable HSACO creation;
5. records accepted, rejected, crash, and failed-stage outcomes; and
6. checkpoints sanitized evidence with no machine-local paths.

A global lowering choice is represented by one all-family module per candidate.
For example, one named-barrier address-space candidate contains every
pointer-taking named-barrier operation. The candidate is accepted only when the
whole family compiles and links.

The study contains no timeout or infrastructure-error outcomes. It records
structural compatibility only. Generated-code fingerprints and performance are
qualification concerns, not configure-time capabilities.

## Structural transition boundaries

Four selected adjacent transition pairs change the structural COMGR vector:

| Boundary | Changed cells | Capability families |
| --- | ---: | --- |
| W34 → W35 | 11 | buffer-resource ABI; named-barrier pointer ABI |
| W40 → W41 | 10 | named-barrier pointer ABI |
| W43 → W44 | 3 | named-barrier pointer ABI |
| W45 → W46 | 10 | named-barrier pointer ABI |

The cells are processor-specific. A transition can change normal rejection to a
compiler crash, move failure between parser and codegen stages, or exchange
which address-space candidate reaches code generation. Those outcomes remain
distinct because they lead to different diagnostics and fallback decisions.

The buffer-resource transition also demonstrates why acceptance and preference
remain separate. A compiler can accept more than one historical spelling or
count width; later rocKE policy must choose and qualify one form.

## Release baselines

| Release package | Embedded LLVM evidence | Purpose |
| --- | --- | --- |
| 7.0.0 | 20.0.0 | legacy profile baseline |
| 7.1.0 | 20.0.0 | second legacy release baseline |
| 7.2.0 | 22.0.0 | indexed-address-space-8 transition baseline |

The release rows use public COMGR packages directly and retain the same processor
coverage as the weekly sample. They are endpoints and are not inserted into the
weekly transition table.

## Interpretation

The reduced study supports five conclusions:

- production behavior must be measured through COMGR, not a nearby Clang or
  `opt` executable;
- global policy choices should be probed once across their complete operation
  family, not once per shape or operation;
- compiler crashes need process isolation and remain distinct from rejection;
- accepted candidates are evidence, not selection policy; and
- LLVM major remains provenance only.

The maintained schema contains six capability families and 15–16 compiler jobs
per processor. A probe retires when its distinction is settled across the
supported compiler window or no active production consumer needs it.

## Replay

From `platform/`:

```bash
PYTHONPATH=python python tools/compiler_capabilities/replay_historical.py
python tools/compiler_capabilities/check_historical_study.py
```

The replay checkpoints after every selected row and resumes a compatible partial
output. The checker validates public provenance, exact corpus closure, source
digests, absence of local paths, and every published transition row.
