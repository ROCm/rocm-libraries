# Pull request

Draft the PR with the `hipdnn-pr-quality` skill (`../hipdnn-pr-quality/SKILL.md`). Show the user the exact title and body; never push, open or post anything without approval.

## Split

One PR per update is the default. Split when a tier 3 item is large:

1. **Data PR**: the tier 0 to 2 changes for the archs and modules already served, with their engine edits.
2. **One PR per new arch, new module or new backward pipeline stage**, each with its plan family, tests and claims.

Forward and backward can share one PR when they move to the same AITER commit; the README then names one commit for both.

Each PR runs its own `plan` and `apply` for its archs, so every PR leaves `verify` clean.

## Title

A Conventional Commits title, for example:

```text
feat(hip-kernel-provider): update ASM SDPA forward kernels to AITER v0.1.28
feat(hip-kernel-provider): update ASM SDPA backward kernels to AITER v0.1.29
```

## Body

Include:

- **Source**: the AITER tag and commit, and the archs and directions updated.
- **Host code**: per module, the host-code commit the engine now mirrors (`abi_commit`). If it lags the object commit, list the reference diffs left to port and where they will land.
- **Provenance comments**: the files outside the pin that step 7 of [SKILL.md](SKILL.md) updated.
- **Plan summary**: the module table from the plan's markdown report, the object and CSV row counts per module, and every finding of tier 2 or higher with what was done about it.
- **Exclusions**: each exclusion and inclusion with its reason, as recorded in `aiter_pins.json`.
- **ASM only**: a statement that `verify` passed, that every object is an AITER ASM kernel with no CK symbols, and that no upstream source was vendored. List each CK fallback case of `fmha_fwd_v3()`, `fmha_v3_bwd()` and the Python gates in `aiter/ops/mha.py` that became a decline.
- **Lost coverage**: graphs the engine no longer serves because AITER dropped an ASM kernel.
- **Validation**: per arch and direction, the GPU tests run and their result, or "not run", and the kernels no test checked numerically (backward rows no case reaches).
- **Not ported**: the modules the report lists as not ported.

The Libraries PR Bot needs a tracking reference in the PR body. Ask the user for it; it goes in the PR body only, never in code or provenance files.
