# ADR 0012: Decouple `test_bigfile_capped_emit` from live `library/src` tuning data

Status:  Accepted
Defect:  https://github.com/ROCm/rocm-libraries/issues/10952

## Context
`test_emit_bigfiles_char.py`'s `test_bigfile_capped_emit` (Phase 3; 10
parametrized cases) read real production tuned-logic YAMLs directly out of
`library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/...` and pinned a
`{basename, err}` digest for the first `cap` kernels (sorted by basename) as a
syrupy golden. Those files are live tuning data that many unrelated engineers
retune constantly, so any change to "which kernels sort first" in one of them
shifted the golden with zero relation to whether a given PR touched
TensileLite. Concretely: PR #10877 (gfx950 Origami work-stealing) changed the
exact files this test read and had to carry a matching `.ambr` update; PR
#10750 (unrelated bf16_r tuning) then failed the same test for a reason that
had nothing to do with its own diff. Root-caused by Brad Nemanich: "The
TensileLite tests should never look under `library/src`." Confirmed via grep
this was the *only* file anywhere under `Tensile/Tests/unit/characterization/`
referencing `library/src` / `asm_full` / `amd_detail` / `rocblaslt/src` — an
isolated defect, not a pattern used elsewhere in the suite.

## Decision
For each of the 10 `_BIG` entries, replace the live-file reference with a
vendored, trimmed, self-contained fixture checked into
`_codegen/data/bigfiles/<label>.yaml` — the same "small copy of a valid tuning
logic file" convention Phase 1's `matrix.py` / `data/<arch>/*.yaml` already
use. Each fixture keeps the original header/`ProblemType`/index-table shape
verbatim and only the `Solutions` entries whose kernels are needed to
reproduce the exact pinned capped-and-sorted kernel set (computed
programmatically from the real file, not hand-picked), with per-solution
`SolutionIndex` and any `Equality`/`GridBased` matching-table rows
remapped/filtered to stay internally consistent. `test_emit_bigfiles_char.py`
now reads only `_DATA_ROOT` (local); the `pytest.skip` fallback for a missing
tuning tree is removed since the fixtures always exist in-checkout. Added
`test_no_library_src_dependency_char.py` as a standing regression guard: it
AST-scans the characterization suite's own `.py` sources for the
product-tree–specific literals `amd_detail` / `rocblaslt` / `asm_full` and
fails if any reappear (a bare `"library"` is deliberately excluded from the
check — it's a common, unrelated dict key/dirname elsewhere in this suite and
would false-positive).

## Consequences
Each of the 10 new fixtures was run against the existing, untouched `.ambr`
golden before the test module was updated — zero snapshot diff in every case,
confirming the golden's provenance changed (vendored fixture vs. live file)
but not its content. The suite no longer has any dependency on
`library/src`'s tuning tree; the new guard test fails CI if that coupling
reappears anywhere in the characterization suite's own sources.

**Rejected alternatives:**
- Vendor the full production files verbatim — rejected: two of the ten are
  59k/64k lines; defeats the purpose of trimming and still churns on unrelated
  retuning if ever refreshed wholesale.
- Hand-pick a small representative subset of solutions — rejected in favor of
  computing the needed subset directly from the same
  sort-by-basename-then-cap logic the harness itself uses, so the fixture is
  *exactly* the minimal set that reproduces the pinned kernels, not a
  best-effort guess that could silently drift from the golden.
- Skip the regression-guard test and rely on review vigilance — rejected: this
  is precisely the kind of anti-pattern (`README.md`/`DECISIONS.md` calls it
  out once, a later PR reintroduces it) tribal knowledge fails to catch;
  operationalizing it as a CI check was Brad's own suggested fix.
