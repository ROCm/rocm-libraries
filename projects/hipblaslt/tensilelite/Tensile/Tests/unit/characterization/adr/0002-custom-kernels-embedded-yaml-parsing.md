# ADR 0002: Re-target `CustomKernels` characterization at `_readEmbeddedYaml`, not the removed `getCustomKernelConfigAndAssembly`

Status:  Accepted
Defect:  none — behavior is intended

## Context
`test_custom_kernels_char.py` (added on `develop` by #7989) pinned
`Tensile.CustomKernels.getCustomKernelConfigAndAssembly`: a naive line-scanner
that split a `.s` file on bare `---`/`...` markers and returned a
`(config_text, assembly_text)` tuple of raw strings, with blank-line padding
in `config_text` to keep YAML error line numbers aligned.

The Gemm-From-Anywhere branch rewrote `CustomKernels.py` to parse the
`.amdgpu_metadata` YAML block properly (supporting the richer external-kernel
`custom.config` schema — `Source`, `Features`, `CustomKernel.args`, etc.) via
a new private helper, `_readEmbeddedYaml`, which returns the **parsed dict**
(typically `{"custom.config": {...}, "amdhsa.kernels": [...]}`) instead of
raw text. `getCustomKernelConfigAndAssembly` was dropped entirely; nothing in
the rewritten module still needs a raw `(config_text, assembly_text)` split.
This branch diverged from `develop` before #7989 landed there, so the two
changes were never reconciled against each other until now: the
characterization test's import of the now-nonexistent symbol failed at
collection time, which aborted the *entire* `-m unit` run (pytest halts a run
on any collection error unless `--continue-on-collection-errors` is passed),
silently hiding every other test in the suite behind it.

Separately, fixing the import exposed that `test_get_custom_kernel_config_ok`
also relied on a minimal `_VALID_S` fixture with no `amdhsa.kernels` entry.
`getCustomKernelConfig`'s no-explicit-`CustomKernel` path (used for
Tensile-generated kernels, which per `CustomKernels/README.md` only need to
embed `InternalSupportParams.KernArgsVersion`) now auto-infers a `CustomKernel`
block from `amdhsa.kernels[0].args` via `_buildCustomKernelFromMetadata` — a
capability that does not exist on `develop`. Real kernel `.s` files always
carry a real `amdhsa.kernels` section (it is mandatory AMDGPU code-object
metadata), so this only affects the test's synthetic fixture, not production
kernels; `Tests/unit/test_CustomKernelMetadata.py`'s own `write_kernel()`
helper (added by this same branch) already includes one for exactly this
reason.

## Decision
1. Replace the `getCustomKernelConfigAndAssembly` import/test with a new
   `test_read_embedded_yaml_parses_custom_config`, pinning `_readEmbeddedYaml`'s
   parsed-dict return instead of the removed raw-text split. This is a
   same-purpose replacement (pin how the module reads its embedded YAML), not
   a scope cut, and it does not restore any removed production code.
2. Give `test_get_custom_kernel_config_ok` its own fixture
   (`_VALID_S_WITH_KERNEL_META`) carrying a minimal, realistic
   `amdhsa.kernels` entry, matching the convention already established by
   `write_kernel()` in `test_CustomKernelMetadata.py`, rather than enriching
   the shared `_VALID_S` used by five other tests that don't need it.

## Consequences
The suite is add-only-compatible (no production code touched) and the two
changed/added `.ambr` entries were reviewed line-by-line in the PR that
carries this ADR. `_readEmbeddedYaml` is private (`_`-prefixed); characterizing
it directly follows existing precedent in this same PR
(`test_CustomKernelMetadata.py` already imports `_parse_tensile_yaml` /
`_read_asm_file` from `Tensile.AddCustomConfig`).

This ADR covers only the `CustomKernels` module's own collection failure.
Fixing it unblocked the rest of the `-m unit` suite to actually run (it had
never executed on this branch's diff, since the collection error aborted the
session before any test executed) and surfaced other, separate stale-golden
failures elsewhere in the suite (e.g. `ValidParameters`, `HandleCustomKernel`,
`ToolchainComponent`, `TensileMain`, `TensileCreateLibraryRun`,
`PublicInputSurface`) tracking other intentional and possibly-unintentional
behavior changes made over the life of this branch. Those are out of scope
for this ADR and are tracked separately.
