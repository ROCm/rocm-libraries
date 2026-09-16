# TensileLite namespace migration: proposed PR split

Status: implemented as a local replacement stack based on local `develop` at `b607a8ae0cfda0a27a0f6a1446cb3043a88c22c4`. The branches have not been pushed. The bridge is deliberately one-way and exists only between the package move and completion of the in-repo consumer migration.

## Objective and inspected scope

Split the migration into PRs that each leave a usable checkout, reducing the number of content edits that must be reconciled whenever the directory move is rebased. The final outcomes remain the physical rename of `projects/hipblaslt/tensilelite/Tensile/` to `projects/hipblaslt/tensilelite/tensilelite/` and migration of Python consumers from the `Tensile` namespace to `tensilelite`.

The inspection compared `users/alvasile/AIHPBLAS-3989-package-namespace` at `539e8403caf6ed0d167a30ebeabbcfb0ed09fd85` with local `develop` at `dd36ef68458fe57248f9ae1532910b77599f4034`. At inspection time, that `develop` commit was also the merge base and the local `origin/develop` ref. These are recorded comparison points, not a claim about subsequent remote updates.

Git's default rename-aware diff reports 1,545 file entries, 3,219 insertions, and 3,228 deletions. Matching old and new package paths directly gives the following breakdown:

| Area | Files moved | Files with identical contents | Files with content edits |
| --- | ---: | ---: | ---: |
| Production code and other package files | 197 | 112 | 85 |
| Custom kernels | 120 | 119 | 1 |
| Tests and test data | 1,136 | 697 | 439 |
| Total package move | 1,453 | 928 | 525 |

Another 89 files change outside the moved package. Three moved files appear as deletion/addition pairs at Git's default similarity threshold, which accounts for the difference between 1,542 distinct logical file changes and the 1,545 diff entries.

The existing commits named “move … part 1,” “part 2,” and “update … references” are not independently usable landing units. For example, the early source-move commit `aa47f13275a` still leaves package discovery and entry points targeting `Tensile`. The proposed PR boundaries must be reconstructed from the aggregate diff.

## Recommended split

Make package modules independent of the public package spelling, prepare a one-way compatibility adapter without activating it, move the package once, and then migrate consumers in reviewable batches. No `tensilelite` import spelling is introduced before the directory move. At the move, `tensilelite` becomes canonical and the temporary `Tensile` package forwards old imports to it.

Suggested branch prefix: `users/alvasile/AIHPBLAS-3989-namespace-split/`.

| PR / branch suffix | Purpose and included changes | Dependency | Size guidance |
| --- | --- | --- | --- |
| `01-package-aware-tests` | Make `TensileLogic` import lazily and replace the three anonymous source-file loaders with normal package imports. | `develop` | 4 files. |
| `02-relative-internal-imports` | Convert implementation imports inside `Tensile/` to explicit relative imports. Tests, launchers, and external tools keep using `Tensile`. | PR 1 | 240 imports in 60 files. |
| `03-inactive-compat-bridge` | Add and test the generic one-way namespace alias machinery. Nothing installs an alias yet, so no `tensilelite` public package is introduced. | PR 2 | Helper plus subprocess tests. |
| `04-move-and-enable-compat` | Move the complete package to `tensilelite/`, update required filesystem/package-discovery paths, and leave `Tensile/__init__.py` as the one-way compatibility shim. | PR 3 | Approximately 1,453 moves plus required build and packaging edits. |
| `05-common-and-characterization-imports` | Migrate the first mixed common/unit/characterization batch from `Tensile` to `tensilelite`. | PR 4 | 99 files. |
| `06-characterization-imports` | Migrate the next characterization batch. | PR 5 | 96 files. |
| `07-codegen-and-unit-imports` | Migrate the codegen harnesses and the adjacent unit-test batch. | PR 6 | 79 files. |
| `08-remaining-unit-imports` | Migrate the remaining ordinary unit-test imports and dotted targets. | PR 7 | 85 files. |
| `09-test-import-fixups` | Apply the original branch's focused test import and mock-target corrections. | PR 8 | 47 files. |
| `10-test-reference-cleanup` | Complete remaining test paths, comments, fixtures, and namespace references while preserving the package-aware loader tests. | PR 9 | 118 files. |
| `11-canonical-package-cleanup` | Apply the original branch's remaining implementation wording and runtime-reference changes while retaining relative imports. | PR 10 | 38 files. |
| `12-sparse-esm2-eligibility` | Carry the original branch's independent sparse ESM2 eligibility change outside the mechanical move. | PR 11 | 1 source line. |
| `13-version-script-repair` | Carry the independent `bump-version.sh` repair from the original branch. | PR 12 | 1 file. |
| `14-tooling-and-entrypoints` | Migrate non-GEKO tooling, documentation, build metadata, and the legacy launchers to the canonical lowercase package and paths. | PR 13 | 53 files. |
| `15-geko-consumers` | Apply only the GEKO changes present in `users/alvasile/AIHPBLAS-3989-package-namespace`; later installed-package/API work remains out of scope. | PR 14 | 24 files. |
| `16-remove-compat-bridge` | Remove the `Tensile` shim and bridge after the in-repo static-import audit is clean. | PR 15 | Small removal. |

The file counts above describe the current diff and help select batches. They are not measured sizes of reconstructed PRs; extracting filesystem edits and wording changes will alter the counts.

```text
develop
└── 1. Package-aware tests
    └── 2. Relative implementation imports
        └── 3. Inactive one-way bridge
            └── 4. Package move + `Tensile` shim activation
                └── 5–10. Test migration batches
                    └── 11. Canonical package cleanup
                        └── 12. Sparse ESM2 eligibility
                            └── 13. Version-script repair
                                └── 14. Tooling and entry points
                                    └── 15. GEKO consumers
                                        └── 16. Remove shim
```

Within PR groups 4 and 5, batches should touch disjoint test files where practical. Tests that depend on a shared harness change must travel with that change or explicitly depend on its PR.

## How the temporary bridge would work

The bridge presents two import spellings for one module tree after the physical move. An import of `Tensile.Common.GlobalParameters` must return the canonical `tensilelite.Common.GlobalParameters` module object, including when either spelling was imported first. Existing consumers continue working while later PRs change their import spelling.

Before the move, the bridge implementation is present but inactive and no `tensilelite` alias is exposed. The move makes `tensilelite` canonical and leaves a minimal `Tensile/__init__.py` adapter. A narrowly scoped import finder/loader recognizes only `Tensile` descendants, resolves the corresponding `tensilelite` module through normal Python importing, and caches the old name without executing the source a second time. The canonical module retains its lowercase `__name__`, `__package__`, `__spec__`, loader, and resource behavior.

Aliasing only the package root is insufficient. Allowing Python to execute a submodule under both names can duplicate module globals, classes, and registration state. The implementation must also preserve the module metadata and loader behavior used by resource lookup and module execution.

The minimum shared-state demonstration is:

```python
import importlib

old = importlib.import_module("Tensile.Common.GlobalParameters")
new = importlib.import_module("tensilelite.Common.GlobalParameters")

assert old is new
assert old.globalParameters is new.globalParameters
```

This matters to the real checkout: `Common/GlobalParameters.py` owns shared mutable state, tests such as `Tests/unit/test_TensileRetuneLibrary.py` patch state using dotted module names, and `resources.py` passes `__package__` to `importlib.resources.files()`.

Validation of the bridge must establish:

- Both import orders work in fresh processes, including imports made before the bridge is first loaded and nested modules loaded afterward.
- Classes and module globals have one identity; patching through either namespace affects callers using the other namespace.
- Normal imports, `from` imports, relative imports, missing-module errors, and import failures behave consistently without eagerly importing the entire generator.
- Module metadata, package resources, command entry points, and `python -m` work for the supported execution paths. A successful root import alone does not establish this.
- Source-checkout imports and imports from installed artifacts work without accidentally falling back to another checkout or an ambient installation.
- Representative consumers can migrate independently while unmigrated consumers continue to pass. Existing unit and characterization tests provide adjacent regression coverage.
- The later directory move can preserve these properties, and removing the bridge leaves only the intended final namespace.

The bridge is temporary migration infrastructure. It should have an explicit removal step tied to this migration, with no permanent duplicate implementation tree. Its implementation and limitations must be established by the branch experiment before relying on it to split the existing PR.

## Changes that must travel with the physical move

An import alias does not make an old filesystem path exist. PR 2 must therefore include the following changes, even when Python import edits in the same files land later:

- `projects/hipblaslt/cmake/hipblaslt_codegen.cmake`: required resource checks and direct launcher paths.
- `projects/hipblaslt/CMakeLists.txt`: installed Python package and executable locations.
- TensileLite `pyproject.toml`, `MANIFEST.in`, `tox.ini`, `pytest.ini`, and test-selection tooling: package contents, discovery, and source/test paths.
- `.github/workflows/component-ci-tensilelite-coverage.yml`: the directly executed coverage-summary script.
- GEKO launcher paths and generated shell-script paths, plus their matching test expectations.
- Test fixture paths, the codegen input manifest, and coverage-baseline file keys. Preserve the recorded coverage values while changing path keys.
- Documentation paths and ownership rules that must point to the moved files.

This separation requires splitting some existing file diffs into path-related and import-related changes. Taking entire changed files wholesale would reintroduce unnecessary dependencies between the PRs.

## Independent changes to extract first

Three changes in the current aggregate diff are separable from the namespace migration:

1. **rocisa adaptor test setup.** Commit `31b4116d817` changes two test files, with 18 insertions and 6 deletions. It adjusts subprocess initialization, assembler selection, and the logical instruction-count expectation. It can be reviewed against `develop` independently.
2. **Sparse ESM2 eligibility.** The added `if state["ProblemType"]["Sparse"]: return False` in `SolutionStructs/Solution.py:evaluateStinkyTofuESM2` changes scheduling eligibility. Review its intent and validation as a separate behavior change before including it in any reconstructed result.
3. **Version-bump script repair.** `bump-version.sh` changes obsolete version-update targets and makes paths relative to the script. Land the repair against the existing layout; the later package-directory substitution belongs in PR 2.

Wording-only changes can also land separately. Actual documentation paths must follow the associated layout change. These extractions should preserve the current branch's intended final result rather than silently discarding changes that are unrelated to renaming.

## Expected benefit and completion criteria

The directory-move PR will still list approximately 1,450 files. Its reduced content churn should make it easier for Git to recognize unchanged moves and carry concurrent edits across the rename. The import and mock-target changes then have smaller, independent landing windows. This can reduce conflict-driven CI restarts, but each PR still needs its required CI.

The proposed split is complete only when every intermediate branch works on its stated base, representative mixed old/new consumers pass, and a reconstruction of the landed changes matches the intended final branch. Temporary bridge code must disappear from that final result. Any deliberate departure from the original tree must be documented and reviewed separately.

If the bridge proves too complex or cannot preserve the required import behavior, the fallback is narrower: extract the unrelated fixes and safe preparatory changes, then keep the namespace switch atomic. Without a working bridge or equivalent preparatory changes, the existing move/import commits cannot be promoted directly into independently usable PRs.
