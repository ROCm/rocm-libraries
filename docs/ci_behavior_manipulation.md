# CI Behavior Manipulation

TheRock CI is controlled by [`configure_ci.py`](../.github/scripts/therock_configure_ci.py), where it controls push, pull request, workflow dispatch and schedule CI behavior.

## Default behavior for push and pull request

TheRock CI will determine [Linux targets](https://github.com/ROCm/rocm-libraries/blob/f4ddefcbc17bf36889295f4a97ee40ebcf1b7cdc/.github/workflows/therock-ci.yml#L94) and [Windows targets](https://github.com/ROCm/rocm-libraries/blob/f4ddefcbc17bf36889295f4a97ee40ebcf1b7cdc/.github/workflows/therock-ci.yml#L107) and [file changes](https://github.com/ROCm/rocm-libraries/blob/f4ddefcbc17bf36889295f4a97ee40ebcf1b7cdc/.github/scripts/therock_matrix.py#L7-L33), then run build and tests accordingly on file changes.

Example: a change made to `projects/rocfft` will only run `FFT` builds and tests.

For [CI changes](https://github.com/ROCm/rocm-libraries/blob/f4ddefcbc17bf36889295f4a97ee40ebcf1b7cdc/.github/scripts/therock_configure_ci.py#L114-L121), we run all build and smoke tests.

## Pull request behavior

Here are additional labels that manipulate the CI behavior. The labels we provide are:

- `skip-therockci`: The CI will skip all builds and tests

### Label-gated cmake options

A label can also turn on a cmake option for a single project's superbuild, so a branch
can exercise a feature flag in CI without changing the default build for everyone else.
The option is added only when the label is on the pull request *and* that project is
already being built; it never applies to pushes to `develop`, nightly runs, or workflow
dispatch.

These labels are declared in
[`LABEL_GATED_CMAKE_OPTIONS`](../.github/scripts/therock_matrix.py) — one entry per
label, naming the target project and the options to inject. Adding an entry requires a
code change plus a matching label in the repository's label set, since these labels are
applied by hand and are not assigned by `labeler.yml`.

The gated build replaces the normal one for that project rather than running alongside
it, so there is no second job and no duplicate artifact — but it also means the
flag-off configuration is not built while the label is applied.

The target project does not need a job of its own. Projects get merged together when
they are built as a set — an optional component folds into its parent, and a dependency
folds into the project that absorbs it — and the option follows its target onto
whichever job ends up building it. Labelling `miopen` works whether the pull request
touches `projects/miopen` directly or only `projects/hipdnn`. The option is applied
last, so it beats a conflicting default that a merge brought in.

## Workflow dispatch behavior

For `workflow_dispatch`, you are able to trigger CI in [GitHub's therock-ci.yml workflow page](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-ci.yml). To trigger a workflow dispatch, click "Run workflow" and fill in the fields accordingly.
