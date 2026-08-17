# CI Behavior Manipulation

TheRock CI is controlled by [`configure_ci.py`](../.github/scripts/therock_configure_ci.py), where it controls push, pull request, workflow dispatch and schedule CI behavior.

## Default behavior for push and pull request

TheRock CI will determine [Linux targets](https://github.com/ROCm/rocm-libraries/blob/f4ddefcbc17bf36889295f4a97ee40ebcf1b7cdc/.github/workflows/therock-ci.yml#L94) and [Windows targets](https://github.com/ROCm/rocm-libraries/blob/f4ddefcbc17bf36889295f4a97ee40ebcf1b7cdc/.github/workflows/therock-ci.yml#L107) and [file changes](https://github.com/ROCm/rocm-libraries/blob/f4ddefcbc17bf36889295f4a97ee40ebcf1b7cdc/.github/scripts/therock_matrix.py#L7-L33), then run build and tests accordingly on file changes.

Example: a change made to `projects/rocfft` will only run `FFT` builds and tests.

For [CI changes](https://github.com/ROCm/rocm-libraries/blob/f4ddefcbc17bf36889295f4a97ee40ebcf1b7cdc/.github/scripts/therock_configure_ci.py#L114-L121), we run all build and smoke tests.

## Pull request behavior

Here are additional labels that manipulate the CI behavior. The labels we provide are:

- `skip-therockci`: The CI will skip all builds and tests

## Workflow dispatch behavior

For `workflow_dispatch`, you are able to trigger CI in [GitHub's therock-ci.yml workflow page](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-ci.yml). To trigger a workflow dispatch, click "Run workflow" and fill in the fields accordingly.
