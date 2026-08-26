# Development Guide: `.github/scripts`

Python automation scripts that support CI/CD workflows for the ROCm Libraries
monorepo. This guide covers setup, the script inventory, how to run tests, and
script-specific guidance for contributors.

---

## Setup

Many Linux distributions manage the system Python through the OS package manager
and reject `pip install` without `--break-system-packages`. Use a venv instead:

```bash
# One-time setup
python3 -m venv ~/.venv/rocm-scripts
source ~/.venv/rocm-scripts/bin/activate
pip install -r .github/requirements.txt
```

Or without activating:

```bash
python3 -m venv /tmp/rocm-scripts-venv
/tmp/rocm-scripts-venv/bin/pip install -r .github/requirements.txt
```

No build step is required. All scripts are plain Python 3.12 and import only
from the standard library plus the packages above.

---

## Directory Layout

> Keep this tree and the [Workflow → Script Reference](#workflow--script-reference)
> table up to date when adding, removing, or renaming scripts.

```
.github/
├── scripts/
│   ├── DEVELOPMENT.md                  ← this file
│   │
│   ├── # Shared libraries
│   ├── ci_utils.py                     ← retry, path matching, GITHUB_OUTPUT helpers
│   ├── config_loader.py                ← loads/validates repos-config.json
│   ├── github_cli_client.py            ← GitHub REST API wrapper (reads GH_TOKEN)
│   ├── repo_config_model.py            ← Pydantic model for repos-config.json entries
│   │
│   ├── # Mirror sync (monorepo → sub-repos)
│   ├── pr_detect_changed_subtrees.py   ← which subtrees did a merged PR touch?
│   ├── pr_merge_sync_patches.py        ← generate patch + push to sub-repo
│   │
│   ├── # TheRock / multi-arch CI
│   ├── resolve_therock_ref.py          ← pin TheRock commit for a PR/push event
│   ├── therock_configure_ci.py         ← select build flags and tests from subtrees
│   ├── therock_matrix.py               ← subtree → TheRock project mapping table
│   ├── component_ci.py                 ← per-component CI trigger flags
│   ├── check_wheel_freshness.py        ← fail CI if nightly wheels are stale
│   │
│   ├── # PR automation
│   ├── pr_category_label.py            ← add/remove category labels from changed paths
│   ├── apply-labels.py                 ← apply labels to a PR via GitHub API
│   ├── collect-labels.py               ← collect labels from a source repo
│   ├── import_subrepo_prs.py           ← import sub-repo PRs into the monorepo
│   ├── notify_teams.py                 ← route CI failure alerts to MS Teams channels
│   │
│   ├── # Repo maintenance
│   ├── get_changed_projects.py         ← git-diff → validated changed project list
│   ├── merge-codeowners.py             ← merge per-project CODEOWNERS into one file
│   ├── merge-submodules.py             ← merge per-project .gitmodules into one file
│   │
│   └── tests/
│       ├── fixtures/                   ← .patch files for commit-message tests
│       ├── resolve_therock_ref_test.py
│       ├── test_pr_detect_changed_subtrees.py
│       ├── test_pr_merge_sync_patches.py
│       ├── therock_configure_ci_test.py
│       └── therock_matrix_test.py
│
├── repos-config.json                   ← subtree → sub-repo mapping (read by config_loader)
└── requirements.txt                    ← pydantic, requests
```

---

## Workflow → Script Reference

> Keep this table up to date when workflows are added or scripts are moved.

| Workflow file | Script(s) invoked |
|---|---|
| `pr-merge-sync-patches.yml` | `pr_detect_changed_subtrees.py`, `pr_merge_sync_patches.py` |
| `pr-merge-sync-patches-manual.yml` | `pr_detect_changed_subtrees.py`, `pr_merge_sync_patches.py` |
| `therock-ci.yml` / `therock-multi-arch-ci.yml` | `resolve_therock_ref.py`, `therock_configure_ci.py` |
| `component-ci.yml` | `component_ci.py` |
| `therock-ci-nightly.yml` | `check_wheel_freshness.py` |
| `labeler.yml` / `pr-org-label.yml` | `apply-labels.py`, `collect-labels.py`, `pr_category_label.py` |
| `pr-import.yml` / `multiple_pr_import.yml` | `import_subrepo_prs.py` |
| `libraries-pr-bot.yml` | `notify_teams.py` |
| `merge-codeowners.yml` | `merge-codeowners.py` |
| `merge-submodules.yml` | `merge-submodules.py` |

---

## Running the Tests

Tests live in `.github/scripts/tests/` and use `unittest`, runnable via
`pytest` with no extra configuration. Run from the `scripts/` directory:

```bash
# Run the full suite
cd .github/scripts
pytest . -v

# Without an activated venv (substitute your venv path)
/tmp/rocm-scripts-venv/bin/pytest . -v

# Run one file
pytest tests/test_pr_merge_sync_patches.py -v
```

`tests/conftest.py` automatically stubs out modules that are only available
in CI (such as `amdgpu_family_matrix` from a TheRock checkout), so `pytest .`
works locally without any extra setup.

> **Note:** there is currently no CI job that runs these tests automatically.
> Until one is added, run the suite locally before merging any change to a
> script that has coverage.

### Test conventions

- File naming: use `test_<script_stem>.py` for new files. Pre-existing files use
  `<script_stem>_test.py`; leave them as-is when making unrelated changes.
- Keep tests free of real network calls. `GitHubCLIClient` is the only class
  that makes HTTP requests — replace it with a `MagicMock` or a hand-written
  stub in every test that reaches it.
- Use `tempfile.TemporaryDirectory` for any test that needs real filesystem or
  git operations.
- Follow the import pattern used by the existing test files:

```python
import sys, os, unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import my_script as sut   # "system under test"
```

---

## TDD Approach for Bug Fixes

When fixing a bug in an existing script:

1. **Write a failing test first.** The test should assert the *desired* behavior
   against the current (unfixed) code — it will fail.
2. **Fix the production code** until the test passes.
3. **Run the full suite** to confirm nothing regressed.

Tests that document *current* buggy behavior (written before the fix) should
use `@pytest.mark.xfail(strict=True)` so the suite stays green until the bug
is fixed, at which point the mark is simply removed. The `reason` string should:
- Describe what is currently broken
- Reference the relevant issue or PR

```python
@pytest.mark.xfail(reason="BUG: <description of what currently happens>, see ALMIOPEN-XXXX", strict=True)
def test_some_buggy_behavior(self):
    result = sut.some_function(...)
    assert result == CORRECT_VALUE  # the value it SHOULD produce
```

---

## Adding a CI Job for Script Tests

Add a lightweight job to `pre-commit.yml` or a new `scripts-unit-tests.yml`:

```yaml
- name: Run script unit tests
  run: |
    pip install -r .github/requirements.txt
    pytest .github/scripts/tests/ -v
```

---

## Script-Specific Guidance

### Mirror Sync — `pr_detect_changed_subtrees.py` + `pr_merge_sync_patches.py`

These scripts run on every merge to `develop` and push changes from the
monorepo to the corresponding sub-repositories. **Never test changes to these
by merging to `develop`** — use the approaches below.

#### Dry-run (read-only, uses real GitHub API)

Both scripts accept `--dry-run --debug`. In this mode every step runs except
writing to `GITHUB_OUTPUT` or pushing to a sub-repo. A `GH_TOKEN` with
read-only PR access is sufficient.

```bash
export GH_TOKEN=<your-personal-access-token>

# Detect changed subtrees for a previously merged PR
python .github/scripts/pr_detect_changed_subtrees.py \
  --repo ROCm/rocm-libraries \
  --pr <pr-number> \
  --require-auto-push \
  --dry-run --debug

# Simulate the full patch-and-push cycle without pushing
python .github/scripts/pr_merge_sync_patches.py \
  --repo ROCm/rocm-libraries \
  --pr <pr-number> \
  --subtrees "projects/rocBLAS" \
  --dry-run --debug
```

Use the PR number of an already-merged PR. The merge commit is already on
`develop`, so a real patch can be generated without touching anything.

#### Local bare-repo harness (no GitHub token needed)

For git-level bugs (staging, patch application, push reliability):

```bash
# Create a fake bare "sub-repo" target
git init --bare /tmp/test-subrepo.git
git clone /tmp/test-subrepo.git /tmp/test-subrepo-seed
git -C /tmp/test-subrepo-seed commit --allow-empty -m "initial"
git -C /tmp/test-subrepo-seed push origin main
rm -rf /tmp/test-subrepo-seed

# Generate a patch from a real or test commit
MERGE_SHA=$(git rev-parse HEAD)
git format-patch -1 "$MERGE_SHA" --relative=projects/mylib \
  --output /tmp/mylib.patch

# Apply manually to inspect behavior
git clone /tmp/test-subrepo.git /tmp/test-apply
git -C /tmp/test-apply apply /tmp/mylib.patch
```

#### Manual workflow re-trigger (nearest thing to a staging run)

1. Open **Actions → Manual Patch Rerun → Run workflow**
2. Enter the PR number of a previously merged PR that touched your subtree
3. The workflow uses real credentials and pushes to the real sub-repo

> ⚠️ Only use this for PRs whose changes were already intended to land.

#### PR checklist for mirror-sync fixes

- [ ] Root cause described in the PR body
- [ ] Fix is in the smallest possible function
- [ ] Unit test in `tests/` reproduces the bug before the fix and passes after
- [ ] `pytest .github/scripts/tests/ -v` passes locally
- [ ] `--dry-run --debug` smoke test passes against a real merged PR number
- [ ] PR references the issue being fixed (link or description in PR body)

---

### TheRock CI — `resolve_therock_ref.py`, `therock_configure_ci.py`, `therock_matrix.py`

`resolve_therock_ref.py` pins the TheRock commit that a multi-arch CI run
builds against. The resolution policy (merge-base for PRs, live tip for
pushes) is documented in the module docstring.

`therock_configure_ci.py` reads the `SUBTREES` environment variable and
selects which build flags and tests to run. It imports from `therock_matrix.py`
(the static subtree → project mapping) and from `pr_detect_changed_subtrees.py`.

All three have test coverage in `tests/`:

```bash
pytest tests/resolve_therock_ref_test.py tests/therock_configure_ci_test.py tests/therock_matrix_test.py -v
```

These tests use an in-process `FakeClient` that stubs out all HTTP calls —
follow the same pattern for any new tests.

---

### PR Automation — `pr_category_label.py`, `apply-labels.py`, `collect-labels.py`

These scripts apply category labels to PRs based on changed file paths.
`apply-labels.py` and `collect-labels.py` use raw `requests` calls rather than
`GitHubCLIClient`. They have no unit tests yet; if you change them, add tests
before merging.

---

### Notifications — `notify_teams.py`

Routes CI failure alerts to the correct Microsoft Teams channel. Webhook URLs
are stored as repository secrets and are not accessible in dry-run/local
testing. The routing logic (PR title tag matching, component-name fallback) is
pure Python and is independently testable without a live webhook.

Use `--dry-run` to exercise the routing logic locally without sending a
notification:

```bash
python .github/scripts/notify_teams.py \
  --failure-stage test \
  --log-path ./test_output.log \
  --webhook-urls '{"miopen": "https://placeholder"}' \
  --pr-number 123 \
  --pr-title "[miopen] Fix something" \
  --component-name miopen \
  --dry-run
```

---

### Repo Maintenance — `merge-codeowners.py`, `merge-submodules.py`

These scripts have no external dependencies beyond the standard library and
operate only on the local filesystem. They are safe to run locally at any time:

```bash
python .github/scripts/merge-codeowners.py   # rewrites .github/CODEOWNERS
python .github/scripts/merge-submodules.py   # rewrites .gitmodules
```

Inspect the diff before committing. Neither script has unit tests yet.
