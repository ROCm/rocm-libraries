# Development Guide: `.github/scripts`

This guide covers how to develop, test, and validate changes to the automation
scripts that power the CK mirror sync system — specifically:

- `pr_detect_changed_subtrees.py` — detects which subtrees a merged PR touched
- `pr_merge_sync_patches.py` — generates a patch from the merge commit and pushes it to the corresponding sub-repositories

These scripts run in GitHub Actions on every merge to `develop`
(`pr-merge-sync-patches.yml`) and can be re-triggered manually
(`pr-merge-sync-patches-manual.yml`). **All testing and validation should be
done locally or via dry-run before any change reaches production.**

---

## Directory Layout

```
.github/
├── scripts/
│   ├── DEVELOPMENT.md              ← this file
│   ├── config_loader.py            ← loads repos-config.json
│   ├── github_cli_client.py        ← thin wrapper around the GitHub REST API
│   ├── pr_detect_changed_subtrees.py
│   ├── pr_merge_sync_patches.py
│   ├── repo_config_model.py        ← Pydantic model for repos-config.json entries
│   └── tests/
│       ├── resolve_therock_ref_test.py
│       ├── therock_configure_ci_test.py
│       └── therock_matrix_test.py
├── repos-config.json               ← subtree → sub-repo mapping
└── requirements.txt                ← pydantic, requests
```

---

## Setup

The system Python on many AMD developer machines is managed by the OS and
rejects `pip install` without `--break-system-packages`. Use a venv instead:

```bash
# One-time setup (from anywhere)
python3 -m venv ~/.venv/mirror-sync
source ~/.venv/mirror-sync/bin/activate
pip install pydantic requests pytest
```

Or, without activating:

```bash
python3 -m venv /tmp/mirror-sync-venv
/tmp/mirror-sync-venv/bin/pip install pydantic requests pytest
```

No build step is required. All scripts are plain Python 3.12 and import only
from the standard library plus the two packages above.

---

## Running the Unit Tests

Tests live in `.github/scripts/tests/` and use `unittest` (runnable via
`pytest` with no configuration needed).

```bash
# If your venv is activated:
pytest .github/scripts/tests/ -v

# Without activating (substituting your venv path):
/tmp/mirror-sync-venv/bin/pytest .github/scripts/tests/ -v

# Run a single test file
pytest .github/scripts/tests/test_pr_merge_sync_patches.py -v
```

**When adding a fix for any mirror-sync bug, include a corresponding test in
`.github/scripts/tests/`.** The naming convention is
`<script_stem>_test.py`, e.g. `pr_merge_sync_patches_test.py`.

### Test pattern

Existing tests use a `FakeClient` / mock-object pattern to avoid any real
GitHub API calls. Follow the same pattern:

```python
import sys, os, unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import pr_merge_sync_patches as sut   # "system under test"

class MyTest(unittest.TestCase):
    def test_something(self):
        ...
```

`GitHubCLIClient` is the only class that makes real network calls. Replace it
with a `MagicMock` or a hand-written stub in every test that exercises code
paths that call it.

---

## Dry-Run: Safe End-to-End Validation

Both scripts accept `--dry-run`. In this mode they perform every step **except**
writing to `GITHUB_OUTPUT` or pushing to a sub-repo. A `GH_TOKEN` with
read-only PR access is still required.

```bash
export GH_TOKEN=<your-personal-access-token>   # read:org, repo (read-only sufficient)

# Detect which subtrees a real merged PR touched (no GITHUB_OUTPUT written)
python .github/scripts/pr_detect_changed_subtrees.py \
  --repo ROCm/rocm-libraries \
  --pr <pr-number> \
  --require-auto-push \
  --dry-run --debug

# Simulate the full patch-and-push cycle without actually pushing
python .github/scripts/pr_merge_sync_patches.py \
  --repo ROCm/rocm-libraries \
  --pr <pr-number> \
  --subtrees "projects/rocBLAS" \
  --dry-run --debug
```

Use a PR number from a **previously merged** PR that touched the subtree you
care about. The merge commit is already on `develop`, so the script can
generate a real patch without touching anything.

---

## Local Bare-Repo Harness (for git-level bugs)

For bugs that involve git behaviour (staging, patch application, push
reliability), you can create fully local fake repositories — no GitHub token
or network access required.

```bash
# 1. Create a fake "source" commit in your working tree
git commit --allow-empty -m "test: trigger mirror sync"
MERGE_SHA=$(git rev-parse HEAD)

# 2. Create a fake bare "sub-repo" target
git init --bare /tmp/test-subrepo.git
git clone /tmp/test-subrepo.git /tmp/test-subrepo-seed
git -C /tmp/test-subrepo-seed commit --allow-empty -m "initial"
git -C /tmp/test-subrepo-seed push origin main
rm -rf /tmp/test-subrepo-seed

# 3. Craft a minimal repos-config override pointing at the local bare repo
cat > /tmp/test-repos-config.json <<'EOF'
[
  {
    "name": "mylib",
    "category": "projects",
    "url": "local-org/mylib",
    "branch": "main",
    "auto_subtree_push": true,
    "monorepo_source_of_truth": true
  }
]
EOF

# 4. Generate a real patch from your test commit
git format-patch -1 "$MERGE_SHA" --relative=projects/mylib \
  --output /tmp/mylib.patch

# 5. Inspect / test patch application directly
git clone /tmp/test-subrepo.git /tmp/test-apply
git -C /tmp/test-apply apply /tmp/mylib.patch   # or apply --3way, etc.
```

This harness lets you reproduce and verify fixes for:

- **AICK-2008** — files silently dropped because `git add .` honors `.gitignore`
- **AICK-2010** — push failures not surfaced (retry / non-zero exit)
- **AICK-2011** — patch application failing on a second run (idempotency)

---

## Testing Each Known Bug

### AICK-2007 — Commit message mangling

The fix lives entirely in `_extract_commit_message_from_patch()`. Test with
fixture `.patch` files that contain:

- Multi-line folded subjects (RFC 2822 line continuation)
- MIME-encoded words (`=?UTF-8?q?...?=`) from non-ASCII author names
- `[PATCH 1/3]` prefixes (not just `[PATCH]`)
- Content after the `---` separator bleeding into the subject

Place fixture files under `.github/scripts/tests/fixtures/` and load them in
the test:

```python
FIXTURES = Path(__file__).parent / "fixtures"

class CommitMessageTest(unittest.TestCase):
    def test_mime_encoded_subject_is_decoded(self):
        patch_path = FIXTURES / "mime_encoded_subject.patch"
        result = sut._extract_commit_message_from_patch(patch_path)
        self.assertNotIn("=?UTF-8?", result)
```

### AICK-2008 — Force-tracked files dropped by `git add .`

Write a test that:
1. Creates a `tempfile.TemporaryDirectory()` with `git init`
2. Commits a file that is listed in `.gitignore`
3. Calls `_stage_changes()` (current implementation)
4. Asserts `git status` shows the file as **not** staged (demonstrating the bug)
5. After the fix, asserts it **is** staged

Use `git add --force` or `git add -A` in the fix, and confirm the test passes.

### AICK-2009 — Silent no-op when no subtrees match

```python
class NoMatchTest(unittest.TestCase):
    def test_unmatched_subtrees_raises_or_exits_nonzero(self):
        config = load_repo_config(".github/repos-config.json")
        result = sut.get_subtree_info(config, ["projects/doesnotexist"])
        # After the fix: either raises, or main() returns non-zero
        self.assertEqual(result, [])   # replace with assertion on exit code
```

### AICK-2010 — Fire-and-forget push

```python
class PushFailureTest(unittest.TestCase):
    @patch("pr_merge_sync_patches._run_git", side_effect=RuntimeError("push failed"))
    def test_push_failure_propagates(self, _):
        with self.assertRaises(RuntimeError):
            sut._push_changes(Path("/tmp/fake"), "main")
```

After the fix (retry + re-raise), assert that the mock was called the expected
number of times and that the exception is still propagated after exhausting
retries.

### AICK-2011 — Non-idempotent patch application

Write a test that applies the same patch twice to a `git init` temp repo and
asserts that:
- First application succeeds
- Second application either succeeds silently (already-synced guard) or raises
  a clear, actionable error — **not** a confusing `git apply` conflict

---

## Manual Workflow Re-trigger (staging equivalent)

If you need to validate against a real sub-repo without modifying `develop`,
use the **Manual Patch Rerun** workflow:

1. Open **Actions → Manual Patch Rerun → Run workflow**
2. Enter the PR number of a previously merged PR that touched your subtree
3. The workflow will detect subtrees, generate the patch, and push — using
   real credentials but on an already-merged commit

> ⚠️ This pushes to the real sub-repo. Only use it for PRs whose changes were
> already intended to land. If you want a pure read-only check, add `--dry-run`
> to the workflow's script invocation temporarily.

---

## Adding a CI Job for Script Tests

There is currently no CI job that runs `.github/scripts/tests/`. When adding
new tests, consider opening a follow-up PR to add a lightweight job to
`pre-commit.yml` or a new `scripts-unit-tests.yml`:

```yaml
- name: Run script unit tests
  run: pytest .github/scripts/tests/ -v
```

This ensures regressions in the mirror sync logic are caught on every PR, not
just when someone runs the tests locally.

---

## Checklist for Mirror-Sync Bug-Fix PRs

- [ ] Root cause is identified and described in the PR body
- [ ] Fix is in the smallest possible function (prefer pure functions over
      side-effectful ones)
- [ ] A unit test in `.github/scripts/tests/` reproduces the bug **before**
      the fix and passes **after**
- [ ] `pytest .github/scripts/tests/ -v` passes locally
- [ ] `--dry-run --debug` smoke test passes against a real merged PR number
- [ ] PR references the relevant AICK ticket
