# TensileLite mutation-analysis execution reference

## Contents

- [Purpose and scope](#purpose-and-scope)
- [Platform support](#platform-support)
- [Record preflight state](#1-record-preflight-state)
- [Configure a mutation slice safely](#2-configure-a-mutation-slice-safely)
- [Run mutmut](#3-run-mutmut)
- [Restore the mutation configuration](#4-restore-the-mutation-configuration)
- [Verify survivor-killing tests](#5-verify-survivor-killing-tests)
- [Run the safety-core self-tests](#6-run-the-safety-core-self-tests)

## Purpose and scope

In this project, mutation testing is an opt-in, offline technique used only to
find behavioral coverage and assertion gaps in the characterization tests. A
surviving mutant is a diagnostic lead, not a product failure. When a survivor
exposes an observable gap, the lasting change is a focused characterization
test in the normal pytest suite—not a mutation score, report, or source change
made only to improve that score. Mutation results are not CI or release gates.

The scripts and Dockerfile in this skill are executable documentation for
maintainers and coding agents. They are not required to build TensileLite or
run its ordinary tests, and no normal CI workflow invokes them. These optional
helpers make a manual, serial mutmut investigation reproducible and protect the
checkout while it runs:

- `slice-preflight.sh` records the source/container environment and refuses to
  proceed when tracked source is dirty.
- `pyproject-mutmut.sh` backs up, rewrites, restores, and verifies the
  `[tool.mutmut]` configuration in `pyproject.toml`.
- `mutmut-verify.sh` checks a proposed characterization test against clean and
  mutated source, then restores every detected mutation change.

Mutmut itself runs without these wrappers. The helpers add the following
TensileLite-specific reproducibility and safety guarantees across repeated runs:

- record the exact source, container image, and mutmut version used;
- change per-slice pyproject selections without leaving tracked changes;
- preserve the rocisa-compatible mutmut configuration;
- bound worker contention so healthy mutants are not misclassified as timeouts;
- distinguish pytest assertion failures from collection, usage, and internal
  errors; and
- prove that every applied mutant was reverted.

## Platform support

Mutmut requires a Linux environment. On Windows, use Windows Subsystem for
Linux (WSL) or another Linux environment; native Windows and PowerShell are
unsupported because mutmut 3.6 relies on `fork`, Unix resource limits, and Unix
signals. The documented workflow normally runs in Docker.

Docker is not required for ordinary TensileLite tests, and mutmut itself can run
in any compatible Linux environment with rocisa installed. The
`mutation-unit` tox environment is opt-in and is not part of tox's default
environment list or the project's CI workflows. Of the bundled helpers,
`pyproject-mutmut.sh` runs on the host, while the current `slice-preflight.sh`
and `mutmut-verify.sh` implementations require a named Docker container.

Run all examples from the `rocm-libraries` repository root. The repository does
not publish a prebuilt image. Build the dedicated mutation image, mount the
current worktree read-write at `/work`, and provision the mutation tox
environment and rocisa from that mounted revision:

```bash
docker build \
  -f projects/hipblaslt/skills/tensilelite-mutation-rerun/scripts/Dockerfile \
  -t hipblaslt-mutation \
  projects/hipblaslt/skills/tensilelite-mutation-rerun/scripts

docker run -d --name tl-mut \
  --mount type=bind,source="$(pwd -P)",target=/work \
  hipblaslt-mutation

docker exec tl-mut /opt/tl-mut-tools/bin/tox -e mutation-unit --notest
docker exec tl-mut \
  /opt/tl-mut/mutation-unit/bin/pip install ./rocisa
```

The image contains the ROCm, Python, native-build, and tox bootstrap tooling,
but it does not copy or clone the repository. Its default working directory is
the mounted TensileLite project, and it keeps the mutation tox environment at
`/opt/tl-mut/mutation-unit` so dependencies are not written into the worktree.

The `/work` mount must point to the same worktree passed through `--root` (or the
current worktree when `--root` is omitted). `mutmut-verify.sh` rejects a missing,
read-only, or mismatched mount so it cannot apply a mutant in one tree and report
that a different tree was restored. Replace `tl-mut` when using another container.

## 1. Record preflight state

The preflight command is read-only with respect to source; it writes only its
`env.json` output. It requires a clean tracked source tree and an existing
Docker container. Untracked mutation output does not fail the check.

```bash
bash projects/hipblaslt/skills/tensilelite-mutation-rerun/scripts/slice-preflight.sh \
  --slice 1 \
  --module Tensile/Common/Utilities.py \
  --container tl-mut \
  --out work/mutation/slices/1-utilities
```

Expected terminal output has this shape (SHA, branch, status, and version vary):

```text
slice-preflight: OK slice=1 module=Tensile/Common/Utilities.py slug=utilities
slice-preflight: wrote work/mutation/slices/1-utilities/env.json
slice-preflight: sha=<commit> branch=<branch> container=tl-mut(running) mutmut=<version>
```

The generated `env.json` records the source SHA/branch, tracked-source-clean
state, container image and ID, container status, and mutmut version. A dirty
tracked file or missing container makes the command exit non-zero without
writing a new artifact.

## 2. Configure a mutation slice safely

Start a transaction by backing up the complete project file before changing the
mutmut selection:

```bash
bash projects/hipblaslt/skills/tensilelite-mutation-rerun/scripts/pyproject-mutmut.sh \
  backup --src projects/hipblaslt/tensilelite
```

Expected output:

```text
pyproject-mutmut: backup <absolute-source>/pyproject.toml -> <absolute-backup>/pyproject.toml.bak
pyproject-mutmut: metadata <absolute-backup>/pyproject.toml.bak.meta.json
```

The backup records the source path, current Git `HEAD`, original file hash and
mode, and the hash and mode produced by each successful `set`. An existing
backup is not overwritten: complete the active transaction before starting
another one. `set` rejects a stale or unrelated source state and atomically
replaces the project file only after generating and validating the complete
result.

Set one or more source modules and pytest selections:

```bash
bash projects/hipblaslt/skills/tensilelite-mutation-rerun/scripts/pyproject-mutmut.sh \
  set \
  --src projects/hipblaslt/tensilelite \
  --only-mutate Tensile/Common/Utilities.py \
  --test-selection Tensile/Tests/unit/characterization/CommonUtilities
```

Expected output:

```text
pyproject-mutmut: set rewrote only_mutate, pytest_add_cli_args_test_selection
pyproject-mutmut: set OK in <absolute-source>/pyproject.toml
```

Do not restore the backup yet. Run and inspect the slice first.

## 3. Run mutmut

Run the configured slice in the container. Always bound worker count; the
committed timeout multiplier assumes no more than 32 concurrent children.

The supported tox entry point runs the complete campaign and applies the same
cap by default:

```bash
docker exec tl-mut /opt/tl-mut-tools/bin/tox -e mutation-unit
```

Alternatively, after provisioning the environment, invoke mutmut directly:

```bash
docker exec \
  -w /work/projects/hipblaslt/tensilelite \
  tl-mut \
  mutmut run --max-children 32
```

For an intentionally smaller campaign, pass the environment variable into the
container when using tox, or pass a lower `--max-children` value directly.
Values above the reviewed default of 32 are unsupported:

```bash
docker exec -e MUTMUT_MAX_CHILDREN=8 tl-mut \
  /opt/tl-mut-tools/bin/tox -e mutation-unit
```

`mutmut run` renders interactive progress while it collects tests, generates
mutants, and executes them. After completion, print all non-killed results:

```bash
docker exec \
  -w /work/projects/hipblaslt/tensilelite \
  tl-mut \
  mutmut results
```

Representative output (mutant names and statuses vary):

```text
    Tensile.Common.Utilities.x__mutmut_1: survived
    Tensile.Common.Utilities.x__mutmut_2: no tests
    Tensile.Common.Utilities.x__mutmut_3: timeout
```

Inspect a survivor's exact source change before deciding whether it exposes a
characterization gap:

```bash
docker exec \
  -w /work/projects/hipblaslt/tensilelite \
  tl-mut \
  mutmut show Tensile.Common.Utilities.x__mutmut_1
```

Rerun one mutant after adding a focused test with:

```bash
docker exec \
  -w /work/projects/hipblaslt/tensilelite \
  tl-mut \
  mutmut run Tensile.Common.Utilities.x__mutmut_1 --max-children 1
```

## 4. Restore the mutation configuration

After the mutmut run, restore the byte-exact backup and verify that
`pyproject.toml` matches `HEAD` before running the survivor verifier, which
requires a clean tracked worktree. Restore validates the transaction's source
path, Git `HEAD`, recorded hashes, and file mode first; it refuses to overwrite
unrelated edits or use a stale backup:

```bash
bash projects/hipblaslt/skills/tensilelite-mutation-rerun/scripts/pyproject-mutmut.sh \
  restore --src projects/hipblaslt/tensilelite

bash projects/hipblaslt/skills/tensilelite-mutation-rerun/scripts/pyproject-mutmut.sh \
  assert-clean --src projects/hipblaslt/tensilelite
```

Expected output:

```text
pyproject-mutmut: restore <absolute-backup>/pyproject.toml.bak -> <absolute-source>/pyproject.toml
pyproject-mutmut: cleared backup transaction
pyproject-mutmut: assert-clean OK (pyproject.toml == HEAD)
```

## 5. Verify survivor-killing tests

`mutmut-verify.sh` changes tracked source while each mutant is active. Run only
one verifier at a time and start from a clean tracked worktree. It discovers
every tracked path and every newly created, non-ignored path changed by mutant
application, then checks that set against the manifest declaration. Cleanup
restores those paths and verifies the complete tracked state plus the original
set of untracked path names, including after an interruption. Existing ignored
artifacts and the contents of existing untracked files are outside that
baseline. A stale target declaration or an unexpected multi-file change fails
the row instead of leaving detected source changes behind.

A newly added, still-untracked characterization test is compatible with this
clean-tracked-worktree requirement. If verification requires editing an existing
tracked test, make a deliberate local commit or use a separate clean worktree;
do not hide, stash, or discard unrelated edits automatically.

Create a tab-separated manifest. The header and column order are required:

```text
mutant_id	file	apply_method	test_node	expect_clean_rc	expect_mutant_rc_nonzero
Tensile.Common.Utilities.x__mutmut_1	Tensile/Common/Utilities.py	mutmut_apply	Tensile/Tests/unit/characterization/CommonUtilities/test_example.py::test_example	0	true
```

For kill-proof rows, `expect_clean_rc` must be `0`: a test that already fails on
clean source cannot prove that a mutant was killed. Set
`expect_mutant_rc_nonzero` to `true` when the mutant must produce pytest's
assertion-failure status `1`. The verifier rejects an empty, malformed, or
duplicate manifest before it changes source.

Then run the verifier:

```bash
bash projects/hipblaslt/skills/tensilelite-mutation-rerun/scripts/mutmut-verify.sh \
  --container tl-mut \
  --manifest work/mutation/manifest.tsv \
  --out work/mutation/verify \
  --src projects/hipblaslt/tensilelite
```

A successful kill has output like:

```text
MUTANT                       VERDICT DETAIL
Tensile...__mutmut_1         KILLED   base_rc=0 mut_rc=1
============================================================
CLEAN: tracked worktree and untracked path set match the pre-run baseline.
RESULT: ALL KILLED (1)
kill_matrix: <absolute-output>/kill_matrix.tsv
```

The verifier writes:

- `kill_matrix.tsv`: one structured result row per manifest entry.
- `verify-report.txt`: a human-readable summary.
- `row-*-clean.log` and `row-*-mutant.log`: the retained pytest output for each
  clean and mutated execution.

Verdict rules are intentionally strict:

- `KILLED`: the clean node returned status `0`, the mutated node returned pytest
  assertion status `1`, and restoration was clean.
- `OK`: an explicitly expected-pass row remained at status `0`. It is successful
  but is not counted as a kill.
- `BAD`: the clean baseline was wrong, the mutant survived, application failed,
  or restoration leaked changes.
- `INCONCLUSIVE`: pytest collection, usage, internal, interruption, or another
  non-assertion error occurred. Infrastructure errors are never counted as kills.

Any `BAD` or `INCONCLUSIVE` row makes the verifier exit non-zero. A successful
report containing only kills says `ALL KILLED` and includes the kill count. A
successful report containing an expected-pass row instead says `SUCCESS` and
reports separate `KILLED` and `EXPECTED_PASS` counts.

## 6. Run the safety-core self-tests

The safety tests use temporary Git repositories and a fake Docker client. They
exercise preflight failures, manifest validation, result classification,
interruption cleanup, changed-path restoration, and configuration transaction
recovery without contacting a Docker daemon or running a mutation campaign:

```bash
bash projects/hipblaslt/skills/tensilelite-mutation-rerun/scripts/tests/run-selftests.sh
```
