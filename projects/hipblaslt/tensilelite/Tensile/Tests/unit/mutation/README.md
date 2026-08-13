# TensileLite mutation-testing core

These scripts provide the safety-critical foundation for a serial mutmut run:

- `slice-preflight.sh` records the source/container environment and refuses to
  proceed when tracked source is dirty.
- `pyproject-mutmut.sh` backs up, rewrites, restores, and verifies the
  `[tool.mutmut]` configuration in `pyproject.toml`.
- `mutmut-verify.sh` proves individual kills by running one pytest node against
  clean and mutated source, then restoring the source file.

Mutmut itself runs without these wrappers. The wrappers preserve the
TensileLite-specific campaign contract across reruns:

- record the exact source, container image, and mutmut version used;
- change per-slice pyproject selections without leaving tracked changes;
- preserve the rocisa-compatible mutmut configuration;
- bound worker contention so healthy mutants are not misclassified as timeouts;
- distinguish pytest assertion failures from collection, usage, and internal
  errors; and
- prove that every applied mutant was reverted.

## Platform support

The supported campaign environment is Linux or WSL, normally using the mutation
Docker container. Mutmut 3.6 does not support native Windows: it exits with a
request to use WSL and relies on `fork`, Unix resource limits, and Unix signals.
These Bash helpers therefore do not reduce the supported platform set for the
actual mutation run. Native PowerShell execution is not supported.

Run all examples from the `rocm-libraries` repository root. The examples assume
an already-created container named `tl-mut` with the repository mounted at
`/work`. Replace that name when using a different container.

## 1. Record preflight state

The preflight command is read-only. It requires a clean tracked source tree and
an existing Docker container. Untracked mutation output does not fail the check.

```bash
bash projects/hipblaslt/tensilelite/Tensile/Tests/unit/mutation/slice-preflight.sh \
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

Back up the complete project file before changing the mutmut selection:

```bash
bash projects/hipblaslt/tensilelite/Tensile/Tests/unit/mutation/pyproject-mutmut.sh \
  backup --src projects/hipblaslt/tensilelite
```

Expected output:

```text
pyproject-mutmut: backup projects/hipblaslt/tensilelite/pyproject.toml -> work/mutation/pyproject.toml.bak
```

Set one or more source modules and pytest selections:

```bash
bash projects/hipblaslt/tensilelite/Tensile/Tests/unit/mutation/pyproject-mutmut.sh \
  set \
  --src projects/hipblaslt/tensilelite \
  --only-mutate Tensile/Common/Utilities.py \
  --test-selection Tensile/Tests/unit/characterization/CommonUtilities
```

Expected output:

```text
pyproject-mutmut: set rewrote only_mutate, pytest_add_cli_args_test_selection
pyproject-mutmut: set OK in projects/hipblaslt/tensilelite/pyproject.toml
```

Do not restore the backup yet. Run and inspect the slice first.

## 3. Run mutmut

Run the configured slice in the container. Always bound worker count; the
committed timeout multiplier assumes no more than 32 concurrent children.

```bash
docker exec \
  -w /work/projects/hipblaslt/tensilelite \
  tl-mut \
  mutmut run --max-children 32
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

Rerun one mutant after adding a focused test with:

```bash
docker exec \
  -w /work/projects/hipblaslt/tensilelite \
  tl-mut \
  mutmut run Tensile.Common.Utilities.x__mutmut_1 --max-children 1
```

## 4. Verify survivor-killing tests

`mutmut-verify.sh` changes tracked source while each mutant is active. Run only
one verifier at a time. The exit trap restores every manifest target, including
after interruption.

Create a tab-separated manifest. The header and column order are required:

```text
mutant_id	file	apply_method	test_node	expect_clean_rc	expect_mutant_rc_nonzero	revert_assert
Tensile.Common.Utilities.x__mutmut_1	Tensile/Common/Utilities.py	mutmut_apply	Tensile/Tests/unit/characterization/CommonUtilities/test_example.py::test_example	0	true	true
```

Then run the verifier:

```bash
bash projects/hipblaslt/tensilelite/Tensile/Tests/unit/mutation/mutmut-verify.sh \
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
CLEAN: no mutated-source leak.
RESULT: ALL KILLED
kill_matrix: work/mutation/verify/kill_matrix.tsv
```

The verifier writes:

- `kill_matrix.tsv`: one structured result row per manifest entry.
- `verify-report.txt`: a human-readable summary.

Verdict rules are intentionally strict:

- `KILLED`: the clean node returned its expected status, the mutated node
  returned pytest assertion status `1`, and restoration was clean.
- `BAD`: the clean baseline was wrong, the mutant survived, application failed,
  or restoration leaked changes.
- `INCONCLUSIVE`: pytest collection, usage, internal, interruption, or another
  non-assertion error occurred. Infrastructure errors are never counted as kills.

Any `BAD` or `INCONCLUSIVE` row makes the verifier exit non-zero.

## 5. Restore the mutation configuration

After the run and survivor verification, restore the byte-exact backup and
verify that `pyproject.toml` matches `HEAD`:

```bash
bash projects/hipblaslt/tensilelite/Tensile/Tests/unit/mutation/pyproject-mutmut.sh \
  restore --src projects/hipblaslt/tensilelite

bash projects/hipblaslt/tensilelite/Tensile/Tests/unit/mutation/pyproject-mutmut.sh \
  assert-clean --src projects/hipblaslt/tensilelite
```

Expected output:

```text
pyproject-mutmut: restore work/mutation/pyproject.toml.bak -> projects/hipblaslt/tensilelite/pyproject.toml
pyproject-mutmut: assert-clean OK (pyproject.toml == HEAD)
```
