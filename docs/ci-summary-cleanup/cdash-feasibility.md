# Per-PR CDash-style CI digest — feasibility study & draft design

Status: draft for review. Author: CI tooling investigation, 2026-06-29.
Companion mockup: [`checks-mockup.html`](checks-mockup.html).
Draft implementation: [`.github/scripts/pr_ci_dashboard.py`](../../.github/scripts/pr_ci_dashboard.py).

## Goal

A table, one per PR, refreshed every time the PR pipelines run, modelled on
<https://open.cdash.org/index.php?project=CMake>. One row per build
(`Linux gfx94X-dcgpu / hipblaslt`, …); columns for revision, configure,
build, tests (Run/Fail/Pass), and timing. Every cell that can link to a log
links to the log — including the configure/build/test logs that today are
hard to find.

## Headline finding (corrects the premise)

The request framed the value as "linking config/build/test logs whose URLs
are buried in the Actions logs." That is only true for **test** logs. The
configure and build logs are **not** buried — their S3 URLs are fully
deterministic.

TheRock's `build_tools/_therock_utils/workflow_outputs.py`
(`WorkflowOutputRoot`) defines the canonical layout. For `rocm-libraries`
(repo `!= ROCm/TheRock`, so the *external* bucket + an `owner-repo/` prefix):

```
https://therock-ci-artifacts-external.s3.amazonaws.com/ROCm-rocm-libraries/<run_id>-<platform>/logs/<arch>/<file>
```

Verified against the live bucket (`s3 list-type=2`,
prefix `ROCm-rocm-libraries/`). Each completed run has, per component, per arch:

- `<Component>_configure.log`
- `<Component>_build.log`
- `<Component>_install.log`
- `index.html` (browsable per-arch index)
- `ninja_logs.tar.gz`

So a configure/build log deep link needs only three values, all available
from the GitHub API: **`run_id`, `platform`, `arch`**. No scraping.

The `<Component>` token is the CMake project name as TheRock spells it
(`hipBLASLt`, `rocBLAS`, `hipSPARSE`, …) — note the casing differs from the
`projects_to_test` tokens in job names (`hipblaslt`). A name-normalization map
is needed to link a row to its component log (see Open questions).

## What is NOT available

Test **results** are not uploaded anywhere. The `*_test_*.tar.zst` objects in
the bucket are the test *binaries* fetched by the test job, not results.
`therock-test-component.yml` runs the test and `tee`s output to
`./test_logs/test_output.log`, which is consumed only by `notify_teams.py` and
then discarded with the runner. The only durable trace of a test run is the
GHA job console log (retained ~90 days, reachable via the jobs API).

## Data sources

| Need | Source | Notes |
|---|---|---|
| Row identity (`Linux gfx94X-dcgpu / hipblaslt`) | jobs API `job.name` | already formatted: `Linux (hipblaslt \| gfx94X-dcgpu) / Build (gfx94X-dcgpu)` |
| Revision | PR head SHA | `pull_request.head.sha` |
| Pipeline link | jobs API `job.html_url` | per job |
| Configure/Build log links | computed S3 URL | verified deterministic |
| Configure/Build error/warning counts | regex over the S3 `*_build.log` | public objects, fetchable; medium cost |
| Test Run/Fail/Pass | **parse test job console log** (chosen) | jobs API → logs; format-fragile |
| Test log link | `job.html_url` | no S3 object today |
| Start / duration | jobs API `startedAt`/`completedAt` | per job |

## Chosen design (from review)

- **Surface:** sticky PR comment, one per PR, rewritten each run. Markdown
  table; cell status via emoji (🟢🟡🔴) because GitHub strips CSS from
  comments. Known-issue rows are listed, never hidden (per mockup rule).
- **Test column:** parse the test job's console log for Run/Fail/Pass. No
  cross-repo change required. Accept format fragility for now; the accurate
  path (uploading ctest/junit to S3 from TheRock's test-component workflow) is
  recorded as a follow-up.

### Trigger / placement

**Chosen (v1, shipped):** a standalone hipBLASLt-owned workflow,
`.github/workflows/hipblaslt-ci-digest.yml`, triggered on `workflow_run`
completion of `TheRock CI` and `hipBLASLt ASAN CI` (plus `workflow_dispatch`
for manual refresh). It checks out only the trusted base-branch digest script
and operates on the upstream run's head SHA, aggregating **all** CI runs for
that SHA so one comment reflects the whole PR. The comment refreshes as each
pipeline settles.

Why this over the shared summary job: `workflow_run` runs in the **base-repo
context with a write token even for fork PRs**, which resolves the comment-
permission gap below; and it keeps the digest in hipBLASLt's lane instead of
editing the shared `therock-ci.yml` (owned by CI infra). The earlier draft's
plan to append a step to `therock_ci_summary` is therefore superseded — it
could not post on fork PRs and would have modified a shared workflow.

`workflow_run` caveat: the workflow only fires once it is on the default
branch (`develop`); it cannot be exercised end-to-end from a feature branch.
Use `workflow_dispatch` (`--pr`) for a manual smoke test before then; the
script also runs locally with `--dry-run`.

### v1 scope (this PR)

- hipBLASLt-relevant rows/checks only (matrix builds that include hipBLASLt,
  the hipBLASLt test component, ASAN, and the always-required gates).
- The native per-check list is rendered but **collapsed by default**
  (`<details>`).
- **No** known-issue classification — every failure is reported truthfully;
  the headline is red if anything hipBLASLt-relevant failed.
- Test column links the job log and shows job-level pass/fail; per-test
  Run/Fail/Pass counts are deferred (see phasing).
- `Build time` column reports the build job's own duration, not
  build-start..last-test-finish (test jobs can queue for hours on GPU
  runners, which would make a span-based duration meaningless).

## Effort / phasing

1. **Phase 1 (ships now, no cross-repo change):** rows from the jobs API,
   computed configure/build log links, job-level test pass/fail + parsed
   Run/Fail/Pass, timing, sticky comment. This is the draft script.
2. **Phase 2:** configure/build error+warning counts by fetching and
   regexing the S3 logs (cache per run_id+arch to bound cost).
3. **Phase 3 (cross-repo, optional):** add a ctest/junit + test-log upload
   step to TheRock's `therock-test-component.yml`; switch the Test column to
   accurate counts with per-shard log deep links.

## Risks / open questions

- **Component↔log name mapping.** Job names use `projects_to_test` tokens
  (`hipblaslt`); S3 logs use CMake project names (`hipBLASLt`). A build job
  often covers several components (`rocblas,hipblaslt,tensilelite,hipblas`).
  The row's build cell should probably link the per-arch `index.html` and let
  the user pick the component log, OR expand one sub-row per component. Draft
  links `index.html`; per-component expansion is a TODO.
- **Test-log regex.** Needs validating against a real `test_output.log`
  (ctest summary vs pytest summary vs gtest). Draft ships a best-effort
  multi-pattern parser, off by default until a sample is confirmed.
- **Comment write permission.** *Resolved by the chosen trigger.* The
  `workflow_run`-triggered digest workflow runs in the base-repo context with
  `pull-requests: write`, so the sticky upsert works for branch and fork PRs
  alike. No PAT/App token needed for v1 (`github.token` suffices).
- **Cost.** Phase 2 log-fetching multiplies S3 GETs per refresh; gate behind
  a flag and cache.
