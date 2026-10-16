# Lane: author testing (`AUTH`)

Compare what the author tested against what the change requires.

## Method

1. **Claimed.** Extract every claim from the PR description. For a testing claim record command, suite, architecture, OS,
   configuration and result; flag claims with no command or arch. Report each unconfirmed claim as a finding; list each
   confirmed one under **Checked, no finding** with its evidence.
2. **Executed in CI.** For every check run at `$HEAD` that executes tests, record suite, arch, OS, and executed and skipped counts:
   `gh api --paginate "repos/<owner>/<repo>/commits/$HEAD/check-runs" --jq '.check_runs[] | [.name,.conclusion,.html_url] | @tsv'`;
   logs via `gh run view <run-id> -R <owner>/<repo> --log`. With no PR or no checks, record "no CI"; absent CI is a finding only
   where step 3 requires an executed run. Sources:
   - `.github/workflows/hipdnn-superbuild-ci.yml` (PRs touching `projects/hipdnn/**` or `dnn-providers/**`): ctest and pytest on one
     Linux and one Windows target. `GPU_TARGET` selects only the build target. The runners (`azure-*-scale-rocm`) have shown no GPU
     (`no ROCm-capable device is detected`), and ctest reports a self-skipping gtest as Passed. Credit an architecture only with a
     device line in the log or per-test output that is not `[  SKIPPED ]`.
   - `.github/workflows/therock-multi-arch-ci.yml`: read the setup job's `tests disabled (...)` lines; count a family only from
     executed test counts ([../policy.md](../policy.md) *Review rules*, green is not coverage).
3. **Required.** Classify the diff: arch-independent → standard PR CI; behavior-shifting → multi-arch; arch-scoped → exactly
   those archs; support-surface expanding → every supported family. Take arch scope from the code (gfx names in paths,
   descriptors, kernel tables, `#if` guards, CMake arch lists). Do not over-escalate: tooling- or docs-only changes need only the
   author's run of that tooling. hipDNN adds:
   - Provider op or support-surface change under `dnn-providers/**` → full multi-arch sweep, or only the op's archs.
   - Newly enabled integration suite → every supported family, not the routine-CI subset.
4. **Compare.** Every required (arch, suite) pair needs a claimed or CI execution; report each missing pair. Reconcile against
   the PR body's `## ASIC Coverage` (or `## Device / Architecture Coverage`) section: report it missing or narrower than step 3.
5. **M4, M5 and PR bot.** Resolve every tracker key and link in the description (`gh issue view`, `gh pr view`, HTTP HEAD). Read
   the `therock-pr-bot` check result and the policy under `tools/libraries_pr_bot/` at `$HEAD`. Give one line each:
   `M4: PASS|FAIL|WAIVED(<code>)|UNVERIFIED <evidence>`, `M5: ...`, `PR bot: pass|fail|absent <rule>`.

## Flags

- Touched gfx90a, gfx942 and gfx950 paths, tested only one: IMPORTANT, listing the rest.
- Testing section absent, or "tested locally" without a command: `process` finding.
- Checklist items marked passed with no log or CI run.
- Results from a head other than `$HEAD` when the delta touches the tested code.
- Performance claims without method, baseline and hardware.

Close each gap with the suite, arch and run that covers it. For a family PR CI does not test, that is a `workflow_dispatch` of
`therock-multi-arch-ci.yml` (inputs `linux_amdgpu_families`, `linux_test_labels`; it runs the `quick` tier, so check the tests
carry it) or the PR labels `ci:<family>` and `ci:test:<family>`, not a PR rerun. Take supported families from TheRock's
`build_tools/github_actions/amdgpu_family_matrix.py` at the pin, not the workflow's default build list.
