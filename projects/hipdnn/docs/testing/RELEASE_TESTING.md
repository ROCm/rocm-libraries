# Release Testing

Owner: @ROCm/hipdnn-reviewers | Last Updated: 2026-10-21

The hipDNN team owns release testing: it runs the `full` tier, runs the benchmark corpus, and decides what to prioritize. Each item below is tagged **today** (automated), **manual** (the team runs it by hand) or **planned**.

| Item | Tag |
|---|---|
| Run the `full` tier; everything must verify | today / manual |
| Run the dnn-benchmarking corpus, prioritized by risk | manual |
| ASAN build | today (nightly) |
| ASAN tests | manual (weekly) |
| Exploratory fuzzy mode within a pre-release budget | planned |

## Full tier

`full` is the broadest tier: much wider cases and regression testing than the per-PR tiers. Tier definitions are in [TESTING.md](../TESTING.md#test-tiers); the commands to run one are in [Building and running tests](../TESTING.md#building-and-running-tests).

- **Today.** TheRock's prerelease flow selects `full`. That flow runs outside this repository.
- **Manual.** The team runs `full` manually ahead of release branching. Every test must verify. A test that cannot run is skipped through a tracked issue, not dropped (see [Flaky tests and skips](TESTING_STRATEGY.md#flaky-tests-and-skips)).

## Performance corpus (manual)

[dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking) holds the benchmark corpus. Before a release the team runs the full corpus. Runs are ordered by importance of the op and by risk, meaning the ops changed most since the last release go first. Approach: [Performance](TESTING_STRATEGY.md#performance). Automation direction: [Roadmap](../Roadmap.md#benchmarking--performance-testing-2).

## Sanitizers

Release sanitizer runs follow [Testing Strategy: Sanitizers](TESTING_STRATEGY.md#sanitizers). Commands: [TESTING.md](../TESTING.md#sanitizers-and-static-analysis).

## Exploratory fuzzy mode (planned)

An exploratory mode that searches for uncovered cases within an affordable pre-release budget. See [Roadmap: Testing](../Roadmap.md#testing-2).

## Recording a release run

[TestPlan.md](../feature_release/TestPlan.md) is the procedure for verifying one release build: confirm CI, then run the tests from a TheRock tarball and from a source build. [TestRunTemplate.md](../feature_release/TestRunTemplate.md) is the document to fill in while following it, pinning results to the build and source commit. The tier and benchmark runs above add to that procedure; they do not replace it.
