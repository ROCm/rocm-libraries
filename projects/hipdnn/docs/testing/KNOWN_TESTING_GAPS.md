# Known Testing Gaps

Owner: @ROCm/hipdnn-reviewers | Last Updated: 2026-10-22

[Back to Testing](../TESTING.md)

Gaps in hipDNN testing that the hipDNN team owns. Planned work is in the Roadmap: [carried over from Q3 2026](../Roadmap.md#testing) and [Q4 2026 and beyond](../Roadmap.md#testing-1).

## Hardware coverage

- **Pre-submit CI covers few GPUs.** Which GPUs PR, push and nightly CI test: [Test tiers](../TESTING.md#test-tiers). Other architectures, including gfx950, are not tested before merge. Run the tests on the architecture your change targets, and validate a provider change on the GPUs it supports ([Before you push](../TESTING.md#before-you-push)).
- **ASAN tests run manually.** PR ASAN jobs build without running tests ([what gates a pull request](../TESTING.md#what-gates-a-pull-request)). The team runs ASAN tests weekly on gfx942 ([Sanitizers](./TESTING_STRATEGY.md#sanitizers)); other architectures need a manual run.

## Test coverage

- **Per-op integration coverage is uneven.** Convolution and normalization have shape sweeps; attention has few cases per tier. Per-op test plans and bundles are on the Roadmap.
- **TSAN covers core only.** Providers have no TSAN option ([Sanitizers](./TESTING_STRATEGY.md#sanitizers)).

## Performance

- **Benchmarks run manually.** No CI job runs [dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking); the team runs it before a release ([Performance](./TESTING_STRATEGY.md#performance)). Automated runs are on the Roadmap.
