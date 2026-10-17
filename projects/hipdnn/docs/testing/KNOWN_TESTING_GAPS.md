# Known Testing Gaps

Owner: @ROCm/hipdnn-core | Last Updated: 2026-10-16

Gaps in hipDNN testing that the hipDNN team owns. Planned work is in the Roadmap: [Q3 2026](../Roadmap.md#testing), [Q4 2026](../Roadmap.md#testing-1) and [Q1 2027](../Roadmap.md#testing-2).

## Hardware coverage

- **Pre-submit CI covers few GPUs.** PR and push test only Linux gfx94X and Windows gfx110X; nightly adds Windows gfx1151. Other architectures, including gfx950, are not tested before merge. Run the tests on the architecture your change targets, and validate a provider change on the GPUs it supports ([Before you push](../TESTING.md#before-you-push)).
- **ASAN tests run manually.** PR ASAN jobs build without running tests. The team runs ASAN tests every week on gfx942; other architectures need a manual run.

## Test coverage

- **Per-op integration coverage is uneven.** Convolution and normalization have shape sweeps; attention has few cases per tier. Per-op test plans and bundles are on the Roadmap.
- **TSAN covers core only.** It runs locally on Linux; providers have no TSAN option.

## Performance

- **Benchmarks run manually.** No CI job runs [dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking); the team runs it before a release. Automated runs are on the Roadmap.
