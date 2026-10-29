# Paired YAML-split experiment

This opt-in experiment measures the test-scheduling effect of [PR #14597](https://github.com/ROCm/rocm-libraries/pull/14597). It lives on an experimental branch, not on the PR branch, and is not a merge gate. It does not measure kernel throughput or qualify the full YAML corpus.

The driver copies the exact YAML bytes from the unsplit tool commit (`4a113302d3e673472871ad7b1be75e2a350f3bd2`) and the generated-file commit (`7bdf24c878e065200086133ebb9ab1fed2bde949`). It checks that both versions contain the same benchmark groups and settings, then runs the existing pytest build/archive/extract/run wrappers. Missing, skipped, duplicate or failing tests invalidate a trial.

The native cohort is four complete strict-mode GEMM YAMLs: BF16, complex F32, F64 and Int8-to-Int32. Splitting produces nine files. Each version runs three times with one pytest worker and three times with four workers, alternating order within pairs. All trials share the same GPU, compiler, client and Python code; only the copied YAML files change. Code generation uses four CPU threads per worker, reference arithmetic uses one, and a GPU lock serializes client execution. Both versions use seed 14597, assembly-size verification, and print every solution verdict. The driver saves hashed case identities and verdict counts, excluding timings and hardware metrics, for an independent equality check. Each trial has fresh output; filesystem caches are not flushed. Dependency and client setup happen once outside the measurements.

Dispatch the existing workflow on the experimental branch:

```sh
gh workflow run gfx1250_hipblaslt_tensilelite_test.yml \
  --ref users/newling/pr14597-split-benchmark \
  -f yaml_split_benchmark=true -f run_hipblaslt=false \
  -f run_tensilelite=true -f send_report=false
```

Normal workflow behavior is unchanged unless `yaml_split_benchmark` is enabled. Native trial durations come from the Actions job-step timestamps; existing hardware-output redaction remains in place. Each trial is capped at ten minutes and the job at 150 minutes. Compare paired split/base ratios separately for each worker count; report every repetition, median and range, never just the fastest trial.

The emulator pilot uses the complete non-strict F64 YAML (one file versus four), the same wrappers, and independent per-worker client locks. Its local timing option must not be enabled in the native workflow. Emulator and native cohorts differ, so their absolute durations are not comparable. The pilot is too narrow to establish a corpus-wide speedup.
