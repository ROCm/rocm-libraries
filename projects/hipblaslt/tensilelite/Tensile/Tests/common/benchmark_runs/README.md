# Benchmark Runs

These tests exercise the benchmark timing code path (`SyncsPerBenchmark >= 1`),
which is disabled (`SyncsPerBenchmark: 0`) in all other test YAMLs to speed up
correctness-only CI runs.

## Architecture Coverage

| File                  | Architectures                          |
|-----------------------|----------------------------------------|
| `fp32_tn.yaml`        | gfx900, gfx906, gfx908, gfx90a,        |
|                       | gfx940, gfx941, gfx942, gfx950,        |
|                       | gfx1010, gfx1011, gfx1012, gfx1030     |
| `fp16_tn_gfx11.yaml`  | gfx1100, gfx1101, gfx1102              |
| `bf16_tn_gfx12.yaml`  | gfx1200, gfx1201                       |

These are trimmed copies of the full test files, kept small for fast execution.
