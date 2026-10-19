# TensileLite common tests: CPU build / GPU run split timing

Measured 2026-10-15 on a gfx942 box, one run per mode, under CI-like conditions.
Uses the `--build-only` / `--use-cache` modes added by PR #6188 (merged).

## Results

| Phase | Command mode | Wall time | Exit | Passed / failed / skipped |
|---|---|---|---|---|
| Build | `--build-only --gpu-targets gfx942 --artifact-dir D` | 1,220 s (20m 20s) | 1 | 133 / 1 / 1,508 |
| Run | `--use-cache --artifact-dir D` | 1,592 s (26m 32s) | 0 | 138 / 0 / 1,509 |
| Combined | default (what CI runs today) | 2,692 s (44m 52s) | 0 | 138 / 0 / 1,509 |

- Split total, back to back: 2,812 s (46m 52s), about 2m longer than combined.
- **GPU-node time: 1,592 s instead of 2,692 s, a saving of 1,100 s (18m 20s, about 41%).**
- The 1,220 s build moves to a CPU-only runner.
- Pipelining: with several arch builds queued on CPU runners, GPU runners only see the run phase.
- The build exit code is 1 because of a single config, see "Findings".
- Not included: artifact upload/download/extract and per-job setup.
  The artifacts total 409 MB compressed (123 `.tar.gz`, about 3.76 GB uncompressed, median 0.46 MB,
  largest `gemm/dtv` 76.7 MB, then `gemm/fp16_tn` 35.7 MB and `gemm/bufferLoadStore0` 31.1 MB).
  Transfer cost must be added when deciding the real saving.

## Conditions

Taken from TheRock's `fetch_test_configurations.py` / `pytest_runner.py` and the log of a failing
`tensilelite-common` job on PR #12151 (job 116559736872, gfx94X). No passing CI job was available to
get CI timings, so none are compared here.

CI (gfx94X): 1 GPU (`ROCR_VISIBLE_DEVICES`), `KUBE_CPU_REQUEST=25` so `OMP_NUM_THREADS` and
`OPENBLAS_NUM_THREADS` are 25, `pytest -v -m "(common) and not skip-gfx94X"`, `--numprocesses=4`
(`parallel_workers: 4` for `hw-common`), `--timeout=5400`, `--prebuilt-client=<client>`,
`--global-parameters=LibraryFormat='msgpack'`, `--tensile-options=--cxx-compiler,<rocm>/bin/amdclang++`.

Local reproduction: `taskset -c 0-24` (25 cores), `ROCR_VISIBLE_DEVICES=0`, `OMP_NUM_THREADS=25`,
`OPENBLAS_NUM_THREADS=25`, `-n 4`, `--timeout=5400`, marker `common and not skip-gfx94X`, the same
`--global-parameters` / `--tensile-options`. `~/.tensile` and `~/.tensilelite` were deleted before
each phase. Script: `/tmp/tl_split/run.sh`; logs in `/tmp/tl_split/`.
The machine has 96 cores and 8 GPUs and is shared, so there was contention (load average up to about
100 at times); treat the numbers as one noisy sample.

To emulate a CPU-only build node, the five GPU-gated tests in `Tests/common/flags/gfx942/`
were deselected in the build phase (see Findings).

## Findings

1. **CI marker uses the family name.** The CI marker is `not skip-gfx94X`, but `pytest.ini` registers
   exact-arch markers (`skip-gfx942`). It selects 1,647 tests instead of 387. The configs that declare
   `skip-gfx942` are still skipped at run time by `configMarks` using the detected arch, so results are
   the same, but the marker is not doing the filtering.
2. **Build phase needs a msgpack-capable client.** CI passes `LibraryFormat='msgpack'` with
   TheRock's client. The locally built `build_tmp` client was built with `HIPBLASLT_ENABLE_YAML=ON`
   and has no msgpack support: 5 tests failed with `Failed to load solution library`.
   The runs here use a client built with `-DHIPBLASLT_ENABLE_YAML=OFF`.
3. **`comm/gfx950/fused_a2a.yaml` fails the build phase** with `Your parameters resulted in 0 valid
   solutions`. It is a gfx950-only config with no `skip-gfx942` mark. In the combined phase it is skipped
   because `comm` tests need 2+ GPUs. With `--build-only --gpu-targets gfx942` it is built anyway and
   fails, so a CI build job for gfx942 would go red on it. It needs a `skip-gfx942` mark or a skip
   when no multi-GPU is possible.
4. **`Tests/common/flags/gfx942/` is collected in every mode.** `pytest_ignore_collect` filters only
   direct children of `common/`. `test_compile` runs on CPU; the other five tests are gated on
   `has_arch("gfx942")` and use the GPU, so they should skip on a CPU node and run in the run phase.
   `has_arch` uses `rocm_agent_enumerator`/`amdgpu-arch` and ignores `ROCR_VISIBLE_DEVICES`,
   so hiding the GPU that way does not skip them. Skipping on a real GPU-less node was not run.
5. **The artifact is a directory of per-config tarballs, not one zip.** CI will need to upload the
   directory per target, and gfx120X (gfx1200/gfx1201) needs one per target.
6. **Setup pitfalls.** `~/.tensile` and `~/.tensilelite` were read-only in the agent sandbox, which made
   early runs fail; the final runs were done unsandboxed.

## Caveats

- Single run per mode; shared, loaded machine. Build phase time scales with cores (25 here); a
  CPU runner with more cores will be faster.
- Run and combined phases include the five `flags/` GPU tests; the build phase does not.
- The run phase measures only GPU-side execution on an already-built artifact; it does not include
  extracting the 409 MB of artifacts (included in the 1,592 s, since `test_config_run` extracts each).
- CI timings were not available for comparison (the observed CI jobs failed early for an unrelated
  `PytestRemovedIn9Warning`).
