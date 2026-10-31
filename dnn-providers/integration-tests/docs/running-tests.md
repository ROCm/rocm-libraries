# Running the Integration Tests

This document shows how to run the cross-provider suite against an engine. It
also explains how tiers and filters decide what runs, and how to read the
output. It includes the ways a run can look green while it tests nothing. The
[index](README.md#terms) defines terms such as *lane*, *oracle* and *cell*.

- File formats named here are described in [File Formats](file-formats.md).
- To act on what a run reports (new bundles, stale claims), see
  [Adding Tests and Updating Claims](adding-tests.md).

Run the commands below from the repository root of a superbuild. The build
directory is `build/`. The suite's binaries are in `build/bin/`. The tests read
a copy of the bundle tree in `build/lib/integration-test-bundles/`
(see [The build reads a copy of the bundles](#the-build-reads-a-copy-of-the-bundles)).

## The binaries

| Binary | Tests | Engine loaded |
|---|---|---|
| `hipdnn_integration_tests` | The bundles (plus any C++ tests compiled in) against **one provider's engine** | yes |
| `hipdnn_golden_data_tests` | Our checked-in golden `.bin` data against the CPU/GPU reference executors | **no** |
| `hipdnn_gpu_ref_tests` | The GPU/CPU reference executors themselves | no |
| `hipdnn_integration_tests_unit_tests` | The harness (discovery, TOML, claims, verdicts) — no device needed | no |

`hipdnn_integration_tests` is built once in `dnn-providers/integration-tests/`
and run by **each provider** against its own plugin. That is how one bundle
tree validates every engine.

## Three ways to run it

### 1. CTest by tier (what CI does)

Each lane registers one CTest suite per tier from its provider's tier YAML
([format](file-formats.md#tier-yaml-files--test_categoriesyaml)). The suites
are in the **provider's** build directory:

```bash
ctest --test-dir build/dnn-providers/miopen-provider -L quick --output-on-failure
ctest --test-dir build/dnn-providers/miopen-provider -L quick -N          # list, do not run
ctest --test-dir build/dnn-providers/miopen-provider -R external-integration_quick -N -V  # print the exact command lines
```

From the superbuild root (`ctest --test-dir build`), this works only if you
configured the build with `-DROCM_LIBS_ENABLE_ROOT_CTEST=ON`. CI sets it. The
default is off. Without it, CTest finds nothing and exits 0.

Every lane's suites have the labels `integration_test`,
`external_integration_test`, `slow` and the engine name, plus the tier labels
from the YAML. If you use two `-L` flags, both must match:

```bash
ctest --test-dir build/dnn-providers/miopen-provider -L quick -L MIOPEN_ENGINE
```

Suite names are `<prefix>_<category>_suite`. The provider sets the prefix, for
example `miopen-provider-external-integration_quick_suite`. A YAML with
`exclude_gpu` entries adds per-arch variants named
`<prefix>_<category>_<arch>_suite`. See [layer 2](#what-decides-whether-a-test-runs).

`-L` is a regular expression matched against each label. It is not an exact
label. In a provider build directory, `-L quick` therefore also selects the
`ffm-quick` suites, the provider's own unit and integration suites (they have
tier labels too), and every per-arch variant. Some of the variants are
`DISABLED`. First list the suites with `-N`. To run exactly one lane's tier,
select the suite by name:

```bash
ctest --test-dir build/dnn-providers/miopen-provider -R '^miopen-provider-external-integration_quick_suite$' --output-on-failure
```

#### Seeing the output of a passing run

`--output-on-failure` prints the output of a test only if that test fails. A
green lane therefore shows neither the coverage summary nor the support-claim
summary. Also, `unclaimed_support` never fails a test. To read the summaries,
use one of these methods:

- Run with `-V` (`ctest ... -L quick -V`).
- Read `build/dnn-providers/<provider>/Testing/Temporary/LastTest.log` after the run.
- Use the check target or the binary directly (below). They always print the
  summaries.

### 2. The provider's check target (no tier filter)

```bash
cmake --build build --target miopen-provider-external-integration-check
```

This target runs the lane without a tier filter and always prints its output.
Use it to see the full picture of an engine. Do not use it to reproduce a tier.

| Provider | Target | Engine | Note |
|---|---|---|---|
| miopen-provider | `miopen-provider-external-integration-check` | `MIOPEN_ENGINE` | |
| hipblaslt-provider | `hipblaslt-provider-external-integration-check` | `HIPBLASLT_ENGINE` | runs with `--gtest_filter=*Matmul*` |
| hip-kernel-provider | `hip-kernel-provider-external-integration-check` | `HIP_MLOPS_ENGINE` | |
| hip-kernel-provider | `hip-kernel-provider-asm-sdpa-external-integration-check` | `ASM_SDPA_ENGINE` | |
| hip-kernel-provider | `hip_kernel_provider_asm_sdpa_gpu_ref_integration_tests` | `ASM_SDPA_ENGINE` | GPU reference, C++ SDPA tests only, no tiers |

A lane registered without a tier YAML, like the last row, is a single CTest
test with no tier labels. The authoritative list is the set of
`add_external_integration_test_target()` calls under `dnn-providers/`. To add
one, see [Add an engine lane](adding-tests.md#add-an-engine-lane).

### 3. The binary directly (debugging one case)

Pass the plugin, pin the engine, and pass the TOML of that engine. The CTest
lanes pass the same arguments. To get the exact plugin path and arguments, run
`ctest -N -V` on the suite of the lane (above).

```bash
build/bin/hipdnn_integration_tests \
    --test-article <path to the provider plugin, e.g. libmiopen_plugin.so> \
    --test-engine  MIOPEN_ENGINE \
    --test-config  dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml \
    --gtest_filter='quick_RMSNorm_Default.*'
```

- Always pass `--test-engine` when you test one provider. Without it, the
  normal engine selection of hipDNN picks the winner. A "pass" can come from a
  different engine, and the run produces no support-claim summary.
- To read the source tree instead of the build copy, add
  `--golden-data-dir dnn-providers/integration-tests/integration-test-bundles`.
  Use this right after you edit bundles or sidecars.
- On Windows, put the ROCm `bin` directory and `build/bin` on `PATH`. Without
  them, the binary stops with `STATUS_DLL_NOT_FOUND` (`0xc0000135`) before it
  runs anything.
- Standard GTest sharding (`GTEST_TOTAL_SHARDS` / `GTEST_SHARD_INDEX`) works.

### 4. Time the engine (`--benchmark`)

`--benchmark` adds a performance measurement to each test. The test first
verifies the engine output in the usual way. If the output passes, the harness
times the same engine on the same tensors with `Graph::autotune()`, limited to
that one engine. The C++ graph tests and the bundles both support it.

A test is not timed when it skips, fails, or passes without output
verification (a bundle at enforcement level APPLICABLE or BUILDABLE). Such a
test prints no `[ BENCHMARK ]` line. A filter that selects only such tests
gives a green run with no timings, so count the `[ BENCHMARK ]` lines.

```bash
build/bin/hipdnn_integration_tests \
    --test-article build/lib/hipdnn_plugins/engines/libmiopen_plugin.so \
    --test-engine  MIOPEN_ENGINE \
    --gtest_filter='quick_ConvolutionFwd_Default.*' \
    --benchmark \
    --gtest_output=json:benchmark.json
```

Each timed test prints one line:

```text
[ BENCHMARK ] quick_ConvolutionFwd_Default.8_16_16_16_bfp16_nhwc_dil1x1_postpad0x0_prepad0x0: engine=MIOPEN_ENGINE tuned=0 timing=DEVICE_ONLY samples=20 min=0.0181 ms robust=0.0184 ms mean=0.0184 ms stddev=0.0002 ms
```

- Autotune runs `--benchmark-warmup` untimed executions (default 30). Then it
  runs `--benchmark-iterations` timed executions (default 20), with the
  `FIXED_AVERAGE` strategy and the same stall gate, retry and statistics
  rules as any other autotune call.
- The GPU idles between tests, so each test starts cold. Too few warmup
  executions make the times read high: on MI210 the quick-tier bundles read
  4-60% slow with 1-10 warmups and settle after about 30. More timed
  iterations do not fix this; more warmup does. Raise `--benchmark-warmup`
  if `stddev` stays large compared with `mean`.
- `robust` is the mean after autotune drops the slow tail: what the engine
  usually costs. `min`, `mean` and `stddev` are over all timed samples.
- `tuned=0` means that the engine chose its solution with its heuristics. The
  harness sets no knobs, so this is the default. To time the solution that the
  engine picks by benchmarking its candidates, set
  `HIPDNN_FORCE_BENCHMARKING=1`. The line then shows `tuned=1` for engines
  that support the `global.benchmarking` knob (MIOpen, the hip-kernel
  ingestor). Engines without the knob, such as hipBLASLt, still show
  `tuned=0`. The selection runs during the verification execution, so its
  cost is not in the timed samples.
- `timing=DEVICE_ONLY` means that the stream was stalled, so the times exclude
  host submission overhead. `timing=UNSTALLED` means that autotune could not
  stall the stream and re-measured without the stall gate, so the times can
  include host time. Do not compare the two kinds.
- The test fails with `benchmark failed: …` when autotune cannot benchmark the
  engine, for example when an execution fails or too many timed executions
  return no measurement.
- `--gtest_output=json:<file>` or `xml:<file>` writes the same values as the
  test properties `benchmark_engine`, `benchmark_tuned`, `benchmark_timing`,
  `benchmark_samples`, `benchmark_min_ms`, `benchmark_robust_ms`,
  `benchmark_mean_ms` and `benchmark_stddev_ms`.
- `engine=` names the engine that ran. To compare engines, run once for each
  `--test-engine`. Without `--test-engine`, hipDNN selects the engine.
- To time your own graph, write it as a single-graph bundle
  (`<dir>/MyGraph/MyGraph.json`, [format](file-formats.md)). Then pass
  `--golden-data-dir <dir> --gtest_filter='MyGraph*'`. Without golden data, a
  reference executor verifies the output. To convert a C++ graph test into a
  bundle, use `--capture-bundles`.
- Verification runs before timing. For large shapes, the CPU reference can be
  slow. Use `--verification-mode gpu` when a GPU reference exists for the op.
- To see the kernels that the engine launched, run the same command after
  `rocprofv3 --kernel-trace --stats --output-format csv -d <dir> --`. In
  `<dir>/*/*_kernel_stats.csv`, an engine kernel of a single-kernel graph has
  1 + warmup + iterations calls. The other kernels fill the inputs or run the
  reference.

### The build reads a copy of the bundles

CMake copies `integration-test-bundles/` into `build/lib/integration-test-bundles/`
**at configure time only**. The lanes, check targets and
`hipdnn_golden_data_tests` read that copy. After you add or edit a bundle or
sidecar, or run `dvc pull`, re-run CMake before you run the tests.
Alternatively, pass `--golden-data-dir` with the path to the source tree.
Otherwise, you test the old tree.

## Command-line reference — `hipdnn_integration_tests`

| Flag | Env var | Default | Purpose |
|---|---|---|---|
| `--test-article`, `--ta` | | plugin discovery | Path to the engine plugin (`.so` / `.dll`). The binary loads only that plugin. |
| `--test-engine`, `--te` | | none | Pin the run to one engine. If the engine is not loaded, the binary exits 1 before any test: `Error: Engine '<name>' is not loaded. Check the plugin path.` |
| `--test-config`, `--tc` | | none | The engine's TOML ([format](file-formats.md#per-engine-test-config--configengine_nametoml)). A missing path or invalid TOML exits 1. |
| `--verification-mode`, `--vm` | `HIPDNN_TEST_VERIFICATION_MODE` | `auto` | How bundle output is checked; see [Verification modes](#verification-modes). |
| `--validator` | `HIPDNN_TEST_VALIDATOR` | `auto` | Where the comparison runs: `auto` (follow the reference), `cpu`, `gpu`. |
| `--reference-executor` | `HIPDNN_TEST_REFERENCE_EXECUTOR` | `cpu` | Reference for C++ graph tests only. |
| `--golden-data-dir`, `--gd` | `HIPDNN_TEST_GOLDEN_DATA_DIR` | `<exe>/../lib/integration-test-bundles/` | Bundle root. The path in the flag must exist. The binary does not check the path in the env var up front. |
| `--outcome-report <file>` | `HIPDNN_TEST_OUTCOME_REPORT` | none | Append one JSON object per line: a `started` record before each bundle test and a `finished` record when it ends. See [Outcome report](#outcome-report). The flag wins over the env var. The binary exits 1 if the file cannot be opened. |
| `--no-bundles` | `HIPDNN_TEST_ALLOW_BUNDLES` | bundles on | Register only the C++ tests compiled into the binary. If the env var is set, it takes priority over the flag in either direction. |
| `--enforce-support-claims[=true\|false]` | | on | Fail a test whose claim the engine breaks. See [Support-claim summary](#support-claim-summary). |
| `--write-support-claims` | | off | Authoring run: record which graphs the engine accepts into sidecars. See [Adding Tests](adding-tests.md#record-claims-with---write-support-claims). |
| `--skip-graph-validation` | | off | Pass as soon as the engine accepts the graph, without executing it or comparing outputs. A bundle is taken only to its applicability rung, whatever its `enforcement_level`, and **skips** when the engine declines. Needs `--test-engine` for bundles. Support claims are still enforced; a claim is left `accepted`, never `confirmed`, because nothing was verified. |
| `--fail-on-unsupported` | | off | Fail instead of skip when no engine supports a graph. **C++ graph tests only.** |
| `--generate-support-matrix[=file]` | | off | Write a markdown matrix. **C++ graph tests only**; on a default build it writes an empty table. |
| `--capture-bundles <dir>` | | off | Dump compiled-in C++ graph tests as bundle JSON (see [Adding Tests](adding-tests.md#from-an-existing-c-graph-test)). |
| `--benchmark` | | off | Time each verified test. See [Time the engine](#4-time-the-engine---benchmark). |
| `--benchmark-iterations` | | 20 | Timed executions per test. Requires `--benchmark`. |
| `--benchmark-warmup` | | 30 | Untimed executions before timing. Requires `--benchmark`. |
| `--gtest_*` | | | Passed through to GTest. |

The binary refuses these flag combinations (exit 1):

- `--enforce-support-claims` or `--enforce-support-claims=true` without
  `--test-engine`.
- `--write-support-claims` together with `--enforce-support-claims` or `=true`.
- `--write-support-claims` without `--test-article`, or without a bundle root
  (`--golden-data-dir` or `HIPDNN_TEST_GOLDEN_DATA_DIR`).
- `--benchmark-iterations` or `--benchmark-warmup` without `--benchmark`, or
  `--benchmark-iterations` less than 1, or `--benchmark-warmup` less than 0.
- `--benchmark` together with `--skip-graph-validation`, `--capture-bundles`
  or `--write-support-claims`. These modes do not execute the graph.

`--enforce-support-claims=false` is accepted in every combination. If
enforcement is only inherited and there is no `--test-engine`, the run
enforces nothing and prints no claim summary.

If the machine has no HIP device, the binary prints
`No HIP devices available; skipping …` and exits 0 immediately. It does this
before it reads any other flag.

Bundle inputs of at least 2^14 elements (float, half, bfloat16 and double) are
generated on the device with rocRAND. Smaller tensors, sub-byte types and fixed
fills use the host RNG. Device-filled values differ from the host fill's, so a
failure that you saw with them reproduces bit for bit only on the same path.
Set `HIPDNN_TEST_HOST_INPUT_FILL=1` to force the host fill for every input.
A build without rocRAND always uses the host fill.

## `hipdnn_golden_data_tests`

This binary recomputes every bundle that has golden data with a reference
executor. It compares the result with the checked-in `.bin` files. It uses no
plugin, no engine and no claims. It validates **our data**, not a provider.

```bash
build/bin/hipdnn_golden_data_tests                        # both references
build/bin/hipdnn_golden_data_tests --reference cpu        # host only, no GPU needed
build/bin/hipdnn_golden_data_tests --gtest_filter='quick_*'
```

| Flag | Default | Purpose |
|---|---|---|
| `--reference cpu\|gpu\|both` | `both` | Which reference suites to register (`…_CpuRef`, `…_GpuRef`). |
| `--golden-data-dir`, `--gd` | as above | Bundle root. |
| `--validator` | `auto` (host) | Where the comparison runs. `gpu` needs a device, even for the CPU suite. |
| `--test-config`, `--tc` | none | The binary accepts it, but the harness applies neither the skips nor the tolerances of the TOML. |

A registered test has no skip path in its body. The harness registers a test
only if the bundle has golden data **and** every node in its graph is in the
required-op set of that reference
(`src/harness/reference-validation/ReferenceOpCoverage.hpp`). Therefore, a
reference that cannot run a registered graph is a failure. Bundles outside the
set of a lane are absent from that lane. The harness prints the counts, and the
ops responsible, at registration. With both lanes selected (the default), a
bundle with golden data that neither lane registered a test for fails as
`<bundle>_Unvalidated`. A tree whose `.bin` files were never pulled registers
nothing and says so. It is registered once, not per provider.

## Outcome report

`--outcome-report <file>` (or `HIPDNN_TEST_OUTCOME_REPORT`) gives every bundle
test one outcome per engine, which the pass/fail tally cannot say. A skip can be a
clean decline or a test that never ran; the report tells them apart. The file is
JSON Lines, appended to, and flushed after every record, so a crash keeps every
earlier line. Each registered bundle test writes a `started` record before it runs
and exactly one `finished` record when it ends: a `SetUp` skip, a bundle that failed
to load (the synthetic failing test) and a test that threw each get one. A `started`
with no `finished` is the test the process died in.

```json
{"event":"started","test":"gauntlet_SdpaFwd_Dense.s913377","bundle":".../sweep.json#s913377"}
{"event":"finished","test":"gauntlet_SdpaFwd_Dense.s913377","bundle":".../sweep.json#s913377",
 "engine":"ASM_SDPA_ENGINE","arch":"gfx950","platform":"linux","purpose":"probe",
 "outcome":"DECLINED","stage":"query","depth":"not-reached","origin":"none","verifier":"none",
 "reason":"Engine ASM_SDPA_ENGINE does not support this graph (1 output tensor(s), 0 ranked engine(s))"}
```

`purpose` is the case's [`purpose`](file-formats.md#metadata-sidecar--namemetajson)
metadata and is absent when the case sets none. `stage` is where the test stopped:
`graph`, `claim`, `query`, `build`, `execute`, `compare`, `oracle` or `setup`
(`none` for a pass).

A case that sets `metadata.inputs`, `validator`, `placement` or `placement_uid` echoes them
into its `finished` record as `inputs`, `validator`, `placement` and `placement_uid`, and a
pass that verified a rejection adds `"expected_status":"BAD_PARAM"`. The fields are
described in [Finished-record fields](file-formats.md#finished-record-fields).

The `finished` record of a case that built a plan also carries `"kernels"`: the distinct
kernel names the engine reported while the case ran, sorted. The SDPA engines (ASM_SDPA,
HIP_FLASH2, `hipkernel:Gfx950AttentionDense`, `hipkernel:Gfx1151WmmaAttention[Decode]`) write
one `kernel: <engine> <kernel>` INFO line per built plan; HIP_MLOPS writes none. The harness
collects the lines between the case's `started` and `finished` records without
`HIPDNN_LOG_LEVEL` being set: it registers a log callback, raises the global log level to
`info` for the case and puts the level back at the `finished` record. A level set by
`HIPDNN_LOG_LEVEL` is restored exactly; with it at `warn` or above the backend's own console
sink, which that variable creates, prints the engines' INFO lines during the cases. `kernels` is
absent when the case reported none.

| Outcome | Meaning |
|---|---|
| `PASS` | The engine accepted the graph and its output matched an oracle. For a case with `placement: "under"` it means instead that the backend rejected the misaligned pointer with `BAD_PARAM` (the record carries `expected_status`); see [reading a placement probe](#reading-an-alignment-or-placement-probe). |
| `DECLINED` | The support query resolved and the engine is absent from the ranked list, or the engine declined later (`EngineNotApplicableError` while building or running). For a claimed graph the run also fails as `CLAIM_BROKEN`. |
| `WRONG` | Accepted, ran, and the output did not match the oracle. |
| `CRASH` | In the ranked list, then failed to build or execute, **including a late `NOT_SUPPORTED`** (accepting a graph is a promise). Also: a hang or a dead process, which only the driver can see. |
| `UNCLAIMED_PASS` | `PASS` on a graph whose sidecar does not claim this engine, arch and platform. The claim is too narrow. |
| `UNEXERCISED` | Nothing verified it: a skip rule or guard, the frontend rejected the graph, the support query did not resolve (unknown is never `DECLINED`), no oracle, an oracle or harness error, or a pass that still left a failure on the test record. Never green. |

The report adds no pass or fail rule: it records what the harness already decided.
A run with no `--outcome-report` writes nothing.

## Gauntlet fuzz runs

A fuzz run draws random legal graphs from a **space** declaration, runs them through
the unchanged `hipdnn_integration_tests` on one engine, and reports one outcome per
sample. It is for finding defects an integration test did not think of; it is **not**
a CTest lane and never runs in CI (RFC 0023). The driver is
`dnn-providers/integration-tests/gauntlet/run_gauntlet_fuzz.py` (Python 3, standard
library only).

```bash
# The lane's own arguments: `ctest -N -V -R '<lane>_quick_suite'` prints them.
python3 dnn-providers/integration-tests/gauntlet/run_gauntlet_fuzz.py \
    --binary build/bin/hipdnn_integration_tests \
    --test-article <path to the provider plugin> \
    --test-engine  HIP_MLOPS_ENGINE \
    --test-config  dnn-providers/hip-kernel-provider/config/HIP_MLOPS_ENGINE.toml \
    --filter '^RMSNorm/' --samples 64 --work-dir gauntlet-runs/hip_mlops
```

The driver prints the seed it used. Run it again with `--seed` to get the same
samples. Spaces live in `gauntlet/spaces/` (`<Op>[.<Name>].space.json`); with no
`--filter` every space runs, and a space whose graphs the engine does not accept
reports `DECLINED`, not a failure.

**Shipped spaces.** `--filter` matches the id in the first column.

| Space | Engine | Template (quick tier) | Draws |
|---|---|---|---|
| `Batchnorm/Default` | `HIP_MLOPS_ENGINE` | `Batchnorm/Default` | forward training, rank 4 or 5, NCHW/NHWC strides |
| `BatchnormBackward/Default` | `HIP_MLOPS_ENGINE` | `BatchnormBackward/Variant2` | backward with saved mean and inverse variance |
| `BatchnormInference/Default` | `HIP_MLOPS_ENGINE` | `BatchnormInference/Default` | forward inference, y in x's type or widened to float |
| `Layernorm/Default` | `HIP_MLOPS_ENGINE` | `Layernorm/Default` | forward training (y, saved mean and inverse variance) |
| `LayernormBackward/Default` | `HIP_MLOPS_ENGINE` | `LayernormBackward/Variant2` | backward with saved mean and inverse variance |
| `RMSNorm/Default` | `HIP_MLOPS_ENGINE` | `RMSNorm/Variant4` | forward inference with scale and bias |
| `RMSNormBackward/Default` | `HIP_MLOPS_ENGINE` | `RMSNormBackward/Default` | backward (dx, dscale, dbias; saved inverse RMS) |
| `ResampleFwd/Default` | `HIP_MLOPS_ENGINE` | `ResampleFwd/Variant2` | pooling forward, no index output |
| `SdpaFwd/Default` | `ASM_SDPA_ENGINE` | `SdpaFwd/Default` | SDPA forward, bfloat16, head size 128 or 192, dense or bottom-right causal; `stats` switches the log-sum-exp output on (`add_tensors` and `node_patches`), `window` draws none, top-left causal or a sliding window (`left_bound` through a node patch). The top-left and sliding rows and stats under a mask are decline probes on ASM_SDPA |
| `SdpaBackward/Default` | `ASM_SDPA_ENGINE` | `SdpaBackward/Default` | SDPA backward on gfx942, half or bfloat16, head size 64, 128 or 192, no/causal/sliding-window masks through `left_bound`/`right_bound`, scale set per sample; grouped-query heads and head size 64 are decline probes; cost ceiling 2**29 (measured) |
| `SdpaFwd/Gfx950Dense` | `hipkernel:Gfx950AttentionDense` | `SdpaFwd/Default` | SDPA forward, BSHD, head size 64 or 128 |
| `SdpaFwd/Gfx1151` | `hipkernel:Gfx1151WmmaAttention` | `SdpaFwd/Default` | SDPA forward, prefill kernel of the gfx1151 WMMA attention pack, **not run on a device**: half or bfloat16, the six shipped head-size pairs (64/64, 96/96, 128/128, 192/128, 256/128, 256/256) and a head size of 100 as decline probe, none, causal (both alignments) and sliding-window masks, `stats` (LSE) and an additive `bias` (FP32 or Q dtype, broadcast over batch, head or query) through `add_tensors` and `node_patches`; draws only what the dispatcher sends to prefill (`seqlen_q` above 64) |
| `SdpaFwd/Gfx1151Decode` | `hipkernel:Gfx1151WmmaAttentionDecode` | `SdpaFwd/Default` | SDPA forward, decode kernel of the same pack, **not run on a device**: as `SdpaFwd/Gfx1151` with the shipped decode pairs 64/64, 128/128 and 256/256, `seqlen_q` mostly up to 64 and `seqlen_kv` up to 8192; draws only what the dispatcher sends to decode |
| `SdpaFwd/Flash2` | `HIP_FLASH2_ENGINE` | `SdpaFwd/Default` | **not a shape space**: one fixed fp16 SDPA forward (BHSD, head size 128, seqlen 256), varying only `alignment` and `placement`; gfx942 builds with `-DENABLE_HIP_FLASH2_ENGINE=ON` only |
| `SdpaFwd/Flash2Region` | `HIP_FLASH2_ENGINE` | `SdpaFwd/Default` | SDPA forward over the FLASH2 region on gfx942: fp16, head size 64 or 128 (V and O the same), no mask or top-left causal, BSHD or BHSD, grouped-query heads, `seqlen_q` a multiple of 64 above the `useFlash2ForShape` crossover; two decline-probe slices just outside it (below the crossover, `seqlen_q` not a multiple of 64); no stats or mask tensor (the engine declines them) |
| `Matmul/Align` | none (probes MIOpen and hipBLASLt) | `Matmul/Default` | **not a shape space**: one fixed float batched matmul, varying only `alignment` and `placement` |

The engine column is informational (the space's `engine` key). A space's `description`
says what it draws, why each constraint exists and which tensor each axis touches.

**Per-engine recipes.** Every SDPA engine runs through the same driver; only the engine name, the
config, the build and the environment differ. `<build>` is the CMake build directory. All of them
live in the one `hip_kernel_provider` plugin, so `--test-article` is the same file for all of them
(relative to `<build>`; the CTest lanes pass `$<TARGET_FILE:hip_kernel_provider>`, see
`HipdnnIntegrationTestHelpers.cmake`). On Windows plugins sit under `bin/` instead of `lib/`
(`bin/hipdnn_plugins/engines/hip_kernel_provider.dll`).

| Engine | Spaces | `--test-article` | `--test-engine` | `--test-config` | Required build flags | Arch | Extra env |
|---|---|---|---|---|---|---|---|
| ASM_SDPA forward | `SdpaFwd/Default` | `lib/hipdnn_plugins/engines/libhip_kernel_provider.so` | `ASM_SDPA_ENGINE` | `dnn-providers/hip-kernel-provider/config/ASM_SDPA_ENGINE.toml` | `-DHIPDNN_ENABLE_SDPA=ON` (`ENABLE_ASM_SDPA_ENGINE` is ON by default) | gfx942, gfx950 | none |
| ASM_SDPA backward | `SdpaBackward/Default` | same | `ASM_SDPA_ENGINE` | same | same | gfx942 (the region the space draws; the engine also ships gfx950 backward kernels) | none |
| HIP_FLASH2 | `SdpaFwd/Flash2Region` (the shape space), `SdpaFwd/Flash2` (alignment probe) | same | `HIP_FLASH2_ENGINE` | `dnn-providers/hip-kernel-provider/src/engines/hip_flash2_engine/config/HIP_FLASH2_ENGINE.toml` | `-DENABLE_HIP_FLASH2_ENGINE=ON -DHIPDNN_ENABLE_SDPA=ON` | gfx942 (fp16; the engine declines gfx950) | `HIP_FLASH2_KERNEL_DIR=<build>/dnn-providers/hip-kernel-provider/src/hip_flash2_kernels`, the directory CMake copies the shipped `.co` files to (its path is the cache variable `HIP_FLASH2_KERNEL_DIR_BUILD`). No CTest sets it, and without it every case fails in `hipModuleLoad` |
| `hipkernel:Gfx950AttentionDense` | `SdpaFwd/Gfx950Dense` | same | `hipkernel:Gfx950AttentionDense` | `dnn-providers/hip-kernel-provider/config/hipkernel_Gfx950AttentionDense.toml` | `-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPKERNELPROVIDER_ENABLE_ROCKE=ON -DHIPDNN_ENABLE_SDPA=ON -DGPU_TARGETS=gfx950`, and the rocKE descriptors pulled (`dvc pull -r ingestor`; add `-DHIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS=ON` to fail configure on a missing one) | gfx950 | none |
| `hipkernel:Gfx1151WmmaAttention` (prefill) | `SdpaFwd/Gfx1151` (no device run yet) | same | `hipkernel:Gfx1151WmmaAttention` | `dnn-providers/hip-kernel-provider/config/hipkernel_Gfx1151WmmaAttention.toml` | as Gfx950, with `-DGPU_TARGETS=gfx1151` | gfx1151 (the descriptors list gfx1100 to gfx1153 and gfx1200/gfx1201; the space's prefill route is the gfx11 rule at 32 to 40 compute units) | none |
| `hipkernel:Gfx1151WmmaAttentionDecode` | `SdpaFwd/Gfx1151Decode` (no device run yet) | same | `hipkernel:Gfx1151WmmaAttentionDecode` | `dnn-providers/hip-kernel-provider/config/hipkernel_Gfx1151WmmaAttentionDecode.toml` | as Gfx950, with `-DGPU_TARGETS=gfx1151` | gfx1151 (as above) | none |

Only `ASM_SDPA_ENGINE`, `HIP_MLOPS_ENGINE` and `hipkernel:Gfx950AttentionDense` have a CTest lane
(`hip-kernel-provider/src/CMakeLists.txt`); HIP_FLASH2 and the two Gfx1151 engines have none, so their
TOMLs reach a run only through an explicit `--test-config`. The two Gfx1151 TOMLs name no tolerance
override or skip: nothing has been run on a gfx1151 device, and neither of the two Gfx1151
spaces has run: they are checked only by the pytest load, sample and render tests, and every claim in their
`description` is read from `Gfx1151WmmaAttentionNative.cpp` and the shipped descriptors. A fuzz run on
FLASH2 therefore reads:

```bash
HIP_FLASH2_KERNEL_DIR=build/dnn-providers/hip-kernel-provider/src/hip_flash2_kernels \
python3 dnn-providers/integration-tests/gauntlet/run_gauntlet_fuzz.py \
    --binary build/bin/hipdnn_integration_tests \
    --test-article build/lib/hipdnn_plugins/engines/libhip_kernel_provider.so \
    --test-engine  HIP_FLASH2_ENGINE \
    --test-config  dnn-providers/hip-kernel-provider/src/engines/hip_flash2_engine/config/HIP_FLASH2_ENGINE.toml \
    --filter '^SdpaFwd/Flash2(Region)?$' --samples 64 --work-dir gauntlet-runs/flash2
```

**Input-value, alignment and placement axes.** Every space except `Matmul/Align` and `SdpaFwd/Flash2` has a
`domain` axis, and every space has an `alignment` and a `placement` axis. They render
case metadata and `tensor_patches` rather than shape values:

| Axis | Values | Renders |
|---|---|---|
| `domain` | `default`, `narrow`, `wide`, `positive`, `constant`, `nonzero`, `zeros`, `large`, `subnormal`, `nan`, `inf` | The [`metadata.inputs`](file-formats.md#inputs--fill-recipes) recipe of the space's domain tensor(s), below. `nan` and `inf` also set `validator: "match_nonfinite"`. |
| `alignment` | `1`, `2`, `4`, `8`, `16`, `32`, `64`, `128`, `256` | A `tensor_patches` entry per device tensor that sets its declared `alignment` (bytes). Host scalars such as epsilon keep the template's value. |
| `placement` | `default`, `exact`, `under` | `metadata.placement`, with `placement_uid` set to one fixed tensor. `under` is rendered as `exact` when the drawn `alignment` is 1, because nothing is below an alignment of 1; the summary still counts the draw as `under`. |

| `domain` | Recipe of the domain tensor |
|---|---|
| `default` | `free` over `[-1, 1]` |
| `narrow` | `free` over `[-0.01, 0.01]` |
| `wide` | `free` over `[-8, 8]` |
| `positive` | `free` over `[0.25, 4]` |
| `constant` | `fixed` at `0.5` |
| `nonzero` | `nonzero` over `[-1, 1]` |
| `zeros` | `zeros` |
| `large` | `large` with `scale` 32 (8 in the SDPA spaces, 4 in `LayernormBackward` and `RMSNormBackward`) |
| `subnormal` | `subnormal` |
| `nan`, `inf` | `special` (`"nan"` or `"inf"`), `fraction` 0.05, planted on tensors chosen so a non-finite element stays local to a few outputs (the space's `description` names them); the case grades with `match_nonfinite` |

A domain a data type cannot hold (`subnormal` on an integer tensor, `inf` on fp8) is
`UNEXERCISED`, not a failure. In the `SdpaFwd` spaces a non-finite V element reaches a whole
output column, so most `nan`/`inf` samples with a long kv sequence produce an all
non-finite reference, which `match_nonfinite` fails by design: expect `WRONG` there
without an engine defect.

`SdpaBackward/Default` draws neither `nan` nor `inf`: one non-finite element of dO makes the
softmax-backward term non-finite for its whole row, so `dK` has no finite element and
`match_nonfinite` fails the reference by design, whichever operand is poisoned (the space's
`description` gives the citations). Its `wide` and `large` domains widen Q and K only, and its
`positive` domain keeps V and dO at `[0.25, 1]`; those choices are provisional until an on-device
run checks the engine tolerances against them.

**`late` axes.** `RMSNorm`, `SdpaFwd/Default` and `SdpaFwd/Gfx950Dense` declare their
`domain`, `alignment` and `placement` axes with `"late": true`; the other spaces declare
them as ordinary axes. A space draws all its axes from one generator seeded with the shape
seed, and a draw rejected by a constraint is retried from that generator. Adding an axis
there would move every later draw, retries included, so `--seed` would no longer reproduce
the shapes of earlier runs. A `late` axis draws from a stream of its own (shape seed,
attempt and axis name), so the earlier axes keep their draws: the same seed gives the
same shape as before, and only the new axes get values. Late axes come last. For the same
reason a feature axis added to an old space must not be read by a constraint: a constraint
that a late axis can violate would reject the draw and move the earlier axes. `SdpaFwd/Default`
gained `stats`, `window`, `wl` and `wr` this way, and its derive chain keeps every window legal
instead of rejecting one.

Before the first batch the driver lists the binary's tests on an empty bundle root with
the run's own flags. A healthy binary lists its compiled-in tests (exit 0) or reports
`Error: zero tests ran.` (exit 1), which also proves it started. If the listing fails any
other way (a bad `--test-engine` or `--test-article`, a missing config, a crash at
startup) or the binary reports that it found no HIP device, the driver prints the log
tail and exits 2 without running a sample, instead of reporting every sample as a load
crash.

| Variable | Flag | Meaning | Default |
|---|---|---|---|
| `HIPDNN_FUZZ_SAMPLES` | `--samples` | Samples per space. `0` is unlimited. | 32 |
| `HIPDNN_FUZZ_SEED` | `--seed` | Shape seed of the first sample. Sample *i* uses `seed + i`. | random, printed |
| `HIPDNN_FUZZ_DATA_SEED` | `--data-seed` | Input-data seed (the case's `metadata.seed`) for every sample. | derived from each shape seed |
| `HIPDNN_FUZZ_FILTER` | `--filter` | Regex on `<Op>/<Space>`. | every space |
| `HIPDNN_FUZZ_TIME_BUDGET` | `--time-budget` | Wall-clock limit, e.g. `30m`, `4h`. | none |
| `HIPDNN_FUZZ_CASE_TIMEOUT` | `--case-timeout` | Longest one case may run before it counts as a hang (the driver kills the binary: `CRASH`). | `10m` |
| `HIPDNN_FUZZ_SHRINK` | `--shrink` | Candidate runs to spend shrinking each `WRONG` or `CRASH` sample (see [Shrinking](#gauntlet-fuzz-runs)). `0` is off. Applies to `--row` runs too. | `0` |

The binary, `--test-article`, `--test-engine` and `--test-config` are flags of the
driver and are passed through; `--binary-arg` adds any other binary flag. Other flags:
`--work-dir`, `--batch-size` (samples per binary invocation), `--spaces-dir` and
`--summary`; `--pairwise`, `--pairwise-repair` and `--generate-only` are described under
[Pairwise runs](#pairwise-runs---pairwise).

**Continuous mode.** With `--time-budget` and no `--samples`, or with `--samples 0`,
nothing is pre-built: each batch draws fresh random samples (new seeds, round-robin
over the spaces) until the budget ends, a `--samples` cap is hit, or you press Ctrl-C.
The current batch finishes and the report is still written. A crash is logged and the
run continues.

**One explicit sample (`--row`).** `--row <file>` runs exactly one sample whose axis values
you give, instead of drawing any. It replays a finding at a modified axis, runs a size
ladder of chosen shapes, and is what a shrunk finding's `replay.sh` calls. The file is
JSON:

```json
{"space": "RMSNorm/Default",
 "axes": {"dtype": "bfloat16", "x_dtype": "bfloat16", "layout": "nhwc", "ndim": 4, "n": 2,
          "c": 115, "d3": 1, "h": 2, "w": 256, "k": 2, "domain": "constant",
          "alignment": 128, "placement": "exact"},
 "data_seed": 913377, "case_id": "s913377"}
```

`axes` names every axis of the space, `late` ones included (a late axis is just a value
here), and nothing else; each value must be one the axis can draw (a listed value, or an
integer inside its `range`; the `axis_values` of a failure's `repro.json` are such a set). `data_seed` becomes the case's `metadata.seed`. `case_id` is
optional, letters, digits and `_` only, and defaults to `row`. It is the case's id and so the
last part of the GTest name: give the id of the drawn sample (`s913377`) to get the name
that run had. The driver derives the other values, checks the space's constraints again and
renders the sample as a drawn one is rendered. A row that is not a legal sample (a missing
or unknown axis, a value the axis cannot draw, a violated constraint) or a file that does not
parse is a usage error: exit status 2, the reason on stderr, and nothing written.

The case goes to the same layout as a drawn batch of one,
`bundles/gauntlet/<Op>/<Space>/<tag>_cases/`, so the GTest name
(`gauntlet_<Op>_<Space>_<tag>_cases.<case_id>`) and the engine TOML tolerances and skip
globs that match it are those of a drawn sample with that id. The outcome, `outcomes.jsonl`,
`summary.json` (with `"row"` holding the file's content and `"seed": null`) and the
failure folder are written as in any run, with the axes in `repro.json` and `replay.txt`,
and `replay.sh` replays with `--row failures/<case>/row.json`. `--row` cannot be combined
with `--samples`, `--seed`, `--data-seed`, `--filter` or `--time-budget` (a usage error); the
`HIPDNN_FUZZ_*` variables for them are ignored. The binary flags, `--spaces-dir`,
`--work-dir` and `--case-timeout` apply as usual.

**Shrinking (`--shrink N`).** A random sample that fails is rarely the smallest one that does.
With `--shrink N` the driver, after the batch a `WRONG` or `CRASH` sample was found in (once its
failure folder is written), spends at most `N` candidate runs on reducing it. A candidate is the
sample with one axis moved toward a simpler value, the other axes kept:

- a `range` axis is binary-searched toward its low end, its declared `boundaries` below the
  current value tried first;
- a `values` axis tries the entries before the current one, first entry first (the list order
  of a space is "simpler first").

The data seed and the case id stay, so a candidate has the finding's GTest name (a changed
`dtype` changes the suite's `<tag>_cases` part, exactly as it would for a drawn sample) and the
engine TOML grades it as it graded the finding. A candidate that violates one of the space's
constraints is skipped and costs no budget. The axes are visited in the space's order, again and
again, until a whole pass moves nothing or the budget is spent. A candidate is **accepted only if
its outcome class equals the finding's**: `WRONG` stays `WRONG`, `CRASH` (a hang included) stays
`CRASH`; a candidate that passes, is declined, or turns one failure into the other is refused.

Each candidate runs as a one-case batch under `<work-dir>/shrink/<case>/` (its own bundles, logs
and reports). It never enters `outcomes.jsonl`, `samples.jsonl` or the summary counts, and leaves
no failure folder. Shrinking is not cut by `--time-budget` (a long budget can overrun by
`N` runs per finding; a hang costs `--case-timeout` each); Ctrl-C ends it early and keeps the best
sample so far.

The result goes to `failures/<case>/shrunk/`: `sweep_row.json` (the shrunk sample in the
`--row` file format), `axes.json` (original and shrunk values, the axes changed, runs spent and
steps accepted), the shrunk graph and its metadata (`<case>.json`, `<case>.meta.json`, written as
in the failure folder), `outcome.json` (the last accepted candidate's outcome record) and
`replay.sh`. `replay.sh` calls the driver with `--row shrunk/sweep_row.json` and the same binary,
article, engine, config and `HIPDNN_*` environment, so it runs under the shrink run's own GTest
name and TOML grading. `summary.json` gives each finding `"shrunk": {"runs", "accepted_steps",
"axes_changed"}` and the console report prints one `shrunk:` line per shrunk finding. When no
simpler value kept the outcome, `shrunk/` holds the original sample and `axes_changed` is empty.

**Every sample gets one outcome.**

| What happened | Outcome |
|---|---|
| A `finished` record | Its outcome. |
| The binary died with a case `started` and not `finished` | `CRASH`; the rest of the batch runs again. |
| A case ran past `--case-timeout` | `CRASH` (hang); the rest runs again. |
| A case written but absent from `--gtest_list_tests` | `UNEXERCISED` (load), with the load message. |
| The binary died before any case started | The batch is halved and rerun until the registration-killing case is isolated: `UNEXERCISED` (load crash). |

**Outputs** under `--work-dir`: `bundles/` (what the binary was given), `outcomes.jsonl`,
`samples.jsonl`, `summary.json` (counts per outcome for each engine and space, decline
and unexercised reasons, findings), `reports/` and `logs/` (one per binary invocation),
`failures/<case>/` for each `WRONG` and `CRASH` (with `shrunk/` under `--shrink`),
`shrink/<case>/` (the candidate runs of a shrink) and, for a [pairwise run](#pairwise-runs---pairwise),
`pairwise/<Op>_<Space>.json`. The exit status is non-zero if any
sample is `WRONG` or `CRASH`; `UNEXERCISED` samples are listed with their reasons and
never count as green. A run whose samples are all `DECLINED` or `UNEXERCISED` has not
tested the engine.

**Sweep directories.** The driver writes one sweep per input data type, under
`bundles/gauntlet/<Op>/<Space>/<tag>_cases/`, where `<tag>` is `fp32`, `fp16` or `bfp16`
(a sample's `x_dtype`, or its `dtype` when the space has no `x_dtype`; a space with neither
axis, such as `Matmul/Align`, has no `<tag>_cases` level). The tag is in the directory,
and so in the GTest suite name (`gauntlet_RMSNorm_Default_fp16_cases.s123`), because
an engine's [TOML](file-formats.md#per-engine-test-config--configengine_nametoml)
tolerance and skip globs match the full test name and the quick bundles carry the dtype
tag in the case id (`*fp16_*`), while a gauntlet case id is only `s<seed>`.

**Failure artifacts.** `failures/<case>/` is written before the run continues and holds
the expanded graph (`<case>.json`, tensor UIDs in canonical order), its metadata
(`<case>.meta.json`, `purpose: regression`), `repro.json` (space, hashes of the space
and template, shape seed and axis values, data seed, host or device fill, engine, arch,
ROCm and git versions, flags and `HIPDNN_*` environment), the outcome record, the
binary's log tail, and `replay.sh`: one command that regenerates the same graph and the
same input tensors (for a [`--row`](#gauntlet-fuzz-runs) run, `row.json` holds the row it
replays from). Device-filled inputs differ from the host fill's, so a replay uses
the fill path of the original run. To turn a finding into a checked-in case, see
[Freeze a fuzz finding](adding-tests.md#freeze-a-fuzz-finding).

**`summary.json` `axis_outcomes`.** For every axis of a space that lists `values` (the
`alignment`, `placement` and `domain` axes, and also the feature axes such as `dtype`, `layout`,
`mask`, `stats`, `window`, `d`) the summary counts outcomes per drawn value:
`axis_outcomes[<space id>][<axis>][<value as a string>]` is an object with a count for each of
`PASS`, `DECLINED`, `WRONG`, `CRASH`, `UNCLAIMED_PASS` and `UNEXERCISED`. The value is the
one drawn, so an `under` that was rendered as `exact` at alignment 1 counts under `under`.
`range` axes (sizes such as `sq`, `skv`, `n`) have no entry: they draw many distinct integers.
Read the `PASS` count of each feature value to see that a feature is exercised and passes, not
only that the run is green. A space has an entry for the axes it declares and no others.

**`summary.json` kernels.** Each engine entry under `engines` has `kernels`, a map from kernel
name to the number of cases that ran it (a case that built more than one plan counts each
distinct name once), and `kernel_unreported`: the cases that ran the engine's plan and whose
`finished` record names no kernel. A case ran the plan when it is `PASS`, `UNCLAIMED_PASS` or
`WRONG`, or `CRASH`/`UNEXERCISED` at stage `execute` or `compare`. `DECLINED` cases, cases that
stopped before a plan existed (stage `graph`, `claim`, `query`, `build`, `oracle`, `setup`) and
the `CRASH` results the driver wrote itself for a hang or a dead process are in neither count.
An engine that does not log a kernel (HIP_MLOPS) shows all its executed cases as
`kernel_unreported`; a non-zero `kernel_unreported` for ASM_SDPA, HIP_FLASH2 or a
`hipkernel:Gfx*Attention*` engine means the engine's `kernel:` line is missing. The record of
each case in `outcomes.jsonl` carries its `kernels`, so a feature value's PASS count (see
`axis_outcomes`) can be read against the kernel that ran.

### Pairwise runs (`--pairwise`)

A random run meets some pairs of axis values and misses others, and says nothing about which. A
**pairwise** run runs, per space, a small covering set in which **every value of every axis meets
every value of every other axis** in at least one legal row (RFC 0023 section 6). A **pair**, or
tuple, is one value of one axis together with one value of another.

```bash
# run the covering sets of two SDPA forward spaces (the seed is the generator's; printed if random)
python3 dnn-providers/integration-tests/gauntlet/run_gauntlet_fuzz.py \
    --binary build/bin/hipdnn_integration_tests --test-article <plugin> \
    --test-engine ASM_SDPA_ENGINE --test-config <its TOML> \
    --pairwise --seed 7 --filter '^SdpaFwd/Default$' --work-dir gauntlet-runs/pw_fwd
# write the same rows as sweeps and reports, running nothing (no binary needed)
python3 .../run_gauntlet_fuzz.py --pairwise --generate-only --seed 7 \
    --filter '^SdpaFwd/' --work-dir gauntlet-runs/pw_sweeps
# the pair coverage of an earlier random run, by the same levels
python3 .../run_gauntlet_fuzz.py --coverage-of gauntlet-runs/random \
    --levels-from gauntlet-runs/pw_fwd/pairwise --first 190
```

| Flag | Meaning |
|---|---|
| `--pairwise` | Run every row of each selected space's covering set instead of drawing samples. `--seed` is the generator's seed (random and printed if absent; the same seed gives the same rows); `--data-seed` fixes the data seed of every row. `--samples` and `--row` are usage errors with it (`HIPDNN_FUZZ_SAMPLES` is ignored), and `--time-budget` still ends the run. |
| `--pairwise-repair N` | After the rows, up to `N` rounds of repair rows (default 2; `0` is off), see below. |
| `--generate-only` | With `--pairwise`: write each space's full row set and its report, run no binary. `--binary` and the engine flags are not needed. |
| `--coverage-of DIR` `--levels-from PATH` `--first N` | Offline coverage of a finished run, see below. |

**Levels.** The driver cannot meet every integer of a `range` axis. A `values` axis has its values
(duplicates, such as the three `none` entries of a weighted list, count once). A `range` axis has
its declared `boundaries` inside the range plus **one interior value**, drawn from the seed
strictly between the lowest and the highest boundary and equal to none of them. The ends of the
range are levels only if they are declared boundaries; a range with no room has no interior level.
Other interior values stay with random runs. `late` axes are levels like any other.

**Live values.** A value that changes nothing is not met: `wl` while `window` is `none`,
`heads_d128` while `d` is 64, `under` placement at alignment 1 (it renders as `exact`) and the like.
A value is **live** in a row when some other level of its axis gives a legal row whose rendered
case differs (the graph values, tensor patches, added tensors, node patches, input recipes,
validator and placement; the id, notes and seeds do not count). A pair counts as covered by a row
only if both values are live in it.

**Pairs no row covers are proven, not dropped.** Each one is reported as

| Kind | Meaning |
|---|---|
| `illegal` | no legal row holds both values (a constraint forbids the pair) |
| `dead` | legal rows hold both, but in each of them at least one of the two is inert (the pair can never be met live) |
| `unproven` | the exhaustive search for a covering row passed its cap before it decided; the pair is never counted as covered |

with the reason in the report. `unproven` is empty for the four SDPA forward spaces
(`SdpaFwd/Default`, `SdpaFwd/Flash2Region`, `SdpaFwd/Gfx950Dense`, `SdpaFwd/Flash2`), where the
generator's tests prove every `illegal` and `dead` pair by an independent search and show that
every other pair is covered. The generator runs on every space, but the other spaces are not
tested for completeness: a space with many axes tied together by constraints (the two Gfx1151
spaces, which are not run on a device either) may report `unproven` pairs.

**Rows.** A row is an explicit row: id `pw<k>` (`k` counts from 0 over the main rows and the repair
rows of the space), `metadata.generator` `gauntlet-pairwise`, the pairwise seed in `notes`
([`file-formats.md`](file-formats.md), metadata sidecar), data seed derived from (space, seed,
`k`) unless `--data-seed` is given. Batches, crash isolation, the failure folders, `--shrink` and
the summary work as for any run. A failure folder has `row.json` and a `replay.sh` that calls
`--row`, which runs the row under the same GTest name; `repro.json` carries the row and no shape
seed.

**Repair (`--pairwise-repair N`).** An engine declines many rows of a space (an unsupported
window, head size or alignment), and a pair that only declining rows drew has tested nothing.
After the main rows, each round takes the **targets**: the pairs every row that drew them ended
`DECLINED` or `UNEXERCISED`. It builds new rows for them, from the construction and liveness of
the generator, using **only the levels that passed**: per axis, the values held by some `PASS` or
`UNCLAIMED_PASS` row of the run. A target with a value outside them is not retried (the report
lists it as `never_passed`); a target no row of those levels can cover live is `unreachable` (listed
with the reason, not tried again). A target an earlier round retried that still did not pass is
given a row of its own, and the other rows avoid holding it, so a pair that always declines does
not drag the others down with it. Repair runs up to `N` rounds, and stops when a round builds no
row. The repair rows are `pw<k>` after the main rows and run through the same batches.

**Report.** `<work-dir>/pairwise/<Op>_<Space>.json`: `levels` (axis to level values),
`tuples_total`, `rows` (the number of main rows), `lower_bound` (the largest product of two axes'
level counts: no covering set is smaller), `illegal`, `dead` and `unproven` (each a list of
`{"pair": [[axisA, valueA], [axisB, valueB]], "detail"}`), `repair` (`max_rounds`, `rows`, one entry
per round with `targets`, `retried`, `never_passed`, `unreachable`, `rows`, `first_row`, and the
`never_passed` pairs of the last round) and `timings` of the generator (`total_s`,
`exhaustive_s`, `exhaustive_searches`, `renders`). It is written when the rows are built and
again at the end.

**`summary.json`.** A pairwise run has `"mode": "pairwise"` (`"random"` and `"row"` otherwise),
`seed` (the generator seed), no `samples_per_space`, `"continuous": false`, and
`stopped_because` `pairwise-complete` when every row and repair round ran (or `time-budget`,
`interrupted`). `pair_coverage[<space id>]` has:

| Field | Meaning |
|---|---|
| `tuples` | every pair of levels |
| `illegal`, `dead`, `unproven` | the counts above |
| `drawn` | pairs some sample drew: both values are levels and live in its row (a `range` axis holding a value that is no level draws none) |
| `passed` | drawn by a `PASS` or `UNCLAIMED_PASS` sample |
| `declined_only` | drawn, and every sample that drew it was `DECLINED` |
| `never_passed` | drawn and not passed (`declined_only`, and pairs a `WRONG`, `CRASH` or `UNEXERCISED` sample drew) |
| `not_passed_total`, `not_passed` | the pairs that could still be passed (everything not passed except the `illegal`, `dead` and `unproven` ones), and the first 50 of them as `[[axisA, valueA], [axisB, valueB]]` |
| `rows`, `repair_rows` | main rows and repair rows run |
| `complete` | `false` when the run stopped before every row and repair round ran; the counts are then partial |

For a complete run, `drawn == tuples - illegal - dead - unproven`; `passed / tuples` is the number
to read, with `not_passed` saying what is missing. It is counted from every axis value of every
row (`range` axes included) and never from a shrink candidate. The console report prints one line
per space.

**`--generate-only`.** Writes `<work-dir>/bundles/gauntlet/<Op>/<Space>/<tag>_cases/sweep.json`
(one per dtype tag, as a run writes them, with `graph.template.json`) holding **all** rows of
each space, however many there are, and the pairwise reports; nothing is run and no repair rows
exist. Only the selected spaces' own earlier sweeps are replaced. The sweeps are for reading or for
running with `--golden-data-dir`; they are not part of any CI tier (that needs claims and a tier
placement, a separate decision).

**Offline coverage of an earlier run.** `--coverage-of WORK_DIR --levels-from PATH [--first N]`
prints, as JSON, the pair coverage of the `samples.jsonl` and `outcomes.jsonl` of `WORK_DIR`
(typically a random run) against the levels, and the `illegal`/`dead`/`unproven` pairs, of
`PATH`: a pairwise report or a directory of them (`<work-dir>/pairwise`). Per space of a report it
gives the `pair_coverage` fields above (without `rows`, `repair_rows` and `complete`) and
`samples` and `with_outcome` (the samples counted and those with an outcome record). `--first N`
counts only the first `N` samples of each space, so a random run of the same size as a pairwise run
can be compared with it. The same is `run_gauntlet_fuzz.coverage_of(work_dir, levels_from,
first=None, spaces_dir=None)`. Random runs have no `pair_coverage` of their own.

**What is tested.** The generator, the driver's pairwise path and the coverage are tested
(`test_gauntlet.py`, stub binary, no GPU). That every pair is covered or proven impossible is
tested for the SDPA forward spaces only; the other spaces get a generator run and nothing more.

### Reading an alignment or placement probe

Read `axis_outcomes[...]["alignment"]` against `["placement"]`, or a record's `placement`
and its `reason`. For `default` and `exact` the case is an ordinary run with the tensors
declared (and placed) at that alignment:

| Outcome | Meaning |
|---|---|
| `PASS` | The engine accepted the declared alignment, and the output matched. |
| `DECLINED` | The engine does not take a graph that declares this alignment (for example through the version gate). The expected answer for an alignment it does not support; not a defect. |
| `CRASH` | The engine accepted the graph, then failed to build or execute: for `exact`, for example, it refused a pointer that sits exactly at the alignment the graph declares. |
| `WRONG` | It ran, and the output did not match the oracle. |
| `UNEXERCISED` | The harness could not set the case up (the `reason` says why). |

What HIP_MLOPS declines (gfx950 probes, `exact` placement):

| Engine / op | Declines a declared alignment | Evidence |
|---|---|---|
| BatchNorm (forward training, backward, inference) | below what its kernels' accesses need: per-channel fp32 tensors need 4, x/y need the vector width times the element size | alignments 1 and 2 decline; 4 and above pass |
| LayerNorm, RMSNorm, Resample | below, or not a multiple of, a tensor's element size | before the check, a float tensor at alignment 1 or 2 gave garbage output; with it, every alignment passes or declines |
| ASM_SDPA, Gfx950AttentionDense | below 16 (Q, K, V, O) | alignments 1 to 8 decline, 16 and above pass |
| hipBLASLt, MIOpen | everything but 16, through the plugin API version gate | only alignment 16 is offered to them |

For `under` the case is written to be refused. The harness misaligns the pointer of
`placement_uid` to half its declared alignment and the engine must accept the graph and then
reject that pointer:

| Outcome | Meaning |
|---|---|
| `PASS` | **`BAD_PARAM` was returned** for the `placement_uid` tensor; the record has `expected_status: "BAD_PARAM"` and nothing was compared. The backend enforces the declared alignment. |
| `DECLINED` | The engine is absent from the ranked list for this graph, so the misaligned pointer was never offered to it. It says nothing about alignment enforcement. |
| `CRASH` | The engine accepted the graph and the backend did not refuse that pointer: it executed the graph (`reason` starts `expected BAD_PARAM for uid N`), declined after accepting it, or failed with another error. A backend that does not enforce the declared alignment lands here. |
| `UNEXERCISED` | The pointer could not be placed: `placement_uid` is not a device buffer of the run (a by-value scalar, a sub-byte packed input, or not a tensor), or its alignment is below 2. |

Compare the `exact` and `under` rows of one alignment. `exact` `PASS` with `under`
`PASS` is a backend that honors the declaration. `exact` `PASS` with `under` `CRASH` is one
that accepts the declaration and ignores it. `DECLINED` in both means the engine's version
or applicability gate removed that alignment from its support.

Run the driver's own tests with
`python -m pytest -q -p no:cacheprovider dnn-providers/integration-tests/gauntlet/test_gauntlet.py`
(CTest: `hipdnn_gauntlet_python_tests`). They use a stub binary and need no GPU.

## Test tiers

Tiers limit how long a run takes. A bundle's tier is its top-level directory
(`quick/`, `standard/`, `comprehensive/`, `full/`), which becomes the prefix of
its GTest suite name. A C++ test's tier is its GTest instantiation prefix.

The table shows what each CTest tier selects in the provider lanes. It applies
to the miopen, hipblaslt and HIP_MLOPS files. The ASM_SDPA lane narrows the
bundle patterns to `*Sdpa*`, uses `Smoke/*Sdpa*` as its only tier-prefixed C++
pattern, and its `full` tier is `*Sdpa*`. The hip-kernel-provider dense
attention lane (see [Tier YAML files](file-formats.md#tier-yaml-files--test_categoriesyaml))
selects only `*_SdpaFwd_Default.*` bundles in every tier, `full` included:

| CTest label | Bundle suites | C++ tests | Other labels on the suite |
|---|---|---|---|
| `quick` | `quick_*` | `Smoke/*`, `Quick/*` | `pre-commit`, `smoke` |
| `standard` | `quick_*`, `standard_*` | `Smoke/*`, `Quick/*`, `Standard/*` | `pr`, `precheckin` |
| `comprehensive` | `quick_*`, `standard_*`, `comprehensive_*` (not `full_*`) | the above, `Comprehensive/*`, and untiered `Integration*` | `nightly`, `extended` |
| `full` | everything (`*`) | everything | `all` |

Each lane YAML also defines an `ffm-quick` category (labels `ffm-quick`,
`ffm-full`). This is the fast-feedback tier. It is not a superset of `quick`.
It differs for each lane:

- miopen lists one sweep case per op by exact name.
- HIP_MLOPS selects `quick_*`, `Smoke/*` and `Quick/*`.
- ASM_SDPA selects the `*Sdpa*` subset of those.
- hipBLASLt selects only `Smoke/*`, `Quick/*` and `TestCpu*GoldenReference*`,
  so its `ffm-quick` runs no bundles.

Because `-L` is a regex, `-L quick` also selects these suites
([above](#1-ctest-by-tier-what-ci-does)).

The timeouts come from `execution_settings.category_timeouts` in each YAML file.
They are different in each file.

### How tiers cascade

A higher tier includes the lower ones only because each YAML category **lists**
the patterns of the lower tiers. CTest labels do not inherit:

```
ctest -L quick           →  quick
ctest -L standard        →  quick + standard
ctest -L comprehensive   →  quick + standard + comprehensive bundles, plus untiered C++ tests (not full bundles)
ctest -L full            →  everything
```

If a category does not list the pattern of a lower tier, it silently drops that
tier. Check the file you use. Two consequences for C++ tests in the provider
lanes follow:

- To run in PR tiers, a test must be instantiated under `Smoke`, `Quick` or
  `Standard`.
- A test with **no** tier prefix runs only from `comprehensive` up.

### The project's own binaries use a catch-all

`dnn-providers/integration-tests/test_categories.yaml` (for
`hipdnn_gpu_ref_tests`, the harness unit tests and `hipdnn_golden_data_tests`)
writes its quick tier as an exclusion: `*` minus `Standard*`,
`Comprehensive*` and `Full*`. Therefore, in this file, any test without a tier
prefix goes into quick. If the quick tier starts to time out, look for a large
shape that has no prefix. In GTest filter syntax, only the first `-` starts the
negative list. A `:-` between patterns does not negate.

## What decides whether a test runs

There are six independent layers, from outside to inside. A test runs only if
it passes all of them.

| # | Layer | Decided | Effect |
|---|---|---|---|
| 1 | Build options (`-DBUILD_CPP_GRAPH_TESTS`, op enables) | configure | A test not compiled cannot be selected. C++ graph tests are **off by default**. |
| 2 | `exclude_gpu` in the tier YAML | configure (suites) and `ctest` (selection) | Adds per-arch variant suites (`…_<category>_<arch>_suite`, label `ex_gpu_<arch>`) carrying that arch's negative patterns, or `DISABLED` for `"*"`. To pick the variant, use `-L ex_gpu_<arch>`. A plain `-L quick` also runs the unfiltered suite. See `shared/ctest/README.md`. |
| 3 | `ctest -L` / `-LE` / `-R` | ctest invocation | Selects suites by label or name. |
| 4 | `--gtest_filter` (from the YAML, or yours) | binary start | Selects cases by GTest name. Bundles that the filter excludes are dropped **before** they are parsed or read, which keeps startup fast. |
| 5 | Metadata guards and TOML `test_skips` | each test's `SetUp()` | Skip on VRAM, arch-locked golden data, or a recorded engine limitation. |
| 6 | The engine's ranked list | each test body | The engine declines the graph → the case is skipped (or fails if a claim promised support). |

A bundle that fails to load (bad JSON, a sweep case with a missing placeholder
or metadata block, a bad `golden.path`) never reaches these layers. The harness
drops it from registration with an `ERROR` log line, and the test count is
lower. Two kinds of failure are different. They fail as a registered test, with
the reason in the message: a malformed `.meta.json` or sweep-case `metadata`
block (a `BundleMetadataError`), and a tensor blob that is unreadable or has the
wrong size. The harness reads blobs when the test that needs them runs, not at
registration. See [File Formats](file-formats.md#where-each-format-is-enforced).

## Verification modes

After an engine runs a bundle, the harness compares its output with an oracle.
`--verification-mode` chooses the oracle:

| Mode | Oracle |
|---|---|
| `auto` (default) | golden data → GPU reference → CPU reference, first available wins; **fails** if none can verify |
| `golden` | golden data only; **fails** a bundle with no golden data |
| `gpu` | the GPU reference executor (no DVC pull needed); **fails** if it cannot run the op |
| `cpu` | the CPU reference executor (no DVC pull needed); **fails** if it cannot run the op |

Which operations, datatypes and layouts each reference executor can run is documented in
[GPU Reference Implementation - Operation Support](OperationSupport-GpuReferenceImpl.md) and
[CPU Reference Implementation - Operation Support](../../../projects/hipdnn/docs/OperationSupport-ReferenceImpl.md).

The comparison runs where the expected values are: on the device for a GPU
reference, and on the host for a CPU reference or golden data.
`--validator cpu|gpu` overrides this for the whole run. With `gpu`, the run
needs a device even for golden or CPU-reference comparisons. Only the pass/fail
decision moves. The host always builds the failure reports. A
`[[validator_overrides]]` entry that uses `allclose_matching_infinities` has a
host validator only. A comparison that resolves to the device therefore fails
that tensor
([details](file-formats.md#per-engine-test-config--configengine_nametoml)). The
`hipdnn_golden_data_tests` binary replaces the retired `golden-check` mode.

The `enforcement_level` metadata of a bundle can stop verification early on
purpose. `applicability` passes after the engine accepts the graph. `buildable`
passes after its plans compile.

## Reading the output

A run of `hipdnn_integration_tests` ends with up to four blocks, in this order.
The first two go to **stdout**. The last two go to **stderr**. Capture both.

### Unverifiable bundles

```text
==== REFERENCE EXECUTOR ERRORS (n) ====
==== UNVERIFIABLE BUNDLES (n) ====
```

- **Unverifiable bundles** ran on the engine with no oracle available (no golden
  data, and no reference executor supports the op). **They fail** with an
  `Unverifiable: …; tried: …` message. It names each oracle that the harness
  tried and why it could not help. The engine's output was not checked, so the
  bundle is untested, not skipped. To quarantine such a bundle until it has golden
  data or a reference, exclude it in the provider's
  `test_categories_integration.yaml` with a comment that says why. Other
  `Unverifiable:` outcomes (inputs that cannot be filled, a bundle with no
  outputs) are not about the oracle and stay skips.
- **Reference executor errors** are bugs in a reference executor that should
  have handled the op. In `auto` mode, a GPU-reference error falls back to the
  CPU reference. An error in the last oracle tried fails the test. An error in
  an explicit `--verification-mode gpu|cpu` also fails the test. That failure
  shows the reference error, and the bundle is not listed under
  `UNVERIFIABLE BUNDLES`.

### Support-claim summary

```text
==== SUPPORT CLAIM SUMMARY (ENFORCING) ====
{ "support_claim_summary": { … } }
```

The run prints this block when it pinned an engine with `--test-engine` and
found at least one bundle with a sidecar (or recorded a verdict).
`--write-support-claims` runs do not print it. The header says `ENFORCING` or
`WARNING ONLY -- NOT ENFORCED`. The JSON is sorted, so two runs over the same
tree print the same document. These keys matter to a developer:

| Key | Meaning | What to do |
|---|---|---|
| `claim_failures` | `CLAIM_BROKEN`: a sidecar claims the engine accepts this graph on this arch and platform, and it no longer does. `QUERY_ERRORED`: the support query failed, so acceptance is unknown. When enforcing, these fail the test. | `CLAIM_BROKEN`: fix the engine. If you intend to drop support, [retract the claim](adding-tests.md#retract-a-claim). `QUERY_ERRORED`: investigate the error. Do not retract the claim because of it. |
| `failed_in_use` | A **claimed** cell: the engine accepted the graph, then the test failed (execution or comparison). The run is already red for the real reason. | Fix the engine, or retract the claim until it is fixed. |
| `unclaimed_support` | The engine accepts graphs that no sidecar claims — support that exists but is not written down. `reached` / `required` show how far each graph got. Only bundles that have a sidecar can appear here. This key does not fail the run. | **Update the sidecars**. See [Updating support claims](adding-tests.md#updating-support-claims). Be careful with entries whose `reached` is below `required`: the engine accepts them, but nothing verified the result. |
| `unenforced.no_applicable_claim` | Graphs with a sidecar that the run read but that claims nothing for this arch and platform (empty sidecars count here). | If `verdicts` are all zero, this run enforced nothing. This is expected on a new arch or where sidecars are still empty. |
| `unenforced.not_selected`, `skipped_before_run`, `not_opened` | Graphs with sidecars that this run did not check: filtered out, skipped in `SetUp()`, or failed to open. | Expected on a filtered lane. In other cases, investigate. |
| `harness_defects` | Anything above zero is a harness bug. | File an issue against the integration-test harness with the summary attached. |
| `verdicts.confirmed` / `verdicts.accepted` | Counts. `confirmed`: claims that held **and** reached the depth that the `enforcement_level` of their bundle requires. `accepted`: the engine accepted the graph but nothing verified the result (see `UNVERIFIABLE BUNDLES`). | Use `confirmed` for a published support matrix. |

`not_selected`, `skipped_before_run` and `harness_defects.missed_query` appear
only when `counters_consistent` is `true`. For the full key list, the coverage
ladder, and the harness lifecycle behind
the verdicts, see
[Support Claim Enforcement](support-claim-enforcement.md#reading-the-summary).

The harness checks claims only for the engine under test, only on the running
arch and platform, and only for bundles. C++ graph tests have no sidecars.

### Test coverage summary

```text
==== TEST COVERAGE SUMMARY ====
Passed:  2457 / 5638 (43.6%)
Skipped: 3181
Failed:  0
```

Each verification test body prints the oracle that graded it between its
`[ RUN ]` and result lines. The coverage summary totals them:

```text
[ VERIFIER ] gpu_ref: .../quick/SdpaFwd/bshd/fp16/hd64_nomask_mqa/Small.json
...
Verified by: golden 0, gpu_ref 9, cpu_ref 0, none 3
```

`golden`, `gpu_ref` and `cpu_ref` name what the harness compared the outputs
against, whether the comparison passed or failed. `none` means the harness
compared nothing. Causes: a skip (the engine declined, or no reference could
run the op), a failure before the comparison, or a bundle whose
`enforcement_level` stops before the comparison. With `--gtest_repeat=N`, the
line counts the last iteration, as do the Passed, Skipped and Failed counts
above it.

**Check all three counts, not just the exit code.** A run that skips every case
is green. To tell an expected skip from a regression, read the skip reasons:

- `[arch <gcnArchName>] <reason>` — a TOML `test_skips` entry: a recorded,
  known limitation.
- `Engine could not execute bundle "<path>": Engine <NAME> does not support this
  graph (…)` — the engine declined the graph. If the cell has no claim, this
  is only a skip. If the cell has a claim, this is a `CLAIM_BROKEN` failure.
- `Bundle requires N MB VRAM …` / `Golden data generated on <arch> …` —
  metadata guards: this machine cannot host the case.

### Hard stops

| Message | Meaning |
|---|---|
| `Error: zero tests ran.` followed by `registered: N … discovered: N … selected: 0 …` | Nothing ran. `discovered: 0` is a discovery problem (wrong plugin, wrong bundle root). If bundles were discovered and the filter excluded them all, it is a filter problem: the line also says how many bundles `--gtest_filter` excluded before loading. The one expected cause is GTest sharding (`GTEST_TOTAL_SHARDS` / `GTEST_SHARD_INDEX`). If GTest printed `Note: This is test shard K of M.`, the shard can be empty because there are more shards than tests that the filter matched. In that case, the `selected:` line also reads `nothing matched --gtest_filter`, although the filter did match. In all other cases, this is a configuration bug. Exit 1. |
| `FATAL: --enforce-support-claims is active and N graph(s) carrying support` (continues over several lines) | Graphs with sidecars ran but the harness never queried any of them: enforcement verified nothing. Usually every one of them failed to open. Exit 1. |
| `support claims exist for X but were never queried` | A code path skipped the claim query: a harness bug, not a data problem. |
| `Error: --enforce-support-claims requires --test-engine …` | No engine named to check claims against. |

The zero-tests guard applies only when bundles are enabled **and** you gave
`--test-engine` or the bundle root exists.

## Runs that look green but are not

| Symptom | Why it is not a pass |
|---|---|
| `No HIP devices available; skipping …`, exit 0 | No GPU: nothing ran, no claim was checked. |
| `ctest -L <label>` prints `No tests were found!!!`, exit 0 | This is the CTest default `--no-tests=ignore`. Possible causes: a misspelled label, the wrong build directory, or a tier whose only suites are `DISABLED`. When you write a script, add `--no-tests=error`. |
| A green `ctest --output-on-failure` run | The run did not print the summaries; see [Seeing the output of a passing run](#seeing-the-output-of-a-passing-run). |
| `WARNING: Bundle tests are enabled but …` / `WARNING: No bundles could be loaded from …` | No bundles registered; only compiled-in C++ tests ran. |
| Fewer tests than expected, with `ERROR` lines at registration | Bundles that failed to load were dropped, not failed. |
| `[  PASSED  ] 0 tests.` from GTest | A filter matched nothing. When bundles are enabled, `hipdnn_integration_tests` turns this into `Error: zero tests ran.` With `--no-bundles`, and in other binaries, the run stays green. |
| `Skipped` equals the total | The engine declined everything, or a TOML skip covers everything. Read the skip reasons. A claim-free graph that the engine silently stopped supporting looks the same. |
| `verdicts` all zero while `no_applicable_claim` > 0 | Sidecars exist but claim nothing for this arch/platform, so nothing was enforced. |
| Header says `WARNING ONLY -- NOT ENFORCED` | `--enforce-support-claims=false`: the run reported broken claims and did not fail them. |
| A run without `--test-engine` | No claim summary at all, and results can come from a different engine. |
| A C++ graph test "does not run" | Expected: C++ graph tests build only with `-DBUILD_CPP_GRAPH_TESTS=ON`. |
| New bundles or sidecars do not show up | The build reads a configure-time copy; re-run CMake. |
| `auto` mode, no `dvc pull` | The golden comparison silently fell back to a reference executor. To require golden data, use `--verification-mode golden`. |
| `hipdnn_golden_data_tests` registers nothing | Golden `.bin` blobs were not pulled (or were pulled after configure). |

## In CI

- `.github/workflows/hipdnn-superbuild-ci.yml` configures with
  `-DROCM_LIBS_ENABLE_ROOT_CTEST=ON` and runs `ctest --test-dir build
  --output-on-failure`. This runs every registered lane and enforces claims. It
  does not run `dvc pull`, so golden comparisons there fall back to references and
  `hipdnn_golden_data_tests` registers nothing.
- No TheRock CI lane currently runs `dvc pull` on Linux. `therock-ci-linux.yml`
  used to, but it was gated off on 2026-09-01 when Multi-Arch CI became the
  default and was removed once it had gone unused; TheRock's multi-arch Linux
  workflows have no `dvc pull` step. Golden comparisons therefore fall back to
  references in every CI lane today.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Engine 'X' is not loaded. Check the plugin path.` | `--test-article` is wrong, or `--test-engine` is misspelled. |
| `Error: Article path does not exist` / `Config path does not exist` | Fix the path. The binary resolves both paths before it loads anything. |
| `WARNING: Bundle tests are enabled but the data directory does not exist: <dir>` | `--golden-data-dir` / `HIPDNN_TEST_GOLDEN_DATA_DIR` is wrong, or the build has no bundle copy (re-run CMake). |
| `WARNING: Bundle tests are enabled but no bundles were found in <dir>` / `WARNING: No bundles could be loaded from <dir>` | The root is empty, or every bundle failed to load. Look for `ERROR` lines above. |
| Bundles missing only in this run | `--no-bundles` or `HIPDNN_TEST_ALLOW_BUNDLES=0`. |
| `Bundle name collision: '…' produced by both:` | Two bundle paths sanitize to the same GTest name; rename one. |
| `verification-mode=golden was requested but this bundle has no golden data` | `dvc pull` the op and re-run CMake, or use `--verification-mode=auto`. |
| `verification-mode 'golden-check' has been retired` | Run `hipdnn_golden_data_tests`, and unset `HIPDNN_TEST_VERIFICATION_MODE`. |
| `STATUS_DLL_NOT_FOUND` / `0xc0000135` on Windows | Put the ROCm `bin` and `build/bin` directories on `PATH`. |
