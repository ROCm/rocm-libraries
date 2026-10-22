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
| `--no-bundles` | `HIPDNN_TEST_ALLOW_BUNDLES` | bundles on | Register only the C++ tests compiled into the binary. If the env var is set, it takes priority over the flag in either direction. |
| `--enforce-support-claims[=true\|false]` | | on | Fail a test whose claim the engine breaks. See [Support-claim summary](#support-claim-summary). |
| `--write-support-claims` | | off | Authoring run: record which graphs the engine accepts into sidecars. See [Adding Tests](adding-tests.md#record-claims-with---write-support-claims). |
| `--skip-graph-validation` | | off | Pass as soon as the engine accepts the graph, without executing it or comparing outputs. A bundle is taken only to its applicability rung, whatever its `enforcement_level`, and **skips** when the engine declines. Needs `--test-engine` for bundles. Support claims are still enforced; a claim is left `accepted`, never `confirmed`, because nothing was verified. |
| `--fail-on-unsupported` | | off | Fail instead of skip when no engine supports a graph. **C++ graph tests only.** |
| `--generate-support-matrix[=file]` | | off | Write a markdown matrix. **C++ graph tests only**; on a default build it writes an empty table. |
| `--capture-bundles <dir>` | | off | Dump compiled-in C++ graph tests as bundle JSON (see [Adding Tests](adding-tests.md#from-an-existing-c-graph-test)). |
| `--gtest_*` | | | Passed through to GTest. |

The binary refuses these flag combinations (exit 1):

- `--enforce-support-claims` or `--enforce-support-claims=true` without
  `--test-engine`.
- `--write-support-claims` together with `--enforce-support-claims` or `=true`.
- `--write-support-claims` without `--test-article`, or without a bundle root
  (`--golden-data-dir` or `HIPDNN_TEST_GOLDEN_DATA_DIR`).

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
