# Host timing boundaries — planning only

Source inspected at `bf27d85949b56da1452d376e45044e9b873f68b4`. No source edits, builds, tests, or GPU work performed. Paths below are relative to `projects/hipblaslt/`.

## Explicit debug categories

The user-selected umbrella is **`HIPBLASLT_JIT_DEBUG`**. Initial named categories are `timing` (final duration report) and `progress` (stage transitions during compilation). Examples: `HIPBLASLT_JIT_DEBUG=timing`, `HIPBLASLT_JIT_DEBUG=progress`, or `HIPBLASLT_JIT_DEBUG=timing,progress`. Unset/empty means no added debug output, collector clock calls, event observer, or timing/progress files. Categories are independent: progress alone does not collect/report durations; timing alone does not stream stage events. Trim comma-separated tokens, deduplicate them, and recognize explicit lowercase names. Extend with future names rather than numeric masks or an implicit `all` category.

Conservative unknown-token policy: ignore unknown categories, warn once for the explicit invalid configuration, and enable only recognized categories. Do not fail a GEMM or silently enable unrelated diagnostics. Configuration is read into private provider state; propagation to the child uses only its request argv/environment overlay, never mutation of the parent process environment.

Environment-prefix invocation works for existing sample and bench executables without public API/ABI expansion. A future per-call provider option would allow mixed requests in one process, but the initial plan does not add fields to public `Options`/`Diagnostics` or invent a bench flag without a clean way to pass that flag to the provider. Structured report location is orthogonal to category selection: use the existing unique `<output>.cwd` workspace initially, with no additional environment knobs. The coordinated direct Python proposal is `--debug timing`, `--debug progress`, or `--debug timing,progress`; Python need not interpret the hipBLASLt environment variable. Host-managed private transport options suppress child human output and select files only for enabled categories.

## Recommended measurements (only when timing is enabled)

The primary coarse measurement should be `generator_process_elapsed_ns`, taken with `std::chrono::steady_clock` immediately before and after the synchronous `process::run(request)` call in `library/src/amd_detail/hipblaslt-jit-tensilelite.cpp:127`. Successful child exit occurs after bundle publication. This measures the caller's elapsed wait for the generator invocation; it is deliberately broader than compiler subprocess time.

It includes process-runner validation and setup, environment copying, exclusive log creation, spawn, Python startup/imports, Python selection/build/bundle publication, any work after publication, child shutdown, wait, cleanup, and result mapping. The runner implements the same synchronous contract on POSIX and Windows. It excludes provider option/argv preparation and `.cwd` creation before `run`, host prediction before `generate`, and bundle validation/loading after the child exits.

If the user also wants the time until the provider has an executable supported bundle, add a distinct `provider_build_ready_elapsed_ns`, covering `Provider::compile` entry through return. On failure, use the same span with an unsuccessful outcome; do not imply readiness. Its success endpoint includes the provider's support check. It is not the same as total public API latency or GEMM adaptation latency.

Two useful provider-level child spans are `host_prediction_elapsed_ns` around `planGemm`, and `bundle_load_elapsed_ns` around `loadGeneratedBundle`. Neither belongs in Python compilation totals. Keep the first version small: the coarse generator span is required; provider total and these two spans are justified if accounting for all runtime preparation is in scope.

| Boundary | Source | Included work / important exclusion |
| --- | --- | --- |
| Request construction | `library/src/amd_detail/hipblaslt-jit-backend.cpp`, `makeGemmRequest` ending at 274 | Separate from submitted compilation |
| Generic submitted JIT request | `hipblaslt-jit-backend.cpp:276–347` | Validation/device query, provider compile at 306–307, another support check at 318, solution allocation at 323–332 |
| Provider compile | `hipblaslt-jit-tensilelite.cpp:217–275` | Operation/target checks, optional prediction, generation, load, support |
| Host prediction | `hipblaslt-jit-tensilelite.cpp:246–250` | `planGemm`; candidate ranking and fresh request JSON serialization occur in C++ |
| Generator preparation | `hipblaslt-jit-tensilelite.cpp:77–126` | Options validation, argv construction, exclusive `.cwd`, request configuration |
| Coarse generator process | `hipblaslt-jit-tensilelite.cpp:127` | Full synchronous process-runner call |
| Failure diagnostic scan | `hipblaslt-jit-tensilelite.cpp:130–137` | After the process timer; retained log scanned for failure prefix |
| Bundle loading | `hipblaslt-jit-tensilelite.cpp:144–204`, invoked at 253–254 | Envelope validation, library read/decompression/deserialization, HIP code-object load at 199–201, symbol resolution at 202 |
| Provider support | `hipblaslt-jit-tensilelite.cpp:255–257` | After bundle load; support failure does not mean generation failed |
| GEMM adaptation | `hipblaslt-jit-backend.cpp:349–390` | Separate public call after `getJitAlgo`; token registration and heuristic-result conversion |

Explicit YAML invokes `Tensile.SingleSolution`; empty `configPath` invokes `Tensile.JitGemm` after C++ prediction. The latter's process duration includes Python candidate handling and post-SingleSolution reporting. Record the actual module/mode so these totals are comparable with clear scope. Python's public-wrapper total excludes initial interpreter/module imports; host minus Python elapsed is **unattributed process overhead**, not specifically Python startup time.

## Process and failure semantics

`library/src/amd_detail/hipblaslt-jit-process.hpp` exposes `started`, `exited`, `exitCode`, `terminationSignal`, and `error`. `run` catches ordinary exceptions into this result (`hipblaslt-jit-process.cpp:465–482`). The Windows implementation uses `CreateProcessW` at 316 and indefinite wait at 331; POSIX uses `posix_spawnp` at 441 and `waitpid` at 449. Do not change process lifecycle behavior or introduce timeout/cancellation as part of instrumentation.

- Collect elapsed immediately after `run`, before converting unsuccessful status to the existing exception. Spawn/setup failure still has an elapsed process-attempt duration and `started=false`.
- An earlier provider validation failure has no generator invocation. Represent its generator timing as absent, not a fabricated zero-duration build.
- Record child process outcome separately from Python bundle publication and provider load/support outcome. A bundle can exist even when later reporting or host loading fails.
- If the child is killed or crashes before telemetry is finalized, retain the host elapsed and actual runner outcome; child phase timings may be missing or partial.
- Use invocation-local clocks and storage. Do not add global timer state, parent environment/cwd mutation, HIP event timing, or GPU synchronization. Distinct calls already require distinct fresh output paths.
- Timing/reporting must preserve the primary result and diagnostic exception. Stop the relevant span before serializing its report; state what report overhead remains inside outer spans. No throwing destructor for telemetry emission.
- Public API comments explicitly promise no persistent cache/reload across invocations (`hipblaslt-jit.hpp:62–64`). Repeated calls are fresh generator requests, although OS/tool caches may change elapsed time. Repeated kernel launches in one prepared algorithm are not new build samples.

## Report transport and client presentation

Keep Tensile-specific phases out of generic `jit::Diagnostics` unless the parent plan deliberately chooses a reusable generic telemetry API. The current generic struct has only `backend` and `message` (`hipblaslt-jit.hpp:56–60`); `Options` is already provider-specific (`hipblaslt-jit-tensilelite.hpp:10–29`).

A versioned, invocation-specific timing sidecar under the existing owned `.cwd` workspace is a practical initial transport. Let the host choose a separate child report path, and keep host and child report files distinct so they cannot overwrite one another. Bundle manifest/loader schemas should not become timing-report storage. Python can atomically write its own sidecar after bundle publication, capturing the publication rename itself. Missing/unreadable timing data should be an optional-telemetry condition, preserving the build result.

Only the `timing` debug category enables the final timing report and its child collector. The host passes a separate child JSON report path when enabled; the child writes that report without printing a second human summary. With timing disabled, do not create timing files or take phase timestamps. Retain a versioned host report in the owned `.cwd` workspace when available; an explicit export destination can be added independently later. If timing is ever exposed through `Diagnostics` instead, explicitly preserve it across GEMM adaptation: the benchmark currently saves only `info.message` before `getGemmAlgo` resets diagnostics (`testing_matmul.hpp:4168–4174`).

Benchmark preparation runs once before reference, warmup, or timing (`clients/common/include/testing_matmul.hpp:4191–4204`). When explicitly requested, emit build duration/report-path diagnostics on stderr alongside recipe/manifest/prediction. Keep existing GEMM latency, GFLOPS, and CSV measurements unchanged. A timer around the whole `select_jit_algo` helper would include request creation, backend construction, device query, provider build, and GEMM adaptation; call that `jit_preparation_elapsed`, not generator duration.

Render the final timing report after the provider attempt, including unsuccessful outcomes, then emit it as a single buffered block under a process-local output lock. Include a unique request ID and process ID; each request owns a distinct structured report file. A process-local lock cannot promise atomic multiline stderr across independent processes, so prefix lines with the request ID and use separate JSON files for dependable concurrent comparison. Reporting errors preserve the primary build result.

Illustrative presentation only, not measured values:

```bash
HIPBLASLT_JIT_DEBUG=timing,progress hipblaslt-bench <existing JIT arguments>
```

```text
JIT [request-id] progress: compiling main kernel
JIT [request-id] progress: publishing bundle
JIT [request-id] timing: generator_process=2.843 s, bundle_load=0.021 s, provider_build_ready=2.870 s
JIT [request-id] timing report: <output>.cwd/host-timing.json
```

## Live progress requires an explicit observer

The existing runner sends child stdout/stderr to the retained per-invocation `.log`, then blocks in `WaitForSingleObject` or `waitpid`. Merely adding Python stderr messages will **not** provide live host-visible progress. Timing's final JSON sidecar is likewise not a progress transport.

The smallest portable initial design keeps the process runner contract unchanged and adds a request-local event file plus observer at the provider layer:

1. Only for `progress`, create a fresh `progress.jsonl` in the provider-owned `.cwd` workspace. Pass the selected debug categories, the managed transport location, and request ID through private child CLI options. A single managed transport directory can select both independent timing/progress files; exact private flag names are an implementation detail. Keep stdout/stderr redirection and the retained log unchanged.
2. The child's main process is the sole event writer. Append a complete bounded JSON line for each meaningful stage transition, and flush immediately. The coordinated event envelope is `version`, `request_id`, `seq`, `kind=stage`, `stage`, `transition=start|end`, and `status=ok|failed|rejected` on end events, with optional `candidate_id` or bounded detail/count. Progress alone requires no duration timestamps. Parallel workers do not independently append into this file.
3. Start a request-local observer thread immediately before the synchronous `process::run`. It checks the file at a modest interval (for example 100 ms), reads from its last offset, buffers an incomplete trailing line, validates request ID/sequence, and forwards complete events to a stderr sink. Use a condition-variable wakeup rather than a busy loop. No OS-specific process-spawn or pipe changes are required.
4. After `run` returns, signal the observer, drain complete remaining events, and join it before destroying the per-call state. Handle exceptional exits with the same scoped shutdown. Do not wait indefinitely for a final child event: killed/failed children may leave only partial progress. Bound line sizes, accepted event counts, and diagnostic detail to prevent report parsing from dominating or exhausting the build.
5. The host also emits its own prediction/generator-start/bundle-load/support transitions. Identify host versus child event sequence, since their ordering is not one shared clock or sequence. Emit actual stage transitions, not estimated completion percentages or compiler-output scraping. Thread-safe output preserves attribution among simultaneous requests; separate process reports may interleave on shared stderr.
6. Telemetry I/O, observer, or parsing failures must not replace generator/provider results. Disable the affected debug channel and issue at most one secondary diagnostic for that request. Child events are best effort; the final host outcome remains authoritative.

When both categories are enabled, progress I/O and observer activity can affect measured wall time. Record enabled categories in the final report so timing-only and timing-plus-progress samples are distinguishable. Starting/joining the observer is outside the narrow `process::run` timer; event production during the child is naturally inside it. This progress work is a distinct implementation step, not something already provided by the final timing report.

## Validation to include in the implementation plan

No validation was executed for this planning task.

1. Reuse the existing host process test seam to inject child delay and assert coarse elapsed contains it; validate setup/spawn/nonzero/signal outcomes without requiring a GPU. `clients/tests/jit/provider_process_test.cpp` already exercises nonzero exit at 181–186, missing executable at 188–191, concurrent isolation at 233–246, and POSIX signal status at 249–255. Since the recommended timer is provider-local rather than part of `process::run`, the implementation should factor only the narrow telemetry seam needed for GPU-free checks; do not move timing into platform-specific spawn internals just to fit a test.
2. Cover explicit SingleSolution and predicted JitGemm modes, with host prediction absent/present respectively and module recorded.
3. Inject a loader failure after a successful child to confirm generator success and provider failure remain distinct.
4. Confirm concurrent distinct output paths produce independent reports and no shared timer state.
5. Preserve the existing bench sentinel: `clients/bench/test_jit_gemm.py:891–907` already checks one generator invocation across warmup/timing, an injected two-second delay, and exclusion of that delay from three measured CPU-timed calls. Extend that test to inspect timing diagnostics when implementation is authorized; retain stderr/stdout provenance checks at 91–95.
6. Validate the debug-category matrix: unset/empty, timing only, progress only, both, duplicates/whitespace, and unknown tokens. Disabled categories must produce no associated output/files/collector activity. A delayed fake child should demonstrate that a flushed progress event is observed before child exit, with complete final drain, no stdout contamination, retained normal logs, and safe shutdown on nonzero/signal outcomes.
