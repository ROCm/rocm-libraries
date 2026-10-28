# Saved paged split-KV decode binding host overhead case study

This case study concerns host submission from saved explicit gfx942/gfx950
non-FP8, LSE-off paged one-token attention bindings. It describes preparation
outside the timed loop and validation of equivalent segment/reducer kernels.
It does not measure the native provider or change heuristic attention routing.

## Caller and preparation

The measured caller is an executable tuning candidate's saved Torch binding:
`attention_execution_candidates` → `bind_tuning_attention_torch` →
`TorchBinding.launch`. The first ordinary eager call goes through the existing
public runner, validating the launch and initializing an existing
`Attention3DExecution`. Binding creation neither launches nor drains.

The binding then holds its compiled pipeline, launch geometry, stream wrapper,
scalar defaults and three private scratch tensors. Compatible warm calls use
the existing ordinary pipeline directly. They avoid allocation, repeated
immutable admission/spec/cache work and redundant Torch context construction.
Kernargs are still packed and retained on every invocation; live pointers and
scalar overrides are read on every call.

The new binding path uses `None` for Torch-current selection. It resolves that
selection before passing an integer to the existing runner and pipeline.
Legacy zero inputs and explicit integer handles preserve the current runtime
API. An explicit stream different from the caller's current stream is entered
using the cached stream object and restored afterwards.

## Ownership and compatibility

Scratch belongs to one saved binding. Separate bindings, including compatible
maximum-KV bindings, share existing immutable compiled caches but own separate
scratch. Each binding uses
`4 * total_q * num_query_heads * num_segments * (head_size + 2)` scratch bytes.
There is no new global cache, eviction policy, until-drain registry or cleanup
API. Generic HIP runtime and launcher code are unchanged.

LSE-on bindings use the general runner on every invocation, preserving live LSE
buffer overrides and the runner's output validation. They keep the existing
integer-stream interface and default; they do not prepare private eager scratch.

Only the binding's creating host thread reuses its scratch under the existing
execution owner's whole-pair guard. Different streams/devices/layouts, foreign
host threads, reentrant calls, internal graphs and external capture use the
general runner with its existing ownership. This path does not introduce a
general concurrent runtime guarantee.

Ordinary per-launch retention keeps submitted tensors and scratch alive after
the binding is discarded. Fenced calls complete through the ordinary pipeline;
unfenced calls require the caller's existing finite-batch completion boundary.
Drains remain outside host enqueue timing and must happen before reading
results. No context remains entered between binding invocations.

## Validation method

[Host regressions](../library/tests/test_attention_decode_binding.py) cover
allocation/context reuse, independent bindings, executable-candidate reachability,
live pointer repacking, scalars, stream selection/overrides, changed layouts,
foreign/reentrant calls, capture/graph fallback, enqueue failure and ordinary
retention after discarding a binding. The focused test is installed in the
attention CPU lane. Tests for the withdrawn global cache and runtime changes
are removed; pre-existing suites retain their baseline source.

The LSE dispatch host regression separately checks repeated calls of one saved
binding on gfx942/gfx950, including LSE-buffer and integer-stream overrides.
It verifies that each call reaches the general runner without preparation.

Fresh hardware validation compares the final binding implementation against the
pinned base using identical executable specs, LLVM/HSACO hashes and geometry.
The synthetic BF16 cases have B/KV = 1/2048 and 4/4096, Hq/Hkv = 32/4, D = 128
and page size 16. Stable bindings and a rotating KV/KV+1/KV+2 maximum-length
series are warmed separately; device lengths stay at KV. Internal and outer
graphs are disabled for measurements.

Binding, setup, compilation warmup and drains are excluded from
`perf_counter_ns` host enqueue timing under `no_fence()`. HIP-event intervals
are measured separately and include host pacing. Three paired trials alternate
arm order on the same GPU and pinned CPU, using 20 warmups, 100 iterations per
finite batch and seven samples; the first sample is excluded from medians.

Every measured arm checks outputs against an independent NumPy FP64 reference
over quantized BF16 inputs before and after timing, with `atol=0.01` and
`rtol=0.02`. Post-checks poison output before a fresh launch. The validation
harness and private raw measurements remain outside this PR. gfx1250 hardware
and a migration of the generic stream API are outside scope.

## Results

The saved binding reduced host enqueue time at every measured configuration
and in every paired trial. Each value below is
`100 * (1 - median(final/base))` over the three paired trials. These are
per-configuration results, not confidence intervals.

| Target | B / KV | Binding mode | Host enqueue reduction | Host-paced HIP-event interval reduction |
| --- | --- | --- | ---: | ---: |
| gfx942 | 1 / 2048 | Stable | 57.1% | 56.8% |
| gfx942 | 1 / 2048 | Maximum-length series | 57.5% | 57.1% |
| gfx942 | 4 / 4096 | Stable | 57.0% | 56.1% |
| gfx942 | 4 / 4096 | Maximum-length series | 57.6% | 57.0% |
| gfx950 | 1 / 2048 | Stable | 53.7% | 54.0% |
| gfx950 | 1 / 2048 | Maximum-length series | 54.1% | 53.3% |
| gfx950 | 4 / 4096 | Stable | 54.2% | 53.9% |
| gfx950 | 4 / 4096 | Maximum-length series | 54.3% | 54.2% |

All compared segment/reducer LLVM and HSACO hashes, executable specs and
grid/block geometry matched across arms and trials. Each target passed 48
independent before/after checks in the measured runs, 64 existing runtime
numerical cases and nine stream/capture/retention/completion checks. Ten
existing LDS-admission numerical cases were explicitly deselected. No hardware
skip or failed check is counted as a pass.

The baseline was `41317132e3f7c081bb36fd228efb59145fe6368d`; the measured
implementation was `b106aeda4da3265c35612b99c69f2a99a2c8c0de`. The explicit
specs were `gfx942_u3d_splitkv_seg128_t1xb` and
`gfx950_u3d_splitkv_seg128_t1xb`. The environment used Python 3.12, the Python
engine, LLVM23, ROCm 7.14.0, Torch `2.13.0+rocm7.14.0`, HIP `7.14.60850` and
NumPy 2.2.6. Compilation ran in Torch-free COMGR workers; launch and compilation
used release-matched wheel HIP/COMGR libraries.

The source-and-adapter archive SHA-256 was
`df8c036a8b55370c134fe0ddeb9db3793ac81d82d8bcd6bf792e426b7c6deb31`;
the separately hashed timing harness was
`bdc02b9d13d80d3c370b15e0276bc331494ae6b801f92a91e03922590b3bfa3e`.
The validation record retains per-file provenance, commands and paired trials.

These results establish savings at the saved Python binding's host submission
boundary for these configurations. HIP-event intervals include host pacing and
do not establish a device-kernel speedup. The runs do not measure a new binding
on every invocation, native provider performance, two-device concurrent
submission or gfx1250 hardware. Earlier measurements of the withdrawn generic
runtime/cache design are not used for these results.
