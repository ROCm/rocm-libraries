# Current multi-GPU behavior of `tensilelite-client` for fused GEMM+A2A

## Scope and sources

This is a factual source-reading note, not a design proposal or decision record. It was
written on 2026-08-27 from worktree branch
`users/alvasile/review_yiding12_gemm-a2a`, whose product-code base is remote
`users/yiding12/gemm-a2a` at `612d4cb83a3317e3e1885b4bb5285dc11febba28`.
No product code was changed and no GPU run was performed for this note.

The branch's stated intent comes from [PR #10925, "feat(tensilelite): fuse an
all-to-all epilogue into the GEMM store path"](https://github.com/ROCm/rocm-libraries/pull/10925)
(head `612d4cb83a`, retrieved 2026-08-27). The PR describes an opt-in GEMM
epilogue that uses SDMA to redistribute output tiles, a four-GPU validation
harness, and a separate receive buffer plus local-output validation.

## Short factual answer

| Area | Existing normal client | Fused GEMM+A2A path in this branch |
| --- | --- | --- |
| Device execution model | One HIP device per client process, selected by scalar `--device-idx`. | One process explicitly owns ranks `0..W-1`, creates resources and launches on every rank. |
| Existing multi-GPU support | Python benchmark orchestration can split *independent problems* into separate per-GPU client processes and merge their CSVs. It is not a collective execution model. | A special branch in `main.cpp`; it does not extend the normal listener/data/validation pipeline. |
| Topology/ranks | No normal-client rank or peer topology abstraction was found in the execution path. | `W` visible HIP devices, fixed ordinal rank mapping, all-pairs P2P enablement, one SDMA queue per source-device/peer pair. |
| Data/validation | `DataInitialization`, `ReferenceValidator`, and `BenchmarkTimer` operate on one set of current-device inputs/results. | Hand-built per-rank buffers, CPU goldens, bespoke synchronization, and bespoke full/none validation. |

The source currently models only `FusedGemmA2A`: a GEMM store-path/epilogue
operation. The generated path divides the first `AM` feature rows among
receivers and leaves `[AM, M)` locally stored. The inspected implementation
does not expose an A2A+GEMM/prologue mode or a shared collective abstraction.
The relevant codegen gate is specifically a store dispatch gate and a
`FusedGemmA2A` ProblemType property. (`Tensile/Components/GlobalWriteBatch.py:75-101`,
`Tensile/KernelWriterAssembly.py:16486-16557`, `Tensile/Contractions.py:296-297`.)

## Observed facts

### 1. The normal client is single-device per process

`main()` calls `GetHardware(args)` once and `GetStream(args)` once before
constructing the normal execution pipeline. `GetHardware` reads scalar
`--device-idx`, calls `hipSetDevice(deviceIdx)`, and returns the current
hardware; `GetStream` creates one stream or returns the default stream.
(`client/main.cpp:679-703`, `client/main.cpp:1039-1069`.)

The normal path then creates one `DataInitialization` object and one solution
iterator. For each problem it calls `prepareGPUInputs`, prepares rotating
buffers from that one input set, solves against the one `hardware` and one
`stream`, and launches through one `SolutionAdapter`.
(`client/main.cpp:1162-1197`, `client/main.cpp:1274-1295`,
`client/main.cpp:1419-1538`.) `DataInitialization` allocates through ordinary
`hipMalloc` and retains vectors of GPU pointers/current inputs rather than a
rank-indexed device context. (`client/src/DataInitialization.cpp:399-410`,
`client/src/DataInitialization.cpp:1493-1602`,
`client/include/DataInitialization.hpp:353-435`.)

Normal reference validation likewise receives one CPU reference input set and
one result input set. It copies each output tensor from that result's device
pointer to one host buffer before comparison. (`client/src/ReferenceValidator.cpp:84-125`,
`client/src/ReferenceValidator.cpp:183-215`,
`client/src/ReferenceValidator.cpp:396-553`, `client/src/ReferenceValidator.cpp:821-898`.)
The normal `BenchmarkTimer` synchronizes the current device or current stream,
not a set of devices. (`client/src/BenchmarkTimer.cpp:436-465`.)

There is an existing Python-level multi-GPU facility, but its semantics are
independent benchmark parallelism: it splits `N` problem entries among GPU
processes, writes a per-GPU `device-idx`, `problem-start-idx`, and
`num-problems`, waits for all processes, then merges separate result CSVs.
(`Tensile/ClientWriter.py:269-305`, `Tensile/ParallelExecution.py:53-107`,
`Tensile/ParallelExecution.py:169-193`, `Tensile/ParallelExecution.py:216-320`.)
It does not establish ranks, exchange pointers, synchronize a collective, or
validate one collective result across devices.

### 2. The fused path is a separate client control flow

`fused-gemm-a2a` is a boolean CLI/config option, defaulting to false. The
generated client INI records it from `ProblemType.fusedGemmA2A`, and
`ClientProblemFactory` copies that option into each `ContractionProblemGemm`.
(`client/main.cpp:239-240`, `Tensile/ClientWriter.py:630-676`,
`Tensile/ClientWriter.py:728-772`, `client/src/ClientProblemFactory.cpp:60-64`,
`client/src/ClientProblemFactory.cpp:365-408`.)

When the option is true, `main()` calls `runFusedA2A(...)` and `continue`s
before `listeners.preProblem`, `DataInitialization::prepareGPUInputs`,
solution-loop benchmarking, `ReferenceValidator`, and `BenchmarkTimer`'s
per-solution hooks. (`client/main.cpp:1260-1282`.) Thus, although the normal
listeners are constructed earlier, their ordinary per-problem/per-solution
flows are not the mechanism that prepares, launches, times, or validates the
fused collective.

The executable must be built with `TENSILELITE_ENABLE_SDMA_A2A`. That CMake
option is OFF by default; when it is enabled the client links hsakmt and HSA,
and when it is disabled `runFusedA2A` fails rather than launching with missing
SDMA queues. (`client/CMakeLists.txt:24-29`, `client/CMakeLists.txt:102-113`,
`client/src/FusedA2AClient.cpp:1078-1085`.)

### 3. Current collective topology and rank configuration

`runFusedA2A` reads `--fused-a2a-world` as `W`; the CLI default is four. The
host ABI reserves eight peer groups, so the code rejects `W < 1` and `W > 8`.
(`client/main.cpp:291-297`, `client/src/FusedA2AClient.cpp:50-76`,
`client/include/FusedA2AKernArg.hpp:22-73`.)

The special path rejects any `--device-idx` other than zero. It assigns rank
`d` to visible HIP device ordinal `d`, requires at least `W` visible devices,
and directs users to `HIP_VISIBLE_DEVICES` to select the visible set. It has no
CLI rank-map or nonzero start-device model. (`client/src/FusedA2AClient.cpp:78-97`.)

For every ordered pair of distinct ranks it checks `hipDeviceCanAccessPeer` and
enables peer access from source to target. It allocates one SDMA queue per
source device and peer, including a loopback queue. (`client/src/FusedA2AClient.cpp:462-500`.)
The queue helper maps HIP ordinals to HSA/KFD topology nodes and chooses the
first recommended SDMA engine for a source-to-destination IO link, with engine
zero as a fallback. (`client/include/SdmaQueue.hpp:74-155`,
`client/include/SdmaQueue.hpp:158-210`, `client/include/SdmaQueue.hpp:212-300`.)

This is consequently single-process, visible-device P2P orchestration. The
source does not present a process-rank transport, a distributed address
exchange, or a generic topology object to the normal client.

The existing benchmark-parallel facility is not compatible with this fused
rank model: its per-GPU config writer replaces `device-idx` with each process's
GPU ordinal, while `runFusedA2A` refuses every value except zero.
(`Tensile/ParallelExecution.py:169-193`, `client/src/FusedA2AClient.cpp:78-86`.)

### 4. Current data preparation and kernel ABI

The path accepts a plain GEMM and rejects unsupported inputs/features such as
batched GEMM, workspace, bias, E, activation, scales, sparse metadata, and
non-bf16 A/B. It derives `M`, `N`, and `K` as `uint32_t`, obtains the macro-tile
shape from the selected solution, and enforces the `AM`, shard, tile, SDMA
field, and D-layout constraints before allocating/launching.
(`client/src/FusedA2AClient.cpp:99-191`, `client/src/FusedA2AClient.cpp:193-275`,
`client/src/FusedA2AClient.cpp:302-357`.)

The data contract implemented by the client is:

- A is initialized once and uploaded identically to every rank; B is initialized
  separately for each rank. CPU goldens are therefore calculated separately for
  every source rank. (`client/src/FusedA2AClient.cpp:359-409`,
  `client/src/FusedA2AClient.cpp:450-459`.)
- The first `AM` feature rows of each source rank's GEMM output are scattered.
  `nShard = AM / W`; destination rank `dst` receives its feature interval
  `[dst*nShard, (dst+1)*nShard)` from every source rank. `[AM, M)` remains in
  that rank's local `outD` buffer. (`client/src/FusedA2AClient.cpp:181-230`,
  `Tensile/Components/GlobalWriteBatch.py:2640-2673`.)
- Each rank has separate A, B, C, D, counter, flag, and receive allocations.
  The flag plus receive allocation is fine-grained because remote peers write
  it; counter, A, B, C, and D use ordinary device allocation. (`client/src/FusedA2AClient.cpp:278-294`,
  `client/src/FusedA2AClient.cpp:416-460`.)
- `ContractionInputs` still carries the normal A/B/C/D inputs. The receive
  buffers, remote flag pointers, queue ring/read/write/doorbell pointers,
  counter pointer, rank, world, drain mask, and `AM` are appended manually as a
  fused kernarg segment. (`client/src/FusedA2AClient.cpp:602-655`,
  `client/include/FusedA2AKernArg.hpp:58-124`,
  `Tensile/Components/Signature.py:437-456`.)

The fixed-size, host-side kernarg packer duplicates layout knowledge from the
Python signature generator: both define eight peer groups and the peer field
ordering. This is an observed implementation boundary, not a proposed design
judgment. (`client/include/FusedA2AKernArg.hpp:6-8`,
`client/include/FusedA2AKernArg.hpp:22-28`, `client/include/FusedA2AKernArg.hpp:75-123`,
`Tensile/Components/Signature.py:33-100`.)

### 5. Synchronization and completion behavior currently implemented

Before each repeat iteration, the host zeros counter, flag, and receive memory
for every rank and then calls `hipDeviceSynchronize()` for every rank. It next
records start/stop events and enqueues every rank's kernel before it starts
stream synchronization. (`client/src/FusedA2AClient.cpp:657-714`.) There is no
host `hipStreamWaitEvent` cross-device dependency in this path.

At generated-kernel level, the PUSH path waits for its stores to reach HBM,
performs a work-group barrier, elects one writer, then emits a copy and a flag
atomic as one SDMA queue reservation before submission. (`Tensile/Components/GlobalWriteBatch.py:2753-2779`,
`Tensile/Components/GlobalWriteBatch.py:2880-2895`,
`Tensile/Components/GlobalWriteBatch.py:3069-3080`.) A counter selects the last
work-group in the grid. If the receive-drain bit is enabled, that work-group
polls all `W` local flag slots until each reaches `tokenTiles`; if the
send-drain bit is enabled, it also polls the outbound counter until it reaches
`W`. (`Tensile/Components/GlobalWriteBatch.py:3175-3233`,
`Tensile/Components/GlobalWriteBatch.py:3245-3353`.)

The current defaults are receive drain enabled and send drain disabled.
(`client/main.cpp:292-295`, `client/src/FusedA2AClient.cpp:50-56`.) Therefore,
under the default mode each launched kernel's completion is intended to imply
the rank's receive buffer is complete; with receive drain explicitly disabled,
the subsequent host stream synchronization does not itself add an independent
cross-device receive-completion fence.

After every launch iteration, the client waits on every rank's stream. It then
checks each counter guard tail and, when `AM != 0`, polls each rank's outbound
counter. Write-pointer monotonicity/alignment checks run only when numeric validation is
enabled. (`client/src/FusedA2AClient.cpp:686-785`,
`client/src/FusedA2AClient.cpp:787-823`.)

### 6. Fused validation is bespoke and does check all successful ranks

The special path rejects a positive `--num-elements-to-validate`: it accepts
only `-1` for all-element numeric validation or `0` for no numeric validation.
With validation enabled it computes a full CPU bf16 golden for every source
rank. (`client/src/FusedA2AClient.cpp:53-65`, `client/src/FusedA2AClient.cpp:385-414`.)

For each successful iteration, it copies and compares every destination rank's
entire receive buffer. The expected value is indexed by both destination and
source, so a source placed in the wrong receive slot is observable. It then
copies each rank's D buffer and compares every local-tail element
`m in [AM, M)` against that rank's own golden. (`client/src/FusedA2AClient.cpp:825-909`.)
The scatter portion `[0, AM)` is covered through the receive comparisons,
including the loopback source/destination slot; the local tail is covered
through D. A passing iteration therefore visits all `W` ranks in both loops.
The loops stop after the first failing rank, which affects failure reporting
but not what a passing iteration has checked. (`client/src/FusedA2AClient.cpp:831-872`,
`client/src/FusedA2AClient.cpp:874-908`.)

Every repeat rezeros collective state and revalidates. The pass/fail verdict
also incorporates HIP failures, guard corruption, write-pointer invariant
failures when enabled, outbound-signal failures, receive validation, and local
validation. (`client/src/FusedA2AClient.cpp:527-541`,
`client/src/FusedA2AClient.cpp:911-952`, `client/src/FusedA2AClient.cpp:1064-1077`.)

Important configuration fact: the supplied fused YAML has
`NumElementsToValidate: 0`. The generated INI passes that value as
`num-elements-to-validate`; under the fused client's rules this skips the CPU
golden and all numeric receive/local comparisons, leaving clean exit plus the
always-run guard/outbound checks. (`Tensile/Tests/common/gemm/gfx950/fused_a2a_disabled.yaml:14-24`,
`Tensile/ClientWriter.py:760-779`, `client/src/FusedA2AClient.cpp:53-65`,
`client/src/FusedA2AClient.cpp:385-414`, `client/src/FusedA2AClient.cpp:825-829`.)

### 7. Current in-tree test coverage versus the PR's reported result

The dedicated in-tree test is a unit characterization test. It generates
fused kernels from the YAML, asserts they assemble for gfx950, and checks for
three emitted instruction markers. It does not build or execute
`FusedA2AClient.cpp` on multiple GPUs. (`Tensile/Tests/unit/characterization/_codegen/test_fused_a2a_gfx950_char.py:3-64`.)

The YAML is deliberately named `fused_a2a_disabled.yaml`; the common config
test helper marks paths containing `disabled` as skipped. The test above
directly consumes that file for code generation, but the ordinary config sweep
does not run it as an end-to-end client test. (`Tensile/Tests/common/gemm/gfx950/fused_a2a_disabled.yaml:3-12`,
`Tensile/Tests/common/config_helpers.py:112-117`.)

PR #10925 reports successful four-GPU numerical runs outside CI. That is a
reported external test result, distinct from the in-tree code-generation
coverage described above. The repository source inspected here does not make
those reported hardware runs a CI-gated test.

## Open design questions (not answers or decisions)

1. What first-class collective/rank/topology object should represent both this
   epilogue operation and a later A2A+GEMM prologue, rather than encoding a
   rank list, resource ownership, and peer pointers in one special client
   function?
2. Which topology and process scopes are in contract: one process over visible
   local HIP devices only, arbitrary visible-device maps, multiple processes,
   multiple nodes, or a defined subset? What capability checks and diagnostics
   belong to that contract?
3. What must completion mean for each exposed buffer: local GEMM output,
   outgoing source data no longer being read, inbound receive data ready, and
   any later prologue input ready? Which of those must compose with normal HIP
   stream ordering?
4. How should collective-owned input/output shards, scratch, queues, flags,
   and reference values be represented so normal data preparation, rotating
   buffers, benchmark timing, and validation can operate on them without a
   parallel bespoke lifecycle?
5. What validation contract should be required for every participating rank,
   every source/destination slot, both fusion directions, partial tiles, and
   repeated reuse? Should a generated/default benchmark configuration ever
   select clean-exit-only validation for a collective correctness run?
6. What test tiers can make the intended behavior enforceable: source/ABI
   tests, generated-code tests, single-device loopback, multi-device numerical
   runs, topology-negative tests, and a hardware CI lane?

## Boundary of this research

The facts above describe the code and current PR text only. They do not
establish that the current SDMA protocol is correct on hardware, choose a new
API, or prescribe a migration. Those require the subsequent design discussion
and decisions.
