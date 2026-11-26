# Lane: execution contract (`EXEC`)

Find memory allocation, synchronization and graph-capture breakage that the change adds to hipDNN or a provider. The public
contract is the section *Execution contract for engines* of `projects/hipdnn/docs/PluginDevelopment.md` (engine authors) and
`projects/hipdnn/docs/user-guides/how-to/capture-hipgraph.rst` (users); cite it in `Expected:`.
The lane applies it as follows:

1. **No allocation, no sync.** hipDNN and providers allocate no device or pinned host memory outside the caller's variant pack
   and workspace, and do not synchronize, at any stage: applicability or support query, options or workspace query, plan build,
   execute, serialization or deserialization, plan destroy. *Synchronize* means device, stream or event sync, a blocking copy or
   memset, `hipFree` or `hipHostFree`, a host-consumed device-to-host readback, and `hipMemcpyAsync` from pageable host memory.
2. **Code objects.** Kernel code objects load at plan build or deserialization and may unload at plan destroy. Loading inside
   `execute()` is a violation.
3. **Host heap.** Host heap is allocated at build. `execute()` uses only small bounded temporaries freed before it returns: no
   growing cache, never pinned memory.
4. **Benchmarking is exempt.** Benchmarking only (the benchmark knob or environment variable, ingestor first-execute sampling
   with benchmarking on, `Graph::autotune()`, `execute_timed_ext`, `PROFILING_CONTROL_EXT`, the cuDNN shim's `time_sorted_plan`)
   may synchronize and allocate timing memory. Handle creation and destruction may allocate only persistent library state (for
   example hipBLASLt reduction synchronization flags), never scratch or workspace.
5. **Capture is the default.** Every engine supports capture of `execute()`, including the first execute, in Global,
   ThreadLocal and Relaxed modes with default settings, and reports `SUPPORTS_GRAPH_CAPTURE`. Where a provider integrates a
   library whose behavior hipDNN cannot control (today MIOpen), the allocation, synchronization or capture behavior that library
   imposes is documented in that provider's section, and its engines do not report the note unless they capture. Behavior not
   documented there is a violation. The exception covers documented library behavior only, never other allocation or sync. The
   note concerns `execute()` only.
6. **By-value scalars.** A pass-by-value scalar frozen into a capture is documented behavior, not a finding. A by-value-only RNG
   seed or offset with no device-tensor form is.
7. **Lifetime.** Plans, the capturing handle, workspace and buffers outlive replays; replay-after-destroy is never an argument.
   Workspace arrives uninitialized; engines zero what they read on the stream.
8. **Cross-thread hazard.** An allocation, free or module unload on one thread, in default Global capture mode, during another
   thread's Global capture invalidates that capture. This is why build and destroy are in scope.

Inputs beyond the packet: Orient's **Options and guards** (the flag that ships the code, below) and the optional
`Contract knowledge:` field (*Contract knowledge* below).

## Stages

Classify every hit by the stage that reaches it. A call counts only when it is reachable from a public entry point.

| Stage | Entry points |
|---|---|
| Applicability | `is_supported_ext`, `create_execution_plans`, matchers, `isApplicable`, `graph_match` |
| Options | knob and workspace queries, `getMaxWorkspaceSize` |
| Build | `build*`, plan builders, pack `prepare()`, plan constructors |
| Execute | `execute`, pack `launch()`, everything they call, first-execute lazy paths |
| Serialize | `serialize_compiled_plan`, `deserialize_compiled_plan`, `restoreLaunch` |
| Destroy | plan, program and module-cache destructors |

## Method

1. Inventory the HIP runtime calls the change adds or makes reachable:
   `grep -noE '\bhip[A-Z][A-Za-z0-9]*\s*\(' <changed files>`. Classify each by stage.
2. A pack or plan template reused for a new arch or engine carries its findings to that engine even when the diff adds no HIP
   call. Re-check the template.
3. Trace every sync to its gate. The gate is the benchmarking value at plan build (`benchmarkingEnabled`, a benchmark plan
   versus a plain plan) or `autotuneImpl`. A delegated library's own benchmark or find flag is not a gate.
4. On the execute path, check:
   - Every stream command uses `handle.getStream()`: no stream `0`, no `hipStreamLegacy`, no internal stream without an event
     fork and join.
   - No host read of device data.
   - No lazy first-call state: `std::call_once`, a function-local `static`, an `std::optional` behind a mutex, a grow-only
     pool, a module load, or a per-call module lookup that loads (`kernelFor`, `kernelForStream`).
   - No `hipStreamIsCapturing` or `hipStreamGetCaptureInfo` branch that changes behavior.
   - Host values frozen at capture: `resolveScalarOperand`, `readHostScalar`, RNG seed and offset, counters, `getenv`.
5. Check copy sources. `hipMemcpyAsync` host-to-device from a `std::vector`, the stack, `new` or a plan member is pageable. It
   is `sync`, settled from source: HIP documents that a copy with a non-pinned side runs synchronously (`hipMemcpyAsync` in
   `hip_runtime_api.h`), so the host waits for earlier work on the stream, on any stream, and again at every `hipGraphLaunch`
   of a captured copy, which also re-reads the host memory on replay.
6. Check plan-constant data routes. These are contract-clean; do not report them as `sync` or `alloc`:
   - kernel arguments;
   - a few on-stream `hipMemsetD32Async` words into the workspace (a fill kernel with the pattern in its arguments; no host
     wait);
   - a kernel that writes the data into the workspace from its kernel arguments.

   A table written with one memset per word or run costs about 1.7 µs of GPU time per call: at most `SUGGESTION`. A
   kernel-argument size below 256 KiB is not a finding: no compiler or HIP API check applies, and the runtime's argument pool
   chunk is 256 KiB. CUDA's 4 KiB parameter limit does not apply.
7. Check where a removed violation went. When the change removes a build- or destroy-stage allocation or sync, find the data's
   new route. Each of these moves the violation to another stage and is a finding in its own class:
   - a per-execute pageable copy (`sync`, step 5);
   - plan-owned pinned memory (`hipHostMalloc` or `hipHostRegister` at build: `alloc`);
   - a handle- or plan-level staging buffer outside the workspace that a later execute overwrites while the device or a
     captured graph still reads it (`alloc`).
8. Check workspace initialization. Atomics, split-K and two-stage reductions need an on-stream `hipMemsetAsync`; compare the
   `workspaceBytes()` regions with the zero fills.
9. Advertised capture. For every engine whose `getDetails()` or `.ued.json` `behavior_notes` gains or keeps
   `supports_graph_capture` (including `GRAPH_CAPTURE_OPS` in `flydsl/gen_descriptors.py`), walk its execute path through steps 4
   to 8. Check that the integration-test capture category covers it (`GRAPH_CAPTURE_test_categories_integration.yaml`).
10. Missing note. For every changed engine (definition below), read its descriptor or `getDetails()` at the head, even when the
   diff does not touch the note. When it lacks `SUPPORTS_GRAPH_CAPTURE` and the provider's documentation section states no
   delegated-library exception for it, file one `capture-support` finding (tier in *Severity*), separate from and in addition
   to any other class the change draws. File it once per review, not per file.
11. Delegated libraries: see *Delegated libraries*.

An engine's descriptor is its own `.ued.json`; ranker and kernel-pack JSON do not count. A *changed engine* is a new engine, or a
change to that engine's own pack, plan, registration or descriptor; shared ingestor or SDK infrastructure does not count.

## Hazard classes

Closed set; name each finding's class in `Actual`.

| Class | Look for |
|---|---|
| `alloc` | `hipMalloc*`, `hipExtMallocWithFlags`, `hipHostMalloc`, `hipHostRegister`, `hipMallocManaged`, `hipMemPool*`, virtual memory management, delegated-library allocation, outside the variant pack and workspace |
| `host-heap` | Execute-time host heap beyond small bounded temporaries freed before return; a growing cache or pool |
| `sync` | Contract 1's definition: `hip{Device,Stream,Event}Synchronize`, `hipEventQuery`, `hipStreamQuery`, `hipEventElapsedTime`, blocking `hipMemcpy` or `hipMemset`, `hipFree` and `hipHostFree` (both documented to synchronize the device), `hipLaunchHostFunc`, device-to-host readback the host consumes, `hipMemcpyAsync` from pageable host memory |
| `lazy-load` | A code-object load, compile, Find or search inside `execute()`, including by a library call that does it on first use; a `hipModuleUnload` inside `execute()` belongs to the same finding |
| `off-stream` | Work on a stream other than the handle's |
| `capture-bake` | A by-value-only RNG seed or offset with no device-tensor form |
| `uninit-workspace` | A read before write of a workspace region with no on-stream zero fill |
| `cross-thread` | `hipModuleUnload` at plan destroy, and nothing else (an unload is permitted but is capture-checked like `hipFree`) |
| `capture-support` | The engine's `SUPPORTS_GRAPH_CAPTURE` note is missing where required (step 10), or a document claims capture support or misstates a library's behavior. Judged on `execute()` only: a build or destroy finding never makes it missing. An engine that reports the note while `execute()` breaks the contract draws the breaking class at `BLOCKING` (name the reported note in `Actual`), not a separate `capture-support` finding |

`alloc` and `sync` at build or destroy stay in their own classes; only the unload is `cross-thread`.

## Severity

Per [../policy.md](../policy.md) tiers. Every EXEC finding is kind `source-claim`: it is confirmed by static analysis of the cited
`path:line` (the forbidden call at the stated stage, a pageable copy source, a missing note, a read of workspace before a write,
a library call that loads). The tiers below are this lane's contract, and `VER` judges findings against it. Kind `defect` only
when a claim cannot be proved from source at all.

| Situation | Tier | Derivation |
|---|---|---|
| `uninit-workspace` | `BLOCKING` | Wrong results; the policy's correctness rule |
| Any class on `execute()` of an engine that reports the note | `BLOCKING` | The reported contract is false |
| New engine without the note and without a documented delegated-library exception: `capture-support` | `BLOCKING` | A new engine must meet the default |
| Existing engine lacking the note, changed by this PR: `capture-support` | `IMPORTANT`, once per review | An existing gap; the policy forbids `BLOCKING` for unrelated changes |
| `alloc` or `sync` at any stage other than execute (applicability, options, build, serialize or deserialize, destroy) | `IMPORTANT` | Invalidates another thread's Global capture |
| `alloc`, `sync`, `lazy-load`, `off-stream` or `capture-bake` on `execute()` of an engine without the note | `IMPORTANT` | Contract broken, no wrong result |
| `cross-thread` | `IMPORTANT` | Same hazard, permitted call |
| `host-heap` | `IMPORTANT`; `BLOCKING` when growth is unbounded | The policy's leak rule |
| Delegated library: allocation or sync the provider's section does not document, an extension of a documented exception to another op, or a new engine relying on one | `IMPORTANT` | Undocumented library behavior |
| Docs that misstate a library's behavior or claim capture support: `capture-support` | `IMPORTANT` | Docs must match the library |
| A new engine, arch or registration that reaches an existing violation (template reuse, shared pack) | Full tier of the violation | Introduced by this change; put the new reach in `Actual` and cite the existing violation |
| The same violation's reach from engines that existed at `$BASE`; unreachable paths; code the change does not touch | `FUTURE WORK` | Not introduced by this change |
| Plan-constant table written with one memset per word or run (step 6) | `SUGGESTION` at most | Contract-clean; GPU cost only |

Build-flag-gated and default-OFF engines take the same tiers as default engines; the finding names the flag that ships the code,
found in Orient's **Options and guards** (or by grepping CMake). A finding at `IMPORTANT` or above carries in **Proof** the
stage, the call path from the public entry point, the traced gate and `path:line`.

## Exempt: do not report

- Benchmarking paths (contract 4).
- Persistent state at handle creation and destruction. These are not stages (*Stages*): allocating per-handle state there, and
  the free and initialization that go with it, are not findings. Report only scratch or workspace allocated per handle
  (contract 4), or a per-handle buffer that execute overwrites while the device or a captured graph still reads it.
- A code-object load at plan build or deserialization.
- A by-value scalar frozen into a capture; replay-after-destroy arguments.
- The approved plan-constant data routes (step 6), and kernel-argument size below 256 KiB.
- The per-device module load in `kernelFor` and `kernelForStream`: it runs only for a device other than the build device, which
  the backend rejects (`validateExecutionDevice`). At most `FUTURE WORK`, and only when the change touches it.
- A documented delegated-library behavior (below).

## Delegated libraries

Checked, not exempted. A library call reached from `execute()` that can load, compile, Find or allocate on first use is a
`lazy-load` or `alloc` candidate unless plan build warms it: hipBLASLt solution and matrix-transform code-object loading, MIOpen
Find, compile and the default allocator, rocBLAS handle scratch that regrows above a size.

A library behavior is accepted only when the provider's documentation section states it (today the first-execute Find of MIOpen
convolution plans, in `capture-hipgraph.rst` and the provider docs). It is then neither re-filed nor a `capture-support` finding,
and its engines do not report the note. Anything else the library does, including MIOpen Find, compile or allocation at
applicability, build or destroy, is reported. Check the provider docs for accuracy against the code.

## Device checks

Capture semantics the source cannot settle are `[INFERENCE]` in **Proof**: a module load inside a capture, implicit syncs in the
runtime. A pageable copy is settled from source (step 5), not an inference. Put a reproducer under **Approval requests**: the
integration-test `--graph-capture` run of the affected suites in `hipStreamCaptureModeGlobal` on the engine's architecture, or a
short capture of `execute()` that shows `hipErrorStreamCaptureInvalidated`. A reproducer strengthens a finding; it never holds
closure.

## Contract knowledge

When the `Contract knowledge:` field of `dispatch-context.md` (or the dispatch `Overrides:`) lists files or directories, read each
file, or a directory's README and the page for the changed provider. Cite matching known violations (issue and PR numbers) in
findings, and write `Contract knowledge: <paths read>` under **Coverage**. Otherwise run from this brief and write `Contract
knowledge: none supplied` under **Coverage** (a non-file line, free of `|`).

## Handoffs

- Missing capture tests: `TESTB`.
- Missing decline of a capture-unsafe configuration: `APPL`.
- Mutation proofs (remove the on-stream zero fill, move a load into `execute()`): `VER`.
- A test that poisons memory with `hipMemset` (null stream) and then runs the engine on a non-blocking stream: racy, because
  `hipMemset` on a device pointer is asynchronous and not ordered against non-blocking streams. `TESTQ`.
- Provider-contract faults that are not execution-contract: `CORR`.
