# The correctness harness

[RUNBOOK.md](RUNBOOK.md) owns execution order; this page owns the oracle, the
comparison, and the assertions without which a green result means nothing. Paths are
relative to the repository root.

## Shape of the harness

A standalone program, built against an **installed** hipDNN
(`find_package(hipdnn_frontend)`, `find_package(hipdnn_data_sdk)`), that does five
things in order:

1. Builds or deserializes the graph.
2. Allocates one buffer per non-virtual tensor UID, twice — a reference set and a
   kernel set — and fills both from the **same seeds**.
3. Runs the hipDNN reference over the reference set.
4. Launches your kernel(s) over the kernel set.
5. Compares every graph output, per element type, at a stated tolerance.

No in-tree program does all five. Each piece exists:

| Step | Existing pattern to imitate |
|---|---|
| 1, 3, 5 for a fused graph | `projects/hipdnn/samples/batchnorm/FusedBnInfDReluBnBwd.cpp:144-207` |
| 2 (buffers, host/device migration) | `hipdnn_data_sdk` `utilities::Tensor<T>` / `MigratableMemory` — `memory().hostData()`, `memory().deviceData()`, `markHostModified()`, `markDeviceModified()` |
| 4 (hipRTC compile + module launch) | `projects/hipdnn/samples/example_engine_plugin/src/hip/` — `HipKernelCompiler`, `HipCompiledProgram`, `HipRunnableKernel`; MIT-licensed and self-contained |
| 5 (validators) | `hipdnn_test_sdk::utilities::createAllCloseValidator<T>()`, `CpuFpReferenceValidation<T>`, `validateAndReport<T>` |

Copy `example_engine_plugin/src/hip/` rather than re-deriving the hipRTC sequence.
It is `hiprtcCreateProgram` → `hiprtcCompileProgram` (on failure,
`hiprtcGetProgramLogSize` / `hiprtcGetProgramLog` — always print it) →
`hiprtcGetCodeSize` / `hiprtcGetCode` → `hipModuleLoadData` → `hipModuleGetFunction`
→ `hipModuleLaunchKernel`.

Do **not** build the harness inside the hip-kernel-provider's ingestor path. That
path is the integration skill's territory, and a kernel proven there is proven
through descriptors you have not yet authored.

## The oracle

`hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor` consumes the serialized graph
directly:

```cpp
auto [serializedGraph, serErr] = graph->to_binary();
hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor cpuExecutor;
cpuExecutor.execute(serializedGraph.data(), serializedGraph.size(), cpuVariantPack);
```

The variant pack is `std::unordered_map<int64_t, void*>`, UID to buffer — **host**
pointers for the CPU reference. The GPU reference
(`ReferenceGraphExecutorFactory::create(ReferenceExecutorType::GPU)` in
`dnn-providers/integration-tests/src/harness/`) takes device pointers; ask it with
`requiresDeviceMemory()` rather than assuming.

The same serialized bytes drive the reference and the engine. That is the invariant
worth exploiting: inspect, validate and execute the identical object.

### What the reference cannot do

Declined up front by both CPU and GPU references, per
`dnn-providers/integration-tests/README.md` ("What the reference executors cannot
verify"): paged KV (`page_table_k/v_tensor_uid`), varlen (`seq_len_q/kv_tensor_uid`),
ragged offsets, block-sparse masks (`block_mask_tensor_uid`), sink tokens
(`sink_token_tensor_uid`), dropout, FP8 descale, softmax statistics.

**CPU is not a fallback for any of these.** A decline is an unmeasured bucket. Under
the integration tests' `auto` verification mode a decline becomes a skip, and a skip
is aggregate-indistinguishable from a pass
(`notes/hipdnn/integration-tests-green-is-not-coverage.md`) — which is why this
harness must assert the reference ran rather than infer it from an exit code.

If the graph needs a declined feature: narrow the graph and state the narrowing,
bring an independently trusted reference and state its provenance, or record BLOCKED.
Never derive the reference from the kernel under test.

## Input parity

Both buffer sets must receive byte-identical inputs. The integration harness achieves
this by seeding once and visiting UIDs in **sorted** order so the draw sequence does
not depend on discovery order, then filling each bundle with the same per-UID seeds
(`dnn-providers/integration-tests/src/harness/input-init/FillInputs.cpp`). Restructuring
that loop has already produced a false kernel-defect report once
(`notes/hipdnn/harness-reference-input-parity.md`). Reproduce the pattern; do not
improve on it.

Fill values matter. Uniform random over a wide range will make an
exponential-and-normalize operation overflow and a comparison meaningless; the
integration harness carries per-op recipes (SDPA scale in `[0.1, 1.0]`, epsilon fixed
at `1e-5`, and so on) for that reason. State the range you used.

## Tolerance

Tolerance is per operation and per element type, resolved in
`dnn-providers/integration-tests/src/harness/tolerance/ToleranceResolver.hpp`
(`resolveTolerance(...)`, dispatching over Conv/Batchnorm/Matmul/MoE/Reduction/
RMSNorm/Pointwise/LayerNorm/SDPA; unknown ops fall back to `1e-3`). For a fused
graph, `MAX_ACROSS_NODES` — the loosest per-node tolerance in the graph — is the
conservative default; `OUTPUT_OP_TOLERANCE` is tighter.

Rules:

- State the tolerance **and where it came from**. A number chosen because it made the
  test pass is a defect report in disguise.
- The TOML per-test overrides exist to loosen a shipping engine's drift budget. They
  must never be applied to a hand-written kernel's first correctness proof.
- Infinities: `createAllCloseValidator` rejects matching infinities by default;
  `createAllCloseMatchingInfinitiesValidator` accepts them, which is correct for
  legitimately infinite outputs such as SDPA statistics rows over a fully masked
  query. Choose deliberately and say which.
- Where the arithmetic is exactly representable — small integers in floating point —
  compare exactly. Tolerance on an exact result hides real errors.

A worked provenance statement is in tree:
`samples/batchnorm/FusedBnInfDReluBnBwd.cpp:175-178` uses `4e-2` and says in a
comment exactly why — the reference does not split input and output dtypes for that
graph. That is the standard to meet: a number plus the reason it is not tighter. A
loosened tolerance with no named cause is an unreported defect.

## The three assertions

Each of these failures is otherwise indistinguishable from success:

| Assertion | How |
|---|---|
| The reference executed | check the executor's return/exception path explicitly; a declined or skipped reference fails the gate |
| Your kernel launched and wrote | fill outputs with a sentinel before launch (`fillWithSentinelValue()`), synchronize, and confirm they changed; check `hipGetLastError()` after the launch |
| Every output was compared | iterate the graph's non-virtual output UIDs from the tensor map, not a hand-written list |

Add a fourth where a multi-launch decomposition is used: assert the launch order and
that each intermediate was written before it was read, by sentinel-filling the
scratch too.

## Coverage to run

Run the shape set claimed, not one point of it, and include the boundaries the
specification implies:

- a non-contiguous stride pattern, since layout comes only from strides;
- a dimension of 1, and a non-tile-multiple dimension;
- the smallest and largest shapes in the claimed envelope;
- every dtype claimed — a `BFLOAT16` path that was never compiled is not supported;
- every architecture claimed, separately.

## Reporting

Per output, per shape, per dtype, per architecture: pass or fail, the tolerance and
its provenance, the max absolute and relative difference, and the device the run
happened on. Then the does-not-prove list: untested shapes, untested dtypes, untested
architectures, declined reference features, and the absence of any performance claim.

Report file paths and job identifiers for logs rather than pasting them. A harness
that passed proves this kernel computed these outputs within this tolerance on these
inputs on this device — nothing wider.
