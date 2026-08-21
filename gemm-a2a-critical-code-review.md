# Critical code review: `users/yiding12/gemm-a2a`

## Scope and result

This review covers the TensileLite changes from `54dcb4f36c` to `c6a0236269`. The branch adds a fused GEMM plus all-to-all (A2A) path that uses the GPU direct-memory-access engine (SDMA) to copy output shards between GPUs.

Result: **CHANGES REQUESTED**. The branch contains several launch paths that can report success without performing A2A, reuse memory while SDMA may still write it, or copy the wrong host buffer size. The review did not modify product code or run a GPU workload.

## Blocking findings

### 1. The harness can run a normal GEMM and report A2A success

**Locations:**

- `projects/hipblaslt/tensilelite/client/src/FusedA2AClient.cpp:90-105`
- `projects/hipblaslt/tensilelite/client/src/FusedA2AClient.cpp:535-567`
- `projects/hipblaslt/tensilelite/Tensile/Components/Signature.py:425-442`
- `projects/hipblaslt/tensilelite/client/src/FusedA2AClient.cpp:707-712`
- `projects/hipblaslt/tensilelite/client/src/FusedA2AClient.cpp:953-958`

`runFusedA2A()` selects an ordinary solution, then unconditionally appends the A2A arguments. The generator adds those arguments only when `FusedGemmA2A=1`. A normal code object can therefore ignore the appended data.

With `--fused-a2a-validate=0`, the client skips the receive-buffer comparison and returns `0` after a clean kernel exit. It can claim success even though no A2A packets were issued.

Require a durable marker on the selected solution or code object and reject it before allocating queues or appending arguments unless it was generated with `FusedGemmA2A=1`. Keep that check active when numeric validation is disabled.

### 2. The host accepts GEMM problems whose input buffers it cannot represent

**Locations:**

- `projects/hipblaslt/tensilelite/client/src/FusedA2AClient.cpp:76-88`
- `projects/hipblaslt/tensilelite/client/src/FusedA2AClient.cpp:212-215`
- `projects/hipblaslt/tensilelite/client/src/FusedA2AClient.cpp:279-301`
- `projects/hipblaslt/tensilelite/client/src/FusedA2AClient.cpp:370-376`
- `projects/hipblaslt/tensilelite/client/src/FusedA2AClient.cpp:526-533`
- `projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Solution.py:1805-1823`

The harness accepts any plain GEMM and allocates host A and B vectors as `BFloat16`. It then copies `problem->a().totalAllocatedBytes()` and `problem->b().totalAllocatedBytes()`. The solution check requires only BF16 D, not BF16 A and B. A valid BF16-D/F32-A or BF16-D/F32-B problem can make `hipMemcpy` read beyond the host vector.

The same path supplies only A, B, C, D, alpha, and beta to `ContractionInputs`. The fused validation does not reject bias, E, scale, gate-residual, workspace, or other optional operands that can require additional inputs.

Restrict the harness and generator to the exact A/B/D types and epilogue features that it initializes, or implement type-aware initialization and every required input pointer.

### 3. `counter2` is dead protocol state but still emits an atomic and changes allocation layout

**Locations:**

- `projects/hipblaslt/tensilelite/Tensile/Components/GlobalWriteBatch.py:2988-3025`
- `projects/hipblaslt/tensilelite/Tensile/Components/Signature.py:64-69`
- `projects/hipblaslt/tensilelite/client/include/FusedA2ACounterSentinel.hpp:40-64`

The generated code increments `counter2`, waits for the result, compares it, and branches to `skipReleaseLabel`. That label is the next emitted label, so both comparison outcomes take the same path. The comment at `GlobalWriteBatch.py:3017-3019` explicitly calls the branch inert.

The unused state costs an atomic and reserves a mirrored region in the kernel and host counter layouts. Delete `counter2`, its offsets, allocation bytes, and comments together. If it is meant to decide a protocol transition, implement that transition and add a test that distinguishes the two outcomes.

### 4. New rocISA instruction models have no direct regression tests

**Locations:**

- `projects/hipblaslt/tensilelite/rocisa/rocisa/include/instruction/common.hpp:1409-1435`
- `projects/hipblaslt/tensilelite/rocisa/rocisa/include/instruction/mem.hpp:2318-2361`
- `projects/hipblaslt/tensilelite/rocisa/rocisa/include/instruction/mem.hpp:3643-3717`

The branch adds `SBfmB64`, `GlobalAtomicAddU32`, `SAtomicCmpswapX2`, and `SAtomicUmaxX2`, and changes the shared scalar-atomic base class. No changed rocISA test covers their rendered assembly, Python bindings, cloning, operands, or lowering through StinkyTofu.

Add focused rocISA tests for each instruction and a regression for existing `SAtomicInc` and `SAtomicDec` output. Run an assembler/lowering test on gfx950, the target architecture for the new producer.

## Important findings

### 5. Code generation admits batching although the completion count is not batch-aware

**Locations:**

- `projects/hipblaslt/tensilelite/client/src/FusedA2AClient.cpp:126-131`
- `projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Solution.py:1805-1823`
- `projects/hipblaslt/tensilelite/Tensile/KernelWriterAssembly.py:3396-3404`

The harness rejects batched GEMM, but solution generation does not. The last-work-group election counts only `NumWorkGroups0 * NumWorkGroups1`; it does not include the third work-group dimension used for batches. A generated batched A2A kernel can elect DRAIN too early.

Reject batched problems in `Solution.py` until the counters, flags, and DRAIN election include the batch dimension.

### 6. The work-group remap exceeds its divider's documented range on valid large shapes

**Locations:**

- `projects/hipblaslt/tensilelite/Tensile/Components/WorkGroupMappingAlgos.py:1167-1193`
- `projects/hipblaslt/tensilelite/rocisa/rocisa/include/functions/f_math.hpp:657-659`
- `projects/hipblaslt/tensilelite/client/src/FusedA2AClient.cpp:143-196`

The remap forms a linear work-group index and divides it with `scalarUInt24DivideAndRemainder`. That helper documents accuracy only for dividends up to `2^24` and divisors up to `2^16`. The fused shape checks do not bound either the grid product or the M-tile count.

Use the 32-bit division helper, or reject shapes beyond the documented range. Add mapping tests at `AM=0`, `AM=M`, the PUSH/local boundary, and large valid dimensions.

### 7. `--device-idx` does not select the ranks that the fused harness launches

**Locations:**

- `projects/hipblaslt/tensilelite/client/main.cpp:685-699`
- `projects/hipblaslt/tensilelite/client/src/FusedA2AClient.cpp:352-379`
- `projects/hipblaslt/tensilelite/client/src/FusedA2AClient.cpp:404-445`
- `projects/hipblaslt/tensilelite/client/src/FusedA2AClient.cpp:522-524`

Main builds its hardware description from `--device-idx`, but the fused path always uses visible HIP devices `0` through `W-1`. It then uses one hardware architecture name when initializing adapters for every rank. A nonzero device index or a mixed visible-device set can select a code object for one GPU and launch it on another.

Reject nonzero `--device-idx` until the harness accepts an explicit rank-to-device map. Before loading the code object, verify that every selected device has the same supported architecture.

### 8. The private kernel argument layout is maintained in four independent places

**Locations:**

- `projects/hipblaslt/tensilelite/Tensile/Components/Signature.py:42-101`
- `projects/hipblaslt/tensilelite/Tensile/Components/SdmaRingEmitter.py:27-44`
- `projects/hipblaslt/tensilelite/client/include/FusedA2AKernArg.hpp:22-106`
- `projects/hipblaslt/tensilelite/client/include/FusedA2ACounterSentinel.hpp:40-64`

The maximum rank count, peer-field order, counter offsets, and size calculations are copied across Python and C++. The host size assertion checks only the host calculation. It cannot prove that generated kernel offsets agree with the host.

Generate both representations from one schema, or add a test that compares the generated kernel metadata and every host offset. This is especially important because a field-order error can make a valid pointer look like a queue control address.

## Spec

The tracker for this branch, `ROCM-27524`, asks for both GEMM+A2A and A2A+GEMM plus a measured improvement against a four-rank PyTorch baseline. This branch implements only output-side A2A: `Signature.py:35-40` places fusion in the D-store epilogue, and `ValidParameters.py:943-947` describes an output PUSH. The harness reports only fused-path latency at `FusedA2AClient.cpp:849-866`; it has no baseline launch or pass/fail performance criterion.

Either add the input-side operation and target benchmark, or narrow the work item and link this branch to a GEMM+A2A-only child issue.

## Standards

The review contains 4 blocking and 4 important code or test findings that were not already raised in an inline GitHub review. The most urgent defects are the unverified selected kernel and typed host-buffer mismatch. The most urgent specification gap is that the branch does not meet the broader tracker requirement for A2A+GEMM or a baseline performance result.
