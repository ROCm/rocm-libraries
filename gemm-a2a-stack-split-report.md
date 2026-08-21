# Proposed PR stack for `users/yiding12/gemm-a2a`

## Dependency graph

```text
current develop after rebase
├── 01-conversion-runtime-strides             independent sibling
└── 02-rocisa-sdma-instructions               independent prerequisite
    └── 03-fused-gemm-a2a-functional          functional, default-off feature
        └── 04-fused-a2a-dispatch-tuning       schedule and cache-policy tuning
```

Use a new prefix such as `users/yiding12/gemm-a2a-stack/`; the existing branch name is already a Git ref and cannot also be a parent directory for child refs.

## Proposed PRs

### 01 — conversion kernels honor runtime C/D strides

Category: defect fix. Suggested title: `AIHPBLAS-<new> fix(tensilelite): honor runtime C/D strides in conversion kernels`.

Move only `Tensile/KernelWriterConversion.py` and a focused regression test. This change replaces an existing `assert 0` for `UseInitialStridesCD` with generated macros for runtime C, D, and optional E strides. It is unrelated to A2A: the feature uses `KernelWriterAssembly`, not conversion kernels.

The test should fail on the base revision, then verify that generated code reads `arg.strideC*` and `arg.strideD*` for index 0 and that a following conversion kernel does not inherit those macros. Extend `Tensile/Tests/unit/test_KernelWriterConversion_64bit_offset.py` or add a nearby focused test. Because this fixes a defect, the test must also pass in a shared hipBLASLt lane for the affected architecture.

Paste-ready summary: “Conversion kernels now read the caller-provided first C and D strides when `UseInitialStridesCD` is enabled. The previous generator stopped with an assertion. The regression test checks both the generated stride expressions and macro cleanup between kernels.”

### 02 — rocISA models the instructions used by GPU-issued SDMA

Category: groundwork. Suggested title: `ROCM-<child> feat(rocisa): model scalar atomics used by SDMA producers`.

Keep the five rocISA implementation files together: `common.hpp`, `common.cpp`, `mem.hpp`, and `mem.cpp`, plus focused tests under `rocisa/test/`. This PR adds `s_bfm_b64`, returned `global_atomic_add_u32`, and scalar `cmpswap_x2` and `umax_x2`. It also restructures the shared scalar-atomic base class.

This is a real prerequisite. `GlobalWriteBatch.py` imports the returned global atomic and `s_bfm_b64` at module load. `Signature.py` imports `SdmaRingEmitter`, which imports the new scalar atomics. Landing the generator first can therefore break ordinary, non-fused generator imports.

Add tests that construct, clone, render, and lower each new instruction. Keep a regression for existing `SAtomicInc` and `SAtomicDec` behavior after the base-class change. Run the rocISA Linux and Windows package lanes, plus an assembly-generation check for gfx950. The current branch changes no rocISA test file.

Paste-ready summary: “rocISA can now emit the scalar atomic instructions required to reserve and publish work in a GPU-owned DMA queue. Tests verify the generated operands, modifiers, cloning behavior, and existing scalar-atomic output.”

### 03 — functional fused GEMM+A2A path, disabled by default

Category: new device path and build/runtime integration. Suggested title: `ROCM-<child> feat(tensilelite): add a default-off fused GEMM+A2A path for gfx950`.

This PR must keep the complete kernel-to-launcher change together:

- the `FusedGemmA2A` parameter, rejection rules, signature tail, SDMA packet/ring emitters, and generated epilogue;
- the mirrored C++ argument packer, counter allocation, queue wrapper, CMake option, and `--fused-a2a` client mode;
- the three saved expected-result updates and helper default caused by registering the new parameter; and
- a tracked ahead-of-time (AOT) gfx950 fixture with `FusedGemmA2A: 1`.

Keep `FusedGemmA2A=0`, `TENSILELITE_ENABLE_SDMA_A2A=OFF`, and `--fused-a2a=false` as normal defaults. Those settings preserve ordinary behavior. They do not replace tests for the enabled path.

Build this first version for correctness, not overlap. Remove the currently inert `counter2` rather than carrying it forward: both branches after its comparison immediately reach the same label, and the source comment calls the path inert. Keep only the cursor pairs, the global completion counter, the per-token counter, and the sentinel. Also mark every fused D store `sc1`/SLC in this baseline. That may bypass L2 for local tiles, but it guarantees that the DMA engine reads completed data from memory.

Before this PR is ready, add four guards that the current branch lacks:

1. Reject a selected solution that was not generated with `FusedGemmA2A=1` before appending the private arguments.
2. Reject unsupported architectures during generation. Current packet encoding is documented for gfx9xx/gfx95x, while the ticket and evidence point to gfx950.
3. Reject batched GEMM during generation, or include the batch dimension in the completion count. The client rejects batching, but the generated count currently uses only work-group dimensions 0 and 1.
4. Add a test-only enabled configuration so AOT generation, assembly, and the host argument layout are reproducible.

Required evidence is a focused solution-validation test, Python/C++ argument-layout tests, packet and ring arithmetic tests, builds with SDMA both off and on, the full TensileLite unit lane, and a shared 4-GPU gfx950 run. The device run must validate received shards and the local tail, include a partial token tile, and send enough packet pairs to wrap the 256 KiB ring. The saved expected-result changes must be recorded only for their affected nodes and classified in the PR; add an ADR if review treats the changed behavior as non-obvious.

### 04 — dispatch outbound tiles first and narrow the cache bypass

Category: kernel tuning. Suggested title: `ROCM-<child> perf(tensilelite): schedule fused A2A transfers before local tiles`.

Move the work-group remap, the two-pass PUSH/local store emission, and `fusedA2ADispatchMode` here. This PR changes the functional baseline in two ways: it schedules peer-bound tiles before local-only tiles, and it applies `sc1` only to peer-bound stores rather than every fused store.

This is a safe later boundary because `FusedA2AWgRemap` is bijective and its source states that correctness does not depend on dispatch order. The functional PR already launches and validates the feature. This PR only changes when tiles are issued and which stores bypass L2.

Add a reference-model test that proves the remap is bijective at `AM=0`, `AM=MT0`, `AM=M`, segment boundaries, and multiple token-tile counts. Re-run the 4-GPU numerical/race test. If the PR claims lower latency, report before/after p50 and p90 on the same topology, shapes, warmup count, iteration count, and drain setting.

Paste-ready summary: “The fused A2A kernel now issues peer-bound tiles before local-only tiles. It also bypasses L2 only for stores that the DMA engine will read. The remap test proves that every matrix tile remains covered once, and the gfx950 run verifies that the reordered kernel returns the same data.”

## Changes that must not become separate PRs

- Do not split the parameter and signature from the client argument packer. One defines a 408-byte private tail; the other supplies it.
- Do not split the packet/ring emitters from the functional A2A PR. The ring code owns the A2A peer-field and counter layout and has no second user.
- Do not split the ROCm queue setup or its CMake workaround from the client harness. Neither has a useful caller by itself.
- Do not make the saved expected-result changes their own PR. They only record a default parameter and do not test the feature.

## Merge and review requirements

Rebase before creating the stack. The locally available `origin/develop` is eight commits ahead of merge-base and overlaps `GlobalParameters.py`, `ValidParameters.py`, `KernelWriter.py`, `KernelWriterAssembly.py`, `Solution.py`, two saved expected-result files, and `client/main.cpp`. The `KernelWriter*` overlap triggers the hipBLASLt rule requiring a rebase and re-run.

PRs 03 and 04 modify shared generator and component code. They need two hipBLASLt code-owner approvals after local review. Their CMake changes also require the build-infrastructure owners listed in `.github/CODEOWNERS`.

Use a tracker key in every title. Link the side defect fix to its own issue. Link the A2A stack to a scoped child of `ROCM-27524`, not only to the broad parent. Do not use a known-bug waiver for missing multi-GPU coverage.

The current PR description should be replaced, not shortened. It relies on commit history, uses undefined GPU terms, says every commit is self-contained, and claims a fused gfx950 generation path that is not present in the tracked test inputs. The four summaries above are the starting point for stack-specific descriptions.
