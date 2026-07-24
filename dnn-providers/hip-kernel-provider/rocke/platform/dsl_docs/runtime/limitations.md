# Runtime, Compiler, And DSL Limitations

This page is blunt. It lists what the DSL does not currently handle generally, plus traps that look like compiler bugs but are not.

## Scope

`rocke` is a high-performance AMDGPU kernel DSL, not a full Python-to-GPU compiler. It expects kernel authors to understand:

- GPU grid / block decomposition;
- wave lane behavior (wave64 on gfx90a/gfx942/gfx950; wave32 on
  gfx11-generic/gfx1151/gfx1201/gfx1250);
- LDS allocation, layout, and bank-conflict semantics;
- MFMA operand and accumulator layouts;
- explicit synchronization (waitcnt, barriers);
- dtype and numeric tolerance;
- ROCm / AMDGPU code-object constraints.

It is strongest for kernels whose performance depends on explicit CK Tile-style patterns.

## Architecture

Current default and byte-identity baseline: `amdgcn-amd-amdhsa--gfx950`.
`known_arches()` currently lists gfx90a, gfx942, gfx950, gfx11-generic,
gfx1151, gfx1201, and gfx1250. This is the platform target catalog, not a claim
that every kernel family accepts every target; family validators own that
contract. Backends are registered in `core/isa/backend.py` (`backend_for`).

Risk areas when changing architecture:

- MFMA / WMMA intrinsic availability — K-packed MFMA atoms
  (`f16_16x16x32`, `f16_32x32x16`) are gfx950-only; the selected target's
  catalog and backend determine whether a bf16 MFMA or WMMA form is legal.
  Wave width alone does not select the atom.
- `s_waitcnt` encoding (vmcnt bit layout differs by target family; gfx11/gfx12 use a different field layout than gfx9/10).
- LDS bank-conflict expectations.
- COMGR target ISA behavior.
- Wave size comes from `ArchTarget`: gfx90a/gfx942/gfx950 are wave64;
  gfx11-generic/gfx1151/gfx1201/gfx1250 are wave32.

Do not assume a kernel tuned for one target is optimal — or even valid — on another.

## Type Coverage

The type vocabulary includes `f16`, `bf16`, `f32`, `fp8e4m3`, `bf8e5m2`, and integer types. Lowering and instances do not support every combination.

Current production center:

- `f16` storage with `f32` accumulation (every shipped GEMM / conv hero path);
- `bf16` storage with `f32` accumulation (attention `bf16_decode_d128_b64` scenario);
- selected fp8 / bf8 / i8 paths through `helpers/quant.py` (attention FP8 K/V cache is the active extension);
- one-element conversion ops (`cvt_*`) are scalar; vectorized packed-conversion (`cvt.pk.fp8.f32` for two f32 in one instruction) is exposed only via the low-byte extract path.

Coverage caveats:

- f32 math intrinsics (`exp2`, `sqrt`, `rsqrt`, `tanh`) are better covered than bf16 / fp8 math.
- `arith.constant_vec` is mostly zero-vector oriented.
- Every new dtype needs load / store / cast / MFMA / manifest / reference coverage.
- HIP debug lowering can lag LLVM lowering in dtype robustness.

## Lowering Coverage

Every IR op must have explicit production lowering. Unsupported ops raise at lowering time.

There are two interchangeable lowering engines reached through the same authoring API: the native Python lowerer (`core/lower_llvm.py`) and a C++ engine (the `cpp/` port, reached via the `rocke_engine` binding). They emit byte-identical LLVM IR across every family. The active engine is selected by `core/backend.py::resolve_backend` (`ROCKE_BACKEND`, default `cpp`, falling back to the Python lowerer when the binding is not built); see `ir_lowering/` and `development/engine_parity.md`. The notes below apply to both engines.

HIP debug lowering is narrower than LLVM production lowering. CK Tile spec emission is narrower still and operates from selected specs, not `KernelDef`.

Implications:

```text
"It prints clean in MLIR-style IR" does not prove the LLVM lowering succeeds.
"It lowers to LLVM" does not prove the HIP debug backend handles it.
"It has a spec dataclass" does not prove CK Tile parity emission works.
```

## MLIR Is A Non-Goal

The textual IR is MLIR-style for humans. There is no MLIR pipeline in the current production path.

Do not expect:

- parsing printed IR back into the compiler;
- MLIR dialect verification;
- MLIR canonicalization;
- MLIR GPU lowering;
- automatic tensorization from high-level loops.

The DSL owns its own IR and lowering.

## Autotuning Limits

`helpers/autotune.py::Autotuner` and the `sweep_bench` driver build and benchmark many configs, but they do not guarantee a global optimum.

Autotuning still needs:

- correct config search space (most "missing perf" is a missing config);
- representative shapes (not just the hero shape);
- stable benchmark methodology (median + spread; discard first run unless measuring cold start);
- correctness checks before timing claims;
- ISA / resource inspection to attribute the win;
- awareness of launch overhead and cache behavior.

A bad search space produces a bad winner quickly.

## Persistent Kernels And Split-K

Some primitives exist but are not shipped as polished instances:

- Persistent grouped GEMM is not shipped (`GroupedGemmLauncher` does multi-launch grouped GEMM).
- Split-K is not a universal `TileSpec` field. The atomic primitive (`global_atomic_add_f32`) exists, but the dispatcher / scheduler / reduction policy is not generalized.
- Cross-kernel fusion beyond `matmul + epilogue` is documented (`helpers/fusion_*`) but the pattern table is small.

Treat these as extension work, not hidden features.

## Async Loader Limits

`AsyncTileLoader` is powerful but constrained:

- `dwords in {1, 3, 4}` (no 2-dword form on this LLVM target).
- LDS writes are lane-contiguous.
- Destination base must be uniform within a wave (the loader uses `to_sgpr_u32` to hoist it).
- Consumers must call `b.s_waitcnt(vmcnt=0)` before reading the LDS.
- Swizzled destination pointers are not allowed; put swizzle in the consumer read arithmetic (`TransposeLdsReader`).

Async is not automatically faster. It can regress if it increases scheduling complexity, drops occupancy via extra LDS, or forces a worse consumer LDS layout.

## Barrier Hazards

Barriers must be reached consistently by participating threads / waves.

Danger patterns:

- barrier inside a divergent `scf.if`;
- `sync_half_block` without strict wave partitioning;
- early return before a later barrier;
- tail predicates controlling synchronization instead of memory-op validity;
- one pipeline stage skipped by some lanes.

Most intermittent correctness bugs in tiled kernels should trigger a barrier / waitcnt review first.

## Memory Safety Limits

The DSL provides safe patterns; it does not make all memory accesses safe by default.

Safe patterns:

- buffer-resource path with OOB sentinel (`select(valid, off_bytes, INT32_MAX)`);
- clamped index before global pointer load (`masked_global_load`);
- descriptor validity threaded through loader / epilogue helpers;
- tail store masks.

Unsafe patterns:

- raw pointer load on an invalid index followed by `select` (the load can fault before the select matters).
- vector load that crosses from valid into invalid memory (the AMDGPU buffer load is per-vector, not per-lane bound).
- treating element offsets as byte offsets (forget the `* 2` for fp16);
- assuming padding bytes exist in user tensors;
- passing a byte-count ABI in elements.

## Numerical Limits

Changing atom shape, K packing, or accumulation order can change numeric results.

For fp16 inputs with f32 accumulation over O(100) terms (e.g. conv with `K=R*S*C ~ 576`), expected `max_abs vs f32 reference ~ 1e-5` and `mean_abs ~ 1e-6`. Errors two orders of magnitude larger almost always indicate structural bugs:

- wrong lane packing (K-packed atom misuse);
- stale accumulator (forgot to reset between tiles);
- missing mask in tail handling;
- invalid LDS read crossing into another wave's slot;
- wrong epilogue lane-to-output mapping.

Do not dismiss large error as "fp16 noise" without evidence.

## Runtime Assumptions

Runtime launchers assume:

- ROCm HIP runtime available through the dynamic linker.
- `libamd_comgr` available for compile through the dynamic linker.
- torch stream resolution is valid for torch launch flows.
- packed-args ctypes buffers and tensors remain alive until launch completion (`Runtime._pending_args` queue handles this when launches go through `KernelLauncher`).
- workspace lifetimes are managed for pipelines (use `WorkspacePool` for stage intermediates).

CI / dev machines without matching ROCm libraries can still build docs, import `rocke`, run the static unit suite, and inspect Python code. They cannot run end-to-end HSACO compile + launch.

## Multi-GPU and Collectives

Each rocKE kernel launch targets one current HIP device; the IR and kernel ABI
do not contain collectives. The runtime does have the device plumbing needed to
respect a host-selected device:

- `runtime/hip_module.py` binds `hipSetDevice`, `hipGetDevice`, and
  `hipGetDeviceCount`; `_ensure_hip_init()` preserves an already-selected
  device and selects device 0 only when none is current.
- `runtime/torch_interop.py::resolve_stream(stream, device=None)` resolves the
  current torch stream for the requested device.
- `benchmark/gemm/tests/test_fp16_rcr_multigpu.py` is a guarded two-or-more-GPU
  smoke test for compiling a sweep once and running independent GEMM variants
  across visible gfx950 GPU lanes.

That support is independent per-device execution, not a collective runtime.
There are no RCCL/NCCL IR operations, fused communication kernels, peer-access
helpers, or HIP IPC helpers in the platform. Applications may compose rocKE
kernels with `torch.distributed`/RCCL outside the kernel, but those collectives
are owned and synchronized by the application. Do not pass a pointer owned by
another device to a launcher or infer collective correctness from the
multi-GPU GEMM sweep smoke test.

## Benchmark Methodology

Benchmark numbers are only meaningful when methodology is labeled. Always label:

- per-launch vs graph / pipeline amortized;
- warm vs cold module / cache;
- attempts and warmups;
- stream / timer mechanism (`time_launches` uses HIP events on the resolved stream);
- GPU and ROCm version;
- correctness status;
- shape and dtype;
- baseline implementation (which CK Tile config? which Triton kernel? which torch ref?).

Single-run results are especially risky: frequency state, cache warmth, module
load, and execution order can create bimodal or cold-start outliers. Use
interleaved repeated measurements and report the distribution rather than
copying an old point estimate into a current support claim.

## Extension Guidelines

When adding a feature:

1. Start with one narrow, correct instance.
2. Add a helper abstraction only after the repetition is real.
3. Keep unsupported combinations rejected explicitly in `is_valid_spec`.
4. Make generated IR / LLVM / ISA inspectable.
5. Add docs in this tree if it's part of the public surface.
6. Add a manifest / test / reference path.
7. Benchmark with stable methodology before claiming speed.

See `development/extending.md` for recipes.

## Known Gremlins (Symptom → Likely Cause)

- Async code that only fails under benchmark pressure → missing `s_waitcnt(vmcnt=0)` before LDS read; or workspace tensor reallocated between iterations.
- Output close but not close enough → MFMA lane packing (K-packed atom wrong K slice); or epilogue lane-to-output mapping doesn't match the atom.
- Kernel validates for aligned hero shapes only → invalid pointer load in tail; switch to buffer-resource path with OOB sentinel.
- Fast first benchmark number, slow subsequent → cold cache / clock luck; report median + spread, discard first.
- "Optimized" kernel slower than baseline → increased VGPR / LDS dropped occupancy; check `analyze_hsaco().resources`.
- HIP debug output looks fine but LLVM build fails → HIP backend dispatches a different op handler; LLVM is canonical.
- Manifest works but launcher misbehaves → pack order / element vs byte mismatch; print the packed bytes and compare to the expected `struct.pack` layout.
- Attention max_abs spikes on a specific scenario → segment count too low/high for that scenario; check `select_3d_config` choice.
