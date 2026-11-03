# Backend Details And Edge Cases

This page collects backend-specific facts that matter when extending the DSL, debugging generated code, or porting to a new target. Everything below is verified against `core/lower_llvm.py`, `core/lower_hip.py`, `core/lower_cktile.py`, and the static unit suite.

## Two Lowering Engines

There are two interchangeable lowering engines that emit byte-identical AMDGPU LLVM IR for the same kernel: the native Python lowerer (`core/lower_llvm.py`) and a peer C++ engine (under `cpp/`, reached through the `rocke_engine` Python extension). The active engine is chosen by `core/backend.py::resolve_backend()` — precedence is the explicit `backend=` argument, then the `ROCKE_BACKEND` environment variable (`python` / `cpp` / `both`), then the package default, which is now **`cpp`**. The `cpp` path falls back to the native Python lowerer automatically if the `rocke_engine` extension is not built, and `both` runs both engines and asserts they agree (the differential gate). The Python lowerer remains the differential oracle. The descriptions of LLVM lowering below apply to both engines; this page documents the lowering contract, not a single implementation.

## LLVM Backend Is Canonical

`core/lower_llvm.py` is the reference description of the production lowering contract. If LLVM lowering and HIP debug lowering differ, treat LLVM lowering as the runtime path.

The backend emits AMDGPU LLVM IR text directly. Benefits:

- very fast iteration vs hipcc / template instantiation (typical end-to-end build under 30 ms warm);
- primitives map closely to AMDGPU intrinsics;
- generated text can be inspected, disassembled, and diffed;
- no external compiler frontend needed.

Costs (you own them):

- every DSL op must have explicit lowering support;
- architecture details (`s_waitcnt` encoding, intrinsic signatures, `make.buffer.rsrc` flags) are owned by this repo;
- LLVM type coverage is hand-maintained;
- backend bugs are less likely to be caught by a higher-level frontend.

## Target Triple and Datalayout

```text
target triple = "amdgcn-amd-amdhsa"
target datalayout = (clang-emitted gfx950 string; see _DATALAYOUT in core/lower_llvm.py)
```

`_DATALAYOUT` was copied verbatim from `clang -target amdgcn-amd-amdhsa -mcpu=gfx950 -emit-llvm -S`. The datalayout/triple are now served per target by an ISA backend (`core/isa/backend.py::backend_for(arch)`), keyed off the chosen gfx; the supported CDNA and RDNA targets share this clang-verified layout/triple on the ROCm releases targeted, but other ISA details (buffer-resource word3, `s_waitcnt` encoding, MFMA vs WMMA matrix ops) diverge per backend. If you bump the ROCm version or move to a different target, regenerate the datalayout. Mismatched datalayouts produce subtle codegen drift; the comgr stage rarely flags them.

The default `compile_kernel` ISA is `amdgcn-amd-amdhsa--gfx950`, and gfx950 (MI355X) is the default-target / byte-identical baseline. The lowerer also supports gfx942 (CDNA, wave64) and the RDNA targets gfx1151 (RDNA3/3.5) and gfx1201 (RDNA4) — wave32 WMMA, including attention. The CDNA targets use MFMA atoms where the chosen shape exists (16x16x16, 32x32x8, 4x4x4); the K-packed atoms (16x16x32, 32x32x16) are gfx950-only. RDNA targets use WMMA atoms (`core/isa/backend.py`) instead of MFMA.

## LLVM Intrinsic Flavor

rocKE separates a **compiler lane** from an **emission profile**. A lane records a qualified compiler major; a profile records the structural LLVM IR that rocKE emits: datalayout generation, declaration ABI, call shapes, and admitted target operations. The qualified lanes are `llvm20`, `llvm22`, `llvm23`, and `llvm24`, enumerated by `LLVM_FLAVORS` in `core/llvm_profile.py` and mirrored by the neutral C++ profile registry in `cpp/core/llvm_profile.cpp`. LLVM24 retains LLVM23's datalayout and target-op surface, but qualification found one structural delta: named-barrier pointer declarations and calls use addrspace(15) instead of addrspace(3).

The profiles capture these differences:

- LLVM20 uses the plain-`p8` datalayout and legacy buffer-resource/FP8 MFMA declarations.
- LLVM22 uses indexed `p8`, `make.buffer.rsrc.p8.p1` with an `i64` count, scalar-`i64` FP8/BF8 MFMA operands, and the eleven-argument scaled-MFMA form.
- LLVM23/24 add `m:e` and address spaces `p10`–`p15`, use the nine-argument scaled-MFMA form, and admit the LLVM23-generation target-op surface.
- LLVM24 casts LDS named-barrier pointers to addrspace(15), matching that compiler's intrinsic declarations; LLVM23 keeps addrspace(3).

The compiler resolver never infers structural compatibility from a lane name. Emission decisions consult the selected profile rather than exact `llvm23` comparisons. Python and C++ must emit byte-identical IR for the same kernel, target, and lane.

Detection (`core/lower_llvm.py::_detect_llvm_flavor`):

1. A supported `ROCKE_LLVM_FLAVOR` value selects offline emission without loading COMGR.
2. Otherwise COMGR's loaded compiler maps as follows: LLVM <=20 → `llvm20`, 21–22 → `llvm22`, 23 → `llvm23`, and 24 → `llvm24`.
3. A compiler newer than the latest qualified lane fails instead of silently inheriting the newest profile. Missing compiler evidence also fails. Offline callers can select an explicit qualified lane.

Resolution is lazy and compiler evidence is retained with the loaded COMGR handle. Python-driven C++ lowering receives Python's resolved lane explicitly. Standalone native AUTO applies the same mapping and unknown-value failures. A native caller that owns compilation should pass the lane matching that compiler. See [Compiler detection and provenance](../reference/env_flags.md#compiler-detection-and-provenance).

**Compile-time guard.** `runtime/comgr.py::build_hsaco_from_llvm_ir` compares the recognized IR `p8` generation with the profile selected from loaded compiler evidence. `p8:128:128:128:48` is `P8_INDEXED` (llvm22/llvm23/llvm24); `p8:128:128` is `P8_PLAIN` (llvm20). This catches a known ABI-generation mismatch but does not replace real-toolchain validity and datalayout-drift checks.

When adding a lane, first characterize its real compiler output. Add a lane only for qualification; add a distinct profile only for a structural IR difference. Put declaration and call-shape decisions in the profile instead of branching on a lane name. Python/C++ profile tests, the real-toolchain drift guard, emitted-IR validity, and byte identity cover different contracts and remain independently required.

Goldens store one sub-document per distinct emission profile. A lane in `aliases` is still lowered and compared, but reuses a profile-identical baseline instead of duplicating thousands of hashes. LLVM24 has its own baseline because its named-barrier call shape differs from LLVM23.

## Wave Size

The CDNA targets (gfx942, gfx950) are wave64: `MfmaAtom.lane_to_output`, `lane_id`, `ds_bpermute` addressing, and the loader/epilogue helpers assume wave64 there. The RDNA targets (gfx1151, gfx1201) are wave32 and use the WMMA atoms (`wave_size=32` in `helpers/atoms.py`); helpers that span both — e.g. the attention softmax wave-reduce (`helpers/attention.py`) — are parameterized by `wave_size`. Pick lane mappings for the wave size of the chosen target.

## LLVM Type Coverage

`_llvm_type` covers the types used by current kernels, not every type representable by `Type`.

Verified mappings:

- pointers: `ptr addrspace(1)` (`global`), `ptr addrspace(3)` (`lds`), `ptr` otherwise.
- vectors: `<N x elem>`.
- SmemType: lowers to `ptr addrspace(3)` for value-level usage.
- scalars: `i1`, `i8` (also `u8`), `i16`, `i32`, `i64`, `f16`, `bf16`, `f32`, `f64`. (FP8E4M3 / BF8E5M2 lower through the convert intrinsics rather than as first-class load/store types.)

To add a dtype, follow [Adding a dtype](../development/extending.md#54-adding-a-dtype).

## Address Spaces

```text
global  -> addrspace(1)
lds     -> addrspace(3)
buffer  -> addrspace(8)   (from llvm.amdgcn.make.buffer.rsrc; see
                           "LLVM Intrinsic Flavor" above for the LLVM
                           20 vs 21+ signature split)
```

Buffer-resource operations are modeled as AMDGPU buffer descriptors rather than normal pointer GEPs. They are essential for OOB-safe access in conv, attention, tails, and epilogues. The DWORD3 ("word3") format/OOB-select bits are ISA-specific: the CDNA value `0x00027000` is not binary-compatible with RDNA, so the RDNA backends (`Gfx11RdnaBackend` / `Gfx12RdnaBackend` in `core/isa/backend.py`) override it with the gfx10/11/12 word3 (`0x31014000`). Each ISA backend exposes the right value via `buffer_rsrc_word3`.

The canonical masked access pattern is:

```text
off_bytes = off_elems * sizeof(elem)
safe = select(valid, off_bytes, INT32_MAX)
buffer_load(resource, voffset=safe, soffset=0)
```

`INT32_MAX = 2147483647 = (1 << 31) - 1` is the default `oob_sentinel` everywhere (`AsyncTileLoader.issue`, `CoalescedTileLoader.load`, and most epilogues). Replacing it with `0` will fault when `num_bytes` is small or when DW3 is missing the bounds-check flags.

## Waitcnt Details

Two wait counters matter:

- LDS operations are tied to LGKM (a 4-bit counter on gfx950 — values 0..15).
- VMEM operations (including `raw.ptr.buffer.load.lds`) are tied to VMEM (six bits split across `[3:0]` and `[15:14]`).

Op semantics:

- `b.sync()` emits `s_waitcnt(vmcnt=0, lgkmcnt=0)` + `s.barrier`. Heavy but safe.
- `b.sync_lds_only()` emits `s_waitcnt(lgkmcnt=0)` + `s.barrier`. Useful only when the surrounding schedule knows what VMEM may remain in flight.
- `b.s_waitcnt(vmcnt=N, lgkmcnt=M)` directly emits the encoded immarg. Test `test_s_waitcnt_encodes_extended_vmcnt_without_wrapping` pins the gfx950 encoding for `vmcnt=16, lgkmcnt=16` to `i32 20336`.

`AsyncTileLoader` + `raw_ptr_buffer_load_lds` require VMEM wait before consumption:

```text
issue async loads (one or more passes)
b.s_waitcnt(vmcnt=0)
b.sync()        # or sync_lds_only() if VMEM is intentionally still pending
read LDS
```

A missing VMEM wait is a classic intermittent correctness bug.

## Structured Control Flow Lowering

`scf.for` and `scf.for_iter` lower to explicit basic blocks (`entry -> header -> body -> latch -> exit`) plus phi nodes for the induction variable and every loop-carried value. `scf.yield` operands are recorded by the body region; the latch block emits the IV increment, back-edge, and the phi inputs.

Risk areas:

- mismatch between the number of `init_vars` to `scf_for_iter` and the operand count of `scf_yield`;
- using `static_for` when a runtime dimension is needed;
- excessive static unrolling causing very large LLVM IR and slow comgr time;
- placing barriers inside a divergent `scf.if`.

When in doubt, inspect the LLVM text via `art.llvm_text` and the generated ISA via `analyze_hsaco`.

## HIP Debug Backend

`core/lower_hip.py` is for human-readable inspection. It emits:

- HIP typedefs for vector types;
- builtin shims for intrinsics clang may not expose directly (e.g. `_llvm_amdgcn_raw_ptr_buffer_load_lds`, `_llvm_amdgcn_ds_read_tr16_b64`);
- `__shared__` storage for LDS allocations;
- `__builtin_amdgcn_make_buffer_rsrc((void*)ptr, /*stride=*/(short)0, /*num_records=*/num_bytes, /*flags=*/word3)`, where `word3` is the target ISA backend's `buffer_rsrc_word3` (the same word the LLVM route emits);
- `__builtin_amdgcn_mfma_*`;
- C++ `for` and `if` for structured control flow.

Known caveats:

- unsupported ops raise `NotImplementedError`;
- op coverage is narrower than LLVM lowering;
- a few op handlers exist twice in the class; Python uses the later definition (`memref.global_store_vN`, `memref.global_atomic_add_f32`);
- vector constant / type paths for bf16 / fp8 are less tested.

Use HIP output as a debugging lens, not as a guarantee that production LLVM lowering behaves identically.

## CK Tile Spec Backend

`core/lower_cktile.py` is spec-to-C++, not IR-to-C++. Coverage is narrow:

- `UniversalGemmSpec`: fp16 in/out, fp32 acc, `RCR` layout;
- `ImplicitGemmConvSpec`: fp16 NHWC/KYXC/NHWK, 2D spatial.

Unsupported pipeline / scheduler / epilogue / spec types raise `NotImplementedError`. Keep this backend as parity/reference glue.

## Adding A New Primitive

A new primitive should be added in layers:

1. Define the IR operation and builder method in `core/ir.py`. Decide if it is pure (CSE-able, DCE-able) or side-effecting (loads, stores, barriers, MFMA, atomics).
2. Add printer support in `core/ir_print.py` if the op should appear cleanly in textual IR.
3. Add LLVM lowering and any new intrinsic declaration in `core/lower_llvm.py`, and port the same lowering into the C++ engine (`cpp/`) so both engines stay byte-identical (the differential gate flags any divergence).
4. Add HIP debug lowering if source-level inspection is valuable.
5. Add a helper wrapper in `helpers/` if multiple kernels will use the primitive.
6. Add analysis hooks (`analysis/ir.py`, `analysis/isa.py`) if the primitive should be counted in generated IR / ISA.
7. Add a minimal instance / example / test that emits and verifies it.

For performance primitives, document the mapping in `primitives/intrinsics_and_primitives.md` and add a `runbook_compliance.md` row.

## Debugging Backend Problems

Suggested order:

1. `print_ir(kernel)` — confirm the builder emitted what you think it did.
2. Inspect `art.llvm_text` for missing intrinsic declarations, wrong types, or wrong address spaces.
3. Build HSACO and disassemble (`llvm-objdump -d`) via `analyze_hsaco`.
4. Check resource metadata for VGPR / SGPR / LDS surprises (`llvm-readelf --notes`).
5. Run correctness with adversarial tails and masks (`run_manifest --verify` against the manifest's reference path).
6. Benchmark only after correctness is clean (`benchmark_manifest` with `attempts >= 5`, `discard_first=True`).
7. If a primitive disappears, inspect passes and purity flags — pure ops with unused results are eliminated.
8. If a memory op faults, check false-lane address clamping before load; do not rely on post-load `select`.
9. If async code is flaky, check `s_waitcnt(vmcnt=0)` placement before LDS reads.
10. If output is numerically close but wrong, suspect MFMA lane packing and epilogue lane-to-output mapping first.

## Op Purity Cheat Sheet

Pure (CSE-able, DCE-able when unused):

- arithmetic (`arith.*`), comparisons, casts (including `cvt_*`), `clamp_f32`, `bitcast`, `vec_bitcast`;
- vector ops (`vector.*`);
- constants (`arith.constant`, `arith.constant_vec`);
- thread/block id and lane id;
- `readfirstlane`, `pin_sgpr`, `to_sgpr_u32`;
- `wave_all`, `wave_any`, `wave_ballot`;
- `tile.smem_addr_of`, `tile.smem_ptr_add`;
- `ds_bpermute`, `warp_shuffle_xor` (uses `ds_bpermute` internally);
- `static_for`, `unroll` (Python-time only).

Side-effecting (never DCE'd, never reordered by passes):

- all loads, stores, atomics;
- LDS reads and writes;
- async buffer-to-LDS;
- MFMA ops;
- barriers (`tile.sync`, `tile.sync_lds_only`, `tile.sync_half_block`);
- waitcnt, sched_group_barrier, sched_barrier, s_setprio.

When in doubt, set the op's `attrs["pure"]` explicitly to override the default classification (see `Op.is_pure`).
