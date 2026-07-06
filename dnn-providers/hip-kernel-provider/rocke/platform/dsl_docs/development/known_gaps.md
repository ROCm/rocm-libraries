# Known gaps — Cpp engine

This is the authoritative, named-debt inventory for the C++ port of the rocke
lowering engine. It records the parts of the Python engine that are **not yet
fully ported**, grouped by subsystem. Each entry states *what* is missing, *why*
it is currently blocked, and *what would unblock it*.

## How to read this file

The C++ engine's definition-of-done is **byte-identical LLVM-IR emission**
versus the Python engine across every kernel family (see
`tools/check_byte_identity.py`). As of the latest authoritative verification:

- Full fresh `-Werror` build of `librocke_core.a`: **passes**.
- Byte-identity gate at both LLVM flavors (`llvm20`, `llvm22`): **GREEN for all
  65 families**, `bad=0`, `canon=0`.
- Random-spec differential fuzz (gemm / elementwise / reduce): **all IDENTICAL,
  0 mismatch**.

None of the gaps below are on the *valid-config emission path*, which is why the
gate is fully green. They fall into three kinds:

1. **Faithful rejection mirrors** — a `ROCKE_ERR_NOTIMPL` (or a reject-message)
   that translates a Python `raise NotImplementedError` / `ValueError` /
   `KeyError` at the *exact same point*. These are **not gaps**: completing them
   would *diverge* from Python and break byte-identity. They are intentionally
   left as-is and are *not* listed here.
2. **Unreachable / dead branches in this codegen-only library** — code paths
   whose producers (HIP runtime, device tensors, RDNA-only or gfx1250-only
   backends) do not exist in the C engine's current scope, so the branch can
   never be hit by a supported build. These are documented as named gaps below.
3. **Host-orchestration not yet ported** — the fused-MoE end-to-end forward
   path, which depends on a HIP runtime that this codegen-only engine does not
   carry. This is the **largest single gap** and is described first.

---

## 1. Fused-MoE end-to-end orchestration (LARGEST GAP)

**Subsystem:** `instances/common/fused_moe_e2e_*`, `helpers/fused_moe_e2e_*`

The fused-MoE *kernel emission* (router, gate/up GEMM, down GEMM, sorting,
gather, silu-mul, top-k weighted reduce, and the fused-mega kernels) is **fully
ported and byte-identity GREEN**. What is missing is the **host-side end-to-end
forward orchestration** that, in Python, compiles those kernels to HSACO and
drives them on device.

This is the bulk of the engine's remaining `TODO(port)` markers (74 of 80).

| File | What's missing | Why blocked |
|---|---|---|
| `fused_moe_e2e_launchers.cpp` (17) | Each `_ensure_*` / `_moe_*` / `_grouped_*` launcher drives its (already golden-verified) IR builder, then records a non-NULL **sentinel** where a real `KernelLauncher` belongs. | No `compile_kernel(IR) -> HSACO` and no `KernelLauncher(hsaco, signature, cache_key)` allocator exist in this codegen-only library. The opaque `rocke_kernel_launcher_t` has no constructor. |
| `fused_moe_e2e_forward_dynamic.cpp` (22 + 6 NOTIMPL) | Host-roundtrip forward path. All host *sizing arithmetic* (blocks-per-expert, num-m-blocks, total-packed, padded counts, max-padded-m, grid dims, launcher / B-tensor selection) is ported 1:1; the launch chain returns NOTIMPL. | Device-launch orchestration: HIP `launch_kernel` chain, torch `zero_/fill_/copy_`, D→H Counts/Offsets copy, `rocke_tensor_t` device pointers + slice copies. No IR-builder analogue. |
| `fused_moe_e2e_forward_static.cpp` (19 + 4 NOTIMPL) | Static no-roundtrip forward + `capture_graph` / `replay_graph`. All path-selection booleans and `sel_*` launcher/weight selections reproduced faithfully on ctx. | HIP runtime: workspace `pool.prepare/get_spec`, per-call buffer resets, `make_kernel` / `launch_kernel` dispatch, `torch.arange` static offsets, CUDA-graph capture/replay. |
| `fused_moe_e2e_host_helpers.cpp` (13) | `data_ptr`-keyed lazy weight-packing caches (`w_down` / `gu_concat` / `gu_interleaved` ± preshuffled). Cache-key arithmetic and hit/miss decision ported exactly. | `rocke_tensor_t` has no data accessor/allocator; the actual torch `cat/stack/permute/reshape/contiguous` and the host pre-shuffle element permutation are runtime concerns. A `data_ptr` surrogate uses handle identity until a real device pointer is exposed. |
| `fused_moe_e2e_ctx_init.cpp` (1) | `rocke_fmoe_build_ctx_destroy` should free runtime-owned resources. | Nothing to free yet: pool / launcher / tensor peers are opaque forward-decls with no allocator. The lifecycle hook exists so a future HIP backend fills it in. |
| `helpers/fused_moe_e2e_spec.cpp` (1) | `rocke_fmoe_resolve_launch_arch` device-arch probe. | Python probes `get_device_arch()` via `rocke.runtime.hip_module`. The codegen-only library has no HIP runtime; it takes the documented no-device `gfx950` fallback (byte-faithful to Python's except-path). |
| `helpers/fused_moe_e2e_orchestrator.cpp` (4 + 3 NOTIMPL) | Legacy standalone orchestrator surface: forward/static/dynamic stub bodies. | Same device-launch chain. Tile-policy is intentionally *not* duplicated here — the golden-verified policy lives in `rocke_fmoe_build_ctx_init`; duplicating it would risk a drifting second copy. Blocked until this legacy surface is retired or re-pointed at `ctx_init`. |

**What unblocks all of the above:** a HIP runtime backend for the C engine
exposing (a) `compile_kernel(IR) -> HSACO`, (b) a `rocke_kernel_launcher_t`
allocator + `launch_kernel` dispatch, (c) `rocke_tensor_t` device pointers /
allocator / data accessor, and (d) a device-arch probe. Until then the IR
emission is complete and verified; only the runtime wrap is absent. **Do not
fabricate device behavior** in the codegen library.

---

## 2. LLVM lowering — gfx1250 WMMA op-id families

**Subsystem:** `core/lower_llvm/`

| File | What's missing | Why blocked |
|---|---|---|
| `core/lower_llvm/mma.cpp` | The C `WMMA_SPECS` / `WMMA_INT_SPECS` tables omit the Python `_GFX1250_WMMA` (16x16x32 f16/bf16) and `_GFX1250_WMMA_FP8` (16x16x64 fp8/bf8) op-id families. The WMMA op-id fallthrough currently rejects them as "unsupported RDNA WMMA op". |

**Why blocked:** there is no gfx1250 entry in the C backend resolver
(`rocke_ll_backend_for`). A gfx1250 build is rejected up front with `ROCKE_ERR_KEY`,
so these op-ids are unreachable through the C lowerer. The rejection is faithful
for the supported RDNA3/RDNA4 set.

**What unblocks it:** porting the gfx1250 ISA backend (split wait-counters +
57-bit SRD word3) into `rocke_ll_backend_for`, then adding the two op-id families
to the WMMA spec tables.

---

## 3. Helpers — buffer-view, epilogue staging dtypes, schedule assertion

**Subsystem:** `helpers/`

| File | What's missing | Why blocked / what unblocks it |
|---|---|---|
| `helpers/tensor_view.cpp` | The `ROCKE_ADDR_BUFFER` address-space path for `load_scalar` / `store_scalar` / `load_vec` / `store_vec` / `load_vec_at` / `store_vec_at` (6 sites). | Python's buffer branch needs `view.base` to *be* a `BufferResource` carrying `{rsrc, soffset, num_bytes}`; the shared `rocke_tensor_view_t` struct stores `base` as a bare `rocke_value_t*` with no `soffset` / `num_bytes` slots. A faithful port **requires adding buffer-resource fields to `rocke_tensor_view_t` in `include/rocke/helper_rocke.helpers.tensor_view.h`** — a header included by many TUs. The buffer builder prims already exist; only the view-struct plumbing is missing. **No producer in C scope builds a `ROCKE_ADDR_BUFFER` view through this shared path**, so the branch is unreachable dead code. *(See "Structural debt" — this is a flagged cross-TU header change, deliberately not made under the conservative byte-identity rule.)* |
| `helpers/epilogues.cpp` | `CShuffleEpilogue` with `out_dtype != f16` (bf16 / fp8e4m3 / bf8e5m2 staging element types). | Blocked on missing bf16/fp8 staging-store builder prims (bf16 `ds_write` + 1-byte `global_store_vN` fp8 stores) plus lds-view dtype plumbing. **Note:** in Python `out_dtype` is a declared field that `emit()` never reads — Python always emits f16. The C guard is strictly *more* conservative; byte-identity is unaffected because only the f16 default is ever passed. |
| `helpers/epilogues.cpp` | CShuffle publish routed via `StaticDistributedTensor` / `store_tile_cshuffle`. | **Code-organization gap with no IR effect:** the byte-identical publish IR is already emitted directly. The distribution machinery (`make_static_distributed_tensor` + `LoadStoreTraits` + `store_tile_cshuffle`) is not ported; routing through it would not change emitted IR. |
| `helpers/schedule.cpp` | `SchedulePolicy.assert_expected_ir(stats)` post-lowering assertion. | Host-side post-lowering check, not on the emission path. Blocked on the `LlvmIrStats` analysis surface (`rocke.analysis.ir`): the C engine has no ported IR-statistics type, so there is nothing to inspect. Signature documented for when it lands. |

---

## 4. Instances — GEMM / conv host-side and shared-validator gaps

**Subsystem:** `instances/common/` (GEMM + conv families)

All of these emit **no IR** (host-side enumerators / validators / reject text)
or are link-resolved-but-forward-declared; the byte-identity path is unaffected.

| File | What's missing | Why blocked / what unblocks it |
|---|---|---|
| `gemm_universal.cpp` | `all_dispatcher_configs()` host-side enumerator. | A streaming generator does not translate to a single C call; needs a callback-walker entry point added to the **public header `rocke/instance_gemm_universal.h`** (out of body-chunk edit scope). Emits no IR. |
| `gemm_epilogue.cpp` | Heterogeneous-dtype `_storage_dtype` `ValueError` text. | No public `_storage_dtype` validator exists in a shared header. Emit-path call sites only need the resolved IR type (dtypes already validated upstream by `ck_gemm_storage_dtype` in `gemm_spec.cpp`), so IR is byte-identical; only the heterogeneous-reject *message* is deferred until a shared validator lands. |
| `deep_fused_conv_pool_value_types_and_spec.cpp` | Leading underlying-conv gate (`conv_spec=spec.conv_spec()`; `is_valid_conv_spec`) in `is_valid_spec`. | The per-family gate `rocke_implicit_gemm_conv_is_valid_spec` is public, but `spec.conv_spec()`'s only C port requires an `ir_builder_t*` for `kernel_name_join`; this validator is pure-compute with no builder. Needs a builder-free `conv_spec()` / validate path. |
| `deep_fused_conv_pool_conv1_gemm_and_lds_staging.cpp` | `rocke_conv_apply_accumulator_epilogue` peer (called via a local extern-C forward decl). | The peer's `ConvAccumulatorEpilogue` struct in its private header collides with this TU's struct of the same name; needs the peer epilogue type exposed through a non-conflicting shared header. Currently **link-resolved (correct at runtime)**; forward-decl retained. |

---

## Structural debt (not per-family gaps)

These are codebase-level risks worth tracking independently of any single
family's parity.

### Binding bulk — `rocke_engine.cpp`

The pybind/C-interface binding layer (`rocke_engine.cpp`) is a single large
translation unit that fans out across every kernel family. It is the integration
choke point: extending the C engine to a new family means another block in this
file. It compiles and is correct, but its size and breadth make it the highest
churn-and-merge-risk surface in the tree. Treat additions here with care and
prefer keeping per-family logic in the family TUs.

### Recursive CMake-glob risk

The build assembles `librocke_core.a` from a recursive glob of `src/`. This is
convenient but means: (a) a newly added `.cpp` is silently swept in (good for
velocity, bad for review visibility), and (b) a broken or half-merged TU
anywhere under `src/` breaks the whole archive link, with the failure attributed
to the archive rather than the offending file. Past staged merges have produced
exactly this symptom (unrelated `-Werror=format-truncation` / missing-member
errors blocking the archive). Consider an explicit source manifest, or at least
CI that compiles each TU independently, so breakage is localized to its file.

### Flagged cross-TU header change (deliberately deferred)

A faithful buffer-view port in `helpers/tensor_view.cpp` (gap #3) would require
adding buffer-resource fields `{rsrc, soffset, num_bytes}` to
`struct rocke_tensor_view_t` in
`include/rocke/helper_rocke.helpers.tensor_view.h`, which is included by many
TUs. This was intentionally **not** made under the conservative byte-identity
rule (the branch is unreachable dead code today). It is recorded here so the
header change is a conscious decision when a `ROCKE_ADDR_BUFFER` producer lands.

---

## Verification commands

```bash
# Full fresh build (must succeed under -Werror)
cmake -S Cpp -B /tmp/rocke_verify -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rocke_verify -j

# Byte-identity gate, both LLVM flavors (must be GREEN for all families)
ROCKE_LLVM_FLAVOR=llvm20 tools/check_byte_identity.py --build-root /tmp/rocke_verify
ROCKE_LLVM_FLAVOR=llvm22 tools/check_byte_identity.py --build-root /tmp/rocke_verify

# Random-spec differential fuzz (must be all IDENTICAL)
python3 tests/instances/differential/fuzz_diff.py --only gemm,elementwise,reduce \
    --cli /tmp/rocke_irart/ir_lower_cli
```
