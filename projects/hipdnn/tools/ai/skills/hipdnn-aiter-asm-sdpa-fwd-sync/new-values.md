# Tier 2: new values and rules

Each tier 2 finding needs an engine edit, or an exclusion in its place. Propose each edit with the finding it answers, and wait for approval (SKILL.md rule 4).

All paths are relative to `dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine/`. Functions are named rather than given by line, because lines move.

## Where the forward engine reads CSV values

| Column | What reads it | What to check for a new value |
|---|---|---|
| `dtype` | `getDataTypeIdentifier` and the dtype gates in `SdpaFwdPlanBuilder::isApplicable` (`plans/SdpaFwdPlanBuilder.cpp`), including `kernelServesHeadCounts`; `fwdInputBytesPerElement`, whose input element size also selects the launch rules in `computeFwdLaunchParams`; the descale checks | Which hipDNN tensor types map to the string. The same string can mean different encodings on different archs: `fp8bf16` is FNUZ fp8 on gfx942 and OCP fp8 on gfx950. A new mapping needs its own descale and stride checks |
| `hdim_q`, `hdim_v` | The registry lookup in `getKernelNameKey`; `computeFwdLaunchParams` in `plans/SdpaFwdLaunchParams.hpp`; the head-dim declines in `isApplicable` (`kernelServesHeadCounts`, and the hd192x128 K/V row-extent decline) | Two kernels are special-cased in `computeFwdLaunchParams`: hd192x128 (block size 256 on every arch; tuning and grid rules on gfx942) and gfx950 bf16 hd256 (`tgDiv = 1`; block size 256 in group mode). Any other head dim silently gets block size 512 and the default tuning, so set them from AITER's `fmha_fwd_v3()` and `get_grid_dim()` for the new head dim. Also check the head-dim conditions in `fmha_fwd_v3()` and `can_impl_fmha_v3_fwd` (see below) |
| `mask` | `plan_utils::MaskType` in `plans/SdpaPlanUtils.hpp`, whose ordinals equal the CSV column; mask classification in `isApplicable` | A new ordinal needs a `MaskType` value and its mapping from the graph. Never reorder the enum |
| `mode` | Batch versus group (ragged) mode in `isApplicable` and `hasAiterCompatibleRaggedAddressing`; `SdpaFwdGroupModeParams` in `plans/SdpaFwdParams.hpp`; the group-mode grid in `computeFwdLaunchParams` (`plans/SdpaFwdLaunchParams.hpp`) | Stats are declined in group mode today |
| `bf16_cvt` | `getKernelNameKey`: matched on gfx942 and ignored on gfx950; the requested value is RTNE for bf16 and RTNA for fp8 | Whether a new arch matches or ignores it |
| `ts_qo` | The Ts overflow gate at the end of `isApplicable`; `buildPlan` passes it to the grid (`computeFwdLaunchParams`) and the `s_Ts` argument (`plans/SdpaFwdArgsBuilder.hpp`) | A change can change which graphs are accepted, not just the grid. `ts_kv` is not read by the forward engine |
| `knl_name`, `co_name` | `buildPlan`: the symbol to load, and the archive key through `getKernelTocKey` | Nothing per value; a new object directory is a `layout_changed` |

Stats (LSE) output is declined with a mask and in group mode. Recheck both rules when AITER adds kernels that write LSE in those cases.

## New combinations (`row_added_new_combination`)

Every value in the row is already served, but not in this combination, such as fp8 in group mode. Gates are written per feature, so check each pair the combination creates. For fp8 in group mode that is the descale checks (`isScalarTensor`), the ragged addressing checks (`hasAiterCompatibleRaggedAddressing`) and the group-mode grid. Add a decline for any pair the engine cannot serve, or exclude the row.

## AITER dispatch changes (`reference_changed`)

The report shows a diff of AITER's ASM dispatch code: `fmha_fwd_v3()`, `get_kernel_name_key()` and `get_cfg_mask_type()` in `csrc/cpp_itfs/mha_fwd.cu`, and the Python dispatch gate `can_impl_fmha_v3_fwd` in `aiter/ops/mha.py`, whose batch and varlen copies the report names `_flash_attn_forward.can_impl_fmha_v3_fwd` and `_flash_attn_varlen_forward.can_impl_fmha_v3_fwd`. A `fmha_fwd_v3()` diff is tier 3 because the function also sets the launch block size; port that part as [structural.md](structural.md) describes. A `can_impl_fmha_v3_fwd` diff is tier 3 too.

- Every early `return -1` in `fmha_fwd_v3()` is a case AITER sends to CK. For each one the diff adds, make `SdpaFwdPlanBuilder::isApplicable` decline the same case, with a reason in its `HIP_KERNEL_RETURN_FALSE_IF` message. Never port or call a CK kernel for it (SKILL.md rule 1). For example, between `17d4a33` and `v0.1.28.post1` AITER starts accepting head dim 256/256 and returns -1 for gfx950 bf16 hd256 with GQA. It also launches gfx950 bf16 hd256 without tile merging (`tg_div = 1` in `get_grid_dim()`), and with 256 threads in group mode. Keeping the hd256 rows on gfx950 therefore needs the GQA decline (`kernelServesHeadCounts`), and the grid and block-size rules in `computeFwdLaunchParams`.
- Every condition `can_impl_fmha_v3_fwd` adds also sends a case to CK. Decline it the same way when it is a kernel limit. A condition that only states a routing preference needs no decline; agree with the user which it is. For example, between `17d4a33` and `v0.1.28.post1` AITER declines hd192x128 when the K or V per-head row extent (`max_seqlen_k` times the token stride times the element size) reaches 2^32 bytes, which wraps the kernel's 32-bit buffer offset. The engine mirrors it in `isApplicable` on both archs. The fp8 power-of-two GQA rule added in the same range is not ported ([structural.md](structural.md)).
- A changed selection key means `getKernelNameKey` must change the same way.
- A changed mask mapping needs care: the engine does not mirror `get_cfg_mask_type()`. AITER sends top-left causal with `seqlen_q == seqlen_k` to the bottom-right rows (mask 2); the engine classifies it as `MaskType` 1, finds no row and declines (`getMaskType` in `plans/SdpaPlanUtils.hpp`). Decide with the user whether to follow a change, and never let the engine map a graph onto a mask the kernel does not compute.
- A removed `return -1` widens what AITER sends to ASM. The engine may follow, but only with rows and GPU tests that cover the new case.

These diffs come from the module's host-code commit, so they repeat in every plan until ported. When all of a module's diffs are handled, run `mark-ported --module <module>` (see SKILL.md).

## Rows dropped or removed

- **`row_removed_upstream`**: AITER no longer ships the kernel as ASM. The registry drops it, so the engine declines those graphs. Remove its `.support.json` claims for that arch, and tell the user which graphs lost ASM coverage. Do not replace it with a CK kernel.
- **`row_dropped_by_rule`**: an exclusion now removes a vendored row. Confirm the user meant it, then handle it like a removal.

## Vendored codegen (`reference_changed` on `hsa/codegen.py`)

`asm/asm_kernels/codegen.py` is a derived copy of AITER's `hsa/codegen.py`; it is not a verbatim one. The report diffs the upstream file from the host-code commit, and the diff repeats in every plan until `mark-ported`, as the host-code diffs do. Read it and port only what the local CSVs need. The local copy differs on purpose:

- It reads CSVs with the `csv` module, not pandas, and stores `co_name` with an arch prefix (`<arch>/<module>/<co_name>`). The builder and `asm/AsmKernelPath.hpp` depend on that path and on the `arch + knl_name` key, so keep both.
- It does not strip comments. It does not need to: `apply` writes every CSV without comments, using the same comment rules as upstream (`/* */` blocks, whole-line and line-end `#`, `//` and `;`).
- It merges same-named CSVs of every arch directory on disk into one struct and stops when their headers differ. Upstream instead takes the union of the columns. Port the union only when a `header_mismatch_across_archs` finding needs it ([structural.md](structural.md)).

Run the build after a port; it regenerates `asm_<module>_configs.hpp` for every arch.
