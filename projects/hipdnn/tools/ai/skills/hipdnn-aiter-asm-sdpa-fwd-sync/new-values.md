# Tier 2: new values and rules

Each tier 2 finding needs an engine edit, or an exclusion in its place. Propose each edit with the finding it answers, and wait for approval (SKILL.md rule 4).

All paths are relative to `dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine/`. Functions are named rather than given by line, because lines move.

## Where the forward engine reads CSV values

| Column | What reads it | What to check for a new value |
|---|---|---|
| `dtype` | `getDataTypeIdentifier` and the dtype gates in `SdpaFwdPlanBuilder::isApplicable` (`plans/SdpaFwdPlanBuilder.cpp`); `fwdInputBytesPerElement`; the descale checks | Which hipDNN tensor types map to the string. The same string can mean different encodings on different archs: `fp8bf16` is FNUZ fp8 on gfx942 and OCP fp8 on gfx950. A new mapping needs its own descale and stride checks |
| `hdim_q`, `hdim_v` | The registry lookup in `getKernelNameKey`; `plans/SdpaFwdLaunchParams.hpp` (block size and tuning per head dim) | Launch parameters for the new head dim; head-dim conditions in AITER's `fmha_fwd_v3()` (see below) |
| `mask` | `plan_utils::MaskType` in `plans/SdpaPlanUtils.hpp`, whose ordinals equal the CSV column; mask classification in `isApplicable` | A new ordinal needs a `MaskType` value and its mapping from the graph. Never reorder the enum |
| `mode` | Batch versus group (ragged) mode in `isApplicable` and `hasAiterCompatibleRaggedAddressing`; `SdpaFwdGroupModeParams` in `plans/SdpaFwdParams.hpp`; the group-mode grid in `computeFwdLaunchParams` (`plans/SdpaFwdLaunchParams.hpp`) | Stats are declined in group mode today |
| `bf16_cvt` | `getKernelNameKey`: matched on gfx942 and ignored on gfx950; the requested value is RTNE for bf16 and RTNA for fp8 | Whether a new arch matches or ignores it |

Stats (LSE) output is declined with a mask and in group mode. Recheck both rules when AITER adds kernels that write LSE in those cases.

## New combinations (`row_added_new_combination`)

Every value in the row is already served, but not in this combination, such as fp8 in group mode. Gates are written per feature, so check each pair the combination creates. For fp8 in group mode that is the descale checks (`isScalarTensor`), the ragged addressing checks (`hasAiterCompatibleRaggedAddressing`) and the group-mode grid. Add a decline for any pair the engine cannot serve, or exclude the row.

## AITER dispatch changes (`reference_changed`, tier 2)

The report shows a diff of AITER's ASM dispatch code: `fmha_fwd_v3()`, `get_kernel_name_key()` and `get_cfg_mask_type()` in `csrc/cpp_itfs/mha_fwd.cu`.

- Every early `return -1` in `fmha_fwd_v3()` is a case AITER sends to CK. For each one the diff adds, make `SdpaFwdPlanBuilder::isApplicable` decline the same case, with a reason in its `HIP_KERNEL_RETURN_FALSE_IF` message. Never port or call a CK kernel for it (SKILL.md rule 1). For example, between `17d4a33` and `v0.1.27` AITER starts accepting head dim 256/256 and returns -1 for gfx950 bf16 hd256 with GQA, so keeping the hd256 rows on gfx950 also needs that GQA decline.
- A changed selection key or mask mapping means `getKernelNameKey` or the mask classification must change the same way.
- A removed `return -1` widens what AITER sends to ASM. The engine may follow, but only with rows and GPU tests that cover the new case.

These diffs come from the module's host-code commit, so they repeat in every plan until ported. When all of a module's diffs are handled, run `mark-ported --module <module>` (see SKILL.md).

## Rows dropped or removed

- **`row_removed_upstream`**: AITER no longer ships the kernel as ASM. The registry drops it, so the engine declines those graphs. Remove its `.support.json` claims for that arch, and tell the user which graphs lost ASM coverage. Do not replace it with a CK kernel.
- **`row_dropped_by_rule`**: an exclusion now removes a vendored row. Confirm the user meant it, then handle it like a removal.

## Vendored codegen (`codegen_changed`)

`asm/asm_kernels/codegen.py` is a derived copy of AITER's `hsa/codegen.py`; it is not a verbatim one. It keeps a local registry key format and arch-prefixed object paths. Read the upstream diff and port only what the local CSVs need, such as comment stripping or new column handling. Keep the generated struct and key format, because the builder depends on them.
