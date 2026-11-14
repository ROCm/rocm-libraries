# Change tiers

`plan` gives every finding a tier. A module's tier is the highest of its findings. The tier says how much judgement the change needs.

| Tier | Meaning | Who acts |
|---|---|---|
| 0 | Nothing changes but the pin | `apply` alone |
| 1 | Data the engine already handles: changed object bytes, rows whose values the engine already serves | `apply`, then revalidate on the GPU ([data-changes.md](data-changes.md)) |
| 2 | A value or rule the engine has not seen before | An approved engine edit per finding ([new-values.md](new-values.md)) |
| 3 | A change of structure: new arch or module, a kernel argument change, a layout change | Stop and decide with the user ([structural.md](structural.md)) |

## Findings

| Kind | Tier | Meaning | Page |
|---|---|---|---|
| `object_changed` | 1 | A kept object's bytes changed upstream | data-changes |
| `object_removed` | 1 | A vendored object is not in the new set, because its row was removed or excluded | data-changes |
| `row_added` | 1 | A new upstream row whose values, and whose combination of dispatch values, the tree already has | data-changes |
| `row_changed` | 1 or 2 | A vendored row changed. Tier 1 when only tile sizes (`ts_*`) changed. `ts_qo` still feeds an applicability gate | data-changes or new-values |
| `local_differs_from_baseline` | 1 | A vendored object is not byte-identical to the baseline commit, so its recorded provenance was wrong; the update replaces it | data-changes |
| `row_added_new_value` | 2 | A new upstream row with a value the module's tree has not used: dtype, hdim, mask, mode, bf16_cvt | new-values |
| `row_added_new_combination` | 2 | A new upstream row whose values are all known, but never together, such as fp8 in group mode when the tree has fp8 batch rows and bf16 group rows | new-values |
| `row_dropped_by_rule` | 2 | A vendored row is now excluded by a rule | new-values |
| `row_removed_upstream` | 2 | AITER no longer ships a vendored row as an ASM kernel | new-values |
| `reference_changed` (selection key, mask mapping) | 2 | AITER's `get_kernel_name_key()` or `get_cfg_mask_type()` changed; the diff is in the report | new-values |
| `reference_changed` (`hsa/codegen.py`) | 2 | AITER's registry codegen changed; the vendored `codegen.py` is a derived copy. The diff is in the report and repeats until `mark-ported` | new-values |
| `reference_changed` (`fmha_fwd_v3`) | 3 | AITER's ASM dispatch changed: which cases it sends to CK, and the block size it launches with; the diff is in the report | new-values (declines), structural (block size) |
| `reference_changed` (`can_impl_fmha_v3_fwd`) | 3 | AITER's Python dispatch gate in `aiter/ops/mha.py` changed, in its batch or varlen copy, in a helper it calls (`is_fmha_v3_fp8`, `_fmha_kv_byte_extent_ge_u32`), or in the caller's code it reads (`swa`, the window sizes, the signature): which cases it sends to CK; the diff is in the report | new-values (declines), structural |
| `reference_changed` (argument struct, padding structs, packing, grid, object path, MI308 detection) | 3 | The kernel ABI or launch changed upstream; the diff is in the report | structural |
| `abi_unported` | 3 | The engine still mirrors older AITER code than the vendored objects need; `reference_changed` lists what is left to port | structural |
| `host_code_shared_with_unplanned_archs` | 3 | The module has a `reference_changed` or `reference_missing`, and other archs vendor it but are not in the plan. The engine launches them with the same host code, so porting the change reaches them too | structural |
| `new_arch`, `new_module` | 3 | An arch or module the tree has never held | structural |
| `cmake_arch_missing`, `cmake_module_missing` | 3 | The engine build does not know the arch or module | structural |
| `header_changed`, `csv_removed_upstream` | 3 | A CSV gained, lost or reordered columns, or disappeared | structural |
| `header_mismatch_across_archs` | 3 | After apply, archs would hold same-named CSVs with different headers, which the vendored `codegen.py` rejects. Archs the plan does not update count too | structural |
| `layout_changed` | 3 | The object directories changed, such as the gfx942 `MI300/` and `MI308/` split | structural |
| `reference_missing` | 3 | A mirrored upstream function or struct is gone or renamed | structural |
| `baseline_unknown` | 3 | No arch of the module records a host-code commit and no `--baseline` was given, so ABI changes cannot be diffed | structural |
| `ck_symbol`, `not_elf`, `symbol_missing`, `object_missing_upstream`, `unexpected_files` | 3, blocks `apply` | The upstream content is not what this skill ports, or the module directory holds files the script does not manage | structural |

## Deciding new rows

For each `row_added_new_value` or `row_added_new_combination`, ask the user to keep the row or exclude it.

- Keep it when the engine should serve the case. Then make the engine changes in [new-values.md](new-values.md).
- Exclude it when the engine should not serve it, or not yet. Re-run `plan` with an `--exclude` that states why, and prefer one column rule over a list of kernel names. For example, gfx942 forward keeps only the round-to-nearest-even bf16 rows, because the engine asks for RTNE on bf16 (it asks for RTNA on fp8, whose rows carry `bf16_cvt=1`, so the rule must name the dtype too):

  ```bash
  python3 <skill-directory>/scripts/aiter_fwd_sync.py plan --ref <aiter-ref> --arch gfx942 --exclude gfx942/fmha_v3_fwd "dtype=bf16;bf16_cvt=1,2" "the engine requests RTNE conversion for bf16" --report <report-dir>/plan.json
  ```

  Here `<skill-directory>`, `<aiter-ref>` and `<report-dir>` are as defined in SKILL.md.

A kept row is not inert. The forward builder selects a kernel by looking up the graph's dtype, head dims, mask and mode in the registry (`getKernelNameKey` in `plans/SdpaFwdPlanBuilder.cpp`). Once a row is in the CSV, any graph that passes the gates and matches those values runs it. So for every kept row, check that the gates in [new-values.md](new-values.md) accept exactly the graphs the kernel computes correctly. The gfx950 `fp8bf16` rows are the standing example: gfx950 fp8 is the OCP encoding, while the engine maps only FNUZ fp8 inputs to `fp8bf16` on every arch (see the comment on `getDataTypeIdentifier`). Keeping those rows without changing the gate would run FNUZ data through OCP kernels. No unit test catches this: `TestSdpaFwdPlanBuilder.IsApplicableAvailableKernels` builds FNUZ graphs for fp8 rows, so it passes. Those rows also need AITER's fp8 power-of-two GQA decline, which the engine does not port today ([structural.md](structural.md)). Exclude a row the engine should not serve yet.
