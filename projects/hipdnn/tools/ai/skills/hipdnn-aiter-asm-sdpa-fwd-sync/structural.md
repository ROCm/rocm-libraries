# Tier 3: structural changes

Stop at every tier 3 finding. Show the user the finding and, where there is one, its diff from the report, then decide together: do the work in this update, exclude the affected rows or module, or split it into its own PR. A plan with a blocking finding cannot be applied.

Engine paths are relative to `dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine/`. Paths starting with `tests/`, `integration_tests/` or `core/` are relative to `dnn-providers/hip-kernel-provider/src/`.

## Blocking findings

These mean the upstream content is not what the skill ports, or the tree holds files the script does not manage. `apply` refuses them.

- **`ck_symbol`**: an object defines Composable Kernel symbols. It is not one of AITER's ASM kernels. Never port it. Exclude its rows and tell the user.
- **`not_elf`**, **`symbol_missing`**: the object is not a code object, or it does not define the kernel its CSV row names. Exclude the row and report it upstream.
- **`object_missing_upstream`**: a CSV row names an object AITER does not ship. Exclude the row.
- **`unexpected_files`**: the module directory holds a file that is neither a CSV, an object nor `SOURCE.md`. Find out where it came from before anything is deleted.

## Kernel ABI and launch (`reference_changed`, `abi_unported`, tier 3)

The report diffs AITER's host code from the module's host-code commit (`abi_commit`) to the target. For forward modules that is the argument struct (`fmha_fwd_v3_args` in `csrc/include/mha_fwd.h`), the SGPR padding structs it is built from (`p1`, `p2` and `p3` in `csrc/include/aiter_hip_common.h`), its packing (`init_fmha_fwd_v3_args`), the grid (`get_grid_dim`), the block size (`bdx` in `fmha_fwd_v3`), the object path per device (`get_kernel_co_name`) and the MI308 detection it uses (`is_mi308_device` in `aiter_hip_common.h`). The engine mirrors each by hand:

| Upstream | Engine |
|---|---|
| `fmha_fwd_v3_args` | `asm/SdpaFwdKernelArgs.hpp` (packed struct with a `static_assert` on its size; the field offsets are checked in `tests/engines/asm_sdpa_engine/TestSdpaFwdKernelArgs.cpp`) |
| `p2`, `p3` | `asm/SgprPadding.hpp` (`SgprPad2`, `SgprPad3`). `p1` has no mirror; add one if a struct starts using it |
| `init_fmha_fwd_v3_args` | `buildFwdKernelArgs` in `plans/SdpaFwdArgsBuilder.hpp` |
| `get_grid_dim`, and `bdx` in `fmha_fwd_v3` | `computeFwdLaunchParams` in `plans/SdpaFwdLaunchParams.hpp` |
| `get_kernel_co_name`, `is_mi308_device` | `getKernelTocKey` and `isMi308Device` in `plans/SdpaFwdPlanBuilder.cpp`, and `asm/AsmKernelPath.hpp` |

Port the change field by field. Update the size assert from the new upstream struct, and the offset checks in `TestSdpaFwdKernelArgs.cpp` for every field the change touches. `TestSdpaFwdKernelArgs.cpp` and `TestSdpaFwdPlanGridMath.cpp` must pass, with expected values taken from upstream, never from the old engine. A kernel argument change applies to every object of the module, so all its kernels need GPU validation on every arch that ships them.

For `fmha_v3_bwd` the report diffs the three argument structs in `csrc/include/mha_bwd.h`, the padding structs, `fmha_v3_bwd()` in `csrc/cpp_itfs/mha_bwd.cu` (argument packing, mask mapping, the post stage, block size and grids) and `compute_mask_coordinates` in `csrc/include/ck_tile_shim.h`:

| Upstream | Engine |
|---|---|
| `fmha_bwd_odo_args`, `fmha_bwd_dqdkdv_args`, `fmha_bwd_post_kernel_args` | `asm/SdpaBwdKernelArgs.hpp` (packed structs with `static_assert`s on size and offsets; the field offsets are also checked in `tests/engines/asm_sdpa_engine/TestSdpaBwdKernelArgs.cpp`) |
| Packing in `fmha_v3_bwd` | `buildOdoArgs`, `buildDqdkdvArgs` and `buildPostArgs` in `plans/SdpaBwdPlan.cpp` |
| Block size, grids and stage order in `fmha_v3_bwd` | `K_BWD_BLOCK_DIM` and `SdpaBwdPlan::execute` in `plans/SdpaBwdPlan.cpp`; tile math in `TestSdpaBwdPlanGridMath.cpp` |
| Mask mapping and window arguments in `fmha_v3_bwd`, `compute_mask_coordinates` | `dqdkdvMaskCoordinates` in `plans/SdpaBwdPlan.cpp`, `computeMaskCoordinates` in `plans/SdpaPlanUtils.hpp`, and the mask ordinal chosen in `plans/SdpaBwdPlanBuilder.cpp` |
| Workspace (dq_acc layout per accumulator and mode) | `SdpaBwdPlan::getWorkspaceSize` and `SdpaBwdPlanBuilder::getMaxWorkspaceSize` |

A new post stage, such as gfx950's dq_shuffle for the A16 accumulator, is a new pipeline stage in `SdpaBwdPlanBuilder::buildPlan` and `SdpaBwdPlan`, with its own registry lookup; treat it like a new module's plan family below. `TestSdpaBwdKernelArgs.cpp`, `TestSdpaBwdPlanGridMath.cpp` and `TestSdpaBwdPlanBuilder.cpp` must pass, with expected values taken from upstream.

The engine has one copy of each module's host code for every arch: one `fmha_fwd_v3_args`, one `buildFwdKernelArgs` and one `computeFwdLaunchParams` for forward, and one set of backward argument structs, packing functions and `SdpaBwdPlan::execute` for backward. A ported change therefore reaches every arch that vendors the module, not only the archs in the plan. The plan reports `host_code_shared_with_unplanned_archs` for each arch it leaves out. Either add those archs to the same plan, so their objects move to the new ABI with the host code, or do not port the change. Agree which with the user. Rules gated on the arch in the engine reach only their arch: `bf16_cvt`, the MI300/MI308 split, hd192x128 on gfx942 and hd256 on gfx950 for forward, and the post stage, the window rules and the head dims served for backward.

`asm/SgprPadding.hpp` is shared by the forward and backward kernel arguments. A padding change therefore needs both directions reviewed, even when the plan covers one. Stop and agree it with the user before editing that file.

When every `reference_changed` diff of a module is ported, run `mark-ported --module <module>` (`<module>` is `<arch>/<module-name>`). Until then the pin keeps the older host-code commit and every later plan repeats the diffs, so a change deferred to another PR is not lost.

## Python dispatch gate (`reference_changed`, tier 3)

AITER also decides in Python whether a call reaches the ASM kernels: `can_impl_fmha_v3_fwd` in `aiter/ops/mha.py` for forward, with one copy for batch mode and one for varlen (group) mode, and `_flash_attn_backward` and `_flash_attn_varlen_backward` (with the per-arch gates nested in them) for backward. The report diffs each gate from `abi_commit`, together with every function the gate calls that `mha.py` defines, transitively: for forward, the caller's nested `is_fmha_v3_fp8` (fp8 archs, head dims and descale shapes) and the module-level `_fmha_kv_byte_extent_ge_u32`; for batch-mode backward, the module-level `can_impl_fmha_v3_bwd` (the stride equalities per kernel variant). `_flash_attn_varlen_backward` defines its own `can_impl_fmha_v3_bwd`, which is part of its region. A change confined to a helper is reported under each gate that calls it. A case a gate rejects goes to CK, so port each new kernel limit as a decline in the builder's `isApplicable`, as [new-values.md](new-values.md) describes for `fmha_fwd_v3()` and `fmha_v3_bwd()`. A condition that only states a routing preference needs no decline. The bullets below record the engine's choice for each gate condition that needed one, including the routing preferences it deliberately does not follow; keep them when a gate diff touches them.

- The engine mirrors the hd192x128 K/V row-extent decline (2^32 bytes) on gfx942 and gfx950. Both gate copies compute the extent in `_fmha_kv_byte_extent_ge_u32`.
- The engine does not port the fp8 rule that `nhead_q / nhead_k` be a power of two. The gfx950 fp8 rows are excluded, and the gfx942 fp8 objects divide by the GQA ratio in full. Port the rule before enabling the gfx950 fp8 rows.
- The engine serves backward at `seqlen_q <= 16` on both archs, although `_flash_attn_backward` sends it to CK: AITER chooses CK there for speed (its CK backward has an optimization for short queries), not correctness. `IntegrationGpuSdpaBackward.cpp` covers Sq = 1 and 16.
- The engine serves the gfx942 backward windows with left bound 0 and right bound 0 or -1. `can_impl_fmha_v3_bwd` does not count them as windows, so they fall out of every ASM family; that is a classification artifact, and the gfx942 window kernel computes them like any other window (`isBatchWindowSupported` in `plans/SdpaBwdPlanBuilder.cpp`).

## New arch (`new_arch`, `cmake_arch_missing`)

1. Add the arch to `ASM_SDPA_GPU_ARCHS` in `CMakeLists.txt`. That runs codegen and packs a `hip_kernel_provider_sdpa_<arch>.kpack` for it.
2. Allow the arch in the device check of each builder that will serve it. `SdpaFwdPlanBuilder::isApplicable` accepts only gfx942 and gfx950; the backward builder checks the arch in `isSupportedArch` (`plans/SdpaBwdPlanBuilder.cpp`).
3. Work out what is arch-specific. The forward builder has gfx942-only rules: `bf16_cvt` matching in `getKernelNameKey`, the MI300/MI308 split in `getKernelTocKey`, and the hd192x128 tuning and grid rules in `computeFwdLaunchParams`. It has gfx950-only rules too: the bf16 hd256 grid and block-size rules in `computeFwdLaunchParams`, and the bf16 hd256 GQA/MQA decline (`kernelServesHeadCounts` in `plans/SdpaFwdPlanBuilder.cpp`). The backward path has per-arch rules in `fmha_v3_bwd()` and `get_heuristic_kernel()`: the mask ordinal and window arguments, the post kernel (dq_convert or dq_shuffle), the `atomic32=0` tile size for pssk, and the head dims each arch sends to ASM in `aiter/ops/mha.py`. Decide each one for the new arch from AITER's own host code for it, including the arch checks in `can_impl_fmha_v3_fwd` and `can_impl_fmha_v3_bwd`.
4. Check that the arch's objects pack and load. `asm/asm_kernels/pack.py` names the archive's target, and the runtime loads by device string. Extend `tests/engines/asm_sdpa_engine/TestAsmKpackLoading.cpp` with one object of the new arch.
5. Update the tests that name archs: the gfx942 and gfx950 skip guards in `tests/engines/asm_sdpa_engine/TestSdpaFwdPlanBuilder.cpp`, `TestSdpaBwdPlanBuilder.cpp` and `TestAsmSdpaEngine.cpp`, and the arch strings of the shape-sweep cases in `integration_tests/asm_sdpa_engine/IntegrationGpuSdpaForward.cpp` and the arch checks in `IntegrationGpuSdpaBackward.cpp`.
6. Add the arch to the GPU tests of each direction and to the cross-provider claims once it passes ([validation.md](validation.md)).

## New module (`new_module`, `cmake_module_missing`)

A new module, such as gfx1250's `fmha_fwd_bf16` and `fmha_fwd_bf16_varlen`, has its own CSV schema, argument struct and launch. It is new engine work, normally its own PR.

1. **Codegen.** Add a custom command for `-m <module>` to `CMakeLists.txt`, beside the `fmha_v3_fwd` one, and add the header to the `asm_kernel_codegen` target. Every generated header defines the global names `CFG` and `ADD_CFG`, and one `cfg_<csv-name>` per CSV name (see `asm/asm_kernels/codegen.py`). Two module headers in one translation unit therefore collide, so give each module its own names before a source file includes two.
2. **Host interface.** There is no baseline, so read the module's reference sources in full. For gfx1250 these are `csrc/py_itfs_cu/asm_fmha_fwd_with_sink.cu` or `asm_fmha_fwd_with_sink_varlen.cu`, plus the padding structs in `csrc/include/aiter_hip_common.h`. Note the argument struct and its asserted size, tensor layout and stride rules, required outputs (LSE is mandatory there), optional inputs (sink), and the launch geometry.
3. **Plan family.** Serve the module with its own argument struct, launch parameters and applicability rules. Either add a plan builder registered in `core/Container.cpp`, or add a per-module path inside `SdpaFwdPlanBuilder`; agree which with the user. Map hipDNN features onto the kernel's contract: ragged tensors onto `cu_seqlens`, a workspace LSE when the graph asks for no stats, and the sink tensor (declined by the engine today; ask the user before enabling it).
4. **Tests.** Unit tests for the argument struct and launch math; forward GPU tests driven by the module's registry, as `integration_tests/asm_sdpa_engine/IntegrationGpuSdpaForward.cpp` does for `cfg_fmha_fwd`.
5. **Record the host code.** A new module's pin starts with no host-code commit, so later plans report `baseline_unknown`. Once the plan family mirrors the pinned commit's host code, run `mark-ported --module <module>`.

## CSV and layout changes

- **`header_changed`**: codegen builds the registry struct from the header, so a renamed or removed column breaks every C++ use of it. Find the uses of the generated config struct, update them, and check that `codegen.py` still parses the CSV. An added column is usable only after the engine reads it.
- **`csv_added`**: the module gains a CSV, so codegen emits one more registry, `cfg_<csv-name>`, in the module's header. A registry no code reads is inert, but keep it only when the engine will use it, such as gfx950's `fmha_bwd_dq_shuffle.csv`, which the dq_shuffle post stage of the A16 backward accumulator needs. Its objects then count as named by a CSV row, which `verify` requires of every object. Otherwise leave the file out with `--exclude <module> csv=<file> <reason>`, and its objects go with it.
- **`header_mismatch_across_archs`**: the vendored `codegen.py` stops the build when same-named CSVs of two arch directories have different headers, and it reads every arch directory on disk, not only the archs being updated. Either update the other archs in the same plan so the headers agree again, or port AITER's column union into `codegen.py` (see [new-values.md](new-values.md)). Agree which with the user.
- **`csv_removed_upstream`**: AITER dropped a CSV, and every row in it is lost. Treat it as a removal of all its kernels ([new-values.md](new-values.md)).
- **`layout_changed`**: the object directories changed. The gfx942 forward objects live in `MI300/` and `MI308/`, chosen at runtime by `isMi308Device` and `getKernelTocKey`. Mirror the new split in both, and in the archive-key tests.
- **`reference_missing`**: a mirrored upstream function or struct is gone or renamed. Find where its logic now lives upstream. If it moved, record the new name in `REFERENCES` in the helper script, in a change to this skill. `apply` keeps the older host-code commit while a reference is missing, so once the new name is recorded, later plans report the moved code until it is ported.
- **`baseline_unknown`**: re-run `plan` with `--baseline`, and with `--abi-baseline` when the engine mirrors older host code than its objects (SKILL.md, "Baselines for a first update"). If no one knows the baseline, read the reference sources in full and compare them with the engine by hand.

## Modules not ported

The report lists AITER modules of the planned directions that the skill does not port, with the reason. `fmha_fwd_mxfp8` needs MX block-scale descales, which the hipDNN SDPA graph does not model. `fmha_v4_fwd` is not used by the engine. Tracking one is a change to `TRACKED_MODULES` and `REFERENCES` in the helper script, plus the new-module work above. Agree it with the user first.
