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

The report diffs AITER's host code from the module's host-code commit (`abi_commit`) to the target: the argument struct (`fmha_fwd_v3_args` in `csrc/include/mha_fwd.h`), the SGPR padding structs it is built from (`p1`, `p2` and `p3` in `csrc/include/aiter_hip_common.h`), its packing (`init_fmha_fwd_v3_args`), the grid (`get_grid_dim`) and the object path per device (`get_kernel_co_name`). The engine mirrors each by hand:

| Upstream | Engine |
|---|---|
| `fmha_fwd_v3_args` | `asm/SdpaFwdKernelArgs.hpp` (packed struct with `static_assert`s on size and field offsets) |
| `p1`, `p2`, `p3` | `asm/SgprPadding.hpp` (`SgprPad2`, `SgprPad3`) |
| `init_fmha_fwd_v3_args` | `plans/SdpaFwdArgsBuilder.hpp` |
| `get_grid_dim` | `computeFwdLaunchParams` in `plans/SdpaFwdLaunchParams.hpp` |
| `get_kernel_co_name` | `getKernelTocKey` and `isMi308Device` in `plans/SdpaFwdPlanBuilder.cpp`, and `asm/AsmKernelPath.hpp` |

Port the change field by field. Update the size and offset asserts from the new upstream struct, and add an offset assert for every field the change touches. `tests/engines/asm_sdpa_engine/TestSdpaFwdKernelArgs.cpp` and `TestSdpaFwdPlanGridMath.cpp` must pass, with expected values taken from upstream, never from the old engine. A kernel argument change applies to every object of the module, so all its kernels need GPU validation on every arch that ships them.

`asm/SgprPadding.hpp` is shared with the backward kernel arguments, which this skill does not touch. A padding change therefore needs a backward review too. Stop and agree it with the user before editing that file.

When every `reference_changed` diff of a module is ported, run `mark-ported --module <module>` (`<module>` is `<arch>/<module-name>`). Until then the pin keeps the older host-code commit and every later plan repeats the diffs, so a change deferred to another PR is not lost.

## New arch (`new_arch`, `cmake_arch_missing`)

1. Add the arch to `ASM_SDPA_GPU_ARCHS` in `CMakeLists.txt`. That runs codegen and packs a `hip_kernel_provider_sdpa_<arch>.kpack` for it.
2. Allow the arch in the device check of the builder that will serve it. `SdpaFwdPlanBuilder::isApplicable` accepts only gfx942 and gfx950.
3. Work out what is arch-specific. The forward builder has gfx942-only rules: `bf16_cvt` matching in `getKernelNameKey`, the MI300/MI308 split in `getKernelTocKey`, and the hd192x128 launch override in `computeFwdLaunchParams`. Decide each one for the new arch from AITER's own host code for it.
4. Check that the arch's objects pack and load. `asm/asm_kernels/pack.py` names the archive's target, and the runtime loads by device string. Extend `tests/engines/asm_sdpa_engine/TestAsmKpackLoading.cpp` with one object of the new arch.
5. Add the arch to the forward GPU tests and to the cross-provider claims once it passes ([validation.md](validation.md)).

## New module (`new_module`, `cmake_module_missing`)

A new module, such as gfx1250's `fmha_fwd_bf16` and `fmha_fwd_bf16_varlen`, has its own CSV schema, argument struct and launch. It is new engine work, normally its own PR.

1. **Codegen.** Add a custom command for `-m <module>` to `CMakeLists.txt`, beside the `fmha_v3_fwd` one, and add the header to the `asm_kernel_codegen` target. Every generated header defines the global names `CFG` and `ADD_CFG` (see `asm/asm_kernels/codegen.py`). Two module headers in one translation unit therefore collide, so give each module its own names before a source file includes two.
2. **Host interface.** There is no baseline, so read the module's reference sources in full. For gfx1250 these are `csrc/py_itfs_cu/asm_fmha_fwd_with_sink.cu` or `asm_fmha_fwd_with_sink_varlen.cu`, plus the padding structs in `csrc/include/aiter_hip_common.h`. Note the argument struct and its asserted size, tensor layout and stride rules, required outputs (LSE is mandatory there), optional inputs (sink), and the launch geometry.
3. **Plan family.** Serve the module with its own argument struct, launch parameters and applicability rules. Either add a plan builder registered in `core/Container.cpp`, or add a per-module path inside `SdpaFwdPlanBuilder`; agree which with the user. Map hipDNN features onto the kernel's contract: ragged tensors onto `cu_seqlens`, a workspace LSE when the graph asks for no stats, and the sink tensor (declined by the engine today; ask the user before enabling it).
4. **Tests.** Unit tests for the argument struct and launch math; forward GPU tests driven by the module's registry, as `integration_tests/asm_sdpa_engine/IntegrationGpuSdpaForward.cpp` does for `cfg_fmha_fwd`.
5. **Record the host code.** A new module's pin starts with no host-code commit, so later plans report `baseline_unknown`. Once the plan family mirrors the pinned commit's host code, run `mark-ported --module <module>`.

## CSV and layout changes

- **`header_changed`**: codegen builds the registry struct from the header, so a renamed or removed column breaks every C++ use of it. Find the uses of the generated config struct, update them, and check that `codegen.py` still parses the CSV. An added column is usable only after the engine reads it.
- **`csv_removed_upstream`**: AITER dropped a CSV, and every row in it is lost. Treat it as a removal of all its kernels ([new-values.md](new-values.md)).
- **`layout_changed`**: the object directories changed. The gfx942 forward objects live in `MI300/` and `MI308/`, chosen at runtime by `isMi308Device` and `getKernelTocKey`. Mirror the new split in both, and in the archive-key tests.
- **`reference_missing`**: a mirrored upstream function or struct is gone or renamed. Find where its logic now lives upstream. If it moved, record the new name in `REFERENCES` in the helper script, in a change to this skill.
- **`baseline_unknown`**: re-run `plan` with `--baseline`. If no one knows the baseline, read the reference sources in full and compare them with the engine by hand.

## Forward modules not ported

The report lists AITER forward modules the skill does not port, with the reason. `fmha_fwd_mxfp8` needs MX block-scale descales, which the hipDNN SDPA graph does not model. `fmha_v4_fwd` is not used by the engine. Tracking one is a change to `TRACKED_MODULES` in the helper script, plus the new-module work above. Agree it with the user first.
