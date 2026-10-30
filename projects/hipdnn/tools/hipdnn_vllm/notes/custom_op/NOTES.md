# custom_op (megakernel route, design step 6)

Branch `users/sareeder/hipdnn-vllm-custom-op`: PR #14255 (`users/dpeabody/hipdnn-vllm`)
plus `users/sareeder/megakernel-engine` merged in.

## Decisions

- `custom_op` opt-in is enforced in `_C`, not only by callers: the bootstrap calls
  `custom_op_set_enabled(op_enabled("custom_op"))`; while off, prepare returns
  `DISABLED` and exec declines `DISABLED` without building a graph. This is what
  makes H6 hold by construction (no CUSTOM_OP node reaches hipDNN).
- Graph: one CUSTOM_OP node, inputs bound uid 1..n in call order, no outputs,
  compute type FLOAT (required by the frontend, unused by the engine), data = payload bytes.
- Plan key: `custom_op|len|op_id|len|payload|<dims:strides:dtype per input>`; never addresses.
- Declines are counted as `fallback` for op `custom_op`; `strict` raises with the
  standard `hipdnn_vllm decline [custom_op]: REASON :: message` (parse_op_error).
- Per-family CMake flags: `[provider.family_cmake_flags]` in BUILD_PINS.toml,
  appended by build_wheel.sh; `gfx94X = ["-DENABLE_MEGAKERNEL_ENGINE=ON"]`. Other
  family wheels do not build the engine, so the H6 check is a gfx94X-wheel check.
- Capture/replay test uses the engine's stand-in kernel (test content, loaded
  with `HIPDNN_DESCRIPTOR_DIR=<build>/lib/hipdnn_plugins/engines/test_arch_content/hip-kernel-provider/integration`),
  since no production code object runs without model weights.

## Evidence

| What | Log |
|---|---|
| gfx94X wheel without engine sources (flag on, engine not merged yet), commit e8d652163c3 | `build_gfx94X_noengine.log` |
| gfx94X wheel with the engine, commit 9fdb4197af7, tree clean, WHEEL_OK | `build_gfx94X_engine.log` |
| custom_op tests, local MI210 (gfx90a), no engine | `test_custom_op_gfx90a_local.log` |
| custom_op tests incl. H6, local MI210 (gfx90a), engine wheel | `test_custom_op_gfx90a_engine.log` |

Builds ran in the `hipdnn_env:fullbuild_20261017` docker (Ubuntu 24.04): the
Ubuntu 22.04 host's libstdc++ 12 fails `-Werror` on `get_temporary_buffer`.
The pinned `rocm[devel]==10.0.0` resolves only with
`--extra-index-url https://stable.repo.amd.com/rocm/whl-next/` in addition to `vllm_index`.
