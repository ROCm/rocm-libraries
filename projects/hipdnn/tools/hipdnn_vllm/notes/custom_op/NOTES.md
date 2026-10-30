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
| gfx94X wheel after the re-merge of `users/sareeder/megakernel-engine` 92ed944eeb0, commit 55181f60133, tree clean, WHEEL_OK | `build_gfx94X_engine_55181f60133.log` |
| custom_op tests, local MI210 (gfx90a), no engine | `test_custom_op_gfx90a_local.log` |
| custom_op tests incl. H6, local MI210 (gfx90a), engine wheel | `test_custom_op_gfx90a_engine.log` |
| gfx942 (MI300X, job 68215745), wheel 55181f60133: custom_op 6 passed + 2 skipped (production catalog), stand-in capture/replay and key mismatch 2 passed; unit 28 failed and `tests/gpu` collection error, both the torchvision mismatch below | `gfx942_run_68215745.log` |
| tests/unit on PR #14255 head cb84aebee6f vs this branch, same venv: identical 18 failures with torchvision 0.29.0, 196 passed for both with 0.28.0 | `unit_base_vs_ours_cpu.log` |
| gfx942 rerun with torchvision pinned to 0.28.0 (shared job 68215720, one free GPU), wheel 55181f60133, tests at 66d0838cd8f: unit 204 passed, 1 failed; custom_op 6+2 passed; `tests/gpu -m "not conv"` 507 passed, 74 failed | `gfx942_overlap_setup.log`, `gfx942_overlap_test.log` (driver `hvco_overlap.sh`) |
| The 74 GPU failures on the PR #14255 head wheel (cb84aebee6f, built the same way) and on this branch's wheel, same node and venv: 74 failed, 40 passed for both | `gfx942_paged_base_vs_ours.log` |

Builds ran in the `hipdnn_env:fullbuild_20261017` docker (Ubuntu 24.04): the
Ubuntu 22.04 host's libstdc++ 12 fails `-Werror` on `get_temporary_buffer`.
The pinned `rocm[devel]==10.0.0` resolves only with
`--extra-index-url https://stable.repo.amd.com/rocm/whl-next/` in addition to `vllm_index`.

whl-next also serves `torchvision 0.29.0+rocm10.0.0`, which pip prefers over the
vLLM index's `0.28.0+rocm10.0.0`. Its `_C.so` is built against another torch
(`undefined symbol _ZN3c104cuda19CUDAErrorLogCaptureC1Ev`), so `torchvision::nms`
is missing. vLLM imports torchvision, so every test that imports vLLM fails (the
28 unit failures and the `tests/gpu` collection error on job 68215745). The cause is
the environment, not this branch: PR #14255's head fails the same tests in the same venv.
Pin `torchvision==0.28.0+rocm10.0.0` whenever whl-next is an index.

Remaining failures, all present before this branch:
- `tests/gpu/test_paged_attention.py` and the paged cases of `tests/gpu/test_sdpa.py`
  (74): they decline `NO_BUILD (SdpaFwdNode: batch size mismatch between Q and K: 16 vs 96)`.
  The frontend's rank/batch check compares Q's batch with the paged K pool's block
  count. The PR #14255 head fails the same 74 tests on the same node.
- `tests/unit/test_declines.py::test_cpp_taxonomy_matches_python` (1): it reads
  `csrc/decline.h` next to `tests/`. The node run copies only `tests/` out of
  the tree, so the installed wheel is imported. The test passes in the
  source-tree runs (`unit_base_vs_ours_cpu.log`).
