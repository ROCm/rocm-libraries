# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# CTest enrolment of the library test suites in the installed test tree.
#
# Included by platform/CMakeLists.txt (inside if(ROCKE_INSTALL_PYTHON_TESTS),
# only when this library tree sits beside platform/), so a library test is
# enrolled -- or a new lane added -- by editing this file alone. Each entry is
# appended to _ROCKE_INSTALLED_CTEST_ENTRIES in the including scope. The library
# tree is staged under tests/library of the install, and every path below is
# relative to the CTestTestfile, hence the ROCKE_TEST_CTEST_REL prefix.
#
# A test that is in library/tests but in no list here does not run in CI.
#
# Every pytest entry but the explicit gates is registered with
# rocke_library_pytest_entry below, so its command starts with
# ${_ROCKE_INSTALLED_PYTEST}, which platform/CMakeLists.txt defines; the
# configure fails for any other pytest entry that does not. An entry that takes
# more than a few seconds of a CI shard and launches kernels in few or none of
# its tests passes WORKERS.

# Append the library pytest entry <name> to _ROCKE_INSTALLED_CTEST_ENTRIES:
#
#   rocke_library_pytest_entry(<name>
#       TESTS <path>... [IGNORE <path>...] [ARGUMENTS <argument>...]
#       LABELS <label>... TIMEOUT <seconds> [PYTHONPATH <value>]
#       [WORKERS] [GPU] [COMPILER_TOOLS] [COMPILER_REQUIREMENT])
#
# TESTS (paths or node ids) and IGNORE (each an --ignore, after the tests) are
# relative to the staged library tests directory, tests/library/tests of the
# install. PYTHONPATH defaults to the staged library tree. The rest is
# rocke_installed_pytest_entry's (platform/tests/cmake/InstalledPytestEntry.cmake).
function(rocke_library_pytest_entry name)
  cmake_parse_arguments(PARSE_ARGV 1 _library
                        "WORKERS;GPU;COMPILER_TOOLS;COMPILER_REQUIREMENT"
                        "PYTHONPATH;TIMEOUT"
                        "TESTS;IGNORE;ARGUMENTS;LABELS")
  if(_library_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
            "rocke_library_pytest_entry(${name}): unknown arguments "
            "${_library_UNPARSED_ARGUMENTS}")
  endif()
  set(_tests_root "${ROCKE_TEST_CTEST_REL}tests/library/tests")
  list(TRANSFORM _library_TESTS PREPEND "${_tests_root}/")
  set(_arguments "")
  foreach(_ignored IN LISTS _library_IGNORE)
    list(APPEND _arguments --ignore "${_tests_root}/${_ignored}")
  endforeach()
  list(APPEND _arguments ${_library_ARGUMENTS})
  if(NOT DEFINED _library_PYTHONPATH)
    set(_library_PYTHONPATH "${ROCKE_TEST_CTEST_REL}tests/library")
  endif()
  set(_flags "")
  foreach(_flag IN ITEMS WORKERS GPU COMPILER_TOOLS COMPILER_REQUIREMENT)
    if(_library_${_flag})
      list(APPEND _flags ${_flag})
    endif()
  endforeach()
  rocke_installed_pytest_entry(_entry ${name} ${_flags}
    TESTS ${_library_TESTS}
    ARGUMENTS ${_arguments}
    LABELS ${_library_LABELS}
    ENVIRONMENT "PYTHONPATH=${_library_PYTHONPATH}"
    TIMEOUT ${_library_TIMEOUT})
  set(_ROCKE_INSTALLED_CTEST_ENTRIES "${_ROCKE_INSTALLED_CTEST_ENTRIES}${_entry}"
      PARENT_SCOPE)
endfunction()

# Oracle-free launches of the shipped dense kernels: the uniform-probability
# gate and the key-permutation relation. Only the GPU modules run here; their
# CPU contracts run in the attention CPU lane.
rocke_library_pytest_entry(rocke_oracle_free_pytest GPU
  TESTS oracle_free/test_analytical_dense.py oracle_free/test_permutation_dense.py
  ARGUMENTS -v -rs
  LABELS ckc python gpu standard
  TIMEOUT ${ROCKE_TEST_TIMEOUT_GPU})

# Library conv CPU tests (no GPU required; dispatch selection + signature guards).
# The test environment installs the platform wheel. PYTHONPATH puts the staged
# library tree first so kernels.*, builders.*, dispatch.*, and benchmarks.* do
# not collide with rocke's platform subpackages. Both the test paths and that
# tree are relative to the CTestTestfile's own directory, which sits one level
# below the install root in standalone mode -- hence the ROCKE_TEST_CTEST_REL
# prefix on each, matching the other installed tests.
set(_ROCKE_LIBRARY_CONV_TESTS
    "dispatch/test_grouped_conv_dispatch.py"
    "test_conv_wgrad_streamk.py"
    "test_conv_mma_operand_roles.py"
    "test_conv_problem_reference.py"
    "test_conv_kouter_fragment_reader.py"
    "test_wgrad_cached_candidate_filter.py"
    "test_conv_wgrad_prefetch_addr.py"
    "test_conv_wgrad_prefetch_global_regs.py"
    "test_conv_dgrad_spec_policy.py"
    "test_streamk_sweep_launch.py"
    "test_conv_build_signatures.py"
    "test_img2col_spec.py"
    "test_conv_aot_grid_limits.py"
    "test_dgrad_stem_wmma.py"
    "test_fwd_small_wmma.py"
    "test_fwd_small_mfma.py"
    "test_direct_dw_fwd_wmma.py"
    "test_direct_fwd_gstream_wmma.py"
    "test_direct_fwd_tapouter_wmma.py"
    "test_direct_dgrad_wmma.py"
    "test_direct_dgrad_pp_mfma.py"
    "test_direct_fwd_wmma.py"
    "test_direct_fwd_pp_mfma.py"
    "test_direct_fwd_flex_mfma.py"
    "test_conv_fwd_split_k.py"
    "test_conv_winograd_fwd_cpu.py"
    "test_conv_winograd_fwd_rt_cpu.py"
    "test_conv_winograd_gemm_wmma.py"
    "test_conv_winograd_gen_cpu.py"
    "test_conv_winograd_select.py"
    "dispatch/test_conv_winograd_dispatch.py"
    "dispatch/test_conv_winograd_aot_dispatch.py"
    "test_direct_kernel_cache.py"
    "test_direct_wgrad_refusals.py"
    "test_direct_depthwise_wgrad_cachepolicy.py"
    "test_conv_tr_lane_map.py"
    "test_direct_depthwise_fwd_cpl.py"
    "test_conv_direct_grouped_backend_parity.py"
    "test_conv_engine_bridge_sched.py"
    "test_direct_wgrad_specs.py"
    "test_conv_gradient_contract.py"
    "test_builder_signature_contract.py"
    "test_library_layering.py"
    "test_aot_launch_zeroing.py"
    "test_batched_timing.py"
    "test_conv_fill.py"
    "test_capability_keys.py"
    "test_module_cache.py"
    "test_run_conv_shards.py"
    "test_convolution_backward_data_aot_packaging.py"
    "test_conv_abi.py"
    "test_conv_batch_and_miopen_parse.py"
    "test_conv_dgrad_patch.py"
    "test_conv_dgrad_uniform_k_decode.py"
    "test_conv_fwd_group_merge_gate.py"
    "test_conv_implicit_gemm_vector_width.py"
    "test_conv_wgrad_group_merge_gate.py"
    "test_direct_depthwise_col_validate.py"
    "test_fwd_merge_window.py"
    "test_kernel_label.py")
# pytest-timeout applies per test item and is intentionally well below the
# CTest limit so a hung import or structural check produces pytest diagnostics.
# CTest remains the hard backstop for collection, cumulative runtime, and
# failures outside an individual test item. The exhaustive items of these files
# run in rocke_library_conv_exhaustive_pytest below instead.
rocke_library_pytest_entry(rocke_library_conv_pytest WORKERS
  TESTS ${_ROCKE_LIBRARY_CONV_TESTS}
  ARGUMENTS -m "not exhaustive and not gpu" -v -rs --timeout=30
  LABELS ckc python conv host standard
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})

# The exhaustive items of the conv files: the kernel-label round trip over
# every direct identity the AOT grid produces (about a million names, minutes
# of one core), and the forward AOT-packaging checks over the whole bf16 sweep.
# Scheduled CI only (the comprehensive tier); the conv entry above runs each
# capabilities row and knob setting once, and the AOT-packaging entry below the
# whole fp16 sweep.
rocke_library_pytest_entry(rocke_library_conv_exhaustive_pytest
  TESTS test_kernel_label.py test_convolution_forward_aot_packaging.py
  ARGUMENTS -m exhaustive -v -rs --timeout=600
  LABELS ckc python conv host comprehensive full
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})

# Library host unit tests that belong to no operator family: the shared
# dispatch helpers (rocke.dispatch.core), the benchmarks' early-stop rule and
# the occupancy calibration tool's parse/compare logic. No GPU, no compiler.
set(_ROCKE_LIBRARY_HOST_TESTS
    "test_build_time_imports.py"
    "dispatch/test_core_helpers.py"
    "test_early_stop.py"
    "test_calibrate_occupancy.py")
rocke_library_pytest_entry(rocke_library_host_pytest
  TESTS ${_ROCKE_LIBRARY_HOST_TESTS}
  ARGUMENTS -v -rs --timeout=30
  LABELS ckc python host standard
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})

# The forward AOT-packaging cross-check gets its own entry: it samples far more of
# the sweep's identity space than the other conv files above, so its slowest cases
# (building and comparing IR for a sampled instance) exceed the shared entry's
# per-test pytest-timeout; the dgrad twin above stays in the shared entry only
# because its own sample is smaller. Its exhaustive items (the whole bf16 sweep)
# run in rocke_library_conv_exhaustive_pytest above.
rocke_library_pytest_entry(rocke_library_conv_fwd_aot_packaging_pytest WORKERS
  TESTS test_convolution_forward_aot_packaging.py
  ARGUMENTS -m "not exhaustive" -v -rs --timeout=600
  LABELS ckc python conv host
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})

# Library conv IR golden, platform-regression, split-K planner, ceil_div and
# conv doc-example (the conv part of dsl_docs' verify_dsl_docs.py) CPU tests
# (no GPU). Their own entry, with the conv entry's arguments, labels and
# PYTHONPATH, so they do not add to rocke_library_conv_pytest's time, which is
# already long on Windows runners.
set(_ROCKE_LIBRARY_CONV_IR_TESTS
    "test_conv_ir_golden.py"
    "test_conv_platform_regressions.py"
    "test_conv_split_k.py"
    "test_conv_split_k_planner.py"
    "test_conv_split_k_wgrad.py"
    "test_ceil_div_callers.py"
    "test_verify_dsl_docs_conv.py")
rocke_library_pytest_entry(rocke_library_conv_ir_pytest
  TESTS ${_ROCKE_LIBRARY_CONV_IR_TESTS}
  ARGUMENTS -v -rs --timeout=30
  LABELS ckc python conv host standard
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})

# Library conv TF32 tests (no GPU required), as two entries:
#
#   rocke_library_conv_tf32_pytest: validation and rejections, kernel names,
#   the spec dicts, the register guard, the benchmark/AOT dtype plumbing,
#   the validate gate, and one or two builds and lowerings per kernel
#   family. Registered on every platform.
#
#   rocke_library_conv_tf32_lowering_pytest: the build-and-lower matrices
#   (every variant, atom and geometry on gfx942 and gfx950) and the comgr
#   ISA checks (skipped without comgr/llvm-objdump), the tests that
#   _ROCKE_LIBRARY_CONV_TF32_LOWERING_K names. Not registered on Windows:
#   every hipkernelprovider CI shard runs every entry, and the Windows
#   shards already run close to their step time limit. The lowering is the
#   same Python on every platform, so the Linux entry covers it; Windows
#   keeps the per-family builds and lowerings of the first entry.
#
# Both entries collect the same files and split them with one -k
# expression and its negation, so every test runs in exactly one of them; a
# test the expression does not name runs in the first. The GPU classes (all
# named *Gpu*, and marked gpu where they launch through the runtime directly)
# are deselected from both and run in the GPU list.
# test_conv_direct_grouped_backend_parity.py runs in
# rocke_library_conv_pytest, not here. Their own entries rather than part
# of rocke_library_conv_pytest: the build-and-lower matrices take tens of
# seconds, past that entry's per-test timeout. Both run in the standard tier
# (ROCKE_ENGINE_test_categories_external.yaml), and both bake in the standard
# label so `ctest -L standard` finds them without that file's labelling.
set(_ROCKE_LIBRARY_CONV_TF32_TESTS
    "test_conv_fwd_tf32.py"
    "test_conv_dgrad_tf32.py"
    "test_conv_wgrad_tf32.py"
    "test_conv_tf32_register_guard.py"
    "test_conv_direct_tf32_mfma.py"
    "test_conv_direct_tf32_scalar.py"
    "test_conv_direct_nongrouped_tf32.py"
    "test_benchmark_tf32_plumbing.py"
    "test_direct_tf32_plumbing.py"
    "test_conv_tf32_validate_gate.py")
# pytest -k matches case-insensitive substrings of the test, class and
# module names: "F32Isa" and "test_isa" are the ISA classes and tests.
set(_ROCKE_LIBRARY_CONV_TF32_LOWERING_K
    "test_forward_variants or (TestConvFwdTf32Build and test_matrix) or (TestTf32BuildMatrix and test_matrix) or (TestTf32Lowering and test_grid) or test_build_and_lower_matrix or test_fp32_tf32_build_lower_same_ir or F32Isa or test_isa")
rocke_library_pytest_entry(rocke_library_conv_tf32_pytest
  TESTS ${_ROCKE_LIBRARY_CONV_TF32_TESTS}
  ARGUMENTS -m "not gpu" -k "not Gpu and not (${_ROCKE_LIBRARY_CONV_TF32_LOWERING_K})"
            -v -rs --timeout=60
  LABELS ckc python conv host standard
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})
if(WIN32)
  message(STATUS "rocKE: rocke_library_conv_tf32_lowering_pytest not registered on Windows (CI shard time; see comment above)")
else()
  rocke_library_pytest_entry(rocke_library_conv_tf32_lowering_pytest
    TESTS ${_ROCKE_LIBRARY_CONV_TF32_TESTS}
    ARGUMENTS -m "not gpu" -k "not Gpu and (${_ROCKE_LIBRARY_CONV_TF32_LOWERING_K})"
              -v -rs --timeout=120
    LABELS ckc python conv host standard
    TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})
endif()

# The one test of test_attention_builds.py that needs no compiler tools: it runs
# in the CPU entry below, and the builds entry further down deselects it. Kept as
# the pytest node id relative to the library tests' rootdir (tests/library/tests,
# where their pytest.ini is), the form --deselect matches.
set(_ROCKE_LIBRARY_ATTENTION_BUILDS_HOST_TEST
    "test_attention_builds.py::TestAttentionHelpers::test_gfx950_signature_matches_the_built_kernels_params")

# Library attention CPU tests (no GPU required): gfx942 Q staging when K is
# read from HBM on the emitted IR, the paged attention tests' fp16/bf16-only
# host encoding, gfx950 LDS budget model,
# dense block-order decodes, the dense paged launch checks, spec-wrapper
# parity gate, the 3D decode split rule and K/V cache policy, the scalar
# fallback's cache key and block-table stride kernarg, the gfx950 dense prefill
# builder's spec construction, the unified attention running-max carry and keyless
# guards on the emitted IR, the tiled 2D LSE store on the emitted IR, the
# gfx950 tiled-2D K reads from LDS on the emitted IR, the
# gfx942 cfvst V loader's i64 K/V addressing on the emitted IR, the
# LDS alignment claims of the backward kernel with the slices dK / dV sink, the
# backward/gfx1250 MMA accumulator-input versus result-role regressions, the
# gfx1250 3D kernel's paged reads staying bounded to the sequence's last page, the
# host-side SDPA sequence-length and padding-mask adapters/specs, the
# fmha_mfma free-stride ABI and layout contracts, the 2D/3D mask_type spec
# fields, the gfx942 2D head counts and batch being IR-neutral at a fixed GQA
# ratio, the 3D segment+reduce module's pairing, the reduce's LDS segment
# bound against the compiled code object, the tiled C++ entries' head-count
# and arch refusals, the dense segment-combine launch ABI, the drift guard
# that pins the shared dense spec's C++ mirror to the Python, the
# oracle-free uniform-probability checker and subject contracts, the gfx942
# dense golden IR hashes (pure text lowering, a few seconds), the dense
# varlen-keys host checks, the gfx950 dense varlen K/V buffer bound on the emitted
# IR, and the backward-test memo bound in tests/conftest.py. Then the forward
# routing and admission checks: the spec builder's arch argument, the gfx950
# tiled-2D selector and its per-engine spec builders, the 2D LDS budget
# resolver, decode routing to the 3D kernel, non-power-of-two GQA ratios in the
# 2D and 3D gates, the gfx942 D128 sliding-window route, GQA head fold, sliced-K
# width, 4-warp exp2 softmax and 3D D192 LDS footprint, the dense LDS budgets
# (gfx942, D64 pad), dense bottom-right causal and LSE output checks, the
# split-KV workspace ownership, the strided (non-paged) KV decode contracts and
# LDS admission, the FLOP helper, the SDPA reference-transfer contract and the
# attention doc claims.
# File-list style (same pattern as conv_pytest) so PYTHONPATH resolves rocke/
# correctly from the installed tree.
set(_ROCKE_LIBRARY_ATTENTION_CPU_TESTS
    "test_attention_bwd_gemm.py"
    "test_attention_bwd_body.py"
    "test_attention_bwd_lds_alignment.py"
    "test_gfx1250_attention_operand_roles.py"
    "test_gfx1151_fmha_operand_roles.py"
    "test_gfx942_k_hbm_direct_q_staging.py"
    "test_attention_paged_reference.py"
    "test_attention_dense_decode.py"
    "test_attention_dense_gfx942_golden.py"
    "test_attention_dense_gfx950_lds.py"
    "test_attention_dense_prefill_builder_gfx950.py"
    "test_attention_dense_paged_launch.py"
    "test_attention_dense_gfx950_varlen_kv_bound.py"
    "test_attention_dense_spec_vectors.py"
    "test_attention_dense_varlen_keys.py"
    "test_attention_scalar_cache_key.py"
    "test_attention_spec_predicate_parity.py"
    "test_attention_tiled_2d_mask_type.py"
    "test_attention_tiled_2d_head_counts_gfx942.py"
    "test_attn_tiled_2d_gfx950_k_lds_read.py"
    "test_attention_tiled_3d_decode_split.py"
    "test_attention_tiled_3d_kv_cache_policy.py"
    "test_attention_tiled_launch_signature.py"
    "test_attention_tiled_cpp_entry_guards.py"
    "test_attention_dense_kv_split_combine_abi.py"
    "test_attention_tiled_3d_mask_type.py"
    "test_attention_tiled_3d_module.py"
    "test_attention_tiled_3d_module_gfx942.py"
    "test_attention_tiled_3d_reduce_lds.py"
    "test_attention_unified_masked_tile_ir.py"
    "test_attention_unified_2d_lse_ir.py"
    "test_gfx942_cfvst_i64_kv_addr.py"
    "test_d256_gfx942_fast_routing.py"
    "test_gfx1250_tiled_3d_bounds.py"
    "test_sdpa_seqlen.py"
    "test_sdpa_padding_mask.py"
    "test_fmha_mfma_free_stride.py"
    "test_sdpa_free_stride_layouts.py"
    "test_bwd_memo_bound.py"
    "oracle_free/test_contracts.py"
    "oracle_free/test_permutation_contracts.py"
    "test_attention_dense_gfx942.py"
    "test_attention_combo_sweep_fp8.py"
    "test_dense_prefill_emit_shapes.py"
    "dispatch/attention/test_review_followups.py"
    "test_arch_binding_guard.py"
    "test_selector_parity.py"
    "test_per_engine_spec_fns.py"
    "test_lds_budget_resolver.py"
    "test_attn_decode_route_3d.py"
    "test_attn_tiled_2d_nqk_nonpow2.py"
    "test_attn_tiled_3d_nqk_nonpow2.py"
    "test_gfx942_d128_sw_routing.py"
    "test_gfx942_gqa_head_fold.py"
    "test_attn_k_slice_hd.py"
    "test_gfx942_4wgqa_softmax_exp2fast.py"
    "test_gfx942_3d_d192_lds.py"
    "test_attention_dense_gfx942_lds.py"
    "test_attention_dense_d64_lds.py"
    "test_attention_dense_bottom_right.py"
    "test_attention_dense_lse.py"
    "test_attention_workspace.py"
    "test_strided_kv_decode.py"
    "test_strided_kv_lds.py"
    "test_attention_flops.py"
    "test_verify_dsl_docs_attention.py"
    # One host-only test of test_attention_builds.py (signature/params
    # mirroring); the rest of the file runs in
    # rocke_library_attention_builds_pytest below, which deselects this one.
    "${_ROCKE_LIBRARY_ATTENTION_BUILDS_HOST_TEST}")
rocke_library_pytest_entry(rocke_library_attention_cpu_pytest WORKERS
  TESTS ${_ROCKE_LIBRARY_ATTENTION_CPU_TESTS}
  ARGUMENTS -m "not gpu" -v -rs --timeout=30
  LABELS ckc python attention host standard
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})

# The rest of test_attention_builds.py (no GPU): the attention builders' compile
# smoke tests and the HSACO resource budgets of the built kernels (scratch,
# LDS, waves per SIMD), read with llvm-readelf / llvm-objdump. The entry takes
# the compiler environment, so a missing tool fails instead of skipping. It
# deselects _ROCKE_LIBRARY_ATTENTION_BUILDS_HOST_TEST, which
# rocke_library_attention_cpu_pytest runs.
#
# Not registered on Windows: the compiler tool environment is Linux-only (see
# platform/CMakeLists.txt), so the budget checks there would fail on the
# missing tools rather than run.
if(WIN32)
  message(STATUS "rocKE: rocke_library_attention_builds_pytest not registered on Windows (compiler tools; see comment above)")
else()
  rocke_library_pytest_entry(rocke_library_attention_builds_pytest
    WORKERS COMPILER_TOOLS COMPILER_REQUIREMENT
    TESTS test_attention_builds.py
    ARGUMENTS --deselect "${_ROCKE_LIBRARY_ATTENTION_BUILDS_HOST_TEST}" -v -rs --timeout=120
    LABELS ckc python attention host standard
    TIMEOUT ${ROCKE_TEST_TIMEOUT_TOOLCHAIN})
endif()

# The pure runner policy and the real subprocess/process-tree contracts run as
# two entries, each with its own whole-entry budget.
rocke_library_pytest_entry(rocke_library_runner_contract_pytest
  TESTS test_library_test_tiers.py
  ARGUMENTS -m "not worker_integration" -v -rs --timeout=30
  LABELS ckc python host runner comprehensive full
  PYTHONPATH "${_ROCKE_LIBRARY_PYTHONPATH}"
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})
rocke_library_pytest_entry(rocke_library_worker_integration_pytest
  TESTS test_library_test_tiers.py
  ARGUMENTS -m worker_integration -v -rs --timeout=30
  LABELS ckc python host runner integration comprehensive full
  PYTHONPATH "${_ROCKE_LIBRARY_PYTHONPATH}"
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})

# Tiled attention cache-key contract (no GPU). The targeted share/split
# checks take seconds. The invariant sweep compares key equality with build
# equality across every selector threshold for each tiled arch, so each sweep
# case runs for tens of seconds: it is a separate entry with its own budget,
# run by scheduled CI only (the comprehensive tier).
rocke_library_pytest_entry(rocke_library_attention_cache_key_pytest
  TESTS test_attention_tiled_cache_key.py
  ARGUMENTS -k "not test_key_matches_build_identity" -v -rs --timeout=30
  LABELS ckc python attention host standard
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})
rocke_library_pytest_entry(rocke_library_attention_cache_key_sweep_pytest
  TESTS test_attention_tiled_cache_key.py
  ARGUMENTS -k test_key_matches_build_identity -v -rs --timeout=240
  LABELS ckc python attention host comprehensive full
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})

# Exhaustive counted-sync gates are intentionally outside normal pytest
# discovery. The structural gate is host-only; the compiler gate additionally
# requires the reference COMGR/LLVM toolchain and llvm-objdump. They share a
# lock to cap aggregate host memory and compiler pressure in parallel runs.
# The gate conftest stops a partial run, so the gates run whole on every CI shard
# and must not load the shard plugin. platform/CMakeLists.txt reads this list
# when it checks the installed pytest entries.
set(_ROCKE_EXPLICIT_PYTEST_GATES
    rocke_library_counted_structural_gate
    rocke_library_counted_compiler_gate)
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_counted_structural_gate python -m pytest -p pytest_timeout \"${ROCKE_TEST_CTEST_REL}tests/library/tests/gates/attention_bwd_counted_structural_gate.py\" -v -rs --timeout=60)\n"
       "set_tests_properties(rocke_library_counted_structural_gate PROPERTIES\n"
       "  LABELS \"ckc;python;attention;host;structural;comprehensive;full;hip-kernel-provider\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${_ROCKE_LIBRARY_PYTHONPATH};PYTEST_ADDOPTS=;PYTEST_PLUGINS=;PYTEST_DISABLE_PLUGIN_AUTOLOAD=1;ROCKE_EXPLICIT_GATE_RUN=1;ROCKE_EXPLICIT_GATE_PARTIAL=0;ROCKE_EXPLICIT_GATE_PREFLIGHT=\"\n"
       "  PROCESSORS 4\n"
       "  RESOURCE_LOCK rocke_counted_sync_gate\n"
       "  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})\n"
       "set(_rocke_counted_isa_environment\n"
       "    \"PYTHONPATH=${_ROCKE_LIBRARY_PYTHONPATH}\"\n"
       "    \"ROCKE_COUNTED_ISA_OPTIONAL=0\"\n"
       "    \"PYTEST_ADDOPTS=\"\n"
       "    \"PYTEST_PLUGINS=\"\n"
       "    \"PYTEST_DISABLE_PLUGIN_AUTOLOAD=1\"\n"
       "    \"ROCKE_EXPLICIT_GATE_RUN=1\"\n"
       "    \"ROCKE_EXPLICIT_GATE_PARTIAL=0\"\n"
       "    \"ROCKE_EXPLICIT_GATE_PREFLIGHT=\"\n"
       "    \"ROCKE_BWD_NO_COMGR_PIN=\"\n"
       "    \"ROCKE_BWD_ISA_ARCH=\"\n"
       "    \"ROCKE_ISA_INSPECT_PROBE=${ROCKE_TEST_CTEST_REL}tests/support/probe_isa_inspect.py\")\n"
       "list(APPEND _rocke_counted_isa_environment\n"
       "     \${_rocke_compiler_tool_environment})\n"
       "add_test(rocke_library_counted_compiler_gate python -m pytest -p pytest_timeout \"${ROCKE_TEST_CTEST_REL}tests/library/tests/gates/attention_bwd_counted_compiler_gate.py\" -v -rs --timeout=1200)\n"
       "set_tests_properties(rocke_library_counted_compiler_gate PROPERTIES\n"
       "  LABELS \"ckc;python;attention;host;compiler;comprehensive;full;hip-kernel-provider\"\n"
       "  ENVIRONMENT \"\${_rocke_counted_isa_environment}\"\n"
       "  PROCESSORS 4\n"
       "  RESOURCE_LOCK rocke_counted_sync_gate\n"
       "  SKIP_RETURN_CODE 77\n"
       "  TIMEOUT ${ROCKE_TEST_TIMEOUT_COUNTED_COMPILER_GATE})\n")

# DSA lightning-indexer CPU tests (no GPU: validator and dispatch wiring; the
# emitted IR is pinned by the corpus family lightning_indexer, checked by
# rocke_golden_static). Same staged-library PYTHONPATH as the conv lane. The on-GPU numeric tests are enrolled in the GPU lane below.
set(_ROCKE_LIBRARY_DSA_TESTS
    "test_dsa_indexer_gfx942_spec.py"
    "dispatch/dsa/test_gfx942_wiring.py"
    "dispatch/dsa/test_gfx950_wiring.py")
rocke_library_pytest_entry(rocke_library_dsa_pytest
  TESTS ${_ROCKE_LIBRARY_DSA_TESTS}
  ARGUMENTS -v -rs --timeout=30
  LABELS ckc python dsa host standard
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})

# Library MLA CPU tests (no GPU required): kernel lowering/admission, dispatch
# wiring, the IR golden, the fp32 oracle, the benchmark shape loader, the
# shared read-ahead helper, and the compile-test skip policy. The prefill
# occupancy / agpr_acc guards compile through comgr and read the code object
# with llvm-objdump, so the entry also takes the compiler environment; with it
# they fail rather than skip, and the skip-policy tests check that they do. Same
# PYTHONPATH contract as the conv entry above; the staged library tree carries
# kernels/mla, dispatch/mla, builders/mla, benchmarks/common/mla_shapes.py and
# tests/golden/mla_prefill_gfx942_ir_sha256.json.
set(_ROCKE_LIBRARY_MLA_TESTS
    "test_mla_prefill_gfx942.py"
    "dispatch/mla/test_mla_prefill_gfx942_wiring.py"
    "test_mla_prefill_gfx942_golden.py"
    "test_mla_reference.py"
    "test_mla_benchmark_shapes.py"
    "test_read_ahead.py"
    "test_compiler_test_utils.py")
rocke_library_pytest_entry(rocke_library_mla_pytest
  WORKERS COMPILER_TOOLS COMPILER_REQUIREMENT
  TESTS ${_ROCKE_LIBRARY_MLA_TESTS}
  ARGUMENTS -v -rs --timeout=120
  LABELS ckc python mla host standard
  TIMEOUT ${ROCKE_TEST_TIMEOUT_TOOLCHAIN})

# The builder-surface contract: every builder another project names still
# imports, takes (spec, *, arch) and carries its required fields.
rocke_library_pytest_entry(rocke_contracts_pytest
  TESTS contracts/test_builder_surface.py
  ARGUMENTS -v -rs --timeout=60
  LABELS ckc python host standard
  PYTHONPATH "${_ROCKE_LIBRARY_PYTHONPATH}"
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})

rocke_library_pytest_entry(rocke_reference_common_pytest WORKERS
  TESTS test_reference_common.py
  ARGUMENTS -v -rs --timeout=30
  LABELS ckc python host standard
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})

# Library attention dispatch and RDNA WMMA CPU tests: dispatcher selection,
# signature guards, spec validation and the tuned-candidate contract. Pointed at the
# tests/dispatch/attention DIRECTORY, not a file list, so a new file there runs
# automatically; the RDNA WMMA spec tests and the unified LSE host contract sit
# outside it and are named beside it.
# The directory's GPU numerics file runs in rocke_library_gpu_pytest, so it is
# ignored here. Same PYTHONPATH rule as the conv entry.
#
# tests/dispatch/attention/ cannot shadow the real dispatch.attention package
# because library/tests/conftest.py puts the library root ahead of the test
# root on sys.path; importlib import mode alone does not prevent it
# (checked by tests/dispatch/attention/test_import_resolution.py).
rocke_library_pytest_entry(rocke_library_attention_pytest WORKERS
  TESTS test_wmma_fmha_fwd_v2_spec.py
        test_wmma_fmha_decode_spec.py
        test_attention_unified_lse.py
        dispatch/attention
  IGNORE dispatch/attention/test_gfx1151_attention_dispatch_numeric.py
  ARGUMENTS -v -rs --timeout=60
  LABELS ckc python attention dispatch host standard
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})

# Library SDPA CPU tests (no GPU): the hipDNN SDPA request contract and
# from_hipdnn, the request-driven and array oracles, and the forward/backward
# case tables. Pointed at the DIRECTORY so a new file there runs automatically;
# a GPU test added there must be --ignore'd here and enrolled in the GPU lane.
# Distinct from rocke_sdpa_reference_unit_pytest below, which runs only
# tests/test_sdpa_reference_contract.py. The rocKE umbrella sets the requirement
# when it stages the optional hipDNN assets; standalone platform builds leave it
# disabled.
set(_ROCKE_SDPA_PYTEST_ARGS "")
if(ROCKE_REQUIRE_INSTALLED_HIPDNN_SDPA_ASSETS)
  set(_ROCKE_SDPA_PYTEST_ARGS --rocke-require-hipdnn-assets)
endif()
rocke_library_pytest_entry(rocke_library_sdpa_pytest WORKERS
  TESTS sdpa
  IGNORE sdpa/test_unified_sdpa_numeric.py sdpa/test_dense_sdpa_numeric.py
  ARGUMENTS ${_ROCKE_SDPA_PYTEST_ARGS} -v -rs --timeout=30
  LABELS ckc python sdpa host standard
  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})

# GDN/KDA decode CPU tests: spec validation, IR emission and load order,
# the golden IR hashes, the performance knobs, dispatch selection, and the
# sweep / defaults-report tools and their shared timing helpers; the GDN
# decode and prefill host guards; the KDA chunkwise specs, decode reference
# and resources, host packing, golden IR hashes and dispatch wiring. None
# needs a GPU. The fp32 pass test and the rotation sizing are torch-free
# (dispatch/gdn/gate_rules.py) and tested without torch, as is how the
# sweep and both benchmarks act on a failed verdict. The tests that build
# CPU tensors (the driver's tensor code) need torch: where torch is
# missing they skip (see -rs) instead of running, and the tensor
# reductions inside gate() are then covered only by the GPU numeric suites
# in rocke_library_gpu_pytest. Same PYTHONPATH rule as the conv entry.
#
# Not registered on Windows. The Windows CI shard runs close to its time
# limit, and test_gdn_decode_fused.py::test_lowers_and_compiles_without_scratch
# ran close to the entry's per-test limit (then 60 s) on Windows runners. Re-enable on
# Windows once that case is cheaper or split out. The GDN/KDA numeric files
# in rocke_library_gpu_pytest need gfx950/gfx942 devices and already skip
# at collection on the Windows GPU lanes, so they are unchanged. This entry
# stays registered on every other platform.
set(_ROCKE_LIBRARY_GDN_TESTS
    "test_gdn_decode_spec.py"
    "test_gdn_decode_fused.py"
    "test_gdn_decode_load_order.py"
    "test_gdn_decode_golden.py"
    "test_gdn_decode_knobs.py"
    "test_gdn_decode_timing.py"
    "test_gdn_decode_table_sweep.py"
    "test_gdn_decode_defaults_report.py"
    "test_kda_decode_benchmark.py"
    "dispatch/gdn/test_gfx950_registry.py"
    "dispatch/gdn/test_gfx950_wiring.py"
    "dispatch/gdn/test_tuning_contract.py"
    "dispatch/gdn/test_cli_args.py"
    "dispatch/gdn/test_gate_rules.py"
    "dispatch/gdn/test_gfx950_prefill_wiring.py"
    "test_gdn_decode_prepare.py"
    "test_gdn_prefill_decay_guard.py"
    "test_kda_chunkwise_spec.py"
    "test_kda_chunkwise_gfx942_spec.py"
    "test_kda_decode_reference.py"
    "test_kda_decode_resources.py"
    "test_kda_gfx942_golden.py"
    "test_kda_gfx950_golden.py"
    "test_kda_hostpack.py"
    "dispatch/kda/test_gfx942_wiring.py"
    "dispatch/kda/test_gfx950_wiring.py")
if(WIN32)
  message(STATUS "rocKE: rocke_library_gdn_pytest not registered on Windows (slow GDN fused case; see comment above)")
else()
  # The scratch and resource checks compile through comgr and read the code
  # object with llvm-objdump, so the entry takes the compiler environment (as
  # rocke_library_mla_pytest does); with it a missing tool fails rather than
  # skips. Two compile cases of test_gdn_decode_fused.py take about 30 s each
  # with the tools on one host core, so the per-test limit is 120 s, keeping
  # them under half of it.
  rocke_library_pytest_entry(rocke_library_gdn_pytest
    WORKERS COMPILER_TOOLS COMPILER_REQUIREMENT
    TESTS ${_ROCKE_LIBRARY_GDN_TESTS}
    ARGUMENTS -m "not gpu" -v -rs --timeout=120
    LABELS ckc python gdn host standard
    TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})
endif()

# The C++/Python engine parity checks of library kernels, through two different
# C++ paths:
#
#   test_attention_ir_cpp_parity.py and test_gdn_decode_ir_cpp_parity.py lower
#   the attention, lightning-indexer, MLA prefill and GDN/KDA decode kernels
#   with both engines and compare the LLVM IR byte for byte. Their C++ side is
#   the shared engine (librocke, through rocke.core.native_engine), which a
#   Linux provider build installs beside the rocke package, so they run in CI.
#
#   test_selector_cpp_parity.py compares the gfx950 tiled-2D selector with its
#   C++ twin through the engine's Python module (rocke_engine), which only a
#   configure with ROCKE_BUILD_PYBIND=ON builds; a provider build does not, so
#   there its cases skip.
#
# Without its C++ engine a file's cases skip, and rocke_engine_modules_check
# says why. The GDN/KDA decode case takes about 20 s on one host core (it lowers
# 341 kernels at four LLVM flavors with each engine), so the per-test limit is
# 120 s. The selector's full grid (its exhaustive items) runs in the
# comprehensive entry below, registered only where the build makes
# rocke_engine: anywhere else it would run nothing.
#
# Not registered on Windows: provider builds there make neither the shared
# engine nor the engine modules, so every case would skip.
set(_ROCKE_LIBRARY_ENGINE_PARITY_TESTS
    "test_selector_cpp_parity.py"
    "test_attention_ir_cpp_parity.py"
    "test_gdn_decode_ir_cpp_parity.py")
if(WIN32)
  message(STATUS "rocKE: rocke_library_engine_parity_pytest not registered on Windows (no C++ engine; see comment above)")
else()
  rocke_library_pytest_entry(rocke_library_engine_parity_pytest
    TESTS ${_ROCKE_LIBRARY_ENGINE_PARITY_TESTS}
    ARGUMENTS -m "not exhaustive" -v -rs --timeout=120
    LABELS ckc python host standard
    PYTHONPATH "${_ROCKE_LIBRARY_PYTHONPATH}"
    TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})
endif()
if(ROCKE_BUILD_PYBIND AND NOT WIN32)
  rocke_library_pytest_entry(rocke_library_engine_parity_exhaustive_pytest
    TESTS test_selector_cpp_parity.py
    ARGUMENTS -m exhaustive -v -rs --timeout=60
    LABELS ckc python host comprehensive full
    PYTHONPATH "${_ROCKE_LIBRARY_PYTHONPATH}"
    TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})
else()
  message(STATUS "rocKE: rocke_library_engine_parity_exhaustive_pytest not registered (the build makes no rocke_engine; see comment above)")
endif()

# Library attention GPU numeric tests (gfx950 kernels, and the unified mask
# tests' scalar-kernel cases, which run on any GPU; require a GPU).
# Separate from the omnibus GPU suite so it can be targeted on its own when only
# attention hardware is available: select it by name (`ctest -R
# rocke_library_attention_gpu_pytest`), as `-L gpu` and `-L attention` also
# match other entries.
set(_ROCKE_LIBRARY_ATTENTION_GPU_TESTS
    "test_attention_dense_gfx950_numeric.py"
    "test_attn_tiled_2d_gfx950_fp8_k_lds_numeric.py"
    "test_extended_parity_attention.py"
    "test_attention_unified_mask_gfx950_numeric.py"
    "test_attention_unified_swa_sink_gfx950_numeric.py"
    # Each score modifier and each transposed 32x32 route once; the mixes run
    # in rocke_library_attention_biased_mixes_gpu_pytest below.
    "test_attention_tiled_2d_gfx950_biased_numeric.py::test_each_modifier_on_each_route")
rocke_library_pytest_entry(rocke_library_attention_gpu_pytest GPU
  TESTS ${_ROCKE_LIBRARY_ATTENTION_GPU_TESTS}
  ARGUMENTS -v -rs
  LABELS ckc python attention gpu
  TIMEOUT ${ROCKE_TEST_TIMEOUT_GPU})

# The score-modifier mixes of the gfx950 transposed 32x32 routes (the exhaustive
# items of test_attention_tiled_2d_gfx950_biased_numeric.py). Scheduled CI only
# (the comprehensive tier); the entry above runs each modifier and route once.
rocke_library_pytest_entry(rocke_library_attention_biased_mixes_gpu_pytest GPU
  TESTS test_attention_tiled_2d_gfx950_biased_numeric.py::test_modifier_mixes
  ARGUMENTS -v -rs
  LABELS ckc python attention gpu comprehensive full
  TIMEOUT ${ROCKE_TEST_TIMEOUT_GPU})

# gfx950 additive-bias numeric lane. Its own entry (selected by name, like the
# one above) so its runtime does not eat into that entry's timeout.
rocke_library_pytest_entry(rocke_library_attention_bias_gpu_pytest GPU
  TESTS test_attention_additive_bias_gfx950_numeric.py
  ARGUMENTS -v -rs
  LABELS ckc python attention gpu
  TIMEOUT ${ROCKE_TEST_TIMEOUT_GPU})

# Library GPU numeric suites: convolution, attention, KDA, and GDN. The gfx950
# dense numeric and extended-parity attention tests run only in
# rocke_library_attention_gpu_pytest above, not again here.
# Select files rather than the "gpu" pytest marker because several runtime
# tests lack that marker. A file whose other tests run in a host entry is named
# by its gpu-marked tests' node ids instead: the host entry deselects them with
# -m "not gpu", so each runs here only. Their dependency and architecture
# checks report skips when the required torch installation or GPU is
# unavailable; the RDNA WMMA suites skip on devices without a gfx11/gfx12
# generic target.
set(_ROCKE_LIBRARY_GPU_TESTS
    "test_conv_fwd_correctness.py"
    "test_conv_gradient_correctness.py"
    "test_conv_wgrad_correctness.py"
    "test_conv_wgrad_streamk_gpu.py"
    "test_conv_dgrad_correctness.py"
    "test_direct_conv_correctness.py"
    "test_conv_dgrad_depthwise_windowed.py"
    "test_direct_depthwise_dgrad_v2.py"
    "test_conv_dgrad_4c.py"
    "test_conv_dgrad_fused_weights.py"
    "test_direct_mfma_dgrad_correctness.py"
    "test_direct_dw_fwd_wmma.py::TestDwFwdWmmaGpu"
    "test_direct_fwd_gstream_wmma.py::TestFwdGstreamGpu"
    "test_direct_fwd_tapouter_wmma.py::TestTapOuterGpu"
    "test_conv_winograd_fwd_gpu.py"
    "test_conv_winograd_fwd_rt_gpu.py"
    "test_conv_winograd_gemm_wmma.py::TestGemmGpu"
    "test_conv_winograd_gen_gpu.py"
    "test_conv_winograd_dispatch_gpu.py"
    "test_conv_implicit_gemm_numeric.py"
    "test_convolution_forward_aot_numeric.py"
    "test_conv_fwd_split_k.py::TestSplitKNumerics"
    "test_conv_fwd_split_k.py::TestSplitKAutoNumerics"
    "test_conv_fwd_split_k.py::TestClusterSplitKNumerics"
    "test_dgrad_stem_wmma.py::TestStemGpu"
    "test_direct_depthwise_fwd_cpl.py::TestDepthwiseFwdCplGpu"
    "test_direct_dgrad_pp_mfma.py::test_dgrad_matches_the_reference"
    "test_direct_dgrad_wmma.py::TestDgradWmmaGpu"
    "test_direct_fwd_flex_mfma.py::test_forward_matches_the_reference"
    "test_direct_fwd_pp_mfma.py::test_forward_matches_the_reference"
    "test_direct_fwd_wmma.py::TestFwdWmmaGpu"
    "test_fwd_small_mfma.py::TestSmallGpu"
    "test_fwd_small_wmma.py::TestSmallGpu"
    "test_streamk_sweep_launch.py::TestStreamKSweepLaunchGpu"
    "test_conv_direct_tf32_mfma.py::TestDirectTf32Gpu"
    "test_conv_direct_tf32_scalar.py::TestGpuCorrectness"
    # The sweep sampler (test_sweep_candidates), an exhaustive item, runs in
    # rocke_library_conv_sweep_gpu_pytest below.
    "test_conv_direct_nongrouped_tf32.py::TestNongroupedF32Gpu::test_cases"
    "test_conv_direct_nongrouped_tf32.py::TestNongroupedF32Gpu::test_one_binary_serves_every_shape"
    "test_attention_dense_gfx942_numeric.py"
    "test_attention_dense_strict_numeric.py"
    "test_sdpa_declines_gpu.py"
    "test_attention_scalar_block_table_numeric.py"
    "test_attention_unified_lse_numeric.py"
    "test_strided_kv_decode_numeric.py"
    "test_strided_kv_decode.py::test_dense_decode_binding_views_device_arrays"
    "test_fmha_mfma_free_stride_numeric.py"
    "sdpa/test_unified_sdpa_numeric.py"
    "sdpa/test_dense_sdpa_numeric.py"
    "test_attn_bf16_d128_ring.py"
    "test_dtype_dispatch.py"
    "test_gdn_decode_gfx950_numeric.py"
    "test_gdn_decode_fused_gfx950_numeric.py"
    "test_gdn_decode_knobs_gfx950_numeric.py"
    "test_kda_decode_gfx950_numeric.py"
    "test_gdn_decode_timing.py::test_value_sets_are_independent_copies"
    "test_gfx1250_attention.py"
    "test_gfx942_gqa_head_fold_numeric.py"
    "test_gfx942_cfvst_i64_kv_addr_numeric.py"
    "test_gfx942_4wgqa_bounds_numeric.py"
    "test_attention_gfx942_kv_pages.py"
    "test_attention_tiled_runtime_shape_numeric.py"
    "test_gfx942_k_hbm_direct_numeric.py"
    "test_attention_tiled_3d_kv_lds_swizzle.py"
    "test_attention_tiled_3d_running_max.py"
    "test_attention_unified_2d_masked_tile_numeric.py"
    "test_attention_unified_scalar_masked_tile_numeric.py"
    "test_attention_tiled_3d_wave_page_id_load.py"
    "test_kda_chunkwise_gfx942_numeric.py"
    "test_kda_chunkwise_gfx950_numeric.py"
    "test_kda_gdn_gfx950_numeric.py"
    "test_mla_prefill_gfx942_numeric.py"
    "test_wmma_fmha_fwd_v2_numeric.py"
    "test_wmma_fmha_fwd_v2_features.py"
    "test_wmma_fmha_decode_numeric.py"
    "test_wmma_fmha_paged_guard.py"
    "test_wmma_fp8_decode_numeric.py"
    "dispatch/attention/test_gfx1151_attention_dispatch_numeric.py"
    "test_dsa_indexer_numeric.py")
# The exhaustive items of these files (the conv dgrad sweep's full sample) run
# in rocke_library_conv_dgrad_sweep_gpu_pytest below instead.
rocke_library_pytest_entry(rocke_library_gpu_pytest WORKERS GPU
  TESTS ${_ROCKE_LIBRARY_GPU_TESTS}
  ARGUMENTS -m "not exhaustive" -v -rs
  LABELS ckc python gpu
  TIMEOUT ${ROCKE_TEST_TIMEOUT_GPU})

# Every candidate of the conv dgrad sweep's pinned sample (the exhaustive class
# of test_conv_dgrad_correctness.py): a few thousand compiles. Scheduled CI only
# (the comprehensive tier); the entry above checks the first candidates of each
# pipeline and epilogue class of the same sample.
rocke_library_pytest_entry(rocke_library_conv_dgrad_sweep_gpu_pytest WORKERS GPU
  TESTS test_conv_dgrad_correctness.py
  ARGUMENTS -m exhaustive -v -rs
  LABELS ckc python conv gpu comprehensive full
  TIMEOUT ${ROCKE_TEST_TIMEOUT_GPU})

# The fp32/tf32 non-grouped direct-conv sweep sampler (the exhaustive item of
# test_conv_direct_nongrouped_tf32.py): it compiles and runs a sample of the
# exact-fp32 sweep grid, 14 subtests (measured wall time on gfx942: about
# 7 s). Scheduled CI only (the comprehensive tier); the entry above runs the
# file's fixed cases and its one-binary check.
rocke_library_pytest_entry(rocke_library_conv_sweep_gpu_pytest GPU
  TESTS test_conv_direct_nongrouped_tf32.py::TestNongroupedF32Gpu::test_sweep_candidates
  ARGUMENTS -v -rs
  LABELS ckc python conv gpu comprehensive full
  TIMEOUT ${ROCKE_TEST_TIMEOUT_GPU})
