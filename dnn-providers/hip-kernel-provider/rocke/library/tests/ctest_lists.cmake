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

# Oracle-free uniform-probability launches of the shipped dense kernels. Only
# the GPU module runs here; its CPU contracts run in the attention CPU lane.
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_oracle_free_pytest python -m pytest \"${ROCKE_TEST_CTEST_REL}tests/library/tests/oracle_free/test_analytical_dense.py\" -v -rs)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_oracle_free_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;gpu\\;quick\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
       "  TIMEOUT 300)\n")

# Library conv CPU tests (no GPU required; dispatch selection + signature guards).
# The test environment installs the platform wheel. PYTHONPATH puts the staged
# library tree first so kernels.*, builders.*, dispatch.*, and benchmarks.* do
# not collide with rocke's platform subpackages. Both the test paths and that
# tree are relative to the CTestTestfile's own directory, which sits one level
# below the install root in standalone mode -- hence the ROCKE_TEST_CTEST_REL
# prefix on each, matching the other installed tests.
set(_ROCKE_LIBRARY_CONV_TEST_RELPATHS
    "tests/library/tests/dispatch/test_grouped_conv_dispatch.py"
    "tests/library/tests/test_conv_wgrad_streamk.py"
    "tests/library/tests/test_conv_mma_operand_roles.py"
    "tests/library/tests/test_conv_kouter_fragment_reader.py"
    "tests/library/tests/test_wgrad_cached_candidate_filter.py"
    "tests/library/tests/test_conv_wgrad_prefetch_addr.py"
    "tests/library/tests/test_conv_wgrad_prefetch_global_regs.py"
    "tests/library/tests/test_conv_dgrad_spec_policy.py"
    "tests/library/tests/test_streamk_sweep_launch.py"
    "tests/library/tests/test_conv_build_signatures.py"
    "tests/library/tests/test_conv_aot_grid_limits.py"
    "tests/library/tests/test_dgrad_stem_wmma.py"
    "tests/library/tests/test_fwd_small_wmma.py"
    "tests/library/tests/test_fwd_small_mfma.py"
    "tests/library/tests/test_direct_dw_fwd_wmma.py"
    "tests/library/tests/test_direct_fwd_gstream_wmma.py"
    "tests/library/tests/test_direct_fwd_tapouter_wmma.py"
    "tests/library/tests/test_direct_dgrad_wmma.py"
    "tests/library/tests/test_direct_fwd_wmma.py"
    "tests/library/tests/test_conv_fwd_split_k.py"
    "tests/library/tests/test_conv_winograd_fwd_cpu.py"
    "tests/library/tests/test_conv_winograd_fwd_rt_cpu.py"
    "tests/library/tests/test_conv_winograd_gemm_wmma.py"
    "tests/library/tests/test_conv_winograd_gen_cpu.py"
    "tests/library/tests/test_conv_winograd_select.py"
    "tests/library/tests/dispatch/test_conv_winograd_dispatch.py"
    "tests/library/tests/dispatch/test_conv_winograd_aot_dispatch.py"
    "tests/library/tests/test_direct_kernel_cache.py"
    "tests/library/tests/test_direct_wgrad_refusals.py"
    "tests/library/tests/test_direct_depthwise_wgrad_cachepolicy.py"
    "tests/library/tests/test_conv_tr_lane_map.py"
    "tests/library/tests/test_direct_depthwise_fwd_cpl.py"
    "tests/library/tests/test_conv_direct_grouped_backend_parity.py"
    "tests/library/tests/test_conv_engine_bridge_sched.py"
    "tests/library/tests/test_direct_wgrad_specs.py"
    "tests/library/tests/test_conv_gradient_contract.py"
    "tests/library/tests/test_builder_signature_contract.py"
    "tests/library/tests/test_library_layering.py"
    "tests/library/tests/test_aot_launch_zeroing.py"
    "tests/library/tests/test_batched_timing.py"
    "tests/library/tests/test_conv_fill.py"
    "tests/library/tests/test_capability_keys.py"
    "tests/library/tests/test_module_cache.py"
    "tests/library/tests/test_run_conv_shards.py"
    "tests/library/tests/test_convolution_backward_data_aot_packaging.py")
set(_ROCKE_LIBRARY_CONV_TESTS "")
foreach(_rocke_conv_test ${_ROCKE_LIBRARY_CONV_TEST_RELPATHS})
  string(APPEND _ROCKE_LIBRARY_CONV_TESTS
         " \"${ROCKE_TEST_CTEST_REL}${_rocke_conv_test}\"")
endforeach()
# pytest-timeout applies per test item and is intentionally well below the
# CTest limit so a hung import or structural check produces pytest diagnostics.
# CTest remains the hard backstop for collection, cumulative runtime, and
# failures outside an individual test item.
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_conv_pytest python -m pytest${_ROCKE_LIBRARY_CONV_TESTS} -v -rs --timeout=30)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_library_conv_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;conv\\;host\\;quick\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
        "  TIMEOUT 180)\n")

# Library conv IR golden, platform-regression, split-K planner, ceil_div and
# conv doc-example (the conv part of dsl_docs' verify_dsl_docs.py) CPU tests
# (no GPU). Their own entry, with the conv entry's arguments, labels and
# PYTHONPATH, so they do not share rocke_library_conv_pytest's CTest budget,
# which is already close to its limit on Windows runners.
set(_ROCKE_LIBRARY_CONV_IR_TEST_RELPATHS
    "tests/library/tests/test_conv_ir_golden.py"
    "tests/library/tests/test_conv_platform_regressions.py"
    "tests/library/tests/test_conv_split_k.py"
    "tests/library/tests/test_conv_split_k_planner.py"
    "tests/library/tests/test_conv_split_k_wgrad.py"
    "tests/library/tests/test_ceil_div_callers.py"
    "tests/library/tests/test_verify_dsl_docs_conv.py")
set(_ROCKE_LIBRARY_CONV_IR_TESTS "")
foreach(_rocke_conv_ir_test ${_ROCKE_LIBRARY_CONV_IR_TEST_RELPATHS})
  string(APPEND _ROCKE_LIBRARY_CONV_IR_TESTS
         " \"${ROCKE_TEST_CTEST_REL}${_rocke_conv_ir_test}\"")
endforeach()
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_conv_ir_pytest python -m pytest${_ROCKE_LIBRARY_CONV_IR_TESTS} -v -rs --timeout=30)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_library_conv_ir_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;conv\\;host\\;quick\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
       "  TIMEOUT 180)\n")

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
# named *Gpu*) are deselected from both and run in the GPU list.
# test_conv_direct_grouped_backend_parity.py runs in
# rocke_library_conv_pytest, not here. Their own entries rather than part
# of rocke_library_conv_pytest: the build-and-lower matrices take tens of
# seconds, past that entry's per-test timeout. Both run in the quick and
# ffm-quick tiers (ROCKE_ENGINE_test_categories_external.yaml), and both
# bake in the quick label so `ctest -L quick` finds them without that
# file's labelling.
set(_ROCKE_LIBRARY_CONV_TF32_TEST_RELPATHS
    "tests/library/tests/test_conv_fwd_tf32.py"
    "tests/library/tests/test_conv_dgrad_tf32.py"
    "tests/library/tests/test_conv_wgrad_tf32.py"
    "tests/library/tests/test_conv_tf32_register_guard.py"
    "tests/library/tests/test_conv_direct_tf32_mfma.py"
    "tests/library/tests/test_conv_direct_tf32_scalar.py"
    "tests/library/tests/test_conv_direct_nongrouped_tf32.py"
    "tests/library/tests/test_benchmark_tf32_plumbing.py"
    "tests/library/tests/test_direct_tf32_plumbing.py"
    "tests/library/tests/test_conv_tf32_validate_gate.py")
# pytest -k matches case-insensitive substrings of the test, class and
# module names: "F32Isa" and "test_isa" are the ISA classes and tests.
set(_ROCKE_LIBRARY_CONV_TF32_LOWERING_K
    "test_forward_variants or (TestConvFwdTf32Build and test_matrix) or (TestTf32BuildMatrix and test_matrix) or (TestTf32Lowering and test_grid) or test_build_and_lower_matrix or test_fp32_tf32_build_lower_same_ir or F32Isa or test_isa")
set(_ROCKE_LIBRARY_CONV_TF32_TESTS "")
foreach(_rocke_conv_tf32_test ${_ROCKE_LIBRARY_CONV_TF32_TEST_RELPATHS})
  string(APPEND _ROCKE_LIBRARY_CONV_TF32_TESTS
         " \"${ROCKE_TEST_CTEST_REL}${_rocke_conv_tf32_test}\"")
endforeach()
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_conv_tf32_pytest python -m pytest${_ROCKE_LIBRARY_CONV_TF32_TESTS} -k \"not Gpu and not (${_ROCKE_LIBRARY_CONV_TF32_LOWERING_K})\" -v -rs --timeout=60)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_library_conv_tf32_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;conv\\;host\\;quick\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
       "  TIMEOUT 300)\n")
if(WIN32)
  message(STATUS "rocKE: rocke_library_conv_tf32_lowering_pytest not registered on Windows (CI shard time; see comment above)")
else()
  string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
         "add_test(rocke_library_conv_tf32_lowering_pytest python -m pytest${_ROCKE_LIBRARY_CONV_TF32_TESTS} -k \"not Gpu and (${_ROCKE_LIBRARY_CONV_TF32_LOWERING_K})\" -v -rs --timeout=120)\n")
  string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
         "set_tests_properties(rocke_library_conv_tf32_lowering_pytest PROPERTIES\n"
         "  LABELS \"ckc\\;python\\;conv\\;host\\;quick\"\n"
         "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
         "  TIMEOUT 600)\n")
endif()

# Library attention CPU tests (no GPU required): gfx950 LDS budget model,
# dense block-order decodes, spec-wrapper parity gate, the 3D decode split rule
# and K/V cache policy, the scalar fallback's cache key and block-table
# stride kernarg, the unified attention running-max carry and keyless
# guards on the emitted IR, the tiled 2D LSE store on the emitted IR, the
# gfx942 cfvst V loader's i64 K/V addressing on the emitted IR, the
# backward/gfx1250 MMA accumulator-input versus result-role regressions, the
# gfx1250 3D kernel's paged reads staying bounded to the sequence's last page, the
# host-side SDPA sequence-length and padding-mask adapters/specs, the
# fmha_mfma free-stride ABI and layout contracts, the 2D/3D mask_type spec
# fields, the 3D segment+reduce module's pairing, the reduce's LDS segment
# bound against the compiled code object, the tiled C++ entries' head-count
# and arch refusals, the dense segment-combine launch ABI, the drift guard
# that pins the shared dense spec's C++ mirror to the Python, the
# oracle-free uniform-probability checker and subject contracts, the gfx942
# dense golden IR hashes (pure text lowering, a few seconds), the dense
# varlen-keys host checks, the gfx950 dense varlen K/V buffer bound on the emitted
# IR, and the backward-test memo bound in tests/conftest.py.
# File-list style (same pattern as conv_pytest) so PYTHONPATH resolves rocke/
# correctly from the installed tree.
set(_ROCKE_LIBRARY_ATTENTION_CPU_TEST_RELPATHS
    "tests/library/tests/test_attention_bwd_gemm.py"
    "tests/library/tests/test_attention_bwd_body.py"
    "tests/library/tests/test_gfx1250_attention_operand_roles.py"
    "tests/library/tests/test_gfx1151_fmha_operand_roles.py"
    "tests/library/tests/test_attention_dense_decode.py"
    "tests/library/tests/test_attention_dense_gfx942_golden.py"
    "tests/library/tests/test_attention_dense_gfx950_lds.py"
    "tests/library/tests/test_attention_dense_gfx950_varlen_kv_bound.py"
    "tests/library/tests/test_attention_dense_spec_vectors.py"
    "tests/library/tests/test_attention_dense_gfx942_golden.py"
    "tests/library/tests/test_attention_dense_varlen_keys.py"
    "tests/library/tests/test_attention_dense_varlen_keys.py"
    "tests/library/tests/test_attention_scalar_cache_key.py"
    "tests/library/tests/test_attention_spec_predicate_parity.py"
    "tests/library/tests/test_attention_tiled_2d_mask_type.py"
    "tests/library/tests/test_attention_tiled_3d_decode_split.py"
    "tests/library/tests/test_attention_tiled_3d_kv_cache_policy.py"
    "tests/library/tests/test_attention_tiled_launch_signature.py"
    "tests/library/tests/test_attention_tiled_cpp_entry_guards.py"
    "tests/library/tests/test_attention_dense_kv_split_combine_abi.py"
    "tests/library/tests/test_attention_tiled_3d_mask_type.py"
    "tests/library/tests/test_attention_tiled_3d_module.py"
    "tests/library/tests/test_attention_tiled_3d_module_gfx942.py"
    "tests/library/tests/test_attention_tiled_3d_reduce_lds.py"
    "tests/library/tests/test_attention_unified_masked_tile_ir.py"
    "tests/library/tests/test_attention_unified_2d_lse_ir.py"
    "tests/library/tests/test_gfx942_cfvst_i64_kv_addr.py"
    "tests/library/tests/test_d256_gfx942_fast_routing.py"
    "tests/library/tests/test_gfx1250_tiled_3d_bounds.py"
    "tests/library/tests/test_sdpa_seqlen.py"
    "tests/library/tests/test_sdpa_padding_mask.py"
    "tests/library/tests/test_fmha_mfma_free_stride.py"
    "tests/library/tests/test_sdpa_free_stride_layouts.py"
    "tests/library/tests/test_bwd_memo_bound.py"
    "tests/library/tests/oracle_free/test_contracts.py")
set(_ROCKE_LIBRARY_ATTENTION_CPU_TESTS "")
foreach(_t ${_ROCKE_LIBRARY_ATTENTION_CPU_TEST_RELPATHS})
  string(APPEND _ROCKE_LIBRARY_ATTENTION_CPU_TESTS
         " \"${ROCKE_TEST_CTEST_REL}${_t}\"")
endforeach()
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_attention_cpu_pytest python -m pytest${_ROCKE_LIBRARY_ATTENTION_CPU_TESTS} -v -rs --timeout=30)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_library_attention_cpu_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;attention\\;host\\;quick\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
       "  TIMEOUT 180)\n")

# Pure runner policy remains quick. Real subprocess/process-tree contracts have
# a separate standard-tier owner and whole-entry budget.
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_runner_contract_pytest python -m pytest \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_library_test_tiers.py\" -m \"not worker_integration\" -v -rs --timeout=30)\n"
       "set_tests_properties(rocke_library_runner_contract_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;host\\;runner\\;comprehensive\\;full\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${_ROCKE_LIBRARY_PYTHONPATH}\"\n"
       "  TIMEOUT 180)\n"
       "add_test(rocke_library_worker_integration_pytest python -m pytest \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_library_test_tiers.py\" -m worker_integration -v -rs --timeout=30)\n"
       "set_tests_properties(rocke_library_worker_integration_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;host\\;runner\\;integration\\;comprehensive\\;full\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${_ROCKE_LIBRARY_PYTHONPATH}\"\n"
       "  TIMEOUT 180)\n")

# Tiled attention cache-key contract (no GPU). The targeted share/split
# checks take seconds and run in the quick tier. The invariant sweep compares
# key equality with build equality across every selector threshold for each
# tiled arch, so each sweep case runs for tens of seconds: it is a separate
# standard-tier entry with its own budget.
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_attention_cache_key_pytest python -m pytest \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_attention_tiled_cache_key.py\" -k \"not test_key_matches_build_identity\" -v -rs --timeout=30)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_library_attention_cache_key_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;attention\\;host\\;quick\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
       "  TIMEOUT 180)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_attention_cache_key_sweep_pytest python -m pytest \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_attention_tiled_cache_key.py\" -k test_key_matches_build_identity -v -rs --timeout=240)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_library_attention_cache_key_sweep_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;attention\\;host\\;standard\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
       "  TIMEOUT 900)\n")

# Exhaustive counted-sync gates are intentionally outside normal pytest
# discovery. The structural gate is host-only; the compiler gate additionally
# requires the reference COMGR/LLVM toolchain and llvm-objdump. They share a
# lock to cap aggregate host memory and compiler pressure in parallel runs.
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_counted_structural_gate python -m pytest -p pytest_timeout \"${ROCKE_TEST_CTEST_REL}tests/library/tests/gates/attention_bwd_counted_structural_gate.py\" -v -rs --timeout=60)\n"
       "set_tests_properties(rocke_library_counted_structural_gate PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;attention\\;host\\;structural\\;comprehensive\\;full\\;hip-kernel-provider\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${_ROCKE_LIBRARY_PYTHONPATH};PYTEST_ADDOPTS=;PYTEST_PLUGINS=;PYTEST_DISABLE_PLUGIN_AUTOLOAD=1;ROCKE_EXPLICIT_GATE_RUN=1;ROCKE_EXPLICIT_GATE_PARTIAL=0;ROCKE_EXPLICIT_GATE_PREFLIGHT=\"\n"
       "  PROCESSORS 4\n"
       "  RESOURCE_LOCK rocke_counted_sync_gate\n"
       "  TIMEOUT 1800)\n"
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
       "  LABELS \"ckc\\;python\\;attention\\;host\\;compiler\\;comprehensive\\;full\\;hip-kernel-provider\"\n"
       "  ENVIRONMENT \"\${_rocke_counted_isa_environment}\"\n"
       "  PROCESSORS 4\n"
       "  RESOURCE_LOCK rocke_counted_sync_gate\n"
       "  TIMEOUT 7200)\n")

# DSA lightning-indexer CPU tests (no GPU: validator, dispatch wiring, and the
# Python-lowered IR-sha golden). Same staged-library PYTHONPATH as the conv
# lane. The on-GPU numeric tests are enrolled in the GPU lane below.
set(_ROCKE_LIBRARY_DSA_TEST_RELPATHS
    "tests/library/tests/test_dsa_indexer_gfx942_spec.py"
    "tests/library/tests/test_dsa_indexer_golden.py"
    "tests/library/tests/dispatch/dsa/test_gfx942_wiring.py"
    "tests/library/tests/dispatch/dsa/test_gfx950_wiring.py")
set(_ROCKE_LIBRARY_DSA_TESTS "")
foreach(_rocke_dsa_test ${_ROCKE_LIBRARY_DSA_TEST_RELPATHS})
  string(APPEND _ROCKE_LIBRARY_DSA_TESTS
         " \"${ROCKE_TEST_CTEST_REL}${_rocke_dsa_test}\"")
endforeach()
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_dsa_pytest python -m pytest${_ROCKE_LIBRARY_DSA_TESTS} -v -rs --timeout=30)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_library_dsa_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;dsa\\;host\\;quick\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
       "  TIMEOUT 180)\n")

# Library MLA CPU tests (no GPU required): kernel lowering/admission, dispatch
# wiring, the IR golden, the fp32 oracle, the benchmark shape loader, the
# shared read-ahead helper, and the compile-test skip policy. The prefill
# occupancy / agpr_acc guards compile through comgr and read the code object
# with llvm-objdump, so the entry also takes the compiler environment; with it
# they fail rather than skip, and the skip-policy tests check that they do. Same
# PYTHONPATH contract as the conv entry above; the staged library tree carries
# kernels/mla, dispatch/mla, builders/mla, benchmarks/common/mla_shapes.py and
# tests/golden/mla_prefill_gfx942_ir_sha256.json.
set(_ROCKE_LIBRARY_MLA_TEST_RELPATHS
    "tests/library/tests/test_mla_prefill_gfx942.py"
    "tests/library/tests/dispatch/mla/test_mla_prefill_gfx942_wiring.py"
    "tests/library/tests/test_mla_prefill_gfx942_golden.py"
    "tests/library/tests/test_mla_reference.py"
    "tests/library/tests/test_mla_benchmark_shapes.py"
    "tests/library/tests/test_read_ahead.py"
    "tests/library/tests/test_compiler_test_utils.py")
set(_ROCKE_LIBRARY_MLA_TESTS "")
foreach(_rocke_mla_test ${_ROCKE_LIBRARY_MLA_TEST_RELPATHS})
  string(APPEND _ROCKE_LIBRARY_MLA_TESTS
         " \"${ROCKE_TEST_CTEST_REL}${_rocke_mla_test}\"")
endforeach()
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set(_rocke_library_mla_pytest_environment \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\")\n"
       "list(APPEND _rocke_library_mla_pytest_environment\n"
       "     \${_rocke_compiler_tool_environment}\n"
       "     \${_rocke_compiler_requirement_environment})\n"
       "add_test(rocke_library_mla_pytest python -m pytest${_ROCKE_LIBRARY_MLA_TESTS} -v -rs --timeout=120)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_library_mla_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;mla\\;host\\;quick\"\n"
       "  ENVIRONMENT \"\${_rocke_library_mla_pytest_environment}\"\n"
       "  TIMEOUT 900)\n")

string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_reference_common_pytest python -m pytest \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_reference_common.py\" -v -rs --timeout=30)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_reference_common_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;host\\;quick\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
       "  TIMEOUT 180)\n")

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
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_attention_pytest python -m pytest"
       " \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_wmma_fmha_fwd_v2_spec.py\""
       " \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_wmma_fmha_decode_spec.py\""
       " \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_attention_unified_lse.py\""
       " \"${ROCKE_TEST_CTEST_REL}tests/library/tests/dispatch/attention\""
       " --ignore \"${ROCKE_TEST_CTEST_REL}tests/library/tests/dispatch/attention/test_gfx1151_attention_dispatch_numeric.py\""
       " -v -rs --timeout=60)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_library_attention_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;attention\\;dispatch\\;host\\;quick\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
       "  TIMEOUT 300)\n")

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
  set(_ROCKE_SDPA_PYTEST_ARGS " --rocke-require-hipdnn-assets")
endif()
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_sdpa_pytest python -m pytest"
       " \"${ROCKE_TEST_CTEST_REL}tests/library/tests/sdpa\""
       " --ignore=\"${ROCKE_TEST_CTEST_REL}tests/library/tests/sdpa/test_unified_sdpa_numeric.py\""
       " --ignore=\"${ROCKE_TEST_CTEST_REL}tests/library/tests/sdpa/test_dense_sdpa_numeric.py\""
       "${_ROCKE_SDPA_PYTEST_ARGS} -v -rs --timeout=30)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_library_sdpa_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;sdpa\\;host\\;quick\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
       "  TIMEOUT 180)\n")

# GDN/KDA decode CPU tests: spec validation, IR emission and load order,
# the golden IR hashes, the performance knobs, dispatch selection, and the
# sweep / defaults-report tools and their shared timing helpers. None
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
# runs close to the 60 s per-test limit on Windows runners. Re-enable on
# Windows once that case is cheaper or split out. The GDN/KDA numeric files
# in rocke_library_gpu_pytest need gfx950/gfx942 devices and already skip
# at collection on the Windows GPU lanes, so they are unchanged. This entry
# stays registered on every other platform.
set(_ROCKE_LIBRARY_GDN_TEST_RELPATHS
    "tests/library/tests/test_gdn_decode_spec.py"
    "tests/library/tests/test_gdn_decode_fused.py"
    "tests/library/tests/test_gdn_decode_load_order.py"
    "tests/library/tests/test_gdn_decode_golden.py"
    "tests/library/tests/test_gdn_decode_knobs.py"
    "tests/library/tests/test_gdn_decode_timing.py"
    "tests/library/tests/test_gdn_decode_table_sweep.py"
    "tests/library/tests/test_gdn_decode_defaults_report.py"
    "tests/library/tests/test_kda_decode_benchmark.py"
    "tests/library/tests/dispatch/gdn/test_gfx950_registry.py"
    "tests/library/tests/dispatch/gdn/test_gfx950_wiring.py"
    "tests/library/tests/dispatch/gdn/test_tuning_contract.py"
    "tests/library/tests/dispatch/gdn/test_cli_args.py"
    "tests/library/tests/dispatch/gdn/test_gate_rules.py"
    "tests/library/tests/dispatch/gdn/test_gfx950_prefill_wiring.py")
set(_ROCKE_LIBRARY_GDN_TESTS "")
foreach(_rocke_gdn_test ${_ROCKE_LIBRARY_GDN_TEST_RELPATHS})
  string(APPEND _ROCKE_LIBRARY_GDN_TESTS
         " \"${ROCKE_TEST_CTEST_REL}${_rocke_gdn_test}\"")
endforeach()
if(WIN32)
  message(STATUS "rocKE: rocke_library_gdn_pytest not registered on Windows (slow GDN fused case; see comment above)")
else()
  string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
         "add_test(rocke_library_gdn_pytest python -m pytest${_ROCKE_LIBRARY_GDN_TESTS} -v -rs --timeout=60)\n")
  string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
         "set_tests_properties(rocke_library_gdn_pytest PROPERTIES\n"
         "  LABELS \"ckc\\;python\\;gdn\\;host\\;quick\"\n"
         "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
         "  TIMEOUT 600)\n")
endif()

# Library attention GPU numeric tests (gfx950-specific; require a GPU).
# Separate from the omnibus GPU suite so it can be targeted on its own when only
# attention hardware is available. Select it by name (`ctest -R
# rocke_library_attention_gpu_pytest`), not by label: the ROCKE_ENGINE YAML
# category pass replaces the "ckc;python;attention;gpu" labels below with its
# own, so `-L gpu` and `-L attention` do not match it in a YAML-categorized
# install tree.
set(_ROCKE_LIBRARY_ATTENTION_GPU_TEST_RELPATHS
    "tests/library/tests/test_attention_dense_gfx950_numeric.py"
    "tests/library/tests/test_extended_parity_attention.py")
set(_ROCKE_LIBRARY_ATTENTION_GPU_TESTS "")
foreach(_rocke_attn_gpu_test ${_ROCKE_LIBRARY_ATTENTION_GPU_TEST_RELPATHS})
  string(APPEND _ROCKE_LIBRARY_ATTENTION_GPU_TESTS
         " \"${ROCKE_TEST_CTEST_REL}${_rocke_attn_gpu_test}\"")
endforeach()
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_attention_gpu_pytest python -m pytest${_ROCKE_LIBRARY_ATTENTION_GPU_TESTS} -v -rs)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_library_attention_gpu_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;attention\\;gpu\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
       "  TIMEOUT 1800)\n")

# gfx950 additive-bias numeric lane. Its own entry (selected by name, like the
# one above) so its runtime does not eat into that entry's timeout.
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_attention_bias_gpu_pytest python -m pytest \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_attention_additive_bias_gfx950_numeric.py\" -v -rs)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_library_attention_bias_gpu_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;attention\\;gpu\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
       "  TIMEOUT 1200)\n")

# Library GPU numeric suites: convolution, attention, KDA, and GDN. The gfx950
# dense numeric and extended-parity attention tests run only in
# rocke_library_attention_gpu_pytest above, not again here.
# Select files rather than the "gpu" pytest marker because several runtime
# tests lack that marker. Their dependency and architecture checks report
# skips when the required torch installation or GPU is unavailable; the
# RDNA WMMA suites skip on devices without a gfx11/gfx12 generic target.
set(_ROCKE_LIBRARY_GPU_TEST_RELPATHS
    "tests/library/tests/test_conv_fwd_correctness.py"
    "tests/library/tests/test_conv_gradient_correctness.py"
    "tests/library/tests/test_conv_wgrad_correctness.py"
    "tests/library/tests/test_conv_wgrad_streamk_gpu.py"
    "tests/library/tests/test_conv_dgrad_correctness.py"
    "tests/library/tests/test_direct_conv_correctness.py"
    "tests/library/tests/test_conv_dgrad_depthwise_windowed.py"
    "tests/library/tests/test_direct_depthwise_dgrad_v2.py"
    "tests/library/tests/test_conv_dgrad_4c.py"
    "tests/library/tests/test_conv_dgrad_fused_weights.py"
    "tests/library/tests/test_direct_mfma_dgrad_correctness.py"
    "tests/library/tests/test_direct_dw_fwd_wmma.py"
    "tests/library/tests/test_direct_fwd_gstream_wmma.py"
    "tests/library/tests/test_direct_fwd_tapouter_wmma.py"
    "tests/library/tests/test_conv_winograd_fwd_gpu.py"
    "tests/library/tests/test_conv_winograd_fwd_rt_gpu.py"
    "tests/library/tests/test_conv_winograd_gemm_wmma.py"
    "tests/library/tests/test_conv_winograd_gen_gpu.py"
    "tests/library/tests/test_conv_winograd_dispatch_gpu.py"
    "tests/library/tests/test_conv_implicit_gemm_numeric.py"
    "tests/library/tests/test_conv_direct_tf32_mfma.py::TestDirectTf32Gpu"
    "tests/library/tests/test_conv_direct_tf32_scalar.py::TestGpuCorrectness"
    # The sweep sampler (test_sweep_candidates) compiles exact-fp32 grid
    # specs for minutes; it stays a manual run.
    "tests/library/tests/test_conv_direct_nongrouped_tf32.py::TestNongroupedF32Gpu::test_cases"
    "tests/library/tests/test_conv_direct_nongrouped_tf32.py::TestNongroupedF32Gpu::test_one_binary_serves_every_shape"
    "tests/library/tests/test_attention_dense_gfx942_numeric.py"
    "tests/library/tests/test_attention_scalar_block_table_numeric.py"
    "tests/library/tests/test_attention_unified_lse_numeric.py"
    "tests/library/tests/test_fmha_mfma_free_stride_numeric.py"
    "tests/library/tests/sdpa/test_unified_sdpa_numeric.py"
    "tests/library/tests/sdpa/test_dense_sdpa_numeric.py"
    "tests/library/tests/test_attn_bf16_d128_ring.py"
    "tests/library/tests/test_dtype_dispatch.py"
    "tests/library/tests/test_gdn_decode_gfx950_numeric.py"
    "tests/library/tests/test_gdn_decode_fused_gfx950_numeric.py"
    "tests/library/tests/test_gdn_decode_knobs_gfx950_numeric.py"
    "tests/library/tests/test_kda_decode_gfx950_numeric.py"
    "tests/library/tests/test_gfx1250_attention.py"
    "tests/library/tests/test_gfx942_gqa_head_fold_numeric.py"
    "tests/library/tests/test_gfx942_cfvst_i64_kv_addr_numeric.py"
    "tests/library/tests/test_gfx942_4wgqa_bounds_numeric.py"
    "tests/library/tests/test_attention_gfx942_kv_pages.py"
    "tests/library/tests/test_attention_tiled_runtime_shape_numeric.py"
    "tests/library/tests/test_attention_tiled_3d_kv_lds_swizzle.py"
    "tests/library/tests/test_attention_tiled_3d_running_max.py"
    "tests/library/tests/test_attention_unified_2d_masked_tile_numeric.py"
    "tests/library/tests/test_attention_unified_scalar_masked_tile_numeric.py"
    "tests/library/tests/test_attention_tiled_3d_wave_page_id_load.py"
    "tests/library/tests/test_kda_chunkwise_gfx942_numeric.py"
    "tests/library/tests/test_kda_chunkwise_gfx950_numeric.py"
    "tests/library/tests/test_kda_gdn_gfx950_numeric.py"
    "tests/library/tests/test_mla_prefill_gfx942_numeric.py"
    "tests/library/tests/test_wmma_fmha_fwd_v2_numeric.py"
    "tests/library/tests/test_wmma_fmha_fwd_v2_features.py"
    "tests/library/tests/test_wmma_fmha_decode_numeric.py"
    "tests/library/tests/test_wmma_fmha_paged_guard.py"
    "tests/library/tests/test_wmma_fp8_decode_numeric.py"
    "tests/library/tests/dispatch/attention/test_gfx1151_attention_dispatch_numeric.py"
    "tests/library/tests/test_dsa_indexer_numeric.py")
set(_ROCKE_LIBRARY_GPU_TESTS "")
foreach(_rocke_gpu_test ${_ROCKE_LIBRARY_GPU_TEST_RELPATHS})
  string(APPEND _ROCKE_LIBRARY_GPU_TESTS
         " \"${ROCKE_TEST_CTEST_REL}${_rocke_gpu_test}\"")
endforeach()
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_gpu_pytest python -m pytest${_ROCKE_LIBRARY_GPU_TESTS} -v -rs)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_library_gpu_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;gpu\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
       "  TIMEOUT 1800)\n")
