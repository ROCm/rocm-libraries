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
    "tests/library/tests/test_conv_dgrad_spec_policy.py"
    "tests/library/tests/test_conv_ir_golden.py"
    "tests/library/tests/test_conv_platform_regressions.py"
    "tests/library/tests/test_conv_split_k.py"
    "tests/library/tests/test_conv_split_k_planner.py"
    "tests/library/tests/test_conv_split_k_wgrad.py"
    "tests/library/tests/test_ceil_div_callers.py"
    "tests/library/tests/test_streamk_sweep_launch.py"
    "tests/library/tests/test_conv_build_signatures.py"
    "tests/library/tests/test_conv_aot_grid_limits.py"
    "tests/library/tests/test_direct_dgrad_wmma.py"
    "tests/library/tests/test_direct_kernel_cache.py"
    "tests/library/tests/test_conv_gradient_contract.py"
    "tests/library/tests/test_builder_signature_contract.py"
    "tests/library/tests/test_library_layering.py")
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

# Library attention CPU tests (no GPU required): gfx950 LDS budget model,
# dense block-order decodes, spec-wrapper parity gate, and the 3D decode split
# rule and K/V cache policy.
# File-list style (same pattern as conv_pytest) so PYTHONPATH resolves rocke/
# correctly from the installed tree.
set(_ROCKE_LIBRARY_ATTENTION_CPU_TEST_RELPATHS
    "tests/library/tests/test_attention_dense_decode.py"
    "tests/library/tests/test_attention_dense_gfx950_lds.py"
    "tests/library/tests/test_attention_spec_predicate_parity.py"
    "tests/library/tests/test_attention_tiled_3d_decode_split.py"
    "tests/library/tests/test_attention_tiled_3d_kv_cache_policy.py")
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
# wiring, the IR golden, the fp32 oracle, and the benchmark shape loader. Same
# PYTHONPATH contract as the conv entry above; the staged library tree carries
# kernels/mla, dispatch/mla, builders/mla, benchmarks/common/mla_shapes.py and
# tests/golden/mla_prefill_gfx942_ir_sha256.json.
set(_ROCKE_LIBRARY_MLA_TEST_RELPATHS
    "tests/library/tests/test_mla_prefill_gfx942.py"
    "tests/library/tests/dispatch/mla/test_mla_prefill_gfx942_wiring.py"
    "tests/library/tests/test_mla_prefill_gfx942_golden.py"
    "tests/library/tests/test_mla_reference.py"
    "tests/library/tests/test_mla_benchmark_shapes.py")
set(_ROCKE_LIBRARY_MLA_TESTS "")
foreach(_rocke_mla_test ${_ROCKE_LIBRARY_MLA_TEST_RELPATHS})
  string(APPEND _ROCKE_LIBRARY_MLA_TESTS
         " \"${ROCKE_TEST_CTEST_REL}${_rocke_mla_test}\"")
endforeach()
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_mla_pytest python -m pytest${_ROCKE_LIBRARY_MLA_TESTS} -v -rs --timeout=120)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_library_mla_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;mla\\;host\\;quick\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
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

# GDN/KDA decode CPU tests: spec validation, IR emission and load order,
# the golden IR hashes, the performance knobs, and dispatch selection. None
# needs a GPU; the GPU numeric suites are in rocke_library_gpu_pytest. Same
# PYTHONPATH rule as the conv entry.
set(_ROCKE_LIBRARY_GDN_TEST_RELPATHS
    "tests/library/tests/test_gdn_decode_spec.py"
    "tests/library/tests/test_gdn_decode_fused.py"
    "tests/library/tests/test_gdn_decode_load_order.py"
    "tests/library/tests/test_gdn_decode_golden.py"
    "tests/library/tests/test_gdn_decode_knobs.py"
    "tests/library/tests/test_gdn_decode_tune.py"
    "tests/library/tests/dispatch/gdn/test_gfx950_registry.py"
    "tests/library/tests/dispatch/gdn/test_gfx950_wiring.py"
    "tests/library/tests/dispatch/gdn/test_gfx950_prefill_wiring.py")
set(_ROCKE_LIBRARY_GDN_TESTS "")
foreach(_rocke_gdn_test ${_ROCKE_LIBRARY_GDN_TEST_RELPATHS})
  string(APPEND _ROCKE_LIBRARY_GDN_TESTS
         " \"${ROCKE_TEST_CTEST_REL}${_rocke_gdn_test}\"")
endforeach()
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_library_gdn_pytest python -m pytest${_ROCKE_LIBRARY_GDN_TESTS} -v -rs --timeout=60)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_library_gdn_pytest PROPERTIES\n"
       "  LABELS \"ckc\\;python\\;gdn\\;host\\;quick\"\n"
       "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
       "  TIMEOUT 600)\n")

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
    "tests/library/tests/test_conv_dgrad_4c.py"
    "tests/library/tests/test_conv_dgrad_fused_weights.py"
    "tests/library/tests/test_direct_mfma_dgrad_correctness.py"
    "tests/library/tests/test_conv_implicit_gemm_numeric.py"
    "tests/library/tests/test_attention_dense_gfx942_numeric.py"
    "tests/library/tests/test_attention_unified_lse_numeric.py"
    "tests/library/tests/test_attn_bf16_d128_ring.py"
    "tests/library/tests/test_dtype_dispatch.py"
    "tests/library/tests/test_gdn_decode_gfx950_numeric.py"
    "tests/library/tests/test_gdn_decode_fused_gfx950_numeric.py"
    "tests/library/tests/test_gdn_decode_knobs_gfx950_numeric.py"
    "tests/library/tests/test_kda_decode_gfx950_numeric.py"
    "tests/library/tests/test_gfx1250_attention.py"
    "tests/library/tests/test_gfx942_gqa_head_fold_numeric.py"
    "tests/library/tests/test_attention_tiled_3d_kv_lds_swizzle.py"
    "tests/library/tests/test_attention_tiled_3d_running_max.py"
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
