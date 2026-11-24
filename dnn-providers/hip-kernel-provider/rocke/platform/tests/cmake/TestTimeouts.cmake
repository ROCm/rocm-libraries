# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The CTest TIMEOUT of every rocKE test entry, build tree and installed tree.
# Included from platform/CMakeLists.txt before any test is registered, so the
# limits reach tests/, library/cpp/tests/ and library/tests/ctest_lists.cmake.
#
# One generous limit per resource class, not one tuned limit per entry. An
# entry takes the limit of the resource it needs:
#
#   cpu        host Python or a native C++ binary; no compiler, no GPU
#   toolchain  also runs or probes the compiler (COMGR, hipcc, llvm-objdump):
#              its environment carries the compiler tools, or it measures them
#   gpu        launches kernels on a GPU
#
# Each limit sits well above the slowest entry of its class seen on the slowest
# CI runners, so a limit fails an entry only when it hangs. A tight limit per
# entry turned runner speed into red runs: every recurring CI flake of these
# entries was a time limit. The cost of a generous limit is that a hang is
# reported later; the per-shard time budget, not these limits, keeps a shard
# inside its CI step limit. When an entry nears its class limit, make it cheaper
# or split it; do not raise the limit for one entry.
#
# The only exception is a gate that compiles a whole matrix: it has its own
# limit, below.
set(ROCKE_TEST_TIMEOUT_CPU 900)
set(ROCKE_TEST_TIMEOUT_TOOLCHAIN 1800)
set(ROCKE_TEST_TIMEOUT_GPU 1800)
# rocke_library_counted_compiler_gate compiles and disassembles the full
# counted-sync matrix and runs only in the comprehensive and full tiers.
set(ROCKE_TEST_TIMEOUT_COUNTED_COMPILER_GATE 7200)

# The per-test limit of the GPU pytest entries (platform/CMakeLists.txt,
# _ROCKE_INSTALLED_PYTEST_GPU_TEST_LIMIT, says why): more than twice the slowest
# GPU test measured with a cold compiler cache, and half the gpu CTest limit.
set(ROCKE_TEST_ITEM_TIMEOUT_GPU 900)

# Give every test the calling directory registered without a TIMEOUT the cpu
# limit. Call it after the directory's registrations. It covers the native C++
# test binaries, which are all cpu entries, so their many registrations need
# not repeat the limit; a test of another class sets its limit where it is
# registered. It must run in the directory that registered the tests: before
# CMake 3.28 a directory can set properties only on its own tests.
function(rocke_set_cpu_timeout_on_untimed_tests)
  get_property(_rocke_tests DIRECTORY PROPERTY TESTS)
  foreach(_rocke_test IN LISTS _rocke_tests)
    get_test_property(${_rocke_test} TIMEOUT _rocke_timeout)
    if(NOT _rocke_timeout)
      set_tests_properties(${_rocke_test} PROPERTIES TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})
    endif()
  endforeach()
endfunction()
