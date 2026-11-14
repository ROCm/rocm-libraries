/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/
#pragma once

#include <cstdint>
#include <string>

struct Arguments;

// Process-wide CLI settings for hipblaslt-bench. apply_to resolves the settings
// represented in Arguments before a benchmark case enters the shared client code.
//
// cotenant_cus is the workgroup count of a busy kernel kept resident on a separate
// stream during timed runs (0 disables it); cotenant_max_occupancy caps its
// workgroups per CU through LDS reservation.
//
// sm_count_target sets the handle's compute-unit target for both C and C++ problems.
// streamk_tile_scheduling_mode sets the matmul descriptor, C++ GEMM instance,
// and C++ heuristic preference as a tri-state {0=OFF, 1=ON, 2=AUTO}.
// The resolved mode defaults to -1
// ("unset"): a negative value tells the bench to leave the setting untouched
// so the library default applies. streamk_tile_scheduling_mode_str() holds the raw
// CLI token (off|on|auto or 0|1|2) before client.cpp resolves it.
//
// uniform_summation_order is {0=off, 1=on, -1=unspecified}. apply_to preserves an
// explicit case value and otherwise uses the CLI value. The client handle applies
// the resolved request; leaving both inputs unspecified keeps the library default.
//
// verification_failures counts the -v checks of D that failed: one per gemm, per solution,
// for each of the norm and allclose checks, plus one per solution whose post_bench_verify_iters
// relaunches differed from the first launch (however many gemms or relaunches differed). With
// verify_strict set, verification_exit_code turns a run that would exit 0 into exit code 2 when
// any check failed. apply_to enables these checks for every case when verify_strict
// is set, even if a case disables them; strict verification is a run-wide requirement.
//
// post_bench_verify_iters is the number of untimed relaunches of each solution after timing
// (0 turns them off); every relaunch must give the same D, bit for bit, as the first launch.
namespace hipblaslt_bench_options
{
    int32_t&     cotenant_cus();
    int32_t&     cotenant_max_occupancy();
    int32_t&     sm_count_target();
    int32_t&     streamk_tile_scheduling_mode();
    std::string& streamk_tile_scheduling_mode_str();
    std::string& hybrid_assignment_policy_str();
    int32_t resolve_hybrid_assignment_policy(std::string const& canonical,
                                            std::string legacy);
    int32_t&     uniform_summation_order();
    std::string& uniform_summation_order_str();
    bool&        verify_strict();
    int64_t&     verification_failures();
    int          verification_exit_code(int ret);
    // Defined in hipblaslt_arguments.cpp because tensilelite-tests compiles
    // hipblaslt_bench_options.cpp without the headers required by Arguments.
    void         apply_to(Arguments& arg);
    int32_t&     post_bench_verify_iters();
}
