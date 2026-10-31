/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef TESTING_BACKEND_HPP
#define TESTING_BACKEND_HPP

#include "utility.hpp"

#include <cstdio>
#include <cstdlib>
#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace rocalution;

void testing_backend_init_order(void)
{
    bool use_acc   = false;
    bool omp_aff   = false;
    int  dev       = 0;
    int  nthreads  = 4;
    int  threshold = 20000;

    // Try to stop rocalution without initialization
    stop_rocalution();

    // Set OpenMP threads
    ASSERT_DEATH(set_omp_threads_rocalution(nthreads), ".*Assertion.*");

    // Set OpenMP threshold size
    ASSERT_DEATH(set_omp_threshold_rocalution(threshold), ".*Assertion.*");

    // Initialize rocalution platform
    set_device_rocalution(device);
    init_rocalution();

    // Set OpenMP thread affinity after init_rocalution should terminate
    ASSERT_DEATH(set_omp_affinity_rocalution(omp_aff), ".*Assertion.*");

    // Select a device after init_rocalution should terminate
    ASSERT_DEATH(set_device_rocalution(dev), ".*Assertion.*");

    // Enable/disable accelerator after init_rocalution should terminate
    ASSERT_DEATH(disable_accelerator_rocalution(use_acc), ".*Assertion.*");

    // Stop rocalution platform
    stop_rocalution();
}

void testing_backend_disable_accelerator_after_stop(void)
{
    // Initialize rocalution platform with the accelerator
    set_device_rocalution(device);
    disable_accelerator_rocalution(false);
    init_rocalution();
    stop_rocalution();

    // Initialize it again without the accelerator
    disable_accelerator_rocalution(true);
    init_rocalution();

    // Stop rocalution platform
    stop_rocalution();
    disable_accelerator_rocalution(false);
}

void testing_backend(Arguments argus)
{
    int  rank         = argus.rank;
    int  dev_per_node = argus.dev_per_node;
    int  dev          = argus.dev;
    int  nthreads     = argus.omp_nthreads;
    bool affinity     = argus.omp_affinity;
    int  threshold    = argus.omp_threshold;
    bool use_acc      = argus.use_acc;

    // Select a device
    set_device_rocalution(dev);

    // Enable/disable accelerator
    disable_accelerator_rocalution(use_acc);

    // Set OpenMP thread affinity
    set_omp_affinity_rocalution(affinity);

    // Initialize rocalution platform
    init_rocalution(rank, dev_per_node);

    // Set OpenMP threads
    set_omp_threads_rocalution(nthreads);

    // Set OpenMP threshold size
    set_omp_threshold_rocalution(threshold);

    // Stop rocalution platform
    stop_rocalution();
}

// On Windows, omp.h only declares the OpenMP 2.0 API
#if defined(_OPENMP) && _OPENMP >= 200805 && !defined(_WIN32)
// Set the pre-init OpenMP state, then initialize a host-only platform with 2 threads
void backend_omp_init(int levels, int nthreads)
{
    omp_set_dynamic(0);
    omp_set_max_active_levels(levels);
    omp_set_num_threads(nthreads);

    disable_accelerator_rocalution(true);
    set_omp_affinity_rocalution(false);
    set_device_rocalution(device);
    init_rocalution();
    set_omp_threads_rocalution(2);
}

void backend_omp_stop(void)
{
    stop_rocalution();
    set_omp_affinity_rocalution(true);
    disable_accelerator_rocalution(false);
}

// Team size of a region on the calling thread: same thread and ICVs as the library's host
// kernels, which use "omp parallel for" after omp_set_num_threads(OpenMP_threads)
int backend_omp_team_size(void)
{
    int team_size = 0;
#pragma omp parallel
    {
        if(omp_get_thread_num() == 0)
        {
            team_size = omp_get_num_threads();
        }
    }
    return team_size;
}
#endif

void testing_backend_omp_parallel_region(void)
{
#if defined(_OPENMP) && _OPENMP >= 200805 && !defined(_WIN32)
    // OMP_THREAD_LIMIT=1 legitimately caps every team at one thread
    if(omp_get_thread_limit() < 2)
    {
        GTEST_SKIP() << "OpenMP thread limit is " << omp_get_thread_limit();
    }

    int def_dynamic = omp_get_dynamic();
    int def_levels  = omp_get_max_active_levels();
    int def_threads = omp_get_max_threads();

    // Nesting disabled before init: the outermost level must stay active
    backend_omp_init(1, 3);
    EXPECT_EQ(omp_get_max_active_levels(), 1);
    EXPECT_EQ(omp_get_max_threads(), 2);
    EXPECT_EQ(backend_omp_team_size(), 2);
    backend_omp_stop();

    // Nesting enabled before init (OMP_MAX_ACTIVE_LEVELS=4, OMP_NESTED=true,
    // OMP_NUM_THREADS=4,2): rocALUTION disables nesting only. Run in a child process
    // without stop_rocalution(), which is covered by testing_backend_omp_restore_state().
    EXPECT_EXIT(
        {
            backend_omp_init(4, 3);
            int levels    = omp_get_max_active_levels();
            int team_size = backend_omp_team_size();
            fprintf(stderr, "max_active_levels %d, team size %d\n", levels, team_size);
            exit((levels == 1 && team_size == 2) ? 0 : 1);
        },
        ::testing::ExitedWithCode(0),
        "max_active_levels 1, team size 2");

    // All parallel regions disabled by the user before init: rocALUTION keeps them disabled
    backend_omp_init(0, 3);
    EXPECT_EQ(omp_get_max_active_levels(), 0);
    EXPECT_EQ(backend_omp_team_size(), 1);
    backend_omp_stop();

    omp_set_dynamic(def_dynamic);
    omp_set_max_active_levels(def_levels);
    omp_set_num_threads(def_threads);
#else
    GTEST_SKIP() << "rocalution-test is built without the OpenMP 3.0 API";
#endif
}

void testing_backend_omp_restore_state(void)
{
#if defined(_OPENMP) && _OPENMP >= 200805 && !defined(_WIN32)
    int def_dynamic = omp_get_dynamic();
    int def_levels  = omp_get_max_active_levels();
    int def_threads = omp_get_max_threads();

    backend_omp_init(1, 3);
    backend_omp_stop();
    EXPECT_EQ(omp_get_max_active_levels(), 1);
    EXPECT_EQ(omp_get_max_threads(), 3);

    backend_omp_init(0, 3);
    backend_omp_stop();
    EXPECT_EQ(omp_get_max_active_levels(), 0);
    EXPECT_EQ(omp_get_max_threads(), 3);

    // Run in a child process, since stop_rocalution() may abort
    EXPECT_EXIT(
        {
            backend_omp_init(4, 3);
            backend_omp_stop();
            int levels  = omp_get_max_active_levels();
            int threads = omp_get_max_threads();
            fprintf(stderr, "max_active_levels %d, max_threads %d\n", levels, threads);
            exit((levels == 4 && threads == 3) ? 0 : 1);
        },
        ::testing::ExitedWithCode(0),
        "max_active_levels 4, max_threads 3");

    omp_set_dynamic(def_dynamic);
    omp_set_max_active_levels(def_levels);
    omp_set_num_threads(def_threads);
#else
    GTEST_SKIP() << "rocalution-test is built without the OpenMP 3.0 API";
#endif
}

#endif // TESTING_BACKEND_HPP
