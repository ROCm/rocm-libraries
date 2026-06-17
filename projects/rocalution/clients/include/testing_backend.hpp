/* ************************************************************************
 * Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

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

#endif // TESTING_BACKEND_HPP
