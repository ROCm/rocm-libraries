/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

//
// Device (GPU) unit tests for rocSPARSE internal block/warp collectives.
//
// Compiled into the rocsparse-unit-test-device binary (links hip::device); must
// run on a GPU (via the serializer, e.g. HIP_VISIBLE_DEVICES=0 gpu-run).
//
// Pattern: write a thin __global__ wrapper around an internal
// ROCSPARSE_DEVICE_ILF collective (blockreduce_*, wfreduce_*, wfsegmented_reduce,
// segmented_blockreduce, dichotomic_search, popc, shfl_*, assign_ilu0_boost_value,
// ...), launch it on one block/warp via rocsparse_ut::launch_single_block /
// launch_single_warp, and assert on the readback via rocsparse_ut::to_host.
//
// Wavefront-size policy: warp collectives are templated on the wavefront size and
// instantiated for BOTH 32 and 64. Tests dispatch at runtime to the instantiation
// matching rocsparse_ut::device_warp_size() (via launch_warp_by_size), so the
// 32-lane path runs on wave32 parts (e.g. gfx1201) and the 64-lane path on wave64
// parts (e.g. gfx94x/gfx950). No wavefront path is skipped or hard-coded. The real
// per-routine tests live in the collectives PR; this foundation TU only proves the
// harness links and runs.
//
#include "unit_test_utils.hpp"

#include "rocsparse_common.hpp" // ROCSPARSE_DEVICE_ILF collectives (blockreduce_*, wfreduce_*, shfl_*)

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

// Placeholder that proves the device unit-test harness compiles, links the
// compiled-in library TUs, and runs on the GPU. Always passes.
TEST(internal_collectives, harness_smoke)
{
    SUCCEED();
}
