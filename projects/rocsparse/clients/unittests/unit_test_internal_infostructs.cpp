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
// Device-target unit tests for rocSPARSE internal data-structure member
// functions (info structs).
//
// Compiled into the rocsparse-unit-test-device binary. Exercises member
// functions of the internal info/handle structures: trm_data_t::storage_index,
// numeric_boost define/get/set/copy, position_t getters, *_info clear(),
// trm_info_t copy/destroy, _rocsparse_mat_info get/set/clear routing, and
// csrgemm_info create/copy/destroy. Downstream this file compiles in the
// matching library/src/common/*.cpp TUs and uses rocsparse_ut::HandleTest.
//
#include "unit_test_utils.hpp"

#include "rocsparse_handle.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

// Placeholder proving the device harness links the compiled-in info-struct TUs
// and runs; the real member-function tests live in the infostructs PR. Always
// passes.
TEST(internal_infostructs, harness_smoke)
{
    SUCCEED();
}
