/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <type_traits>

#include "detail/load_store_matrix_coop_sync.hpp"
#include "kernel_generator.hpp"
#include "load_store_matrix_coop_sync_test_emulation_params.hpp"
#include "unit_test.hpp"
#include "unit_test_macros.hpp"

namespace rocwmma
{

    using TestParams
        = EmulationLoadStoreMatrixCoopSyncTestParams<UnitTestParams::TestAllSizeTypes,
                                                     UnitTestParams::TestBlockSizes32,
                                                     LoadStoreMatrixCoopSyncGeneratorB>;

} // namespace rocwmma

// Test suite for unique parameterization
ROCWMMA_GENERATE_UNIT_GTEST_SUITE(EmulationRegressionLoadStoreMatrixSyncCoopBTest32, TestParams)
