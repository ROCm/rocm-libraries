/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "pooling2d.hpp"
#include "../driver.hpp"
#include <half/half.hpp>

namespace pooling2d {

class GPU_Pooling2d_FP32 : public testing::Test
{
};

class GPU_Pooling2d_FP16 : public testing::Test
{
};

template <typename T>
void RunPooling2dTests()
{
    pooling2d_driver<T> driver;
    driver.type = miopen_type<T>{};
    driver.full_set = true; // Equivalent to --all flag
    driver.dataset_id = 0;   // Default dataset

    // Get data arguments (arguments that weren't passed via command line)
    std::vector<typename pooling2d_driver<T>::argument*> data_args;
    for(auto&& arg : driver.arguments)
    {
        data_args.push_back(&arg);
    }

    // Manually iterate over all combinations using the driver's iteration logic
    prng::reset_seed();
    driver.iteration = 0;
    run_data(data_args.begin(), data_args.end(), [&] { driver.template base_run<pooling2d_driver<T>>(); });
}

void Run2dDriver(miopenDataType_t prec)
{
    switch(prec)
    {
    case miopenFloat:
        RunPooling2dTests<float>();
        break;
    case miopenHalf:
        RunPooling2dTests<half_float::half>();
        break;
    case miopenBFloat16:
    case miopenInt8:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
        FAIL() << "miopenBFloat16, miopenInt8, miopenInt32, miopenDouble, miopenFloat8_fnuz, "
                  "miopenBFloat8_fnuz "
                  "data type not supported by "
                  "pooling2d test";

    default: RunPooling2dTests<float>();
    }
};

bool IsTestSupportedForDevice(const miopen::Handle& handle) { return true; }


} // namespace pooling2d
using namespace pooling2d;

TEST_F(GPU_Pooling2d_FP32, FloatTest_pooling2d)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle))
    {
        Run2dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
}

TEST_F(GPU_Pooling2d_FP16, HalfTest_pooling2d)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle))
    {
        Run2dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }
}
