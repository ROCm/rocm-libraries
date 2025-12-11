/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include "pooling3d.hpp"
#include "../driver.hpp"
#include <half/half.hpp>

namespace pooling3d {

class GPU_Pooling3d_FP32 : public testing::Test
{
};

class GPU_Pooling3d_FP16 : public testing::Test
{
};

template <typename T>
void RunPooling3dTests()
{
    pooling3d_driver<T> driver;
    driver.type              = miopen_type<T>{};
    driver.full_set          = false;
    driver.dataset_id        = 0;
    driver.config_iter_start = 0;

    std::vector<typename pooling3d_driver<T>::argument*> data_args;
    for(auto&& arg : driver.arguments)
    {
        data_args.push_back(&arg);
    }

    prng::reset_seed();
    driver.iteration = 0;
    try
    {
        run_data(data_args.begin(), data_args.end(), [&] {
            driver.template base_run<pooling3d_driver<T>>();
        });
    }
    catch(const std::exception& e)
    {
        FAIL() << "Exception in pooling3d test: " << e.what();
    }
    catch(...)
    {
        FAIL() << "Unknown exception in pooling3d test";
    }
}

void Run3dDriver(miopenDataType_t prec)
{
    switch(prec)
    {
    case miopenFloat: RunPooling3dTests<float>(); break;
    case miopenHalf: RunPooling3dTests<half_float::half>(); break;
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
                  "pooling3d test";

    default: RunPooling3dTests<float>();
    }
};

} // namespace pooling3d
using namespace pooling3d;

TEST_F(GPU_Pooling3d_FP32, FloatTest_pooling3d)
{
    if(!IsTestSupportedByDevice(Gpu::All))
    {
        GTEST_SKIP();
    }
    Run3dDriver(miopenFloat);
}

TEST_F(GPU_Pooling3d_FP16, HalfTest_pooling3d)
{
    if(!IsTestSupportedByDevice(Gpu::All))
    {
        GTEST_SKIP();
    }
    Run3dDriver(miopenHalf);
}
