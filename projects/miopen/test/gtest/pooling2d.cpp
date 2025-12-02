// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

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
    driver.full_set = false; // Set to false to reduce test cases and avoid OOM on smaller GPUs
    driver.dataset_id = 0;   // Use default dataset
    driver.config_iter_start = 0;

    // Get data arguments (arguments that weren't passed via command line)
    std::vector<typename pooling2d_driver<T>::argument*> data_args;
    for(auto&& arg : driver.arguments)
    {
        data_args.push_back(&arg);
    }

    // Manually iterate over all combinations using the driver's iteration logic
    prng::reset_seed();
    driver.iteration = 0;
    try
    {
        run_data(data_args.begin(), data_args.end(), [&] { driver.template base_run<pooling2d_driver<T>>(); });
    }
    catch(const std::exception& e)
    {
        FAIL() << "Exception in pooling2d test: " << e.what();
    }
    catch(...)
    {
        FAIL() << "Unknown exception in pooling2d test";
    }
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
