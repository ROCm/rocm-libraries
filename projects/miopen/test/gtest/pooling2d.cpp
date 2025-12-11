// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "pooling2d.hpp"
#include "../driver.hpp"
#include <half/half.hpp>

namespace {

class GPU_Pooling2d_FP32 : public testing::TestWithParam<miopenDataType_t>
{
    void SetUp() override
    {
        prng::reset_seed();
        const auto& handle = get_handle();
        if(!IsTestSupportedForDevice(handle))
        {
            GTEST_SKIP();
        }
        // Decrease log level to reduce output
        lib_env::update(MIOPEN_LOG_LEVEL, 2);
    }
};

class GPU_Pooling2d_FP16 : public testing::TestWithParam<miopenDataType_t>
{
    void SetUp() override
    {
        prng::reset_seed();
        const auto& handle = get_handle();
        if(!IsTestSupportedForDevice(handle))
        {
            GTEST_SKIP();
        }
        // Decrease log level to reduce output
        lib_env::update(MIOPEN_LOG_LEVEL, 2);
    }
};

template <typename T>
void RunPooling2dTests()
{
    // Run 1: full_set = false with default dataset (reduced combinations, various tensor sizes)
    {
        pooling2d_driver<T> driver;
        driver.type              = miopen_type<T>{};
        driver.full_set          = false;
        driver.dataset_id        = 0;
        driver.config_iter_start = 0;

        std::vector<typename pooling2d_driver<T>::argument*> data_args;
        for(auto&& arg : driver.arguments)
        {
            data_args.push_back(&arg);
        }

        driver.iteration = 0;
        try
        {
            run_data(data_args.begin(), data_args.end(), [&] {
                driver.template base_run<pooling2d_driver<T>>();
            });
        }
        catch(const std::exception& e)
        {
            FAIL() << "Exception in pooling2d test (full_set=false, dataset=0): " << e.what();
        }
        catch(...)
        {
            FAIL() << "Unknown exception in pooling2d test (full_set=false, dataset=0)";
        }
    }

    // Run 2: full_set = true with minimal dataset (all combinations, small tensors)
    {
        pooling2d_driver<T> driver;
        driver.type              = miopen_type<T>{};
        driver.full_set          = true;
        driver.dataset_id        = 1; // Minimal dataset to avoid OOM
        driver.config_iter_start = 0;

        std::vector<typename pooling2d_driver<T>::argument*> data_args;
        for(auto&& arg : driver.arguments)
        {
            data_args.push_back(&arg);
        }

        driver.iteration = 0;
        try
        {
            run_data(data_args.begin(), data_args.end(), [&] {
                driver.template base_run<pooling2d_driver<T>>();
            });
        }
        catch(const std::exception& e)
        {
            FAIL() << "Exception in pooling2d test (full_set=true, dataset=1): " << e.what();
        }
        catch(...)
        {
            FAIL() << "Unknown exception in pooling2d test (full_set=true, dataset=1)";
        }
    }
}

void Run2dDriver(miopenDataType_t prec)
{
    switch(prec)
    {
    case miopenFloat: RunPooling2dTests<float>(); break;
    case miopenHalf: RunPooling2dTests<half_float::half>(); break;
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

} // namespace

TEST_P(GPU_Pooling2d_FP32, FloatTest_pooling2d) { Run2dDriver(GetParam()); }

TEST_P(GPU_Pooling2d_FP16, HalfTest_pooling2d) { Run2dDriver(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_Pooling2d_FP32, testing::ValuesIn({miopenFloat}));

INSTANTIATE_TEST_SUITE_P(Full, GPU_Pooling2d_FP16, testing::ValuesIn({miopenHalf}));
