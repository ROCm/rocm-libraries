// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "pooling3d.hpp"
#include "../driver.hpp"
#include <half/half.hpp>

namespace {

class GPU_Pooling3d_FP32 : public testing::TestWithParam<miopenDataType_t>
{
    void SetUp() override
    {
        prng::reset_seed();
        if(!IsTestSupportedByDevice(Gpu::All))
        {
            GTEST_SKIP();
        }
    }
};

class GPU_Pooling3d_FP16 : public testing::TestWithParam<miopenDataType_t>
{
    void SetUp() override
    {
        prng::reset_seed();
        if(!IsTestSupportedByDevice(Gpu::All))
        {
            GTEST_SKIP();
        }
    }
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

} // namespace

TEST_P(GPU_Pooling3d_FP32, FloatTest_pooling3d)
{
    Run3dDriver(GetParam());
}

TEST_P(GPU_Pooling3d_FP16, HalfTest_pooling3d)
{
    Run3dDriver(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Full, GPU_Pooling3d_FP32, testing::ValuesIn({miopenFloat}));

INSTANTIATE_TEST_SUITE_P(Full, GPU_Pooling3d_FP16, testing::ValuesIn({miopenHalf}));
