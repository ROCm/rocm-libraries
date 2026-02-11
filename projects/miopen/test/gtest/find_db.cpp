// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "find_db.hpp"
#include "../lib_env_var.hpp"

MIOPEN_LIB_ENV_VAR(MIOPEN_ENABLE_LOGGING_ELAPSED_TIME)
MIOPEN_LIB_ENV_VAR(MIOPEN_COMPILE_PARALLEL_LEVEL)

namespace {

void RunFindDbDriver(miopenDataType_t prec)
{
    switch(prec)
    {
    case miopenFloat: break;
    case miopenHalf:
    case miopenBFloat16:
    case miopenInt8:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
        FAIL() << "miopenHalf, miopenBFloat16, miopenInt8, miopenInt32, miopenDouble, "
                  "miopenFloat8_fnuz, miopenBFloat8_fnuz "
                  "data type not supported by "
                  "find_db test";

    default: break;
    }

    // Set up environment variables with automatic restoration
    ScopedEnvironment<int> logging_elapsed_time(MIOPEN_ENABLE_LOGGING_ELAPSED_TIME, 1);
    ScopedEnvironment<int> log_level(MIOPEN_LOG_LEVEL, 6);
    ScopedEnvironment<int> compile_parallel_level(MIOPEN_COMPILE_PARALLEL_LEVEL, 1);

    // Create driver instance and run the test directly
    miopen::find_db_driver<float> driver;
    driver.run();
}

} // namespace

class GPU_FindDb_FP32 : public testing::TestWithParam<miopenDataType_t>
{
    void SetUp() override { prng::reset_seed(); }
};

TEST_P(GPU_FindDb_FP32, FloatTest_find_db) { RunFindDbDriver(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_FindDb_FP32, testing::ValuesIn({miopenFloat}));
