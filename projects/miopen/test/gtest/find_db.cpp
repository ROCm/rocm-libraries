// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "workspace.hpp"

#include <miopen/convolution.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/find_db.hpp>
#include <miopen/logger.hpp>
#include <miopen/temp_file.hpp>
#include <miopen/tensor.hpp>
#include <miopen/hip_build_utils.hpp>

#include <chrono>
#include <functional>

MIOPEN_LIB_ENV_VAR(MIOPEN_ENABLE_LOGGING_ELAPSED_TIME)
MIOPEN_LIB_ENV_VAR(MIOPEN_LOG_LEVEL)
MIOPEN_LIB_ENV_VAR(MIOPEN_COMPILE_PARALLEL_LEVEL)

namespace miopen {

struct TestRordbEmbedFsOverrideLock
{
    TestRordbEmbedFsOverrideLock() : cached(debug::rordb_embed_fs_override())
    {
        debug::rordb_embed_fs_override() = true;
    }

    ~TestRordbEmbedFsOverrideLock() { debug::rordb_embed_fs_override() = cached; }

private:
    bool cached;
};

static auto Duration(const std::function<void()>& func)
{
    const auto start = std::chrono::steady_clock::now();
    func();
    return std::chrono::steady_clock::now() - start;
}

class GPU_FindDb_FP32 : public testing::Test
{
protected:
    void SetUp() override
    {
        filter.findMode.Set(FindMode::Values::Hybrid);
        x = {100, 25, 32, 32};
        w = {300, 25, 3, 3};
        y = tensor<float>{filter.GetForwardOutputTensor(x.desc, w.desc)};

        x_dev = handle.Write(x.data);
        w_dev = handle.Write(w.data);
        y_dev = handle.Write(y.data);

        const TempFile temp_file{"miopen.test.find_db"};
        debug::testing_find_db_path_override() = temp_file;
    }

    Handle handle{};
    tensor<float> x;
    tensor<float> w;
    tensor<float> y;
    Allocator::ManageDataPtr x_dev;
    Allocator::ManageDataPtr w_dev;
    Allocator::ManageDataPtr y_dev;
    // --input 100,25,32,32 --weights 300,25,3,3 --filter 0,0,1,1,1,1,
    miopen::ConvolutionDescriptor filter = {
        2, miopenConvolution, miopenPaddingDefault, {0, 0}, {1, 1}, {1, 1}};

    void TestBwdData()
    {
        MIOPEN_LOG_I("Starting backward find-db test.");

        const auto ctx = ExecutionContext{&handle};
        const auto problem =
            conv::ProblemDescription{y.desc, w.desc, x.desc, filter, conv::Direction::BackwardData};
        Workspace wspace{filter.GetWorkSpaceSize(ctx, problem)};

        auto filterCall = [&]() {
            int ret_algo_count;
            miopenConvAlgoPerf_t perf[1];

            filter.FindConvBwdDataAlgorithm(handle,
                                            y.desc,
                                            y_dev.get(),
                                            w.desc,
                                            w_dev.get(),
                                            x.desc,
                                            x_dev.get(),
                                            1,
                                            &ret_algo_count,
                                            perf,
                                            wspace.ptr(),
                                            wspace.size(),
                                            false);
        };

        Test(filterCall);
    }

    void TestForward()
    {
        std::cout << "Starting forward find-db test." << std::endl;

        const auto ctx = ExecutionContext{&handle};
        const auto problem =
            conv::ProblemDescription{x.desc, w.desc, y.desc, filter, conv::Direction::Forward};
        Workspace wspace{filter.GetWorkSpaceSize(ctx, problem)};

        auto filterCall = [&]() {
            int ret_algo_count;
            miopenConvAlgoPerf_t perf[1];

            filter.FindConvFwdAlgorithm(handle,
                                        x.desc,
                                        x_dev.get(),
                                        w.desc,
                                        w_dev.get(),
                                        y.desc,
                                        y_dev.get(),
                                        1,
                                        &ret_algo_count,
                                        perf,
                                        wspace.ptr(),
                                        wspace.size(),
                                        false);
        };

        Test(filterCall);
    }

    void TestWeights()
    {
        MIOPEN_LOG_I("Starting wrw find-db test.");

        const auto ctx     = ExecutionContext{&handle};
        const auto problem = conv::ProblemDescription{
            y.desc, w.desc, x.desc, filter, conv::Direction::BackwardWeights};
        Workspace wspace{filter.GetWorkSpaceSize(ctx, problem)};

        auto filterCall = [&]() {
            int ret_algo_count;
            miopenConvAlgoPerf_t perf[1];

            filter.FindConvBwdWeightsAlgorithm(handle,
                                               y.desc,
                                               y_dev.get(),
                                               x.desc,
                                               x_dev.get(),
                                               w.desc,
                                               w_dev.get(),
                                               1,
                                               &ret_algo_count,
                                               perf,
                                               wspace.ptr(),
                                               wspace.size(),
                                               false);
        };

        Test(filterCall);
    }

    void Test(const std::function<void()>& func)
    {
        using mSeconds = std::chrono::duration<double, std::ratio<1, 1000>>;

        TestRordbEmbedFsOverrideLock rordb_embed_fs_override;

        const auto time0   = Duration(func);
        const auto time0ms = std::chrono::duration_cast<mSeconds>(time0);
        MIOPEN_LOG_I(
            "Find(), 1st call (populating kcache in RAM, updating find-db): " << time0ms.count());

        debug::testing_find_db_enabled = false;
        const auto time1               = Duration(func);
        const auto time1ms             = std::chrono::duration_cast<mSeconds>(time1);
        MIOPEN_LOG_I("Find(), find-db disabled: " << time1ms.count());

        debug::testing_find_db_enabled = true;
        const auto time2               = Duration(func);
        const auto time2ms             = std::chrono::duration_cast<mSeconds>(time2);
        MIOPEN_LOG_I("Find(), find-db enabled: " << time2ms.count());

        const auto find_db_speedup = time1ms / time2ms;
        MIOPEN_LOG_I("Speedup: " << find_db_speedup);
#if !MIOPEN_DISABLE_USERDB
        double limit = 3.0;
        EXPECT_GE(find_db_speedup, limit);
#endif
    }
};

TEST_F(GPU_FindDb_FP32, FloatTest_find_db)
{
    ScopedEnvironment<int> logging_time(MIOPEN_ENABLE_LOGGING_ELAPSED_TIME, 1);
    ScopedEnvironment<int> log_level(MIOPEN_LOG_LEVEL, 2);
    ScopedEnvironment<int> compile_parallel(MIOPEN_COMPILE_PARALLEL_LEVEL, 1);

    TestForward();
    TestBwdData();
    TestWeights();
}

} // namespace miopen

