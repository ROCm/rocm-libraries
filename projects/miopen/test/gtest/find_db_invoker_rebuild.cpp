// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Regression tests for find-db reuse across processes (ROCm/rocm-libraries#3553).
//
// Invokers are kept only in the in-memory InvokerCache of a Handle. A find-db
// record written by another process therefore never has invokers, and Find used
// to discard such records and re-run the whole search/benchmark. A fresh Handle
// has an empty InvokerCache, so it is used here to emulate a new process.

#include <gtest/gtest.h>

#include <miopen/convolution.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/find_db.hpp>
#include <miopen/handle.hpp>
#include <miopen/temp_file.hpp>

#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

/// Redirects find-db to a temporary file and restores the previous state.
class ScopedTempFindDb
{
public:
    ScopedTempFindDb()
        : prev_enabled(miopen::debug::testing_find_db_enabled),
          prev_path(miopen::debug::testing_find_db_path_override())
    {
        miopen::debug::testing_find_db_enabled         = true;
        miopen::debug::testing_find_db_path_override() = temp_file.Path();
    }

    ~ScopedTempFindDb()
    {
        miopen::debug::testing_find_db_enabled         = prev_enabled;
        miopen::debug::testing_find_db_path_override() = prev_path;
    }

    ScopedTempFindDb(const ScopedTempFindDb&)            = delete;
    ScopedTempFindDb& operator=(const ScopedTempFindDb&) = delete;

    miopen::fs::path Path() const { return temp_file.Path(); }

private:
    miopen::TempFile temp_file{"miopen.test.find_db_invoker_rebuild"};
    bool prev_enabled;
    std::optional<miopen::fs::path> prev_path;
};

struct ConvFixture
{
    miopen::TensorDescriptor x_desc{miopenFloat, {2, 16, 16, 16}};
    miopen::TensorDescriptor w_desc{miopenFloat, {32, 16, 3, 3}};
    miopen::ConvolutionDescriptor conv{
        2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};
    miopen::TensorDescriptor y_desc = conv.GetForwardOutputTensor(x_desc, w_desc);

    ConvFixture() { conv.findMode.Set(miopen::FindMode::Values::Normal); }

    miopen::conv::ProblemDescription Problem() const
    {
        return {x_desc, w_desc, y_desc, conv, miopen::conv::Direction::Forward};
    }
};

std::vector<float> MakeData(std::size_t n, float scale)
{
    std::vector<float> v(n);
    for(std::size_t i = 0; i < n; ++i)
        v[i] = scale * static_cast<float>(static_cast<int>(i % 13) - 6);
    return v;
}

/// Runs Find (Normal mode, like PyTorch with cudnn.benchmark=True) followed by
/// ConvolutionForward with the best algorithm, entirely on the given handle.
struct FindAndRunResult
{
    miopenConvAlgoPerf_t perf;
    std::vector<float> y;
};

FindAndRunResult FindAndRun(const miopen::Handle& handle, const ConvFixture& f)
{
    const auto x = MakeData(f.x_desc.GetElementSize(), 0.25f);
    const auto w = MakeData(f.w_desc.GetElementSize(), 0.125f);
    auto y       = std::vector<float>(f.y_desc.GetElementSize(), 0.0f);

    auto x_dev = handle.Write(x);
    auto w_dev = handle.Write(w);
    auto y_dev = handle.Write(y);

    const auto ctx      = miopen::ExecutionContext{&handle};
    const auto ws_size  = f.conv.GetWorkSpaceSize(ctx, f.Problem());
    auto ws_dev         = handle.Create(ws_size);
    auto* const ws_ptr  = ws_size > 0 ? ws_dev.get() : nullptr;
    auto ret_algo_count = 0;
    auto perf           = miopenConvAlgoPerf_t{};

    f.conv.FindConvFwdAlgorithm(handle,
                                f.x_desc,
                                x_dev.get(),
                                f.w_desc,
                                w_dev.get(),
                                f.y_desc,
                                y_dev.get(),
                                1,
                                &ret_algo_count,
                                &perf,
                                ws_ptr,
                                ws_size,
                                false);
    EXPECT_EQ(ret_algo_count, 1);

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    f.conv.ConvolutionForward(handle,
                              &alpha,
                              f.x_desc,
                              x_dev.get(),
                              f.w_desc,
                              w_dev.get(),
                              perf.fwd_algo,
                              &beta,
                              f.y_desc,
                              y_dev.get(),
                              ws_ptr,
                              ws_size);

    return {perf, handle.Read<float>(y_dev, y.size())};
}

/// Adds `offset` to every timing in the plain-text find-db, keeping their order.
void OffsetFindDbTimes(const miopen::fs::path& path, float offset)
{
    std::stringstream content;
    {
        std::ifstream in(path);
        ASSERT_TRUE(in.good()) << "find-db was not written to " << path;
        content << in.rdbuf();
    }
    // Record format: <key>=<solver>:<time>,<workspace>,<algo>;...
    const auto text = content.str();
    const auto re   = std::regex(R"(:([-+0-9.eE]+),)");
    auto updated    = std::string{};
    auto last       = text.cbegin();
    auto replaced   = 0;
    for(auto it = std::sregex_iterator(text.begin(), text.end(), re); it != std::sregex_iterator();
        ++it, ++replaced)
    {
        updated.append(last, (*it)[0].first);
        updated += ":" + std::to_string(std::stof((*it)[1].str()) + offset) + ",";
        last = (*it)[0].second;
    }
    updated.append(last, text.cend());
    ASSERT_GT(replaced, 0) << "no timings found in find-db: " << text;
    std::ofstream(path, std::ios::trunc) << updated;
}

} // namespace

TEST(GPU_FindDbInvokerRebuild_FP32, FindReusesFindDbAcrossHandles)
{
    const ScopedTempFindDb find_db;
    const auto f = ConvFixture{};

    // "Process 1": search and benchmark, populating find-db.
    const auto first = [&]() {
        const auto handle = miopen::Handle{};
        return FindAndRun(handle, f);
    }();

    // Mark the cached record. If Find benchmarks again, real (small) timings
    // will be reported instead of the shifted ones from find-db.
    constexpr float offset = 10000.0f;
    OffsetFindDbTimes(find_db.Path(), offset);

    // "Process 2": fresh handle, empty invoker cache.
    const auto handle = miopen::Handle{};
    const auto second = FindAndRun(handle, f);

    EXPECT_GE(second.perf.time, offset) << "Find re-ran the search instead of using find-db";
    EXPECT_EQ(second.perf.fwd_algo, first.perf.fwd_algo);
    ASSERT_EQ(second.y.size(), first.y.size());
    for(std::size_t i = 0; i < first.y.size(); ++i)
        ASSERT_FLOAT_EQ(second.y[i], first.y[i]) << "at index " << i;
}
