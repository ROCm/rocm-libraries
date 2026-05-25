// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared scaffolding for the 2D and 3D large-stride API applicability sweeps
// (conv_api_solution_count_{2d,3d}_large_stride.cpp).
//
// The two sweeps differ only in shape rank and per-dimension data; the
// descriptor lifecycle, gtest plumbing, and the GTEST_SKIP pattern below are
// identical, so they live here.
//
// Why skip-early for known-failing shapes:
//   The IsKnownFailing* lists in the .cpp files enumerate shapes that CK
//   currently rejects (most are >INT_MAX where no large-tensor instance is
//   registered yet). The helpers skip these shapes *before* calling
//   CompileSolution, so the failing CK code path never runs and its error
//   logs stay out of test output. The lists are non-blocking documentation:
//   when upstream CK fills a gap, the corresponding entry simply becomes
//   stale and that shape continues to report SKIPPED (never FAILED), so CK
//   integration cannot break the build. Trim stale entries opportunistically.
//
// WrappingPrinter: gtest's default summary lists SKIPPED test names without
// their GTEST_SKIP() messages. We wrap the default pretty-printer and replace
// only OnTestIterationEnd so the summary appends each reason inline:
//   [  SKIPPED ] Suite.Test -- <solver> known-failing (CK applicability gap)

#pragma once

#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

#include "gtest_common.hpp"

namespace miopen_test_large_stride {

// Wraps gtest's default pretty-printer so the end-of-run summary appends each
// SKIPPED test's GTEST_SKIP() message inline:
//   [  SKIPPED ] Suite.Test -- <solver> known-failing (CK applicability gap)
// All other listener events are forwarded unchanged to the default printer, so
// per-test [ RUN ] / [  OK  ] / [  SKIPPED ] / [ FAILED ] output is unaffected.
class WrappingPrinter : public ::testing::EmptyTestEventListener
{
public:
    explicit WrappingPrinter(::testing::TestEventListener* inner) : inner_(inner) {}

    void OnTestProgramStart(const ::testing::UnitTest& u) override
    {
        inner_->OnTestProgramStart(u);
    }
    void OnTestIterationStart(const ::testing::UnitTest& u, int i) override
    {
        inner_->OnTestIterationStart(u, i);
    }
    void OnEnvironmentsSetUpStart(const ::testing::UnitTest& u) override
    {
        inner_->OnEnvironmentsSetUpStart(u);
    }
    void OnEnvironmentsSetUpEnd(const ::testing::UnitTest& u) override
    {
        inner_->OnEnvironmentsSetUpEnd(u);
    }
    void OnTestSuiteStart(const ::testing::TestSuite& s) override { inner_->OnTestSuiteStart(s); }
    void OnTestStart(const ::testing::TestInfo& t) override { inner_->OnTestStart(t); }
    void OnTestPartResult(const ::testing::TestPartResult& r) override
    {
        inner_->OnTestPartResult(r);
    }
    void OnTestEnd(const ::testing::TestInfo& t) override { inner_->OnTestEnd(t); }
    void OnTestSuiteEnd(const ::testing::TestSuite& s) override { inner_->OnTestSuiteEnd(s); }
    void OnEnvironmentsTearDownStart(const ::testing::UnitTest& u) override
    {
        inner_->OnEnvironmentsTearDownStart(u);
    }
    void OnEnvironmentsTearDownEnd(const ::testing::UnitTest& u) override
    {
        inner_->OnEnvironmentsTearDownEnd(u);
    }
    void OnTestIterationEnd(const ::testing::UnitTest& u, int /*i*/) override
    {
        // Replaces the default's summary so SKIPPED lines carry their reason.
        PrintEnhancedSummary(u);
    }
    void OnTestProgramEnd(const ::testing::UnitTest& u) override { inner_->OnTestProgramEnd(u); }

private:
    std::unique_ptr<::testing::TestEventListener> inner_;

    static std::string GetSkipReason(const ::testing::TestInfo& info)
    {
        const auto& result = *info.result();
        for(int k = 0; k < result.total_part_count(); ++k)
        {
            const auto& part = result.GetTestPartResult(k);
            if(part.skipped())
            {
                std::string reason = part.message();
                if(auto nl = reason.find('\n'); nl != std::string::npos)
                    reason.resize(nl);
                return reason;
            }
        }
        return {};
    }

    static void PrintEnhancedSummary(const ::testing::UnitTest& u)
    {
        std::printf("[==========] %d tests from %d test suites ran. (%lld ms total)\n",
                    u.test_to_run_count(),
                    u.test_suite_to_run_count(),
                    static_cast<long long>(u.elapsed_time()));
        std::printf("[  PASSED  ] %d tests.\n", u.successful_test_count());

        const int skipped = u.skipped_test_count();
        if(skipped > 0)
        {
            std::printf("[  SKIPPED ] %d tests, listed below:\n", skipped);
            for(int i = 0; i < u.total_test_suite_count(); ++i)
            {
                const auto* suite = u.GetTestSuite(i);
                for(int j = 0; j < suite->total_test_count(); ++j)
                {
                    const auto* info = suite->GetTestInfo(j);
                    if(!info->result()->Skipped())
                        continue;
                    const std::string reason = GetSkipReason(*info);
                    if(reason.empty())
                        std::printf("[  SKIPPED ] %s.%s\n", info->test_suite_name(), info->name());
                    else
                        std::printf("[  SKIPPED ] %s.%s -- %s\n",
                                    info->test_suite_name(),
                                    info->name(),
                                    reason.c_str());
                }
            }
        }

        const int failed = u.failed_test_count();
        if(failed > 0)
        {
            std::printf("[  FAILED  ] %d tests, listed below:\n", failed);
            for(int i = 0; i < u.total_test_suite_count(); ++i)
            {
                const auto* suite = u.GetTestSuite(i);
                for(int j = 0; j < suite->total_test_count(); ++j)
                {
                    const auto* info = suite->GetTestInfo(j);
                    if(!info->result()->Failed())
                        continue;
                    std::printf("[  FAILED  ] %s.%s\n", info->test_suite_name(), info->name());
                }
            }
            std::printf("\n%2d FAILED TEST%s\n", failed, failed == 1 ? "" : "S");
        }
    }
};

inline bool RegisterWrappingPrinter()
{
    auto& listeners       = ::testing::UnitTest::GetInstance()->listeners();
    auto* default_printer = listeners.Release(listeners.default_result_printer());
    listeners.Append(new WrappingPrinter(default_printer));
    return true;
}

// Static initializer -- runs once per binary that #includes this header.
// Inline variable (C++17) ensures a single definition across translation units.
inline const bool g_wrapping_printer_registered = RegisterWrappingPrinter();

struct Descriptors
{
    miopenHandle_t handle                  = nullptr;
    miopenTensorDescriptor_t xDesc         = nullptr;
    miopenTensorDescriptor_t wDesc         = nullptr;
    miopenTensorDescriptor_t yDesc         = nullptr;
    miopenConvolutionDescriptor_t convDesc = nullptr;

    ~Descriptors()
    {
        if(yDesc != nullptr)
            miopenDestroyTensorDescriptor(yDesc);
        if(convDesc != nullptr)
            miopenDestroyConvolutionDescriptor(convDesc);
        if(wDesc != nullptr)
            miopenDestroyTensorDescriptor(wDesc);
        if(xDesc != nullptr)
            miopenDestroyTensorDescriptor(xDesc);
        if(handle != nullptr)
            miopenDestroy(handle);
    }
};

inline uint64_t SolverIdFromName(const char* name) { return miopen::solver::Id(name).Value(); }

// SetupDescriptorsImpl -- shared rank-templated descriptor builder. The
// reproducer-family tests use uniform pad=1, stride=1, dilation=1 in all
// spatial dims, so those are hard-coded here.
template <int Ndim>
inline ::testing::AssertionResult
SetupDescriptorsImpl(const int* x_dims, const int* w_dims, miopenDataType_t dtype, Descriptors& d)
{
    static_assert(Ndim == 2 || Ndim == 3, "Only 2D/3D supported");
    constexpr int rank = Ndim + 2;

    if(miopenCreateWithStream(&d.handle, /*stream=*/nullptr) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "miopenCreateWithStream failed";

    if(miopenCreateTensorDescriptor(&d.xDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create xDesc failed";
    if(miopenSetTensorDescriptor(d.xDesc, dtype, rank, const_cast<int*>(x_dims), nullptr) !=
       miopenStatusSuccess)
        return ::testing::AssertionFailure() << "set xDesc failed";

    if(miopenCreateTensorDescriptor(&d.wDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create wDesc failed";
    if(miopenSetTensorDescriptor(d.wDesc, dtype, rank, const_cast<int*>(w_dims), nullptr) !=
       miopenStatusSuccess)
        return ::testing::AssertionFailure() << "set wDesc failed";

    if(miopenCreateConvolutionDescriptor(&d.convDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create convDesc failed";
    {
        int pads[Ndim];
        int strides[Ndim];
        int dils[Ndim];
        for(int i = 0; i < Ndim; ++i)
        {
            pads[i]    = 1;
            strides[i] = 1;
            dils[i]    = 1;
        }
        if(miopenInitConvolutionNdDescriptor(
               d.convDesc, Ndim, pads, strides, dils, miopenConvolution) != miopenStatusSuccess)
            return ::testing::AssertionFailure() << "init convDesc failed";
    }

    if(miopenCreateTensorDescriptor(&d.yDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create yDesc failed";
    {
        int yDim[rank] = {0};
        int yNbDims    = 0;
        if(miopenGetConvolutionNdForwardOutputDim(d.convDesc, d.xDesc, d.wDesc, &yNbDims, yDim) !=
           miopenStatusSuccess)
            return ::testing::AssertionFailure() << "get yDim failed";
        if(yNbDims != rank)
            return ::testing::AssertionFailure() << "yNbDims != " << rank;
        if(miopenSetTensorDescriptor(d.yDesc, dtype, rank, yDim, nullptr) != miopenStatusSuccess)
            return ::testing::AssertionFailure() << "set yDesc failed";
    }
    return ::testing::AssertionSuccess();
}

// CI-covered architectures for these sweeps. IsKnownFailing* lists below were
// hand-tuned for gfx942 CK tile coverage; we only run on arches whose CK
// coverage has been characterized in CI, so non-allowlisted GPUs SKIP instead
// of emitting stale FAILED lines. gfx115X (RDNA 3.5) is intentionally omitted:
// the CK *Xdlops solvers target CDNA MFMA, and a gfx1151 CI run produced 26
// failures across sub- and >INT_MAX shapes -- re-add once gfx115X CK coverage
// is characterized and the IsKnownFailing* lists grow per-arch entries.
inline bool IsArchInCiAllowlist()
{
    return IsTestSupportedByDevice(Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950);
}

// Run* helpers -- templated on Shape, SetupFn (per-rank descriptor wrapper),
// and KnownFailingFn (per-direction predicate). Direction-specific Compile
// API call is the only thing that varies between the three.
//
// Known-failing shapes are skipped *before* descriptor setup and the
// CompileSolution call, so the failing CK code path never runs. Diagnostic
// note: the test framework's parameter machinery prints the shape, so we
// keep the skip/failure messages terse (solver name only).

template <typename Shape, typename SetupFn, typename KnownFailingFn>
void RunCompileFwd(const Shape& s,
                   miopenDataType_t dtype,
                   SetupFn setup_fn,
                   KnownFailingFn is_known_failing,
                   const char* solver_name)
{
    if(!IsArchInCiAllowlist())
        GTEST_SKIP() << "Architecture not in CI allowlist (gfx90A/gfx94X/gfx950)";

    if(is_known_failing(dtype, s))
        GTEST_SKIP() << solver_name << " known-failing (CK applicability gap)";

    Descriptors d;
    ASSERT_TRUE(setup_fn(s, dtype, d));

    const auto status = miopenConvolutionForwardCompileSolution(
        d.handle, d.wDesc, d.xDesc, d.convDesc, d.yDesc, SolverIdFromName(solver_name));
    EXPECT_EQ(status, miopenStatusSuccess) << solver_name << " not applicable/compilable";
}

template <typename Shape, typename SetupFn, typename KnownFailingFn>
void RunCompileBwdData(const Shape& s,
                       miopenDataType_t dtype,
                       SetupFn setup_fn,
                       KnownFailingFn is_known_failing,
                       const char* solver_name)
{
    if(!IsArchInCiAllowlist())
        GTEST_SKIP() << "Architecture not in CI allowlist (gfx90A/gfx94X/gfx950)";

    if(is_known_failing(dtype, s))
        GTEST_SKIP() << solver_name << " known-failing (CK applicability gap)";

    Descriptors d;
    ASSERT_TRUE(setup_fn(s, dtype, d));

    // dyDesc has y's shape, dxDesc has x's shape.
    const auto status = miopenConvolutionBackwardDataCompileSolution(
        d.handle, d.yDesc, d.wDesc, d.convDesc, d.xDesc, SolverIdFromName(solver_name));
    EXPECT_EQ(status, miopenStatusSuccess) << solver_name << " not applicable/compilable";
}

template <typename Shape, typename SetupFn, typename KnownFailingFn>
void RunCompileWrw(const Shape& s,
                   miopenDataType_t dtype,
                   SetupFn setup_fn,
                   KnownFailingFn is_known_failing,
                   const char* solver_name)
{
    if(!IsArchInCiAllowlist())
        GTEST_SKIP() << "Architecture not in CI allowlist (gfx90A/gfx94X/gfx950)";

    if(is_known_failing(dtype, s))
        GTEST_SKIP() << solver_name << " known-failing (CK applicability gap)";

    Descriptors d;
    ASSERT_TRUE(setup_fn(s, dtype, d));

    // dyDesc has y's shape, xDesc has x's shape, dwDesc has w's shape.
    const auto status = miopenConvolutionBackwardWeightsCompileSolution(
        d.handle, d.yDesc, d.xDesc, d.convDesc, d.wDesc, SolverIdFromName(solver_name));
    EXPECT_EQ(status, miopenStatusSuccess) << solver_name << " not applicable/compilable";
}

} // namespace miopen_test_large_stride
