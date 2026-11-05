// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Exercise the benchmark definitions in a separate translation unit of the host test binary.
#undef GOOGLE_TEST
#include "client_reporting.hpp"
#include <gtest/gtest.h>

namespace
{
    class CaptureClientStream
    {
        hipblaslt_internal_ostream& stream;
        hipblaslt_internal_ostream  saved;

    public:
        explicit CaptureClientStream(hipblaslt_internal_ostream& target)
            : stream(target)
            , saved(std::move(target))
        {
            stream = hipblaslt_internal_ostream{};
        }

        ~CaptureClientStream()
        {
            stream = std::move(saved);
        }

        std::string str() const
        {
            return stream.str();
        }
    };

    TEST(HostUnitBenchmarkReporting, smoke_FatalReportReturnsFromCaller)
    {
        CaptureClientStream capture(hipblaslt_cerr);
        int                 evaluations = 0;
        bool                continued   = false;
        [&] {
            for(int i = 0; i < 2; ++i)
                HIPBLASLT_FAIL("failure " << ++evaluations);
            continued = true;
        }();
        const std::string output = capture.str();
        EXPECT_FALSE(continued);
        EXPECT_EQ(evaluations, 1);
        EXPECT_EQ(output, "failure 1\n");
    }

    TEST(HostUnitBenchmarkReporting, smoke_SkipReturnsFromCaller)
    {
        CaptureClientStream capture(hipblaslt_cout);
        bool                continued = false;
        [&] {
            HIPBLASLT_SKIP("unsupported case");
            continued = true;
        }();
        const std::string output = capture.str();
        EXPECT_FALSE(continued);
        EXPECT_EQ(output, "unsupported case\n");
    }

    TEST(HostUnitBenchmarkReporting, smoke_NonfatalReportContinues)
    {
        CaptureClientStream capture(hipblaslt_cerr);
        bool                continued = false;
        [&] {
            HIPBLASLT_REPORT_FAILURE("validation failed");
            continued = true;
        }();
        const std::string output = capture.str();
        EXPECT_TRUE(continued);
        EXPECT_EQ(output, "validation failed\n");
    }
}
