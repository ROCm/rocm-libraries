/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "rocsparse_test_listeners.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

namespace
{
    // Per-test stream-ordered mempool management for memory-constrained runners.
    //
    // rocSPARSE allocates temporaries via hipMallocAsync/hipFreeAsync on the
    // device default pool. With a non-zero pool release threshold (e.g. on
    // Windows), freed blocks are cached rather than returned to the device, so
    // across a large single-process suite the reserved pool grows monotonically
    // and exhausts a small (4 GB) card. Trimming the pool after each test
    // returns cached blocks to the device and bounds peak usage.
    //
    // Both behaviors are opt-in via environment variables so default runs are
    // byte-for-byte unchanged:
    //   ROCSPARSE_TEST_TRIM_POOL=1  -> hipMemPoolTrimTo(pool, 0) after each test
    //   ROCSPARSE_TEST_VRAM_LOG=1   -> log device free/used + pool reserved/test
    void rocsparse_test_pool_maintenance(const testing::TestInfo& test_info)
    {
        static const bool trim = []() {
            const char* e = getenv("ROCSPARSE_TEST_TRIM_POOL");
            return e && atoi(e) != 0;
        }();
        static const bool vram_log = []() {
            const char* e = getenv("ROCSPARSE_TEST_VRAM_LOG");
            return e && atoi(e) != 0;
        }();

        if(!trim && !vram_log)
        {
            return;
        }

        hipMemPool_t pool;
        if(hipDeviceGetDefaultMemPool(&pool, 0) != hipSuccess)
        {
            return;
        }

        if(trim)
        {
            (void)hipMemPoolTrimTo(pool, 0);
        }

        if(vram_log)
        {
            size_t   free_b = 0, total_b = 0;
            uint64_t reserved = 0, used = 0;
            (void)hipMemGetInfo(&free_b, &total_b);
            (void)hipMemPoolGetAttribute(pool, hipMemPoolAttrReservedMemCurrent, &reserved);
            (void)hipMemPoolGetAttribute(pool, hipMemPoolAttrUsedMemCurrent, &used);
            fprintf(stderr,
                    "[vram] %s.%s free=%zuMB used=%zuMB pool_reserved=%lluMB pool_used=%lluMB\n",
                    test_info.test_suite_name(),
                    test_info.name(),
                    (total_b - free_b) / 1048576u,
                    free_b / 1048576u,
                    (unsigned long long)(reserved / 1048576u),
                    (unsigned long long)(used / 1048576u));
        }
    }
} // namespace

rocsparse_clients::configurable_event_listener::configurable_event_listener(
    testing::TestEventListener* theEventListener)
    : m_eventListener(theEventListener)
    , m_pendingTestInfo(nullptr)
    , showTestCases(true)
    , showTestNames(true)
    , showSuccesses(true)
    , showInlineFailures(true)
    , showSkipped(true)
    , showEnvironment(true)
    , redirectOutput(true)
{
}

rocsparse_clients::configurable_event_listener::~configurable_event_listener()
{
    delete m_eventListener;
}

void rocsparse_clients::configurable_event_listener::OnTestProgramStart(
    const testing::UnitTest& unit_test)
{
    m_eventListener->OnTestProgramStart(unit_test);
}

void rocsparse_clients::configurable_event_listener::OnTestIterationStart(
    const testing::UnitTest& unit_test, int iteration)
{
    m_eventListener->OnTestIterationStart(unit_test, iteration);
}

void rocsparse_clients::configurable_event_listener::OnEnvironmentsSetUpStart(
    const testing::UnitTest& unit_test)
{
    if(showEnvironment)
    {
        m_eventListener->OnEnvironmentsSetUpStart(unit_test);
    }
}

void rocsparse_clients::configurable_event_listener::OnEnvironmentsSetUpEnd(
    const testing::UnitTest& unit_test)
{
    if(showEnvironment)
    {
        m_eventListener->OnEnvironmentsSetUpEnd(unit_test);
    }
}

void rocsparse_clients::configurable_event_listener::OnTestCaseStart(
    const testing::TestCase& test_case)
{
    if(showTestCases)
    {
        m_eventListener->OnTestCaseStart(test_case);
    }
}

void rocsparse_clients::configurable_event_listener::OnTestStart(const testing::TestInfo& test_info)
{
    if(redirectOutput)
    {
        // Clear and redirect streams before each test
        this->m_redirector.clear();
        this->m_redirector.redirect();
    }

    if(showTestNames)
    {
        if(!showSkipped)
        {
            // Buffer the test info - we'll print it in OnTestEnd if test didn't skip
            m_pendingTestInfo = &test_info;
        }
        else
        {
            m_eventListener->OnTestStart(test_info);
        }
    }
}

void rocsparse_clients::configurable_event_listener::OnTestPartResult(
    const testing::TestPartResult& result)
{
    // Suppress skip messages when showSkipped is false
    if(!showSkipped && result.skipped())
    {
        return;
    }
    m_eventListener->OnTestPartResult(result);
}

void rocsparse_clients::configurable_event_listener::OnTestEnd(const testing::TestInfo& test_info)
{
    if(redirectOutput)
    {
        // Restore streams after test
        this->m_redirector.restore();

        // Check if test failed
        if(test_info.result()->Failed())
        {
            const std::string content = this->m_redirector.get_stream().str();

            if(!content.empty())
            {
                std::cerr << content << std::endl;
            }
        }
    }

    bool show = false;
    if(test_info.result()->Failed())
    {
        show = showInlineFailures;
    }
    else if(test_info.result()->Skipped())
    {
        show = showSkipped;
    }
    else
    {
        show = showSuccesses;
    }

    if(show)
    {
        // Print buffered OnTestStart if we were deferring it
        if(m_pendingTestInfo != nullptr)
        {
            m_eventListener->OnTestStart(*m_pendingTestInfo);
        }
        m_eventListener->OnTestEnd(test_info);
    }
    m_pendingTestInfo = nullptr;

    rocsparse_test_pool_maintenance(test_info);
}

void rocsparse_clients::configurable_event_listener::OnTestCaseEnd(
    const testing::TestCase& test_case)
{
    if(showTestCases)
    {
        m_eventListener->OnTestCaseEnd(test_case);
    }
}

void rocsparse_clients::configurable_event_listener::OnEnvironmentsTearDownStart(
    const testing::UnitTest& unit_test)
{
    if(showEnvironment)
    {
        m_eventListener->OnEnvironmentsTearDownStart(unit_test);
    }
}

void rocsparse_clients::configurable_event_listener::OnEnvironmentsTearDownEnd(
    const testing::UnitTest& unit_test)
{
    if(showEnvironment)
    {
        m_eventListener->OnEnvironmentsTearDownEnd(unit_test);
    }
}

void rocsparse_clients::configurable_event_listener::OnTestIterationEnd(
    const testing::UnitTest& unit_test, int iteration)
{
    if(!showSkipped)
    {
        // Print custom summary without listing skipped tests
        std::cout << "[==========] " << unit_test.total_test_count() << " tests from "
                  << unit_test.total_test_suite_count() << " test suites ran. ("
                  << unit_test.elapsed_time() << " ms total)" << std::endl;
        std::cout << "[  PASSED  ] " << unit_test.successful_test_count() << " tests." << std::endl;
        if(unit_test.skipped_test_count() > 0)
        {
            std::cout << "[  SKIPPED ] " << unit_test.skipped_test_count() << " tests."
                      << std::endl;
        }
        if(unit_test.failed_test_count() > 0)
        {
            std::cout << "[  FAILED  ] " << unit_test.failed_test_count() << " tests." << std::endl;
        }
    }
    else
    {
        m_eventListener->OnTestIterationEnd(unit_test, iteration);
    }
}

void rocsparse_clients::configurable_event_listener::OnTestProgramEnd(
    const testing::UnitTest& unit_test)
{
    m_eventListener->OnTestProgramEnd(unit_test);
}

void rocsparse_clients::stream_redirector::redirect()
{
    // Save original buffers
    this->m_old_cout_buf = std::cout.rdbuf();
    this->m_old_cerr_buf = std::cerr.rdbuf();

    // Redirect to our ostringstream
    std::cout.rdbuf(this->m_stream.rdbuf());
    std::cerr.rdbuf(this->m_stream.rdbuf());
}

void rocsparse_clients::stream_redirector::restore()
{
    // Restore original buffers
    std::cout.rdbuf(this->m_old_cout_buf);
    std::cerr.rdbuf(this->m_old_cerr_buf);
}

std::ostringstream& rocsparse_clients::stream_redirector::get_stream()
{
    return this->m_stream;
}

void rocsparse_clients::stream_redirector::clear()
{
    this->m_stream.str("");
}
