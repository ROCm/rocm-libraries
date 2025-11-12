/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

rocsparse_clients::configurable_event_listener::configurable_event_listener(
    testing::TestEventListener* theEventListener)
    : m_eventListener(theEventListener)
    , showTestCases(true)
    , showTestNames(true)
    , showSuccesses(true)
    , showInlineFailures(true)
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
        m_eventListener->OnTestStart(test_info);
    }
}

void rocsparse_clients::configurable_event_listener::OnTestPartResult(
    const testing::TestPartResult& result)
{
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

    if(test_info.result()->Failed() ? showInlineFailures : showSuccesses)
    {
        m_eventListener->OnTestEnd(test_info);
    }
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
    m_eventListener->OnTestIterationEnd(unit_test, iteration);
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
