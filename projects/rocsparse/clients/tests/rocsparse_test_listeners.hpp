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
#pragma once

#include <gtest/gtest.h>
#include <sstream>
#include <streambuf>

namespace rocsparse_clients
{
    // Helper class to redirect streams
    class stream_redirector
    {
    private:
        std::streambuf*    m_old_cout_buf{};
        std::streambuf*    m_old_cerr_buf{};
        std::ostringstream m_stream;

    public:
        void redirect();

        void restore();

        std::ostringstream& get_stream();

        void clear();
    };

    class configurable_event_listener : public testing::TestEventListener
    {
    private:
        stream_redirector           m_redirector;
        testing::TestEventListener* m_eventListener;

    public:
        bool showTestCases; // Show the names of each test case.
        bool showTestNames; // Show the names of each test.
        bool showSuccesses; // Show each success.
        bool showInlineFailures; // Show each failure as it occurs.
        bool showEnvironment; // Show the setup of the global environment.
        bool redirectOutput; // Redirect output to a stringstream.

        explicit configurable_event_listener(testing::TestEventListener* theEventListener);

        ~configurable_event_listener() override;

        void OnTestProgramStart(const testing::UnitTest& unit_test) override;

        void OnTestIterationStart(const testing::UnitTest& unit_test, int iteration) override;

        void OnEnvironmentsSetUpStart(const testing::UnitTest& unit_test) override;

        void OnEnvironmentsSetUpEnd(const testing::UnitTest& unit_test) override;

        void OnTestCaseStart(const testing::TestCase& test_case) override;

        void OnTestStart(const testing::TestInfo& test_info) override;

        void OnTestPartResult(const testing::TestPartResult& result) override;

        void OnTestEnd(const testing::TestInfo& test_info) override;

        void OnTestCaseEnd(const testing::TestCase& test_case) override;

        void OnEnvironmentsTearDownStart(const testing::UnitTest& unit_test) override;

        void OnEnvironmentsTearDownEnd(const testing::UnitTest& unit_test) override;

        void OnTestIterationEnd(const testing::UnitTest& unit_test, int iteration) override;

        void OnTestProgramEnd(const testing::UnitTest& unit_test) override;
    };

} // namespace rocsparse_clients
