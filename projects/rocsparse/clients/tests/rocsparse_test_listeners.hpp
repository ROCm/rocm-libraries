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

#include <sstream>
#include <streambuf>
#include <gtest/gtest.h>

namespace rocsparse_clients
{

    class ConfigurableEventListener : public testing::TestEventListener
    {
        testing::TestEventListener* eventListener;

    public:
        bool showTestCases; // Show the names of each test case.
        bool showTestNames; // Show the names of each test.
        bool showSuccesses; // Show each success.
        bool showInlineFailures; // Show each failure as it occurs.
        bool showEnvironment; // Show the setup of the global environment.

        explicit ConfigurableEventListener(testing::TestEventListener* theEventListener)
            : eventListener(theEventListener)
            , showTestCases(true)
            , showTestNames(true)
            , showSuccesses(true)
            , showInlineFailures(true)
            , showEnvironment(true)
        {
        }

        ~ConfigurableEventListener() override
        {
            delete eventListener;
        }

        void OnTestProgramStart(const testing::UnitTest& unit_test) override
        {
            eventListener->OnTestProgramStart(unit_test);
        }

        void OnTestIterationStart(const testing::UnitTest& unit_test, int iteration) override
        {
            eventListener->OnTestIterationStart(unit_test, iteration);
        }

        void OnEnvironmentsSetUpStart(const testing::UnitTest& unit_test) override
        {
            if(showEnvironment)
            {
                eventListener->OnEnvironmentsSetUpStart(unit_test);
            }
        }

        void OnEnvironmentsSetUpEnd(const testing::UnitTest& unit_test) override
        {
            if(showEnvironment)
            {
                eventListener->OnEnvironmentsSetUpEnd(unit_test);
            }
        }

        void OnTestCaseStart(const testing::TestCase& test_case) override
        {
            if(showTestCases)
            {
                eventListener->OnTestCaseStart(test_case);
            }
        }

        void OnTestStart(const testing::TestInfo& test_info) override
        {
            if(showTestNames)
            {
                eventListener->OnTestStart(test_info);
            }
        }

        void OnTestPartResult(const testing::TestPartResult& result) override
        {
            eventListener->OnTestPartResult(result);
        }

        void OnTestEnd(const testing::TestInfo& test_info) override
        {
            if(test_info.result()->Failed() ? showInlineFailures : showSuccesses)
            {
                eventListener->OnTestEnd(test_info);
            }
        }

        void OnTestCaseEnd(const testing::TestCase& test_case) override
        {
            if(showTestCases)
            {
                eventListener->OnTestCaseEnd(test_case);
            }
        }

        void OnEnvironmentsTearDownStart(const testing::UnitTest& unit_test) override
        {
            if(showEnvironment)
            {
                eventListener->OnEnvironmentsTearDownStart(unit_test);
            }
        }

        void OnEnvironmentsTearDownEnd(const testing::UnitTest& unit_test) override
        {
            if(showEnvironment)
            {
                eventListener->OnEnvironmentsTearDownEnd(unit_test);
            }
        }

        void OnTestIterationEnd(const testing::UnitTest& unit_test, int iteration) override
        {
            eventListener->OnTestIterationEnd(unit_test, iteration);
        }

        void OnTestProgramEnd(const testing::UnitTest& unit_test) override
        {
            eventListener->OnTestProgramEnd(unit_test);
        }
    };

    // Helper class to redirect streams
    class StreamRedirector
    {
    private:
        std::streambuf*    m_old_cout_buf{};
        std::streambuf*    m_old_cerr_buf{};
        std::ostringstream m_cout_stream;
        std::ostringstream m_cerr_stream;

    public:
        void redirect()
        {
            // Save original buffers
            this->m_old_cout_buf = std::cout.rdbuf();
            this->m_old_cerr_buf = std::cerr.rdbuf();

            // Redirect to our stringstreams
            std::cout.rdbuf(this->m_cout_stream.rdbuf());
            std::cerr.rdbuf(this->m_cerr_stream.rdbuf());
        }

        void restore()
        {
            // Restore original buffers
            std::cout.rdbuf(this->m_old_cout_buf);
            std::cerr.rdbuf(this->m_old_cerr_buf);
        }

        std::string get_cout_content() const
        {
            return this->m_cout_stream.str();
        }

        std::string get_cerr_content() const
        {
            return this->m_cerr_stream.str();
        }

        void clear()
        {
            this->m_cout_stream.clear();
            this->m_cerr_stream.clear();
        }
    };

    // Custom test listener to handle output redirection
    class OutputRedirectListener : public ::testing::TestEventListener
    {
    private:
        StreamRedirector              m_redirector;
        ::testing::TestEventListener* m_default_listener{};

    public:
        explicit OutputRedirectListener(::testing::TestEventListener* listener)
            : m_default_listener(listener)
        {
        }

        ~OutputRedirectListener() override
        {
            delete this->m_default_listener;
        }

        void OnTestProgramStart(const ::testing::UnitTest& unit_test) override
        {
            this->m_default_listener->OnTestProgramStart(unit_test);
        }

        void OnTestIterationStart(const ::testing::UnitTest& unit_test, int iteration) override
        {
            this->m_default_listener->OnTestIterationStart(unit_test, iteration);
        }

        void OnEnvironmentsSetUpStart(const ::testing::UnitTest& unit_test) override
        {
            this->m_default_listener->OnEnvironmentsSetUpStart(unit_test);
        }

        void OnEnvironmentsSetUpEnd(const ::testing::UnitTest& unit_test) override
        {
            this->m_default_listener->OnEnvironmentsSetUpEnd(unit_test);
        }

        void OnTestCaseStart(const ::testing::TestCase& test_case) override
        {
            this->m_default_listener->OnTestCaseStart(test_case);
        }

        void OnTestStart(const ::testing::TestInfo& test_info) override
        {
            // Clear and redirect streams before each test
            this->m_redirector.clear();
            this->m_redirector.redirect();
            this->m_default_listener->OnTestStart(test_info);
        }

        void OnTestPartResult(const ::testing::TestPartResult& test_part_result) override
        {
            this->m_default_listener->OnTestPartResult(test_part_result);
        }

        void OnTestEnd(const ::testing::TestInfo& test_info) override
        {
            // Restore streams after test
            this->m_redirector.restore();

            // Check if test failed
            if(test_info.result()->Failed())
            {
                const std::string cout_content = this->m_redirector.get_cout_content();
                const std::string cerr_content = this->m_redirector.get_cerr_content();

                if(!cout_content.empty())
                {
                    std::cout << "=== CAPTURED STDOUT ===" << std::endl
                              << cout_content << std::endl;
                }
                if(!cerr_content.empty())
                {
                    std::cerr << "=== CAPTURED STDERR ===" << std::endl
                              << cerr_content << std::endl;
                }
            }

            this->m_default_listener->OnTestEnd(test_info);
        }

        void OnTestCaseEnd(const ::testing::TestCase& test_case) override
        {
            this->m_default_listener->OnTestCaseEnd(test_case);
        }

        void OnEnvironmentsTearDownStart(const ::testing::UnitTest& unit_test) override
        {
            this->m_default_listener->OnEnvironmentsTearDownStart(unit_test);
        }

        void OnEnvironmentsTearDownEnd(const ::testing::UnitTest& unit_test) override
        {
            this->m_default_listener->OnEnvironmentsTearDownEnd(unit_test);
        }

        void OnTestIterationEnd(const ::testing::UnitTest& unit_test, int iteration) override
        {
            this->m_default_listener->OnTestIterationEnd(unit_test, iteration);
        }

        void OnTestProgramEnd(const ::testing::UnitTest& unit_test) override
        {
            this->m_default_listener->OnTestProgramEnd(unit_test);
        }
    };

}