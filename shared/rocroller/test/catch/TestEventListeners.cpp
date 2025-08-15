/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <iostream>

#include <omp.h>

#include <catch2/catch_test_case_info.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

class OpenMPSetupListener : public Catch::EventListenerBase
{
private:
    int originalMaxActiveLevels = 0;

public:
    using Catch::EventListenerBase::EventListenerBase;

    void testCaseStarting(Catch::TestCaseInfo const& testInfo) override
    {
        originalMaxActiveLevels = omp_get_max_active_levels();
        
        // Ensure all functions that use OpenMP have [OPENMP] tag
        // Note: All uses of #pragma omp should include an assertion for omp_get_max_active_levels() >= 1;
        if(std::find(testInfo.tags.begin(), testInfo.tags.end(), Catch::Tag("openmp"))
           != testInfo.tags.end())
            omp_set_max_active_levels(1);
        else
            omp_set_max_active_levels(0);
    }

    void testCaseEnded(Catch::TestCaseStats const&) override
    {
        omp_set_max_active_levels(originalMaxActiveLevels);
    }
};

CATCH_REGISTER_LISTENER(OpenMPSetupListener)