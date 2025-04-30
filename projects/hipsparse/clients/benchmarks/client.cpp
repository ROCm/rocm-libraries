/*! \file */
/* ************************************************************************
* Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "hipsparse_bench.hpp"
#include "hipsparse_bench_app.hpp"
#include "hipsparse_routine.hpp"
#include "utility.hpp"
#include <hipsparse.h>
#include <iostream>

hipsparseStatus_t hipsparse_record_output_legend(const std::string& s)
{
    auto* s_bench_app = hipsparse_bench_app::instance();
    if(s_bench_app)
    {
        auto status = s_bench_app->record_output_legend(s);
        return status;
    }
    else
    {
        return HIPSPARSE_STATUS_SUCCESS;
    }
}

hipsparseStatus_t hipsparse_record_output(const std::string& s)
{
    auto* s_bench_app = hipsparse_bench_app::instance();
    if(s_bench_app)
    {
        auto status = s_bench_app->record_output(s);
        return status;
    }
    else
    {
        return HIPSPARSE_STATUS_SUCCESS;
    }
}

hipsparseStatus_t hipsparse_record_timing(double msec, double gflops, double gbs)
{
    auto* s_bench_app = hipsparse_bench_app::instance();
    if(s_bench_app)
    {
        return s_bench_app->record_timing(msec, gflops, gbs);
    }
    else
    {
        return HIPSPARSE_STATUS_SUCCESS;
    }
}

bool display_timing_info_is_stdout_disabled()
{
    auto* s_bench_app = hipsparse_bench_app::instance();
    if(s_bench_app)
    {
        return s_bench_app->is_stdout_disabled();
    }
    else
    {
        return false;
    }
}

int main(int argc, char* argv[])
{
    if(hipsparse_bench_app::applies(argc, argv))
    {
        try
        {
            auto* s_bench_app = hipsparse_bench_app::instance(argc, argv);
            //
            // RUN CASES.
            //
            hipsparseStatus_t status = s_bench_app->run_cases();
            if(status != HIPSPARSE_STATUS_SUCCESS)
            {
                return status;
            }

            //
            // EXPORT FILE.
            //
            status = s_bench_app->export_file();
            if(status != HIPSPARSE_STATUS_SUCCESS)
            {
                return status;
            }

            return status;
        }
        catch(const hipsparseStatus_t& status)
        {
            return status;
        }
    }
    else
    {
        //
        // old style.
        //
        try
        {
            hipsparse_bench bench(argc, argv);

            //
            // Print info devices.
            //
            bench.info_devices(std::cout);

            //
            // Run benchmark.
            //
            hipsparseStatus_t status = bench.run();
            if(status != HIPSPARSE_STATUS_SUCCESS)
            {
                return status;
            }

            return status;
        }
        catch(const hipsparseStatus_t& status)
        {
            return status;
        }
    }
}
