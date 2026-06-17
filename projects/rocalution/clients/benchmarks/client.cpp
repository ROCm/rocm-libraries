/*! \file */
/* ************************************************************************
* Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocalution/rocalution.hpp"
#include "rocalution_bench.hpp"

#include "utility.hpp"
#include <iostream>

#include "rocalution_bench_app.hpp"

//
// REQUIRED ROUTINES:
//

bool rocalution_bench_record_results(const rocalution_bench_solver_parameters& params,
                                     const rocalution_bench_solver_results&    results)
{
    auto* s_bench_app = rocalution_bench_app::instance();
    if(s_bench_app)
    {
        return s_bench_app->record_results(params, results);
    }
    else
    {
        params.WriteNicely(std::cout);
        results.WriteNicely(std::cout);
        return true;
    }
}

int device = 0;
int main(int argc, char* argv[])
{
    if(rocalution_bench_app::applies(argc, argv))
    {
        std::cout << "main " << std::endl;
        try
        {
            auto* s_bench_app = rocalution_bench_app::instance(argc, argv);
            //
            // RUN CASES.
            //
            bool success = s_bench_app->run_cases();
            if(!success)
            {
                std::cout << "ERROR SUCCESS FAILURE client.cpp line " << __LINE__ << std::endl;
                return !success;
            }
            //
            // EXPORT FILE.
            //
            success = s_bench_app->export_file();
            if(!success)
            {
                std::cout << "ERROR SUCCESS FAILURE client.cpp line " << __LINE__ << std::endl;
                return !success;
            }
            return !success;
        }
        catch(const bool& success)
        {
            std::cout << "ERROR SUCCESS EXCEPTION FAILURE client.cpp line " << __LINE__
                      << std::endl;
            return !success;
        }
        catch(const std::exception&)
        {
            std::cout << "ERROR UNKNOWN EXCEPTION FAILURE client.cpp line " << __LINE__
                      << std::endl;
            throw;
        }
    }
    else
    {
        try
        {
            rocalution_bench bench(argc, argv);
            //
            // Print info devices.
            //
            bench.info_devices(std::cout);
            //
            // Run benchmark.
            //
            bool success = bench.run();
            if(!success)
            {
                std::cout << "ERROR SUCCESS FAILURE client.cpp line " << __LINE__ << std::endl;
                return !success;
            }
            return !success;
        }
        catch(const std::exception&)
        {
            std::cout << "ERROR UNKNOWN EXCEPTION FAILURE client.cpp line " << __LINE__
                      << std::endl;
            throw;
        }
    }
    return 0;
}
