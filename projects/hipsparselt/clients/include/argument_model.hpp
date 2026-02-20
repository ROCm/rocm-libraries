/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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

#pragma once

#include "hipsparselt_arguments.hpp"

namespace ArgumentLogging
{
    const double NA_value = -1.0; // invalid for time, GFlop, GB
}

void ArgumentModel_set_log_function_name(bool f);
bool ArgumentModel_get_log_function_name();

// ArgumentModel template has a variadic list of argument enums
template <hipsparselt_argument... Args>
class ArgumentModel
{
    // Whether model has a particular parameter
    // TODO: Replace with C++17 fold expression ((Args == param) || ...)
    static constexpr bool has(hipsparselt_argument param)
    {
        for(auto x : {Args...})
            if(x == param)
                return true;
        return false;
    }

public:
    void log_perf(hipsparselt_internal_ostream& name_line,
                  hipsparselt_internal_ostream& val_line,
                  const Arguments&              arg,
                  double                        gpu_us,
                  double                        gflops,
                  double                        gbytes,
                  double                        cpu_us,
                  double                        norm1,
                  double                        norm2,
                  double                        norm3,
                  double                        norm4,
                  int                           config_id)
    {
        constexpr bool has_batch_count = has(e_batch_count);
        int64_t        batch_count     = has_batch_count ? arg.batch_count : 1;
        int64_t        hot_calls       = arg.iters < 1 ? 1 : arg.iters;

        if(config_id != ArgumentLogging::NA_value)
        {
            name_line << ",config_id";
            val_line << ", " << config_id;
        }

        // gpu time is total cumulative over hot calls, cpu is not
        if(hot_calls > 1)
            gpu_us /= hot_calls;

        // per/us to per/sec *10^6
        double hipsparselt_gflops = gflops * batch_count / gpu_us * 1e6;
        double hipsparselt_GBps   = gbytes * batch_count / gpu_us * 1e6;

        // append performance fields
        if(gflops != ArgumentLogging::NA_value)
        {
            name_line << ",hipsparselt-Gflops";
            val_line << ", " << hipsparselt_gflops;
        }

        if(gbytes != ArgumentLogging::NA_value)
        {
            // GB/s not usually reported for non-memory bound functions
            name_line << ",hipsparselt-GB/s";
            val_line << ", " << hipsparselt_GBps;
        }

        name_line << ",us";
        val_line << ", " << gpu_us;

        if(arg.unit_check || arg.norm_check)
        {
            if(cpu_us != ArgumentLogging::NA_value)
            {
                if(gflops != ArgumentLogging::NA_value)
                {
                    double cblas_gflops = gflops * batch_count / cpu_us * 1e6;
                    name_line << ",CPU-Gflops";
                    val_line << "," << cblas_gflops;
                }

                name_line << ",CPU-us";
                val_line << "," << cpu_us;
            }
            if(arg.norm_check)
            {
                if(norm1 != ArgumentLogging::NA_value)
                {
                    name_line << ",norm_error_1";
                    val_line << "," << norm1;
                }
                if(norm2 != ArgumentLogging::NA_value)
                {
                    name_line << ",norm_error_2";
                    val_line << "," << norm2;
                }
                if(norm3 != ArgumentLogging::NA_value)
                {
                    name_line << ",norm_error_3";
                    val_line << "," << norm3;
                }
                if(norm4 != ArgumentLogging::NA_value)
                {
                    name_line << ",norm_error_4";
                    val_line << "," << norm4;
                }
            }
        }
    }

    template <typename T>
    void log_args(hipsparselt_internal_ostream& str,
                  const Arguments&              arg,
                  double                        gpu_us,
                  double                        gflops,
                  double                        gpu_bytes = ArgumentLogging::NA_value,
                  double                        cpu_us    = ArgumentLogging::NA_value,
                  double                        norm1     = ArgumentLogging::NA_value,
                  double                        norm2     = ArgumentLogging::NA_value,
                  double                        norm3     = ArgumentLogging::NA_value,
                  double                        norm4     = ArgumentLogging::NA_value,
                  int                           config_id = ArgumentLogging::NA_value)
    {
        hipsparselt_internal_ostream name_list;
        hipsparselt_internal_ostream value_list;

        if(ArgumentModel_get_log_function_name())
        {
            auto delim = ",";
            name_list << "function" << delim;
            value_list << arg.function << delim;
        }

        // Output (name, value) pairs to name_list and value_list
        auto print = [&, delim = ""](const char* name, auto&& value) mutable {
            name_list << delim << name;
            value_list << delim << value;
            delim = ",";
        };

#if __cplusplus >= 201703L
        // C++17
        (ArgumentsHelper::apply<Args>(print, arg, T{}), ...);
#else
        // C++14. TODO: Remove when C++17 is used
        (void)(int[]){(ArgumentsHelper::apply<Args>{}()(print, arg, T{}), 0)...};
#endif

        if(arg.timing)
            log_perf(name_list,
                     value_list,
                     arg,
                     gpu_us,
                     gflops,
                     gpu_bytes,
                     cpu_us,
                     norm1,
                     norm2,
                     norm3,
                     norm4,
                     config_id);

        str << name_list << "\n" << value_list << std::endl;
    }
};
