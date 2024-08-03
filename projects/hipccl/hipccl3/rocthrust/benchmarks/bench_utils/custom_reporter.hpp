// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef ROCTHRUST_BENCHMARKS_BENCH_UTILS_CUSTOM_REPORTER_HPP_
#define ROCTHRUST_BENCHMARKS_BENCH_UTILS_CUSTOM_REPORTER_HPP_

// Utils
#include "common/types.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

namespace bench_utils
{
/// \brief Custom Google Benchmark reporter for formatting the benchmarks' report matching Thrust's.
///
/// This reporter is a ConsoleReporter that outputs:
/// - GPU Time: measured with events and registered as manual time.
/// - CPU Time: measured by Google Benchmark.
/// - Iterations: the number of kernel executions done on each benchmark run.
/// - GlobalMem BW: number of GB read and written to global memory per second of execution.
/// - BW util: percentage of the theoretical peak global memory bandwith utilised by the benchmark.
/// - Elements/s: number of G elements (G = 10**6) processed by second of execution.
/// - Labels: for extra labels. For instance, labels are used by the benchmarks generating random
///            input values to report the seed used.
///
/// Additionally, when the number of \p repetitions  is greater than one, each benchmark run is
/// repeated \p repetitions times to measure the stability of results. In this case, the mean,
/// median, standard deviation (stddev) and coefficient of variation (cv) of the above-described
/// metrics are also reported after all the \p repetitions have ben run.
class CustomReporter : public benchmark::ConsoleReporter
{
private:
    enum LogColor
    {
        COLOR_DEFAULT,
        COLOR_RED,
        COLOR_GREEN,
        COLOR_YELLOW,
        COLOR_BLUE,
        COLOR_CYAN,
        COLOR_WHITE
    };

    std::string get_log_color(LogColor&& color)
    {
        std::string s("unknown");
        switch(color)
        {
        case COLOR_RED:
            return "\033[31m";
        case COLOR_GREEN:
            return "\033[32m";
        case COLOR_YELLOW:
            return "\033[33m";
        case COLOR_BLUE:
            return "\033[34m";
        case COLOR_CYAN:
            return "\033[36m";
        case COLOR_WHITE:
            return "\033[37m";
        default: // COLOR_DEFAULT
            return "\033[0m";
        }
    }

    std::string get_complexity(const benchmark::BigO& complexity)
    {
        switch(complexity)
        {
        case benchmark::oN:
            return "N";
        case benchmark::oNSquared:
            return "N^2";
        case benchmark::oNCubed:
            return "N^3";
        case benchmark::oLogN:
            return "lgN";
        case benchmark::oNLogN:
            return "NlgN";
        case benchmark::o1:
            return "(1)";
        default:
            return "f(N)";
        }
    }

    static std::string FormatString(const char* msg, va_list args)
    {
        // we might need a second shot at this, so pre-emptivly make a copy
        va_list args_cp;
        va_copy(args_cp, args);

        std::size_t size = 256;
        char        local_buff[256];
        auto        ret = vsnprintf(local_buff, size, msg, args_cp);

        if(ret <= 0)
        {
            return {};
        }
        else if(ret < 0)
            if(static_cast<size_t>(ret) < size)
            {
                return local_buff;
            }
        // we did not provide a long enough buffer on our first attempt.
        size = static_cast<size_t>(ret) + 1; // + 1 for the null byte
        std::unique_ptr<char[]> buff(new char[size]);
        ret = vsnprintf(buff.get(), size, msg, args);
        return buff.get();
    }

    static std::string FormatString(const char* msg, ...)
    {
        va_list args;
        va_start(args, msg);
        auto tmp = FormatString(msg, args);
        va_end(args);
        return tmp;
    }

    void PrintColoredString(std::ostream& os, std::string color, std::string str, ...)
    {
        os << color;
        va_list args;
        va_start(args, str);
        os << FormatString(str.data(), args);
        va_end(args);
    }

    static std::string FormatTime(double time)
    {
        // Assuming the time is at max 9.9999e+99 and we have 10 digits for the
        // number, we get 10-1(.)-1(e)-1(sign)-2(exponent) = 5 digits to print.
        if(time > 9999999999 /*max 10 digit number*/)
        {
            return FormatString("%1.4e", time);
        }
        return FormatString("%10.3f", time);
    }

public:
    void PrintHeader(const Run& /*run*/)
    {
        // Assume run.counters (with elements processed and global mem reads/writes)
        // will not be empty
        std::string str  = FormatString("%-*s %13s %15s %12s %16s %12s %14s %14s",
                                       static_cast<int>(name_field_width_),
                                       "Benchmark",
                                       "GPU Time",
                                       "CPU Time",
                                       "Iterations",
                                       "GlobalMem BW",
                                       "BW util",
                                       "GPU Noise",
                                       "Elements/s");
        std::string line = std::string(str.length(), '-');
        GetOutputStream() << line << "\n" << str << "\n" << line << "\n";
    }

    void PrintRunData(const Run& result)
    {
        // Report benchmark name
        auto& sout = GetOutputStream();
        auto  color
            = get_log_color((result.report_big_o || result.report_rms) ? COLOR_BLUE : COLOR_GREEN);

        PrintColoredString(
            sout, color, "%-*s ", name_field_width_, result.benchmark_name().c_str());

        if(benchmark::internal::SkippedWithError == result.skipped)
        {
            PrintColoredString(sout,
                               get_log_color(COLOR_RED),
                               "ERROR OCCURRED: \'%s\'",
                               result.skip_message.c_str());
            PrintColoredString(sout, get_log_color(COLOR_DEFAULT), "\n");
            return;
        }
        else if(benchmark::internal::SkippedWithMessage == result.skipped)
        {
            PrintColoredString(
                sout, get_log_color(COLOR_WHITE), "SKIPPED: \'%s\'", result.skip_message.c_str());
            PrintColoredString(sout, get_log_color(COLOR_DEFAULT), "\n");
            return;
        }

        // Report GPU and CPU time
        const double      gpu_time     = result.GetAdjustedRealTime();
        const double      cpu_time     = result.GetAdjustedCPUTime();
        const std::string gpu_time_str = FormatTime(gpu_time);
        const std::string cpu_time_str = FormatTime(cpu_time);

        if(result.report_big_o)
        {
            std::string big_o = get_complexity(result.complexity);
            PrintColoredString(sout,
                               get_log_color(COLOR_YELLOW),
                               "%10.3f %-4s %10.3f %-4s ",
                               gpu_time,
                               big_o.c_str(),
                               cpu_time,
                               big_o.c_str());
        }
        else if(result.report_rms)
        {
            PrintColoredString(sout,
                               get_log_color(COLOR_YELLOW),
                               "%10.3f %-4s %10.3f %-4s ",
                               gpu_time * 100,
                               "%",
                               cpu_time * 100,
                               "%");
        }
        else if(result.run_type != Run::RT_Aggregate
                || result.aggregate_unit == benchmark::StatisticUnit::kTime)
        {
            const char* timeLabel = GetTimeUnitString(result.time_unit);
            PrintColoredString(sout,
                               get_log_color(COLOR_YELLOW),
                               "%s %-4s %s %-4s ",
                               gpu_time_str.c_str(),
                               timeLabel,
                               cpu_time_str.c_str(),
                               timeLabel);
        }
        else
        {
            assert(result.aggregate_unit == benchmark::StatisticUnit::kPercentage);
            PrintColoredString(sout,
                               get_log_color(COLOR_YELLOW),
                               "%10.3f %-4s %10.3f %-4s ",
                               (100. * result.real_accumulated_time),
                               "%",
                               (100. * result.cpu_accumulated_time),
                               "%");
        }

        // Report iterations
        if(!result.report_big_o && !result.report_rms)
        {
            PrintColoredString(sout, get_log_color(COLOR_CYAN), "%10lld", result.iterations);
        }

        // Report counters
        std::string s;
        const char* unit = "";
        for(auto& c : result.counters)
        {
            std::size_t cNameLen = std::max(std::string::size_type(10), c.first.length());
            if(c.first == "items_per_second")
            {
                // Print elements processed in G/s
                if(result.run_type == Run::RT_Aggregate
                   && result.aggregate_unit == benchmark::StatisticUnit::kPercentage)
                {
                    s    = FormatString("%.3f", 100. * c.second.value);
                    unit = "%";
                }
                else
                {
                    s    = FormatString("%.3f", c.second.value / 1e9);
                    unit = "G";
                }
            }
            else if(c.first == "bytes_per_second")
            {
                // Print GlobalMem BW in GB/s
                if(result.run_type == Run::RT_Aggregate
                   && result.aggregate_unit == benchmark::StatisticUnit::kPercentage)
                {
                    s    = FormatString("%.3f", 100. * c.second.value);
                    unit = "%";
                }
                else
                {
                    s = FormatString("%.3f", c.second.value / 1e9);
                    if(c.second.flags & benchmark::Counter::kIsRate)
                        unit = "GB/s";
                }
                PrintColoredString(sout,
                                   get_log_color(COLOR_DEFAULT),
                                   " %*s%s",
                                   cNameLen - strlen(unit),
                                   s.c_str(),
                                   unit);

                // Print BW util
                std::map<std::string, std::string>* global_context
                    = benchmark::internal::GetGlobalContext();
                if(global_context != nullptr)
                {
                    s = FormatString("%.2f", c.second.value);
                    for(const auto& keyval : *global_context)
                    {
                        if(keyval.first == "hdp_peak_global_mem_bus_bandwidth")
                        {
                            const double global_mem_bw  = std::stod(s);
                            const double peak_global_bw = std::stod(keyval.second);
                            s        = FormatString("%.2f", 100. * global_mem_bw / peak_global_bw);
                            unit     = "%";
                            cNameLen = std::max(std::string::size_type(12), s.length());
                        }
                    }
                }
            }
            else if(c.first == "gpu_noise")
            {
                if(result.run_type == Run::RT_Aggregate
                   && result.aggregate_unit == benchmark::StatisticUnit::kPercentage)
                {
                    s = FormatString("%.2f", 100. * c.second.value);
                }
                else
                {
                    s = FormatString("%.2f", c.second.value);
                }
                unit = "%";
            }
            else
            {
                if(result.run_type == Run::RT_Aggregate
                   && result.aggregate_unit == benchmark::StatisticUnit::kPercentage)
                {
                    s    = FormatString("%.2f", 100. * c.second.value);
                    unit = "%";
                }
                else
                {
                    s = FormatString("%.2f", c.second.value / 1000.);
                    if(c.second.flags & benchmark::Counter::kIsRate)
                        unit = (c.second.flags & benchmark::Counter::kInvert) ? "s" : "/s";
                }
            }

            PrintColoredString(sout,
                               get_log_color(COLOR_DEFAULT),
                               " %*s%s",
                               cNameLen - strlen(unit),
                               s.c_str(),
                               unit);
        }

        // Print labels, that is, GPU noise
        if(!result.report_label.empty())
        {
            std::string s    = FormatString("%.2f", 100. * std::stod(result.report_label));
            const char* unit = "%";
            PrintColoredString(sout, get_log_color(COLOR_DEFAULT), " %13s%s", s.c_str(), unit);
        }

        PrintColoredString(sout, get_log_color(COLOR_DEFAULT), "\n");
    }

    // Called once for each group of benchmark runs, gives information about
    // cpu-time, gpu-time, elements processed, global memory bw and the %
    // the latter represents from the peak global bandwidth during the
    // benchmark run. If the group of runs contained more than two entries
    // then 'report' contains additional elements representing the mean,
    // standard deviation and coefficient of variation of those runs.
    // Additionally if this group of runs was the last in a family of
    // benchmarks, 'reports' contains additional entries representing the
    // asymptotic complexity and RMS of that benchmark family.
    void ReportRuns(const std::vector<Run>& reports)
    {
        for(const auto& run : reports)
        {
            // Print the header if none was printed yet
            bool print_header = !printed_header_;
            if(print_header)
            {
                printed_header_ = true;
                PrintHeader(run);
            }
            PrintRunData(run);
        }
    }
};
} // namespace bench_utils
#endif // ROCTHRUST_BENCHMARKS_BENCH_UTILS_CUSTOM_REPORTER_HPP_
