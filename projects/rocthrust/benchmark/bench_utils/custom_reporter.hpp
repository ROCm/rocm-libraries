// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

namespace bench_utils
{

template <class DstType, class SrcType>
bool IsType(const SrcType* src)
{
  // Check if the src can be casted to the DstType
  return dynamic_cast<const DstType*>(src) != nullptr;
}

static std::string FormatString(const char* msg, va_list args)
{
  // we might need a second shot at this, so pre-emptivly make a copy
  va_list args_cp;
  va_copy(args_cp, args);

  std::size_t size = 256;
  char local_buff[256];
  auto ret = vsnprintf(local_buff, size, msg, args_cp);

  if (ret <= 0)
  {
    return {};
  }
  else if (static_cast<size_t>(ret) < size)
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

std::string get_complexity(const benchmark::BigO& complexity)
{
  switch (complexity)
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

template <class Run>
double calculate_bw_utils(const Run& result)
{
  // Calculates bandwith utilization in %
  std::map<std::string, std::string>* global_context = benchmark::internal::GetGlobalContext();
  if (global_context != nullptr)
  {
    for (const auto& keyval : *global_context)
    {
      if (keyval.first == "hdp_peak_global_mem_bus_bandwidth")
      {
        const double global_mem_bw  = result.counters.at("bytes_per_second").value;
        const double peak_global_bw = std::stod(keyval.second);
        const double bw_util        = 100. * global_mem_bw / peak_global_bw;
        return bw_util;
      }
    }
  }
  return -1;
}

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
class CustomConsoleReporter : public benchmark::ConsoleReporter
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
    switch (color)
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
    if (time > 9999999999 /*max 10 digit number*/)
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
    std::string str = FormatString(
      "%-*s %13s %15s %12s %16s %12s %14s %14s",
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
    auto color = get_log_color((result.report_big_o || result.report_rms) ? COLOR_BLUE : COLOR_GREEN);

    PrintColoredString(sout, color, "%-*s ", name_field_width_, result.benchmark_name().c_str());

    if (benchmark::internal::SkippedWithError == result.skipped)
    {
      PrintColoredString(sout, get_log_color(COLOR_RED), "ERROR OCCURRED: \'%s\'", result.skip_message.c_str());
      PrintColoredString(sout, get_log_color(COLOR_DEFAULT), "\n");
      return;
    }
    else if (benchmark::internal::SkippedWithMessage == result.skipped)
    {
      PrintColoredString(sout, get_log_color(COLOR_WHITE), "SKIPPED: \'%s\'", result.skip_message.c_str());
      PrintColoredString(sout, get_log_color(COLOR_DEFAULT), "\n");
      return;
    }

    // Report GPU and CPU time
    const double gpu_time          = result.GetAdjustedRealTime();
    const double cpu_time          = result.GetAdjustedCPUTime();
    const std::string gpu_time_str = FormatTime(gpu_time);
    const std::string cpu_time_str = FormatTime(cpu_time);

    if (result.report_big_o)
    {
      std::string big_o = get_complexity(result.complexity);
      PrintColoredString(
        sout, get_log_color(COLOR_YELLOW), "%10.3f %-4s %10.3f %-4s ", gpu_time, big_o.c_str(), cpu_time, big_o.c_str());
    }
    else if (result.report_rms)
    {
      PrintColoredString(
        sout, get_log_color(COLOR_YELLOW), "%10.3f %-4s %10.3f %-4s ", gpu_time * 100, "%", cpu_time * 100, "%");
    }
    else if (result.run_type != Run::RT_Aggregate || result.aggregate_unit == benchmark::StatisticUnit::kTime)
    {
      const char* timeLabel = GetTimeUnitString(result.time_unit);
      PrintColoredString(
        sout,
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
      PrintColoredString(
        sout,
        get_log_color(COLOR_YELLOW),
        "%10.3f %-4s %10.3f %-4s ",
        (100. * result.real_accumulated_time),
        "%",
        (100. * result.cpu_accumulated_time),
        "%");
    }

    // Report iterations
    if (!result.report_big_o && !result.report_rms)
    {
      PrintColoredString(sout, get_log_color(COLOR_CYAN), "%10lld", result.iterations);
    }

    // Report counters
    std::string s;
    const char* unit = "";
    for (auto& c : result.counters)
    {
      std::size_t cNameLen = std::max(std::string::size_type(10), c.first.length());
      if (c.first == "items_per_second")
      {
        // Print elements processed in G/s
        if (result.run_type == Run::RT_Aggregate && result.aggregate_unit == benchmark::StatisticUnit::kPercentage)
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
      else if (c.first == "bytes_per_second")
      {
        // Print GlobalMem BW in GB/s
        if (result.run_type == Run::RT_Aggregate && result.aggregate_unit == benchmark::StatisticUnit::kPercentage)
        {
          s    = FormatString("%.3f", 100. * c.second.value);
          unit = "%";
        }
        else
        {
          s = FormatString("%.3f", c.second.value / 1e9);
          if (c.second.flags & benchmark::Counter::kIsRate)
          {
            unit = "GB/s";
          }
        }
        PrintColoredString(sout, get_log_color(COLOR_DEFAULT), " %*s%s", cNameLen - strlen(unit), s.c_str(), unit);

        // Print BW util
        s                    = FormatString("%.2f", c.second.value);
        const double bw_util = calculate_bw_utils(result);
        if (bw_util >= 0)
        {
          s        = FormatString("%.2f", bw_util);
          unit     = "%";
          cNameLen = std::max(std::string::size_type(12), s.length());
        }
      }
      else if (c.first == "gpu_noise")
      {
        if (result.run_type == Run::RT_Aggregate && result.aggregate_unit == benchmark::StatisticUnit::kPercentage)
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
        if (result.run_type == Run::RT_Aggregate && result.aggregate_unit == benchmark::StatisticUnit::kPercentage)
        {
          s    = FormatString("%.2f", 100. * c.second.value);
          unit = "%";
        }
        else
        {
          s = FormatString("%.2f", c.second.value / 1000.);
          if (c.second.flags & benchmark::Counter::kIsRate)
          {
            unit = (c.second.flags & benchmark::Counter::kInvert) ? "s" : "/s";
          }
        }
      }

      PrintColoredString(sout, get_log_color(COLOR_DEFAULT), " %*s%s", cNameLen - strlen(unit), s.c_str(), unit);
    }

    // Print labels, that is, GPU noise
    if (!result.report_label.empty())
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
    for (const auto& run : reports)
    {
      // Print the header if none was printed yet
      bool print_header = !printed_header_;
      if (print_header)
      {
        printed_header_ = true;
        PrintHeader(run);
      }
      PrintRunData(run);
    }
  }
};

class CustomJSONReporter : public benchmark::JSONReporter
{
private:
  bool first_report_ = true;

  std::string StrEscape(const std::string& s)
  {
    std::string tmp;
    tmp.reserve(s.size());
    for (char c : s)
    {
      switch (c)
      {
        case '\b':
          tmp += "\\b";
          break;
        case '\f':
          tmp += "\\f";
          break;
        case '\n':
          tmp += "\\n";
          break;
        case '\r':
          tmp += "\\r";
          break;
        case '\t':
          tmp += "\\t";
          break;
        case '\\':
          tmp += "\\\\";
          break;
        case '"':
          tmp += "\\\"";
          break;
        default:
          tmp += c;
          break;
      }
    }
    return tmp;
  }

  std::string FormatKV(std::string const& key, std::string const& value)
  {
    return FormatString("\"%s\": \"%s\"", StrEscape(key).c_str(), StrEscape(value).c_str());
  }

  std::string FormatKV(std::string const& key, const char* value)
  {
    return FormatString("\"%s\": \"%s\"", StrEscape(key).c_str(), StrEscape(value).c_str());
  }

  std::string FormatKV(std::string const& key, bool value)
  {
    return FormatString("\"%s\": %s", StrEscape(key).c_str(), value ? "true" : "false");
  }

  std::string FormatKV(std::string const& key, int64_t value)
  {
    std::stringstream ss;
    ss << '"' << StrEscape(key) << "\": " << value;
    return ss.str();
  }

  std::string FormatKV(std::string const& key, double value)
  {
    std::stringstream ss;
    ss << '"' << StrEscape(key) << "\": ";

    if (std::isnan(value))
    {
      ss << (value < 0 ? "-" : "") << "NaN";
    }
    else if (std::isinf(value))
    {
      ss << (value < 0 ? "-" : "") << "Infinity";
    }
    else
    {
      const auto max_digits10            = std::numeric_limits<decltype(value)>::max_digits10;
      const auto max_fractional_digits10 = max_digits10 - 1;
      ss << std::scientific << std::setprecision(max_fractional_digits10) << value;
    }
    return ss.str();
  }

public:
  void ReportRuns(std::vector<Run> const& reports)
  {
    if (reports.empty())
    {
      return;
    }
    std::string indent(4, ' ');
    std::ostream& out = GetOutputStream();
    if (!first_report_)
    {
      out << ",\n";
    }
    first_report_ = false;

    for (auto it = reports.begin(); it != reports.end(); ++it)
    {
      out << indent << "{\n";
      PrintRunData(*it);
      out << indent << '}';
      auto it_cp = it;
      if (++it_cp != reports.end())
      {
        out << ",\n";
      }
    }
  }

  void PrintRunData(Run const& run)
  {
    std::string indent(6, ' ');
    std::ostream& out = GetOutputStream();

    auto output_format = [this, &out, &indent](const std::string& label, auto val, bool start_endl = true) {
      if (start_endl)
      {
        out << ",\n";
      }
      out << indent << FormatKV(label, val);
    };

    output_format("name", run.benchmark_name(), false);
    output_format("family_index", run.family_index);
    output_format("per_family_instance_index", run.per_family_instance_index);
    output_format("run_name", run.run_name.str());
    output_format("run_type", [&run]() -> const char* {
      switch (run.run_type)
      {
        case BenchmarkReporter::Run::RT_Iteration:
          return "iteration";
        case BenchmarkReporter::Run::RT_Aggregate:
          return "aggregate";
      }
      BENCHMARK_UNREACHABLE();
    }());
    output_format("repetitions", run.repetitions);
    if (run.run_type != BenchmarkReporter::Run::RT_Aggregate)
    {
      output_format("repetition_index", run.repetition_index);
    }
    output_format("threads", run.threads);
    if (run.run_type == BenchmarkReporter::Run::RT_Aggregate)
    {
      output_format("aggregate_name", run.aggregate_name);
      output_format("aggregate_unit", [&run]() -> const char* {
        switch (run.aggregate_unit)
        {
          case benchmark::StatisticUnit::kTime:
            return "time";
          case benchmark::StatisticUnit::kPercentage:
            return "percentage";
        }
        BENCHMARK_UNREACHABLE();
      }());
    }
    if (benchmark::internal::SkippedWithError == run.skipped)
    {
      output_format("error_occurred", true);
      output_format("error_message", run.skip_message);
    }
    else if (benchmark::internal::SkippedWithMessage == run.skipped)
    {
      output_format("skipped", true);
      output_format("skip_message", run.skip_message);
    }
    if (!run.report_big_o && !run.report_rms)
    {
      output_format("iterations", run.iterations);
      if (run.run_type != Run::RT_Aggregate || run.aggregate_unit == benchmark::StatisticUnit::kTime)
      {
        output_format("gpu_time", run.GetAdjustedRealTime());
        output_format("cpu_time", run.GetAdjustedCPUTime());
      }
      else
      {
        assert(run.aggregate_unit == benchmark::StatisticUnit::kPercentage);
        output_format("gpu_time", run.real_accumulated_time);
        output_format("cpu_time", run.cpu_accumulated_time);
      }
      output_format("time_unit", GetTimeUnitString(run.time_unit));
    }
    else if (run.report_big_o)
    {
      output_format("cpu_coefficient", run.GetAdjustedCPUTime());
      output_format("gpu_coefficient", run.GetAdjustedRealTime());
      output_format("big_o", get_complexity(run.complexity));
      output_format("time_unit", GetTimeUnitString(run.time_unit));
    }
    else if (run.report_rms)
    {
      output_format("rms", run.GetAdjustedCPUTime());
    }

    for (auto& c : run.counters)
    {
      if (c.first == "items_per_second")
      {
        // Report same name as console reporter
        output_format("elements_per_second", c.second);
      }
      else if (c.first == "bytes_per_second")
      {
        // Report same name as console reporter
        output_format("global_mem_bw", c.second);
        const double util_bw = calculate_bw_utils(run);
        if (util_bw >= 0)
        {
          output_format("util_bw", util_bw);
        }
      }
      else
      {
        output_format(c.first, c.second);
      }
    }

    if (run.memory_result)
    {
      const benchmark::MemoryManager::Result memory_result = *run.memory_result;
      output_format("allocs_per_iter", run.allocs_per_iter);
      output_format("max_bytes_used", memory_result.max_bytes_used);

      if (memory_result.total_allocated_bytes != benchmark::MemoryManager::TombstoneValue)
      {
        output_format("total_allocated_bytes", memory_result.total_allocated_bytes);
      }

      if (memory_result.net_heap_growth != benchmark::MemoryManager::TombstoneValue)
      {
        output_format("net_heap_growth", memory_result.net_heap_growth);
      }
    }

    if (!run.report_label.empty())
    {
      output_format("label", run.report_label);
    }
    out << '\n';
  }
};

BENCHMARK_DISABLE_DEPRECATED_WARNING

class CustomCSVReporter : public benchmark::CSVReporter
{
private:
  bool printed_header_ = false;

  std::vector<std::string> elements = {
    "name",
    "iterations",
    "gpu_time",
    "cpu_time",
    "time_unit",
    "global_mem_bw",
    "util_bw",
    "elements_per_second",
    "gpu_noise",
    "label",
    "error_occurred",
    "error_message"};

  std::string CsvEscape(const std::string& s)
  {
    std::string tmp;
    tmp.reserve(s.size() + 2);
    for (char c : s)
    {
      switch (c)
      {
        case '"':
          tmp += "\"\"";
          break;
        default:
          tmp += c;
          break;
      }
    }
    return '"' + tmp + '"';
  }

public:
  void PrintHeader(const Run& /*run*/)
  {
    std::string str = "";
    bool first      = true;
    for (auto element : elements)
    {
      if (first)
      {
        first = false;
      }
      else
      {
        str += ",";
      }
      str += element;
    }
    GetOutputStream() << str << "\n";
  }

  void PrintRunData(const Run& result)
  {
    // Report benchmark name
    auto& sout = GetOutputStream();

    if (result.skipped)
    {
      sout << std::string(elements.size() - 3, ',');
      sout << std::boolalpha << (benchmark::internal::SkippedWithError == result.skipped) << ",";
      sout << CsvEscape(result.skip_message) << "\n";
      return;
    }

    sout << CsvEscape(result.benchmark_name()) << ",";

    if (!result.report_big_o && !result.report_rms)
    {
      sout << result.iterations;
    }
    sout << ",";

    sout << result.GetAdjustedRealTime() << ",";
    sout << result.GetAdjustedCPUTime() << ",";

    if (result.report_big_o)
    {
      sout << get_complexity(result.complexity);
    }
    else if (!result.report_rms)
    {
      sout << benchmark::GetTimeUnitString(result.time_unit);
    }
    sout << ",";

    if (result.counters.find("bytes_per_second") != result.counters.end())
    {
      sout << result.counters.at("bytes_per_second");
    }
    sout << ",";

    if (result.counters.find("bytes_per_second") != result.counters.end())
    {
      const double bw_util = calculate_bw_utils(result);
      if (bw_util >= 0)
      {
        sout << bw_util;
      }
    }
    sout << ",";

    if (result.counters.find("items_per_second") != result.counters.end())
    {
      sout << result.counters.at("items_per_second");
    }
    sout << ",";

    if (result.counters.find("gpu_noise") != result.counters.end())
    {
      sout << result.counters.at("gpu_noise");
    }
    sout << ",";

    if (!result.report_label.empty())
    {
      sout << CsvEscape(result.report_label);
    }

    sout << ",,\n";
  }

  void ReportRuns(const std::vector<Run>& reports)
  {
    for (const auto& run : reports)
    {
      // Print the header if none was printed yet
      bool print_header = !printed_header_;
      if (print_header)
      {
        printed_header_ = true;
        PrintHeader(run);
      }
      PrintRunData(run);
    }
  }
};

benchmark::BenchmarkReporter* ChooseCustomReporter()
{
  // benchmark::BenchmarkReporter is polymorphic as it has a virtual
  // function which allows us to use dynamic_cast to detect the derived type.
  using PtrType                    = benchmark::BenchmarkReporter*;
  PtrType default_display_reporter = benchmark::CreateDefaultDisplayReporter();

  if (IsType<benchmark::CSVReporter>(default_display_reporter))
  {
    return PtrType(new CustomCSVReporter);
  }
  else if (IsType<benchmark::JSONReporter>(default_display_reporter))
  {
    return PtrType(new CustomJSONReporter);
  }
  else if (IsType<benchmark::ConsoleReporter>(default_display_reporter))
  {
    return PtrType(new CustomConsoleReporter);
  }

  return nullptr;
}

BENCHMARK_RESTORE_DEPRECATED_WARNING

} // namespace bench_utils
#endif // ROCTHRUST_BENCHMARKS_BENCH_UTILS_CUSTOM_REPORTER_HPP_
