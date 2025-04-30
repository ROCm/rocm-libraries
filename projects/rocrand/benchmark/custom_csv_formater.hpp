// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <benchmark/benchmark.h>
#include <vector>
namespace benchmark
{

class customCSVReporter : public BenchmarkReporter
{
public:
    customCSVReporter() : printed_header_(false) {}
    bool ReportContext(const Context& context) override;
    void ReportRuns(const std::vector<Run>& reports) override;

private:
    std::string CsvEscape(const std::string& s)
    {
        std::string tmp;
        tmp.reserve(s.size() + 2);
        for(char c : s)
        {
            switch(c)
            {
                case '"': tmp += "\"\""; break;
                default: tmp += c; break;
            }
        }
        return '"' + tmp + '"';
    }

    // Function to return an string for the calculated complexity
    std::string GetBigOString(const BigO complexity)
    {
        switch(complexity)
        {
            case oN: return "N";
            case oNSquared: return "N^2";
            case oNCubed: return "N^3";
            case oLogN: return "lgN";
            case oNLogN: return "NlgN";
            case o1: return "(1)";
            default: return "f(N)";
        }
    }

    void                  PrintRunData(const Run& report);
    bool                  printed_header_;
    std::set<std::string> user_counter_names_;

    std::ostream* nullLog = nullptr;

    std::array<std::string, 15> elements = {"engine",
                                            "distribution",
                                            "mode",
                                            "name",
                                            "iterations",
                                            "real_time",
                                            "cpu_time",
                                            "time_unit",
                                            "bytes_per_second",
                                            "throughput_gigabytes_per_second",
                                            "lambda",
                                            "items_per_second",
                                            "label",
                                            "error_occurred",
                                            "error_message"};
};

inline bool customCSVReporter::ReportContext(const Context& context)
{
    PrintBasicContext(&GetErrorStream(), context);
    return true;
}

inline void customCSVReporter::ReportRuns(const std::vector<Run>& reports)
{
    std::ostream& Out = GetOutputStream();

    if(!printed_header_)
    {
        // save the names of all the user counters
        for(const auto& run : reports)
        {
            for(const auto& cnt : run.counters)
            {
                if(cnt.first == "bytes_per_second" || cnt.first == "items_per_second")
                    continue;
                user_counter_names_.insert(cnt.first);
            }
        }

        // print the header
        for(auto B = elements.begin(); B != elements.end();)
        {
            Out << *B++;
            if(B != elements.end())
                Out << ",";
        }
        for(auto B = user_counter_names_.begin(); B != user_counter_names_.end();)
        {
            Out << ",\"" << *B++ << "\"";
        }
        Out << "\n";

        printed_header_ = true;
    }
    else
    {
        // check that all the current counters are saved in the name set
        for(const auto& run : reports)
        {
            for(const auto& cnt : run.counters)
            {
                if(cnt.first == "bytes_per_second" || cnt.first == "items_per_second")
                    continue;

                // benchmark::internal::GetNullLogInstance()
                *nullLog << "All counters must be present in each run. "
                         << "Counter named \"" << cnt.first
                         << "\" was not in a run after being added to the header";
            }
        }
    }

    // print results for each run
    for(const auto& run : reports)
    {
        PrintRunData(run);
    }
}

inline void customCSVReporter::PrintRunData(const Run& run)
{
    std::ostream& Out = GetOutputStream();
    std::ostream& Err = GetErrorStream();

    //get the name of the engine and distribution:
    std::string temp = run.benchmark_name();
    std::string deviceName = std::string(temp.begin(), temp.begin() + temp.find("<"));
    temp.erase(0, temp.find("<") + 1);
    std::string engineName = std::string(temp.begin(), temp.begin() + temp.find(","));
    temp.erase(0, engineName.size() + 1);
    std::string mode = "default";

    if(deviceName != "device_kernel")
    {
        mode = std::string(temp.begin(), temp.begin() + temp.find(','));
        temp.erase(0, temp.find(",") + 1);
    }
    std::string disName = std::string(temp.begin(), temp.begin() + temp.find(">"));
    std::string lambda = "";

    size_t ePos = disName.find("=");
    if(ePos <= disName.size())
    {
        lambda = std::string(disName.begin() + (ePos + 1), disName.end() - 1);
        disName.erase(disName.begin() + disName.find("("), disName.end());
    }

    Out << engineName << "," << disName << "," << mode << ",";
    Out << CsvEscape(run.benchmark_name()) << ",";
    if(run.skipped)
    {
        Err << std::string(elements.size() - 3, ',');
        Err << "true,";
        Err << CsvEscape(run.skip_message) << "\n";
        return;
    }

    // Do not print iteration on bigO and RMS report
    if(!run.report_big_o && !run.report_rms)
    {
        Out << run.iterations;
    }
    Out << ",";

    Out << run.GetAdjustedRealTime() << ",";
    Out << run.GetAdjustedCPUTime() << ",";

    // Do not print timeLabel on bigO and RMS report
    if(run.report_big_o)
    {
        Out << GetBigOString(run.complexity);
    }
    else if(!run.report_rms)
    {
        Out << GetTimeUnitString(run.time_unit);
    }
    Out << ",";

    if(run.counters.find("bytes_per_second") != run.counters.end())
    {
        Out << run.counters.at("bytes_per_second");
    }
    Out << ",";

    double gbps = run.counters.at("bytes_per_second") / std::pow(1024, 3);

    Out << gbps << "," << lambda << ",";

    if(run.counters.find("items_per_second") != run.counters.end())
    {
        Out << run.counters.at("items_per_second");
    }
    Out << ",";
    if(!run.report_label.empty())
    {
        Out << CsvEscape(run.report_label);
    }
    Out << ",,"; // for error_occurred and error_message

    // Print user counters
    for(const auto& ucn : user_counter_names_)
    {
        auto it = run.counters.find(ucn);
        if(it == run.counters.end())
        {
            Out << ",";
        }
        else
        {
            Out << "," << it->second;
        }
    }
    Out << '\n';
}

} // namespace benchmark
