// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iomanip>

enum class Metric
{
    LATENCY   = 0,
    TFLOPS    = 1,
    BANDWIDTH = 2
};

inline constexpr auto get_metric_name(Metric m)
{
    switch(m)
    {
    case Metric::LATENCY: return "latency";
    case Metric::TFLOPS: return "tflops";
    case Metric::BANDWIDTH: return "bandwidth";
    default: throw std::invalid_argument("Unsupported metric type");
    }
}

struct PerformanceResult
{
    double latency_;
    double tflops_;
    double bandwidth_;

    static bool compare(const PerformanceResult& a, const PerformanceResult& b, Metric m)
    {
        switch(m)
        {
        case Metric::LATENCY: return a.latency_ < b.latency_;
        case Metric::TFLOPS: return a.tflops_ > b.tflops_;
        case Metric::BANDWIDTH: return a.bandwidth_ > b.bandwidth_;
        default: throw std::invalid_argument("Unsupported metric type");
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const PerformanceResult& result)
    {
        os << "{\n"
           << "   \"latency(ms)\": " << std::fixed << std::setprecision(2) << result.latency_
           << ",\n"
           << "   \"tflops(TFlops)\": " << result.tflops_ << ",\n"
           << "   \"bandwidth(GB/s)\": " << result.bandwidth_ << "\n"
           << "}";
        return os;
    }
};

inline std::string get_rocm_version()
{
    std::ifstream version_file("/opt/rocm/.info/version");
    if(version_file.is_open())
    {
        std::string version;
        std::getline(version_file, version);
        return version;
    }
    return "Unknown";
}
