// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <string>

#include <miopen/env.hpp>
#include <miopen/kernel_tuning_mode.hpp>

#ifdef __linux__
#include <unistd.h>
#include <sys/utsname.h>
#endif

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_runtime.h>
#endif

namespace RocmPerf {

#define HIP_CHECK(call)                                                         \
    {                                                                           \
        hipError_t err_ = call;                                                 \
        if(err_ != hipSuccess)                                                  \
        {                                                                       \
            std::cerr << "HIP error: " << hipGetErrorString(err_) << std::endl; \
        }                                                                       \
    }

class SysInfo
{
public:
    SysInfo(size_t major, size_t minor, size_t patch)
        : miopMajor(major), miopMinor(minor), miopPatch(patch)
    {
    }

    void ShowSysInfo()
    {
#ifdef __linux__
        // Check if JSON mode is enabled
        const bool json_mode = miopen::IsPerformanceLoggingEnabled();

        // System information collection
        const std::string timestamp = GetTimestamp();
        const std::string hostname  = GetHostname();
        const std::string osInfo    = GetOsInfo();
        const std::string hipVer    = GetHipVersion();
        auto [cpuVendor, cpuModel]  = GetCpuInfo();
        const std::string ramSize   = GetRamSize();
        const std::string gpuInfo   = GetGpuInfo();
        const std::string amdgpuVer = GetAmdGpuVersion();

        // Format final output
        if(json_mode)
        {
            std::cout << "{\"timestamp\":\"" << JsonEscape(timestamp) << "\","
                      << "\"system_info\":{" << "\"hostname\":\"" << JsonEscape(hostname) << "\","
                      << "\"os\":\"" << JsonEscape(osInfo) << "\"," << "\"cpu_vendor\":\""
                      << JsonEscape(cpuVendor) << "\"," << "\"cpu_model\":\""
                      << JsonEscape(cpuModel) << "\"," << "\"ram_size\":\"" << JsonEscape(ramSize)
                      << "\"," << "\"gpu_model\":\"" << JsonEscape(gpuInfo) << "\"},"
                      << "\"build_info\":{" << "\"rocm\":\"" << JsonEscape(hipVer) << "\","
                      << "\"miopen_version\":\"" << miopMajor << "." << miopMinor << "."
                      << miopPatch << "\"," << "\"amdgpu_driver\":\"" << JsonEscape(amdgpuVer)
                      << "\"}}" << std::endl;
        }
        else
        {
            std::cout << "Timestamp: " << timestamp << "; " << "Host Name: " << hostname << "; "
                      << "Operating System: " << osInfo << "; " << "ROCm: " << hipVer << "; "
                      << "MIOpen Driver: " << miopMajor << "." << miopMinor << "." << miopPatch
                      << "; " << "CPU Vendor: " << cpuVendor << "; " << "CPU Model: " << cpuModel
                      << "; " << "RAM Size: " << ramSize << "; " << "GPU Model: " << gpuInfo << "; "
                      << "AMDGPU Driver: " << amdgpuVer << std::endl;
        }
#else
        const bool json_mode = miopen::IsPerformanceLoggingEnabled();
        if(json_mode)
        {
            std::cout << "{\"build_info\":{" << "\"miopen_version\":\"" << miopMajor << "."
                      << miopMinor << "." << miopPatch << "\"}}" << std::endl;
        }
        else
        {
            (void)miopMajor;
            (void)miopMinor;
            (void)miopPatch;
        }
#endif
    }

private:
    std::string JsonEscape(const std::string& str)
    {
        std::ostringstream oss;
        for(char c : str)
        {
            switch(c)
            {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if(c < 32 || c > 126)
                {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c));
                }
                else
                {
                    oss << c;
                }
            }
        }
        return oss.str();
    }

    std::string GetTimestamp()
    {
        std::stringstream ss;
#ifdef __linux__
        auto now   = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);

        ss << std::put_time(std::gmtime(&now_c), "%Y-%m-%d %H:%M:%S UTC");
#endif
        return ss.str();
    }

    std::string GetHostname()
    {
        char name[256] = "";
#ifdef __linux__
        gethostname(name, sizeof(name));
#endif
        return name;
    }

    std::string GetOsInfo()
    {
#ifdef __linux__
        struct utsname buf;
        uname(&buf);
        return std::string(buf.sysname) + " " + buf.release;
#else
        return "unimplemented";
#endif
    }

    std::pair<std::string, std::string> GetCpuInfo()
    {
        std::string line;
        std::string vendor_id, model_name;
        std::set<std::string> physical_ids;
        std::string socket_info;
#ifdef __linux__
        std::ifstream cpuinfo("/proc/cpuinfo");
        while(getline(cpuinfo, line))
        {
            if(line.find("vendor_id") == 0 && vendor_id.empty())
            {
                vendor_id = line.substr(line.find(": ") + 2);
            }
            if(line.find("model name") == 0 && model_name.empty())
            {
                model_name = line.substr(line.find(": ") + 2);
            }
            if(line.find("physical id") == 0)
            {
                physical_ids.insert(line.substr(line.find(": ") + 2));
            }
        }

        if(vendor_id.find("AuthenticAMD") != std::string::npos)
        {
            vendor_id        = "AMD";
            size_t start_pos = model_name.find("AMD ");
            if(start_pos != std::string::npos)
            {
                model_name = model_name.substr(start_pos + 4);
            }
            std::istringstream iss(model_name);
            std::string part1, part2;
            iss >> part1 >> part2;
            model_name = part1 + " " + part2;
        }
        else if(vendor_id.find("GenuineIntel") != std::string::npos)
        {
            vendor_id = "Intel";
        }
        else
        {
            vendor_id = "unknown";
        }

        // Format socket count
        socket_info = physical_ids.empty() ? "" : std::to_string(physical_ids.size()) + " x ";
        return {vendor_id, socket_info + model_name};
#else
        return {"unimplemented", "unimplemented"};
#endif
    }

    std::string GetRamSize()
    {
#ifdef __linux__
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while(getline(meminfo, line))
        {
            if(line.find("MemTotal") == 0)
            {
                size_t start = line.find(":") + 2;
                size_t end   = line.find(" kB");
                long kb      = std::stol(line.substr(start, end - start));
                return std::to_string(kb / (1024 * 1024)) + " GB";
            }
        }
#endif
        return "Unknown";
    }

    std::string GetAmdGpuVersion()
    {
        std::string version = "0.0.0";
#ifdef __linux__
        std::ifstream amdgpuVer("/sys/module/amdgpu/version");
        if(amdgpuVer.is_open())
        {
            std::getline(amdgpuVer, version);
        }
#endif

        return version;
    }

    std::string GetHipVersion()
    {
        int runtime_version = 0;
#ifndef MIOPEN_HIP_RUNTIME_COMPILE
        HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
#endif
        const int patch = runtime_version % 100000;
        runtime_version = runtime_version / 100000;

        const int major = runtime_version / 100;
        const int minor = runtime_version % 100;
        return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    }

    std::string GetGpuInfo()
    {
        std::string result;
        int deviceCount = 0;
#ifndef MIOPEN_HIP_RUNTIME_COMPILE
        HIP_CHECK(hipGetDeviceCount(&deviceCount));
#endif
        if(deviceCount < 1)
        {
            result = "None";
        }
        else
        {
            std::map<std::string, int> gpuList;
#ifndef MIOPEN_HIP_RUNTIME_COMPILE
            for(int i = 0; i < deviceCount; i++)
            {
                hipDeviceProp_t props;
                HIP_CHECK(hipGetDeviceProperties(&props, i));
                gpuList[props.name]++;
            }
#endif
            for(const auto& [name, count] : gpuList)
            {
                if(!result.empty())
                    result += ", ";
                result += std::to_string(count) + " x " + name;
            }
        }

        return result;
    }

private:
    size_t miopMajor{};
    size_t miopMinor{};
    size_t miopPatch{};
};
} // namespace RocmPerf
