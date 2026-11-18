// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GPU_PROBE_ARGS_HPP
#define GPU_PROBE_ARGS_HPP

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>

// Returns the device index, or nullopt for a usage error. Kept free of HIP so it can be tested
// without a GPU.
inline std::optional<int32_t> parseGpuProbeArgs(int argc, char const* const* argv)
{
    int32_t deviceIdx = 0;
    for(int32_t i = 1; i < argc; ++i)
    {
        if(std::strcmp(argv[i], "--device-idx") != 0 || i + 1 >= argc)
            return std::nullopt;
        char const* text  = argv[++i];
        char*       end   = nullptr;
        errno             = 0;
        int64_t     value = std::strtoll(text, &end, 10);
        if(end == text || *end != '\0' || errno == ERANGE || value < 0 || value > INT32_MAX)
            return std::nullopt;
        deviceIdx = static_cast<int32_t>(value);
    }
    return deviceIdx;
}

#endif // GPU_PROBE_ARGS_HPP
