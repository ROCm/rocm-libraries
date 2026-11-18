// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_Logging.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_Logging.h
 * @brief cuDNN FE v0.x logging helpers; the `CUDNN_FE_LOG*` macros come from graph_helpers.h.
 */

#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

#include <hipdnn_compatibility/cudnn/cudnn_backend_base.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_helpers.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/detail/logging_bridge.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_Logging.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,misc-const-correctness,readability-implicit-bool-conversion,readability-avoid-nested-conditional-operator):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

// Same lookup as hipDNN's logging bridge. The returned pointer stays valid until the
// next call on this thread.
inline const char* get_environment(const char* name)
{
    thread_local std::string value;
    value = detail::getCudnnEnv(name);
    return value.empty() ? nullptr : value.c_str();
}

inline int getLogLevel()
{
    static int log_level = detail::cudnnLogLevel();
    return log_level;
}

inline bool& isLoggingEnabled()
{
#ifdef NV_CUDNN_FRONTEND_DISABLE_LOGGING
    static bool log_enabled = false;
#else
    static bool log_enabled = (getLogLevel() > 0);
#endif
    return log_enabled;
}

inline bool& isLoggingTensorDumpEnabled()
{
#ifdef NV_CUDNN_FRONTEND_DISABLE_LOGGING
    static bool tensor_dump_enabled = false;
#else
    static bool tensor_dump_enabled = []() {
        int level = getLogLevel();
        return level >= 1 && level < 10;
    }();
#endif
    return tensor_dump_enabled;
}

inline std::ostream& getStream()
{
    static std::ofstream outFile;
    static std::ostream& stream
        = get_environment("CUDNN_FRONTEND_LOG_FILE")
              ? (std::strncmp(get_environment("CUDNN_FRONTEND_LOG_FILE"), "stdout", 6) == 0
                     ? std::cout
                     : (std::strncmp(get_environment("CUDNN_FRONTEND_LOG_FILE"), "stderr", 6) == 0
                            ? std::cerr
                            : (outFile.open(get_environment("CUDNN_FRONTEND_LOG_FILE"),
                                            std::ios::out),
                               outFile)))
              : (isLoggingEnabled() = false, std::cout);
    return stream;
}

class ConditionalStreamer
{
private:
    std::ostream& stream;

public:
    ConditionalStreamer(std::ostream& stream_)
        : stream(stream_)
    {
    }

    template <typename T>
    const ConditionalStreamer& operator<<(const T& t) const
    {
        if(isLoggingEnabled())
        {
            stream << t;
        }
        return *this;
    }

    const ConditionalStreamer& operator<<(std::ostream& (*spl)(std::ostream&)) const
    {
        if(isLoggingEnabled())
        {
            stream << spl;
        }
        return *this;
    }
};

inline ConditionalStreamer& getLogger()
{
    static ConditionalStreamer opt(getStream());
    return opt;
}

inline std::ostream& operator<<(std::ostream& os, const BackendDescriptor& desc)
{
    if(isLoggingEnabled())
    {
        os << desc.describe();
    }
    return os;
}

// NOLINTEND(readability-identifier-naming,misc-const-correctness,readability-implicit-bool-conversion,readability-avoid-nested-conditional-operator)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
