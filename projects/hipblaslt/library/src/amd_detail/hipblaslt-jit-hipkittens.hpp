// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit.hpp"
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Not installed. A backend that compiles HipKittens kernel templates with comgr,
// in process. Each kernel variant serves a narrow set of problems; others are
// NOT_SUPPORTED. The HipKittens headers it includes are installed with the
// library under hipblaslt/hipkittens/<commit>.
namespace hipblaslt_ext::experimental::jit::hipkittens
{
    struct Options
    {
        // The header directory; empty uses HIPBLASLT_JIT_HIPKITTENS_PATH, then
        // hipblaslt/hipkittens/<commit> next to libhipblaslt.
        std::string headers;
    };

    // Reads and verifies the headers. When they are missing or do not match
    // their manifest, fails with "JIT backend HipKittens not available: ...".
    HIPBLASLT_EXPORT hipblasStatus_t createBackend(const Options& options,
                                                   Backend&       backend,
                                                   Diagnostics&   diagnostics);

    namespace detail
    {
        struct HeaderFile
        {
            std::string_view path; // relative to the header directory
            uint64_t         size;
            std::string_view sha256;
        };

        // What the built kernel uses.
        struct KernelResources
        {
            uint32_t kernargBytes, ldsBytes, vgprs, vgprSpills;
        };

        struct Variant
        {
            std::string_view         name;
            std::string_view         isa;
            std::string_view         kernelName;
            std::string_view         source; // HIP
            std::string_view         entry; // the one-solution library skeleton
            std::vector<std::string> hipFlags;
            KernelResources          resources;
        };

        // Written at build time by hipkittens/make_entries.py.
        struct Resources
        {
            std::string_view        manifest; // the header directory's manifest.json
            std::vector<HeaderFile> headers;
            std::vector<Variant>    variants;
        };
        HIPBLASLT_EXPORT const Resources& resources();
    }
}
