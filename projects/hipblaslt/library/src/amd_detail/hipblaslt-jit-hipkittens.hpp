// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-code-object.hpp"
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// Not installed. A backend that compiles HipKittens kernel templates with comgr,
// in process. Each kernel variant serves a narrow set of problems; others are
// NOT_SUPPORTED. The HipKittens headers the templates include are installed
// with the library under hipblaslt/hipkittens/<commit>.
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
        // One installed header: its path in the header directory, its size and
        // the SHA-256 the runtime checks. The bytes the compiler includes are a
        // BuildUnit, read from that path.
        struct HeaderFile
        {
            std::filesystem::path path;
            uint64_t              size = 0;
            std::string           sha256;
        };

        // What the built kernel uses.
        struct KernelResources
        {
            uint32_t kernargBytes, ldsBytes, vgprs, vgprSpills;
        };

        struct Variant
        {
            std::string                        name;
            hipblaslt_jit::code_object::Target isa;
            std::string                        kernelName;
            hipblaslt_jit::BuildUnit           source; // HIP
            std::vector<uint8_t>               entry; // the one-solution library skeleton
            std::vector<std::string>           hipFlags;
            KernelResources                    resources;
        };

        // Written at build time by hipkittens/write_entries.cpp. manifest is the
        // MessagePack record installed next to the headers.
        struct Resources
        {
            std::vector<uint8_t>    manifest;
            std::vector<HeaderFile> headers;
            std::vector<Variant>    variants;
        };
        HIPBLASLT_EXPORT const Resources& resources();
    }
}
