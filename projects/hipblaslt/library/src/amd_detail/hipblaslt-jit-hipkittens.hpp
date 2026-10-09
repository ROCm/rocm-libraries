// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-code-object.hpp"
#include <Tensile/ContractionSolution.hpp>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace hipblaslt_jit
{
    class Jit;
}

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
    HIPBLASLT_EXPORT hipblasStatus_t
        createBackend(const Options&                              options,
                      std::shared_ptr<const hipblaslt_jit::Jit>& jit,
                      Diagnostics&                                diagnostics);

    namespace detail
    {
        // One installed header: its path in the header directory, its size and
        // the SHA-256 the runtime checks. The bytes the compiler includes are an
        // Artifact, read from that path.
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

        // One HipKittens kernel: the HIP main kernel, the flags it compiles
        // with, and the TN GEMM it implements for one processor, with A, B, C,
        // D and E of one data type.
        struct Kernel : hipblaslt_jit::Artifact
        {
            hipblaslt_jit::Processor processor;
            std::string              kernelName;
            rocisa::DataType         type;
            std::vector<std::string> hipFlags;
            KernelResources          resources;

            // The solution an executable built from this kernel runs. Its
            // WorkgroupMappingXCCCheck predicate reads the current HIP device.
            HIPBLASLT_EXPORT std::shared_ptr<TensileLite::ContractionSolution> solution() const;
        };
        HIPBLASLT_EXPORT const std::vector<Kernel>& kernels();

        // Written at build time by hipkittens/write_resources.cpp. manifest is
        // the MessagePack record installed next to the headers, and source the
        // HIP template every kernel compiles.
        struct Resources
        {
            std::vector<uint8_t>    manifest;
            std::vector<HeaderFile> headers;
            std::vector<uint8_t>    source;
        };
        HIPBLASLT_EXPORT const Resources& resources();
    }
}
