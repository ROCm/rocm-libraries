// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Contractions.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>

namespace hipblaslt_jit
{
    using TensileLibrary = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;

    // Loads a GEMM library with TensileLite. Throws unless it holds local
    // solutions 0 to N-1. A library holds many solutions; this does not require
    // exactly one.
    std::shared_ptr<TensileLibrary> loadGemmLibrary(const std::vector<uint8_t>& entry);

    // A source bundle and the name of its first solution's kernel. The
    // TensileLite library is not returned; a caller that checks predicates
    // loads it with loadGemmLibrary.
    struct SourceBundleRead
    {
        SourceCodeBundle source;
        std::string      kernelName;
    };

    // Reads a TensileLite source bundle by directory convention: one main
    // kernel, HIP helpers, headers, and the kernel name from the library entry.
    SourceBundleRead readTensileSourceBundle(const std::filesystem::path& bundle);

    // The ContractionSolution a backend contributed. This is not a
    // MasterSolutionLibrary, Hardware, or SolutionAdapter. After publish,
    // hipBLASLt launches that solution through JitLibrary, the same way it
    // launches Equality and Origami solutions. An in-process handle for one
    // index refers to the solution inside JitLibrary.
    struct TensileGemmBundle final : KernelBundle
    {
        using Diagnostics = hipblaslt_ext::experimental::jit::Diagnostics;

        std::shared_ptr<TensileLite::ContractionSolution> solution;
        int32_t                                           libraryIndex = 0; // inside JitLibrary
        int                                               index        = 0; // local, in the record

        int solutionIndex() const noexcept override
        {
            return index;
        }
        std::string name() const override
        {
            return solution ? solution->solutionName : std::string();
        }
        std::string kernelNames() const override
        {
            return solution ? solution->kernelName : std::string();
        }
        hipblasStatus_t support(const GemmRequest&, size_t, size_t&, Diagnostics&) const override;
    };

    // Loads solution records from any backend into TensileGemmBundles. A
    // predicate check that calls loadGemmLibrary stays a local load in the
    // caller; the bundle does not keep that library.
    std::shared_ptr<const SolutionLoader> makeTensileLoader();
}
