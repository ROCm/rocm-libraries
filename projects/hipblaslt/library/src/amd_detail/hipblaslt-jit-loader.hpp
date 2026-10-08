// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include <Tensile/Contractions.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>

namespace hipblaslt_jit
{
    using TensileLibrary = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;

    // Loads a GEMM library entry. Throws unless it holds local solutions 0 to N-1.
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

    // One solution of a TensileLite record as the GEMM API runs it. The
    // TensileLite library is kept so the solution stays alive. Loading a
    // published solution goes through JitLibrary.
    struct TensileGemmBundle final : KernelBundle
    {
        using Diagnostics = hipblaslt_ext::experimental::jit::Diagnostics;

        std::shared_ptr<TensileLibrary>                    library;
        std::shared_ptr<TensileLite::Hardware>             hardware;
        std::shared_ptr<TensileLite::hip::SolutionAdapter> adapter; // null until loaded
        int                                                index = 0; // the local solution index

        int solutionIndex() const noexcept override
        {
            return index;
        }
        std::string name() const override
        {
            return library->solutions.at(index)->solutionName;
        }
        std::string kernelNames() const override
        {
            return library->solutions.at(index)->kernelName;
        }
        hipblasStatus_t support(const GemmRequest&, size_t, size_t&, Diagnostics&) const override;
    };

    // Loads solution records from any backend into TensileGemmBundles.
    std::shared_ptr<const SolutionLoader> makeTensileLoader();
}
