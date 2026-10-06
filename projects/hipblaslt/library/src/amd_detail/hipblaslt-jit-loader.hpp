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

    // A TensileLite library entry with one or more solutions and, once loaded,
    // the code objects that define their kernels.
    struct TensileBundle
    {
        std::shared_ptr<TensileLite::Hardware>             hardware;
        std::shared_ptr<TensileLibrary>                    library;
        std::shared_ptr<TensileLite::hip::SolutionAdapter> adapter; // null until loaded
        std::vector<std::string>                           kernels; // every main kernel
    };

    // Loads a GEMM library entry. Throws unless it holds local solutions 0 to N-1.
    std::shared_ptr<TensileLibrary> loadGemmLibrary(const std::vector<uint8_t>& entry);

    // A source bundle and the library parsed from its entry. The library is the
    // parse of solution.entry, so a caller keeps it instead of reading those
    // bytes again.
    struct TensileSource
    {
        GeneratedSolution               solution;
        std::shared_ptr<TensileLibrary> library;
    };

    // Reads a TensileLite source bundle by directory convention: its entry, every
    // sources/*.s as an assembly unit, and every HIP source as a HIP unit that
    // can include the bundle's headers.
    TensileSource readTensileSourceBundle(const std::filesystem::path& bundle);

    // Reads the entry of a built solution for hardware and loads no code. Throws
    // unless the entry holds local solutions 0 to N-1 and the kernels they name
    // are the built kernels.
    std::shared_ptr<TensileBundle> parseTensileBundle(const BuiltSolution&                   built,
                                                      std::shared_ptr<TensileLite::Hardware> hardware);

    // Loads the built code object into a new adapter and resolves every main
    // kernel. Throws on failure.
    void loadTensileBundle(TensileBundle& bundle, const BuiltSolution& built);

    // One solution of a TensileLite bundle as the GEMM API runs it.
    struct TensileGemmBundle final : KernelBundle
    {
        using Diagnostics = hipblaslt_ext::experimental::jit::Diagnostics;

        std::shared_ptr<const TensileBundle> tensile;
        int                                  index = 0; // the local solution index

        std::string_view operationKind() const noexcept override;
        int              solutionIndex() const noexcept override
        {
            return index;
        }
        std::string name() const override
        {
            return tensile->library->solutions.at(index)->solutionName;
        }
        std::string kernelNames() const override
        {
            return tensile->library->solutions.at(index)->kernelName;
        }
        hipblasStatus_t
            support(const OperationRequest&, size_t, size_t&, Diagnostics&) const override;
    };

    // Loads CustomKernel entries from any backend into TensileGemmBundles.
    std::shared_ptr<const SolutionLoader> makeTensileLoader();
}
