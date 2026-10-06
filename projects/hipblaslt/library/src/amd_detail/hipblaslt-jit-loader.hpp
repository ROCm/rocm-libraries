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
}
