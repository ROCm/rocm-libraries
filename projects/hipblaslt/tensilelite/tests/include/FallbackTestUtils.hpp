// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/ExactLogicLibrary.hpp>

namespace TensileLite
{
namespace testing
{
    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    constexpr int _MI350_CHIP_ID = 0x75a0;
    constexpr int _MI355_CHIP_ID = 0x75a3;
    constexpr int _SPX_CU     = 256;
    constexpr int _CPX_CU     = 64;

    // -----------------------------------------------------------------------
    // Debug helpers (opt-in via TENSILE_DB=0x1 or any non-zero value)
    // -----------------------------------------------------------------------

    inline bool dbgEnabled()
    {
        return Debug::Instance().printDeviceSelection();
    }

    inline void dbg(const std::string& msg)
    {
        if(dbgEnabled())
            std::cout << "[FALLBACK_TEST] " << msg << std::endl;
    }

    inline std::string hexChipId(int chipId)
    {
        std::ostringstream ss;
        ss << "0x" << std::hex << chipId;
        return ss.str();
    }

    // -----------------------------------------------------------------------
    // Device factory
    // -----------------------------------------------------------------------

    /// Create a mock AMDGPU device with the given chip ID and CU count.
    inline AMDGPU makeDevice(int chipId, int cuCount, const std::string& name)
    {
        return AMDGPU(
            AMDGPU::Processor::gfx950, cuCount, name, std::make_optional(chipId));
    }

    // -----------------------------------------------------------------------
    // Solution factory
    // -----------------------------------------------------------------------

    /// Create a ContractionSolution with the given name and index.
    /// All predicates default to True<>, so it passes any library-level check.
    inline std::shared_ptr<ContractionSolution> makeSolution(const std::string& name, int index)
    {
        auto sol          = std::make_shared<ContractionSolution>();
        sol->index        = index;
        sol->solutionName = name;
        return sol;
    }

    // -----------------------------------------------------------------------
    // Hardware predicate builder
    // -----------------------------------------------------------------------

    /// Build a HardwarePredicate wrapping And(ProcessorEqual, [PciChipIdEqual], [CUCountEqual]).
    /// Omitted optional arguments are simply left out of the And predicate.
    inline HardwarePredicate makeHwPred(
        AMDGPU::Processor            processor,
        std::optional<int>           chipId = std::nullopt,
        std::optional<int>           cuCount = std::nullopt)
    {
        std::vector<std::shared_ptr<Predicates::Predicate<AMDGPU>>> preds;
        preds.push_back(std::make_shared<Predicates::GPU::ProcessorEqual>(processor));

        if(chipId.has_value())
            preds.push_back(std::make_shared<Predicates::GPU::PciChipIdEqual>(chipId.value()));

        if(cuCount.has_value())
            preds.push_back(std::make_shared<Predicates::GPU::CUCountEqual>(cuCount.value()));

        std::shared_ptr<Predicates::Predicate<AMDGPU>> inner;
        if(preds.size() == 1)
            inner = preds[0];
        else
            inner = std::make_shared<Predicates::And<AMDGPU>>(preds);

        return HardwarePredicate(
            std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(inner),
            chipId);
    }

    // -----------------------------------------------------------------------
    // Problem predicate row builders
    // -----------------------------------------------------------------------

    using ProbLib    = std::shared_ptr<SolutionLibrary<ContractionProblemGemm, ContractionSolution>>;
    using ProbRow    = ContractionProblemSelectionLibrary::Row;
    using HwRow      = ContractionHardwareSelectionLibrary::Row;

    /// Create a ProblemSelectionLibrary row tagged as EqualityMatching.
    inline ProbRow makeEqRow(ProbLib lib)
    {
        ContractionProblemPredicate pred(
            std::make_shared<Predicates::Contraction::EqualityMatching>());
        return ProbRow(pred, lib);
    }

    /// Create a ProblemSelectionLibrary row tagged as FreeSizeMatching (Origami/OOB).
    inline ProbRow makeOobRow(ProbLib lib)
    {
        ContractionProblemPredicate pred(
            std::make_shared<Predicates::Contraction::FreeSizeMatching>());
        return ProbRow(pred, lib);
    }

    // -----------------------------------------------------------------------
    // Library composition helpers
    // -----------------------------------------------------------------------

    /// Build a ProblemSelectionLibrary with an equality row (optional) + oob row.
    /// If eqLib is nullptr only the oob row is included.
    inline std::shared_ptr<ContractionProblemSelectionLibrary>
        buildProblemLib(ProbLib eqLib, ProbLib oobLib)
    {
        auto lib = std::make_shared<ContractionProblemSelectionLibrary>();
        if(eqLib)
            lib->rows.push_back(makeEqRow(eqLib));
        lib->rows.push_back(makeOobRow(oobLib));
        return lib;
    }

    /// Overload: oob-only ProblemSelectionLibrary.
    inline std::shared_ptr<ContractionProblemSelectionLibrary>
        buildProblemLib(ProbLib oobLib)
    {
        return buildProblemLib(nullptr, oobLib);
    }

    /// Build a HardwareSelectionLibrary from an ordered vector of (predicate, sub-library) pairs.
    /// The vector must already be in the correct evaluation order (most-specific first).
    inline std::shared_ptr<ContractionHardwareSelectionLibrary>
        buildHwLib(const std::vector<std::pair<HardwarePredicate, ProbLib>>& rows)
    {
        auto lib = std::make_shared<ContractionHardwareSelectionLibrary>();
        for(auto const& [pred, subLib] : rows)
            lib->rows.emplace_back(pred, subLib);
        return lib;
    }

    // -----------------------------------------------------------------------
    // Dummy problem (for findBestSolution calls)
    // -----------------------------------------------------------------------

    inline ContractionProblemGemm dummyProblem()
    {
        return ContractionProblemGemm::GEMM(
            false, false, 1024, 1024, 1024, 1024, 1024, 1024, 1.0, false, 1);
    }

    // -----------------------------------------------------------------------
    // Selection helper with debug trace
    // -----------------------------------------------------------------------

    /// Run findBestSolution against a HardwareSelectionLibrary and return the
    /// solution name, or "<none>" if nothing matched.
    inline std::string selectSolution(
        const ContractionHardwareSelectionLibrary& lib,
        const Hardware&                            hw,
        const std::string&                         deviceLabel = "")
    {
        auto problem  = dummyProblem();
        auto solution = lib.findBestSolution(problem, hw);

        std::string name = solution ? solution->solutionName : "<none>";

        if(dbgEnabled())
        {
            std::cout << "[FALLBACK_TEST] "
                      << (deviceLabel.empty() ? "device" : deviceLabel)
                      << " => " << name << std::endl;
        }
        return name;
    }

    /// Convenience: wrap a single solution in a SingleContractionLibrary.
    inline std::shared_ptr<SingleContractionLibrary>
        singleLib(std::shared_ptr<ContractionSolution> sol)
    {
        return std::make_shared<SingleContractionLibrary>(sol);
    }

} // namespace testing
} // namespace TensileLite
