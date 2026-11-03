// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <msgpack.hpp>

#include <Tensile/ContractionSolution_fwd.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    // The static catalogs that SolutionState.cpp accepts when it rebuilds a
    // record from state.
    TENSILELITEHOST_EXPORT std::vector<std::string> supportedBuildChipIds();
    TENSILELITEHOST_EXPORT std::vector<std::string> supportedDataTypeNames();

    /**
     * Builds the library record of one fully derived solution state, the map
     * Solution.FromOriginalState receives after defaults and derived parameters
     * are filled in.
     *
     * The input is not a raw custom.config delta and not an unexpanded logic
     * entry. It must already contain ProblemType.TotalIndices,
     * IndicesSummation, IndicesFree, IndicesBatch, IndexAssignmentsA,
     * IndexAssignmentsB, and the other derived fields the record reads. A
     * metadata note may carry that derived map. A caller that passes an
     * unexpanded map gets a missing-key error.
     *
     * The assembler types each scalar of a metadata note from its YAML text,
     * so an empty string arrives as 0.0 and a null as the string "null"; both
     * are read as the values the state holds.
     *
     * WorkgroupMappingXCCCheck stores only its value. Its compute-unit count
     * stays 0 until the predicate is evaluated, which is when the device is
     * read. Building the record does not query a GPU.
     *
     * Throws std::runtime_error naming the key when a key the record needs is
     * missing or holds a value of the wrong kind.
     */
    TENSILELITEHOST_EXPORT std::shared_ptr<ContractionSolution>
                           solutionFromState(msgpack::object const& state);
}
