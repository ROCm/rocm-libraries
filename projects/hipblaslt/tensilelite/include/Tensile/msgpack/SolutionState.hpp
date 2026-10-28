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
    // One index of a fully derived problem type. kind is 0 for a free index, 1
    // for a batch index, and 2 for a bound index. problemIndex is its position
    // in TotalIndices. kindOrder is its position in the free, batch, or bound
    // list: free and batch follow 0..NumIndicesC, and bound follows
    // IndicesSummation, which is not problem-index order.
    struct ProblemIndexSlot
    {
        int     kind         = 0;
        int64_t problemIndex = 0;
        int64_t kindOrder    = 0;
        bool    isA          = false;
        int64_t i            = 0;
        int64_t c            = 0;
        int64_t d            = 0;
        bool    hasA         = false;
        bool    hasB         = false;
        int64_t a            = 0;
        int64_t b            = 0;
        bool    aMirror      = false;
        bool    bMirror      = false;
    };

    // Index geometry of a fully derived problem type: the operation identifier,
    // the A/B/C/D/summation names, and one slot per problem index.
    struct ProblemIndexLayout
    {
        std::string                    operationIdentifier;
        std::string                    aNames;
        std::string                    bNames;
        std::string                    cNames;
        std::string                    dNames;
        std::string                    sumNames;
        int64_t                        aDims = 0;
        int64_t                        bDims = 0;
        int64_t                        cDims = 0;
        std::vector<ProblemIndexSlot> indices;
    };

    // Reads a fully derived problem-type map. A logic entry or a custom.config
    // delta that has not been filled with defaults is missing keys and throws.
    TENSILELITEHOST_EXPORT ProblemIndexLayout
        problemIndexLayout(msgpack::object const& problemType);

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
