// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/ContractionSolution_fwd.hpp>
#include <Tensile/SolutionLibrary_fwd.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    /// Encodes a library as the msgpack document MessagePackLoadLibraryData
    /// and LoadLibraryFile read. `library` must be a MasterSolutionLibrary.
    /// Indexed (format_version 2) libraries are written in the eager layout.
    /// Throws std::runtime_error naming every node that cannot be written.
    TENSILELITEHOST_EXPORT std::vector<uint8_t> MessagePackWriteLibraryData(
        SolutionLibrary<ContractionProblemGemm, ContractionSolution> const& library);

    /// Encodes one solution as a msgpack map: the form of an element of a
    /// library's `solutions` array and of a slice of an indexed library's
    /// `solutions_blob`.
    TENSILELITEHOST_EXPORT std::vector<uint8_t>
        MessagePackWriteSolutionData(ContractionSolution const& solution);

    /// Encodes solutions as a msgpack array, the form of a library's
    /// `solutions` value.
    TENSILELITEHOST_EXPORT std::vector<uint8_t> MessagePackWriteSolutionListData(
        std::vector<std::shared_ptr<ContractionSolution>> const& solutions);

    /// Encodes the index-to-file-prefix table MessagePackLoadLibraryMapping
    /// reads for a lazy library.
    TENSILELITEHOST_EXPORT std::vector<uint8_t>
        MessagePackWriteLibraryMapping(std::map<int, std::string> const& mapping);
}
