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
    /// Encodes a library as the eager msgpack document MessagePackLoadLibraryData
    /// and LoadLibraryFile read: optional `version`, a `solutions` array, and
    /// `library`. `library` must be a MasterSolutionLibrary. An indexed
    /// in-memory library is expanded into that eager layout.
    ///
    /// The caller must be the only user of `library` for the duration of the
    /// call. The writer copies `solutions` under its lock, then reads the
    /// library tree and any blob cache without a further lock. It does not
    /// modify the library; the serialization traits take a mutable reference,
    /// which is why the implementation casts away const.
    /// Throws std::runtime_error naming every node that cannot be written.
    TENSILELITEHOST_EXPORT std::vector<uint8_t> MessagePackWriteLibraryData(
        SolutionLibrary<ContractionProblemGemm, ContractionSolution> const& library);

    /// Encodes a library as the indexed document (format_version 2) that
    /// writeMsgPackIndexed stores: `format_version`, optional `version`,
    /// `solutions_index` as a flat [index, offset, length, ...] array,
    /// `solutions_blob` as the concatenation of each solution's bytes, and
    /// `library`. Each solution's bytes come from MessagePackWriteSolutionData.
    /// `library` must be a MasterSolutionLibrary. The same exclusive-use rule
    /// as MessagePackWriteLibraryData applies.
    TENSILELITEHOST_EXPORT std::vector<uint8_t> MessagePackWriteIndexedLibraryData(
        SolutionLibrary<ContractionProblemGemm, ContractionSolution> const& library);

    /// Loads `data` with the MessagePack library loader and writes it with
    /// MessagePackWriteLibraryData.
    TENSILELITEHOST_EXPORT std::vector<uint8_t>
        MessagePackWriteLibraryFromBytes(std::vector<uint8_t> const& data);

    /// Loads `data` with the MessagePack library loader and writes it with
    /// MessagePackWriteIndexedLibraryData.
    TENSILELITEHOST_EXPORT std::vector<uint8_t>
        MessagePackWriteIndexedLibraryFromBytes(std::vector<uint8_t> const& data);

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
