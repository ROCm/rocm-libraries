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
    /// Encodes a MasterContractionLibrary the caller already holds. The bytes
    /// are the eager document MessagePackLoadLibraryData and LoadLibraryFile
    /// read: optional `version`, a `solutions` array, and `library`. An indexed
    /// in-memory library is expanded into that eager layout. TensileCreateLibrary
    /// writes its library files with msgpack.packb and does not call this.
    ///
    /// The caller must be the only user of `library` for the duration of the
    /// call. The writer copies `solutions` under its lock, then reads the
    /// library tree and any blob cache without a further lock. The serialization
    /// traits take a mutable reference, which is why the implementation casts
    /// away const.
    ///
    /// The eager layout has nowhere to put unparsed bytes, so writing a lazily
    /// loaded library parses every solution in its blob cache and leaves them
    /// there. That is an observable change in the library's memory footprint.
    /// MessagePackWriteIndexedLibraryData does not do it.
    /// Throws std::runtime_error naming every node that cannot be written.
    TENSILELITEHOST_EXPORT std::vector<uint8_t> MessagePackWriteLibraryData(
        SolutionLibrary<ContractionProblemGemm, ContractionSolution> const& library);

    /// Encodes an in-memory library as the indexed document (format_version 2):
    /// `format_version`, optional `version`, `solutions_index` as a flat
    /// [index, offset, length, ...] array, `solutions_blob` as the concatenation
    /// of each solution's bytes, and `library`. A solution the caller holds as
    /// an object is encoded with MessagePackWriteSolutionData; one that is
    /// still only bytes in a blob cache is copied through unparsed, so a lazily
    /// loaded library is not materialized. `library` must be a
    /// MasterSolutionLibrary.
    /// The same exclusive-use rule as MessagePackWriteLibraryData applies.
    /// writeMsgPackIndexed stores this layout with msgpack.packb.
    TENSILELITEHOST_EXPORT std::vector<uint8_t> MessagePackWriteIndexedLibraryData(
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
