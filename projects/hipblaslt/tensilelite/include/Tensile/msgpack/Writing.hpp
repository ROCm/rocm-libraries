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
    /// Parsing constructs predicates and does not evaluate them.
    /// WorkgroupMappingXCCCheck reads the current device only when it is
    /// evaluated, so neither this function nor the indexed writer queries a GPU.
    /// MessagePackWriteIndexedLibraryData does not parse the cached bytes.
    /// Throws std::runtime_error naming every node that cannot be written.
    TENSILELITEHOST_EXPORT std::vector<uint8_t> MessagePackWriteLibraryData(
        SolutionLibrary<ContractionProblemGemm, ContractionSolution> const& library);

    /// Encodes an in-memory library as the indexed document (format_version 2):
    /// `format_version`, optional `version`, `solutions_index` as a flat
    /// [index, offset, length, ...] array, `solutions_blob` as the concatenation
    /// of each solution's bytes, and `library`. A solution in `lib.solutions`
    /// is encoded with MessagePackWriteSolutionData. Every other index is
    /// copied from the blob unparsed. That includes a solution a query has
    /// already parsed and the caller has modified: the object lives in the
    /// cache, not in `lib.solutions`, so the original bytes are written. The
    /// eager writer has no raw slice to copy, so it writes the modified
    /// object. A lazily loaded library is not materialized. `library` must be
    /// a MasterSolutionLibrary.
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
