// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Writes TensileLite MessagePack libraries in the lazy-loading layout the stock
// loader reads. Solutions are never built here: entries keep the generator's
// bytes. Malformed input is Status::Code::Failed. A YAML-only build has no
// MessagePack support and fails every call.
namespace hipblaslt_jit::msgpack_io
{
    // Sets the index of a one-solution entry's solution and of the Single
    // library node that selects it.
    Status rewriteEntryIndex(const std::vector<uint8_t>& entry,
                             int32_t                     index,
                             std::vector<uint8_t>&       rewritten);

    // The entry's solution index and kernel name, and the predicate of its only
    // library row, re-encoded as MessagePack.
    struct EntryFields
    {
        int64_t              index = -1;
        std::string          kernelName;
        std::vector<uint8_t> predicate;
    };
    Status readEntry(const std::vector<uint8_t>& entry, EntryFields& fields);

    // One master row: And(SizeEqual(i, sizes[i]) for every i, predicate)
    // selecting Placeholder(prefix).
    struct MasterRow
    {
        std::vector<size_t>  sizes;
        std::vector<uint8_t> predicate;
        std::string          prefix;
    };

    // Appends rows to a master whose library is a Problem node, keeping every
    // other field. An empty master starts {"solutions": [], "library": Problem}.
    Status appendMasterRows(const std::vector<uint8_t>&   master,
                            const std::vector<MasterRow>& rows,
                            std::vector<uint8_t>&         appended);

    // The Placeholder prefix of every master row, in order.
    Status readMasterPrefixes(const std::vector<uint8_t>& master, std::vector<std::string>& prefixes);

    // The mapping file: solution index to library file prefix.
    Status writeMapping(const std::map<int32_t, std::string>& mapping, std::vector<uint8_t>& bytes);
    Status readMapping(const std::vector<uint8_t>& bytes, std::map<int32_t, std::string>& mapping);

    // {"next": <first unallocated index>}.
    Status writeAllocator(int64_t next, std::vector<uint8_t>& bytes);
    Status readAllocator(const std::vector<uint8_t>& bytes, int64_t& next);
}
