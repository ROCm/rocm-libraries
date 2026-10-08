// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// The JIT library's own MessagePack documents, which TensileLite does not
// define. Library entries, masters and index mappings are TensileLite's and
// are written with its writer. Malformed input is Status::Code::Failed. A
// YAML-only build has no MessagePack support and fails every call.
namespace hipblaslt_jit::msgpack_io
{
    // {"next": <first unallocated index>}.
    Status writeAllocator(int64_t next, std::vector<uint8_t>& bytes);
    Status readAllocator(const std::vector<uint8_t>& bytes, int64_t& next);

    // cache-key.msgpack. Unknown fields are ignored. A missing field stays at
    // the default; present records which names were read so a caller can tell
    // a default from a stored value.
    struct CacheKeyDocument
    {
        std::string                        backendId, comgr, rocmPath;
        uint32_t                           backendVersion = 0;
        std::string                        targetId, isa, libraryArch;
        int                                codeObjectVersion = 0;
        int                                schema            = 1;
        int                                wavefrontSize     = 0;
        std::map<std::string, std::string> environment;
        bool                               hasBackendId = false, hasBackendVersion = false;
        bool                               hasTargetId = false, hasIsa = false, hasLibraryArch = false;
        bool                               hasWavefront = false;
    };
    Status writeCacheKey(const CacheKeyDocument& document, std::vector<uint8_t>& bytes);
    Status readCacheKey(const std::vector<uint8_t>& bytes, CacheKeyDocument& document);
}
