// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include <cstdint>
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
}
