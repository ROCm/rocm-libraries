// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace hipdnn_backend
{

/// Convert a pointer to a hexadecimal string representation.
/// Returns "null" for null pointers, otherwise "0x" followed by the hex address.
inline std::string ptrToString(const void* ptr)
{
    if(ptr == nullptr)
    {
        return "null";
    }
    std::ostringstream oss;
    oss << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(ptr);
    return oss.str();
}

} // namespace hipdnn_backend
