// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_backend.h"

#include <array>
#include <string>

namespace hipdnn_backend::test_utilities
{

/// Reads and clears the calling thread's last backend error.
inline std::string takeLastErrorString()
{
    std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> buffer{};
    hipdnnGetLastErrorString(buffer.data(), buffer.size());
    return buffer.data();
}

} // namespace hipdnn_backend::test_utilities
