// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/Version.h"

#include <cstring>

namespace stinkytofu {

const char* getRuntimeVersion() {
    return STINKYTOFU_FULL_VERSION;
}

bool versionsMatch(const char* expected, const char* actual) {
    return std::strcmp(expected, actual) == 0;
}

}  // namespace stinkytofu
