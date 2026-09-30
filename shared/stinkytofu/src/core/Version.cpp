// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/Version.h"

#include <cstring>

namespace stinkytofu {

const char* getRuntimeVersion() {
    return STINKYTOFU_FULL_VERSION;
}

bool versionsMatch(const char* expected, const char* actual) {
    // Fail closed on a version that isn't there. loadPlugin() feeds this the
    // return value of a dlsym'd hook in an untrusted binary, so nullptr is a
    // reachable input, not a programming error — one a bare strcmp would answer
    // with a segfault instead of a rejection.
    if (!expected || !actual) return false;
    return std::strcmp(expected, actual) == 0;
}

}  // namespace stinkytofu
