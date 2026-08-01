// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocm/interfaces/loader.h"

int main() {
    rocm_interfaces_device_key key{};
    return key.device_ordinal;
}
