// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * arch_num_cus.cpp -- helpers/split_k.py arch_num_cus (see rocke/arch_num_cus.h).
 * Keep the table in step with _ARCH_NUM_CUS: test_arch_num_cus.py compares them.
 */
#include "rocke/arch_num_cus.h"

#include <string.h>

namespace
{

struct arch_cus
{
    const char* arch;
    int num_cus;
};

/* Python _ARCH_NUM_CUS */
const arch_cus k_arch_num_cus[] = {
    {"gfx942", 304}, /* 8 XCDs * 38 CUs */
    {"gfx950", 256}, /* 8 XCDs * 32 CUs */
    {"gfx1151", 64}, /* RDNA gfx1151 (WMMA) */
};

} // namespace

extern "C" int rocke_arch_num_cus(const char* arch)
{
    if(arch != nullptr)
    {
        for(const arch_cus& e : k_arch_num_cus)
        {
            if(strcmp(e.arch, arch) == 0)
                return e.num_cus;
        }
    }
    return ROCKE_DEFAULT_NUM_CUS;
}
