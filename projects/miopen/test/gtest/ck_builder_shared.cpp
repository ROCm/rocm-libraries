// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_builder_shared.hpp"

void print_instance_strings(std::vector<std::string>& instance_strings)
{
    for(auto&& s : instance_strings)
    {
        MIOPEN_LOG_T("\t" << s);
    }
}
