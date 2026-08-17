// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "conv_driver.hpp"
#include "registry_driver_maker.hpp"

static Driver* makeDriver(const std::string& base_arg)
{
    if(base_arg == "convfp32")
        return new ConvDriver<float, float>();
    return nullptr;
}

REGISTER_DRIVER_MAKER(makeDriver);
