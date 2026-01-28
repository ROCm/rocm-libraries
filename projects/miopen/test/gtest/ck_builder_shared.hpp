// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

#include <miopen/logger.hpp>

namespace ckb = ck_tile::builder;

void print_instance_strings(std::vector<std::string>& instance_strings);

template <typename DeviceOpA, typename DeviceOpB>
void compare_instance_vectors(std::vector<std::unique_ptr<DeviceOpA>>& instancesA,
                              std::vector<std::unique_ptr<DeviceOpB>>& instancesB)
{
    EXPECT_EQ(instancesA.size(), instancesB.size());

    // Convert instances to string lists
    std::vector<std::string> stringsA;
    std::vector<std::string> stringsB;

    for(const auto& instance : instancesA)
    {
        stringsA.push_back(instance->GetInstanceString());
    }

    for(const auto& instance : instancesB)
    {
        stringsB.push_back(instance->GetInstanceString());
    }

    // Sort for efficient set operations
    std::sort(stringsA.begin(), stringsA.end());
    std::sort(stringsB.begin(), stringsB.end());

    // Strings only in A
    std::vector<std::string> only_in_A;
    std::set_difference(stringsA.begin(),
                        stringsA.end(),
                        stringsB.begin(),
                        stringsB.end(),
                        std::back_inserter(only_in_A));

    EXPECT_EQ(only_in_A.size(), 0);

    // Strings only in B
    std::vector<std::string> only_in_B;
    std::set_difference(stringsB.begin(),
                        stringsB.end(),
                        stringsA.begin(),
                        stringsA.end(),
                        std::back_inserter(only_in_B));

    EXPECT_EQ(only_in_B.size(), 0);

    if(only_in_B.size() > 0)
    {
        MIOPEN_LOG_E("There are " << only_in_B.size() << " kernels only in B");
        print_instance_strings(only_in_B);
    }

    // Strings in both
    std::vector<std::string> in_both;
    std::set_intersection(stringsA.begin(),
                          stringsA.end(),
                          stringsB.begin(),
                          stringsB.end(),
                          std::back_inserter(in_both));

    if(in_both.size() > 0)
    {
        MIOPEN_LOG_I("There are " << in_both.size() << " kernels in both");
        print_instance_strings(in_both);
    }
}
