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

template <typename DeviceOpLegacy, typename DeviceOpCKBuilder>
void compare_instance_vectors(std::vector<std::unique_ptr<DeviceOpLegacy>>& legacyInstances,
                              std::vector<std::unique_ptr<DeviceOpCKBuilder>>& ckBuilderInstances)
{
    EXPECT_EQ(legacyInstances.size(), ckBuilderInstances.size());

    // Convert instances to string lists
    std::vector<std::string> legacyInstanceStrings;
    std::vector<std::string> ckBuilderInstanceStrings;

    for(const auto& instance : legacyInstances)
    {
        legacyInstanceStrings.push_back(instance->GetInstanceString());
    }

    for(const auto& instance : ckBuilderInstances)
    {
        ckBuilderInstanceStrings.push_back(instance->GetInstanceString());
    }

    // Sort for efficient set operations
    std::sort(legacyInstanceStrings.begin(), legacyInstanceStrings.end());
    std::sort(ckBuilderInstanceStrings.begin(), ckBuilderInstanceStrings.end());

    // Find instances only in the legacy list
    std::vector<std::string> onlyInLegacyInstances;
    std::set_difference(legacyInstanceStrings.begin(),
                        legacyInstanceStrings.end(),
                        ckBuilderInstanceStrings.begin(),
                        ckBuilderInstanceStrings.end(),
                        std::back_inserter(onlyInLegacyInstances));

    EXPECT_EQ(onlyInLegacyInstances.size(), 0);

    if(onlyInLegacyInstances.size() > 0)
    {
        MIOPEN_LOG_E("There are " << onlyInLegacyInstances.size()
                                  << " kernels only in legacy instance vector");
        print_instance_strings(onlyInLegacyInstances);
    }

    // Find instances only in the CK builder list
    std::vector<std::string> onlyInCKBuilderInstances;
    std::set_difference(ckBuilderInstanceStrings.begin(),
                        ckBuilderInstanceStrings.end(),
                        legacyInstanceStrings.begin(),
                        legacyInstanceStrings.end(),
                        std::back_inserter(onlyInCKBuilderInstances));

    EXPECT_EQ(onlyInCKBuilderInstances.size(), 0);

    if(onlyInCKBuilderInstances.size() > 0)
    {
        MIOPEN_LOG_E("There are " << onlyInCKBuilderInstances.size()
                                  << " kernels only in CK Builder instance vector");
        print_instance_strings(onlyInCKBuilderInstances);
    }

    // Strings in both
    std::vector<std::string> in_both;
    std::set_intersection(legacyInstanceStrings.begin(),
                          legacyInstanceStrings.end(),
                          ckBuilderInstanceStrings.begin(),
                          ckBuilderInstanceStrings.end(),
                          std::back_inserter(in_both));

    if(in_both.size() > 0)
    {
        MIOPEN_LOG_I("There are " << in_both.size() << " kernels in both");
        print_instance_strings(in_both);
    }
}
