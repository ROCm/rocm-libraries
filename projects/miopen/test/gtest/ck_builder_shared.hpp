// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

namespace ckb         = ck_tile::builder;
using BaseOperator    = ck::tensor_operation::device::BaseOperator;
using BaseOperatorPtr = std::unique_ptr<BaseOperator>;

template <typename Builder>
constexpr void do_builder_checks()
{
    // Verify that Builder is a class type
    static_assert(std::is_class_v<Builder>, "Builder should be a class type");

    // Verify that Builder::Instance exists and is the actual device kernel class
    static_assert(std::is_class_v<typename Builder::Instance>,
                  "Builder::Instance should be a class type");

    static_assert(ck_tile::reflect::HasInstanceTraits<typename Builder::Instance>);
}

std::size_t FirstDifference(const std::string& a, const std::string& b);

void print_closest_instance(std::string builderKernelInstanceString, auto&& factoryInstances)
{
    std::size_t m = 0;
    std::string desc{};

    for(auto&& k : factoryInstances)
    {
        auto kernelDescription = k->GetInstanceString();
        auto firstDifferent    = FirstDifference(builderKernelInstanceString, kernelDescription);
        if(firstDifferent > m)
        {
            m    = firstDifferent;
            desc = kernelDescription;
        }
    }

    if(m < builderKernelInstanceString.size())
    {
        std::cout << builderKernelInstanceString << std::endl << desc << std::endl;

        for(auto i = 0; i < m; i++)
        {
            std::cout << ' ';
        }

        std::cout << '^' << std::endl << std::endl;
    }
}
