// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ck_tile/builder/reflect/instance_traits.hpp>
#include <ck_tile/builder/types.hpp>
#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/description.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

namespace ckb = ck_tile::builder;

template <typename Builder>
concept is_valid_builder =
    // Verify that Builder is a class type
    std::is_class_v<Builder> &&

    // Verify that Builder::Instance exists and is the actual device kernel class
    std::is_class_v<typename Builder::Instance> &&

    ck_tile::reflect::HasInstanceTraits<typename Builder::Instance>;

template <auto KernelDescriptor>
concept is_valid_kernel_instance =
    is_valid_builder<ckb::ConvBuilder<KernelDescriptor.signature, KernelDescriptor.algorithm>>;
} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
