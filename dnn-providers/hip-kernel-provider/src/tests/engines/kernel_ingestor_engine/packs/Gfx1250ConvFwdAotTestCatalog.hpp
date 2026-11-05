// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdAotTestVectors.hpp"

/**
 * @file Gfx1250ConvFwdAotTestCatalog.hpp
 * @brief The golden vectors' instances and problems as the engine's port reads them.
 *
 * Gfx1250ConvFwdAotTestVectors.hpp (make_gfx1250_conv_fwd_aot_test_vectors.py --header)
 * carries each catalog instance's KMD fields as `name=value;` text, the fields left at 0
 * or empty omitted. Decoding them here into an InstanceGeometry is what the engine's
 * prepare step does with a kernel's metadata, so the port is tested on the catalog
 * without a packaged product.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx1250_conv_fwd_aot
{

namespace conv = hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd;
namespace vectors = gfx1250_conv_fwd_aot_vectors;

/// The InstanceGeometry of golden instance @p kernel.
inline conv::InstanceGeometry instanceGeometry(const vectors::KernelVector& kernel)
{
    conv::InstanceGeometry::Ints ints;
    conv::InstanceGeometry::Strings strings;
    for(const auto* name : vectors::INT_FIELDS)
    {
        ints.emplace(name, 0);
    }
    for(const auto* name : vectors::STRING_FIELDS)
    {
        strings.emplace(name, "");
    }
    std::string_view text(kernel.metadata);
    while(!text.empty())
    {
        const auto end = text.find(';');
        const auto entry = text.substr(0, end);
        const auto equals = entry.find('=');
        if(end == std::string_view::npos || equals == std::string_view::npos)
        {
            throw std::invalid_argument(std::string("malformed golden metadata: ")
                                        + kernel.metadata);
        }
        const std::string name(entry.substr(0, equals));
        const std::string value(entry.substr(equals + 1));
        if(const auto it = ints.find(name); it != ints.end())
        {
            it->second = std::stoll(value);
        }
        else if(const auto it2 = strings.find(name); it2 != strings.end())
        {
            it2->second = value;
        }
        else
        {
            throw std::invalid_argument("golden metadata names no KMD field " + name);
        }
        text.remove_prefix(end + 1);
    }
    return {std::move(ints), std::move(strings)};
}

/// Golden problem @p index of vectors::PROBLEMS.
inline conv::Problem problemAt(size_t index)
{
    const auto& row = vectors::PROBLEMS[index];
    conv::Problem p;
    p.n = row[0];
    p.c = row[1];
    p.k = row[2];
    p.h = row[3];
    p.w = row[4];
    p.y = row[5];
    p.x = row[6];
    p.groups = row[7];
    p.strideH = row[8];
    p.strideW = row[9];
    p.padH = row[10];
    p.padW = row[11];
    p.dilationH = row[12];
    p.dilationW = row[13];
    p.dtype = row[14] != 0 ? conv::DataType::BF16 : conv::DataType::FP16;
    return p;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx1250_conv_fwd_aot
