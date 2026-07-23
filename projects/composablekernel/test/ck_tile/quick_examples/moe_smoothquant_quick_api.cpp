// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ck_tile/core.hpp>

#include <stdexcept>

#include "moe_smoothquant.hpp"

namespace {

template <typename InType,
          typename OutType,
          ck_tile::index_t RepeatN,
          ck_tile::index_t VectorN>
using QuickTrait =
    moe_smoothquant_traits_<InType, OutType, 1, RepeatN, 4, 64, VectorN, true, false>;

template <typename InType, typename OutType>
float dispatch_quick(moe_smoothquant_args args, const ck_tile::stream_config& stream_config)
{
    if(args.hidden_size <= 64)
    {
        return moe_smoothquant_<QuickTrait<InType, OutType, 1, 1>>(stream_config, args);
    }
    if(args.hidden_size <= 128)
    {
        if(args.hidden_size % 2 == 0)
        {
            return moe_smoothquant_<QuickTrait<InType, OutType, 1, 2>>(stream_config, args);
        }
        return moe_smoothquant_<QuickTrait<InType, OutType, 2, 1>>(stream_config, args);
    }
    throw std::runtime_error("Quick MoE smoothquant supports hidden_size <= 128");
}

} // namespace

float moe_smoothquant(moe_smoothquant_traits traits,
                      moe_smoothquant_args args,
                      const ck_tile::stream_config& stream_config)
{
    if(traits.in_type == "fp16" && traits.out_type == "fp8")
    {
        return dispatch_quick<ck_tile::fp16_t, ck_tile::fp8_t>(args, stream_config);
    }
    if(traits.in_type == "bf16" && traits.out_type == "int8")
    {
        return dispatch_quick<ck_tile::bf16_t, ck_tile::int8_t>(args, stream_config);
    }
    throw std::runtime_error("Unsupported quick MoE smoothquant dtype combination");
}
