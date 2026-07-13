// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "dispatcher/SdpaProblem.hpp"

namespace rocke_client::dispatcher
{

const char* toString(TensorLayout layout)
{
    switch(layout)
    {
    case TensorLayout::BSHD:
        return "BSHD";
    case TensorLayout::BHSD:
        return "BHSD";
    case TensorLayout::OTHER:
    default:
        return "UNKNOWN";
    }
}

std::map<std::string, AttrValue> SdpaProblem::attributes() const
{
    return {
        {"mask_mode", AttrValue{maskMode}},
        {"dropout_probability", AttrValue{dropoutProbability}},
        {"scale_policy", AttrValue{scalePolicy}},
        {"padding_mask", AttrValue{paddingMask}},
        {"alibi_mask", AttrValue{alibiMask}},
    };
}

} // namespace rocke_client::dispatcher
