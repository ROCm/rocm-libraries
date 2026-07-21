// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/datatype.hpp>
#include <miopen/softmax/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>
#include <string_view>

namespace miopen {

namespace softmax {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss(isForward ? "sfmfwd-" : "sfmbwd-");

    // all the tensors must be the same size and types
    // so we can use only one set of values
    const auto& desc            = isForward ? xdxDesc : yDesc;
    const auto [sn, sc, sh, sw] = tien<4>(desc.GetLengths());
    ss << "n" << sn << "c" << sc << "h" << sh << "w" << sw;
    ss << GetDataType(desc.GetType());
    ss << "a" << alpha;
    ss << "b" << beta;
    ss << "algo" << static_cast<int>(algorithm);
    ss << "mode" << static_cast<int>(mode);

    auto printStrides = [&ss](std::string_view name, const miopen::TensorDescriptor& d) {
        const auto [n, c, h, w] = tien<4>(d.GetStrides());
        ss << name << "strides" << n << "x" << c << "x" << h << "x" << w;
    };

    if(isForward)
    {
        printStrides("x", xdxDesc);
        printStrides("y", yDesc);
    }
    else
    {
        printStrides("y", yDesc);
        printStrides("dy", dyDesc);
        printStrides("dx", xdxDesc);
    }

    return NetworkConfig{ss.str()};
}

} // namespace softmax

} // namespace miopen
