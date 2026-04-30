// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <ostream>
#include <variant>

#include "GpuConvolutionFwdSignatureKey.hpp"
#include "GpuPointwiseDummySignatureKey.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// Variant of all GPU plan signature key types.
// Add new signature key types here as GPU plans are implemented.
using GpuPlanRegistrySignatureKey
    = std::variant<GpuPointwiseDummySignatureKey, GpuConvolutionFwdSignatureKey>;

struct GpuPlanRegistrySignatureKeyHash
{
    std::size_t operator()(const GpuPlanRegistrySignatureKey& k) const
    {
        return std::visit([](const auto& x) { return x.hashSelf(); }, k);
    }
};

struct GpuPlanRegistrySignatureKeyEqual
{
    template <typename T, typename U>
    bool operator()([[maybe_unused]] const T& a, [[maybe_unused]] const U& b) const
    {
        return false;
    }

    template <typename T>
    bool operator()(const T& a, const T& b) const
    {
        return a == b;
    }

    // NOLINTNEXTLINE(readability-redundant-casting)
    bool operator()(const GpuPlanRegistrySignatureKey& a,
                    const GpuPlanRegistrySignatureKey& b) const
    {
        return std::visit(*this, a, b);
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuPlanRegistrySignatureKey& key)
{
    std::visit([&os](const auto& arg) { os << arg; }, key);
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
