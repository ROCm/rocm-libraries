// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>

#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/input-init/FillInputs.hpp"
#include "harness/input-init/InputFillRecipes.hpp"

namespace hipdnn_integration_tests::bundle
{

/// The inputs one run of a bundle's graph reads, ready to hand to an executor.
///
/// Shared by the verification harness and the catalog sweep, so both read or generate a
/// bundle's inputs by one rule: with the recipes and seeds that rule applies, the same
/// call twice yields the same values, which is what lets the sweep refill before every
/// entry without the reference ever disagreeing about what the inputs were.
struct BundleInputs
{
    enum class Status
    {
        READY,
        /// The inputs cannot be provided for this graph (no fill recipe, a sub-byte input
        /// with no packed copy). A gap in the harness's coverage, not a fault.
        UNVERIFIABLE,
        /// A device fill failed: the device or the harness, never the graph.
        DEVICE_FILL_FAILED,
        /// A golden tensor blob was unreadable or the wrong size.
        TENSOR_LOAD_FAILED,
    };

    Status status = Status::READY;
    /// The reason, or the underlying error text for the two failures.
    std::string message;

    /// This run's inputs, plus the golden outputs when the bundle has them.
    TensorMap inputs;
    /// A packed copy of the inputs, filled only when an input is sub-byte; empty otherwise.
    TensorMap packedInputs;
    /// How many inputs were generated on the device.
    std::size_t deviceFilled = 0;
};

/// Reads the bundle's golden tensors when it has them, otherwise generates every input
/// from `recipes`. `deviceFiller` may be null (everything is filled on the host); it is
/// also ignored for a graph with a sub-byte tensor, which is always filled on the host.
BundleInputs prepareBundleInputs(const IntegrationTestBundle& bundle,
                                 InputFillRecipes& recipes,
                                 DeviceInputFiller* deviceFiller);

} // namespace hipdnn_integration_tests::bundle
