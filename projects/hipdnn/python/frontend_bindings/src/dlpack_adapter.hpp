// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace hipdnn_python
{

// Resolves an int pointer, a DeviceBuffer, or a __dlpack__ producer on the
// current ROCm device to a raw device pointer. The DLPack byte offset is
// applied and no data is copied. `what` prefixes error messages.
void* toDevicePointer(nanobind::handle value, const std::string& what);

// Converts a {uid: value} dict with toDevicePointer() on each value.
std::unordered_map<int64_t, void*> toVariantPack(const nanobind::dict& variantPack);

// Builds tensor metadata (dims, strides, data type) from a __dlpack__ producer.
// A single-element host (CPU) tensor becomes a compile-time-constant scalar.
std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
    tensorAttributesFromDlpack(nanobind::handle obj, const std::string& name);

} // namespace hipdnn_python
