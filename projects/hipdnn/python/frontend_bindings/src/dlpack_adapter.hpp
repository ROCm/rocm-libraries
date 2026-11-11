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

// Resolves a variant-pack value to a raw pointer: an int pointer, a
// DeviceBuffer, an object with data_ptr(), or a __dlpack__ producer in host
// (cpu), ROCm, or pinned host (rocm_host) memory. The DLPack byte offset is
// applied and no data is copied. The pointer is used after the DLPack import is
// released, so the producer object must own the memory and stay alive until the
// HIP work completes. `what` prefixes error messages.
void* toDataPointer(nanobind::handle value, const std::string& what);

// Converts a {uid or Tensor: value} dict with toDataPointer() on each value.
std::unordered_map<int64_t, void*> toVariantPack(const nanobind::dict& variantPack);

// Builds tensor metadata (dims, strides, data type) from a __dlpack__ producer.
// A host (cpu) tensor becomes a runtime pass-by-value tensor.
std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
    tensorAttributesFromDlpack(nanobind::handle obj, const std::string& name);

} // namespace hipdnn_python
