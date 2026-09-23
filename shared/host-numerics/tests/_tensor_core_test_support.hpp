// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <roc/host_numerics/tensor.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_support.hpp"

#pragma once

namespace tensor_core_test {
namespace {
using roc::host_numerics::test::require;
using roc::host_numerics::test::requireInvalidArgument;
using roc::host_numerics::test::requireNear;
using roc::host_numerics::test::requireOutOfRange;
using roc::host_numerics::test::requireOverflow;
}  // namespace

void testShape();
void testLayoutBasics();
void testTensorBasicsAndBroadcasting();
void testTensorStorageOwnership();
void testTensorTransformations();
void testTensorEncodedCopies();
void testTensorLayoutsAndBounds();
void testTensorConversions();
}  // namespace tensor_core_test
