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
#include <roc/hostnumerics/tensor.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_support.hpp"

#pragma once

namespace tensor_core_test {
namespace {
using roc::hostnumerics::test::require;
using roc::hostnumerics::test::requireInvalidArgument;
using roc::hostnumerics::test::requireNear;
using roc::hostnumerics::test::requireOutOfRange;
using roc::hostnumerics::test::requireOverflow;
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
