// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "_tensor_core_test_support.hpp"

TEST(TensorCore, LayoutBasics) {
    tensor_core_test::testLayoutBasics();
}

TEST(TensorCore, TensorBasicsAndBroadcasting) {
    tensor_core_test::testTensorBasicsAndBroadcasting();
}

TEST(TensorCore, TensorStorageOwnership) {
    tensor_core_test::testTensorStorageOwnership();
}

TEST(TensorCore, TensorTransformations) {
    tensor_core_test::testTensorTransformations();
}

TEST(TensorCore, TensorEncodedCopies) {
    tensor_core_test::testTensorEncodedCopies();
}

TEST(TensorCore, TensorLayoutsAndBounds) {
    tensor_core_test::testTensorLayoutsAndBounds();
}

TEST(TensorCore, TensorConversions) {
    tensor_core_test::testTensorConversions();
}
