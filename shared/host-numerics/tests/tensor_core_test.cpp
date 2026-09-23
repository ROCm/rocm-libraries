// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_tensor_core_test_support.hpp"

int main() {
    using namespace tensor_core_test;
    testShape();
    testLayoutBasics();
    testTensorBasicsAndBroadcasting();
    testTensorStorageOwnership();
    testTensorTransformations();
    testTensorEncodedCopies();
    testTensorLayoutsAndBounds();
    testTensorConversions();
    return 0;
}
