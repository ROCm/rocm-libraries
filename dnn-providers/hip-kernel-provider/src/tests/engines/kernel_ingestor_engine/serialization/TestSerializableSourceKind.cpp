// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/serialization/SerializableSourceKind.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::serialization
{
namespace
{

using hipdnn_plugin_sdk::ingestor::KernelSourceKind;

// Saving accepts only kpack kernels. embedded_source kernels have no recorded argument
// signature, and their code bytes are not exposed. hsaco_file and rocke_builder kernels have
// no adapter that loads them. Change this test together with the gate when one of these
// facts changes.
TEST(TestSerializableSourceKind, AcceptsOnlyKpack)
{
    EXPECT_TRUE(isSerializableSourceKind(KernelSourceKind::KPACK));
    EXPECT_FALSE(isSerializableSourceKind(KernelSourceKind::EMBEDDED_SOURCE));
    EXPECT_FALSE(isSerializableSourceKind(KernelSourceKind::HSACO_FILE));
    EXPECT_FALSE(isSerializableSourceKind(KernelSourceKind::ROCKE_BUILDER));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
