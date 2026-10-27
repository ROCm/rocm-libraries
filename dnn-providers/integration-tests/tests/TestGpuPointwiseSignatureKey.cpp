// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The GPU reference is the default verifier and the CPU reference its fallback, so a
// dtype combination the CPU reference registers but the GPU one does not silently
// moves every such graph onto the slow path, or off the GPU-only lanes entirely. For
// the modes both implement, the GPU registry must cover every CPU dtype tuple.

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PointwiseSignatureKey.hpp>

#include "harness/gpu-graph-executor/detail/GpuPointwiseSignatureKey.hpp"

using hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
using hipdnn_integration_tests::gpu_graph_executor::detail::GpuPointwiseSignatureKey;
using CpuPointwiseSignatureKey = hipdnn_test_sdk::detail::PointwiseSignatureKey;

TEST(TestGpuPointwiseSignatureKey, CoversEveryCpuBinaryDataTypeTuple)
{
    const auto gpu = GpuPointwiseSignatureKey::getPlanBuilders();
    size_t checked = 0;
    for(const auto& [cpu, builder] : CpuPointwiseSignatureKey::getPlanBuilders())
    {
        // ADD stands for the binary arithmetic family: the CPU registry adds every
        // binary mode for the same dtype tuples, and the GPU registry is keyed on
        // dtypes only.
        if(cpu.operation != PointwiseMode::ADD)
        {
            continue;
        }
        ++checked;
        const GpuPointwiseSignatureKey key(
            cpu.inputDataType, cpu.input1DataType, cpu.outputDataType, cpu.computeDataType);
        EXPECT_EQ(gpu.count(key), 1u)
            << "GPU reference has no binary plan for in0=" << EnumNameDataType(cpu.inputDataType)
            << " in1=" << EnumNameDataType(cpu.input1DataType)
            << " out=" << EnumNameDataType(cpu.outputDataType)
            << " compute=" << EnumNameDataType(cpu.computeDataType);
    }
    EXPECT_GT(checked, 0u);
}

TEST(TestGpuPointwiseSignatureKey, CoversEveryCpuUnaryDataTypeTuple)
{
    const auto gpu = GpuPointwiseSignatureKey::getPlanBuilders();
    size_t checked = 0;
    for(const auto& [cpu, builder] : CpuPointwiseSignatureKey::getPlanBuilders())
    {
        if(cpu.operation != PointwiseMode::RELU_FWD)
        {
            continue;
        }
        ++checked;
        const GpuPointwiseSignatureKey key(
            cpu.inputDataType, cpu.outputDataType, cpu.computeDataType);
        EXPECT_EQ(gpu.count(key), 1u)
            << "GPU reference has no unary plan for in=" << EnumNameDataType(cpu.inputDataType)
            << " out=" << EnumNameDataType(cpu.outputDataType)
            << " compute=" << EnumNameDataType(cpu.computeDataType);
    }
    EXPECT_GT(checked, 0u);
}
