// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <string>

#include "compilation/KernelCompileOptions.hpp"
#include "engines/hip_mlops_engine/plans/PlanUtils.hpp"

namespace hip_kernel_provider::batchnorm
{

using namespace compilation;

class BatchnormKernelCompileOptions : public KernelCompileOptions
{
public:
    BatchnormKernelCompileOptions(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* inputTensorAttrs,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* outputTensorAttrs,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* meanTensorAttrs,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scaleTensorAttrs,
        const hipDeviceProp_t& deviceProps,
        const core::utils::ActivationMode optActivationMode = core::utils::ActivationMode::PASTHRU)
        : BatchnormKernelCompileOptions(inputTensorAttrs,
                                        outputTensorAttrs,
                                        meanTensorAttrs,
                                        scaleTensorAttrs,
                                        std::string(deviceProps.gcnArchName),
                                        optActivationMode)
    {
    }

    BatchnormKernelCompileOptions(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* inputTensorAttrs,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* outputTensorAttrs,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* meanTensorAttrs,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scaleTensorAttrs,
        const std::string& gcnArchName,
        const core::utils::ActivationMode optActivationMode = core::utils::ActivationMode::PASTHRU)
        : KernelCompileOptions(inputTensorAttrs, gcnArchName)
    {
        addBatchnormDefaults();

        auto inputDataType = inputTensorAttrs->data_type();
        auto outputDataType = outputTensorAttrs->data_type();
        auto meanDataType = meanTensorAttrs != nullptr
                                ? meanTensorAttrs->data_type()
                                : hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
        auto scaleDataType = scaleTensorAttrs->data_type();

        add("HIP_PLUGIN_BN_INPUT_TYPE", std::string(getKernelParamTypeString(inputDataType)));
        add("HIP_PLUGIN_BN_OUTPUT_TYPE", std::string(getKernelParamTypeString(outputDataType)));
        add("HIP_PLUGIN_BN_SCALE_TYPE", std::string(getKernelParamTypeString(scaleDataType)));
        add("HIP_PLUGIN_BN_MEAN_VAR_TYPE", std::string(getKernelParamTypeString(meanDataType)));

        const int nrnOpId = static_cast<int>(optActivationMode);
        add("HIP_PLUGIN_BN_NRN_OP_ID", nrnOpId);
    }

    ~BatchnormKernelCompileOptions() = default;

    BatchnormKernelCompileOptions(const BatchnormKernelCompileOptions&) = delete;
    BatchnormKernelCompileOptions& operator=(const BatchnormKernelCompileOptions&) = delete;
    BatchnormKernelCompileOptions(BatchnormKernelCompileOptions&&) = default;
    BatchnormKernelCompileOptions& operator=(BatchnormKernelCompileOptions&&) = default;

private:
    void addBatchnormDefaults()
    {
        add("HIP_PLUGIN_BN_SAVE_MEAN_VARIANCE", 0);
        add("HIP_PLUGIN_BN_RUNNING_RESULT", 0);
        add("HIP_PLUGIN_BN_NODPP", 0);
        add("HIP_PLUGIN_BN_GRP0", 1);
        add("HIP_PLUGIN_BN_GRP1", 1);
        add("HIP_PLUGIN_BN_GRP2", 1);
        add("HIP_PLUGIN_BN_VARIANT", 255);
        add("HIP_PLUGIN_BN_USESAVED", 0);
        add("HIP_PLUGIN_BN_VEC_SIZE", 1);
        add("HIP_PLUGIN_BN_STASH_METHOD", 0);
        add("HIP_PLUGIN_BN_LOOP_UNROLL_MAX_BATCH", 768);
        add("HIP_PLUGIN_BN_LOOP_UNROLL_MAX_SPATIAL", 2500);
        add("HIP_PLUGIN_BN_LDSGCN_SIZE", 16);
        add("HIP_PLUGIN_BN_LDS_SIZE", 256);
        add("HIP_PLUGIN_BN_CHANNELS", 1);
        add("HIP_PLUGIN_BN_BATCHES", 1);
        add("HIP_PLUGIN_BN_BATCH_ELEMENTS", std::string("HIP_PLUGIN_BN_BATCHES"));
        add("HIP_PLUGIN_BN_SPATIAL_SIZE", 1);
        add("HIP_PLUGIN_BN_GRPS0", 1);
        add("HIP_PLUGIN_BN_GRPS1", 1);
        add("HIP_PLUGIN_BN_GRPS2", 1);
        add("HIP_PLUGIN_BN_GRP0_FINAL", 1);
        add("HIP_PLUGIN_BN_GRP1_FINAL", 1);
        add("HIP_PLUGIN_BN_GRP2_FINAL", 1);
    }
};

/// What a batchnorm inference plan compiles and launches: the compile options, the entry point
/// with its 3D geometry, and the tensor strides the launch passes. Built by
/// buildBatchnormFwdInferenceKernel() and buildBatchnormFwdInferenceWithVarianceKernel(), which
/// the HIP MLOps engine's plans and the kernel ingestor's HipMlops packs both call, so the two
/// engines compile and launch the same program for the same graph.
struct BatchnormInferenceKernelBuild
{
    BatchnormKernelCompileOptions options;
    const char* entryPoint;
    std::array<unsigned int, 3> blockSize;
    std::array<unsigned int, 3> gridSize;
    unsigned int channelStride;
    unsigned int spatialStride;
    unsigned int batchStride;
};

} // namespace hip_kernel_provider::batchnorm
