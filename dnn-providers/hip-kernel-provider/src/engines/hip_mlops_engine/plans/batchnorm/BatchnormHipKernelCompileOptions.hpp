// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hip/HipKernelCompileOptions.hpp"
#include <optional>

namespace hip_kernel_provider::batchnorm
{

class BatchnormHipKernelCompileOptions : public HipKernelCompileOptions
{
public:
    BatchnormHipKernelCompileOptions(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* inputTensorAttrs,
        const hipDeviceProp_t& deviceProps,
        const std::optional<hip_kernel_utils::ActivationMode>& optActivationMode = std::nullopt)
        : HipKernelCompileOptions(inputTensorAttrs, deviceProps)
    {
        addBatchnormDefaults();

        // Add activation options if activation is fused
        if(optActivationMode.has_value())
        {
            const int nrnOpId = static_cast<int>(optActivationMode.value());
            add("HIP_PLUGIN_BN_NRN_OP_ID", nrnOpId);
        }
        else
        {
            add("HIP_PLUGIN_BN_NRN_OP_ID", 0);
        }
    }

    ~BatchnormHipKernelCompileOptions() = default;

    BatchnormHipKernelCompileOptions(const BatchnormHipKernelCompileOptions&) = delete;
    BatchnormHipKernelCompileOptions& operator=(const BatchnormHipKernelCompileOptions&) = delete;
    BatchnormHipKernelCompileOptions(BatchnormHipKernelCompileOptions&&) = default;
    BatchnormHipKernelCompileOptions& operator=(BatchnormHipKernelCompileOptions&&) = default;

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
        add("HIP_PLUGIN_BN_NCHW", 1);
        add("HIP_PLUGIN_BN_MAXN", 65);
        add("HIP_PLUGIN_BN_VECTORIZE", 0);
        add("HIP_PLUGIN_BN_VEC_SIZE", 1);
        add("HIP_PLUGIN_BN_STASH_METHOD", 0);
        add("HIP_PLUGIN_BN_LOOP_UNROLL_MAXN", 768);
        add("HIP_PLUGIN_BN_LOOP_UNROLL_MAXHW", 2500);
        add("HIP_PLUGIN_BN_LDSGCN_SIZE", 16);
        add("HIP_PLUGIN_BN_LDS_SIZE", 256);
        add("HIP_PLUGIN_BN_C", 1);
        add("HIP_PLUGIN_BN_N", 1);
        add("HIP_PLUGIN_BN_N_ELEMENTS", std::string("HIP_PLUGIN_BN_N"));
        add("HIP_PLUGIN_BN_NHW", 1);
        add("HIP_PLUGIN_BN_INHW", 1);
        add("HIP_PLUGIN_BN_CHW", 1);
        add("HIP_PLUGIN_BN_HW", 1);
    }
};

} // namespace hip_kernel_provider::batchnorm
