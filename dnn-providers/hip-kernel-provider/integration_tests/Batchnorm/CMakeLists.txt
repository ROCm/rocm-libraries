# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hip_kernel_plugin_integration_tests PRIVATE
    IntegrationGpuBatchnormForwardInference.cpp
    IntegrationGpuBatchnormForwardInferenceAndActivation.cpp
    IntegrationGpuBatchnormForwardInferenceWithVariance.cpp
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivation.cpp
)
