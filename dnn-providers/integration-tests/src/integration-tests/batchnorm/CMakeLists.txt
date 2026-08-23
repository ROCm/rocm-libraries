# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_cpp_graph_test_sources(
    IntegrationGpuBatchnormBackward.cpp
    IntegrationGpuBatchnormBackwardActivation.cpp
    IntegrationGpuBatchnormForwardInference.cpp
    IntegrationGpuBatchnormForwardInferenceWithVariance.cpp
    IntegrationGpuBatchnormForwardTraining.cpp
    IntegrationGpuBatchnormFwdInferenceVarianceActiv.cpp
    IntegrationGpuBatchnormFwdInferenceActiv.cpp
    IntegrationGpuBatchnormFwdTrainingActiv.cpp
)
