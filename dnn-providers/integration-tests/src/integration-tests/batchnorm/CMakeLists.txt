# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(${INTEGRATION_TESTS_EXE}
    PRIVATE
        IntegrationGoldenRefBatchnormFwdInference.cpp
)

if(BUILD_CPP_GRAPH_TESTS)
    target_sources(${INTEGRATION_TESTS_EXE}
        PRIVATE
            IntegrationGpuBatchnormBackward.cpp
            IntegrationGpuBatchnormBackwardActivation.cpp
            IntegrationGpuBatchnormForwardInference.cpp
            IntegrationGpuBatchnormForwardInferenceWithVariance.cpp
            IntegrationGpuBatchnormForwardTraining.cpp
            IntegrationGpuBatchnormFwdInferenceVarianceActiv.cpp
            IntegrationGpuBatchnormFwdInferenceActiv.cpp
            IntegrationGpuBatchnormFwdTrainingActiv.cpp
    )
endif()
