# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hip_kernel_plugin_tests PRIVATE
    TestBatchnormApplicability.cpp
    TestBatchnormApplicabilityLayer1.cpp
    TestBatchnormApplicabilityLayer2.cpp 
    TestBatchnormApplicabilityLayer3.cpp 
    TestBatchnormFwdInferencePlan.cpp
    TestBatchnormFwdInferenceWithVariancePlan.cpp
    TestBatchnormPlanBuilder.cpp
)
