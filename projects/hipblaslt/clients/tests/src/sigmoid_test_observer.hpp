// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipblaslt/hipblaslt.h>

// Test-only pass-through: the real descriptor setter always receives unchanged
// arguments. Observation is enabled only by the sigmoid client regression test.
struct SigmoidEpilogueObservation
{
    bool                sent     = false;
    hipblasLtEpilogue_t  epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
};

extern thread_local SigmoidEpilogueObservation* sigmoid_epilogue_observation;

class SigmoidObservationScope
{
    SigmoidEpilogueObservation* previous;
public:
    explicit SigmoidObservationScope(SigmoidEpilogueObservation& result)
        : previous(sigmoid_epilogue_observation)
    {
        sigmoid_epilogue_observation = &result;
    }
    ~SigmoidObservationScope() { sigmoid_epilogue_observation = previous; }
};

hipblasStatus_t sigmoid_test_set_attribute(hipblasLtMatmulDesc_t           desc,
                                          hipblasLtMatmulDescAttributes_t attr,
                                          const void*                     value,
                                          size_t                          size);
