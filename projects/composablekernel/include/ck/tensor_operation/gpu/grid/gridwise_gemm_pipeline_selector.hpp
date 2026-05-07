// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
#include <iostream>
#endif

#include "ck/utility/pipeline_enum.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v2.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v4_direct_load.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v5.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_wavegroup_v1.hpp"

namespace ck {
template <PipelineVersion PipelineVer,
          index_t NumPrefetch          = 1,
          LoopScheduler LoopSched      = LoopScheduler::Default,
          bool AEnableLds              = true,
          bool BEnableLds              = true,
          bool EnableWaveGroup         = false,
          TensorLoadOption ALoadOption = TensorLoadOption::DEFAULT_LOAD,
          TensorLoadOption BLoadOption = TensorLoadOption::DEFAULT_LOAD>
constexpr auto GridwiseGemmPipeline_Selector()
{
    if constexpr(EnableWaveGroup)
    {
        if constexpr(PipelineVer == PipelineVersion::v1)
        {
            if constexpr(LoopSched == LoopScheduler::Default)
            {
                return GridwiseGemmPipeline_Wavegroup_v1<NumPrefetch,
                                                         AEnableLds,
                                                         BEnableLds,
                                                         ALoadOption,
                                                         BLoadOption>{};
            }
            else
            {
#if !(defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC))
                std::cerr << "GridwiseGemmPipeline configuration is not available" << std::endl;
#endif
            }
        }
        else if constexpr(PipelineVer == PipelineVersion::v5)
        {
            return GridwiseGemmPipeline_Wavegroup_v5<NumPrefetch>{};
        }
        else
        {
#if !(defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC))
            std::cerr << "GridwiseGemmPipeline configuration is not available" << std::endl;
#endif
        }
    }
    else if constexpr(PipelineVer == PipelineVersion::v1)
    {
        if constexpr(LoopSched == LoopScheduler::Default)
        {
            return GridwiseGemmPipeline_v1<NumPrefetch,
                                           AEnableLds,
                                           BEnableLds,
                                           ALoadOption,
                                           BLoadOption>{};
        }
        else if constexpr(LoopSched == LoopScheduler::Interwave)
        {
            return GridwiseGemmPipelineInterwave_v1<NumPrefetch>{};
        }
    }
    else if constexpr(PipelineVer == PipelineVersion::v2)
    {
        return GridwiseGemmPipeline_v2{};
    }
    else if constexpr(PipelineVer == PipelineVersion::v4)
    {
        return GridwiseGemmPipeline_v4<NumPrefetch>{};
    }
    else if constexpr(PipelineVer == PipelineVersion::v5)
    {
        return GridwiseGemmPipeline_v5<NumPrefetch, ALoadOption, BLoadOption>{};
    }
    else if constexpr(PipelineVer == PipelineVersion::weight_only)
    {
        return GridwiseGemmPipeline_v1_WeightOnly<NumPrefetch, AEnableLds, BEnableLds>{};
    }
    else
    {
#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
        std::cerr << "GridwiseGemmPipeline configuration is not available" << std::endl;
#endif
    }
}

} // namespace ck
