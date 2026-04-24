// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/host/rotating_buffers.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1.hpp"

#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/fill.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipelines.hpp"

#include <cstddef>
#include <optional>

namespace ck_tile {
template <GemmPipeline PT, typename Problem>
struct GemmPipelineTypeSelector;

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipeline::MEMORY, Problem>
{
    using pipeline = ck_tile::GemmPipelineAgBgCrMem<Problem>;
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipeline::COMPUTE_V3, Problem>
{
    using pipeline = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipeline::COMPUTE_V4, Problem>
{
    using pipeline = ck_tile::GemmPipelineAgBgCrCompV4<Problem>;
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipeline::COMPUTE_V6, Problem>
{
    using pipeline = ck_tile::GemmPipelineAgBgCrCompV6<Problem>;
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipeline::COMPUTE_ASYNC, Problem>
{
    using pipeline = ck_tile::GemmPipelineAgBgCrCompAsync<Problem>;
};

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(size_t k, CDataType max_val)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;

    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(k);
    const auto atol =
        ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(max_val, k);
    return make_tuple(rtol, atol);
}

} // namespace ck_tile
