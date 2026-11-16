// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdDepthwise.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdDirectWmma.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdDwWmma.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdGstreamWmma.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdImplicitGemm.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdSmallWmma.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdTapOuterWmma.hpp"

/**
 * @file Gfx1250ConvFwdGeometry.hpp
 * @brief Admission and launch plan of every hipkernel:Gfx1250ConvFwdAot catalog kernel.
 *
 * A catalog kernel is parsed once from its KMD fields (InstanceGeometry) into its
 * family's kernel; refuse() and plan() then port rocKE's dispatcher
 * for it -- an instance's `_admits` / `launch_plan_for` -- and servedPlan() adds the
 * packaged contract's bounds. The grid caps in the scalar depthwise and small-C WMMA
 * refuse() are additions beyond rocKE's validators; servedPlan() checks the same grid. Pure: no hipDNN includes, so the port is unit-tested
 * against the adapter's golden vectors.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
{

class CatalogKernel
{
public:
    /// The kernel @p g describes; std::invalid_argument for one this port does not run.
    static CatalogKernel parse(const InstanceGeometry& g)
    {
        const auto family = g.algorithm();
        const auto& label = g.label();
        if(family == "implicit_gemm")
        {
            return CatalogKernel{family, ImplicitGemmKernel::parse(g)};
        }
        if(family == "direct_fwd_wmma")
        {
            return CatalogKernel{family, DirectFwdWmmaKernel::parse(g)};
        }
        if(family == "direct_fwd_gstream_wmma")
        {
            return CatalogKernel{family, GstreamWmmaKernel::parse(g)};
        }
        if(family == "direct_fwd_tapouter_wmma")
        {
            return CatalogKernel{family, TapOuterWmmaKernel::parse(g)};
        }
        if(family == "direct_fwd_smallc_wmma")
        {
            return CatalogKernel{family, SmallcWmmaKernel::parse(g)};
        }
        if(family == "direct_fwd_smallk_wmma")
        {
            return CatalogKernel{family, SmallkWmmaKernel::parse(g)};
        }
        if(family == "direct_dw_fwd_wmma")
        {
            return CatalogKernel{family, DwFwdWmmaKernel::parse(g)};
        }
        if(family == "direct_depthwise")
        {
            return CatalogKernel{family, DepthwiseKernel::parse(g)};
        }
        if(family == "direct_depthwise_tiled")
        {
            return CatalogKernel{family, DepthwiseTiledKernel::parse(g)};
        }
        if(family == "direct_depthwise_col")
        {
            return CatalogKernel{family, DepthwiseColKernel::parse(g)};
        }
        if(family == "direct_depthwise_spatial")
        {
            return CatalogKernel{family, DepthwiseSpatialKernel::parse(g)};
        }
        throw std::invalid_argument(label + ": unknown kernel_family " + family);
    }

    const std::string& family() const
    {
        return _family;
    }

    /// The implicit-GEMM kernel, or nullptr for a direct one.
    const ImplicitGemmKernel* implicitGemm() const
    {
        return std::get_if<ImplicitGemmKernel>(&_kernel);
    }

    /// Why this kernel cannot run @p problem; empty when it can.
    Refusal refuse(const Problem& problem) const
    {
        return std::visit(
            [&](const auto& kernel) { return gfx1250_conv_fwd::refuse(kernel, problem); }, _kernel);
    }

    /// The launch plan of an admitted @p problem (refuse() returned empty).
    Plan plan(const Problem& problem) const
    {
        return std::visit(
            [&](const auto& kernel) { return gfx1250_conv_fwd::plan(kernel, problem); }, _kernel);
    }

private:
    using Variant = std::variant<ImplicitGemmKernel,
                                 DirectFwdWmmaKernel,
                                 GstreamWmmaKernel,
                                 TapOuterWmmaKernel,
                                 SmallcWmmaKernel,
                                 SmallkWmmaKernel,
                                 DwFwdWmmaKernel,
                                 DepthwiseKernel,
                                 DepthwiseTiledKernel,
                                 DepthwiseColKernel,
                                 DepthwiseSpatialKernel>;

    CatalogKernel(std::string family, Variant kernel)
        : _family(std::move(family))
        , _kernel(std::move(kernel))
    {
    }

    std::string _family;
    Variant _kernel;
};

/// The bytes of an fp16/bf16 tensor of @p dims elements fit the kernels' i32 byte-size and
/// offset arguments.
inline bool fitsI32Bytes(std::initializer_list<int64_t> dims)
{
    int64_t bytes = 2;
    for(const auto extent : dims)
    {
        if(extent < 1 || extent > I32_MAX / bytes)
        {
            return false;
        }
        bytes *= extent;
    }
    return true;
}

/// The plan @p kernel runs @p problem with, or std::nullopt when it does not serve it:
/// conv_fwd_aot_serves and ConvFwdAotSpec.launch_plan of the packaging adapter
/// (rocke/library/builders/common/convolution_forward.py). Every tensor's bytes fit the i32
/// launch ABI, the kernel admits the problem (refuse()), and a direct kernel's launch fits:
/// grid x one 32-bit dispatch, y and z 65535.
inline std::optional<Plan> servedPlan(const CatalogKernel& kernel, const Problem& problem)
{
    if(!fitsI32Bytes({problem.n, problem.h, problem.w, problem.c})
       || !fitsI32Bytes({problem.k, problem.y, problem.x, problem.cpg()})
       || !fitsI32Bytes({problem.n, problem.ho(), problem.wo(), problem.k})
       || kernel.refuse(problem).has_value())
    {
        return std::nullopt;
    }
    auto plan = kernel.plan(problem);
    if(kernel.family() != "implicit_gemm")
    {
        for(const auto& step : plan.steps)
        {
            if(step.grid[0] > I32_MAX || step.grid[0] * step.block[0] > MAX_DISPATCH_ITEMS
               || step.grid[1] > MAX_GRID_DIM || step.grid[2] > MAX_GRID_DIM)
            {
                return std::nullopt;
            }
        }
    }
    return plan;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
