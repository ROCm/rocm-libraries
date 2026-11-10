// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

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
 * for it -- an instance's `_admits` / `launch_plan_for`. Pure: no hipDNN includes, so the
 * port is unit-tested against the dispatcher's golden vectors.
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

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
