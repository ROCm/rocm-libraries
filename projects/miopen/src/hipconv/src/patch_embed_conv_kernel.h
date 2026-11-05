#pragma once

#include "conv_kernel.h"
#include "explicit_gemm/patch_embed/patch_embed.hpp"
#include "hipconv/conv_params.hpp"

namespace hipconv
{

// Family base for patch-embedding convolutions (stride == filter, no padding).
//
// Such a layer is a GEMM with a permuted input and no data duplication, which
// this family runs as a gather followed by the shared hipBLASLt call.
class PatchEmbedConvKernel : public ConvKernel
{
public:
    using ConvKernel::ConvKernel;

    std::string_view name() const override { return "patch_embed"; }

    hipconv::Algorithm algorithm() const override { return hipconv::Algorithm::ExplicitGemm; }

    bool is_applicable(const hipconv::Device& /*dev*/,
                       const hipconv::ConvParams& par) const override
    {
        const bool ok_fp16bf16 =
            (par.input_type == DataType::fp16 || par.input_type == DataType::bf16) &&
            par.weight_type == par.input_type &&
            (par.direction == Direction::Wgrad ? par.weight_grad_type == DataType::fp32
                                               : par.output_type == par.input_type);
        // tf32 takes tf32 operands and an fp32 result, as the pointwise family does.
        const bool ok_tf32 =
            par.input_type == DataType::tf32 && par.weight_type == DataType::tf32 &&
            par.output_type == DataType::fp32 &&
            (par.direction != Direction::Wgrad || par.weight_grad_type == DataType::fp32);
        if(!ok_fp16bf16 && !ok_tf32)
            return false;
        if(par.order != TensorOrder::NHWC)
            return false;

        return explicit_gemm::is_patch_embedding(par);
    }

    size_t get_workspace_size(const hipconv::Device& /*dev*/,
                              const hipconv::ConvParams& par) const override
    {
        return explicit_gemm::patch_embed_workspace_bytes(par);
    }

    // Two launches and a round trip through the permuted A matrix, where a fused
    // kernel would need one pass; scored below the tuned 1x1 path to say so.
    float get_weighted_throughput_index(const hipconv::Device& /*dev*/,
                                        const hipconv::ConvParams& /*par*/) const override
    {
        return 0.5f;
    }
};

} // namespace hipconv
