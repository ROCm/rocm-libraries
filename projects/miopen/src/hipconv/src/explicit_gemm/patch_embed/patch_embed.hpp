#pragma once

#include "hipconv/conv_params.hpp"

#include <cstddef>
#include <memory>

typedef struct ihipStream_t* hipStream_t;

namespace hipconv::explicit_gemm
{

class GemmPlan;

// Whether `par` is a non-overlapping patch convolution this path can serve.
//
// Checked by the kernel family's is_applicable; exposed here so the workspace
// query and the launch agree on one definition.
bool is_patch_embedding(const ConvParams& par);

// The plan for the GEMM the layer reduces to; see make_gemm_plan.
std::shared_ptr<GemmPlan> make_patch_embed_gemm_plan(const ConvParams& par);

// Bytes of scratch the layer needs: the permuted A matrix, n*p*q rows of kh*kw*c, followed by
// the GEMM's hipBLASLt workspace.
std::size_t patch_embed_workspace_bytes(const ConvParams& par);

// Run the layer as a permute plus `plan`'s GEMM.
void launch_patch_embed(const ConvParams& par,
                        GemmPlan& plan,
                        const void* in,
                        const void* wei,
                        void* out,
                        void* workspace,
                        hipStream_t stream);

} // namespace hipconv::explicit_gemm
