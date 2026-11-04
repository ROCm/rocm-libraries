#pragma once

#include "hipconv/conv_params.hpp"

#include <cstddef>

typedef struct ihipStream_t* hipStream_t;

namespace hipconv::explicit_gemm
{

// Whether `par` is a non-overlapping patch convolution this path can serve.
//
// Checked by the kernel family's is_applicable; exposed here so the workspace
// query and the launch agree on one definition.
bool is_patch_embedding(const ConvParams& par);

// Whether hipBLASLt has an algorithm for the GEMM the layer reduces to; see has_algorithm.
bool patch_embed_has_algorithm(const ConvParams& par);

// Bytes of scratch the permuted A matrix needs: n*p*q rows of kh*kw*c.
std::size_t patch_embed_workspace_bytes(const ConvParams& par);

// Run the layer as a permute plus a GEMM.
void launch_patch_embed(const ConvParams& par,
                        const void* in,
                        const void* wei,
                        void* out,
                        void* workspace,
                        hipStream_t stream);

} // namespace hipconv::explicit_gemm
