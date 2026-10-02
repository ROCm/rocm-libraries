#include "explicit_gemm/patch_embed/patch_embed.hpp"

#include "conv_kernel_table.h"
#include "launch_params.h"
#include "patch_embed_conv_kernel.h"

#include "hip_util.h"
#include "mathutil.h"
#include "explicit_gemm/hipblaslt_matmul.hpp"

#include <hip/hip_runtime.h>

#include <array>

namespace hipconv::explicit_gemm
{
namespace
{

// Elements per copy, picked per shape by copy_vec below. Both dtypes this path serves are
// 2 bytes wide, so VEC 8 is the 16-byte copy this kernel wants and 1 is the scalar fallback.
template <int VEC>
struct copy_unit;
template <>
struct copy_unit<8>
{
    using type = uint4;
};
template <>
struct copy_unit<4>
{
    using type = uint2;
};
template <>
struct copy_unit<2>
{
    using type = unsigned int;
};
template <>
struct copy_unit<1>
{
    using type = unsigned short;
};

// The widest copy this shape admits.
//
// A copy may not straddle a filter row, and every term of the X offset has to be a whole
// number of them: the row origin carries q_i*kw*c and the filter row r carries r*w*c. A
// patch whose row is not 8-aligned -- 14x14 over 3 channels gives 42 -- drops to a narrower
// unit rather than being turned away, which is the only thing that serves ViT-L/14.
inline int copy_vec(const ConvParams& par)
{
    for(int v : {8, 4, 2})
        if((par.kw * par.c) % v == 0 && (par.w * par.c) % v == 0)
            return v;
    return 1;
}

// The correspondence between X (NHWC) and the GEMM's A matrix.
//
// A patch convolution whose stride equals its filter reads every input pixel
// exactly once, so im2col is a pure permutation:
//   A[(n*p + p_i)*q + q_i][(r*kw + s)*c + ch] = X[n][p_i*kh + r][q_i*kw + s][ch]
// The A column index is already KRSC's element order, so the weights need no
// reordering and the layer becomes a 1x1 conv over kh*kw*c channels.
struct PatchMap
{
    int rows;
    int chunks_per_row;
    int kgemm;
    int h, w, c;
    int p, q;
    int kh, kw;
};

PatchMap make_patch_map(const ConvParams& par)
{
    const int kgemm = par.kh * par.kw * par.c;
    return PatchMap{par.n * par.p * par.q,
                    kgemm / copy_vec(par),
                    kgemm,
                    par.h,
                    par.w,
                    par.c,
                    par.p,
                    par.q,
                    par.kh,
                    par.kw};
}

// Copy one 16-byte chunk per thread between X and A.
//
// The A side is walked contiguously so those accesses coalesce; the X side is
// the strided one, but a block's threads stay inside one patch row-strip, whose
// lines it therefore still touches once each.
template <bool ToMatrix, int VEC>
__global__ __launch_bounds__(256) void patch_permute(PatchMap m,
                                                     const unsigned short* __restrict__ src,
                                                     unsigned short* __restrict__ dst)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= m.rows * m.chunks_per_row)
        return;

    const int row = gid / m.chunks_per_row;
    const int k0  = (gid - row * m.chunks_per_row) * VEC;

    const int patch_row = m.kw * m.c;
    const int r         = k0 / patch_row;
    const int within_r  = k0 - r * patch_row;

    const int q_i     = row % m.q;
    const int batched = row / m.q;
    const int p_i     = batched % m.p;
    const int n_i     = batched / m.p;

    const size_t x_off =
        ((static_cast<size_t>(n_i) * m.h + p_i * m.kh + r) * m.w + q_i * m.kw) * m.c + within_r;
    const size_t a_off = static_cast<size_t>(row) * m.kgemm + k0;

    const size_t from = ToMatrix ? x_off : a_off;
    const size_t to   = ToMatrix ? a_off : x_off;

    using unit_t                         = typename copy_unit<VEC>::type;
    *reinterpret_cast<unit_t*>(dst + to) = *reinterpret_cast<const unit_t*>(src + from);
}

// `par` rewritten as the 1x1 conv over kh*kw*c channels that the GEMM runs.
ConvParams gemm_params(const ConvParams& par)
{
    ConvParams gemm = par;
    gemm.h = gemm.p = par.p;
    gemm.w = gemm.q = par.q;
    gemm.c          = par.kh * par.kw * par.c;
    gemm.kh = gemm.kw = 1;
    gemm.pad_h = gemm.pad_w = 0;
    gemm.stride_h = gemm.stride_w = 1;
    return gemm;
}

void launch_permute(bool to_matrix,
                    int vec,
                    const PatchMap& map,
                    const void* src,
                    void* dst,
                    hipStream_t stream)
{
    constexpr int BLOCK = 256;
    const int blocks    = divup(map.rows * map.chunks_per_row, BLOCK);
    const auto* in      = static_cast<const unsigned short*>(src);
    auto* out           = static_cast<unsigned short*>(dst);

    auto dispatch = [&]<int VEC>() {
        if(to_matrix)
            patch_permute<true, VEC><<<blocks, BLOCK, 0, stream>>>(map, in, out);
        else
            patch_permute<false, VEC><<<blocks, BLOCK, 0, stream>>>(map, in, out);
    };

    switch(vec)
    {
    case 8:
        dispatch.template operator()<8>();
        break;
    case 4:
        dispatch.template operator()<4>();
        break;
    case 2:
        dispatch.template operator()<2>();
        break;
    default:
        dispatch.template operator()<1>();
        break;
    }
}

} // namespace

bool is_patch_embedding(const ConvParams& par)
{
    if(par.kh == 1 && par.kw == 1)
        return false; // the 1x1 family already serves these
    if(par.kh != par.stride_h || par.kw != par.stride_w)
        return false;
    if(par.pad_h != 0 || par.pad_w != 0)
        return false;
    if(par.dilation_h != 1 || par.dilation_w != 1)
        return false;
    return par.groups == 1;
}

std::size_t patch_embed_workspace_bytes(const ConvParams& par)
{
    const std::size_t rows  = static_cast<std::size_t>(par.n) * par.p * par.q;
    const std::size_t kgemm = static_cast<std::size_t>(par.kh) * par.kw * par.c;
    return rows * kgemm * sizeof_data_type(par.input_type);
}

void launch_patch_embed(const ConvParams& par,
                        const void* in,
                        const void* wei,
                        void* out,
                        void* workspace,
                        hipStream_t stream)
{
    const PatchMap map    = make_patch_map(par);
    const ConvParams gemm = gemm_params(par);

    if(par.direction == Direction::Dgrad)
    {
        // dA[M, kh*kw*c] = dY[M, K] * W[K, kh*kw*c], then scatter dA over dX.
        // Non-overlapping patches make the scatter a permutation, so no
        // accumulation is needed on the way out -- but it covers only the patches, and the
        // strip past the last whole one is zero.
        if(par.h % par.kh != 0 || par.w % par.kw != 0)
            HIP_CHECK(hipMemsetAsync(out, 0, ConvSize(par).input_grad_bytes(), stream));
        launch_gemm(gemm, in, wei, workspace, stream);
        launch_permute(false, copy_vec(par), map, workspace, out, stream);
        return;
    }

    // Fprop multiplies A by the weights; Wgrad contracts A against dY. Both
    // read X only through A, so they share the gather.
    launch_permute(true, copy_vec(par), map, in, workspace, stream);
    launch_gemm(gemm, workspace, wei, out, stream);
}

} // namespace hipconv::explicit_gemm

namespace hipconv::patch_embed_fp16bf16
{

void launch_impl(const LaunchParams&,
                 const ConvParams& par,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* workspace,
                 hipStream_t stream)
{
    hipconv::explicit_gemm::launch_patch_embed(par, in, wei, out, workspace, stream);
}

// One stateless implementation, like the 1x1 family: the permute reads its
// geometry from `par` and the GEMM dispatches inside hipBLASLt, so there is no
// per-config state to enumerate.
class PatchEmbedConvKernelImpl : public PatchEmbedConvKernel
{
public:
    constexpr PatchEmbedConvKernelImpl() : PatchEmbedConvKernel(&launch_impl) {}

    bool is_valid_config(const ConvParams&) const override { return true; }

    LaunchParams get_launch_params(const ConvParams&) const override { return LaunchParams{}; }
};

PatchEmbedConvKernelImpl kernel;
std::array<ConvKernel*, 1> kernel_ptrs = {&kernel};

} // namespace hipconv::patch_embed_fp16bf16

HIPCONV_EXPORT_KERNEL_TABLE(patch_embed_fp16bf16_kernels, hipconv::patch_embed_fp16bf16);
