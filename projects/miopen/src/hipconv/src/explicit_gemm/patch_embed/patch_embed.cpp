#include "explicit_gemm/patch_embed/patch_embed.hpp"

#include "launch_params.h"
#include "patch_embed_conv_kernel.h"

#include "hip_util.h"
#include "mathutil.h"
#include "explicit_gemm/hipblaslt_matmul.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace hipconv::explicit_gemm
{
namespace
{

// The unit one copy moves, by its width in bytes: 16 is the copy this kernel wants, and the
// element itself -- 2 bytes for fp16 and bf16, 4 for tf32 -- is the scalar fallback.
template <int BYTES>
struct copy_unit;
template <>
struct copy_unit<16>
{
    using type = uint4;
};
template <>
struct copy_unit<8>
{
    using type = uint2;
};
template <>
struct copy_unit<4>
{
    using type = unsigned int;
};
template <>
struct copy_unit<2>
{
    using type = unsigned short;
};

// The widest copy this shape admits, in elements: at most 16 bytes of them.
//
// A copy may not straddle a filter row, and every term of the X offset has to be a whole
// number of them: the row origin carries q_i*kw*c and the filter row r carries r*w*c. A
// patch whose row is not 8-aligned -- 14x14 over 3 channels gives 42 -- drops to a narrower
// unit rather than being turned away, which is the only thing that serves ViT-L/14.
inline int copy_vec(const ConvParams& par)
{
    const int widest = 16 / static_cast<int>(sizeof_data_type(par.input_type));
    for(int v = widest; v > 1; v /= 2)
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
    uint32_t rows;
    uint32_t chunks_per_row;
    uint32_t kgemm;
    uint32_t h, w, c;
    uint32_t p, q;
    uint32_t kh, kw;
};

PatchMap make_patch_map(const ConvParams& par)
{
    const auto u         = [](int v) { return static_cast<uint32_t>(v); };
    const uint32_t kgemm = u(par.kh) * u(par.kw) * u(par.c);
    return PatchMap{u(par.n) * u(par.p) * u(par.q),
                    kgemm / u(copy_vec(par)),
                    kgemm,
                    u(par.h),
                    u(par.w),
                    u(par.c),
                    u(par.p),
                    u(par.q),
                    u(par.kh),
                    u(par.kw)};
}

// Copy one chunk of VEC elements per thread between X and A, over the `copies` chunks that
// start at row `row0` of A.
//
// The A side is walked contiguously so those accesses coalesce; the X side is
// the strided one, but a block's threads stay inside one patch row-strip, whose
// lines it therefore still touches once each.
//
// Every index is unsigned: signed division would spend instructions on signs these never have,
// and a copy as narrow as 2 or 4 bytes is bound by those instructions rather than by memory.
template <bool ToMatrix, int VEC, typename Elem>
__global__ __launch_bounds__(256) void patch_permute(PatchMap m,
                                                     uint32_t row0,
                                                     uint32_t copies,
                                                     const Elem* __restrict__ src,
                                                     Elem* __restrict__ dst)
{
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= copies)
        return;

    const uint32_t slab_row = gid / m.chunks_per_row;
    const uint32_t row      = row0 + slab_row;
    const uint32_t k0       = (gid - slab_row * m.chunks_per_row) * VEC;

    const uint32_t patch_row = m.kw * m.c;
    const uint32_t r         = k0 / patch_row;
    const uint32_t within_r  = k0 - r * patch_row;

    const uint32_t q_i     = row % m.q;
    const uint32_t batched = row / m.q;
    const uint32_t p_i     = batched % m.p;
    const uint32_t n_i     = batched / m.p;

    const size_t x_off =
        ((static_cast<size_t>(n_i) * m.h + p_i * m.kh + r) * m.w + q_i * m.kw) * m.c + within_r;
    const size_t a_off = static_cast<size_t>(row) * m.kgemm + k0;

    const size_t from = ToMatrix ? x_off : a_off;
    const size_t to   = ToMatrix ? a_off : x_off;

    using unit_t = typename copy_unit<VEC* static_cast<int>(sizeof(Elem))>::type;
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

// Offset of the GEMM's workspace past the permuted A matrix; hipblasLtMatmul requires a 16-byte
// aligned workspace (hipblaslt.h).
std::size_t gemm_workspace_offset(const ConvParams& par)
{
    constexpr std::size_t align = 16;
    const std::size_t rows      = static_cast<std::size_t>(par.n) * par.p * par.q;
    const std::size_t kgemm     = static_cast<std::size_t>(par.kh) * par.kw * par.c;
    const std::size_t a_bytes   = rows * kgemm * sizeof_data_type(par.input_type);
    return (a_bytes + align - 1) / align * align;
}

template <typename Elem>
void launch_permute_typed(bool to_matrix,
                          int vec,
                          const PatchMap& map,
                          const void* src,
                          void* dst,
                          hipStream_t stream)
{
    constexpr uint32_t BLOCK = 256;
    // A launch holds at most 2^32 - 1 threads, so a map past that takes several, each a slab of
    // whole rows.
    constexpr uint32_t max_copies = std::numeric_limits<uint32_t>::max() / BLOCK * BLOCK;
    const uint32_t slab_rows      = max_copies / map.chunks_per_row;
    const auto* in                = static_cast<const Elem*>(src);
    auto* out                     = static_cast<Elem*>(dst);

    auto dispatch = [&]<int VEC>() {
        for(uint32_t row0 = 0; row0 < map.rows;)
        {
            const uint32_t rows   = std::min(slab_rows, map.rows - row0);
            const uint32_t copies = rows * map.chunks_per_row;
            const uint32_t blocks = divup(copies, BLOCK);
            if(to_matrix)
                patch_permute<true, VEC><<<blocks, BLOCK, 0, stream>>>(map, row0, copies, in, out);
            else
                patch_permute<false, VEC><<<blocks, BLOCK, 0, stream>>>(map, row0, copies, in, out);
            row0 += rows;
        }
    };

    constexpr int ELEM = sizeof(Elem);
    switch(vec * ELEM)
    {
    case 16:
        dispatch.template operator()<16 / ELEM>();
        break;
    case 8:
        dispatch.template operator()<8 / ELEM>();
        break;
    case 4:
        dispatch.template operator()<4 / ELEM>();
        break;
    default:
        dispatch.template operator()<1>();
        break;
    }
}

void launch_permute(bool to_matrix,
                    int vec,
                    const PatchMap& map,
                    DataType dtype,
                    const void* src,
                    void* dst,
                    hipStream_t stream)
{
    if(sizeof_data_type(dtype) == 4)
        launch_permute_typed<unsigned int>(to_matrix, vec, map, src, dst, stream);
    else
        launch_permute_typed<unsigned short>(to_matrix, vec, map, src, dst, stream);
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
    if(par.groups != 1)
        return false;
    // PatchMap counts A's rows in 32 bits, and the GEMM takes kh*kw*c as an int channel count.
    return static_cast<uint64_t>(par.n) * par.p * par.q <= UINT32_MAX &&
           static_cast<int64_t>(par.kh) * par.kw * par.c <= INT32_MAX;
}

std::shared_ptr<GemmPlan> make_patch_embed_gemm_plan(const ConvParams& par)
{
    return make_gemm_plan(gemm_params(par));
}

std::size_t patch_embed_workspace_bytes(const ConvParams& par)
{
    return gemm_workspace_offset(par) + gemm_workspace_bytes(par.direction);
}

void launch_patch_embed(const ConvParams& par,
                        GemmPlan& plan,
                        const void* in,
                        const void* wei,
                        void* out,
                        void* workspace,
                        hipStream_t stream)
{
    if(workspace == nullptr)
        throw HipError(hipErrorInvalidValue, "patch embedding needs a workspace");
    const PatchMap map   = make_patch_map(par);
    void* gemm_workspace = static_cast<char*>(workspace) + gemm_workspace_offset(par);

    if(par.direction == Direction::Dgrad)
    {
        // dA[M, kh*kw*c] = dY[M, K] * W[K, kh*kw*c], then scatter dA over dX.
        // Non-overlapping patches make the scatter a permutation, so no
        // accumulation is needed on the way out -- but it covers only the patches, and the
        // strip past the last whole one is zero.
        if(par.h % par.kh != 0 || par.w % par.kw != 0)
            HIP_CHECK(hipMemsetAsync(out, 0, ConvSize(par).input_grad_bytes(), stream));
        launch_gemm(plan, in, wei, workspace, gemm_workspace, stream);
        launch_permute(false, copy_vec(par), map, par.input_type, workspace, out, stream);
        return;
    }

    // Fprop multiplies A by the weights; Wgrad contracts A against dY. Both
    // read X only through A, so they share the gather.
    launch_permute(true, copy_vec(par), map, par.input_type, in, workspace, stream);
    launch_gemm(plan, workspace, wei, out, gemm_workspace, stream);
}

} // namespace hipconv::explicit_gemm

namespace hipconv::patch_embed_kernel
{

void launch_impl(const LaunchParams& lp,
                 const Device&,
                 const ConvParams& par,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* workspace,
                 hipStream_t stream)
{
    hipconv::explicit_gemm::launch_patch_embed(par, *lp.gemm_plan, in, wei, out, workspace, stream);
}

// One stateless implementation, like the 1x1 family: the permute reads its
// geometry from `par` and the GEMM dispatches inside hipBLASLt, so there is no
// per-config state to enumerate.
class PatchEmbedConvKernelImpl : public PatchEmbedConvKernel
{
public:
    constexpr PatchEmbedConvKernelImpl() : PatchEmbedConvKernel(&launch_impl) {}

    bool is_valid_config(const Device& /*dev*/, const ConvParams&) const override { return true; }

    LaunchParams get_launch_params(const Device& /*dev*/, const ConvParams& par) const override
    {
        LaunchParams lp;
        lp.gemm_plan = explicit_gemm::make_patch_embed_gemm_plan(par);
        return lp;
    }
};

PatchEmbedConvKernelImpl kernel;
std::array<ConvKernel*, 1> kernel_ptrs = {&kernel};

// Only a single kernel in the patch_embed family with empty config representation
auto get_kernel_by_config(RepresentationHashPair const& cfg) -> hipconv::ConvKernel*
{
    return cfg.repr().empty() ? &kernel : nullptr;
}

} // namespace hipconv::patch_embed_kernel

extern const hipconv::ConvKernelGroup patch_embed_kernels = {
    hipconv::patch_embed_kernel::kernel_ptrs,
    hipconv::patch_embed_kernel::get_kernel_by_config};
