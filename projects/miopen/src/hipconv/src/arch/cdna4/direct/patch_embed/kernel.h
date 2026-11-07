#pragma once

// Fused patch-embedding convolution: a permuting load feeding one MFMA GEMM.
//
// See docs/algorithms/direct/patch-embed-cdna4.md.

#include "config.h"
#include "bunnies_cdna4.hpp"
#include "config_desc.h"
#include "config_table.h"
#include "direct_conv_kernel.h"
#include "grouped/mfma_dispatch.h"
#include "detail.h"
#include "lds_layout.h"

#include <algorithm>
#include <cmath>
#include "hip_util.h"
#include "launch_params.h"
#include "magic_division.h"
#include "memory.h"
#include "packed_ops.h"
#include "persistent_grid.h"
#include "types.h"
#include "hipconv/conv_params.hpp"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <type_traits>

namespace hipconv::cdna4::patch_embed
{

// The bunnies format tag the MFMA computes in: a 16-bit operand type as it is, tf32 as the bf16
// big/small split it computes as.
template <DataType DT>
constexpr bunnies::fpfmt operand_fmt = DT == DataType::tf32   ? bunnies::fpfmt::e8m10_e8m7x2split
                                       : DT == DataType::bf16 ? bunnies::fpfmt::e8m7
                                                              : bunnies::fpfmt::e5m10;


// The layer geometry the kernel indexes with, flattened to what the loops need.
struct LayerPars
{
    int h, w, c;
    int kh, kw;
    int sh, sw; // stride, at least the filter: the patches never overlap
    int p, q;
    int k;               // output channels
    int kgemm;           // kh * kw * c, the GEMM's reduction depth
    int rows;            // n * p * q, the GEMM's M
    int patch_row;       // kw * c, the K span of one filter row
    int kpad_row;        // patch_row rounded up to a granule, when the config pads rows
    int64_t input_elems; // n * h * w * c
    int split_rows;      // wgrad: the reduction rows one split walks, a whole chunk pair
    MagicDiv q_div;      // wgrad decodes a row per staged piece, so its divides go magic
    MagicDiv p_div;
    int tiles_n; // fprop/dgrad: N tiles per M tile in the flattened grid
};

// Whether a staged granule of the A matrix stays inside one filter row.
//
// Not an alignment test: filter rows are w*c apart, so a piece may not straddle two.
inline bool addressing_fits(const Config& cfg, const ConvParams& par)
{
    // The padding config pulls a row's last granule back to end on the row's last element, so
    // it fits any row; one shorter than a granule starts before it, see SHORT_ROW.
    return cfg.row_pad || (par.kw * par.c) % GRANULE == 0;
}

// Wgrad reduces over the spatial rows and leaves only the small dW to tile, which cannot
// fill the machine; the rows split across blockIdx.z instead, and each split adds its share
// into dW. A split is a whole number of chunk pairs, so the walk needs no tail.
//
// A padding config lays each filter row out at a whole number of granules, so wgrad's M axis
// -- the filter, not a reduction here -- is kh rows of that padded width.
inline int wgrad_filter_axis(const Config& cfg, const ConvParams& par)
{
    return par.kh * cfg.padded_row(par.kw * par.c);
}

// Workgroups a CU holds at once: LDS or the accumulator's registers bound it, and past three
// the extra splits' atomics cost more than the occupancy returns.
inline int wgrad_slots(const Config& cfg)
{
    const int lds  = 160 * 1024 / cfg.lds_bytes();
    const int regs = 512 / (cfg.acc_vgprs() + 96) / std::max(1, cfg.waves() / 4);
    return std::clamp(std::min(lds, regs), 1, 3);
}

// The most rows one workgroup's range may cover. Its descriptors rebase on the first image and
// row it touches, so only that range -- the images it reaches in X, the rows it reaches in Y or
// dY -- has to fit the 32-bit offsets; a range starting mid-image reaches into one more.
inline int64_t span_rows(const ConvParams& par)
{
    const int64_t limit = INT32_MAX / static_cast<int64_t>(sizeof_data_type(par.input_type));
    const int64_t image = static_cast<int64_t>(par.h) * par.w * par.c;
    const int64_t by_x  = (limit / image - 2) * par.p * par.q;
    const int64_t by_r  = limit / std::max(par.k, par.kh * par.kw * par.c);
    return std::min(by_x, by_r);
}

inline int wgrad_split_rows(const Config& cfg, const ConvParams& par, int cu_count)
{
    const int pair = 2 * cfg.k_chunk;
    const int rows = par.n * par.p * par.q;
    const int axis = wgrad_filter_axis(cfg, par);
    const int tiles =
        ((axis + cfg.m_tile() - 1) / cfg.m_tile()) * ((par.k + cfg.n_tile() - 1) / cfg.n_tile());
    const int pairs = (rows + pair - 1) / pair;
    // Rounded down, so every split is resident at once: one left over waits a whole walk.
    // Never so few that a split's rows outgrow what one rebased descriptor reaches.
    const int most = static_cast<int>(std::max<int64_t>(1, span_rows(par) / pair));
    const int splits =
        std::clamp(cu_count * wgrad_slots(cfg) / tiles, (pairs + most - 1) / most, pairs);
    return (pairs + splits - 1) / splits * pair;
}

inline int wgrad_splits(const Config& cfg, const ConvParams& par, int cu_count)
{
    const int rows = par.n * par.p * par.q;
    const int per  = wgrad_split_rows(cfg, par, cu_count);
    return (rows + per - 1) / per;
}

inline LayerPars layer_pars(const Config& cfg, const ConvParams& par, int cu_count)
{
    return LayerPars{
        par.h,
        par.w,
        par.c,
        par.kh,
        par.kw,
        par.stride_h,
        par.stride_w,
        par.p,
        par.q,
        par.k,
        par.kh * par.kw * par.c,
        par.n * par.p * par.q,
        par.kw * par.c,
        cfg.padded_row(par.kw * par.c),
        static_cast<int64_t>(par.n) * par.h * par.w * par.c,
        par.direction == Direction::Wgrad ? wgrad_split_rows(cfg, par, cu_count) : 0,
        MagicDiv(static_cast<uint32_t>(par.q)),
        MagicDiv(static_cast<uint32_t>(par.p)),
        ((par.direction == Direction::Dgrad ? par.kh * par.kw * par.c : par.k) + cfg.n_tile() - 1) /
            cfg.n_tile()};
}

namespace detail
{

// The slot a row stores granule g at, XOR-swizzled so a 16-lane MFMA read spreads over
// every granule of the tile instead of hitting one bank column; see GranuleTile.
template <Config cfg>
__device__ inline int swizzled(int row, int g)
{
    return GranuleTile<cfg.granules()>::slot(row, g);
}

// The element offset of a (row, granule) piece in either LDS tile.
template <Config cfg>
__device__ inline int lds_offset(int row, int g)
{
    return row * cfg.k_chunk + swizzled<cfg>(row, g) * GRANULE;
}

// The input element offset an A row reads from: the (n_i, p_i, q_i) it decodes to.
//
// A row past the problem gets an offset past the buffer, which the descriptor zero-fills.
// img0 is the image the caller's descriptor starts on.
__device__ inline int row_origin(const LayerPars& pars, int m, int oob, int img0)
{
    if(m >= pars.rows)
        return oob;
    const int q_i  = m % pars.q;
    const int rest = m / pars.q;
    const int p_i  = rest % pars.p;
    const int n_i  = rest / pars.p - img0;
    return ((n_i * pars.h + p_i * pars.sh) * pars.w + q_i * pars.sw) * pars.c;
}

// row_origin for a row known to be in range, with the two divides magic.
__device__ inline int row_origin_magic(const LayerPars& pars, int m, int img0)
{
    const int mq  = static_cast<int>(pars.q_div.div(static_cast<uint32_t>(m)));
    const int q_i = m - mq * pars.q;
    const int n_i = static_cast<int>(pars.p_div.div(static_cast<uint32_t>(mq)));
    const int p_i = mq - n_i * pars.p;
    return (((n_i - img0) * pars.h + p_i * pars.sh) * pars.w + q_i * pars.sw) * pars.c;
}

// The logical granule a staging slot fetches, since a DTL load's LDS slot is lane order and not
// the loader's to pick -- so the swizzle has to move to the global address. The register split
// fills each plane at the slot the copy would have, so tf32 shares it.
template <Config cfg>
__device__ inline int granule_for_slot(int row, int slot_g)
{
    return GranuleTile<cfg.granules()>::slot(row, slot_g);
}

template <Config cfg>
using TransposeB = TransposeTile<cfg.n_tile()>;

// The dX element that dgrad's result for (row m, filter element j) belongs to.
//
// The inverse of row_origin's gather, and a permutation because the patches do not overlap.
// A whole filter row is contiguous here, which is what lets an aligned row store eight wide.
//
// Written as the row's patch origin plus the filter row's step across the image plus j, rather
// than by decoding (n_i, p_i, q_i, r, s, ch): the two forms agree, and this one divides three
// times where that one divides five, on an axis the epilogue walks per element. What is left
// depends on m alone, so it hoists out of the column loop.
__device__ inline int patch_offset(const LayerPars& pars, int m, int j)
{
    return row_origin(pars, m, 0, 0) + (j / pars.patch_row) * (pars.w - pars.kw) * pars.c + j;
}

} // namespace detail

// Fprop computes Y = A(X) * W; dgrad computes dA = dY * W and scatters dA over dX; wgrad
// computes one split of dW = dY^T * A(X) and adds it into the fp32 dW.
//
// Both walk the same staged GEMM, so the pipeline and the MFMA loop are shared. They differ
// in which side permutes -- fprop on the way in, dgrad on the way out -- and in the
// reduction axis, which is kh*kw*c for fprop, the output channels for dgrad, and the spatial
// rows for wgrad -- whose tiles are the filter axis (M) and the output channels (N).
//
// NARROW_K names a k the granule does not divide. It is a template parameter rather than a branch
// so the kernel that serves every other k compiles exactly as if it were not there: as a runtime
// test it cost the 16x16 stem's dgrad 13%.
//
// SHORT names a filter row shorter than a granule, on the same terms: compiled into every padded
// kernel, its walk cost the 8x8 and 14x14 stems 2-5%.
template <Config cfg, DataType DT, Direction DIR, bool NARROW_K, bool SHORT>
__device__ void patch_embed_impl(LayerPars pars,
                                 const ToType<DT>* __restrict__ in_all,
                                 const ToType<DT>* __restrict__ wei_all,
                                 void* __restrict__ out_raw)
{
    using dtype = ToType<DT>;
    using vec_t = std::conditional_t<DT == DataType::bf16, bf16x8_t, fp16x8_t>;

    constexpr bool is_dgrad = DIR == Direction::Dgrad;
    constexpr bool is_wgrad = DIR == Direction::Wgrad;
    constexpr bool is_tf32  = DT == DataType::tf32;
    static_assert(is_tf32 == cfg.is_tf32(), "a config is compiled for one element width");

    using arch = bunnies::arch_cdna4;
    using dtl  = arch::buffer_load_lds<16>;

    // Filters take the A role and spatial rows the B role: the accumulator hands a lane
    // four contiguous values of A's axis, which is what the epilogue's wide store needs.
    using mat_filt = arch::matrix<operand_fmt<DT>, MFMA_N, MFMA_K, bunnies::use::A>;
    using mat_spat = arch::matrix<operand_fmt<DT>, MFMA_K, MFMA_M, bunnies::use::B>;
    using mat_acc  = arch::matrix<bunnies::fpfmt::e8m23, MFMA_N, MFMA_M, bunnies::use::Acc>;
    using rt_filt  = bunnies::reg_tile<mat_filt, cfg.wave_n16(), 1>;
    using rt_spat  = bunnies::reg_tile<mat_spat, 1, cfg.wave_m16()>;
    using rt_acc   = bunnies::reg_tile<mat_acc, cfg.wave_n16(), cfg.wave_m16()>;
    using mat_out  = arch::matrix<operand_fmt<DT>, MFMA_N, MFMA_M, bunnies::use::Acc>;
    using rt_out   = bunnies::reg_tile<mat_out, cfg.wave_n16(), cfg.wave_m16()>;

    // A tf32 plane is read as the bf16 matrix it is, into one half of the split.
    using plane_filt = arch::matrix<bunnies::fpfmt::e8m7, MFMA_N, MFMA_K, bunnies::use::A>;
    using plane_spat = arch::matrix<bunnies::fpfmt::e8m7, MFMA_K, MFMA_M, bunnies::use::B>;

    constexpr int granules   = cfg.granules();
    constexpr int elem_bytes = sizeof(dtype);

    // What a tile holds: the operand as staged, or tf32's two bf16 planes back to back.
    using lds_t          = std::conditional_t<is_tf32, bf16_t, dtype>;
    constexpr int PLANES = is_tf32 ? 2 : 1;
    constexpr int A_TILE = cfg.m_tile() * cfg.k_chunk;
    constexpr int B_TILE = cfg.n_tile() * cfg.k_chunk;

    // Four declarations, not two arrays of two: a partial vmcnt needs the waitcnt pass to
    // tell a read's object from the in-flight DMA's, and one indexed array is one object.
    __shared__ lds_t a_lds0[PLANES * A_TILE];
    __shared__ lds_t a_lds1[PLANES * A_TILE];
    __shared__ lds_t b_lds0[PLANES * B_TILE];
    __shared__ lds_t b_lds1[PLANES * B_TILE];

    auto a_buf = [&]<int B>() -> lds_t* { return B ? a_lds1 : a_lds0; };
    auto b_buf = [&]<int B>() -> lds_t* { return B ? b_lds1 : b_lds0; };

    // Fprop and dgrad walk the grid with N fastest, so the N tiles of one M tile run together
    // and the M-side operand they share -- X, or dY -- leaves HBM once instead of once per N
    // tile. With M outermost, each re-read came back from DRAM once that operand outgrew MALL.
    //
    // Consecutive workgroups go round-robin to the XCDs, each with its own L2, so the tile
    // order is also regrouped to give every XCD a contiguous run: the N tiles of an M tile then
    // share one L2 instead of reading the same A rows into several.
    int m_idx = static_cast<int>(blockIdx.x);
    int n_idx = static_cast<int>(blockIdx.y);
    if constexpr(!is_wgrad)
    {
        constexpr int XCDS = persistent::NUM_XCD;
        const int block    = static_cast<int>(blockIdx.x);
        const int blocks   = static_cast<int>(gridDim.x);
        const int xcd      = block % XCDS;
        const int local    = block / XCDS;
        const int per      = blocks / XCDS;
        const int extra    = blocks % XCDS; // the first `extra` XCDs take one more
        const int tile =
            xcd < extra ? xcd * (per + 1) + local : extra * (per + 1) + (xcd - extra) * per + local;
        m_idx = tile / pars.tiles_n;
        n_idx = tile - m_idx * pars.tiles_n;
    }
    const int m0  = m_idx * cfg.m_tile();
    const int n0  = n_idx * cfg.n_tile();
    const int tid = threadIdx.x;
    // The first reduction row this workgroup's split walks.
    const int kbase = is_wgrad ? static_cast<int>(blockIdx.z) * pars.split_rows : 0;

    // Every tensor the rows index is rebased on the first image and row this workgroup touches,
    // so offsets span its own range rather than the batch: X and dX by image, Y and dY by row.
    const int row0        = is_wgrad ? kbase : m0;
    const int img0        = row0 / (pars.p * pars.q);
    const int64_t x_base  = static_cast<int64_t>(img0) * pars.h * pars.w * pars.c;
    const int64_t r_base  = static_cast<int64_t>(row0) * pars.k;
    const int64_t r_total = static_cast<int64_t>(pars.rows) * pars.k;
    auto span             = [](int64_t n) {
        return static_cast<int>(std::min<int64_t>(n, INT32_MAX / elem_bytes));
    };
    const dtype* const in  = in_all + (is_dgrad ? r_base : x_base);
    const dtype* const wei = wei_all + (is_wgrad ? r_base : 0);
    // Wgrad writes the fp32 dW through out_raw; the other two write their own dtype here.
    dtype* const out = static_cast<dtype*>(out_raw) + (is_dgrad ? x_base : r_base);

    // Buffer descriptors give the staging loads hardware range checking, so an out-of-range
    // row nulls through its offset instead of a branch.
    const int a_elems = span(is_dgrad ? r_total - r_base : pars.input_elems - x_base);
    const int b_elems = is_wgrad ? span(r_total - r_base) : pars.k * pars.kgemm;
    const auto a_rsrc = arch::make_buffer(in, a_elems);
    const auto b_rsrc = arch::make_buffer(wei, b_elems);
    const int a_oob   = a_elems;
    const int b_oob   = b_elems;

    // A thread keeps one slot column and walks rows a workgroup apart. rows_per_pass is a
    // whole number of swizzle periods, so a slot's row moves but the granule it fetches does not.
    constexpr int rows_per_pass = cfg.threads() / granules;
    static_assert(rows_per_pass % detail::GranuleTile<granules>::period == 0,
                  "a staging slot would change granule columns between passes");
    const int stage_slot_g = tid % granules;
    const int stage_row0   = tid / granules;
    const int stage_g      = detail::granule_for_slot<cfg>(stage_row0, stage_slot_g);
    const int wave_slot0   = tid - (tid % WAVE_SIZE);

    // Dgrad's A side is dY, which is already row-major over its reduction, so its row base
    // is the plain stride; fprop's is where the row's patch starts in X.
    //
    // Wgrad's rows are its reduction and move with the chunk, so fetch_one decodes them; what
    // stays put is a slot's filter column, whose share of the address is kept here instead,
    // -1 past the filter. On a padding config a row's last granule is pulled back to end on
    // the row's last element, as fprop's is; the epilogue drops the columns that repeats.
    // A short row's pulled-back origin is negative, so the past-the-filter mark cannot be -1.
    constexpr int NO_COL = INT32_MIN;
    int a_origin[cfg.a_rounds()];
    if constexpr(is_wgrad)
    {
        using TT              = detail::TransposeTile<cfg.m_tile()>;
        constexpr int per_row = cfg.m_tile() / GRANULE;
#pragma unroll
        for(int i = 0; i < cfg.a_rounds(); ++i)
        {
            const int slot = tid + i * cfg.threads();
            const int col  = m0 + TT::group_at(slot / per_row, (slot % per_row) * 2) * 4;
            const int r    = col / pars.kpad_row;
            int s          = col - r * pars.kpad_row;
            if constexpr(cfg.row_pad)
                s = std::min(s, pars.patch_row - GRANULE);
            a_origin[i] = r < pars.kh ? r * pars.w * pars.c + s : NO_COL;
        }
    }
    else
    {
#pragma unroll
        for(int i = 0; i < cfg.a_rounds(); ++i)
        {
            const int m = m0 + stage_row0 + i * rows_per_pass;
            if constexpr(is_dgrad)
                a_origin[i] = m < pars.rows ? (m - m0) * pars.k : a_oob;
            else
                a_origin[i] = detail::row_origin(pars, m, a_oob, img0);
        }
    }

    int b_origin[cfg.b_rounds()];
    if constexpr(!is_dgrad && !is_wgrad)
    {
#pragma unroll
        for(int i = 0; i < cfg.b_rounds(); ++i)
        {
            const int n = n0 + stage_row0 + i * rows_per_pass;
            b_origin[i] = n < pars.k ? n * pars.kgemm : b_oob;
        }
    }

    // Wave tile.
    const int wave    = tid / WAVE_SIZE;
    const int lane    = tid % WAVE_SIZE;
    const int lane16  = lane % MFMA_M;
    const int lane_hi = lane / MFMA_M; // the lane's eight-element K slice
    const int wave_m  = (wave / cfg.waves_n) * cfg.wave_m16() * MFMA_M;
    const int wave_n  = (wave % cfg.waves_n) * cfg.wave_n16() * MFMA_N;

    rt_acc acc;

    constexpr int W_TAIL  = 1;
    constexpr int DY_TAIL = 2;
    // A filter row shorter than a granule pulls its granule back past the row's start, and for
    // the first patches past the start of X or W, where the descriptor nulls the whole load.
    // 16-bit refills such a granule in LDS; tf32 loads it element by element instead.
    constexpr int SHORT_ROW = 4;

    // tf32's granules in flight: eight fp32 each, from the load that issues with the refill to the
    // split that follows the compute. A dead slot for 16-bit, which compiles away.
    constexpr int STAGED = is_tf32 ? cfg.a_rounds() + cfg.b_rounds() : 1;
    bunnies::uint32x4 staged[STAGED][2];

    // A granule from global element `off`: a DTL copy to the wave's slots, or tf32's two loads
    // into registers, which the buffer range check nulls per element past its end.
    //
    // __device__ keeps the host compile from mangling copy_in: the Microsoft C++ ABI has no
    // encoding for buffer_t.
    auto copy_in = [&] __device__<int I>(const arch::buffer_t& rsrc, void* wave_slots, int off) {
        if constexpr(is_tf32)
        {
            // The range check nulls a load that starts before the tensor whole, so a short row's
            // granule there loads element by element; the elements before the start meet a zeroed
            // B element or a dropped wgrad column. Those take the out-of-range offset rather than
            // their own, since loads a constant offset apart merge back into the whole load.
            if constexpr(SHORT)
                if(off < 0)
                {
                    const int oob = I < cfg.a_rounds() ? a_oob : b_oob;
#pragma unroll
                    for(int e = 0; e < GRANULE; ++e)
                    {
                        uint32_t v;
                        const int at = off + e >= 0 ? off + e : oob;
                        arch::buffer_load<4>::load(rsrc, &v, at * elem_bytes, 0);
                        staged[I][e / 4][e % 4] = v;
                    }
                    return;
                }
            arch::buffer_load<16>::load(rsrc, &staged[I][0], off * elem_bytes, 0);
            arch::buffer_load<16>::load(rsrc, &staged[I][1], (off + 4) * elem_bytes, 0);
        }
        else
            dtl::load(rsrc, wave_slots, off * elem_bytes, 0);
    };

    // Whether a dY granule starting at channel c can go straight to LDS: it has to end inside
    // its row, and an odd k leaves rows off the four-byte alignment the copy needs.
    auto dy_whole = [&](int c) { return pars.k % 2 == 0 && c + GRANULE <= pars.k; };

    // Stage a dY granule through registers, zero past k or on a row past the problem (row < 0).
    auto stage_dy = [&](dtype* dst, const dtype* dy, int row, int c) {
        vec_t value{};
        if(row >= 0)
            for(int e = 0; e < GRANULE; ++e)
                value[e] = c + e < pars.k ? dy[row + c + e] : dtype(0);
        *reinterpret_cast<vec_t*>(dst) = value;
    };

    // One staged piece, global straight to LDS. Fprop's A permutes on the way in: element k
    // of row m is at x[n_i][p_i*sh + r][q_i*sw + s][ch] for r = k / (kw*c).
    //
    // TAIL is a mask of the staging forms a walk needs. W_TAIL names dgrad's last filter tile,
    // which reaches past W: see the staging below. DY_TAIL is NARROW_K on the dY side, where a
    // granule would read into the next row -- or, on the last row, lose the four-byte unit
    // holding its last live element -- so it stages through registers instead.
    auto fetch_one = [&]<int BUF, int I, int TAIL>(int k0) {
        constexpr bool is_a = I < cfg.a_rounds();
        constexpr int i     = is_a ? I : I - cfg.a_rounds();

        if constexpr(is_wgrad)
        {
            // Both tiles sit reduction-major, as the transposed read wants them: A is the patch
            // gather of X along the filter axis, B is dY along the output channels. Each keeps a
            // granule inside one tensor row, so both stay plain DTL copies; the slot's rotation
            // moves the global address, as dgrad's B side does.
            constexpr int width   = is_a ? cfg.m_tile() : cfg.n_tile();
            using TT              = detail::TransposeTile<width>;
            constexpr int per_row = width / GRANULE;
            const int slot        = tid + i * cfg.threads();
            const int r_local     = slot / per_row;
            const int m           = kbase + k0 + r_local;
            const int slot0       = __builtin_amdgcn_readfirstlane(wave_slot0 + i * cfg.threads());

            if constexpr(is_a)
            {
                const int a_off = m < pars.rows && a_origin[i] != NO_COL
                                      ? detail::row_origin_magic(pars, m, img0) + a_origin[i]
                                      : a_oob;
                copy_in.template operator()<I>(
                    a_rsrc, &a_buf.template operator()<BUF>()[slot0 * GRANULE], a_off);
            }
            else
            {
                // A tf32 granule past k reads the next row's channels, which land in the
                // accumulator rows the epilogue drops: an element is its own four-byte unit, so
                // nothing live is lost and no form of its own is needed.
                const int kk = n0 + TT::group_at(r_local, (slot % per_row) * 2) * 4;
                if(is_tf32 || !(TAIL & DY_TAIL) || dy_whole(kk))
                {
                    const int b_off =
                        m < pars.rows && kk < pars.k ? (m - kbase) * pars.k + kk : b_oob;
                    copy_in.template operator()<I>(
                        b_rsrc, &b_buf.template operator()<BUF>()[slot0 * GRANULE], b_off);
                }
                else if constexpr(!is_tf32)
                {
                    // The tile is channel-contiguous, so this is a plain write, not a transpose.
                    stage_dy(&b_buf.template operator()<BUF>()[slot * GRANULE],
                             wei,
                             m < pars.rows ? (m - kbase) * pars.k : -1,
                             kk);
                }
            }
        }
        else if constexpr(is_dgrad && !is_a)
        {
            // W is staged in its own layout -- reduction-major, filter axis contiguous -- so this
            // side is a plain DTL copy and ds_read_b64_tr_b16 does the transpose on the way out.
            // A slot's rotation is what the read expects to find there, so it moves the global
            // address exactly as the A side's swizzle does.
            using TB              = detail::TransposeB<cfg>;
            constexpr int per_row = cfg.n_tile() / GRANULE;
            const int slot        = tid + i * cfg.threads();
            const int n_local     = slot / per_row;
            const int group       = TB::group_at(n_local, (slot % per_row) * 2);
            const int n           = k0 + n_local;
            const int j0          = n0 + group * 4;
            lds_t* b              = b_buf.template operator()<BUF>();

            // tf32 never takes the tail form: its range check nulls per element, so a load
            // running past W loses nothing live.
            static_assert(!(is_tf32 && (TAIL & W_TAIL)), "tf32 stages W's last tile as any other");
            if constexpr(!(TAIL & W_TAIL))
            {
                // Columns past the filter read the next row and land in accumulator entries the
                // epilogue drops; only the rounded-up reduction has to null, and so it does.
                const int slot0  = __builtin_amdgcn_readfirstlane(wave_slot0 + i * cfg.threads());
                const int b_step = n < pars.k ? n * pars.kgemm + j0 : b_oob;
                copy_in.template operator()<I>(b_rsrc, &b[slot0 * GRANULE], b_step);
            }
            else if constexpr(!is_tf32)
            {
                // A partly out-of-range DTL load drops the four-byte unit holding its last live
                // element, which every odd kh*kw*c ends on, so this tile stages through registers.
                // The tile is filter-contiguous, so the write is plain and wide, not a transpose.
                //
                // Running past the filter is not what forces that: those columns reach the next row
                // of W and land in accumulator entries the epilogue drops. Only running past W
                // itself has to go element-wise, which is the last reduction row and no other.
                const int base = n * pars.kgemm + j0;
                vec_t value{};
                if(n < pars.k && base + GRANULE <= b_elems)
                    value = *reinterpret_cast<const vec_t*>(&wei[base]);
                else if(n < pars.k)
                    for(int e = 0; e < GRANULE; ++e)
                        value[e] = base + e < b_elems ? wei[base + e] : dtype(0);
                *reinterpret_cast<vec_t*>(&b[slot * GRANULE]) = value;
            }
        }
        else
        {
            const int row   = stage_row0 + i * rows_per_pass;
            const int g     = detail::granule_for_slot<cfg>(row, stage_slot_g);
            const int slot0 = __builtin_amdgcn_readfirstlane(wave_slot0 + i * cfg.threads());
            const int k     = k0 + g * GRANULE;

            if constexpr(is_dgrad)
            {
                // dY is row-major over the reduction, so this side is a plain copy; the
                // chunks the reduction was rounded up to null through the descriptor. tf32
                // copies a granule past k whole and zeroes its tail at the split.
                if(is_tf32 || !(TAIL & DY_TAIL) || dy_whole(k))
                {
                    const int a_step = k < pars.k ? k : a_oob;
                    copy_in.template operator()<I>(
                        a_rsrc,
                        &a_buf.template operator()<BUF>()[slot0 * GRANULE],
                        a_origin[i] + a_step);
                }
                else if constexpr(!is_tf32)
                {
                    stage_dy(&a_buf.template operator()<BUF>()[(tid + i * cfg.threads()) * GRANULE],
                             in,
                             a_origin[i] != a_oob ? a_origin[i] : -1,
                             k);
                }
            }
            else
            {
                const int r = k / pars.kpad_row;
                // Past the last filter row the granule is pure padding, and an out-of-range
                // DTL load is what zeroes it.
                const bool live = !cfg.row_pad || r < pars.kh;
                int rest        = k - r * pars.kpad_row;
                // A row's last granule ends on its last real element rather than starting on
                // a granule boundary, so the final patch never runs off the tensor -- where a
                // partly out-of-range load drops an odd trailing element. The repeat that
                // creates is what zero_b_pad takes back out of B.
                if constexpr(cfg.row_pad)
                    rest = rest < pars.patch_row - GRANULE ? rest : pars.patch_row - GRANULE;
                if constexpr(is_a)
                {
                    const int a_step = live ? r * pars.w * pars.c + rest : a_oob;
                    copy_in.template operator()<I>(
                        a_rsrc,
                        &a_buf.template operator()<BUF>()[slot0 * GRANULE],
                        a_origin[i] + a_step);
                }
                else
                {
                    // B needs no permutation: KRSC is row-major over exactly this K order.
                    const int b_step = live ? r * pars.patch_row + rest : b_oob;
                    copy_in.template operator()<I>(
                        b_rsrc,
                        &b_buf.template operator()<BUF>()[slot0 * GRANULE],
                        b_origin[i] + b_step);
                }
            }
        }
    };

    // tf32's second half: a granule's eight fp32 split into the bf16 pair, each half stored to
    // its plane at the slot a DTL copy would have filled, so every layout and read above serves
    // the planes unchanged. What the DTL forms fix in LDS is fixed here instead, in registers: a
    // dY granule past k zeroes its tail, and a pulled-back W granule the repeat zero_b_pad would
    // take out.
    auto split_one = [&]<int BUF, int I, int TAIL>(int k0) {
        constexpr bool is_a = I < cfg.a_rounds();
        constexpr int i     = is_a ? I : I - cfg.a_rounds();
        fp32x8_t v          = __builtin_shufflevector(__builtin_bit_cast(fp32x4_t, staged[I][0]),
                                             __builtin_bit_cast(fp32x4_t, staged[I][1]),
                                             0,
                                             1,
                                             2,
                                             3,
                                             4,
                                             5,
                                             6,
                                             7);
        if constexpr(is_dgrad && is_a && (TAIL & DY_TAIL))
        {
            const int row = stage_row0 + i * rows_per_pass;
            const int k   = k0 + detail::granule_for_slot<cfg>(row, stage_slot_g) * GRANULE;
#pragma unroll
            for(int e = 0; e < GRANULE; ++e)
                v[e] = k + e < pars.k ? v[e] : 0.f;
        }
        if constexpr(!is_dgrad && !is_wgrad && !is_a && cfg.row_pad)
        {
            const int k   = k0 + stage_g * GRANULE;
            const int r   = k / pars.kpad_row;
            const int raw = k - r * pars.kpad_row;
            const int dup = r < pars.kh ? raw - std::min(raw, pars.patch_row - GRANULE) : 0;
#pragma unroll
            for(int e = 0; e < GRANULE; ++e)
                v[e] = e < dup ? 0.f : v[e];
        }
        const auto pair = fp32xN_to_bf16_pair<GRANULE>(v);
        lds_t* tile = is_a ? a_buf.template operator()<BUF>() : b_buf.template operator()<BUF>();
        constexpr int PLANE                             = is_a ? A_TILE : B_TILE;
        const int at                                    = (tid + i * cfg.threads()) * GRANULE;
        *reinterpret_cast<bf16x8_t*>(&tile[at])         = pair.big;
        *reinterpret_cast<bf16x8_t*>(&tile[PLANE + at]) = pair.small;
    };

    // A fragment tile out of LDS: as staged, or tf32's big plane and then its small one through
    // the same map, each into its half of the split.
    auto read = [&]<typename Inst, typename Plane, int PLANE>(auto& rt, lds_t* src, auto&& at) {
        if constexpr(is_tf32)
        {
            bunnies::load_tile<Inst, Plane, false>(rt, src, at);
            bunnies::load_tile<Inst, Plane, true>(rt, src + PLANE, at);
        }
        else
            bunnies::load_tile<Inst>(rt, src, at);
    };

    // One MFMA K step over a named buffer. Templated on the parity so the LDS objects it
    // reads are the ones the waitcnt pass can tell apart from the fetch in flight.
    auto compute_step = [&]<int BUF, int STEP, bool ZeroAcc>() {
        // Not const: ds_read_b64_tr_b16 takes a mutable LDS pointer where ds_load_b128 does not.
        lds_t* a_src = a_buf.template operator()<BUF>();
        lds_t* b_src = b_buf.template operator()<BUF>();

        // These functors only say where a (row, col) of the tile lives in LDS; load_tile
        // drives the lane and round mapping off each operand's own layout.
        rt_filt filt;
        if constexpr(is_dgrad || is_wgrad)
        {
            // row is the lane's filter element, 4-aligned so the group divide is exact; col is
            // the reduction index, which the transposed tile carries as its LDS row.
            using TB = detail::TransposeB<cfg>;
            read.template operator()<arch::ds_read_b64_tr_b16, plane_filt, B_TILE>(
                filt, b_src, [&](int nb, int, int row, int col) {
                return TB::elem(STEP * MFMA_K + col, (wave_n + nb * MFMA_N + row) / 4);
            });
        }
        else
        {
            read.template operator()<arch::ds_load_b128, plane_filt, B_TILE>(
                filt, b_src, [&](int nb, int, int row, int col) {
                return detail::lds_offset<cfg>(wave_n + nb * MFMA_N + row,
                                               (STEP * MFMA_K + col) / GRANULE);
            });
        }

        rt_spat spat;
        if constexpr(is_wgrad)
        {
            // row is the reduction index; col is the lane's filter element, 4-aligned.
            using TA = detail::TransposeTile<cfg.m_tile()>;
            read.template operator()<arch::ds_read_b64_tr_b16, plane_spat, A_TILE>(
                spat, a_src, [&](int, int mb, int row, int col) {
                return TA::elem(STEP * MFMA_K + row, (wave_m + mb * MFMA_M + col) / 4);
            });
        }
        else
        {
            read.template operator()<arch::ds_load_b128, plane_spat, A_TILE>(
                spat, a_src, [&](int, int mb, int row, int col) {
                return detail::lds_offset<cfg>(wave_m + mb * MFMA_M + col,
                                               (STEP * MFMA_K + row) / GRANULE);
            });
        }

        // The three-argument form feeds srcC an inline zero, so the opening step never
        // materialises a zeroed accumulator to live across the prologue. Worth 4.3%.
        if constexpr(ZeroAcc)
            bunnies::mma(acc, filt, spat);
        else
            bunnies::mma(acc, filt, spat, acc);
    };

    // Two stages, and the loop unrolled by that period so a phase's stage index is a
    // constant -- which is what lets the waitcnt pass tell its LDS objects from the fetch.
    constexpr int BUFS = 2;
    constexpr int PAIR = BUFS * cfg.k_chunk;
    // Dgrad rounds its reduction up to a chunk pair the way a padding config does, so every
    // output-channel count is served and the tail stages zeros.
    const int kwalk = [&] {
        if constexpr(is_wgrad)
            return pars.split_rows;
        else if constexpr(is_dgrad)
            return (pars.k + PAIR - 1) / PAIR * PAIR;
        else if constexpr(cfg.row_pad)
            return (pars.kh * pars.kpad_row + PAIR - 1) / PAIR * PAIR;
        else
            return pars.kgemm;
    }();
    const int chunks = kwalk / cfg.k_chunk;

    // The overlap a pulled-back B granule repeats, zeroed so it is not counted twice. tf32 zeroes
    // it at the split instead.
    //
    // A thread zeroes only granules it filled, so its own vmcnt is the whole ordering.
    auto zero_b_pad = [&]<int BUF>(int k0) {
        if constexpr(!is_dgrad && !is_wgrad && cfg.row_pad && !is_tf32)
        {
            const int k   = k0 + stage_g * GRANULE;
            const int r   = k / pars.kpad_row;
            const int raw = k - r * pars.kpad_row;
            const int dup = raw - (raw < pars.patch_row - GRANULE ? raw : pars.patch_row - GRANULE);
            if(r < pars.kh && dup > 0)
            {
                dtype* b = b_buf.template operator()<BUF>();
#pragma unroll
                for(int i = 0; i < cfg.b_rounds(); ++i)
                {
                    const int base = (tid + i * cfg.threads()) * GRANULE;
                    for(int e = 0; e < dup; ++e)
                        b[base + e] = dtype(0);
                }
            }
        }
    };

    // Refill from registers a SHORT_ROW granule the descriptor nulled for starting before its
    // tensor. Like zero_b_pad, a thread touches only slots it filled, so its vmcnt orders it.
    auto fix_short_rows = [&]<int BUF>(int k0) {
        // `first` is the index of src[0] in its tensor: elements before the tensor read zero,
        // and meet a zeroed B element or a dropped wgrad column.
        auto refill = [&](dtype* dst, const dtype* src, int64_t first, int off) {
            vec_t value{};
            for(int e = 0; e < GRANULE; ++e)
                value[e] = first + off + e >= 0 ? src[off + e] : dtype(0);
            *reinterpret_cast<vec_t*>(dst) = value;
        };
        dtype* a = a_buf.template operator()<BUF>();
        if constexpr(is_wgrad)
        {
            constexpr int per_row = cfg.m_tile() / GRANULE;
#pragma unroll
            for(int i = 0; i < cfg.a_rounds(); ++i)
            {
                const int slot = tid + i * cfg.threads();
                const int m    = kbase + k0 + slot / per_row;
                if(m < pars.rows && a_origin[i] != NO_COL)
                {
                    const int off = detail::row_origin_magic(pars, m, img0) + a_origin[i];
                    if(off < 0)
                        refill(&a[slot * GRANULE], in, x_base, off);
                }
            }
        }
        else if constexpr(!is_dgrad)
        {
            const int k = k0 + stage_g * GRANULE;
            const int r = k / pars.kpad_row;
            if(r < pars.kh)
            {
                const int rest = pars.patch_row - GRANULE;
                dtype* b       = b_buf.template operator()<BUF>();
#pragma unroll
                for(int i = 0; i < cfg.a_rounds(); ++i)
                {
                    // A row past the problem never reaches Y, and its sentinel is no address.
                    const int off = a_origin[i] + r * pars.w * pars.c + rest;
                    if(a_origin[i] != a_oob && off < 0)
                        refill(&a[(tid + i * cfg.threads()) * GRANULE], in, x_base, off);
                }
#pragma unroll
                for(int i = 0; i < cfg.b_rounds(); ++i)
                {
                    const int off = b_origin[i] + r * pars.patch_row + rest;
                    if(b_origin[i] != b_oob && off < 0)
                        refill(&b[(tid + i * cfg.threads()) * GRANULE], wei, 0, off);
                }
            }
        }
    };

    // One phase: retire this stage's chunk, refill the stage its predecessor freed, compute.
    // The barrier is what makes the last wave's read of that stage happen-before the refill.
    //
    // tf32 refills in two halves around the compute: its loads go out with the refill, as a DTL
    // copy would, and their split lands after the compute has issued, when the loads have had the
    // whole of it to arrive. The next phase's barrier publishes the stores.
    auto phase = [&]<int J, bool Refill, bool ZeroAcc, int TAIL>(int base) {
        if constexpr(!is_tf32)
            wait_vmcnt_all();
        if constexpr(!is_tf32 && (TAIL & SHORT_ROW))
            fix_short_rows.template operator()<J>((base + J) * cfg.k_chunk);
        zero_b_pad.template operator()<J>((base + J) * cfg.k_chunk);
        __syncthreads();

        // The refill is written across the K steps but the scheduler is left to place it:
        // every interleave forced on it, sched_group_barrier included, measured worse.
        constexpr int LOADS = cfg.a_rounds() + cfg.b_rounds();
        constexpr int STEPS = cfg.k_steps();
        const int k0_next   = (base + J + BUFS - 1) * cfg.k_chunk;

        if constexpr(Refill)
            static_for<LOADS>([&]<int I>() {
                fetch_one.template operator()<(J + BUFS - 1) % BUFS, I, TAIL>(k0_next);
            });
        __builtin_amdgcn_sched_barrier(0);
        static_for<STEPS>(
            [&]<int S>() { compute_step.template operator()<J, S, ZeroAcc && S == 0>(); });
        if constexpr(is_tf32 && Refill)
        {
            __builtin_amdgcn_sched_barrier(0);
            static_for<LOADS>([&]<int I>() {
                split_one.template operator()<(J + BUFS - 1) % BUFS, I, TAIL>(k0_next);
            });
        }
    };

    // The whole K walk, with the staging form fixed.
    //
    // TAIL is block-uniform, so it could be a branch inside the staging -- but then the path not
    // taken still holds registers, and the widest tile spilled 32 of them for it. Named here, each
    // form allocates on its own. The shared tiles stay declared once, so the two do not double the
    // LDS the way two instantiations of this function would.
    auto walk = [&]<int TAIL>() {
        // Prime BUFS-1 stages so a phase always has BUFS-2 chunks still in flight under it.
        static_for<BUFS - 1>([&]<int J>() {
            static_for<cfg.a_rounds() + cfg.b_rounds()>(
                [&]<int I>() { fetch_one.template operator()<J, I, TAIL>(J * cfg.k_chunk); });
            if constexpr(is_tf32)
                static_for<cfg.a_rounds() + cfg.b_rounds()>(
                    [&]<int I>() { split_one.template operator()<J, I, TAIL>(J * cfg.k_chunk); });
        });

        // The opening pair is peeled to name the zeroing form and the last to drop the refill;
        // a runtime branch there would split the block the fetch and compute overlap in.
        static_for<BUFS>([&]<int J>() { phase.template operator()<J, true, J == 0, TAIL>(0); });

        for(int base = BUFS; base + BUFS < chunks; base += BUFS)
            static_for<BUFS>(
                [&]<int J>() { phase.template operator()<J, true, false, TAIL>(base); });

        const int last = chunks - BUFS;
        if(last > 0)
        {
            static_for<BUFS - 1>(
                [&]<int J>() { phase.template operator()<J, true, false, TAIL>(last); });
            phase.template operator()<BUFS - 1, false, false, TAIL>(last);
        }
    };

    // tf32 needs neither of wgrad's dY form nor dgrad's W one: with an element per four-byte
    // unit, a load past either tensor's live range loses nothing, and wgrad's dY tail lands in
    // channels the epilogue drops. Only dgrad's dY tail is a reduction, which must read zero.
    constexpr int DY = NARROW_K && (is_dgrad || (is_wgrad && !is_tf32)) ? DY_TAIL : 0;
    if constexpr(is_dgrad && !is_tf32)
    {
        if(n0 + cfg.n_tile() > pars.kgemm)
            walk.template operator()<W_TAIL | DY>();
        else
            walk.template operator()<DY>();
    }
    else
    {
        walk.template operator()<(SHORT ? SHORT_ROW : 0) | DY>();
    }

    // Wgrad adds its split into dW, which the launch zeroed. The accumulator's columns are the
    // filter axis, so each group of sixteen lanes lands one atomic on sixteen contiguous floats.
    if constexpr(is_wgrad)
    {
        float* const dw = static_cast<float*>(out_raw);
#pragma unroll
        for(int nb = 0; nb < cfg.wave_n16(); ++nb)
#pragma unroll
            for(int mb = 0; mb < cfg.wave_m16(); ++mb)
#pragma unroll
                for(int item = 0; item < mat_acc::num_items; ++item)
                {
                    const auto coord = mat_acc::map({lane, item});
                    const int kk     = n0 + wave_n + nb * MFMA_N + coord[0];
                    const int col    = m0 + wave_m + mb * MFMA_M + coord[1];
                    const int r      = col / pars.kpad_row;
                    const int s      = col - r * pars.kpad_row;
                    // A pulled-back granule holds its row's last eight elements, so its column
                    // e is element start + e, and the ones the previous granule has are skipped.
                    const int g0    = s - s % GRANULE;
                    const int start = cfg.row_pad ? std::min(g0, pars.patch_row - GRANULE) : g0;
                    const int elem  = start + s % GRANULE;
                    if(kk < pars.k && r < pars.kh && elem >= g0 && elem < pars.patch_row)
                        unsafeAtomicAdd(&dw[kk * pars.kgemm + r * pars.patch_row + elem],
                                        acc.block(nb, mb).data[item]);
                }
        return;
    }

    // The result leaves through LDS: a lane holds four channels of one row, too narrow to
    // store well, and re-reading the tile in output order widens it. The stage is free.
    constexpr int OUT_ROW = cfg.out_row();
    constexpr int DRAINS  = cfg.drains();
    constexpr int MB_PER  = cfg.wave_m16() / DRAINS; // row blocks a wave drains per part
    constexpr int ROWS    = cfg.m_tile() / DRAINS;   // tile rows a part carries
    constexpr int RUN     = 16 / elem_bytes;         // elements in a lane's 16-byte store

    dtype* out_lds;
    if constexpr(cfg.n_tile() > cfg.m_tile())
        out_lds = reinterpret_cast<dtype*>(b_lds0);
    else
        out_lds = reinterpret_cast<dtype*>(a_lds0);

    // Where output (row, col) sits in the stage: a 16-bit row padded off the banks, an fp32 one
    // swizzled; see OutStage.
    auto out_at = [](int r, int c) {
        if constexpr(is_tf32)
            return detail::OutStage<OUT_ROW>::elem(r, c);
        else
            return r * OUT_ROW + c;
    };

    // The 16-bit types narrow before the stage; tf32's result is the fp32 accumulator itself.
    using rt_staged = std::conditional_t<is_tf32, rt_acc, rt_out>;
    rt_out narrowed;
    if constexpr(!is_tf32)
        bunnies::tile_cast(narrowed, acc);
    rt_staged& result = [&]() -> rt_staged& {
        if constexpr(is_tf32)
            return acc;
        else
            return narrowed;
    }();

    const auto out_rsrc =
        arch::make_buffer(out, span(is_dgrad ? pars.input_elems - x_base : r_total - r_base));
    const int wave_mi = bunnies::wave_id() / cfg.waves_n;

    // Dgrad's scatter address separates: patch_offset is a term in the tile row plus a term in
    // the tile column, and each costs a divide by a runtime extent. The store walks both per
    // element, so they are tabulated once here -- m_tile + n_tile entries against the
    // m_tile * n_tile the drains would otherwise divide for.
    constexpr int TABLE = is_dgrad ? cfg.m_tile() + cfg.n_tile() : 1;
    __shared__ int scatter_base[TABLE];
    int* const row_base = scatter_base;
    int* const col_base = scatter_base + (is_dgrad ? cfg.m_tile() : 0);

    if constexpr(is_dgrad)
    {
        // The walk's last phase still had this stage live.
        __syncthreads();
        const int gap = (pars.w - pars.kw) * pars.c;
        for(int i = tid; i < cfg.m_tile(); i += cfg.threads())
            row_base[i] = detail::row_origin(pars, m0 + i, 0, img0);
        // tf32's whole runs read one column entry a lane run, so only runs are tabulated and they
        // are read densely: an entry a run apart would put a phase's 32 lanes on 32 / RUN banks.
        // 16-bit keeps an entry a column: its dense read would add scratch to M256xN256, already
        // at 256 VGPRs, and run slower.
        const bool dense  = is_tf32 && pars.patch_row % RUN == 0;
        const int entries = dense ? cfg.n_tile() / RUN : cfg.n_tile();
        for(int i = tid; i < entries; i += cfg.threads())
        {
            const int j = n0 + (dense ? i * RUN : i);
            col_base[i] = j / pars.patch_row * gap + j;
        }
        // The drain loop opens on a barrier, which is what publishes these.
    }

    static_for<DRAINS>([&]<int D>() {
        __syncthreads();

        // A part takes MB_PER row blocks from every wave, so its waves are strided in the
        // tile and packed in LDS; the store below undoes that.
        using store_inst = std::conditional_t<is_tf32, arch::ds_store_b128, arch::ds_store_b64>;
        auto part        = result.template sub<cfg.wave_n16(), MB_PER>(0, D * MB_PER);
        bunnies::store_tile<store_inst>(part, out_lds, [&](int nb, int mb, int chan, int row) {
            const int lds_row = (wave_mi * MB_PER + mb) * MFMA_M + row;
            return out_at(lds_row, wave_n + nb * MFMA_N + chan);
        });

        __syncthreads();

        if constexpr(is_dgrad)
        {
            // A lane's run is only whole when a filter row is a multiple of it, since that is
            // what keeps the run inside one row of dX.
            if(pars.patch_row % RUN == 0)
            {
                bunnies::buffer_store_from_lds<arch,
                                               {.rows           = ROWS,
                                                .cols           = cfg.n_tile(),
                                                .bytes_per_lane = 16,
                                                .num_waves      = cfg.waves()}>(
                    bunnies::wave_id(), out_rsrc, [&](int r, int c) -> std::array<int, 2> {
                    const int wv  = r / (MB_PER * MFMA_M);
                    const int rem = r % (MB_PER * MFMA_M);
                    const int mi  = wv * cfg.wave_m16() * MFMA_M + D * MB_PER * MFMA_M + rem;
                    const int voffset =
                        (m0 + mi < pars.rows && n0 + c + RUN <= pars.kgemm)
                            ? static_cast<int>((row_base[mi] + col_base[is_tf32 ? c / RUN : c]) *
                                               sizeof(dtype))
                            : -1;
                    return {voffset, 0};
                }, out_lds, out_at);
            }
            else
            {
                for(int idx = tid; idx < ROWS * cfg.n_tile(); idx += cfg.threads())
                {
                    const int r   = idx / cfg.n_tile();
                    const int c   = idx % cfg.n_tile();
                    const int wv  = r / (MB_PER * MFMA_M);
                    const int rem = r % (MB_PER * MFMA_M);
                    const int mi  = wv * cfg.wave_m16() * MFMA_M + D * MB_PER * MFMA_M + rem;
                    if(m0 + mi < pars.rows && n0 + c < pars.kgemm)
                        out[row_base[mi] + col_base[c]] = out_lds[out_at(r, c)];
                }
            }
        }
        else if constexpr(NARROW_K)
        {
            // A whole run would cross into the next row of Y, so this goes element-wise.
            for(int idx = tid; idx < ROWS * cfg.n_tile(); idx += cfg.threads())
            {
                const int r   = idx / cfg.n_tile();
                const int c   = idx % cfg.n_tile();
                const int wv  = r / (MB_PER * MFMA_M);
                const int rem = r % (MB_PER * MFMA_M);
                const int m   = m0 + wv * cfg.wave_m16() * MFMA_M + D * MB_PER * MFMA_M + rem;
                if(m < pars.rows && n0 + c < pars.k)
                    out[(m - m0) * pars.k + n0 + c] = out_lds[out_at(r, c)];
            }
        }
        else
        {
            bunnies::buffer_store_from_lds<arch,
                                           {.rows           = ROWS,
                                            .cols           = cfg.n_tile(),
                                            .bytes_per_lane = 16,
                                            .num_waves      = cfg.waves()}>(
                bunnies::wave_id(), out_rsrc, [&](int r, int c) -> std::array<int, 2> {
                const int wv  = r / (MB_PER * MFMA_M);
                const int rem = r % (MB_PER * MFMA_M);
                const int m   = m0 + wv * cfg.wave_m16() * MFMA_M + D * MB_PER * MFMA_M + rem;
                const int n   = n0 + c;
                // buffer addressing nulls the row past the problem; the column cannot straddle
                // because both tile origins, k and the lane run are multiples of the run.
                const int voffset = (m < pars.rows && n < pars.k)
                                        ? static_cast<int>(((m - m0) * pars.k + n) * sizeof(dtype))
                                        : -1;
                return {voffset, 0};
            }, out_lds, out_at);
        }
    });
}

// One entry point per direction: fprop reads (X, W) into Y, dgrad reads (dY, W) into dX, and
// wgrad reads (X, dY) into the fp32 dW.
//
// The registry compiles this header for every target it serves, so gate on the builtins:
// that leaves a well-formed empty kernel off gfx950 rather than a build error.
template <Config cfg, DataType DT, Direction DIR, bool NARROW_K, bool SHORT>
__global__
__launch_bounds__(cfg.threads()) void patch_embed_cdna4(LayerPars pars,
                                                        const ToType<DT>* __restrict__ in,
                                                        const ToType<DT>* __restrict__ w,
                                                        void* __restrict__ out)
{
    // A tile the transposed read cannot be laid out on serves fprop alone, and is_valid_config
    // never pairs it with a direction that reads transposed.
    constexpr bool served = DIR == Direction::Fprop   ? true
                            : DIR == Direction::Dgrad ? cfg.serves_dgrad()
                                                      : cfg.serves_wgrad();
    if constexpr(served)
    {
        if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_f16) &&
           __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_bf16))
        {
            patch_embed_impl<cfg, DT, DIR, NARROW_K, SHORT>(pars, in, w, out);
        }
    }
}

// Wgrad's M axis is the filter, not the spatial rows it reduces over.
inline int m_tiles(const Config& cfg, const ConvParams& par)
{
    const int m =
        par.direction == Direction::Wgrad ? wgrad_filter_axis(cfg, par) : par.n * par.p * par.q;
    return (m + cfg.m_tile() - 1) / cfg.m_tile();
}

inline int n_tiles(const Config& cfg, const ConvParams& par)
{
    const int n = par.direction == Direction::Dgrad ? par.kh * par.kw * par.c : par.k;
    return (n + cfg.n_tile() - 1) / cfg.n_tile();
}

inline LaunchParams get_launch_params(const Config& cfg, const ConvParams& par, int cu_count)
{
    LaunchParams launch;
    const int splits  = par.direction == Direction::Wgrad ? wgrad_splits(cfg, par, cu_count) : 1;
    launch.grid       = par.direction == Direction::Wgrad
                            ? dim3(m_tiles(cfg, par), n_tiles(cfg, par), splits)
                            : dim3(m_tiles(cfg, par) * n_tiles(cfg, par), 1, 1);
    launch.block_size = dim3(cfg.threads(), 1, 1);
    launch.dynamic_shared_bytes = 0;
    return launch;
}

// The two ladders tile_cost ranks: aligned filter rows, then padded ones.
constexpr Config NARROW_TILE = configs[0]; // M32xN128 k128
constexpr Config MID_TILE    = configs[1]; // M64xN64  k64
constexpr Config WIDE_TILE   = configs[2]; // M96xN192 k128
constexpr Config HUGE_TILE   = configs[3]; // M256xN256 k64
constexpr Config NARROW_ALT  = configs[4]; // M32xN128 k64, for the K depths 128 misses

constexpr Config TALL_TILE = configs[5]; // M128xN256 k64
constexpr Config SQ_TILE   = configs[6]; // M128xN128 k64

constexpr Config PAD_NARROW = configs[7];  // M32xN128  k128
constexpr Config PAD_MID    = configs[8];  // M64xN128  k64
constexpr Config PAD_WIDE   = configs[9];  // M128xN128 k32
constexpr Config PAD_HUGE   = configs[10]; // M256xN256 k32

// The K the loop walks: the real depth, or a padding config's rows rounded to a chunk pair.
inline int walked_k(const Config& cfg, int kh, int patch_row)
{
    const int pad  = 2 * cfg.k_chunk;
    const int axis = cfg.row_pad ? kh * cfg.padded_row(patch_row) : kh * patch_row;
    return cfg.row_pad ? (axis + pad - 1) / pad * pad : axis;
}

// The same for dgrad, whose reduction is the output channels and always rounds to a pair.
inline int walked_k_dgrad(const Config& cfg, const ConvParams& par)
{
    const int pad = 2 * cfg.k_chunk;
    return (par.k + pad - 1) / pad * pad;
}

// Whether an entry can run the problem at all: the K loop deals whole chunks, and a tail
// would need a masked staging pass the kernel does not have.
inline bool chunks_evenly(const Config& cfg, const ConvParams& par)
{
    // A padding config rounds its own axis up to a pair of chunks, so any K suits it.
    if(cfg.row_pad)
        return true;
    const int kgemm = par.kh * par.kw * par.c;
    if(kgemm % cfg.k_chunk != 0)
        return false;
    // The K loop is unrolled by the buffer period, so the chunks have to fill it.
    return (kgemm / cfg.k_chunk) % 2 == 0;
}

// Wgrad's model: traffic and MFMA over one split's walk, the LDS reads, and the atomics each
// split lands on dW.
//
// The LDS term is what fprop's model lacks. Wgrad reads both operands with ds_read_b64_tr_b16,
// half the bytes per instruction ds_read_b128 moves, and every wave re-reads the tile along the
// axis it does not own -- so the read volume, not the global traffic, is what separates the
// tiles, and M128xN128 wins most shapes by holding it down at two workgroups per CU. Fitted on
// five stems over n 1..1024 under the occupancy split (min of three runs each): the pick lands
// 1.03x off the per-shape best on average, 1.25x at worst. Refit rather than reason about the
// constants.
//
// tf32's are its own; see the TF32 section of patch-embed-cdna4.md.
inline double wgrad_tile_cost(const Config& cfg, const ConvParams& par, int cu_count)
{
    const bool tf32              = cfg.is_tf32();
    const double BYTES_PER_S     = tf32 ? 1e13 : 2e11;
    const double FLOPS_PER_S     = tf32 ? 1e16 : 8e14;
    const double LDS_BYTES_PER_S = tf32 ? 2.6e14 : 2e13;
    const double ATOMICS_PER_S   = tf32 ? 1.3e11 : 2.5e9;
    const double HIDE_WAVES      = tf32 ? 2.2 : 2.0;

    const int splits  = wgrad_splits(cfg, par, cu_count);
    const int64_t wgs = static_cast<int64_t>(m_tiles(cfg, par)) * n_tiles(cfg, par) * splits;
    const int kwalk   = wgrad_split_rows(cfg, par, cu_count);
    const int cap     = wgrad_slots(cfg);

    // A round is every CU holding its full cap, so a split that fits in one costs no tail.
    const int64_t slots  = static_cast<int64_t>(cap) * cu_count;
    const int64_t rounds = (wgs + slots - 1) / slots;
    const double tail    = rounds > 1 ? static_cast<double>(rounds) * slots / wgs : 1.0;
    const int64_t per_cu = (wgs + cu_count - 1) / cu_count;
    const double waves   = std::min<int64_t>(cap, per_cu) * (cfg.threads() / 64.0) / 4.0;
    const double hide    = std::min(1.0, waves / HIDE_WAVES);

    const double walked = static_cast<double>(wgs) * kwalk;
    const double bytes  = walked * (cfg.m_tile() + cfg.n_tile()) * cfg.elem_bytes;
    const double flops  = 2.0 * walked * cfg.m_tile() * cfg.n_tile();
    const double lds =
        walked * (cfg.waves_m * cfg.n_tile() + cfg.waves_n * cfg.m_tile()) * cfg.elem_bytes;
    const double atomics = static_cast<double>(splits) * par.k * par.kh * par.kw * par.c;

    return (bytes / BYTES_PER_S + flops / FLOPS_PER_S + lds / LDS_BYTES_PER_S) * tail / hide +
           atomics / ATOMICS_PER_S;
}

// Modelled seconds for one tile on one shape, which is what ranks the ladder.
//
// Traffic is what a workgroup reads, not what the shape needs, and occupancy splits into
// the tail and latency hiding; folding those two together measured worse. See the doc.
inline double tile_cost(const Config& cfg, const ConvParams& par, int cu_count)
{
    if(par.direction == Direction::Wgrad)
        return wgrad_tile_cost(cfg, par, cu_count);
    const bool dgrad = par.direction == Direction::Dgrad;

    // Fitted, not the hardware's peaks: they absorb the kernel's own efficiency, which is why
    // dgrad carries its own three. Its operand read is ds_read_b64_tr_b16, moving half the bytes
    // an aligned ds_read_b128 does, and it wants a deeper grid before more waves stop paying.
    // Dgrad's were refit under the N-fastest XCD-grouped grid, over five stems at n 16..1024.
    // Refit rather than reason about them.
    //
    // tf32 has its own (the TF32 section of patch-embed-cdna4.md). Its fprop also pays a fixed
    // latency per phase a CU walks: below a workgroup per CU the walk is latency-bound, which the
    // byte and MFMA terms do not see.
    const bool tf32          = cfg.is_tf32();
    const double BYTES_PER_S = tf32 ? (dgrad ? 1.2e13 : 1.8e13) : (dgrad ? 6e11 : 6e12);
    const double FLOPS_PER_S = tf32 ? (dgrad ? 7e14 : 4.5e14) : (dgrad ? 5e13 : 3e14);
    const double HIDE_WAVES =
        dgrad && !tf32 ? 3.0 : 1.5; // waves per SIMD past which more stops helping
    const double PHASE_S = tf32 && !dgrad ? 2.8e-7 : 0.0;

    const int64_t wgs = static_cast<int64_t>(m_tiles(cfg, par)) * n_tiles(cfg, par);
    const int kwalk   = dgrad ? walked_k_dgrad(cfg, par) : walked_k(cfg, par.kh, par.kw * par.c);
    const int cap     = std::max(1, 160 * 1024 / cfg.lds_bytes());

    const int64_t per_cu   = (wgs + cu_count - 1) / cu_count;
    const double tail      = static_cast<double>(per_cu) * cu_count / wgs;
    const int64_t resident = std::min<int64_t>(cap, per_cu);
    double waves           = resident * (cfg.threads() / 64.0) / 4.0;
    // Registers cap the waves a SIMD holds as much as LDS does, and the accumulator is what moves
    // with the tile: a wave owns wave_m16 * wave_n16 * 4 of them before its operands and addresses,
    // which the constant stands in for. Fitted on dgrad alone, so fprop keeps the LDS bound it was
    // fitted under; ATT reports the two bounds separately and dgrad's widest tiles hit this one.
    if(dgrad)
        waves = std::min(waves, 512.0 / (cfg.acc_vgprs() + 128.0));
    const double hide = std::min(1.0, waves / HIDE_WAVES);

    // The axes trade places: dgrad writes the filter axis and reduces over the output channels.
    const double rows = static_cast<double>(par.n) * par.p * par.q;
    const double out  = dgrad ? static_cast<double>(par.kh) * par.kw * par.c : par.k;
    const double bytes =
        (static_cast<double>(wgs) * (cfg.m_tile() + cfg.n_tile()) * kwalk + rows * out) *
        cfg.elem_bytes;
    const double flops = 2.0 * wgs * cfg.m_tile() * cfg.n_tile() * kwalk;
    // A CU walks its resident workgroups' phases side by side, and its rounds one after another.
    const double phases =
        static_cast<double>((per_cu + resident - 1) / resident) * kwalk / cfg.k_chunk;

    return (bytes / BYTES_PER_S + flops / FLOPS_PER_S) * tail / hide + phases * PHASE_S;
}

// Whether any aligned tile of `like`'s element width can chunk this K.
//
// The ladders are per width because their chunks are: tf32 halves the 16-bit ones, so a K
// the 16-bit aligned ladder misses can still pair up on tf32's.
inline bool aligned_ladder_serves(const Config& like, const ConvParams& par)
{
    for(const Config& tile : configs)
        if(tile.elem_bytes == like.elem_bytes && !tile.row_pad && addressing_fits(tile, par) &&
           chunks_evenly(tile, par))
            return true;
    return false;
}

// Whether cfg is one of the tiles this problem may be served by.
//
// A whole ladder, since tile_cost ranks it; the padded one only where the aligned cannot.
inline bool is_valid_config(const ConvParams& par, const Config& cfg)
{
    // A config is compiled for one element width, which alone partitions tf32's from the
    // 16-bit ones.
    if(static_cast<int>(sizeof_data_type(par.input_type)) != cfg.elem_bytes)
        return false;
    const bool dgrad = par.direction == Direction::Dgrad;
    const bool wgrad = par.direction == Direction::Wgrad;
    // Sweep hatch: HIPCONV_PE_SWEEP=<index> pins the run to configs[index].
    if(const char* pin = getenv("HIPCONV_PE_SWEEP"))
    {
        const int idx = atoi(pin);
        // A pin still has to be a tile that can run the problem, or it measures garbage.
        const bool runs = dgrad   ? cfg.serves_dgrad()
                          : wgrad ? cfg.serves_wgrad() && addressing_fits(cfg, par)
                                  : chunks_evenly(cfg, par) && addressing_fits(cfg, par);
        return idx >= 0 && idx < num_configs && cfg == configs[idx] && runs;
    }
    // Wgrad reduces over the rows, which a split pads to a chunk pair, so only the filter axis
    // constrains it: its granules must stay inside one filter row. The two ladders split on that
    // as fprop's do, the padded one taking the rows the granule does not divide.
    // M256xN256 holds one workgroup per CU and never won a wgrad shape against M128xN256.
    if(wgrad)
        return cfg.serves_wgrad() && addressing_fits(cfg, par) && !(cfg == HUGE_TILE) &&
               cfg.row_pad != ((par.kw * par.c) % GRANULE == 0);
    // Dgrad rounds its own reduction up to a chunk pair, so no tile is ruled out by the K axis
    // and the whole table is a ladder tile_cost ranks. row_pad says nothing here -- it names
    // fprop's staging alone -- so the two ladders collapse onto the tiles they share.
    if(dgrad)
        return cfg.serves_dgrad();
    if(!chunks_evenly(cfg, par) || !addressing_fits(cfg, par))
        return false;
    // The padding ladder is the family's floor, not a fallback: explicit-GEMM is off in a
    // default build, so a shape neither ladder takes has no kernel at all.
    return cfg.row_pad != aligned_ladder_serves(cfg, par);
}

template <Config cfg>
void launch_impl(const LaunchParams& lp,
                 const Device& dev,
                 const ConvParams& par,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* /*workspace*/,
                 hipStream_t stream)
{
    auto typed_launch = [&]<DataType DT>() {
        using dtype   = ToType<DT>;
        auto dispatch = [&]<Direction DIR>() {
            auto go = [&]<bool NARROW_K, bool SHORT>() {
                patch_embed_cdna4<cfg, DT, DIR, NARROW_K, SHORT>
                    <<<lp.grid, lp.block_size, lp.dynamic_shared_bytes, stream>>>(
                        layer_pars(cfg, par, dev.cu_count),
                        static_cast<const dtype*>(in),
                        static_cast<const dtype*>(wei),
                        out);
            };
            // A k the granule does not divide narrows fprop's store and dgrad's dY staging; tf32's
            // wgrad has no narrow form, so it is not built twice.
            auto narrow = [&]<bool SHORT>() {
                if constexpr(cfg.is_tf32() && DIR == Direction::Wgrad)
                    go.template operator()<false, SHORT>();
                else if(par.k % GRANULE != 0)
                    go.template operator()<true, SHORT>();
                else
                    go.template operator()<false, SHORT>();
            };
            // Only a padded fprop or wgrad stages a short row; dgrad's reduction is k.
            if constexpr(cfg.row_pad && DIR != Direction::Dgrad)
                if(par.kw * par.c < GRANULE)
                    return narrow.template operator()<true>();
            narrow.template operator()<false>();
        };
        if(par.direction == Direction::Wgrad)
        {
            // `wei` carries dY and `out` the fp32 dW, which every split adds into.
            HIP_CHECK(hipMemsetAsync(out, 0, ConvSize(par).weight_grad_bytes(), stream));
            dispatch.template operator()<Direction::Wgrad>();
        }
        else if(par.direction == Direction::Dgrad)
        {
            // The scatter writes only pixels a patch covers; the strips and gaps it skips are zero.
            if(par.p * par.kh != par.h || par.q * par.kw != par.w)
                HIP_CHECK(hipMemsetAsync(out, 0, ConvSize(par).input_grad_bytes(), stream));
            dispatch.template operator()<Direction::Dgrad>();
        }
        else
            dispatch.template operator()<Direction::Fprop>();
    };
    // A config is built for one element width; is_valid_config never pairs it with another.
    if constexpr(cfg.is_tf32())
        typed_launch.template operator()<DataType::tf32>();
    else if(par.input_type == DataType::bf16)
        typed_launch.template operator()<DataType::bf16>();
    else
        typed_launch.template operator()<DataType::fp16>();
}

class PatchEmbed_ConvKernel : public DirectConvKernel
{
public:
    constexpr PatchEmbed_ConvKernel(const Config& cfg, LaunchFn launch_fn)
        : DirectConvKernel(launch_fn)
        , cfg_(cfg)
    {
    }

    std::string_view name() const override { return "patch_embed"; }

    KVDescriptor config_descriptor() const override { return ConfigMatcher(cfg_); }

    // Does not chain to DirectConvKernel::is_applicable, which rejects any stride but one.
    // A patch embedding is exactly the case where the stride equals the filter; a stride past
    // it only skips pixels, and a 1x1 is the one-pixel patch.
    bool is_applicable(const Device& /*dev*/, const ConvParams& par) const override
    {
        // The 16-bit types are same-type; tf32 takes tf32 operands and an fp32 result.
        const bool ok_16bit =
            (par.input_type == DataType::fp16 || par.input_type == DataType::bf16) &&
            par.input_type == par.weight_type && par.input_type == par.output_type;
        const bool ok_tf32 = par.input_type == DataType::tf32 &&
                             par.weight_type == DataType::tf32 && par.output_type == DataType::fp32;
        if(!ok_16bit && !ok_tf32)
            return false;
        if(par.order != TensorOrder::NHWC)
            return false;
        // Wgrad accumulates across splits, so its dW is fp32 whatever the operands are.
        if(par.direction == Direction::Wgrad && par.weight_grad_type != DataType::fp32)
            return false;
        if(par.groups != 1)
            return false;
        if(par.dilation_h != 1 || par.dilation_w != 1)
            return false;
        // The patch geometry: filters that tile the image without overlap or padding.
        if(par.kh > par.stride_h || par.kw > par.stride_w)
            return false;
        if(par.pad_h != 0 || par.pad_w != 0)
            return false;
        // Offsets are 32-bit, but every descriptor over the batch rebases per workgroup, so what
        // has to fit is one workgroup's range -- the widest M tile, or a wgrad chunk pair -- and W.
        const ConvSize sz(par);
        const size_t w_bytes =
            par.direction == Direction::Wgrad ? sz.weight_grad_bytes() : sz.weight_bytes();
        if(w_bytes > INT32_MAX || static_cast<int64_t>(par.n) * par.p * par.q > INT32_MAX)
            return false;
        return span_rows(par) >= 256;
    }

    bool is_valid_config(const Device& /*dev*/, const ConvParams& par) const override
    {
        return patch_embed::is_valid_config(par, cfg_);
    }

    LaunchParams get_launch_params(const Device& dev, const ConvParams& par) const override
    {
        return patch_embed::get_launch_params(cfg_, par, dev.cu_count);
    }

    // Wgrad with more than one row split adds into dW atomically.
    bool is_deterministic(const Device& dev, const ConvParams& par) const override
    {
        return par.direction != Direction::Wgrad || wgrad_splits(cfg_, par, dev.cu_count) == 1;
    }

    // Ranks the ladder in every direction, and places the family against explicit-GEMM: where it
    // outruns that path the rank folds into (1, 2), and where it yields into (0, 0.5). 16-bit
    // outruns it in fprop and wgrad and yields in dgrad; tf32 swaps fprop and dgrad (see the TF32
    // section of patch-embed-cdna4.md).
    //
    // An unstrided 1x1 also yields to hipBLASLt's pointwise with its 128 MiB workspace (#289),
    // except where that loses. Over 824 workload 1x1s: fprop wins c < 256 with n*h*w <= 64K (1.23x
    // geomean) and loses elsewhere, most on small n*h*w over c >= 1024, which hipBLASLt splits
    // along K. On ROCm 10.1 wgrad wins c <= 8 past n*h*w = 8M (1.44x over 18 layers) and loses
    // elsewhere. ROCm 7.x hipBLASLt, which past n*h*w ~3.3M offers only workspace-less
    // algorithms 3-600x slower, is not what this is fit to.
    //
    // tf32 has its own fit, over 749 workload 1x1s on ROCm 10.2. Fprop yields throughout: it wins
    // only c <= 128 under n*h*w = 2K, by 1.10x. Dgrad wins k <= 128 with n*h*w <= 8K (1.29x over
    // 126 layers). Wgrad wins c <= 1024 past n*h*w = 2K, and every c past 1M (3.33x over 416
    // layers): past 8M hipBLASLt's tf32 wgrad runs 4-144x behind.
    float get_weighted_throughput_index(const Device& dev, const ConvParams& par) const override
    {
        constexpr double UNIT = 1e-5; // seconds, about where these shapes land
        const double rank     = 1.0 / (1.0 + tile_cost(cfg_, par, dev.cu_count) / UNIT);
        const bool yields =
            cfg_.is_tf32() ? par.direction == Direction::Fprop : par.direction == Direction::Dgrad;
        if(yields || yields_to_pointwise(par, cfg_.is_tf32()))
            return static_cast<float>(0.5 * rank);
        return static_cast<float>(1.0 + rank);
    }

    void
    get_tolerance(const Device& dev, const ConvParams& par, float& atol, float& rtol) const override
    {
        if(par.direction == Direction::Dgrad)
            get_mixed_precision_tolerance(par, static_cast<size_t>(par.k), atol, rtol);
        else
            DirectConvKernel::get_tolerance(dev, par, atol, rtol);
    }

private:
    static bool yields_to_pointwise(const ConvParams& par, bool tf32)
    {
        const bool unstrided_1x1 =
            par.kh == 1 && par.kw == 1 && par.stride_h == 1 && par.stride_w == 1;
        if(!unstrided_1x1)
            return false;
        const int64_t rows = static_cast<int64_t>(par.n) * par.p * par.q;
        if(tf32)
        {
            if(par.direction == Direction::Dgrad)
                return !(par.k <= 128 && rows <= 8192);
            if(par.direction == Direction::Wgrad)
                return !((par.c <= 1024 && rows > 2048) || rows > 1048576);
            return true;
        }
        if(par.direction == Direction::Fprop)
            return !(par.c < 256 && rows <= 65536);
        return par.direction == Direction::Wgrad && !(par.c <= 8 && rows > 8388608);
    }

    const Config& cfg_;
};

} // namespace hipconv::cdna4::patch_embed
