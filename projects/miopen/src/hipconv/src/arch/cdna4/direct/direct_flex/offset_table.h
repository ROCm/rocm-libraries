#pragma once

// Input offset tables: a fibre offset per filter position and pixel of an X tile (Input fibres).
// Parenthesized section names refer to docs/algorithms/direct/direct-flex-cdna4.md.

#include "kernel_args.h"
#include "layer_pars.h"
#include "tile_order.h"
#include "traits.h"
#include "bunnies.hpp"
#include "tensor_view.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdint>

namespace hipconv::cdna4::direct_flex
{

// A tile's bases: descriptors rebase per tile, so offsets span only its range (Offset range).
struct Base
{
    int img0, h_base;
    int64_t x_base;
    int d_base = 0;
};

// Filter tiles [lo, hi) of an X tile.
struct Range
{
    int lo, hi;
};

// build_next's default has_tile: every target X tile exists.
struct AnyTile
{
    __device__ constexpr bool operator()(int) const { return true; }
};

// An X tile's input offset tables, double-buffered in LDS, built one filter tile per call.
template <class T>
class OffsetTable
{
    using Args = KernelArgs<T::PHASED>;
    using Pars = typename Args::Pars;
    using Div  = detail::KernargDiv;

    static constexpr Config cfg        = T::cfg;
    static constexpr int ROWS_PER_PASS = cfg.threads() / T::granules;
    static constexpr bool SKIP         = T::DEPTH && T::N > 0;

public:
    // A csplit tile's rows hold its X(128) pixels alone, so three stages fit in LDS.
    static constexpr int COLS = T::CSPLIT ? cfg.x_tile() : TAB_ROW_BYTES / 4;
    using Tab                 = int[2][TAB_ROWS][COLS];

    // Build state in one register: filter tile, buffer, X tiles ahead and the mailbox parity.
    static constexpr int BUILD_X      = 1 << 9;
    static constexpr int BUILD_PARITY = 1 << 12;

    __device__ __forceinline__ OffsetTable(const Args& args, Tab& tab)
        : args_(args)
        , tab_(tab)
        , wave_(bunnies::wave_id())
    {
    }

    // A pixel's column: a staging thread's pixels of both rounds are adjacent. A csplit tile's
    // column is its pixel.
    // Unsigned: signed, it compiled to 16-bit ops whose shift VGPR stayed live in the main loop.
    __device__ __forceinline__ static int col(int x)
    {
        if constexpr(T::CSPLIT)
            return x;
        const auto u = static_cast<unsigned>(x);
        return static_cast<int>(((u / (2 * ROWS_PER_PASS)) * ROWS_PER_PASS + u % ROWS_PER_PASS) *
                                    2 +
                                (u / ROWS_PER_PASS) % 2);
    }

    // Fprop skips geom, which moved its allocation (Offset range in depth).
    // hot, if given: rebase_in and the q and p divisors come from SGPRs (2D, unphased).
    __device__ __forceinline__ static Base
    origin(const Args& args, const Tile& t, const Hot* hot = nullptr)
    {
        Pars& kp = args.cold();
        if constexpr(T::DEPTH)
        {
            const auto& dp = kp.dep;
            int p          = kp.p;
            int ph         = kp.ph;
            int e          = dp.e;
            int pd         = dp.pd;
            int sd         = dp.sd;
            Div* q_div     = &kp.q_div;
            Div* p_div     = &kp.p_div;
            Div* e_div     = &dp.e_div;
            if constexpr(T::PHASED)
            {
                const Geom g = args.geom();
                const int f  = args.seg().phase;
                p            = g.p;
                ph           = g.ph;
                q_div        = g.q_div;
                p_div        = g.p_div;
                e            = dp.pe[f];
                pd           = dp.ppd[f];
                sd           = 1;
                e_div        = &dp.pe_div[f];
            }
            const int row_x0 = __builtin_amdgcn_readfirstlane(static_cast<int>(
                detail::magic_div(*q_div, static_cast<uint32_t>(t.x0 * kp.rebase_in))));
            const int plane0 = __builtin_amdgcn_readfirstlane(
                static_cast<int>(detail::magic_div(*p_div, static_cast<uint32_t>(row_x0))));
            const int img0 = __builtin_amdgcn_readfirstlane(
                static_cast<int>(detail::magic_div(*e_div, static_cast<uint32_t>(plane0))));
            const int d0     = (plane0 - img0 * e) * sd - pd;
            const int h0     = (row_x0 - plane0 * p) * kp.sh - ph;
            const int d_base = std::min(std::max(d0, 0), static_cast<int>(dp.d));
            const int h_base =
                d0 < 0 || d0 >= dp.d ? 0 : std::min(std::max(h0, 0), static_cast<int>(kp.h));
            return Base{img0,
                        h_base,
                        ((static_cast<int64_t>(img0) * dp.d + d_base) * kp.h + h_base) * kp.w *
                            kp.c,
                        d_base};
        }
        if constexpr(T::PHASED)
        {
            const Geom g     = args.geom();
            const int row_x0 = __builtin_amdgcn_readfirstlane(static_cast<int>(
                detail::magic_div(*g.q_div, static_cast<uint32_t>(t.x0 * kp.rebase_in))));
            const int img0   = __builtin_amdgcn_readfirstlane(
                static_cast<int>(detail::magic_div(*g.p_div, static_cast<uint32_t>(row_x0))));
            const int h_base =
                std::min(std::max((row_x0 - img0 * g.p) * kp.sh - g.ph, 0), static_cast<int>(kp.h));
            return Base{img0, h_base, (static_cast<int64_t>(img0) * kp.h + h_base) * kp.w * kp.c};
        }
        const auto x0    = static_cast<uint32_t>(t.x0 * (hot ? hot->rebase_in : kp.rebase_in));
        const int row_x0 = __builtin_amdgcn_readfirstlane(
            static_cast<int>(hot ? hot->q_div.div(x0) : detail::magic_div(kp.q_div, x0)));
        const auto row = static_cast<uint32_t>(row_x0);
        const int img0 = __builtin_amdgcn_readfirstlane(
            static_cast<int>(hot ? hot->p_div.div(row) : detail::magic_div(kp.p_div, row)));
        const int h_base =
            std::min(std::max((row_x0 - img0 * kp.p) * kp.sh - kp.ph, 0), static_cast<int>(kp.h));
        return Base{img0, h_base, (static_cast<int64_t>(img0) * kp.h + h_base) * kp.w * kp.c};
    }

    // Live filter tiles [cur_lo, cur_hi) of the current X tile, and the next X tile's first.
    __device__ __forceinline__ int cur_lo() const { return cur_lo_; }
    __device__ __forceinline__ int cur_hi() const { return cur_hi_; }
    __device__ __forceinline__ int next_lo() const { return next_lo_; }

    // The mailbox slot the epilogue posts its claim to; it reads the other.
    __device__ __forceinline__ int parity() const
    {
        return (static_cast<unsigned>(build_) / BUILD_PARITY) & 1;
    }

    // Enters the next X tile: the X tile being built is one fewer ahead; the mailbox slot flips.
    __device__ __forceinline__ void next_tile() { build_ = (build_ - BUILD_X) ^ BUILD_PARITY; }

    // Sets the live filter-tile bounds on entering an X tile; returns the first live filter tile.
    template <class TileAhead>
    __device__ __forceinline__ int enter_tile(const TileAhead& tile_ahead)
    {
        const Range r = live_ftiles(tile_ahead(0));
        cur_lo_       = r.lo;
        cur_hi_       = r.hi;
        next_lo_      = live_ftiles(tile_ahead(1)).lo;
        return r.lo;
    }

    // Builds the next filter tile's table; after an X tile's last, moves to the next X tile.
    // ALL: with one filter tile, the calling thread builds every position of its pixel.
    // origin0: the origin of the tile built, if the caller computed it; null computes it here.
    // has_tile(ahead): whether the X tile `ahead` past the current exists; read only by SKIP.
    template <bool ALL = false, class TileAhead, class HasTile = AnyTile>
    __device__ __forceinline__ void build_next(const TileAhead& tile_ahead,
                                               const Base* origin0     = nullptr,
                                               const HasTile& has_tile = {})
    {
        // Only conv3d on positions inner nulls the tables past the workgroup's last X tile.
        // Its builds cost the most; with the has_tile test, the other K256 kernels spilled.
        if constexpr(SKIP)
        {
            const int ahead   = (build_ >> 9) & 7;
            const int k       = build_ & 0xff;
            const bool exists = has_tile(ahead);
            Tile tb{};
            if(k == 0)
            {
                // An X tile past the workgroup's last gets a null table for every filter tile.
                Range r{0, args_.cold().n_ftiles};
                if(exists)
                {
                    tb = tile_ahead(ahead);
                    r  = live_ftiles(tb);
                }
                build_lo_ = r.lo;
                build_hi_ = r.hi;
            }
            const int f = build_lo_ + k;
            if(exists)
                build_plane(f, (build_ >> 8) & 1, tb, k == 0);
            else
                fill_null((build_ >> 8) & 1);
            build_ = f + 1 == build_hi_ ? ((build_ & ~0x1ff) + BUILD_X) | (~build_ & 0x100)
                                        : build_ + 1 ^ 0x100;
            return;
        }
        // Read after the build, and not with one filter tile: either moved fprop's tail allocation.
        const int f = build_ & 0xff;
        build<ALL>(f, (build_ >> 8) & 1, tile_ahead((build_ >> 9) & 7), origin0);
        bool last;
        if constexpr(T::PHASED)
            last = f + 1 == args_.seg().n_ftiles;
        else
            last = T::ONE_FTILE || f + 1 == args_.cold().n_ftiles;
        build_ = last ? ((build_ & ~0x1ff) + BUILD_X) | (~build_ & 0x100) : build_ + 1 ^ 0x100;
    }

    // Moves to the next X tile's table as build_next does, for a thread that builds none of it.
    __device__ __forceinline__ void skip_next()
    {
        static_assert(T::ONE_FTILE);
        build_ = ((build_ & ~0x1ff) + BUILD_X) | (~build_ & 0x100);
    }

private:
    // Entry (row, c) of buffer buf. Typed indexing tab_[buf][row][c] adds an s_waitcnt vmcnt
    // before the store; see bunnies::tensor_view::advance.
    //
    // The element index is scaled to bytes in int: make_memref's 64-bit byte offset cost a VALU
    // instruction per entry address.
    __device__ __forceinline__ int& entry_at(int buf, int row, int c) const
    {
        constexpr int X = COLS;
        const int i = bunnies::make_view<3>(0, {2, TAB_ROWS, X}, {TAB_ROWS * X, X, 1})(buf, row, c);
        return *reinterpret_cast<int*>(reinterpret_cast<char*>(&tab_[0][0][0]) + i * 4);
    }

    // NULL_OFFSET in the thread's entries of buffer buf, split between threads as build splits
    // them.
    __device__ __forceinline__ void fill_null(int buf) const
    {
        const int t        = bunnies::pin_sgpr(wave_) * WAVE_SIZE + bunnies::volatile_lane_id();
        const int tab_part = __builtin_amdgcn_readfirstlane(t / cfg.x_tile());
        const int dst_col  = col(t % cfg.x_tile());
#pragma unroll
        for(int p = 0; p < T::N; ++p)
            if((p & 1) == tab_part)
                entry_at(buf, p, dst_col) = NULL_OFFSET;
    }

    // The first filter tile's entries from tap_h, tap_w and in_row.
    //
    // "Building a table", step 5, in docs/algorithms/details/direct/direct-flex-offset-tables.md.
    // Loads every position's kernel arguments before the first store: interleaved, each entry
    // reloads them behind an s_waitcnt.
    template <bool ALL = false>
    __device__ __forceinline__ void
    write_taps(Pars& kp, int buf, int dst_col, int tab_part, bool live, int h0, int w0, int base)
        const
    {
        const int h = kp.h, w = kp.w;
        const int base_bytes = base * T::elem_bytes;
        // Pinned: unpinned, the compiler hoisted NULL_OFFSET out of the main loop and held a VGPR
        // through it.
        const int null_off = bunnies::pin_vgpr(NULL_OFFSET);
        // A pixel not live gets an h0 that fails every image-row test, whatever tap_h it adds.
        const int top     = live ? h0 : INT_MIN / 2;
        constexpr int PER = ALL ? T::N : (T::N + 1) / 2;
        int tap_h[PER], tap_w[PER], in_row[PER];
#pragma unroll
        for(int i = 0; i < PER; ++i)
        {
            const int p = ALL ? i : 2 * i + tab_part;
            tap_h[i]    = kp.tap_h[p];
            tap_w[i]    = kp.tap_w[p];
            in_row[i]   = kp.seq[p].in_row;
        }
#pragma unroll
        for(int i = 0; i < PER; ++i)
        {
            const int p = ALL ? i : 2 * i + tab_part;
            if(ALL || 2 * i + 1 < T::N || p < T::N)
            {
                const int hh     = top + tap_h[i];
                const int ww     = w0 + tap_w[i];
                const bool in_im = (static_cast<unsigned>(hh) < static_cast<unsigned>(h)) &
                                   (static_cast<unsigned>(ww) < static_cast<unsigned>(w));
                const int entry           = base_bytes + in_row[i];
                entry_at(buf, p, dst_col) = in_im ? entry : null_off;
            }
        }
    }

    // A conv3d pixel's top-left tap and its offset in elements from the X tile's origin.
    struct DepthPixel
    {
        int d0, h0, w0, base;
        bool live; // m < rows
    };

    // Decodes output pixel m of a conv3d (Building a table, steps 1 to 3, in
    // docs/algorithms/details/direct/direct-flex-offset-tables.md).
    __device__ __forceinline__ DepthPixel depth_pixel(Pars& kp,
                                                      const Geom& g,
                                                      int m,
                                                      const Base& base0) const
    {
        const auto& dp = kp.dep;
        int p          = kp.p;
        int q          = kp.q;
        int ph         = kp.ph;
        int pw         = kp.pw;
        int rows       = kp.rows;
        int e          = dp.e;
        int pd         = dp.pd;
        int sd         = dp.sd;
        Div* q_div     = &kp.q_div;
        Div* p_div     = &kp.p_div;
        Div* e_div     = &dp.e_div;
        if constexpr(T::PHASED)
        {
            const int ph_i = args_.seg().phase;
            p              = g.p;
            q              = g.q;
            ph             = g.ph;
            pw             = g.pw;
            rows           = g.rows;
            q_div          = g.q_div;
            p_div          = g.p_div;
            e              = dp.pe[ph_i];
            pd             = dp.ppd[ph_i];
            sd             = 1;
            e_div          = &dp.pe_div[ph_i];
        }
        DepthPixel px{0, 0, 0, 0, m < rows};
        if(px.live)
        {
            const int mq  = static_cast<int>(detail::magic_div(*q_div, static_cast<uint32_t>(m)));
            const int q_i = m - mq * q;
            const int mp  = static_cast<int>(detail::magic_div(*p_div, static_cast<uint32_t>(mq)));
            const int p_i = mq - mp * p;
            const int n_i = static_cast<int>(detail::magic_div(*e_div, static_cast<uint32_t>(mp)));
            const int e_i = mp - n_i * e;
            px.d0         = e_i * sd - pd;
            px.h0         = p_i * kp.sh - ph;
            px.w0         = q_i * kp.sw - pw;
            px.base       = ((((n_i - base0.img0) * dp.d + px.d0 - base0.d_base) * kp.h + px.h0 -
                        base0.h_base) *
                           kp.w +
                       px.w0) *
                      kp.c;
        }
        return px;
    }

    // Filter tile f of a conv3d on positions inner: depth plane f, with tile 0's Kh × Kw positions.
    //
    // An X tile's first build keeps each pixel's decode in buffer 0 for its later builds; see
    // "Other uses of the tables' LDS" in docs/algorithms/details/direct/direct-flex-lds.md.
    __device__ __forceinline__ void build_plane(int f, int buf, const Tile& tile, bool first) const
    {
        static_assert(T::N > 0 && T::N + 4 <= TAB_ROWS, "rows N to N + 3 must fit in the table");
        Pars& kp            = args_.cold();
        const auto& dp      = kp.dep;
        const int t         = bunnies::pin_sgpr(wave_) * WAVE_SIZE + bunnies::volatile_lane_id();
        const int tab_pixel = t % cfg.x_tile();
        const int tab_part  = __builtin_amdgcn_readfirstlane(t / cfg.x_tile());
        const int dst_col   = col(tab_pixel);
        int d0, h0, w0, base;
        if(first)
        {
            const Base base0 = origin(args_, tile);
            const DepthPixel px =
                depth_pixel(kp, Geom{}, bunnies::pin_sgpr(tile.x0) + tab_pixel, base0);
            // A pixel past rows gets a d0 that fails every depth test.
            d0                             = px.live ? px.d0 : INT_MIN / 2;
            h0                             = px.h0;
            w0                             = px.w0;
            base                           = px.base;
            entry_at(0, T::N, dst_col)     = d0;
            entry_at(0, T::N + 1, dst_col) = h0;
            entry_at(0, T::N + 2, dst_col) = w0;
            entry_at(0, T::N + 3, dst_col) = base;
        }
        else
        {
            d0   = entry_at(0, T::N, dst_col);
            h0   = entry_at(0, T::N + 1, dst_col);
            w0   = entry_at(0, T::N + 2, dst_col);
            base = entry_at(0, T::N + 3, dst_col);
        }
        const int dd    = d0 + f * dp.dd;
        const bool in_d = static_cast<unsigned>(dd) < static_cast<unsigned>(dp.d);
        write_taps(kp, buf, dst_col, tab_part, in_d, h0, w0, base + f * dp.dd * kp.h * kp.w * kp.c);
    }

    // Unless ALL, two threads per pixel take even and odd positions, split by wave group to stay
    // uniform.
    template <bool ALL = false>
    __device__ __forceinline__ void
    build(int f, int buf, const Tile& tile, const Base* origin0) const
    {
        Pars& kp     = args_.cold();
        const int x0 = tile.x0;
        Geom g{};
        if constexpr(T::PHASED)
        {
            g = args_.geom();
            f += args_.seg().ftile0;
        }
        // Lane from volatile_lane_id: a threadIdx.x read in the main loop holds v0 through the
        // loop.
        const int t         = bunnies::pin_sgpr(wave_) * WAVE_SIZE + bunnies::volatile_lane_id();
        const int tab_pixel = t % cfg.x_tile();
        const int tab_part  = __builtin_amdgcn_readfirstlane(t / cfg.x_tile());
        const uint32_t d    = kp.ftiles[f];
        const int r0        = d & 0xff;
        const int s0        = (d >> 8) & 0xff;
        const int nr        = (d >> 16) & 0xff;
        const int nc        = d >> 24;
        const Base base0    = origin0 ? *origin0 : origin(args_, tile);
        // Not hoisted out of the filter-tile loop: live through the main loop, it spilled K256.
        const int m = bunnies::pin_sgpr(x0) + tab_pixel;
        // A conv3d also decodes the output plane and tests the input plane.
        if constexpr(T::DEPTH)
        {
            const auto& dp      = kp.dep;
            const int khp       = T::PHASED ? dp.pkh[args_.seg().phase] : dp.kh;
            const DepthPixel px = depth_pixel(kp, g, m, base0);
            const int d0 = px.d0, h0 = px.h0, w0 = px.w0, base = px.base;
            const bool live   = px.live;
            const int dst_col = col(tab_pixel);
            const uint32_t tr = dp.ftile_tr[f];
            int tap           = tr & 0xff;
            int row           = tr >> 8;
            int local         = 0;
            for(int lr = 0; lr < nr; ++lr)
            {
                for(int ls = 0; ls < nc; ++ls, ++local)
                    if((local & 1) == tab_part)
                    {
                        const int s  = s0 + ls;
                        const int dd = d0 + tap * dp.dd;
                        const int hh = h0 + row * kp.dh;
                        const int ww = w0 + s * kp.dw;
                        const bool in_im =
                            live && static_cast<unsigned>(dd) < static_cast<unsigned>(dp.d) &&
                            static_cast<unsigned>(hh) < static_cast<unsigned>(kp.h) &&
                            static_cast<unsigned>(ww) < static_cast<unsigned>(kp.w);
                        entry_at(buf, local, dst_col) =
                            in_im
                                ? (base +
                                   ((tap * dp.dd * kp.h + row * kp.dh) * kp.w + s * kp.dw) * kp.c) *
                                      T::elem_bytes
                                : NULL_OFFSET;
                    }
                if(++row == khp)
                {
                    row = 0;
                    ++tap;
                }
            }
            return;
        }
        int h0 = 0, w0 = 0, base = 0;
        const bool live_row = m < (T::PHASED ? g.rows : kp.rows);
        if(live_row)
        {
            int n_i, p_i, q_i;
            if constexpr(T::PHASED)
            {
                const int mq =
                    static_cast<int>(detail::magic_div(*g.q_div, static_cast<uint32_t>(m)));
                q_i = m - mq * g.q;
                n_i = static_cast<int>(detail::magic_div(*g.p_div, static_cast<uint32_t>(mq)));
                p_i = mq - n_i * g.p;
                h0  = p_i * kp.sh - g.ph;
                w0  = q_i * kp.sw - g.pw;
            }
            else
            {
                const int mq =
                    static_cast<int>(detail::magic_div(kp.q_div, static_cast<uint32_t>(m)));
                q_i = m - mq * kp.q;
                n_i = static_cast<int>(detail::magic_div(kp.p_div, static_cast<uint32_t>(mq)));
                p_i = mq - n_i * kp.p;
                h0  = p_i * kp.sh - kp.ph;
                w0  = q_i * kp.sw - kp.pw;
            }
            base = (((n_i - base0.img0) * kp.h + h0 - base0.h_base) * kp.w + w0) * kp.c;
        }
        const int dst_col = col(tab_pixel);
        int local         = 0;
        if constexpr(T::ONE_FTILE)
        {
            // With one filter tile the build runs only in the prologue and the epilogue phase.
            // Built in a sweep step, as with several filter tiles, the kernel arguments write_taps
            // holds spilled.
            write_taps<ALL>(kp, buf, dst_col, tab_part, live_row, h0, w0, base);
        }
        else
        {
            for(int lr = 0; lr < nr; ++lr)
                for(int ls = 0; ls < nc; ++ls, ++local)
                    if((local & 1) == tab_part)
                    {
                        const int r  = r0 + lr;
                        const int s  = s0 + ls;
                        const int hh = h0 + r * kp.dh;
                        const int ww = w0 + s * kp.dw;
                        const bool in_im =
                            live_row && static_cast<unsigned>(hh) < static_cast<unsigned>(kp.h) &&
                            static_cast<unsigned>(ww) < static_cast<unsigned>(kp.w);
                        entry_at(buf, local, dst_col) =
                            in_im ? (base + (r * kp.dh * kp.w + s * kp.dw) * kp.c) * T::elem_bytes
                                  : NULL_OFFSET;
                    }
        }
    }

    // Depth taps live in X tile t, at least one (Live depth taps).
    __device__ __forceinline__ Range live_ftiles(const Tile& t) const
    {
        Pars& kp       = args_.cold();
        const auto& dp = kp.dep;
        const int n_ft = kp.n_ftiles;
        const int rows = kp.rows;
        if(dp.all_live)
            return Range{0, n_ft};
        auto plane_of = [&](int x) {
            const int row = static_cast<int>(detail::magic_div(kp.q_div, static_cast<uint32_t>(x)));
            return __builtin_amdgcn_readfirstlane(
                static_cast<int>(detail::magic_div(kp.p_div, static_cast<uint32_t>(row))));
        };
        const int pl0 = plane_of(t.x0);
        const int pl1 = plane_of(std::min(t.x0 + cfg.x_tile(), rows) - 1);
        const int n0  = static_cast<int>(detail::magic_div(dp.e_div, static_cast<uint32_t>(pl0)));
        const int n1  = static_cast<int>(detail::magic_div(dp.e_div, static_cast<uint32_t>(pl1)));
        // t * dd lies in [pd - e1 * sd, D - 1 + pd - e0 * sd].
        const int lo_num = dp.pd - (pl1 - n1 * dp.e) * dp.sd;
        const int hi_num = dp.d - 1 + dp.pd - (pl0 - n0 * dp.e) * dp.sd;
        auto div_dd      = [&](int x) {
            return static_cast<int>(detail::magic_div(dp.dd_div, static_cast<uint32_t>(x)));
        };
        int lo = lo_num <= 0 ? 0 : div_dd(lo_num + dp.dd - 1);
        int hi = hi_num < 0 ? 0 : std::min(div_dd(hi_num) + 1, n_ft);
        if(n0 != n1)
        {
            lo = 0;
            hi = n_ft;
        }
        lo = std::min(lo, n_ft - 1);
        hi = std::max(hi, lo + 1);
        return Range{__builtin_amdgcn_readfirstlane(lo), __builtin_amdgcn_readfirstlane(hi)};
    }

    const Args& args_;
    Tab& tab_;
    const int wave_;
    int build_    = 0;
    int cur_lo_   = 0;
    int cur_hi_   = 0;
    int next_lo_  = 0;
    int build_lo_ = 0;
    int build_hi_ = 0;
};

} // namespace hipconv::cdna4::direct_flex
