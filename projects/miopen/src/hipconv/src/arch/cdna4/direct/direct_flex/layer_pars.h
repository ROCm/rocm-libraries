#pragma once

// Kernel arguments: LayerPars and the layer geometry, filter tiles and offset tables it holds.
// Parenthesized section names refer to docs/algorithms/direct/direct-flex-cdna4.md.

#include "config.h"
#include "config_table.h"
#include "magic_division.h"
#include "persistent_grid.h"
#include "types.h"
#include "hipconv/conv_params.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>

namespace hipconv::cdna4::direct_flex
{

// The layer geometry the kernel indexes with, flattened to what the loops need.
// The fields a workgroup reads before its main loop come first, to share scalar-cache lines.
struct LayerPars
{
    int h, w, c;
    int k;
    int kh, kw;
    int p, q;
    int sh, sw;
    int dh, dw;
    int ph, pw;
    int rows;    // n * p * q, the GEMM's M
    int chunks;  // C_CHUNK-channel chunks per filter position
    int steps;   // kh * kw * chunks, the reduction steps a tile runs
    int tiles_k; // K tiles per X tile in the flattened grid
    // Tile order: groups of k_group K tiles, X tiles within a group, K tiles fastest (Grid).
    int x_tiles;
    int tiles;
    int k_group;
    int k_full;
    int in_full;
    MagicDiv group_div; // k_group * x_tiles
    MagicDiv kg_div;    // k_group
    MagicDiv kr_div;    // tiles_k - k_full, or 1
    int blocks;         // workgroups in the grid
    // Split-K: `pieces` chunk ranges per tile, one per workgroup, each to its own fp32 copy.
    int pieces;
    int split_tiles;
    int c_last; // channels in the layer's last chunk
    int n_ftiles;
    // Dynamic order's counters and mailboxes; null for the static order (Dynamic tile order).
    uint32_t* ws;
    float* partials;
    // Host-computed: a device divide would put the descriptors in VGPRs and waterfall every load.
    int64_t in_elems;
    int wei_elems;
    int64_t out_elems;
    int rebase_in; // 1 if the input passes the 32-bit range, which rebases its descriptor
    MagicDiv q_div;
    MagicDiv p_div;
    MagicDiv chunk_div;
    MagicDiv kw_div;
    uint32_t ftiles[MAX_FTILES]; // first row, first column, rows, columns: a byte each
    // Filter positions in loop order, in filter tiles of at most TAB_POSITIONS (Filter tiles).
    // Entries past the filter repeat the first ones, for the workgroup's next tile.
    struct Seq
    {
        int in_row;
        int wgt_off;
    };
    Seq seq[MAX_SEQ];
    int dyn_first[persistent::NUM_XCD];
    int run_end[persistent::NUM_XCD];
    // Phased dgrad's phases and segments; unset in fprop (Strided dgrad, Partitioned grid).
    struct Phase
    {
        int p, q;   // output rows and columns
        int ph, pw; // padding of the convolution over dY
        int rows;   // n * p * q
        int out0;   // dX pixel of the phase's first, within an image
        int x_tiles;
        int in_full; // k_full * x_tiles
        MagicDiv q_div;
        MagicDiv p_div;
        MagicDiv group_div; // k_group * x_tiles
    };
    struct Seg
    {
        int phase;
        int ftile0, n_ftiles; // its filter tiles in ftiles
        int seq0, taps;       // taps 0: the phase's pixels take no tap
        int tiles;            // in the tile order: x_tiles * tiles_k, times pieces in a split
        int copy0;
        // Workgroups in `runs` runs; with eight, a workgroup's run is its XCD (Partitioned grid).
        int wg0, wgs;
        int runs, run_q, run_r, wg_q, wg_r;
        int dyn; // claims its tiles at run time
    };
    Phase phases[MAX_PHASES];
    Seg segs[MAX_SEGS];
    int n_segs;
    int strided;                 // the phases write every out_sh-th row: out_sh * out_sw > 1
    int out_hw, out_row, out_sw; // dX's pixels per image, per out_sh rows, and its column stride
    // Conv3d, read by the depth kernels only (Depth as filter rows, Dgrad in depth).
    struct Depth
    {
        int d, e;   // input and output depth
        int kh;     // filter rows per depth plane
        int sd, dd; // depth stride and dilation
        int pd;     // depth padding
        MagicDiv e_div;
        // A filter tile's first row as its depth tap and its row in the plane, a byte each.
        uint16_t ftile_tr[MAX_FTILES];
        // Per phase: output depth, depth padding, filter rows per plane.
        int pe[MAX_PHASES], ppd[MAX_PHASES], pkh[MAX_PHASES];
        MagicDiv pe_div[MAX_PHASES];
        int out_plane; // dX pixels between a phase's consecutive depth planes
        MagicDiv dd_div;
        int all_live; // every output plane's depth taps all read an input plane
    };
    Depth dep;
    // Positions inner: r * dh and s * dw of each position in the first filter tile; read by
    // ONE_FTILE and conv3d builds.
    int tap_h[TAB_ROWS];
    int tap_w[TAB_ROWS];
};

// A layer's output positions and filter positions, whatever its dimensions.
inline int out_depth(const ConvParams& par)
{
    return par.dims == 3 ? par.e : 1;
}
inline int64_t out_rows(const ConvParams& par)
{
    return static_cast<int64_t>(par.n) * out_depth(par) * par.p * par.q;
}
inline int filter_taps(const ConvParams& par)
{
    return (par.dims == 3 ? par.kd : 1) * par.kh * par.kw;
}

// An input offset table's row: one 32-bit offset per pixel of the X tile, and its buffer.
constexpr int TAB_ROW_BYTES = 256 * 4;
constexpr int TAB_BYTES     = TAB_ROWS * TAB_ROW_BYTES;

// An out-of-image pixel's offset: past any buffer's NUM_RECORDS with a granule offset added.
constexpr int NULL_OFFSET = INT32_MIN;

// Positions per filter tile with positions inner; 0 keeps positions outer (Positions inner).
inline int sweep_positions(const ConvParams& par)
{
    // Conv3d's depth kernels sweep 3x3 planes, each a filter tile.
    if(par.dims == 3)
        return par.kd > 1 && par.kh * par.kw == DEPTH_SWEEP ? DEPTH_SWEEP : 0;
    const int n = par.kh * par.kw <= TAB_ROWS ? par.kh * par.kw : par.kw;
    for(const int s : SWEEPS)
        if(s == n)
            return n;
    return 0;
}

// The fprop layer a unit-stride dgrad runs as: rotated filter, KRSC weights (Dgrad).
inline ConvParams as_fprop(const ConvParams& par)
{
    if(par.direction != Direction::Dgrad)
        return par;
    ConvParams d = par;
    d.direction  = Direction::Fprop;
    d.h          = par.p;
    d.w          = par.q;
    d.p          = par.h;
    d.q          = par.w;
    d.c          = par.k;
    d.k          = par.c;
    d.pad_h      = (par.kh - 1) * par.dilation_h - par.pad_h;
    d.pad_w      = (par.kw - 1) * par.dilation_w - par.pad_w;
    if(par.dims == 3)
    {
        d.d     = par.e;
        d.e     = par.d;
        d.pad_d = (par.kd - 1) * par.dilation_d - par.pad_d;
    }
    return d;
}

// The compiled positions-inner sweep that fits the layer, or 0; phased() decides if dgrad runs it.
inline int sweep_of(const ConvParams& layer)
{
    const ConvParams par = as_fprop(layer);
    const int n          = sweep_positions(par);
    if(n == 0 || par.dims == 3)
        return n;
    const bool dgrad  = layer.direction == Direction::Dgrad;
    const bool narrow = !dgrad && par.k % GRANULE != 0;
    const bool one    = par.kh * par.kw == n;
    return inner_compiled(dgrad, narrow, par.c % C_CHUNK != 0, n, one) ? n : 0;
}

// A 3D filter as the 2D filter of Kd * Kh rows its KRSC weights hold (Depth as filter rows).
inline ConvParams filter_rows(const ConvParams& par)
{
    ConvParams f = par;
    if(par.dims == 3)
    {
        f.kh = par.kd * par.kh;
        f.kd = 1;
    }
    return f;
}

// Byte offset of position (r, s) in a weight row, dgrad's filter rotated (Weights in KRSC).
inline int wgt_position_bytes(const ConvParams& par, bool dgrad, int r, int s)
{
    return dgrad ? ((par.kh - 1 - r) * par.kw + par.kw - 1 - s) * par.k * 2
                 : (r * par.kw + s) * par.c * 2;
}

// Filter tile shape: columns, then rows, in equal parts of at most TAB_POSITIONS (Filter tiles).
struct FtileShape
{
    int rows, cols;
};
inline FtileShape ftile_shape(int nr, int nc)
{
    const int col_parts = (nc + TAB_POSITIONS - 1) / TAB_POSITIONS;
    const int cols      = (nc + col_parts - 1) / col_parts;
    const int row_parts = (nr + TAB_POSITIONS / cols - 1) / (TAB_POSITIONS / cols);
    return {(nr + row_parts - 1) / row_parts, cols};
}

// Writes the filter tiles of the nr x nc positions at (r0, c0) to ftiles[f...]; returns the next f.
inline int cut_ftiles(uint32_t* ftiles, int f, int r0, int c0, int nr, int nc)
{
    const FtileShape t = ftile_shape(nr, nc);
    for(int r = 0; r < nr; r += t.rows)
        for(int c = 0; c < nc; c += t.cols)
            ftiles[f++] = (r0 + r) | ((c0 + c) << 8) | (std::min(t.rows, nr - r) << 16) |
                          (std::min(t.cols, nc - c) << 24);
    return f;
}

// Kernel arguments; sweep_n is 0 for positions outer, pieces the split-K ranges per tile.
inline LayerPars
layer_pars(const Config& cfg, const ConvParams& layer, int sweep_n, int pieces, bool dgrad)
{
    // A 3D layer runs as its filter rows, with the output depth in the rows of X.
    const ConvParams par = filter_rows(layer);
    const int depth      = out_depth(layer);
    const int all_chunks = (par.c + C_CHUNK - 1) / C_CHUNK;
    const int chunks     = all_chunks / pieces;
    LayerPars pars{par.h,
                   par.w,
                   par.c,
                   par.k,
                   par.kh,
                   par.kw,
                   par.p,
                   par.q,
                   par.stride_h,
                   par.stride_w,
                   par.dilation_h,
                   par.dilation_w,
                   par.pad_h,
                   par.pad_w,
                   par.n * depth * par.p * par.q,
                   chunks,
                   par.kh * par.kw * chunks,
                   (par.k + cfg.k_tile() - 1) / cfg.k_tile(),
                   0,
                   0,
                   0,
                   0,
                   0,
                   {},
                   {},
                   {},
                   0,
                   pieces,
                   0,
                   par.c - (all_chunks - 1) * C_CHUNK,
                   0,
                   nullptr,
                   nullptr,
                   static_cast<int64_t>(par.n) * (layer.dims == 3 ? par.d : 1) * par.h * par.w *
                       par.c,
                   par.k * par.kh * par.kw * par.c,
                   static_cast<int64_t>(par.n) * depth * par.p * par.q * par.k,
                   ConvSize(par).input_bytes() > INT32_MAX,
                   MagicDiv(static_cast<uint32_t>(par.q)),
                   MagicDiv(static_cast<uint32_t>(par.p)),
                   MagicDiv(static_cast<uint32_t>(chunks)),
                   MagicDiv(static_cast<uint32_t>(par.kw))};

    // Positions inner: weights carry most L2 misses, so a K group of 2 shares each weight line.
    pars.x_tiles     = (pars.rows + cfg.x_tile() - 1) / cfg.x_tile();
    pars.split_tiles = pars.x_tiles * pars.tiles_k;
    pars.tiles       = pars.split_tiles * pieces;
    pars.k_group     = std::min(pars.tiles_k, sweep_n > 0 ? 2 : 4);
    pars.k_full      = pars.tiles_k / pars.k_group * pars.k_group;
    pars.in_full     = pars.k_full * pars.x_tiles;
    pars.group_div   = MagicDiv(static_cast<uint32_t>(pars.k_group * pars.x_tiles));
    pars.kg_div      = MagicDiv(static_cast<uint32_t>(pars.k_group));
    pars.kr_div      = MagicDiv(static_cast<uint32_t>(std::max(pars.tiles_k - pars.k_full, 1)));

    // A filter tile's first row as its depth tap and its row in the plane.
    auto depth_tiles = [&] {
        if(layer.dims != 3)
            return;
        pars.dep          = {layer.d,
                             layer.e,
                             layer.kh,
                             layer.stride_d,
                             layer.dilation_d,
                             layer.pad_d,
                             MagicDiv(static_cast<uint32_t>(layer.e))};
        pars.dep.dd_div   = MagicDiv(static_cast<uint32_t>(layer.dilation_d));
        const int last_hi = layer.d - 1 + layer.pad_d - (layer.e - 1) * layer.stride_d;
        pars.dep.all_live =
            layer.pad_d <= 0 && last_hi >= 0 && last_hi / layer.dilation_d + 1 >= layer.kd;
        for(int f = 0; f < pars.n_ftiles; ++f)
        {
            const int u          = pars.ftiles[f] & 0xff;
            pars.dep.ftile_tr[f] = static_cast<uint16_t>(u / layer.kh | (u % layer.kh) << 8);
        }
    };

    if(sweep_n > 0)
    {
        // Positions inner: filter tiles of sweep_n positions, each entry a tap and a weight offset.
        const bool whole    = par.kh * par.kw == sweep_n;
        const int tile_rows = layer.dims == 3 ? layer.kh : 1;
        pars.n_ftiles       = whole ? 1 : par.kh / tile_rows;
        for(int f = 0; f < pars.n_ftiles; ++f)
            pars.ftiles[f] =
                whole ? 0u | (par.kh << 16) | (par.kw << 24)
                      : static_cast<uint32_t>(f * tile_rows) | (tile_rows << 16) | (par.kw << 24);
        depth_tiles();
        // in_row: the position's input byte offset from its pixel's top-left tap.
        int q = 0;
        for(int r = 0; r < par.kh; ++r)
            for(int s = 0; s < par.kw; ++s)
            {
                if(q < sweep_n)
                {
                    pars.tap_h[q] = r * par.dilation_h;
                    pars.tap_w[q] = s * par.dilation_w;
                }
                const int tap = (r * par.dilation_h * par.w + s * par.dilation_w) * par.c * 2;
                pars.seq[q++] = {tap, wgt_position_bytes(par, dgrad, r, s)};
            }
        for(int j = 0; q < MAX_SEQ; ++q, ++j)
            pars.seq[q] = j < sweep_n ? pars.seq[j] : LayerPars::Seq{0, 0};
        return pars;
    }

    pars.n_ftiles = cut_ftiles(pars.ftiles, 0, 0, 0, par.kh, par.kw);
    int q         = 0;
    for(int f = 0; f < pars.n_ftiles; ++f)
    {
        const uint32_t d = pars.ftiles[f];
        const int r0 = d & 0xff, s0 = (d >> 8) & 0xff, nr = (d >> 16) & 0xff, nc = d >> 24;
        for(int local = 0; local < nr * nc; ++local)
        {
            const int r   = r0 + local / nc;
            const int s   = s0 + local % nc;
            pars.seq[q++] = {(f % 2) * TAB_BYTES + local * TAB_ROW_BYTES |
                                 (local == 0 ? 1 << 16 : 0),
                             wgt_position_bytes(par, dgrad, r, s)};
        }
    }
    for(int j = 0; q < MAX_SEQ; ++q, ++j)
        pars.seq[q] = j < 3 ? pars.seq[j] : LayerPars::Seq{0, 0};
    depth_tiles();
    return pars;
}

// Input elements an X tile's rebased descriptor must reach (Offset range).
inline int64_t span_input(const ConvParams& par)
{
    int x_tile = 0;
    for(const Config& cfg : tiles)
        x_tile = std::max(x_tile, cfg.x_tile());
    const int64_t rows = (x_tile - 1) / par.q + 2;
    if(par.dims == 3)
    {
        // Rows of W * C over (n, d, h) (Offset range in depth).
        const int64_t row_end = (par.p - 1) * static_cast<int64_t>(par.stride_h);
        const int64_t advance =
            std::max({static_cast<int64_t>(par.stride_h),
                      static_cast<int64_t>(par.stride_d) * par.h - row_end,
                      static_cast<int64_t>(par.d - (par.e - 1) * par.stride_d) * par.h - row_end});
        // OffsetTable::origin's clamp can put the base this far before the first tap row
        // (docs/algorithms/details/direct/direct-flex-offset-tables.md).
        const int64_t clamp_gap =
            static_cast<int64_t>(std::abs(par.pad_d)) * par.h + std::max(par.pad_h, 0);
        const int64_t span = (rows - 1) * advance +
                             static_cast<int64_t>(par.kd - 1) * par.dilation_d * par.h +
                             (par.kh - 1) * par.dilation_h + 1 + clamp_gap;
        return span * par.w * par.c;
    }
    const int64_t advance = std::max<int64_t>(par.stride_h, par.h - (par.p - 1) * par.stride_h);
    const int64_t span    = (rows - 1) * advance + (par.kh - 1) * par.dilation_h + 1;
    return span * par.w * par.c;
}

inline int tile_count(const Config& cfg, const ConvParams& par)
{
    const int x_tiles = static_cast<int>((out_rows(par) + cfg.x_tile() - 1) / cfg.x_tile());
    return x_tiles * ((par.k + cfg.k_tile() - 1) / cfg.k_tile());
}

} // namespace hipconv::cdna4::direct_flex
