#pragma once

// Dgrad over stride phases in one launch: segments, their workgroups, workspace, arguments.
// Parenthesized section names refer to docs/algorithms/direct/direct-flex-cdna4.md.

#include "dynamic_order.h"
#include "layer_pars.h"
#include "split_partials.h"
#include "tile_order.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace hipconv::cdna4::direct_flex
{

// One axis's dgrad phases (Strided dgrad).
struct AxisPhase
{
    int f, taps, extent, first, pad;
};

inline AxisPhase axis_phase(int size, int k, int pad, int stride, int dilation, int f)
{
    if(stride == 1)
        return {0, k, size, 0, (k - 1) * dilation - pad};
    const auto floor_div = [](int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); };
    const int taps       = f < k ? (k - f + stride - 1) / stride : 0;
    const int u0         = -floor_div(f - pad, stride);
    const int u1         = floor_div(size - 1 + pad - f, stride);
    return {f, taps, std::max(u1 - u0 + 1, 0), stride * u0 + f - pad, taps - 1 - u0};
}

// A dgrad's phases: the product of its strides, each capped at MAX_PHASES + 1 to avoid overflow.
inline int stride_phases(const ConvParams& layer)
{
    const auto cap = [](int stride) { return std::min(stride, MAX_PHASES + 1); };
    return cap(layer.stride_h) * cap(layer.stride_w) * (layer.dims == 3 ? cap(layer.stride_d) : 1);
}

// Whether a dgrad runs as phases: strided, or on positions outer.
inline bool phased(const ConvParams& layer)
{
    return layer.direction == Direction::Dgrad &&
           (stride_phases(layer) > 1 || sweep_of(layer) == 0);
}

// A phased dgrad's phases, segments, chunk ranges and workgroups (Partitioned grid).
struct DgradPlan
{
    // A conv3d phase pairs a depth phase with its row and column phases (Dgrad in depth).
    struct Phase
    {
        AxisPhase h, w;
        AxisPhase d = {0, 1, 1, 0, 0};
        int rows() const { return d.taps * h.taps; }
        int taps() const { return rows() * w.taps; }
    };
    struct Rect
    {
        int phase, r0, c0, nr, nc;
        int wg0 = 0, wgs = 0; // its workgroups, blocks [wg0, wg0 + wgs)
        bool dyn = false;     // claims its tiles at run time
    };
    Phase phases[MAX_PHASES];
    int n_phases = 0;
    Rect segs[MAX_SEGS];
    int n_segs = 0;
    int pieces = 1;
    bool split = false;
    int grid   = 0;

    int x_tiles(const Config& cfg, int n, int phase) const
    {
        const int64_t rows = static_cast<int64_t>(n) * phases[phase].d.extent *
                             phases[phase].h.extent * phases[phase].w.extent;
        return static_cast<int>((rows + cfg.x_tile() - 1) / cfg.x_tile());
    }
    // Segment s's items: tiles, once per chunk range in a split.
    int seg_tiles(const Config& cfg, const ConvParams& layer, int s) const
    {
        const int tiles_k = (layer.c + cfg.k_tile() - 1) / cfg.k_tile();
        return x_tiles(cfg, layer.n, segs[s].phase) * tiles_k * pieces;
    }
    int tiles(const Config& cfg, const ConvParams& layer) const
    {
        int t = 0;
        for(int s = 0; s < n_segs; ++s)
            t += seg_tiles(cfg, layer, s);
        return t;
    }
    int max_taps() const
    {
        int m = 0;
        for(int s = 0; s < n_segs; ++s)
            m = std::max(m, segs[s].nr * segs[s].nc);
        return m;
    }
    bool dynamic() const
    {
        for(int s = 0; s < n_segs; ++s)
            if(segs[s].dyn)
                return true;
        return false;
    }
};

// HIPCONV_DIRECT_FLEX_SPLIT_TAPS=<n> caps segments at n positions and splits; read once.
inline int split_taps_pin()
{
    static const int pin = env_int("HIPCONV_DIRECT_FLEX_SPLIT_TAPS");
    return pin;
}

// Cuts a phase into rectangles of at most `limit` positions; may exceed MAX_SEGS.
inline void cut_phase(DgradPlan& plan, int phase, int limit)
{
    const int nr = plan.phases[phase].rows();
    const int nc = plan.phases[phase].w.taps;
    auto add     = [&](int r0, int c0, int rows, int cols) {
        if(plan.n_segs < MAX_SEGS)
            plan.segs[plan.n_segs] = {phase, r0, c0, rows, cols};
        ++plan.n_segs;
    };
    if(nr * nc <= limit)
        add(0, 0, nr, nc);
    else if(nc <= limit)
        for(int r = 0; r < nr; r += limit / nc)
            add(r, 0, std::min(limit / nc, nr - r), nc);
    else
        for(int r = 0; r < nr; ++r)
            for(int c = 0; c < nc; c += limit)
                add(r, c, 1, std::min(limit, nc - c));
}

// A key for the cut cut_phase makes at `limit`: limits with equal keys cut the phase alike.
//
// 0: whole; positive: rows per rectangle of all columns; negative: minus columns per rectangle
// of one row.
inline int cut_key(const DgradPlan& plan, int phase, int limit)
{
    const int nr = plan.phases[phase].rows();
    const int nc = plan.phases[phase].w.taps;
    if(nr * nc <= limit)
        return 0;
    return nc <= limit ? limit / nc : -limit;
}

// The filter tiles a rectangle of nr x nc positions cuts into, as layer_pars cuts a filter.
inline int rect_ftiles(int nr, int nc)
{
    if(nr * nc == 0)
        return 0;
    const FtileShape t = ftile_shape(nr, nc);
    return (nr + t.rows - 1) / t.rows * ((nc + t.cols - 1) / t.cols);
}

inline int plan_ftiles(const DgradPlan& plan)
{
    int n = 0;
    for(int s = 0; s < std::min(plan.n_segs, MAX_SEGS); ++s)
        n += rect_ftiles(plan.segs[s].nr, plan.segs[s].nc);
    return n;
}

// Estimated cycles of a step, a tile's remainder and a tapless tile (Partitioned grid).
constexpr int STEP_CYCLES  = 2200;
constexpr int STORE_CYCLES = 16500;
constexpr int ZERO_CYCLES  = 3600;

// HIPCONV_DIRECT_FLEX_ZERO_CYCLES=<n> overrides ZERO_CYCLES. Read once.
inline int zero_cycles()
{
    static const int cycles = env_int("HIPCONV_DIRECT_FLEX_ZERO_CYCLES", ZERO_CYCLES);
    return cycles;
}

// Assigns each segment its workgroups (Partitioned grid).
inline void partition(const Config& cfg, const ConvParams& layer, DgradPlan& plan, int cu_count)
{
    constexpr int XCDS = persistent::NUM_XCD;
    const int n        = plan.n_segs;
    int tiles[MAX_SEGS];
    for(int s = 0; s < n; ++s)
        tiles[s] = plan.seg_tiles(cfg, layer, s);
    if(plan.split)
        for(int s = 0; s < n; ++s)
            plan.segs[s].wgs = tiles[s];
    else
    {
        const int chunks = (layer.k + C_CHUNK - 1) / C_CHUNK;
        double est[MAX_SEGS];
        double total     = 0;
        bool whole_units = true;
        for(int s = 0; s < n; ++s)
        {
            const int taps = plan.segs[s].nr * plan.segs[s].nc;
            est[s]         = taps == 0 ? zero_cycles() : taps * chunks * STEP_CYCLES + STORE_CYCLES;
            total += tiles[s] * est[s];
            whole_units &= tiles[s] >= XCDS;
        }
        int left       = std::max(n, cu_count);
        const bool by8 = whole_units && left / XCDS >= n;
        int size[MAX_SEGS], got[MAX_SEGS];
        for(int s = 0; s < n; ++s)
            size[s] = by8 ? XCDS : 1;
        // Finds the least finishing time whose units fit.
        auto claims = [&](int s, int wgs) { return tiles[s] >= MIN_ROUNDS * wgs; };
        auto need   = [&](int s, double t) {
            const int rounds = std::max(static_cast<int>(t / est[s]), 1);
            const int wgs    = std::min((tiles[s] + rounds - 1) / rounds, tiles[s]);
            int units        = std::max((wgs + size[s] - 1) / size[s], 1);
            const int frac =
                static_cast<int>(std::ceil(tiles[s] * est[s] / std::max(t, 1.0) / size[s] - 1e-9));
            if(frac >= 1 && frac < units && claims(s, frac * size[s]))
                units = frac;
            return units;
        };
        auto fits = [&](double t) {
            int used = 0;
            for(int s = 0; s < n; ++s)
                used += need(s, t) * size[s];
            return used <= left;
        };
        // Estimates the least t at which segment s needs at most u units.
        //
        // By rounds, or by claims where they apply; 0 when u units give each tile a workgroup.
        auto threshold = [&](int s, int u) {
            const int per = u * size[s];
            if(per >= tiles[s])
                return 0.0;
            double t     = static_cast<double>((tiles[s] + per - 1) / per) * est[s];
            const int uc = std::min(u, tiles[s] / (MIN_ROUNDS * size[s]));
            if(uc >= 1)
                t = std::min(t, tiles[s] * est[s] / size[s] / (uc + 1e-9));
            return t;
        };
        // The least t at which the units fit: fits is monotone, so bisecting [0, total] ends there.
        //
        // Gives units to the segment with the latest threshold while they fit, then steps t by
        // nextafter to the least double that fits; bisects if that takes 32 steps.
        double hi = total;
        if(n > 0)
        {
            int u[MAX_SEGS];
            double th[MAX_SEGS];
            int used = 0;
            for(int s = 0; s < n; ++s)
            {
                u[s]  = 1;
                th[s] = threshold(s, 1);
                used += size[s];
            }
            for(;;)
            {
                const int m = static_cast<int>(std::max_element(th, th + n) - th);
                if(u[m] * size[m] >= tiles[m] || used + size[m] > left)
                    break;
                ++u[m];
                used += size[m];
                th[m] = threshold(m, u[m]);
            }
            double t  = *std::max_element(th, th + n);
            int steps = 0;
            // Every t fits: the bisection halves total 64 times.
            if(t == 0)
                t = std::ldexp(total, -64);
            else
            {
                while(steps < 32 && !fits(t))
                {
                    t = std::nextafter(t, total);
                    ++steps;
                }
                while(steps < 32 && fits(std::nextafter(t, 0.0)))
                {
                    t = std::nextafter(t, 0.0);
                    ++steps;
                }
            }
            if(steps < 32)
                hi = t;
            else
            {
                double lo = 0;
                for(int i = 0; i < 64; ++i)
                {
                    const double mid = (lo + hi) / 2;
                    if(fits(mid))
                        hi = mid;
                    else
                        lo = mid;
                }
            }
        }
        for(int s = 0; s < n; ++s)
        {
            got[s] = need(s, hi);
            left -= got[s] * size[s];
        }
        // The CUs left over to the segment that would finish last.
        auto finish = [&](int s) {
            const int wgs = got[s] * size[s];
            return claims(s, wgs) ? static_cast<double>(tiles[s]) / wgs * est[s]
                                  : (tiles[s] + wgs - 1) / wgs * est[s];
        };
        for(;;)
        {
            int last = -1;
            for(int s = 0; s < n; ++s)
                if(size[s] <= left && (got[s] + 1) * size[s] <= tiles[s] &&
                   (last < 0 || finish(s) > finish(last)))
                    last = s;
            if(last < 0)
                break;
            ++got[last];
            left -= size[last];
        }
        for(int s = 0; s < n; ++s)
            plan.segs[s].wgs = std::min(got[s] * size[s], tiles[s]);
        // Segments of whole units first, then the rest.
        DgradPlan::Rect order[MAX_SEGS];
        int k = 0;
        for(int pass = 0; pass < 2; ++pass)
            for(int s = 0; s < n; ++s)
                if((plan.segs[s].wgs % XCDS != 0) == (pass == 1))
                    order[k++] = plan.segs[s];
        std::copy(order, order + n, plan.segs);
    }

    int at = 0;
    for(int s = 0; s < n; ++s)
    {
        auto& sg = plan.segs[s];
        sg.wg0   = at;
        at += sg.wgs;
        const int t = plan.seg_tiles(cfg, layer, s);
        sg.dyn      = !plan.split && sg.wgs < t && runs_claim(t, sg.wgs, std::min(XCDS, sg.wgs));
    }
    plan.grid = at;
}

// A phased dgrad's phases, most taps first, each one segment.
inline DgradPlan dgrad_phases(const ConvParams& layer)
{
    DgradPlan plan;
    const bool depth = layer.dims == 3;
    for(int fd = 0; fd < (depth ? layer.stride_d : 1); ++fd)
        for(int fh = 0; fh < layer.stride_h; ++fh)
            for(int fw = 0; fw < layer.stride_w; ++fw)
            {
                const AxisPhase h = axis_phase(
                    layer.h, layer.kh, layer.pad_h, layer.stride_h, layer.dilation_h, fh);
                const AxisPhase w = axis_phase(
                    layer.w, layer.kw, layer.pad_w, layer.stride_w, layer.dilation_w, fw);
                const AxisPhase d =
                    depth
                        ? axis_phase(
                              layer.d, layer.kd, layer.pad_d, layer.stride_d, layer.dilation_d, fd)
                        : AxisPhase{0, 1, 1, 0, 0};
                if(d.extent > 0 && h.extent > 0 && w.extent > 0 && plan.n_phases < MAX_PHASES)
                    plan.phases[plan.n_phases++] = {h, w, d};
            }
    // Longest first: a split's items, numbered segment by segment, then end with the shortest.
    std::stable_sort(plan.phases, plan.phases + plan.n_phases, [](const auto& a, const auto& b) {
        return a.taps() > b.taps();
    });
    const int full = plan.n_phases > 0 ? plan.phases[0].taps() : 0;
    for(int f = 0; f < plan.n_phases; ++f)
        if(plan.phases[f].taps() > 0)
            cut_phase(plan, f, full);
    return plan;
}

// A phased dgrad's plan. Its split and dynamic tile order use the workspace plan_workspace sizes.
inline DgradPlan dgrad_plan(const Config& cfg, const ConvParams& layer, int cu_count)
{
    DgradPlan plan = dgrad_phases(layer);
    // The segments of the phases with taps; and unless split, the phases without.
    auto cut = [&](int limit, int pieces) {
        plan.n_segs = 0;
        for(int f = 0; f < plan.n_phases; ++f)
            if(plan.phases[f].taps() > 0)
                cut_phase(plan, f, limit);
        plan.pieces = pieces;
    };
    auto finish = [&] {
        if(!plan.split)
            for(int f = 0; f < plan.n_phases; ++f)
                if(plan.phases[f].taps() == 0 && plan.n_segs < MAX_SEGS)
                    plan.segs[plan.n_segs++] = {f, 0, 0, 0, 0};
        partition(cfg, layer, plan, cu_count);
        return plan;
    };
    const int full = plan.n_phases > 0 ? plan.phases[0].taps() : 0;
    if(full == 0)
        return finish();

    // As split_pieces, with the split's fixed cost.
    const int chunks  = (layer.k + C_CHUNK - 1) / C_CHUNK;
    const int tpin    = split_taps_pin();
    const int pin     = pinned_pieces(chunks);
    const bool pinned = pin > 0 || tpin > 0;
    int best_limit = full, best_pieces = 1;
    int best_steps = full * chunks;
    bool found     = false;
    if(pinned)
    {
        const int limit  = tpin > 0 ? std::min(tpin, full) : full;
        const int pieces = pin > 0 ? pin : 1;
        if(limit != full || pieces != 1)
        {
            cut(limit, pieces);
            if(plan.n_segs <= MAX_SEGS && plan_ftiles(plan) <= MAX_FTILES)
            {
                best_limit  = limit;
                best_pieces = pieces;
                found       = true;
            }
        }
    }
    int keys[MAX_PHASES];
    for(int limit = full; !pinned && limit >= 1; --limit)
    {
        // A limit cutting every phase as the previous one did repeats its steps; `<` rejects them.
        bool same = limit < full;
        for(int f = 0; f < plan.n_phases; ++f)
        {
            const int key = cut_key(plan, f, limit);
            same          = same && key == keys[f];
            keys[f]       = key;
        }
        if(same)
            continue;
        cut(limit, 1);
        if(plan.n_segs > MAX_SEGS || plan_ftiles(plan) > MAX_FTILES)
            continue;
        // Tiles of one piece; a split into pieces multiplies them.
        const int tiles = plan.tiles(cfg, layer);
        const int taps  = plan.max_taps();
        for(int pieces = 1; pieces <= chunks; ++pieces)
        {
            if(chunks % pieces != 0 || (limit == full && pieces == 1) || tiles * pieces > cu_count)
                continue;
            const int steps = split_steps(cfg, taps, chunks, pieces);
            if(steps < best_steps)
            {
                best_limit  = limit;
                best_pieces = pieces;
                best_steps  = steps;
                found       = true;
            }
        }
    }
    cut(best_limit, best_pieces);
    plan.split = found;
    return finish();
}

// A split's partials: per segment and chunk range, an fp32 copy of its phase's padded output.
inline size_t plan_partials(const Config& cfg, const ConvParams& layer, const DgradPlan& plan)
{
    if(!plan.split)
        return 0;
    const size_t k_pad = (layer.c + cfg.k_tile() - 1) / cfg.k_tile() * cfg.k_tile();
    size_t floats      = 0;
    for(int s = 0; s < plan.n_segs; ++s)
        floats += static_cast<size_t>(plan.pieces) *
                  plan.x_tiles(cfg, layer.n, plan.segs[s].phase) * cfg.x_tile() * k_pad;
    return floats * sizeof(float);
}

// The dynamic order's counters, a line per segment and run, and then each workgroup's mailbox.
inline size_t plan_counter_bytes(const DgradPlan& plan)
{
    return static_cast<size_t>(plan.n_segs) * CTR_BYTES;
}

inline size_t plan_workspace(const Config& cfg, const ConvParams& layer, int cu_count)
{
    const DgradPlan plan = dgrad_plan(cfg, layer, cu_count);
    if(plan.split)
        return plan_partials(cfg, layer, plan);
    return plan.dynamic() ? plan_counter_bytes(plan) +
                                static_cast<size_t>(plan.grid) * MAILBOX_SLOTS * sizeof(int)
                          : 0;
}

// Whether a phased dgrad fits the kernel arguments and the 32-bit offsets.
inline bool phases_fit(const Config& cfg, const ConvParams& layer)
{
    const DgradPlan plan = dgrad_phases(layer);
    if(plan.n_phases == 0 || plan.phases[0].taps() == 0 || plan_ftiles(plan) > MAX_FTILES)
        return false;
    const ConvParams par = as_fprop(layer);
    for(int f = 0; f < plan.n_phases; ++f)
    {
        const auto& g  = plan.phases[f];
        ConvParams sub = par;
        sub.kh         = std::max(g.h.taps, 1);
        sub.kw         = std::max(g.w.taps, 1);
        sub.pad_h      = g.h.pad;
        sub.pad_w      = g.w.pad;
        sub.p          = g.h.extent;
        sub.q          = g.w.extent;
        sub.stride_h   = 1;
        sub.stride_w   = 1;
        if(layer.stride_h * layer.stride_w > 1)
            sub.dilation_h = sub.dilation_w = 1;
        // dX rows lie stride_h rows apart, or in a conv3d up to two depth strides of planes.
        int64_t row_advance = static_cast<int64_t>(layer.stride_h) * layer.w;
        if(layer.dims == 3)
        {
            sub.kd         = std::max(g.d.taps, 1);
            sub.pad_d      = g.d.pad;
            sub.e          = g.d.extent;
            sub.stride_d   = 1;
            sub.dilation_d = stride_phases(layer) > 1 ? 1 : sub.dilation_d;
            row_advance =
                std::max(row_advance, 2 * static_cast<int64_t>(layer.stride_d) * layer.h * layer.w);
        }
        const int64_t tile_rows = (cfg.x_tile() - 1) / g.w.extent + 2;
        if(static_cast<int64_t>(par.n) * g.d.extent * g.h.extent * g.w.extent > INT32_MAX - 256 ||
           span_input(sub) > INT32_MAX / 2 || tile_rows * row_advance * layer.c > INT32_MAX / 2)
            return false;
    }
    return true;
}

// Kernel arguments of a phased dgrad.
inline LayerPars phased_pars(const Config& cfg,
                             const ConvParams& layer,
                             const DgradPlan& plan,
                             uint32_t* ws,
                             float* partials)
{
    constexpr int XCDS   = persistent::NUM_XCD;
    const ConvParams par = as_fprop(layer);
    const bool strided   = stride_phases(layer) > 1;
    const bool depth     = layer.dims == 3;
    LayerPars pars{};
    pars.h         = par.h;
    pars.w         = par.w;
    pars.c         = par.c;
    pars.k         = par.k;
    pars.kh        = filter_rows(par).kh;
    pars.kw        = par.kw;
    pars.p         = par.p;
    pars.q         = par.q;
    pars.sh        = 1;
    pars.sw        = 1;
    pars.dh        = strided ? 1 : par.dilation_h;
    pars.dw        = strided ? 1 : par.dilation_w;
    pars.ph        = par.pad_h;
    pars.pw        = par.pad_w;
    pars.chunks    = (par.c + C_CHUNK - 1) / C_CHUNK / plan.pieces;
    pars.tiles_k   = (par.k + cfg.k_tile() - 1) / cfg.k_tile();
    pars.k_group   = std::min(pars.tiles_k, 4);
    pars.k_full    = pars.tiles_k / pars.k_group * pars.k_group;
    pars.kg_div    = MagicDiv(static_cast<uint32_t>(pars.k_group));
    pars.kr_div    = MagicDiv(static_cast<uint32_t>(std::max(pars.tiles_k - pars.k_full, 1)));
    pars.in_elems  = static_cast<int64_t>(par.n) * (depth ? par.d : 1) * par.h * par.w * par.c;
    pars.wei_elems = par.k * pars.kh * par.kw * par.c;
    pars.out_elems = static_cast<int64_t>(par.n) * out_depth(par) * par.p * par.q * par.k;
    pars.rebase_in = ConvSize(par).input_bytes() > INT32_MAX;
    pars.blocks    = plan.grid;
    pars.ws        = ws;
    pars.pieces    = plan.pieces;
    pars.c_last    = par.c - ((par.c + C_CHUNK - 1) / C_CHUNK - 1) * C_CHUNK;
    pars.partials  = partials;
    pars.strided   = strided;
    pars.out_hw    = (depth ? layer.d : 1) * layer.h * layer.w;
    pars.out_row   = layer.stride_h * layer.w;
    pars.out_sw    = layer.stride_w;
    if(depth)
    {
        pars.dep.d         = par.d;
        pars.dep.dd        = strided ? 1 : par.dilation_d;
        pars.dep.out_plane = layer.stride_d * layer.h * layer.w;
    }

    const int k_pad = pars.tiles_k * cfg.k_tile();
    for(int f = 0; f < plan.n_phases; ++f)
    {
        const auto& g     = plan.phases[f];
        const int x_tiles = plan.x_tiles(cfg, layer.n, f);
        pars.phases[f]    = {g.h.extent,
                             g.w.extent,
                             g.h.pad,
                             g.w.pad,
                             par.n * g.d.extent * g.h.extent * g.w.extent,
                             (g.d.first * layer.h + g.h.first) * layer.w + g.w.first,
                             x_tiles,
                             pars.k_full * x_tiles,
                             MagicDiv(static_cast<uint32_t>(g.w.extent)),
                             MagicDiv(static_cast<uint32_t>(g.h.extent)),
                             MagicDiv(static_cast<uint32_t>(pars.k_group * x_tiles))};
        if(depth)
        {
            pars.dep.pe[f]     = g.d.extent;
            pars.dep.ppd[f]    = g.d.pad;
            pars.dep.pkh[f]    = g.h.taps;
            pars.dep.pe_div[f] = MagicDiv(static_cast<uint32_t>(g.d.extent));
        }
    }
    pars.rows = pars.phases[0].rows;

    // Each segment's positions in filter tiles, then its first again; weights at its tap (r, s).
    int q = 0, ft = 0, copy = 0;
    pars.n_segs = plan.n_segs;
    for(int s = 0; s < plan.n_segs; ++s)
    {
        const auto& rc = plan.segs[s];
        const auto& g  = plan.phases[rc.phase];
        const int ft0  = ft;
        const int seq0 = rc.nr * rc.nc > 0 ? q : 0;
        if(rc.nr * rc.nc > 0)
        {
            ft = cut_ftiles(pars.ftiles, ft, rc.r0, rc.c0, rc.nr, rc.nc);
            for(int f = ft0; f < ft; ++f)
            {
                const uint32_t d = pars.ftiles[f];
                const int fr0 = d & 0xff, fc0 = (d >> 8) & 0xff, nr = (d >> 16) & 0xff,
                          nc = d >> 24;
                if(depth)
                    pars.dep.ftile_tr[f] =
                        static_cast<uint16_t>(fr0 / g.h.taps | (fr0 % g.h.taps) << 8);
                for(int local = 0; local < nr * nc; ++local)
                {
                    // Rotated position (j, l) is tap taps - 1 - j along each axis.
                    const int j   = fr0 + local / nc;
                    const int l   = fc0 + local % nc;
                    const int jd  = j / g.h.taps;
                    const int jh  = j % g.h.taps;
                    const int rd  = g.d.f + (depth ? layer.stride_d : 1) * (g.d.taps - 1 - jd);
                    const int r   = g.h.f + layer.stride_h * (g.h.taps - 1 - jh);
                    const int cc  = g.w.f + layer.stride_w * (g.w.taps - 1 - l);
                    pars.seq[q++] = {((f - ft0) % 2) * TAB_BYTES + local * TAB_ROW_BYTES |
                                         (local == 0 ? 1 << 16 : 0),
                                     ((rd * layer.kh + r) * layer.kw + cc) * layer.c * 2};
                }
            }
            pars.seq[q++] = pars.seq[seq0];
        }
        const int tiles = plan.seg_tiles(cfg, layer, s);
        const int runs  = std::min(XCDS, rc.wgs);
        auto& sg        = pars.segs[s];
        sg.phase        = rc.phase;
        sg.ftile0       = ft0;
        sg.n_ftiles     = ft - ft0;
        sg.seq0         = seq0;
        sg.taps         = rc.nr * rc.nc;
        sg.tiles        = tiles;
        sg.copy0        = copy;
        sg.wg0          = rc.wg0;
        sg.wgs          = rc.wgs;
        sg.runs         = runs;
        sg.run_q        = tiles / runs;
        sg.run_r        = tiles % runs;
        sg.wg_q         = rc.wgs / runs;
        sg.wg_r         = rc.wgs % runs;
        sg.dyn          = rc.dyn;
        copy += plan.pieces * plan.x_tiles(cfg, layer.n, rc.phase) * cfg.x_tile() * k_pad;
    }
    return pars;
}

inline ReducePars reduce_pars(const Config& cfg, const DgradPlan& plan, const LayerPars& pars)
{
    ReducePars rp{};
    rp.k        = pars.k;
    rp.k_pad    = pars.tiles_k * cfg.k_tile();
    rp.out_hw   = pars.out_hw;
    rp.out_row  = pars.out_row;
    rp.out_sw   = pars.out_sw;
    rp.n_phases = plan.n_phases;
    for(int f = 0; f < plan.n_phases; ++f)
    {
        const auto& g = pars.phases[f];
        rp.phases[f]  = {
            g.rows, g.q, g.p, g.out0, 0, 0, g.x_tiles * cfg.x_tile() * rp.k_pad, g.q_div, g.p_div};
        rp.phases[f].e     = std::max(pars.dep.pe[f], 1);
        rp.phases[f].e_div = MagicDiv(static_cast<uint32_t>(rp.phases[f].e));
    }
    rp.out_plane = pars.dep.out_plane;
    for(int s = 0; s < pars.n_segs; ++s)
    {
        const auto& sg = pars.segs[s];
        ReducePhase& r = rp.phases[sg.phase];
        if(r.copies == 0)
            r.copy0 = sg.copy0;
        r.copies += pars.pieces;
    }
    return rp;
}

} // namespace hipconv::cdna4::direct_flex
