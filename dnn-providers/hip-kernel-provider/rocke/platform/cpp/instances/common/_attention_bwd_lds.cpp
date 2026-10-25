// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_bwd_lds.py: the LDS layouts and
 * the phase planner of the attention backward (host side only, no IR). See
 * rocke/helper_rocke.instances.common._attention_bwd_lds.h for the mapping.
 *
 * Every Python raise is one ckc::Error throw with the same message; the C
 * entry points catch it. Python's divmod / // run on non-negative operands
 * here (validated shapes), where C's / and % agree with them.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_lds.h"

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "rocke/arch_target.h"
#include "rocke/error.hpp"

namespace rocke_attn_bwd
{

static const int LINE = ROCKE_ATTN_BWD_LDS_LINE_BYTES;
static const int ALIGN = ROCKE_ATTN_BWD_BUFFER_ALIGN_BYTES;

/* _SWIZZLES (static spellings) */
static const char* const SWIZZLES[] = {"xor", "none", "pad8", "pad16", "pad32"};
static const char* const SWIZZLES_REPR = "('xor', 'none', 'pad8', 'pad16', 'pad32')";

static const char* const PHASE_NAMES[4] = {"stage0_k", "stage0_v", "loop", "epilogue"};

static const char* static_swizzle(const std::string& s)
{
    for(const char* x : SWIZZLES)
        if(s == x)
            return x;
    return NULL;
}

static int align_up(long long n, int a = ALIGN)
{
    return (int)((n + a - 1) / a * a);
}

static int pad_bytes(const std::string& swizzle)
{
    if(swizzle.rfind("pad", 0) == 0)
        return std::stoi(swizzle.substr(3));
    return 0;
}

static void check_swizzle(const std::string& swizzle)
{
    if(static_swizzle(swizzle) == NULL)
        throw ckc::ValueError("unknown LDS swizzle " + py_repr(swizzle) + "; expected one of "
                              + SWIZZLES_REPR);
}

static int bit_length(unsigned v)
{
    int n = 0;
    while(v)
    {
        ++n;
        v >>= 1;
    }
    return n;
}

/* ====================================================================== *
 * XtLayout
 * ====================================================================== */

static rocke_attn_bwd_xt_layout_t
    xt_layout(int rows, int cols, int elem_bytes, const std::string& swizzle, int chunk_bytes)
{
    check_swizzle(swizzle);
    if(rows <= 0 || cols <= 0)
        throw ckc::ValueError("XtLayout rows and cols must be positive");
    if(chunk_bytes != 4 && chunk_bytes != 8 && chunk_bytes != 16)
        throw ckc::ValueError("XtLayout chunk_bytes must be 4, 8 or 16");
    int p = cols * elem_bytes;
    if(p % chunk_bytes)
        throw ckc::ValueError("row of " + std::to_string(p) + " bytes is not a multiple of the "
                              + std::to_string(chunk_bytes) + "-byte chunk");
    if(swizzle == "xor" && !(p % LINE == 0 || LINE % p == 0))
        throw ckc::ValueError("xor swizzle needs the row (" + std::to_string(p)
                              + " bytes) to divide or be a multiple of the " + std::to_string(LINE)
                              + "-byte line");
    int pad = pad_bytes(swizzle);
    if(pad % elem_bytes)
        throw ckc::ValueError("row pad must be a whole number of elements");
    rocke_attn_bwd_xt_layout_t l;
    l.rows = rows;
    l.cols = cols;
    l.elem_bytes = elem_bytes;
    l.swizzle = static_swizzle(swizzle);
    l.chunk_bytes = chunk_bytes;
    return l;
}

static int xt_row_pitch(const rocke_attn_bwd_xt_layout_t& l)
{
    return l.cols * l.elem_bytes + pad_bytes(l.swizzle);
}

static int xt_size(const rocke_attn_bwd_xt_layout_t& l)
{
    return l.rows * xt_row_pitch(l);
}

static int xt_hash(const rocke_attn_bwd_xt_layout_t& l, int row)
{
    int nch = LINE / l.chunk_bytes;
    int p = l.cols * l.elem_bytes;
    int rpl = std::max(1, LINE / p);
    int g = row / rpl;
    int shift = bit_length((unsigned)nch) - 1;
    return (g ^ (g >> shift)) % nch;
}

static int xt_offset(const rocke_attn_bwd_xt_layout_t& l, int row, int col)
{
    if(!(0 <= row && row < l.rows && 0 <= col && col < l.cols))
        throw ckc::ValueError("(" + std::to_string(row) + ", " + std::to_string(col)
                              + ") outside the " + std::to_string(l.rows) + "x"
                              + std::to_string(l.cols) + " image");
    if(strcmp(l.swizzle, "xor") != 0)
        return row * xt_row_pitch(l) + col * l.elem_bytes;
    int lin = row * l.cols * l.elem_bytes + col * l.elem_bytes;
    int line = lin / LINE, within = lin % LINE;
    int chunk = within / l.chunk_bytes, inner = within % l.chunk_bytes;
    return line * LINE + (chunk ^ xt_hash(l, row)) * l.chunk_bytes + inner;
}

static void xt_check_vector(const rocke_attn_bwd_xt_layout_t& l, int col, int n)
{
    int nbytes = n * l.elem_bytes;
    if(strcmp(l.swizzle, "xor") == 0)
    {
        if(nbytes > l.chunk_bytes || (col * l.elem_bytes) % nbytes)
            throw ckc::ValueError(std::to_string(nbytes) + "-byte vector at col "
                                  + std::to_string(col) + " crosses a "
                                  + std::to_string(l.chunk_bytes) + "-byte xor chunk");
    }
    else if((col * l.elem_bytes) % nbytes)
        throw ckc::ValueError(std::to_string(nbytes) + "-byte vector at col " + std::to_string(col)
                              + " is misaligned");
}

/* ====================================================================== *
 * RowXorLayout
 * ====================================================================== */

static rocke_attn_bwd_xt_layout_t row_xor_image(const rocke_attn_bwd_row_xor_layout_t& l)
{
    return xt_layout(l.rows, l.cols, l.elem_bytes, "xor", l.chunk_bytes);
}

static rocke_attn_bwd_row_xor_layout_t row_xor_layout(int rows, int cols, int eb, int chunk)
{
    rocke_attn_bwd_row_xor_layout_t l{rows, cols, eb, chunk};
    (void)row_xor_image(l); /* validates the shape */
    return l;
}

static int row_xor_size(const rocke_attn_bwd_row_xor_layout_t& l)
{
    return l.rows * (l.cols * l.elem_bytes);
}

/* ====================================================================== *
 * XtWriter
 * ====================================================================== */

static int writer_n_dg(const rocke_attn_bwd_xt_writer_t& w)
{
    return w.depth / w.vec_d;
}
static int writer_n_sg(const rocke_attn_bwd_xt_writer_t& w)
{
    return w.seq / w.k_per_thread;
}
static int writer_d_lanes_eff(const rocke_attn_bwd_xt_writer_t& w)
{
    return std::min(w.d_lanes, w.depth / w.vec_d);
}
static int writer_units(const rocke_attn_bwd_xt_writer_t& w)
{
    return writer_n_dg(w) * writer_n_sg(w);
}
static int writer_iterations(const rocke_attn_bwd_xt_writer_t& w)
{
    return (writer_units(w) + w.threads - 1) / w.threads;
}
static int writer_write_bytes(const rocke_attn_bwd_xt_writer_t& w)
{
    return w.k_per_thread * w.elem_bytes;
}

static bool in_1248(int v)
{
    return v == 1 || v == 2 || v == 4 || v == 8;
}

static rocke_attn_bwd_xt_writer_t xt_writer(
    int seq, int depth, int threads, int vec_d, int k_per_thread, int d_lanes, int elem_bytes)
{
    rocke_attn_bwd_xt_writer_t w{seq, depth, threads, vec_d, k_per_thread, d_lanes, elem_bytes};
    /* Python raises ZeroDivisionError on a zero vec_d / k_per_thread or a zero
     * effective lane count; the C port refuses those inputs as not ported. */
    if(vec_d == 0 || k_per_thread == 0)
        throw ckc::NotImplError("XtWriter: a zero vec_d / k_per_thread is not ported");
    if(depth % vec_d || seq % k_per_thread)
        throw ckc::ValueError("XtWriter: depth % vec_d and seq % k_per_thread must be 0");
    if(!in_1248(vec_d) || !in_1248(k_per_thread))
        throw ckc::ValueError("XtWriter: vec_d and k_per_thread must be in {1, 2, 4, 8}");
    if(writer_d_lanes_eff(w) == 0)
        throw ckc::NotImplError("XtWriter: a zero d_lanes is not ported");
    if(writer_n_dg(w) % writer_d_lanes_eff(w))
        throw ckc::ValueError("XtWriter: head-dim groups must be a multiple of d_lanes");
    if(threads <= 0)
        throw ckc::ValueError("XtWriter: threads must be positive");
    return w;
}

static bool writer_unit_coords(
    const rocke_attn_bwd_xt_writer_t& w, int tid, int iteration, int* s0, int* d0)
{
    int u = iteration * w.threads + tid;
    if(u >= writer_units(w))
        return false;
    int dl = writer_d_lanes_eff(w);
    int dg_lo = u % dl;
    int rest = u / dl;
    int sg = rest % writer_n_sg(w);
    int dg_hi = rest / writer_n_sg(w);
    int dg = dg_hi * dl + dg_lo;
    *s0 = sg * w.k_per_thread;
    *d0 = dg * w.vec_d;
    return true;
}

static rocke_attn_bwd_xt_writer_t
    default_xt_writer(int seq, int depth, int threads, int elem_bytes, int max_vec_bytes)
{
    int max_vec = std::max(1, max_vec_bytes / elem_bytes);
    int per_thread = std::max(1, (seq * depth) / threads);
    int vec_d = std::min(max_vec, depth);
    while(vec_d > 1 && (per_thread / vec_d < 2 || depth % vec_d))
        vec_d /= 2;
    int kpt = 2;
    if(per_thread / vec_d >= 4 && seq % 4 == 0)
        kpt = 4;
    kpt = std::min(kpt, seq);
    int d_lanes = std::min(16, depth / vec_d);
    return xt_writer(seq, depth, threads, vec_d, kpt, d_lanes, elem_bytes);
}

/* ====================================================================== *
 * access lists
 * ====================================================================== */

using Access = rocke_attn_bwd_lds_access_t;

static std::vector<Access>
    xt_read_accesses(const rocke_attn_bwd_xt_layout_t& l, int n0, int k0, int atom_k, int wave_size)
{
    if(wave_size != 64)
        throw ckc::ValueError("xt_read_accesses models the wave64 MFMA B layout");
    int per_lane = atom_k / 4;
    int width = per_lane * l.elem_bytes;
    std::vector<Access> out;
    for(int lane = 0; lane < wave_size; ++lane)
    {
        int row = n0 + lane % 16;
        int col = k0 + (lane / 16) * per_lane;
        xt_check_vector(l, col, per_lane);
        out.push_back(Access{lane, xt_offset(l, row, col), width});
    }
    return out;
}

static std::vector<Access> xt_write_accesses(const rocke_attn_bwd_xt_layout_t& l,
                                             const rocke_attn_bwd_xt_writer_t& w,
                                             int wave,
                                             int iteration,
                                             int j,
                                             int wave_size)
{
    if(l.rows != w.depth || l.cols != w.seq)
        throw ckc::ValueError("XtWriter tile does not match the XtLayout image");
    std::vector<Access> out;
    for(int lane = 0; lane < wave_size; ++lane)
    {
        int tid = wave * wave_size + lane;
        if(tid >= w.threads)
            continue;
        int s0, d0;
        if(!writer_unit_coords(w, tid, iteration, &s0, &d0))
            continue;
        xt_check_vector(l, s0, w.k_per_thread);
        out.push_back(Access{lane, xt_offset(l, d0 + j, s0), writer_write_bytes(w)});
    }
    return out;
}

/* ====================================================================== *
 * conflict predictor (rocke.analysis.lds, the wave64 profiles)
 * ====================================================================== */

/* _READ_OPCODE / _WRITE_OPCODE: width -> opcode; KeyError when absent. */
static const char* read_opcode(int width)
{
    switch(width)
    {
    case 4:
        return "ds_read_b32";
    case 8:
        return "ds_read_b64";
    case 16:
        return "ds_read_b128";
    }
    throw ckc::KeyError(std::to_string(width));
}

static bool has_write_opcode(int width)
{
    return width == 4 || width == 8 || width == 16;
}

static const char* write_opcode(int width)
{
    switch(width)
    {
    case 4:
        return "ds_write_b32";
    case 8:
        return "ds_write_b64";
    case 16:
        return "ds_write_b128";
    }
    throw ckc::KeyError(std::to_string(width));
}

/* _predictor_targets(): the profiles with wave64 (gfx90a, gfx942, gfx950). */
static bool predictor_target(const std::string& arch)
{
    return arch == "gfx90a" || arch == "gfx942" || arch == "gfx950";
}

/* _predict_max(arch, opcode, accesses): the largest conflict multiplicity of
 * one instruction (1 = conflict-free). Ports predict_lds_conflicts for the
 * ds_read / ds_write b32 / b64 / b128 opcodes on the wave64 profiles: the
 * request checks a candidate layout can fail (capacity, alignment), then the
 * (phase key, address class) groups of two or more accesses (a read group of
 * one address is a broadcast and still counts; a write one does not). */
static int
    predict_max(const std::string& arch, const std::string& opcode, const std::vector<Access>& acc)
{
    bool read = opcode.rfind("ds_read", 0) == 0;
    int width = opcode.find("b128") != std::string::npos  ? 16
                : opcode.find("b64") != std::string::npos ? 8
                                                          : 4;
    int capacity = arch == "gfx950" ? 163840 : 65536;
    for(size_t i = 0; i < acc.size(); ++i)
    {
        const Access& a = acc[i];
        if(a.width != width)
            throw ckc::ValueError("access " + std::to_string(i) + " width "
                                  + std::to_string(a.width) + " does not match " + opcode
                                  + " width " + std::to_string(width));
        if(a.byte + a.width > capacity)
            throw ckc::ValueError("access " + std::to_string(i) + " end address "
                                  + std::to_string(a.byte + a.width) + " exceeds " + arch
                                  + " LDS capacity " + std::to_string(capacity));
        if(a.byte % width)
            throw ckc::ValueError("access " + std::to_string(i) + " address "
                                  + std::to_string(a.byte) + " must be " + std::to_string(width)
                                  + "-byte aligned");
    }
    /* key: (phase key tuple, address class) */
    std::map<std::pair<std::vector<int>, int>, std::vector<int>> buckets;
    for(const Access& a : acc)
    {
        int lane = a.lane;
        int bit5 = (lane >> 5) & 1, bit4 = (lane >> 4) & 1, bit3 = (lane >> 3) & 1,
            bit2 = (lane >> 2) & 1;
        std::vector<int> phase;
        int cls;
        if(arch == "gfx950" && read && width == 8)
            phase = {bit5};
        else if(arch == "gfx950" && read && width == 16)
            phase = {bit5, bit4 ^ bit3 ^ bit2};
        else if(width == 4)
            phase = {bit5};
        else if(width == 8)
            phase = {bit5, bit4};
        else if(read)
            phase = {bit5, bit4 ^ bit2, bit3};
        else
            phase = {bit5, bit4, bit3};
        if(arch == "gfx950" && read && (width == 8 || width == 16))
            cls = a.byte % 256;
        else
            cls = a.byte % 128;
        buckets[{phase, cls}].push_back(a.byte);
    }
    int groups = 0, worst = 0;
    for(const auto& kv : buckets)
    {
        const std::vector<int>& m = kv.second;
        if(m.size() < 2)
            continue;
        bool one_addr = std::all_of(m.begin(), m.end(), [&](int x) { return x == m[0]; });
        if(one_addr && !read)
            continue;
        ++groups;
        worst = std::max(worst, (int)m.size());
    }
    return groups == 0 ? 1 : worst;
}

struct Conflicts
{
    bool has;
    int read, write;
};

static Conflicts xt_conflicts(const rocke_attn_bwd_xt_layout_t& l,
                              const rocke_attn_bwd_xt_writer_t* writer,
                              const std::string& arch,
                              int atom_k)
{
    if(!predictor_target(arch))
        return Conflicts{false, 0, 0};
    int read_w = (atom_k / 4) * l.elem_bytes;
    int worst_r = 1;
    for(int n0 = 0; n0 < l.rows; n0 += 16)
        for(int k0 = 0; k0 < l.cols; k0 += atom_k)
        {
            std::vector<Access> acc = xt_read_accesses(l, n0, k0, atom_k, 64);
            worst_r = std::max(worst_r, predict_max(arch, read_opcode(read_w), acc));
        }
    int worst_w = 1;
    if(writer != NULL)
    {
        const rocke_attn_bwd_xt_writer_t& w = *writer;
        if(!has_write_opcode(writer_write_bytes(w)))
            throw ckc::ValueError("no ds_write opcode for " + std::to_string(writer_write_bytes(w))
                                  + "-byte stores");
        int waves = (w.threads + 63) / 64;
        for(int it = 0; it < writer_iterations(w); ++it)
            for(int wave = 0; wave < waves; ++wave)
                for(int j = 0; j < w.vec_d; ++j)
                {
                    std::vector<Access> acc = xt_write_accesses(l, w, wave, it, j, 64);
                    if(!acc.empty())
                        worst_w = std::max(
                            worst_w, predict_max(arch, write_opcode(writer_write_bytes(w)), acc));
                }
    }
    return Conflicts{true, worst_r, worst_w};
}

struct Choice
{
    std::string sw;
    bool illegal;
    std::string msg;
    Conflicts pred;
    int bytes;
};

static std::string choose_xt_swizzle(int rows,
                                     int cols,
                                     const rocke_attn_bwd_xt_writer_t* writer,
                                     const std::string& arch,
                                     const std::vector<std::string>& candidates,
                                     int elem_bytes,
                                     int atom_k,
                                     std::vector<Choice>* table_out)
{
    int chunk = (atom_k / 4) * elem_bytes;
    std::vector<Choice> table; /* dict order: first insertion, last value */
    for(const std::string& sw : candidates)
    {
        Choice c;
        c.sw = sw;
        try
        {
            rocke_attn_bwd_xt_layout_t lay = xt_layout(rows, cols, elem_bytes, sw, chunk);
            c.pred = xt_conflicts(lay, writer, arch, atom_k);
            c.illegal = false;
            c.bytes = xt_size(lay);
        }
        catch(const ckc::ValueError& e)
        {
            c.illegal = true;
            c.msg = e.what();
            c.pred = Conflicts{false, 0, 0};
            c.bytes = 0;
        }
        bool found = false;
        for(Choice& t : table)
            if(t.sw == sw)
            {
                t = c;
                found = true;
            }
        if(!found)
            table.push_back(c);
    }
    auto row_of = [&](const std::string& sw) -> const Choice& {
        for(const Choice& t : table)
            if(t.sw == sw)
                return t;
        throw ckc::KeyError(py_repr(sw));
    };
    std::vector<std::string> legal;
    for(const std::string& sw : candidates)
        if(!row_of(sw).illegal)
            legal.push_back(sw);
    if(table_out != NULL)
        *table_out = table;
    if(legal.empty())
        throw ckc::ValueError("no legal XT swizzle among " + py_tuple(candidates));
    if(!row_of(legal[0]).pred.has)
    {
        if(std::find(legal.begin(), legal.end(), "pad8") != legal.end())
            return "pad8";
        return legal[0];
    }
    auto index_of = [&](const std::string& sw) {
        return (int)(std::find(candidates.begin(), candidates.end(), sw) - candidates.begin());
    };
    auto key = [&](const std::string& sw) {
        const Choice& c = row_of(sw);
        return std::make_tuple(c.pred.read, c.pred.write, c.bytes, index_of(sw));
    };
    std::string best = legal[0];
    for(const std::string& sw : legal)
        if(key(sw) < key(best))
            best = sw;
    return best;
}

static const std::vector<std::string>& default_candidates()
{
    static const std::vector<std::string> c = {"xor", "pad8", "pad16", "pad32", "none"};
    return c;
}

std::string planner_xt_swizzle(
    int rows, int cols, int threads, const std::string& arch, int elem_bytes, int atom_k)
{
    typedef std::tuple<int, int, int, std::string, int, int> Key;
    static std::map<Key, std::string> cache;
    static std::mutex mu;
    Key key(rows, cols, threads, arch, elem_bytes, atom_k);
    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(key);
        if(it != cache.end())
            return it->second;
    }
    rocke_attn_bwd_xt_writer_t writer = default_xt_writer(cols, rows, threads, elem_bytes, 16);
    std::string best = choose_xt_swizzle(
        rows, cols, &writer, arch, default_candidates(), elem_bytes, atom_k, NULL);
    std::lock_guard<std::mutex> lock(mu);
    cache[key] = best;
    return best;
}

/* ====================================================================== *
 * DmaSlabLayout
 * ====================================================================== */

static const int DMA_CHUNK = 16;

static int dma_row_bytes(const rocke_attn_bwd_dma_slab_layout_t& l)
{
    return l.cols * l.elem_bytes;
}
static int dma_fold(const rocke_attn_bwd_dma_slab_layout_t& l)
{
    return std::max(1, LINE / dma_row_bytes(l));
}
static int dma_slabs(const rocke_attn_bwd_dma_slab_layout_t& l)
{
    return std::max(1, dma_row_bytes(l) / LINE);
}
static int dma_size(const rocke_attn_bwd_dma_slab_layout_t& l)
{
    return l.rows * dma_row_bytes(l);
}

static rocke_attn_bwd_dma_slab_layout_t dma_slab_layout(int rows, int cols, int elem_bytes)
{
    rocke_attn_bwd_dma_slab_layout_t l{rows, cols, elem_bytes};
    int r = dma_row_bytes(l);
    if(rows <= 0 || r <= 0)
        throw ckc::ValueError("DmaSlabLayout rows and cols must be positive");
    if(r % DMA_CHUNK)
        throw ckc::ValueError("DMA slab rows must be whole 16-byte chunks");
    if(r >= LINE && r % LINE)
        throw ckc::ValueError("DMA slab rows of 128 bytes or more must be whole lines");
    if(r < LINE && (LINE % r || rows % (LINE / r)))
        throw ckc::ValueError("folded DMA slab rows must tile whole lines");
    return l;
}

static int dma_offset(const rocke_attn_bwd_dma_slab_layout_t& l, int row, int col)
{
    if(!(0 <= row && row < l.rows && 0 <= col && col < l.cols))
        throw ckc::ValueError("(" + std::to_string(row) + ", " + std::to_string(col)
                              + ") outside the " + std::to_string(l.rows) + "x"
                              + std::to_string(l.cols) + " tile");
    int b = col * l.elem_bytes;
    int line_row, within, base;
    if(dma_row_bytes(l) >= LINE)
    {
        int s = b / LINE;
        within = b % LINE;
        line_row = row;
        base = s * l.rows * LINE + row * LINE;
    }
    else
    {
        line_row = row / dma_fold(l);
        int sub = row % dma_fold(l);
        within = sub * dma_row_bytes(l) + b;
        base = line_row * LINE;
    }
    int chunk = within / DMA_CHUNK, inner = within % DMA_CHUNK;
    return base + (chunk ^ (line_row % 8)) * DMA_CHUNK + inner;
}

static void dma_source(const rocke_attn_bwd_dma_slab_layout_t& l, int slot, int* row, int* col)
{
    long long p = (long long)slot * DMA_CHUNK;
    if(!(0 <= p && p < dma_size(l)))
        throw ckc::ValueError("slot " + std::to_string(slot) + " outside the buffer");
    if(dma_row_bytes(l) >= LINE)
    {
        int per_slab = l.rows * LINE;
        int s = (int)(p / per_slab), rem = (int)(p % per_slab);
        int r = rem / LINE, w = rem % LINE;
        int chunk = (w / DMA_CHUNK) ^ (r % 8);
        *row = r;
        *col = (s * LINE + chunk * DMA_CHUNK) / l.elem_bytes;
        return;
    }
    int line_row = (int)(p / LINE), w = (int)(p % LINE);
    int within = ((w / DMA_CHUNK) ^ (line_row % 8)) * DMA_CHUNK;
    int sub = within / dma_row_bytes(l), b = within % dma_row_bytes(l);
    *row = line_row * dma_fold(l) + sub;
    *col = b / l.elem_bytes;
}

static std::vector<Access> dma_slab_read_accesses(
    const rocke_attn_bwd_dma_slab_layout_t& l, int m0, int k0, int atom_k, int wave_size)
{
    if(wave_size != 64)
        throw ckc::ValueError("dma_slab_read_accesses models the wave64 MFMA A layout");
    int per_lane = atom_k / 4;
    std::vector<Access> out;
    for(int lane = 0; lane < wave_size; ++lane)
        out.push_back(Access{lane,
                             dma_offset(l, m0 + lane % 16, k0 + (lane / 16) * per_lane),
                             per_lane * l.elem_bytes});
    return out;
}

static Conflicts dma_slab_read_conflicts(const rocke_attn_bwd_dma_slab_layout_t& l,
                                         const std::string& arch,
                                         int atom_k)
{
    if(!predictor_target(arch))
        return Conflicts{false, 0, 0};
    int width = (atom_k / 4) * l.elem_bytes;
    int worst = 1;
    for(int m0 = 0; m0 < l.rows - l.rows % 16; m0 += 16)
        for(int k0 = 0; k0 < l.cols; k0 += atom_k)
        {
            std::vector<Access> acc = dma_slab_read_accesses(l, m0, k0, atom_k, 64);
            worst = std::max(worst, predict_max(arch, read_opcode(width), acc));
        }
    return Conflicts{true, worst, 0};
}

/* ====================================================================== *
 * BwdLdsRequest
 * ====================================================================== */

/* _ELEM_BYTES */
static int elem_bytes_of(const std::string& dtype)
{
    if(dtype == "fp16" || dtype == "f16" || dtype == "bf16")
        return 2;
    throw ckc::KeyError(py_repr(dtype));
}

static std::string strip(const std::string& s)
{
    /* str.strip(): Python whitespace (ASCII subset). */
    size_t a = 0, b = s.size();
    auto ws = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
    };
    while(a < b && ws(s[a]))
        ++a;
    while(b > a && ws(s[b - 1]))
        --b;
    return s.substr(a, b - a);
}

/* BwdLdsRequest.canonical_swizzle(str): sorted (name, value) pairs. */
static std::vector<std::pair<std::string, std::string>> canonical_swizzle(const char* value)
{
    std::vector<std::pair<std::string, std::string>> items;
    if(value == NULL)
        return items;
    std::string v(value);
    size_t start = 0;
    while(true)
    {
        size_t comma = v.find(',', start);
        std::string part = strip(
            v.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if(!part.empty())
        {
            size_t eq = part.find('=');
            std::string name = eq == std::string::npos ? part : part.substr(0, eq);
            std::string sw = eq == std::string::npos ? std::string() : part.substr(eq + 1);
            items.emplace_back(strip(name), strip(sw));
        }
        if(comma == std::string::npos)
            break;
        start = comma + 1;
    }
    std::sort(items.begin(), items.end());
    return items;
}

static std::string swizzle_of(const rocke_attn_bwd_lds_request_t& req,
                              const std::string& name,
                              const std::string& dflt)
{
    /* dict(sorted pairs).get(name, default): the last pair of the name wins. */
    std::string out = dflt;
    for(const auto& kv : canonical_swizzle(req.lds_swizzle))
        if(kv.first == name)
            out = kv.second;
    return out;
}

static std::string s_(const char* s)
{
    return s ? std::string(s) : std::string();
}

static bool in_strs(const std::string& v, std::initializer_list<const char*> allowed)
{
    for(const char* a : allowed)
        if(v == a)
            return true;
    return false;
}

void lds_request_check(const rocke_attn_bwd_lds_request_t& r)
{
    std::string dtype = s_(r.dtype);
    if(!in_strs(dtype, {"fp16", "f16", "bf16"}))
        throw ckc::ValueError("unsupported dtype " + py_repr(dtype));
    std::string kv = s_(r.kv_residency), kt = s_(r.kt_source);
    if(!in_strs(kv, {"reg", "lds"}))
        throw ckc::ValueError("kv_residency must be 'reg' or 'lds'");
    if(!in_strs(kt, {"reg", "lds"}))
        throw ckc::ValueError("kt_source must be 'reg' or 'lds'");
    if(kt == "reg" && kv != "reg")
        throw ckc::ValueError("kt_source = 'reg' requires kv_residency = 'reg'");
    std::string ts = s_(r.transpose_source);
    if(!in_strs(ts, {"xt_lds", "tr_read", "lds_plain", "wmma_lds"}))
        throw ckc::ValueError("unknown transpose_source " + py_repr(ts));
    if(r.ring_depth != 1 && r.ring_depth != 2 && r.ring_depth != 3)
        throw ckc::ValueError("ring_depth must be 1, 2 or 3");
    std::string gp = s_(r.global_path);
    if(!in_strs(gp, {"vgpr", "dma"}))
        throw ckc::ValueError("global_path must be 'vgpr' or 'dma'");
    if(gp == "dma" && ts == "xt_lds")
        throw ckc::ValueError("global_path = 'dma' excludes transpose_source = 'xt_lds'");
    if(!in_strs(s_(r.pt_route), {"relabel", "lds"}))
        throw ckc::ValueError("pt_route must be 'relabel' or 'lds'");
    if(!in_strs(s_(r.dq_mode), {"atomic", "split"}))
        throw ckc::ValueError("dq_mode must be 'atomic' or 'split'");
    if(!in_strs(s_(r.dkv_epilogue), {"lds_staged", "lds_staged_pair", "direct"}))
        throw ckc::ValueError(
            "dkv_epilogue must be one of ('lds_staged', 'lds_staged_pair', 'direct')");
    if(!in_strs(s_(r.lds_order), {"resident_low", "phase_low"}))
        throw ckc::ValueError("lds_order must be one of ('resident_low', 'phase_low')");
    if(r.ds_slots != 1 && r.ds_slots != 2)
        throw ckc::ValueError("ds_slots must be 1 or 2");
    for(const auto& kvp : canonical_swizzle(r.lds_swizzle))
        check_swizzle(kvp.second);
    if((r.atom_k_g13 != 16 && r.atom_k_g13 != 32) || (r.atom_k_g4 != 16 && r.atom_k_g4 != 32))
        throw ckc::ValueError("atom_k_g13 / atom_k_g4 must be 16 or 32");
    for(int v : {r.head_size, r.k_m0, r.k_n0, r.waves})
        if(v <= 0)
            throw ckc::ValueError("tile extents and waves must be positive");
}

/* ====================================================================== *
 * plan_bwd_lds
 * ====================================================================== */

struct Item
{
    const char* name;
    int slot;
    int nbytes;
    std::string desc;
    rocke_attn_bwd_lds_layout_t obj; /* kind DESC: no object */
};

static rocke_attn_bwd_lds_layout_t no_object()
{
    rocke_attn_bwd_lds_layout_t o;
    memset(&o, 0, sizeof(o));
    o.kind = ROCKE_ATTN_BWD_LAYOUT_DESC;
    return o;
}

static void set_desc(char* dst, const std::string& s)
{
    if(s.size() >= ROCKE_ATTN_BWD_LDS_DESC_CAP)
        throw ckc::ValueError("LDS layout description " + py_repr(s)
                              + " exceeds the C port's limit");
    memcpy(dst, s.c_str(), s.size() + 1);
}

/* _normal_layout(rows, cols, eb, swizzle, dma) */
static Item normal_layout(
    const char* name, int rows, int cols, int eb, const std::string& swizzle, bool dma)
{
    Item it{name, 0, 0, "", no_object()};
    if(dma)
    {
        it.obj.kind = ROCKE_ATTN_BWD_LAYOUT_DMA_SLAB;
        it.obj.dma = dma_slab_layout(rows, cols, eb);
        it.nbytes = dma_size(it.obj.dma);
        it.desc = "dma_slab_xor";
        return it;
    }
    if(swizzle == "xor")
    {
        it.obj.kind = ROCKE_ATTN_BWD_LAYOUT_ROW_XOR;
        it.obj.row_xor = row_xor_layout(rows, cols, eb, 16);
        it.nbytes = row_xor_size(it.obj.row_xor);
        it.desc = "rowmajor:xor";
        return it;
    }
    int pitch = cols * eb + pad_bytes(swizzle);
    it.nbytes = rows * pitch;
    it.desc = "rowmajor:" + swizzle;
    return it;
}

/* _xt_layout(rows, cols, eb, swizzle, atom_k) */
static Item
    xt_item(const char* name, int rows, int cols, int eb, const std::string& swizzle, int atom_k)
{
    Item it{name, 0, 0, "", no_object()};
    it.obj.kind = ROCKE_ATTN_BWD_LAYOUT_XT;
    it.obj.xt = xt_layout(rows, cols, eb, swizzle, (atom_k / 4) * eb);
    it.nbytes = xt_size(it.obj.xt);
    it.desc = "xt:" + swizzle;
    return it;
}

static bool in_dma_filled(const char* name)
{
    return strcmp(name, "q") == 0 || strcmp(name, "do") == 0; /* DMA_FILLED */
}

static void plan_check(const rocke_attn_bwd_lds_plan_t& p);
static void plan_check_capacity(const rocke_attn_bwd_lds_plan_t& p);

rocke_attn_bwd_lds_plan_t
    plan_bwd_lds(const rocke_attn_bwd_lds_request_t& req, const std::string& arch, bool strict)
{
    lds_request_check(req);
    int eb = elem_bytes_of(s_(req.dtype));
    int d = req.head_size, m0 = req.k_m0, n0 = req.k_n0;
    int threads = req.waves * req.wave_size;
    bool dma = s_(req.global_path) == "dma";
    bool xt = s_(req.transpose_source) == "xt_lds";
    std::vector<Item> resident, stage_k, stage_v, loop, epilogue;

    auto normal = [&](const char* name, int rows, int cols) {
        return normal_layout(
            name, rows, cols, eb, swizzle_of(req, name, "none"), dma && in_dma_filled(name));
    };
    auto xt_img = [&](const char* name, int rows, int cols, int atom_k) {
        std::string sw = swizzle_of(req, name, "");
        if(sw.empty())
            sw = planner_xt_swizzle(rows, cols, threads, arch, eb, atom_k);
        return xt_item(name, rows, cols, eb, sw, atom_k);
    };

    bool kt_live = s_(req.dq_mode) == "atomic";
    std::string kv = s_(req.kv_residency), kt = s_(req.kt_source);
    if(kv == "lds")
    {
        resident.push_back(normal("k_res", n0, d));
        resident.push_back(normal("v_res", n0, d));
    }
    if(kt == "lds" && kt_live)
    {
        if(xt)
            resident.push_back(xt_img("kt_res", d, n0, req.atom_k_g4));
        else if(kv == "reg")
            resident.push_back(normal("k_res", n0, d));
    }
    if(req.acc_in_lds)
    {
        Item a{"dk_acc", 0, n0 * d * 4, "rowmajor:f32", no_object()};
        Item b{"dv_acc", 0, n0 * d * 4, "rowmajor:f32", no_object()};
        resident.push_back(a);
        resident.push_back(b);
    }
    if(kv == "reg")
    {
        bool has_k_res = false;
        for(const Item& it : resident)
            if(strcmp(it.name, "k_res") == 0)
                has_k_res = true;
        if(!has_k_res)
            stage_k.push_back(normal("k_stage", n0, d));
        if(kt == "reg" && xt && kt_live)
            stage_k.push_back(xt_img("kt_stage", d, n0, req.atom_k_g4));
        stage_v.push_back(normal("v_stage", n0, d));
    }

    int stats_bytes = std::max(m0, req.wave_size) * 4;
    bool stats_in_lds = req.ring_depth > 1 || dma;
    for(int slot = 0; slot < req.ring_depth; ++slot)
    {
        for(const char* name : {"q", "do"})
        {
            Item it = normal(name, m0, d);
            it.slot = slot;
            loop.push_back(it);
        }
        if(xt)
            for(const char* name : {"qt", "dot"})
            {
                Item it = xt_img(name, d, m0, req.atom_k_g13);
                it.slot = slot;
                loop.push_back(it);
            }
        if(stats_in_lds)
        {
            loop.push_back(Item{"lse2", slot, stats_bytes, "vector:f32", no_object()});
            loop.push_back(Item{"dsum", slot, stats_bytes, "vector:f32", no_object()});
        }
    }
    std::string pt_route = s_(req.pt_route);
    if(s_(req.dq_mode) == "atomic" || pt_route == "lds")
    {
        Item it = normal("ds", m0, n0);
        for(int slot = 0; slot < req.ds_slots; ++slot)
        {
            it.slot = slot;
            loop.push_back(it);
        }
    }
    if(pt_route == "lds")
    {
        Item it = normal("pt", m0, n0);
        it.slot = 0;
        loop.push_back(it);
    }
    std::string epi = s_(req.dkv_epilogue);
    if(!req.acc_in_lds && epi != "direct")
    {
        std::string sw = swizzle_of(req, "dkv_out", "none");
        int out_bytes = n0 * (d * eb + pad_bytes(sw));
        rocke_attn_bwd_lds_layout_t out_obj = no_object();
        if(sw == "xor")
        {
            out_obj.kind = ROCKE_ATTN_BWD_LAYOUT_ROW_XOR;
            out_obj.row_xor = row_xor_layout(n0, d, eb, 16);
        }
        int slots = epi == "lds_staged_pair" ? 2 : 1;
        for(int slot = 0; slot < slots; ++slot)
            epilogue.push_back(Item{"dkv_out", slot, out_bytes, "rowmajor:" + sw, out_obj});
    }

    rocke_attn_bwd_lds_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    std::vector<rocke_attn_bwd_lds_buffer_t> placed;
    std::map<std::string, rocke_attn_bwd_lds_layout_t> layouts;

    auto layout_value = [&](const Item& it) {
        rocke_attn_bwd_lds_layout_t v = it.obj;
        if(v.kind == ROCKE_ATTN_BWD_LAYOUT_DESC)
            set_desc(v.desc, it.desc);
        v.name = it.name;
        return v;
    };
    auto add_buffer = [&](const Item& it, int off, unsigned phases, bool res) {
        rocke_attn_bwd_lds_buffer_t b;
        memset(&b, 0, sizeof(b));
        b.name = it.name;
        b.slot = it.slot;
        b.offset = off;
        b.nbytes = it.nbytes;
        b.phases = phases;
        set_desc(b.layout, it.desc);
        b.resident = res;
        placed.push_back(b);
    };
    auto place_phase = [&](const std::vector<Item>& items, unsigned phase, int base) {
        int cur = base;
        for(const Item& it : items)
        {
            add_buffer(it, cur, phase, false);
            if(layouts.find(it.name) == layouts.end()) /* setdefault */
                layouts[it.name] = layout_value(it);
            cur = align_up((long long)cur + it.nbytes);
        }
        return cur - base;
    };
    auto place_resident = [&](int base) {
        int off = base;
        for(const Item& it : resident)
        {
            add_buffer(it, off, ROCKE_ATTN_BWD_PHASES_ALL, true);
            layouts[it.name] = layout_value(it);
            off = align_up((long long)off + it.nbytes);
        }
        return off - base;
    };
    int phase_sizes[4];
    auto place_phases = [&](int base) {
        phase_sizes[0] = place_phase(stage_k, ROCKE_ATTN_BWD_PHASE_STAGE0_K, base);
        phase_sizes[1] = place_phase(stage_v, ROCKE_ATTN_BWD_PHASE_STAGE0_V, base);
        phase_sizes[2] = place_phase(loop, ROCKE_ATTN_BWD_PHASE_LOOP, base);
        phase_sizes[3] = place_phase(epilogue, ROCKE_ATTN_BWD_PHASE_EPILOGUE, base);
    };
    int resident_bytes;
    if(s_(req.lds_order) == "phase_low")
    {
        place_phases(0);
        resident_bytes = place_resident(*std::max_element(phase_sizes, phase_sizes + 4));
    }
    else
    {
        resident_bytes = place_resident(0);
        place_phases(resident_bytes);
    }
    int total = resident_bytes + *std::max_element(phase_sizes, phase_sizes + 4);

    if(placed.size() > ROCKE_ATTN_BWD_LDS_MAX_BUFFERS)
        throw ckc::ValueError("LDS plan of " + std::to_string(placed.size())
                              + " buffers exceeds the C port's limit");
    const rocke_arch_target_t* t = rocke_arch_target_from_gfx(arch.c_str());
    if(t == NULL)
    {
        char reason[512];
        rocke_set_unknown_arch_reason(reason, sizeof(reason), arch.c_str());
        throw ckc::KeyError(reason);
    }
    plan.arch = t->gfx;
    plan.request = req;
    for(size_t i = 0; i < placed.size(); ++i)
        plan.buffers[i] = placed[i];
    plan.n_buffers = (int)placed.size();
    plan.resident_bytes = resident_bytes;
    plan.total_bytes = total;
    plan.capacity_bytes = t->lds_capacity_bytes;
    for(int i = 0; i < 4; ++i)
        plan.phase_bytes[i] = phase_sizes[i];
    int n = 0;
    for(const auto& kvp : layouts) /* sorted(layouts.items()) */
        plan.layouts[n++] = kvp.second;
    plan.n_layouts = n;
    plan_check(plan);
    if(strict)
        plan_check_capacity(plan);
    return plan;
}

static bool plan_fits(const rocke_attn_bwd_lds_plan_t& p)
{
    return p.total_bytes <= p.capacity_bytes;
}

static void plan_check(const rocke_attn_bwd_lds_plan_t& p)
{
    for(int i = 0; i < p.n_buffers; ++i)
    {
        const rocke_attn_bwd_lds_buffer_t& b = p.buffers[i];
        if(b.offset % ALIGN || b.offset < 0 || b.offset + b.nbytes > p.total_bytes)
            throw ckc::ValueError(std::string("LDS buffer ") + b.name + "[" + std::to_string(b.slot)
                                  + "] misplaced");
    }
    for(int ph = 0; ph < 4; ++ph)
    {
        std::vector<const rocke_attn_bwd_lds_buffer_t*> live;
        for(int i = 0; i < p.n_buffers; ++i)
            if(p.buffers[i].phases & (1u << ph))
                live.push_back(&p.buffers[i]);
        std::stable_sort(
            live.begin(), live.end(), [](auto* a, auto* b) { return a->offset < b->offset; });
        for(size_t i = 0; i + 1 < live.size(); ++i)
        {
            const rocke_attn_bwd_lds_buffer_t* a = live[i];
            const rocke_attn_bwd_lds_buffer_t* c = live[i + 1];
            if(a->offset + a->nbytes > c->offset)
                throw ckc::ValueError(std::string("LDS buffers ") + a->name + "["
                                      + std::to_string(a->slot) + "] and " + c->name + "["
                                      + std::to_string(c->slot) + "] overlap in phase "
                                      + PHASE_NAMES[ph]);
        }
    }
}

static void plan_check_capacity(const rocke_attn_bwd_lds_plan_t& p)
{
    if(!plan_fits(p))
        throw ckc::ValueError("LDS footprint " + std::to_string(p.total_bytes)
                              + " bytes exceeds the " + p.arch + " capacity of "
                              + std::to_string(p.capacity_bytes) + " bytes");
}

static const rocke_attn_bwd_lds_buffer_t*
    plan_buffer(const rocke_attn_bwd_lds_plan_t& p, const std::string& name, int slot)
{
    for(int i = 0; i < p.n_buffers; ++i)
        if(name == p.buffers[i].name && p.buffers[i].slot == slot)
            return &p.buffers[i];
    /* str(KeyError(msg)) is repr(msg) */
    throw ckc::KeyError(
        py_repr("no LDS buffer " + py_repr(name) + " slot " + std::to_string(slot)));
}

static std::vector<int> plan_ring_offsets(const rocke_attn_bwd_lds_plan_t& p,
                                          const std::string& name)
{
    std::vector<const rocke_attn_bwd_lds_buffer_t*> bs;
    for(int i = 0; i < p.n_buffers; ++i)
        bs.push_back(&p.buffers[i]);
    std::stable_sort(bs.begin(), bs.end(), [](auto* a, auto* b) { return a->slot < b->slot; });
    std::vector<int> out;
    for(auto* b : bs)
        if(name == b->name)
            out.push_back(b->offset);
    return out;
}

static int plan_ring_stride(const rocke_attn_bwd_lds_plan_t& p, const std::string& name)
{
    std::vector<int> offs = plan_ring_offsets(p, name);
    if(offs.empty())
        throw ckc::KeyError(py_repr("no LDS buffer " + py_repr(name)));
    if(offs.size() == 1)
        return 0;
    int stride = offs[1] - offs[0];
    bool even = stride > 0;
    for(size_t i = 0; even && i + 1 < offs.size(); ++i)
        if(offs[i + 1] - offs[i] != stride)
            even = false;
    if(!even)
        throw ckc::ValueError("ring slots of " + py_repr(name)
                              + " are not evenly spaced: " + py_tuple(offs));
    return stride;
}

template <class Fn>
static rocke_status_t lds_boundary(rocke_attn_bwd_err_t* err, Fn&& fn)
{
    store_err(err, ROCKE_OK, "");
    try
    {
        fn();
        return ROCKE_OK;
    }
    catch(const ckc::Error& e)
    {
        return store_err(err, e.code(), e.what());
    }
    catch(const std::bad_alloc&)
    {
        return store_err(err, ROCKE_ERR_OOM, "out of memory");
    }
}

static void copy_accesses(const std::vector<Access>& acc, Access* out, int* n_out)
{
    if(acc.size() > 64)
        throw ckc::ValueError("more than 64 accesses for one wave");
    for(size_t i = 0; i < acc.size(); ++i)
        out[i] = acc[i];
    if(n_out != NULL)
        *n_out = (int)acc.size();
}

} // namespace rocke_attn_bwd

using namespace rocke_attn_bwd;

extern "C" {

const char* rocke_attn_bwd_lds_phase_name(rocke_attn_bwd_lds_phase_t phase)
{
    for(int i = 0; i < 4; ++i)
        if((unsigned)phase == (1u << i))
            return PHASE_NAMES[i];
    return NULL;
}

rocke_status_t
    rocke_attn_bwd_lds_capacity_bytes(const char* arch, int* out, rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] {
        const rocke_arch_target_t* t = rocke_arch_target_from_gfx(arch ? arch : "");
        if(t == NULL)
        {
            char reason[512];
            rocke_set_unknown_arch_reason(reason, sizeof(reason), arch ? arch : "");
            throw ckc::KeyError(reason);
        }
        if(out != NULL)
            *out = t->lds_capacity_bytes;
    });
}

rocke_status_t rocke_attn_bwd_xt_layout_init(rocke_attn_bwd_xt_layout_t* out,
                                             int rows,
                                             int cols,
                                             int elem_bytes,
                                             const char* swizzle,
                                             int chunk_bytes,
                                             rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] {
        rocke_attn_bwd_xt_layout_t l = xt_layout(rows, cols, elem_bytes, s_(swizzle), chunk_bytes);
        if(out != NULL)
            *out = l;
    });
}

int rocke_attn_bwd_xt_layout_row_pitch(const rocke_attn_bwd_xt_layout_t* l)
{
    return xt_row_pitch(*l);
}

int rocke_attn_bwd_xt_layout_size(const rocke_attn_bwd_xt_layout_t* l)
{
    return xt_size(*l);
}

int rocke_attn_bwd_xt_layout_offset(const rocke_attn_bwd_xt_layout_t* l,
                                    int row,
                                    int col,
                                    rocke_attn_bwd_err_t* err)
{
    int v = -1;
    lds_boundary(err, [&] { v = xt_offset(*l, row, col); });
    return v;
}

rocke_status_t rocke_attn_bwd_xt_layout_check_vector(const rocke_attn_bwd_xt_layout_t* l,
                                                     int col,
                                                     int n,
                                                     rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] { xt_check_vector(*l, col, n); });
}

rocke_status_t rocke_attn_bwd_row_xor_layout_init(rocke_attn_bwd_row_xor_layout_t* out,
                                                  int rows,
                                                  int cols,
                                                  int elem_bytes,
                                                  int chunk_bytes,
                                                  rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] {
        rocke_attn_bwd_row_xor_layout_t l = row_xor_layout(rows, cols, elem_bytes, chunk_bytes);
        if(out != NULL)
            *out = l;
    });
}

rocke_attn_bwd_xt_layout_t
    rocke_attn_bwd_row_xor_layout_image(const rocke_attn_bwd_row_xor_layout_t* l)
{
    rocke_attn_bwd_xt_layout_t x{l->rows, l->cols, l->elem_bytes, "xor", l->chunk_bytes};
    return x;
}

int rocke_attn_bwd_row_xor_layout_row_pitch(const rocke_attn_bwd_row_xor_layout_t* l)
{
    return l->cols * l->elem_bytes;
}

int rocke_attn_bwd_row_xor_layout_size(const rocke_attn_bwd_row_xor_layout_t* l)
{
    return row_xor_size(*l);
}

int rocke_attn_bwd_row_xor_layout_offset(const rocke_attn_bwd_row_xor_layout_t* l,
                                         int row,
                                         int col,
                                         rocke_attn_bwd_err_t* err)
{
    int v = -1;
    lds_boundary(err, [&] { v = xt_offset(row_xor_image(*l), row, col); });
    return v;
}

rocke_status_t rocke_attn_bwd_row_xor_layout_check_vector(const rocke_attn_bwd_row_xor_layout_t* l,
                                                          int col,
                                                          int n,
                                                          rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] { xt_check_vector(row_xor_image(*l), col, n); });
}

rocke_status_t rocke_attn_bwd_xt_writer_init(rocke_attn_bwd_xt_writer_t* out,
                                             int seq,
                                             int depth,
                                             int threads,
                                             int vec_d,
                                             int k_per_thread,
                                             int d_lanes,
                                             int elem_bytes,
                                             rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] {
        rocke_attn_bwd_xt_writer_t w
            = xt_writer(seq, depth, threads, vec_d, k_per_thread, d_lanes, elem_bytes);
        if(out != NULL)
            *out = w;
    });
}

int rocke_attn_bwd_xt_writer_n_dg(const rocke_attn_bwd_xt_writer_t* w)
{
    return writer_n_dg(*w);
}
int rocke_attn_bwd_xt_writer_n_sg(const rocke_attn_bwd_xt_writer_t* w)
{
    return writer_n_sg(*w);
}
int rocke_attn_bwd_xt_writer_d_lanes_eff(const rocke_attn_bwd_xt_writer_t* w)
{
    return writer_d_lanes_eff(*w);
}
int rocke_attn_bwd_xt_writer_units(const rocke_attn_bwd_xt_writer_t* w)
{
    return writer_units(*w);
}
int rocke_attn_bwd_xt_writer_iterations(const rocke_attn_bwd_xt_writer_t* w)
{
    return writer_iterations(*w);
}
int rocke_attn_bwd_xt_writer_write_bytes(const rocke_attn_bwd_xt_writer_t* w)
{
    return writer_write_bytes(*w);
}
int rocke_attn_bwd_xt_writer_stores_per_thread(const rocke_attn_bwd_xt_writer_t* w)
{
    return writer_iterations(*w) * w->vec_d;
}
int rocke_attn_bwd_xt_writer_loads_per_thread(const rocke_attn_bwd_xt_writer_t* w)
{
    return writer_iterations(*w) * w->k_per_thread;
}

bool rocke_attn_bwd_xt_writer_unit_coords(
    const rocke_attn_bwd_xt_writer_t* w, int tid, int iteration, int* s0, int* d0)
{
    return writer_unit_coords(*w, tid, iteration, s0, d0);
}

rocke_status_t rocke_attn_bwd_default_xt_writer(rocke_attn_bwd_xt_writer_t* out,
                                                int seq,
                                                int depth,
                                                int threads,
                                                int elem_bytes,
                                                int max_vec_bytes,
                                                rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] {
        rocke_attn_bwd_xt_writer_t w
            = default_xt_writer(seq, depth, threads, elem_bytes, max_vec_bytes);
        if(out != NULL)
            *out = w;
    });
}

rocke_status_t rocke_attn_bwd_xt_read_accesses(const rocke_attn_bwd_xt_layout_t* layout,
                                               int n0,
                                               int k0,
                                               int atom_k,
                                               int wave_size,
                                               rocke_attn_bwd_lds_access_t out[64],
                                               int* n_out,
                                               rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] {
        copy_accesses(xt_read_accesses(*layout, n0, k0, atom_k, wave_size), out, n_out);
    });
}

rocke_status_t rocke_attn_bwd_xt_write_accesses(const rocke_attn_bwd_xt_layout_t* layout,
                                                const rocke_attn_bwd_xt_writer_t* writer,
                                                int wave,
                                                int iteration,
                                                int j,
                                                int wave_size,
                                                rocke_attn_bwd_lds_access_t out[64],
                                                int* n_out,
                                                rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] {
        copy_accesses(
            xt_write_accesses(*layout, *writer, wave, iteration, j, wave_size), out, n_out);
    });
}

rocke_status_t rocke_attn_bwd_xt_conflicts(const rocke_attn_bwd_xt_layout_t* layout,
                                           const rocke_attn_bwd_xt_writer_t* writer,
                                           const char* arch,
                                           int atom_k,
                                           bool* has_out,
                                           int* read_out,
                                           int* write_out,
                                           rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] {
        Conflicts c = xt_conflicts(*layout, writer, s_(arch), atom_k);
        if(has_out != NULL)
            *has_out = c.has;
        if(read_out != NULL)
            *read_out = c.read;
        if(write_out != NULL)
            *write_out = c.write;
    });
}

rocke_status_t rocke_attn_bwd_choose_xt_swizzle(int rows,
                                                int cols,
                                                const rocke_attn_bwd_xt_writer_t* writer,
                                                const char* arch,
                                                const char* const* candidates,
                                                int n_candidates,
                                                int elem_bytes,
                                                int atom_k,
                                                const char** best,
                                                rocke_attn_bwd_xt_choice_t* table,
                                                int* n_table,
                                                rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] {
        std::vector<std::string> cands;
        if(candidates == NULL || n_candidates == 0)
            cands = default_candidates();
        else
            for(int i = 0; i < n_candidates; ++i)
                cands.push_back(s_(candidates[i]));
        if((int)cands.size() > ROCKE_ATTN_BWD_XT_CANDIDATES_MAX)
            throw ckc::ValueError("choose_xt_swizzle: more candidates than the C port's limit");
        std::vector<Choice> rows_out;
        std::string pick;
        try
        {
            pick = choose_xt_swizzle(
                rows, cols, writer, s_(arch), cands, elem_bytes, atom_k, &rows_out);
        }
        catch(...)
        {
            if(n_table != NULL)
                *n_table = 0;
            throw;
        }
        /* static spelling of the pick: a legal candidate, so in _SWIZZLES */
        if(best != NULL)
            *best = static_swizzle(pick);
        for(size_t i = 0; i < rows_out.size(); ++i)
        {
            rocke_attn_bwd_xt_choice_t& t = table[i];
            memset(&t, 0, sizeof(t));
            t.swizzle = NULL;
            for(int k = 0; k < (int)cands.size(); ++k)
                if(cands[k] == rows_out[i].sw)
                {
                    t.swizzle = candidates ? candidates[k] : static_swizzle(cands[k]);
                    break;
                }
            t.illegal = rows_out[i].illegal;
            snprintf(t.illegal_msg, sizeof(t.illegal_msg), "%s", rows_out[i].msg.c_str());
            t.has_prediction = rows_out[i].pred.has;
            t.read = rows_out[i].pred.read;
            t.write = rows_out[i].pred.write;
            t.bytes = rows_out[i].bytes;
        }
        if(n_table != NULL)
            *n_table = (int)rows_out.size();
    });
}

const char* rocke_attn_bwd_planner_xt_swizzle(int rows,
                                              int cols,
                                              int threads,
                                              const char* arch,
                                              int elem_bytes,
                                              int atom_k,
                                              rocke_attn_bwd_err_t* err)
{
    const char* out = NULL;
    lds_boundary(err, [&] {
        out = static_swizzle(planner_xt_swizzle(rows, cols, threads, s_(arch), elem_bytes, atom_k));
    });
    return out;
}

rocke_status_t rocke_attn_bwd_dma_slab_layout_init(rocke_attn_bwd_dma_slab_layout_t* out,
                                                   int rows,
                                                   int cols,
                                                   int elem_bytes,
                                                   rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] {
        rocke_attn_bwd_dma_slab_layout_t l = dma_slab_layout(rows, cols, elem_bytes);
        if(out != NULL)
            *out = l;
    });
}

int rocke_attn_bwd_dma_slab_layout_row_bytes(const rocke_attn_bwd_dma_slab_layout_t* l)
{
    return dma_row_bytes(*l);
}
int rocke_attn_bwd_dma_slab_layout_fold(const rocke_attn_bwd_dma_slab_layout_t* l)
{
    return dma_fold(*l);
}
int rocke_attn_bwd_dma_slab_layout_slabs(const rocke_attn_bwd_dma_slab_layout_t* l)
{
    return dma_slabs(*l);
}
int rocke_attn_bwd_dma_slab_layout_size(const rocke_attn_bwd_dma_slab_layout_t* l)
{
    return dma_size(*l);
}

int rocke_attn_bwd_dma_slab_layout_offset(const rocke_attn_bwd_dma_slab_layout_t* l,
                                          int row,
                                          int col,
                                          rocke_attn_bwd_err_t* err)
{
    int v = -1;
    lds_boundary(err, [&] { v = dma_offset(*l, row, col); });
    return v;
}

rocke_status_t rocke_attn_bwd_dma_slab_layout_dma_source(const rocke_attn_bwd_dma_slab_layout_t* l,
                                                         int slot,
                                                         int* row,
                                                         int* col,
                                                         rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] {
        int r = 0, c = 0;
        dma_source(*l, slot, &r, &c);
        if(row != NULL)
            *row = r;
        if(col != NULL)
            *col = c;
    });
}

rocke_status_t rocke_attn_bwd_dma_slab_read_accesses(const rocke_attn_bwd_dma_slab_layout_t* l,
                                                     int m0,
                                                     int k0,
                                                     int atom_k,
                                                     int wave_size,
                                                     rocke_attn_bwd_lds_access_t out[64],
                                                     int* n_out,
                                                     rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] {
        copy_accesses(dma_slab_read_accesses(*l, m0, k0, atom_k, wave_size), out, n_out);
    });
}

rocke_status_t rocke_attn_bwd_dma_slab_read_conflicts(const rocke_attn_bwd_dma_slab_layout_t* l,
                                                      const char* arch,
                                                      int atom_k,
                                                      bool* has_out,
                                                      int* worst_out,
                                                      rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] {
        Conflicts c = dma_slab_read_conflicts(*l, s_(arch), atom_k);
        if(has_out != NULL)
            *has_out = c.has;
        if(worst_out != NULL)
            *worst_out = c.read;
    });
}

rocke_attn_bwd_lds_request_t rocke_attn_bwd_lds_request_default(void)
{
    rocke_attn_bwd_lds_request_t r;
    memset(&r, 0, sizeof(r));
    r.wave_size = 64;
    r.dtype = "fp16";
    r.kv_residency = "reg";
    r.kt_source = "reg";
    r.transpose_source = "xt_lds";
    r.ring_depth = 1;
    r.global_path = "vgpr";
    r.pt_route = "relabel";
    r.acc_in_lds = false;
    r.dq_mode = "atomic";
    r.lds_swizzle = NULL;
    r.dkv_epilogue = "lds_staged";
    r.lds_order = "resident_low";
    r.ds_slots = 1;
    r.atom_k_g13 = 16;
    r.atom_k_g4 = 16;
    return r;
}

rocke_status_t rocke_attn_bwd_lds_request_init(const rocke_attn_bwd_lds_request_t* req,
                                               rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] { lds_request_check(*req); });
}

int rocke_attn_bwd_lds_request_elem_bytes(const rocke_attn_bwd_lds_request_t* req)
{
    (void)req;
    return 2; /* _ELEM_BYTES of every legal dtype */
}

int rocke_attn_bwd_lds_request_threads(const rocke_attn_bwd_lds_request_t* req)
{
    return req->waves * req->wave_size;
}

const char* rocke_attn_bwd_lds_request_swizzle_of(const rocke_attn_bwd_lds_request_t* req,
                                                  const char* name,
                                                  const char* dflt)
{
    std::string v = swizzle_of(*req, s_(name), s_(dflt));
    const char* st = static_swizzle(v);
    if(st != NULL)
        return st;
    return dflt; /* only the default can be outside _SWIZZLES once validated */
}

rocke_status_t rocke_attn_bwd_plan_lds(const rocke_attn_bwd_lds_request_t* req,
                                       const char* arch,
                                       bool strict,
                                       rocke_attn_bwd_lds_plan_t* out,
                                       rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] {
        rocke_attn_bwd_lds_plan_t p = plan_bwd_lds(*req, s_(arch), strict);
        if(out != NULL)
            *out = p;
    });
}

bool rocke_attn_bwd_lds_plan_fits(const rocke_attn_bwd_lds_plan_t* p)
{
    return plan_fits(*p);
}

int rocke_attn_bwd_lds_plan_phase_offset(const rocke_attn_bwd_lds_plan_t* p)
{
    return s_(p->request.lds_order) == "phase_low" ? 0 : p->resident_bytes;
}

int rocke_attn_bwd_lds_plan_unaliased_bytes(const rocke_attn_bwd_lds_plan_t* p)
{
    int s = 0;
    for(int i = 0; i < p->n_buffers; ++i)
        s += align_up(p->buffers[i].nbytes);
    return s;
}

const rocke_attn_bwd_lds_buffer_t* rocke_attn_bwd_lds_plan_buffer(
    const rocke_attn_bwd_lds_plan_t* p, const char* name, int slot, rocke_attn_bwd_err_t* err)
{
    const rocke_attn_bwd_lds_buffer_t* b = NULL;
    lds_boundary(err, [&] { b = plan_buffer(*p, s_(name), slot); });
    return b;
}

bool rocke_attn_bwd_lds_plan_has(const rocke_attn_bwd_lds_plan_t* p, const char* name)
{
    for(int i = 0; i < p->n_buffers; ++i)
        if(s_(name) == p->buffers[i].name)
            return true;
    return false;
}

int rocke_attn_bwd_lds_plan_offset(const rocke_attn_bwd_lds_plan_t* p,
                                   const char* name,
                                   int slot,
                                   rocke_attn_bwd_err_t* err)
{
    int v = -1;
    lds_boundary(err, [&] { v = plan_buffer(*p, s_(name), slot)->offset; });
    return v;
}

int rocke_attn_bwd_lds_plan_ring_offsets(const rocke_attn_bwd_lds_plan_t* p,
                                         const char* name,
                                         int out[ROCKE_ATTN_BWD_LDS_MAX_BUFFERS])
{
    std::vector<int> offs = plan_ring_offsets(*p, s_(name));
    for(size_t i = 0; i < offs.size(); ++i)
        out[i] = offs[i];
    return (int)offs.size();
}

int rocke_attn_bwd_lds_plan_ring_stride(const rocke_attn_bwd_lds_plan_t* p,
                                        const char* name,
                                        rocke_attn_bwd_err_t* err)
{
    int v = -1;
    lds_boundary(err, [&] { v = plan_ring_stride(*p, s_(name)); });
    return v;
}

const rocke_attn_bwd_lds_layout_t* rocke_attn_bwd_lds_plan_layout(
    const rocke_attn_bwd_lds_plan_t* p, const char* name, rocke_attn_bwd_err_t* err)
{
    const rocke_attn_bwd_lds_layout_t* l = NULL;
    lds_boundary(err, [&] {
        for(int i = 0; i < p->n_layouts; ++i)
            if(s_(name) == p->layouts[i].name)
            {
                l = &p->layouts[i];
                return;
            }
        throw ckc::KeyError(py_repr(s_(name)));
    });
    return l;
}

rocke_status_t rocke_attn_bwd_lds_plan_check(const rocke_attn_bwd_lds_plan_t* p,
                                             rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] { plan_check(*p); });
}

rocke_status_t rocke_attn_bwd_lds_plan_check_capacity(const rocke_attn_bwd_lds_plan_t* p,
                                                      rocke_attn_bwd_err_t* err)
{
    return lds_boundary(err, [&] { plan_check_capacity(*p); });
}

} /* extern "C" */
