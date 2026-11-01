// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tensor_gett_pack.cpp -- C++ mirror of rocke/instances/common/tensor/pack.py
 * (TensorPackSpec, is_valid_spec, build_tensor_pack, tensor_pack_grid /
 * _block): one strided f16 / bf16 operand copied into a packed mode order, the
 * pack step of the tensor GETT routes. See rocke/instance_tensor_gett.h.
 *
 * Every builder call is made in the order of the Python function it mirrors
 * (an argument that emits ops is computed in its own statement first).
 */
#include "rocke/instance_tensor_gett.h"

#include "rocke/error_boundary.hpp"
#include "rocke/ir_internal.h"

#include <string.h>

#include <algorithm>
#include <set>

namespace rocke
{
namespace gett
{
namespace
{

using V = rocke_value_t;

/* tiled form: tile edge (elements) along O and I, LDS row pad (elements) */
constexpr int TILE = 64;
constexpr int TILE_PAD = 2;

V* ci(rocke_ir_builder_t* b, int64_t v)
{
    return rocke_b_const_i32(b, v);
}

bool set_why(std::string* why, const std::string& m)
{
    if(why != nullptr)
        *why = m;
    return false;
}

/* Python repr of a str */
std::string repr(const std::string& s)
{
    return "'" + s + "'";
}

const rocke_type_t* ty_of(const std::string& dtype)
{
    if(dtype == "f16" || dtype == "fp16")
        return rocke_f16();
    if(dtype == "bf16")
        return rocke_bf16();
    return nullptr;
}

/* _digits(b, x, digits): the source offset of index x over (extent, stride)
 * digits, inner first */
V* digits(rocke_ir_builder_t* b, V* x, const std::vector<Digit>& dg)
{
    V* off = nullptr;
    V* q = x;
    for(size_t j = 0; j < dg.size(); ++j)
    {
        bool last = j + 1 == dg.size();
        V* d = q;
        if(!last)
        {
            V* e = ci(b, dg[j].e);
            d = rocke_b_mod(b, q, e);
        }
        V* term = d;
        if(dg[j].s != 1)
        {
            V* s = ci(b, dg[j].s);
            term = rocke_b_mul(b, d, s);
        }
        off = off == nullptr ? term : rocke_b_add(b, off, term);
        if(!last)
        {
            V* e = ci(b, dg[j].e);
            q = rocke_b_div(b, q, e);
        }
    }
    return off;
}

int64_t src_bytes(const TensorPackSpec& spec)
{
    std::map<std::string, int64_t> st;
    spec.src_strides(&st);
    std::map<std::string, int64_t> ext = spec.ext();
    int64_t span = 0;
    for(const auto& m : spec.modes_src)
        span += (ext.at(m) - 1) * st.at(m);
    return 2 * (1 + span);
}

int64_t prod(const std::vector<std::string>& modes, const std::map<std::string, int64_t>& ext)
{
    int64_t p = 1;
    for(const auto& m : modes)
        p *= ext.at(m);
    return p;
}

void store(rocke_ir_builder_t* b, bool bf, V* rs, V* off, V* c0, V* vec)
{
    if(bf)
        rocke_b_buffer_store_vN_bf16(b, rs, off, c0, vec, 4);
    else
        rocke_b_buffer_store_vN_f16(b, rs, off, c0, vec, 4);
}

/* _build_tiled: one 64 (O) x 64 (I) tile per workgroup through a padded
 * [I][O] LDS tile */
void build_tiled(rocke_ir_builder_t* b,
                 const TensorPackSpec& spec,
                 const rocke_type_t* dt,
                 V* rs_x,
                 V* rs_y,
                 V* c0,
                 V* big,
                 bool bf)
{
    std::vector<std::string> O, I;
    spec.tiled_groups(&O, &I);
    std::map<std::string, int64_t> ext = spec.ext(), st;
    spec.src_strides(&st);
    int64_t p_o = prod(O, ext);
    int64_t p_i = prod(I, ext);
    std::vector<Digit> odig = merge_digits(O, ext, st);
    std::vector<Digit> idig = merge_digits(I, ext, st);
    int shape[2] = {TILE, TILE + TILE_PAD};
    V* sm = rocke_b_smem_alloc(b, dt, shape, 2, "pack_tile");
    V* tid = rocke_b_thread_id_x(b);
    V* bx = rocke_b_block_id_x(b);
    V* t64a = ci(b, TILE);
    V* i_blk = rocke_b_mul(b, bx, t64a);
    V* by = rocke_b_block_id_y(b);
    V* t64b = ci(b, TILE);
    V* o_blk = rocke_b_mul(b, by, t64b);
    /* load: thread -> O pair op (32 per row) and I rows ib + 8 j */
    V* h1 = ci(b, TILE / 2);
    V* op = rocke_b_mod(b, tid, h1);
    V* h2 = ci(b, TILE / 2);
    V* ib = rocke_b_div(b, tid, h2);
    V* two = ci(b, 2);
    V* op2 = rocke_b_mul(b, op, two);
    V* o = rocke_b_add(b, o_blk, op2);
    V* po = ci(b, p_o);
    V* ok_o = rocke_b_cmp_lt(b, o, po);
    V* ro = digits(b, o, odig);
    const int rows_per_pass = 256 / (TILE / 2);
    for(int j = 0; j < TILE / rows_per_pass; ++j)
    {
        V* il = ib;
        if(j)
        {
            V* jo = ci(b, j * rows_per_pass);
            il = rocke_b_add(b, ib, jo);
        }
        V* i = rocke_b_add(b, i_blk, il);
        V* pi = ci(b, p_i);
        V* lt = rocke_b_cmp_lt(b, i, pi);
        V* ok = rocke_b_land(b, ok_o, lt);
        V* di = digits(b, i, idig);
        V* sum = rocke_b_add(b, ro, di);
        V* c2 = ci(b, 2);
        V* so = rocke_b_mul(b, sum, c2);
        so = rocke_b_select(b, ok, so, big);
        V* pr = rocke_b_buffer_load_vN(b, rs_x, so, c0, dt, 2);
        V* c2b = ci(b, 2);
        V* col = rocke_b_mul(b, op, c2b);
        V* idx[2] = {il, col};
        rocke_b_smem_store_vN(b, sm, idx, 2, pr, 2);
    }
    rocke_b_sync(b);
    /* store: 16-byte vectors of 8 consecutive I per (o, chunk) */
    const int chunks = TILE / 8;
    for(int jj = 0; jj < TILE * chunks / 256; ++jj)
    {
        V* cidx = tid;
        if(jj)
        {
            V* off = ci(b, 256 * jj);
            cidx = rocke_b_add(b, tid, off);
        }
        V* ch1 = ci(b, chunks);
        V* ol = rocke_b_div(b, cidx, ch1);
        V* ch2 = ci(b, chunks);
        V* m = rocke_b_mod(b, cidx, ch2);
        V* c8 = ci(b, 8);
        V* ic = rocke_b_mul(b, m, c8);
        V* els[8];
        for(int e = 0; e < 8; ++e)
        {
            V* r = ic;
            if(e)
            {
                V* ce = ci(b, e);
                r = rocke_b_add(b, ic, ce);
            }
            V* idx[2] = {r, ol};
            V* ld = rocke_b_smem_load_vN(b, sm, idx, 2, dt, 1);
            els[e] = rocke_b_vec_extract(b, ld, 0);
        }
        V* vec = rocke_b_vec_pack(b, els, 8, dt);
        V* o2 = rocke_b_add(b, o_blk, ol);
        V* i2 = rocke_b_add(b, i_blk, ic);
        V* po2 = ci(b, p_o);
        V* lo = rocke_b_cmp_lt(b, o2, po2);
        V* pi2 = ci(b, p_i);
        V* li = rocke_b_cmp_lt(b, i2, pi2);
        V* ok = rocke_b_land(b, lo, li);
        V* pi3 = ci(b, p_i);
        V* rowo = rocke_b_mul(b, o2, pi3);
        V* lin = rocke_b_add(b, rowo, i2);
        V* c2 = ci(b, 2);
        V* dof = rocke_b_mul(b, lin, c2);
        V* sel = rocke_b_select(b, ok, dof, big);
        store(b, bf, rs_y, sel, c0, vec);
    }
}

rocke_kernel_def_t* build_impl(rocke_ir_builder_t* b, const TensorPackSpec& spec, const char* arch)
{
    std::string why;
    bool ok = pack_is_valid_spec(spec, arch, &why);
    std::string name = spec.kernel_name();
    if(rocke_ir_builder_init(b, name.c_str()) != ROCKE_OK)
        return nullptr;
    if(!ok)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid tensor pack spec: %s", why.c_str());
        return nullptr;
    }
    const rocke_type_t* dt = ty_of(spec.dtype);
    bool bf = dt == rocke_bf16();
    const rocke_type_t* pty = rocke_ptr_type(b, dt, "global");
    rocke_param_opts_t xo;
    memset(&xo, 0, sizeof xo);
    xo.noalias = xo.noalias_set = true;
    xo.readonly = xo.readonly_set = true;
    xo.align = 4;
    xo.align_set = true;
    V* X = rocke_b_param(b, "X", pty, &xo);
    rocke_param_opts_t yo;
    memset(&yo, 0, sizeof yo);
    yo.noalias = yo.noalias_set = true;
    yo.writeonly = yo.writeonly_set = true;
    yo.align = 16;
    yo.align_set = true;
    V* Y = rocke_b_param(b, "Y", pty, &yo);
    int64_t n = spec.numel();
    V* xb = ci(b, src_bytes(spec));
    V* rs_x = rocke_b_buffer_rsrc(b, X, xb);
    V* yb = ci(b, 2 * n);
    V* rs_y = rocke_b_buffer_rsrc(b, Y, yb);
    V* c0 = ci(b, 0);
    V* big = ci(b, 0x7FFFFFF0);
    std::string form = spec.resolved_form();
    if(form == "tiled")
    {
        build_tiled(b, spec, dt, rs_x, rs_y, c0, big, bf);
        return rocke_ir_builder_kernel(b);
    }
    std::vector<Digit> dig = spec.src_digits();
    V* tid = rocke_b_thread_id_x(b);
    V* bx = rocke_b_block_id_x(b);
    V* per = ci(b, (int64_t)spec.block * spec.groups);
    V* t = rocke_b_mul(b, bx, per);
    V* t2 = rocke_b_add(b, t, tid);
    V* c8 = ci(b, 8);
    V* base = rocke_b_mul(b, t2, c8);
    const rocke_type_t* vec8 = rocke_vector_type(b, dt, 8);
    for(int g = 0; g < spec.groups; ++g)
    {
        V* e0 = base;
        if(g)
        {
            V* go = ci(b, (int64_t)g * spec.block * 8);
            e0 = rocke_b_add(b, base, go);
        }
        V* nn = ci(b, n);
        V* inb = rocke_b_cmp_lt(b, e0, nn);
        V* vec = nullptr;
        if(form == "pairs")
        {
            V* words[4];
            for(int j = 0; j < 8; j += 2)
            {
                V* e = e0;
                if(j)
                {
                    V* cj = ci(b, j);
                    e = rocke_b_add(b, e0, cj);
                }
                V* d = digits(b, e, dig);
                V* c2 = ci(b, 2);
                V* so = rocke_b_mul(b, d, c2);
                so = rocke_b_select(b, inb, so, big);
                V* ld = rocke_b_buffer_load_vN(b, rs_x, so, c0, dt, 2);
                words[j / 2] = rocke_b_bitcast(b, ld, rocke_i32());
            }
            V* pk = rocke_b_vec_pack(b, words, 4, rocke_i32());
            vec = rocke_b_vec_bitcast(b, pk, vec8);
        }
        else
        {
            V* els[8];
            for(int j = 0; j < 8; ++j)
            {
                V* e = e0;
                if(j)
                {
                    V* cj = ci(b, j);
                    e = rocke_b_add(b, e0, cj);
                }
                V* d = digits(b, e, dig);
                V* c2 = ci(b, 2);
                V* so = rocke_b_mul(b, d, c2);
                so = rocke_b_select(b, inb, so, big);
                els[j] = rocke_b_buffer_load(b, rs_x, so, c0, dt);
            }
            vec = rocke_b_vec_pack(b, els, 8, dt);
        }
        V* c2 = ci(b, 2);
        V* dm = rocke_b_mul(b, e0, c2);
        V* dof = rocke_b_select(b, inb, dm, big);
        store(b, bf, rs_y, dof, c0, vec);
    }
    return rocke_ir_builder_kernel(b);
}

} // namespace

/* ------------------------------------------------------------ TensorPackSpec */

TensorPackSpec TensorPackSpec::make(const std::vector<std::string>& modes_src,
                                    const std::vector<std::string>& modes_dst,
                                    const std::map<std::string, int64_t>& extents,
                                    const std::map<std::string, int64_t>* strides_src,
                                    const std::string& dtype)
{
    TensorPackSpec s;
    s.modes_src = modes_src;
    s.modes_dst = modes_dst;
    s.extents = ModeMap(extents.begin(), extents.end());
    if(strides_src != nullptr)
    {
        s.has_strides_src = true;
        s.strides_src = ModeMap(strides_src->begin(), strides_src->end());
    }
    s.dtype = dtype;
    return s;
}

std::map<std::string, int64_t> TensorPackSpec::ext() const
{
    return std::map<std::string, int64_t>(extents.begin(), extents.end());
}

bool TensorPackSpec::src_strides(std::map<std::string, int64_t>* out) const
{
    std::map<std::string, int64_t> e = ext();
    if(!has_strides_src)
    {
        for(const auto& m : modes_src)
            if(e.find(m) == e.end())
                return false;
        *out = packed_strides(modes_src, e);
        return true;
    }
    *out = std::map<std::string, int64_t>(strides_src.begin(), strides_src.end());
    for(const auto& m : modes_src)
        if(out->find(m) == out->end())
            return false;
    return true;
}

int64_t TensorPackSpec::numel() const
{
    return prod(modes_dst, ext());
}

std::vector<Digit> TensorPackSpec::src_digits() const
{
    std::map<std::string, int64_t> st;
    src_strides(&st);
    return merge_digits(modes_dst, ext(), st);
}

bool TensorPackSpec::unit_mode(std::string* out) const
{
    std::map<std::string, int64_t> st, e = ext();
    src_strides(&st);
    for(const auto& m : modes_src)
        if(st.at(m) == 1 && e.at(m) > 1)
        {
            *out = m;
            return true;
        }
    return false;
}

bool TensorPackSpec::pairs() const
{
    std::map<std::string, int64_t> st;
    src_strides(&st);
    if(st.at(modes_dst.back()) != 1)
        return false;
    std::vector<Digit> sd = src_digits();
    if(sd[0].e % 2 != 0)
        return false;
    for(size_t i = 1; i < sd.size(); ++i)
        if(sd[i].s % 2 != 0)
            return false;
    return true;
}

bool TensorPackSpec::tiled_groups(std::vector<std::string>* O, std::vector<std::string>* I) const
{
    std::string u;
    if(!unit_mode(&u))
        return false;
    auto it = std::find(modes_dst.begin(), modes_dst.end(), u);
    if(it == modes_dst.end())
        return false;
    size_t k = (size_t)(it - modes_dst.begin());
    std::vector<std::string> o(modes_dst.begin(), modes_dst.begin() + k + 1);
    std::vector<std::string> i(modes_dst.begin() + k + 1, modes_dst.end());
    if(i.empty())
        return false;
    std::map<std::string, int64_t> e = ext(), st;
    src_strides(&st);
    if(prod(i, e) % 8 != 0)
        return false;
    std::vector<Digit> od = merge_digits(o, e, st);
    if(od[0].s != 1 || od[0].e % 2 != 0)
        return false;
    for(size_t j = 1; j < od.size(); ++j)
        if(od[j].s % 2 != 0)
            return false;
    for(const auto& m : i)
        if(st.at(m) % 2 != 0)
            return false;
    *O = o;
    *I = i;
    return true;
}

std::string TensorPackSpec::resolved_form() const
{
    if(form != "auto")
        return form;
    if(pairs())
        return "pairs";
    std::vector<std::string> o, i;
    if(tiled_groups(&o, &i))
        return "tiled";
    return "elements";
}

std::string TensorPackSpec::json_key() const
{
    /* json.dumps(asdict(spec), sort_keys=True) */
    std::string o = "{";
    o += "\"block\": " + std::to_string(block);
    o += ", \"dtype\": " + json_str(dtype);
    o += ", \"extents\": " + json_modemap(extents);
    o += ", \"form\": " + json_str(form);
    o += ", \"groups\": " + std::to_string(groups);
    o += ", \"modes_dst\": " + json_modes(modes_dst);
    o += ", \"modes_src\": " + json_modes(modes_src);
    o += std::string(", \"strides_src\": ")
         + (has_strides_src ? json_modemap(strides_src) : std::string("null"));
    return o + "}";
}

std::string TensorPackSpec::kernel_name() const
{
    std::string dt = dtype == "f16" || dtype == "fp16" ? "f16" : dtype;
    return "tensor_pack_" + dt + "_" + sha1_hex(json_key()).substr(0, 12);
}

bool pack_is_valid_spec(const TensorPackSpec& spec, const char* /*arch*/, std::string* why)
{
    if(!(spec.dtype == "fp16" || spec.dtype == "bf16" || spec.dtype == "f16"))
        return set_why(why, "dtype " + repr(spec.dtype) + " not in ('fp16', 'bf16')");
    std::vector<std::string> a = spec.modes_src, d = spec.modes_dst;
    std::sort(a.begin(), a.end());
    std::sort(d.begin(), d.end());
    if(a != d)
        return set_why(why, "source and destination modes differ");
    std::set<std::string> em, sm(spec.modes_src.begin(), spec.modes_src.end());
    for(const auto& kv : spec.extents)
        em.insert(kv.first);
    if(em != sm)
        return set_why(why, "extents must name exactly the tensor's modes");
    if(!(spec.form == "auto" || spec.form == "pairs" || spec.form == "tiled"
         || spec.form == "elements"))
        return set_why(
            why, "form " + repr(spec.form) + " not in ('auto', 'pairs', 'tiled', 'elements')");
    if(!(spec.groups == 1 || spec.groups == 2 || spec.groups == 4))
        return set_why(why, "groups " + std::to_string(spec.groups) + " not in (1, 2, 4)");
    if(!(spec.block == 64 || spec.block == 128 || spec.block == 256 || spec.block == 512))
        return set_why(why, "block " + std::to_string(spec.block) + " not in (64, 128, 256, 512)");
    std::map<std::string, int64_t> ext = spec.ext(), st;
    if(!spec.src_strides(&st))
        return set_why(why, "strides_src must name every source mode");
    for(const auto& kv : ext)
        if(kv.second <= 0)
            return set_why(why, "extents must be positive");
    if(spec.numel() % 8 != 0)
        return set_why(why,
                       std::to_string(spec.numel())
                           + " elements is not a multiple of 8 (16-byte stores)");
    std::string form = spec.resolved_form();
    if(form == "pairs" && !spec.pairs())
        return set_why(
            why, "pairs form: the destination's fastest mode is not the source's unit-stride run");
    if(form == "tiled")
    {
        std::vector<std::string> o, i;
        if(!spec.tiled_groups(&o, &i))
            return set_why(why, "tiled form does not apply (see TensorPackSpec.tiled_groups)");
        if(spec.block != 256)
            return set_why(why, "tiled form: block must be 256");
    }
    int64_t span = 0;
    for(const auto& m : spec.modes_src)
        span += (ext.at(m) - 1) * st.at(m);
    span = 2 * (1 + span);
    if(span > MAX_PACK_BYTES || 2 * spec.numel() > MAX_PACK_BYTES)
        return set_why(why, "tensor spans more than 2 GiB (32-bit buffer offsets)");
    return true;
}

rocke_kernel_def_t*
    build_tensor_pack(rocke_ir_builder_t* b, const TensorPackSpec& spec, const char* arch)
{
    return ckc::guard_builder(b,
                              [&]() -> rocke_kernel_def_t* { return build_impl(b, spec, arch); });
}

void tensor_pack_grid(const TensorPackSpec& spec, int grid[3])
{
    grid[1] = grid[2] = 1;
    if(spec.resolved_form() == "tiled")
    {
        std::vector<std::string> O, I;
        spec.tiled_groups(&O, &I);
        std::map<std::string, int64_t> e = spec.ext();
        grid[0] = (int)((prod(I, e) + TILE - 1) / TILE);
        grid[1] = (int)((prod(O, e) + TILE - 1) / TILE);
        return;
    }
    int64_t per = (int64_t)spec.block * spec.groups * 8;
    grid[0] = (int)std::max<int64_t>(1, (spec.numel() + per - 1) / per);
}

int tensor_pack_block(const TensorPackSpec& spec)
{
    return spec.block;
}

} // namespace gett
} // namespace rocke
