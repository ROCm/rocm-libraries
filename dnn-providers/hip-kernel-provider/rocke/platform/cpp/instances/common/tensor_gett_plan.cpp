// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tensor_gett_plan.cpp -- C++ mirror of
 * rocke/instances/common/tensor/gett_plan.py: the host-side plan of a native
 * GETT contraction (no IR). See rocke/instance_tensor_gett.h.
 */
#include "rocke/instance_tensor_gett.h"

#include <algorithm>
#include <set>
#include <sstream>

namespace rocke
{
namespace gett
{

std::map<std::string, int64_t> packed_strides(const std::vector<std::string>& modes,
                                              const std::map<std::string, int64_t>& ext)
{
    std::map<std::string, int64_t> st;
    int64_t s = 1;
    for(auto it = modes.rbegin(); it != modes.rend(); ++it)
    {
        st[*it] = s;
        s *= ext.at(*it);
    }
    return st;
}

std::vector<Digit> merge_digits(const std::vector<std::string>& order,
                                const std::map<std::string, int64_t>& ext,
                                const std::map<std::string, int64_t>& strides)
{
    std::vector<Digit> dig;
    for(auto it = order.rbegin(); it != order.rend(); ++it)
    {
        int64_t e = ext.at(*it), s = strides.at(*it);
        if(e == 1)
            continue;
        if(!dig.empty() && s == dig.back().e * dig.back().s)
            dig.back().e *= e;
        else
            dig.push_back({e, s});
    }
    if(dig.empty())
        dig.push_back({1, 0});
    return dig;
}

namespace
{

/* k_split: (local, tile) mode-level digits, or false */
bool k_split(const std::vector<std::string>& order,
             const std::map<std::string, int64_t>& ext,
             int64_t T,
             std::vector<KDigit>* local,
             std::vector<KDigit>* tile)
{
    int64_t rem = T;
    local->clear();
    tile->clear();
    for(const auto& m : order)
    {
        int64_t e = ext.at(m);
        if(rem == 1)
            tile->push_back({e, m, 1});
        else if(e <= rem)
        {
            if(rem % e)
                return false;
            local->push_back({e, m, 1});
            rem /= e;
        }
        else
        {
            if(e % rem)
                return false;
            local->push_back({rem, m, 1});
            tile->push_back({e / rem, m, rem});
            rem = 1;
        }
    }
    return rem == 1;
}

/* k_split_padded: (local, tile, (mode, real, padded)) with the least padded K */
bool k_split_padded(const std::vector<std::string>& order,
                    const std::map<std::string, int64_t>& ext,
                    int64_t T,
                    std::vector<KDigit>* local,
                    std::vector<KDigit>* tile,
                    std::string* pmode,
                    int64_t* preal,
                    int64_t* ppad)
{
    bool have = false;
    int64_t best_cost = 0;
    for(size_t j = 0; j < order.size(); ++j)
    {
        const std::string& m = order[j];
        int64_t rem = T;
        for(size_t x = 0; x < j; ++x)
        {
            int64_t e = ext.at(order[x]);
            if(e > rem || rem % e)
            {
                rem = 0;
                break;
            }
            rem /= e;
        }
        if(!rem || ext.at(m) >= rem)
            continue;
        int64_t d = 0;
        for(int64_t v = ext.at(m) + 1; v <= rem; ++v)
            if(rem % v == 0)
            {
                d = v;
                break;
            }
        std::map<std::string, int64_t> e2 = ext;
        e2[m] = d;
        std::vector<KDigit> lo, ti;
        if(!k_split(order, e2, T, &lo, &ti))
            continue;
        bool found = false;
        for(const auto& kd : lo)
            found = found || (kd.m == m && kd.r == d);
        if(!found)
            continue;
        int64_t cost = 1;
        for(const auto& x : order)
            cost *= e2.at(x);
        if(!have || cost < best_cost)
        {
            have = true;
            best_cost = cost;
            *local = lo;
            *tile = ti;
            *pmode = m;
            *preal = ext.at(m);
            *ppad = d;
        }
    }
    return have;
}

int vec_of(const std::vector<Digit>& digits, const std::vector<int64_t>& other, int cap = 8)
{
    if(digits[0].s != 1)
        return 1;
    for(int v = cap; v > 1; v /= 2)
    {
        bool ok = digits[0].e % v == 0;
        for(size_t i = 1; ok && i < digits.size(); ++i)
            ok = digits[i].s % v == 0;
        for(size_t i = 0; ok && i < other.size(); ++i)
            ok = other[i] % v == 0;
        if(ok)
            return v;
    }
    return 1;
}

bool contains(const std::vector<std::string>& v, const std::string& x)
{
    return std::find(v.begin(), v.end(), x) != v.end();
}

std::string py_list_ext(const std::vector<std::string>& k_modes,
                        const std::map<std::string, int64_t>& ext)
{
    std::ostringstream o;
    o << "[";
    for(size_t i = 0; i < k_modes.size(); ++i)
        o << (i ? ", " : "") << "('" << k_modes[i] << "', " << ext.at(k_modes[i]) << ")";
    o << "]";
    return o.str();
}

} // namespace

int64_t Plan::klocal(int64_t kl, char role) const
{
    const auto& st = op(role).strides;
    int64_t off = 0;
    for(const auto& d : k_local)
    {
        off += (kl % d.r) * d.mult * st.at(d.m);
        kl /= d.r;
    }
    return off;
}

int64_t Plan::ktile(int64_t kt, char role) const
{
    const auto& st = op(role).strides;
    int64_t off = 0;
    for(const auto& d : k_tile)
    {
        off += (kt % d.r) * d.mult * st.at(d.m);
        kt /= d.r;
    }
    return off;
}

bool plan_gett(const std::vector<std::string>& modes_a,
               const std::vector<std::string>& modes_b,
               const std::vector<std::string>& modes_d,
               const std::map<std::string, int64_t>& ext,
               const std::map<std::string, int64_t>* strides_a,
               const std::map<std::string, int64_t>* strides_b,
               const std::map<std::string, int64_t>* strides_d,
               int tile_k,
               bool kc,
               char d_fast_role,
               Plan* out,
               std::string* why)
{
    auto fail = [&](const std::string& m) {
        if(why != nullptr)
            *why = m;
        return false;
    };
    if(d_fast_role != 'A' && d_fast_role != 'B')
        return fail(std::string("d_fast_role '") + d_fast_role + "' not in ('A', 'B')");
    auto sa
        = strides_a != nullptr && !strides_a->empty() ? *strides_a : packed_strides(modes_a, ext);
    auto sb
        = strides_b != nullptr && !strides_b->empty() ? *strides_b : packed_strides(modes_b, ext);
    auto sd
        = strides_d != nullptr && !strides_d->empty() ? *strides_d : packed_strides(modes_d, ext);
    std::vector<std::string> m_modes, n_modes, k_modes;
    for(const auto& x : modes_d)
        if(contains(modes_a, x) && !contains(modes_b, x))
            m_modes.push_back(x);
    for(const auto& x : modes_d)
        if(contains(modes_b, x) && !contains(modes_a, x))
            n_modes.push_back(x);
    for(const auto& x : modes_a)
        if(contains(modes_b, x) && !contains(modes_d, x))
            k_modes.push_back(x);
    {
        std::set<std::string> mn(m_modes.begin(), m_modes.end());
        mn.insert(n_modes.begin(), n_modes.end());
        std::set<std::string> dset(modes_d.begin(), modes_d.end());
        if(mn != dset || k_modes.empty() || m_modes.empty() || n_modes.empty())
            return fail("not a plain bilinear contraction (batch / Hadamard modes unsupported)");
    }
    /* min(modes_d, key=sd): the first of the smallest */
    std::string d_fast = modes_d[0];
    for(const auto& x : modes_d)
        if(sd.at(x) < sd.at(d_fast))
            d_fast = x;
    bool swapped = contains(n_modes, d_fast) == (d_fast_role == 'A');
    struct RoleSrc
    {
        std::string nm;
        const std::vector<std::string>* modes;
        const std::map<std::string, int64_t>* st;
        const std::vector<std::string>* free;
    };
    RoleSrc roles[2];
    if(swapped)
    {
        roles[0] = {"B", &modes_b, &sb, &n_modes};
        roles[1] = {"A", &modes_a, &sa, &m_modes};
    }
    else
    {
        roles[0] = {"A", &modes_a, &sa, &m_modes};
        roles[1] = {"B", &modes_b, &sb, &n_modes};
    }
    std::vector<int> tks = tile_k ? std::vector<int>{tile_k} : std::vector<int>{64, 32};
    bool have = false;
    int T = 0;
    std::vector<std::string> k_order;
    std::vector<KDigit> k_local, k_tile;
    if(kc)
    {
        k_order = k_modes;
        std::stable_sort(
            k_order.begin(), k_order.end(), [&](const std::string& x, const std::string& y) {
                return sa.at(x) < sa.at(y);
            });
        T = tks[0];
        have = true;
    }
    for(size_t ti = 0; !have && ti < tks.size(); ++ti)
    {
        struct Cand
        {
            std::vector<int64_t> key;
            std::vector<std::string> perm;
            std::vector<KDigit> lo, tl;
        };
        std::vector<Cand> cands;
        std::vector<int> idx(k_modes.size());
        for(size_t i = 0; i < idx.size(); ++i)
            idx[i] = (int)i;
        do
        {
            std::vector<std::string> perm;
            for(int i : idx)
                perm.push_back(k_modes[i]);
            Cand c;
            if(!k_split(perm, ext, tks[ti], &c.lo, &c.tl))
                continue;
            for(const auto& m : perm)
                c.key.push_back(std::min(sa.at(m), sb.at(m)));
            c.perm = perm;
            cands.push_back(c);
        } while(std::next_permutation(idx.begin(), idx.end()));
        if(!cands.empty())
        {
            std::stable_sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) {
                return x.key < y.key;
            });
            T = tks[ti];
            k_order = cands[0].perm;
            k_local = cands[0].lo;
            k_tile = cands[0].tl;
            have = true;
        }
    }
    bool has_kp = false;
    std::string kp_mode;
    int64_t kp_real = 0, kp_pad = 0;
    if(!have && !kc)
    {
        /* no exact split: pad one in-tile K mode (padded rows read zeros) */
        for(size_t ti = 0; !have && ti < tks.size(); ++ti)
        {
            struct Cand
            {
                std::vector<int64_t> key;
                std::vector<std::string> perm;
                std::vector<KDigit> lo, tl;
                std::string m;
                int64_t real, pad;
            };
            std::vector<Cand> cands;
            std::vector<int> idx(k_modes.size());
            for(size_t i = 0; i < idx.size(); ++i)
                idx[i] = (int)i;
            do
            {
                std::vector<std::string> perm;
                for(int i : idx)
                    perm.push_back(k_modes[i]);
                Cand c;
                if(!k_split_padded(perm, ext, tks[ti], &c.lo, &c.tl, &c.m, &c.real, &c.pad))
                    continue;
                int64_t kv = 1;
                for(const auto& m : k_modes)
                    kv *= m != c.m ? ext.at(m) : c.pad;
                c.key.push_back(kv);
                for(const auto& m : perm)
                    c.key.push_back(std::min(sa.at(m), sb.at(m)));
                c.perm = perm;
                cands.push_back(c);
            } while(std::next_permutation(idx.begin(), idx.end()));
            if(!cands.empty())
            {
                std::stable_sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) {
                    return x.key < y.key;
                });
                T = tks[ti];
                k_order = cands[0].perm;
                k_local = cands[0].lo;
                k_tile = cands[0].tl;
                has_kp = true;
                kp_mode = cands[0].m;
                kp_real = cands[0].real;
                kp_pad = cands[0].pad;
                have = true;
            }
        }
    }
    if(!have)
    {
        std::ostringstream o;
        o << "K modes " << py_list_ext(k_modes, ext) << " not separable for tiles [";
        for(size_t i = 0; i < tks.size(); ++i)
            o << (i ? ", " : "") << tks[i];
        o << "]";
        return fail(o.str());
    }
    Plan pl;
    for(int r = 0; r < 2; ++r)
    {
        const RoleSrc& rs = roles[r];
        OpGeom g;
        g.name = rs.nm;
        g.rows_order = *rs.free;
        std::stable_sort(g.rows_order.begin(),
                         g.rows_order.end(),
                         [&](const std::string& x, const std::string& y) {
                             return -rs.st->at(x) < -rs.st->at(y);
                         });
        g.row_digits.d = merge_digits(g.rows_order, ext, *rs.st);
        g.rows = 1;
        for(const auto& x : *rs.free)
            g.rows *= ext.at(x);
        std::vector<int64_t> other;
        for(const auto& m : k_modes)
            other.push_back(rs.st->at(m));
        int v = vec_of(g.row_digits.d, other);
        if(kc)
            v = 0;
        else if(v < 2)
        {
            std::ostringstream o;
            o << "operand " << rs.nm << " is not MN-contiguous (vec " << v
              << "); GETT needs the K-outer path";
            return fail(o.str());
        }
        if(!kc)
            v = v >= 8 ? 8 : 2;
        g.vec = v;
        g.strides = *rs.st;
        int64_t span = 0;
        for(const auto& m : *rs.modes)
            span += (ext.at(m) - 1) * rs.st->at(m);
        g.nbytes = 2 * (1 + span);
        pl.ops[r] = g;
    }
    std::vector<Digit> d_row = merge_digits(pl.ops[0].rows_order, ext, sd);
    std::vector<Digit> d_col = merge_digits(pl.ops[1].rows_order, ext, sd);
    int vd = 4;
    while(vd > 1)
    {
        bool ok = d_row[0].s == 1 && d_row[0].e % vd == 0;
        for(size_t i = 1; ok && i < d_row.size(); ++i)
            ok = d_row[i].s % vd == 0;
        for(size_t i = 0; ok && i < d_col.size(); ++i)
            ok = d_col[i].s % vd == 0;
        if(ok)
            break;
        vd /= 2;
    }
    auto all8 = [](const std::vector<Digit>& fast, const std::vector<Digit>& other) {
        bool ok = fast[0].s == 1 && fast[0].e % 8 == 0;
        for(size_t i = 1; ok && i < fast.size(); ++i)
            ok = fast[i].s % 8 == 0;
        for(size_t i = 0; ok && i < other.size(); ++i)
            ok = other[i].s % 8 == 0;
        return ok;
    };
    int64_t K = 1;
    for(const auto& m : k_modes)
        K *= (!has_kp || m != kp_mode) ? ext.at(m) : kp_pad;
    pl.modes_a = modes_a;
    pl.modes_b = modes_b;
    pl.modes_d = modes_d;
    pl.ext = ext;
    pl.swapped = swapped;
    pl.M = pl.ops[0].rows;
    pl.N = pl.ops[1].rows;
    pl.K = K;
    pl.tile_k = T;
    pl.k_order = k_order;
    pl.k_local = k_local;
    pl.k_tile = k_tile;
    pl.d_row_digits.d = d_row;
    pl.d_col_digits.d = d_col;
    pl.vec_d = vd;
    pl.strides_d = sd;
    pl.vec_d8 = all8(d_row, d_col);
    pl.has_k_pad = has_kp;
    pl.k_pad_mode = kp_mode;
    pl.k_pad_real = kp_real;
    pl.k_pad_padded = kp_pad;
    pl.kc = kc;
    pl.vec_dc8 = all8(d_col, d_row);
    *out = pl;
    return true;
}

} // namespace gett
} // namespace rocke
