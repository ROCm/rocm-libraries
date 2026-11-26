// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host reference of the target model's top-k router and of the route table the
// router phase writes. Plain C++ (no HIP), so router_reference_test.cpp can build it with g++.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace router_ref {

inline float bf16_to_float(uint16_t b)
{
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

struct Routing
{
    int M = 0, E = 0, top_k = 0;
    std::vector<double> logits; // [M, E], scaled
    std::vector<double> s;      // sigmoid
    std::vector<double> choice; // s + bias
    std::vector<int> ids;       // [M, top_k], selection order
    std::vector<double> weights;
};

/// Top-k of one row of choice: larger choice first, the smaller expert on equal choice (the tie
/// rule of RouterFinalizePhase).
inline std::vector<int> topk_row(const double* choice, int E, int k)
{
    std::vector<int> idx(E);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return choice[a] > choice[b]; });
    idx.resize(k);
    return idx;
}

/// Sigmoid, bias, top-k and normalisation from given (scaled) logits, in double.
inline Routing route_from_logits(const std::vector<double>& logits,
                                 const std::vector<float>& bias,
                                 int M,
                                 int E,
                                 int top_k,
                                 double routed_scaling_factor = 1.0,
                                 double eps                   = 1e-15)
{
    Routing r;
    r.M      = M;
    r.E      = E;
    r.top_k  = top_k;
    r.logits = logits;
    r.s.resize(size_t(M) * E);
    r.choice.resize(size_t(M) * E);
    r.ids.resize(size_t(M) * top_k);
    r.weights.resize(size_t(M) * top_k);
    for(int m = 0; m < M; ++m)
    {
        for(int e = 0; e < E; ++e)
        {
            const double s              = 1.0 / (1.0 + std::exp(-logits[size_t(m) * E + e]));
            r.s[size_t(m) * E + e]      = s;
            r.choice[size_t(m) * E + e] = s + bias[e];
        }
        const auto idx = topk_row(&r.choice[size_t(m) * E], E, top_k);
        double sum     = 0;
        for(int k = 0; k < top_k; ++k)
            sum += r.s[size_t(m) * E + idx[k]];
        for(int k = 0; k < top_k; ++k)
        {
            r.ids[size_t(m) * top_k + k] = idx[k];
            r.weights[size_t(m) * top_k + k] =
                r.s[size_t(m) * E + idx[k]] / (sum + eps) * routed_scaling_factor;
        }
    }
    return r;
}

/// logits = (h fp32) @ w^T * scale in double; h bf16 bits [M, K], w fp32 [E, K].
/// mag = sum |h * w| * scale per entry, for error bounds.
struct Logits
{
    std::vector<double> logits; // [M, E]
    std::vector<double> mag;    // [M, E]
};

inline Logits router_logits(
    const std::vector<uint16_t>& h, const std::vector<float>& w, int M, int K, int E, double scale)
{
    Logits out;
    out.logits.assign(size_t(M) * E, 0.0);
    out.mag.assign(size_t(M) * E, 0.0);
    std::vector<double> hrow(K);
    for(int m = 0; m < M; ++m)
    {
        for(int k = 0; k < K; ++k)
            hrow[k] = bf16_to_float(h[size_t(m) * K + k]);
        for(int e = 0; e < E; ++e)
        {
            double acc = 0, a = 0;
            const float* wr = &w[size_t(e) * K];
            for(int k = 0; k < K; ++k)
            {
                const double p = hrow[k] * wr[k];
                acc += p;
                a += std::fabs(p);
            }
            out.logits[size_t(m) * E + e] = acc * scale;
            out.mag[size_t(m) * E + e]    = a * scale;
        }
    }
    return out;
}

/// The moe_route_table fields, built on the host from topk ids [M, top_k].
struct RouteTable
{
    std::vector<int> count;  // [E]
    std::vector<int> routes; // [E, max_m], only the first count[e] entries are defined
    std::vector<int> active; // [E]: active experts ascending, then idle experts ascending
    int num_active = 0;
};

inline RouteTable build_route_table(const std::vector<int>& ids, int M, int top_k, int E, int max_m)
{
    RouteTable t;
    t.count.assign(E, 0);
    t.routes.assign(size_t(E) * max_m, -1);
    for(int route = 0; route < M * top_k; ++route) // ascending routes
    {
        const int e                              = ids[route];
        t.routes[size_t(e) * max_m + t.count[e]] = route;
        ++t.count[e];
    }
    for(int e = 0; e < E; ++e)
        if(t.count[e] > 0)
            t.active.push_back(e);
    t.num_active = static_cast<int>(t.active.size());
    for(int e = 0; e < E; ++e)
        if(t.count[e] == 0)
            t.active.push_back(e);
    return t;
}

/// Checks a device top-k against the reference. A set difference is accepted when every expert
/// in it is within tol of the reference's k-th choice (a near-tie that fp32 may order
/// differently). Returns the number of bad rows; *near_ties gets the accepted differences.
inline int compare_topk(const Routing& ref,
                        const std::vector<int>& ids,
                        const std::vector<float>& weights,
                        double choice_tol,
                        double weight_tol,
                        int* near_ties = nullptr)
{
    int bad = 0, ties = 0;
    const int E = ref.E, k = ref.top_k;
    for(int m = 0; m < ref.M; ++m)
    {
        const double* c = &ref.choice[size_t(m) * E];
        std::vector<int> want(&ref.ids[size_t(m) * k], &ref.ids[size_t(m) * k] + k);
        std::vector<int> got(&ids[size_t(m) * k], &ids[size_t(m) * k] + k);
        const double kth = c[want.back()];
        bool ok          = true;
        for(int e : got)
            ok = ok && e >= 0 && e < E;
        std::vector<int> gs = got, ws = want;
        std::sort(gs.begin(), gs.end());
        std::sort(ws.begin(), ws.end());
        ok = ok && std::adjacent_find(gs.begin(), gs.end()) == gs.end();
        if(ok && gs != ws)
        {
            for(int e : gs)
                if(!std::binary_search(ws.begin(), ws.end(), e))
                    ok = ok && std::fabs(c[e] - kth) <= choice_tol;
            for(int e : ws)
                if(!std::binary_search(gs.begin(), gs.end(), e))
                    ok = ok && std::fabs(c[e] - kth) <= choice_tol;
            ties += ok ? 1 : 0;
        }
        if(ok)
        {
            // Weights of the experts the device picked, normalised in double.
            double sum = 0;
            for(int e : got)
                sum += ref.s[size_t(m) * E + e];
            for(int j = 0; j < k; ++j)
            {
                const double w = ref.s[size_t(m) * E + got[j]] / sum;
                ok             = ok && std::fabs(weights[size_t(m) * k + j] - w) <= weight_tol;
            }
        }
        bad += ok ? 0 : 1;
    }
    if(near_ties)
        *near_ties = ties;
    return bad;
}

} // namespace router_ref
