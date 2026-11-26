// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only checks of the router reference and of an fp32 emulation of RouterFinalizePhase (its
// key order, tie rule, normalisation and route-table construction). Needs no GPU:
//   g++ -std=c++17 -O2 router_reference_test.cpp -o router_reference_test

#include <cstdio>
#include <random>

#include "router_reference.hpp"

namespace {

constexpr int kE    = 256;
constexpr int kTopK = 8;

// RouterFinalizePhase's key: larger choice first, smaller expert on bitwise-equal choice.
uint64_t key_of(float c, int e)
{
    uint32_t b;
    std::memcpy(&b, &c, 4);
    const uint32_t o = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return (uint64_t(o) << 32) | (0xffffffffu - uint32_t(e));
}

// fp32 emulation of the device finalize: per row kTopK arg-max rounds over the keys, fp32 sum
// in slot order, then the table from a slot[e][row] array scanned by expert.
struct DeviceEmu
{
    std::vector<int> ids;
    std::vector<float> weights;
    router_ref::RouteTable table;
};

DeviceEmu
emulate(const std::vector<float>& logits, const std::vector<float>& bias, int M, int max_m)
{
    DeviceEmu d;
    d.ids.resize(size_t(M) * kTopK);
    d.weights.resize(size_t(M) * kTopK);
    std::vector<uint8_t> slot(size_t(kE) * 64, 0xff);
    for(int m = 0; m < M; ++m)
    {
        float s[kE];
        uint64_t key[kE];
        for(int e = 0; e < kE; ++e)
        {
            s[e]   = 1.f / (1.f + std::exp(-logits[size_t(m) * kE + e]));
            key[e] = key_of(s[e] + bias[e], e);
        }
        float sum = 0.f;
        float se[kTopK];
        for(int k = 0; k < kTopK; ++k)
        {
            uint64_t best = 0;
            for(int e = 0; e < kE; ++e)
                best = key[e] > best ? key[e] : best;
            const int e = int(0xffffffffu - uint32_t(best));
            key[e]      = 0;
            se[k]       = s[e];
            sum += s[e];
            d.ids[size_t(m) * kTopK + k] = e;
            slot[size_t(e) * 64 + m]     = uint8_t(k);
        }
        for(int k = 0; k < kTopK; ++k)
            d.weights[size_t(m) * kTopK + k] = se[k] / (sum + 1e-15f) * 1.f;
    }
    auto& t = d.table;
    t.count.assign(kE, 0);
    t.routes.assign(size_t(kE) * max_m, -1);
    for(int e = 0; e < kE; ++e)
        for(int m = 0; m < M; ++m)
            if(slot[size_t(e) * 64 + m] != 0xff)
                t.routes[size_t(e) * max_m + t.count[e]++] = m * kTopK + slot[size_t(e) * 64 + m];
    t.active.assign(kE, -1);
    int pos = 0;
    for(int e = 0; e < kE; ++e)
        pos += t.count[e] > 0;
    t.num_active = pos;
    int a        = 0;
    for(int e = 0; e < kE; ++e)
    {
        if(t.count[e] > 0)
            t.active[a++] = e;
        else
            t.active[pos + e - a] = e;
    }
    return d;
}

bool same_table(const router_ref::RouteTable& x, const router_ref::RouteTable& y, int max_m)
{
    if(x.count != y.count || x.active != y.active || x.num_active != y.num_active)
        return false;
    for(int e = 0; e < kE; ++e)
        for(int i = 0; i < x.count[e]; ++i)
            if(x.routes[size_t(e) * max_m + i] != y.routes[size_t(e) * max_m + i])
                return false;
    return true;
}

int g_fail = 0;
void check(bool ok, const char* what)
{
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    g_fail += ok ? 0 : 1;
}

} // namespace

int main()
{
    // 1. All logits and biases equal: the lowest 8 experts, slots ascending, weights 1/8.
    {
        const int M = 3;
        std::vector<float> lf(size_t(M) * kE, 0.25f), bias(kE, 0.f);
        std::vector<double> ld(lf.begin(), lf.end());
        const auto r = router_ref::route_from_logits(ld, bias, M, kE, kTopK);
        const auto d = emulate(lf, bias, M, 64);
        bool ok      = true;
        for(int m = 0; m < M; ++m)
            for(int k = 0; k < kTopK; ++k)
                ok = ok && r.ids[m * kTopK + k] == k && d.ids[m * kTopK + k] == k &&
                     std::fabs(d.weights[m * kTopK + k] - 0.125f) < 1e-7f;
        check(ok, "all-equal row picks experts 0..7 in order with weight 1/8");
    }
    // 2. Tie at the cut: 7 clear winners, then {200, 50, 5} equal: expert 5 takes slot 8.
    {
        const int M = 1;
        std::vector<float> lf(kE, -4.f), bias(kE, 0.f);
        for(int e : {3, 9, 100, 101, 150, 254, 255})
            bias[e] = 1.f;
        for(int e : {200, 50, 5})
            lf[e] = 2.f;
        std::vector<double> ld(lf.begin(), lf.end());
        const auto r = router_ref::route_from_logits(ld, bias, M, kE, kTopK);
        const auto d = emulate(lf, bias, M, 64);
        check(r.ids[7] == 5 && d.ids[7] == 5 && r.ids == d.ids,
              "tie at the cut goes to the smaller expert (reference and emulation)");
    }
    // 3. Random rows: emulation (fp32) vs reference (double), and the route tables.
    {
        std::mt19937_64 rng(7);
        std::normal_distribution<float> nd(0.f, 1.f);
        for(int M : {1, 8, 16, 37, 64})
        {
            std::vector<float> lf(size_t(M) * kE), bias(kE);
            for(auto& x : lf)
                x = 2.f * nd(rng);
            for(auto& x : bias)
                x = 0.05f * nd(rng);
            // Exact duplicates in some rows, so ties are exercised.
            for(int m = 0; m < M; m += 3)
                lf[size_t(m) * kE + 17] = lf[size_t(m) * kE + 230];
            bias[17] = bias[230];
            std::vector<double> ld(lf.begin(), lf.end());
            const auto r  = router_ref::route_from_logits(ld, bias, M, kE, kTopK);
            const auto d  = emulate(lf, bias, M, 64);
            int ties      = 0;
            const int bad = router_ref::compare_topk(r, d.ids, d.weights, 1e-6, 1e-6, &ties);
            const auto t  = router_ref::build_route_table(d.ids, M, kTopK, kE, 64);
            int routes    = 0;
            for(int c : t.count)
                routes += c;
            char msg[160];
            std::snprintf(msg,
                          sizeof(msg),
                          "M=%d: fp32 emulation vs double reference (%d near-ties), table "
                          "(%d active, %d routes)",
                          M,
                          ties,
                          t.num_active,
                          routes);
            check(bad == 0 && same_table(t, d.table, 64) && routes == M * kTopK, msg);
        }
    }
    std::printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
