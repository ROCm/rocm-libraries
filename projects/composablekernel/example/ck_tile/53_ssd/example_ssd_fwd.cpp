// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Mamba-2 SSD forward — ck_tile example driver.
//
// Usage:
//   ./tile_example_ssd_fwd -B=2 -E=2 -H=2 -v=1

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

#include "ssd_problem.hpp"
#include "ssd_fwd.hpp"

// ---- CPU reference (self-contained, matches ssd_amd/ssd_cpu_reference.h) ---

static void ssd_forward_cpu_ref(float* Y,
                                float* Fstate,
                                const float* X,
                                const float* DeltaA,
                                const float* Delta,
                                const float* B_mat,
                                const float* C_mat,
                                const float* D_param,
                                const float* Z,
                                int B,
                                int G,
                                int EH,
                                int C,
                                int L,
                                int D,
                                int N)
{
    auto idx5 = [](int a, int b, int c, int d, int e, int B_, int C_, int D_, int E_) -> size_t {
        return static_cast<size_t>(((a * B_ + b) * C_ + c) * D_ + d) * E_ + e;
    };
    auto idx4 = [](int a, int b, int c, int d, int B_, int C_, int D_) -> size_t {
        return static_cast<size_t>((a * B_ + b) * C_ + c) * D_ + d;
    };
    auto idx3 = [](int a, int b, int c, int B_, int C_) -> size_t {
        return static_cast<size_t>(a * B_ + b) * C_ + c;
    };
    auto idx2 = [](int a, int b, int B_) -> size_t { return static_cast<size_t>(a) * B_ + b; };

    int gr = EH / G;

    std::vector<float> intra_bmm1(L * L);
    std::vector<float> segsum_mat(L * L);
    std::vector<float> pre_intra2(L * L);
    std::vector<float> intra_bmm2(L * D);
    std::vector<float> cum(L);
    std::vector<float> cum_exp_last(L);
    std::vector<float> cum_exp(L);
    std::vector<float> pre_inter1(N * L);
    std::vector<float> inter_bmm1(C * N * D, 0.0f);
    std::vector<float> state(C * N * D, 0.0f);
    std::vector<float> inter_bmm2(L * D);
    std::vector<float> last_per_chunk(C);

    for(int bi = 0; bi < B; ++bi)
    {
        for(int eh = 0; eh < EH; ++eh)
        {
            int g = eh / gr;
            std::fill(inter_bmm1.begin(), inter_bmm1.end(), 0.0f);
            std::fill(state.begin(), state.end(), 0.0f);

            for(int ci = 0; ci < C; ++ci)
            {
                // cumsum
                cum[0] = DeltaA[idx4(bi, eh, ci, 0, EH, C, L)];
                for(int l = 1; l < L; ++l)
                    cum[l] = cum[l - 1] + DeltaA[idx4(bi, eh, ci, l, EH, C, L)];
                float last         = cum[L - 1];
                last_per_chunk[ci] = last;

                // segsum
                for(int i = 0; i < L; ++i)
                    for(int j = 0; j < L; ++j)
                    {
                        if(j < i)
                            segsum_mat[i * L + j] = expf(cum[i] - cum[j]);
                        else if(j == i)
                            segsum_mat[i * L + j] = 1.0f;
                        else
                            segsum_mat[i * L + j] = 0.0f;
                    }

                // IntraBMM1
                std::fill(intra_bmm1.begin(), intra_bmm1.end(), 0.0f);
                for(int l1 = 0; l1 < L; ++l1)
                    for(int l2 = 0; l2 < L; ++l2)
                    {
                        float acc = 0.0f;
                        for(int n = 0; n < N; ++n)
                            acc += C_mat[idx5(bi, g, n, ci, l1, G, N, C, L)] *
                                   B_mat[idx5(bi, g, n, ci, l2, G, N, C, L)];
                        intra_bmm1[l1 * L + l2] = acc;
                    }

                // pre_intra2
                for(int i = 0; i < L; ++i)
                    for(int j = 0; j < L; ++j)
                        pre_intra2[i * L + j] = segsum_mat[i * L + j] *
                                                Delta[idx4(bi, eh, ci, j, EH, C, L)] *
                                                intra_bmm1[i * L + j];

                // IntraBMM2
                std::fill(intra_bmm2.begin(), intra_bmm2.end(), 0.0f);
                for(int l = 0; l < L; ++l)
                    for(int d_ = 0; d_ < D; ++d_)
                    {
                        float acc = 0.0f;
                        for(int lp = 0; lp < L; ++lp)
                            acc +=
                                pre_intra2[l * L + lp] * X[idx5(bi, eh, d_, ci, lp, EH, D, C, L)];
                        intra_bmm2[l * D + d_] = acc;
                    }

                // cumsum exp
                for(int l = 0; l < L; ++l)
                {
                    cum_exp_last[l] = expf(last - cum[l]);
                    cum_exp[l]      = expf(cum[l]);
                }

                // pre_inter1
                for(int n = 0; n < N; ++n)
                    for(int l = 0; l < L; ++l)
                        pre_inter1[n * L + l] = cum_exp_last[l] *
                                                Delta[idx4(bi, eh, ci, l, EH, C, L)] *
                                                B_mat[idx5(bi, g, n, ci, l, G, N, C, L)];

                // InterBMM1
                for(int n = 0; n < N; ++n)
                    for(int d_ = 0; d_ < D; ++d_)
                    {
                        float acc = 0.0f;
                        for(int l = 0; l < L; ++l)
                            acc += pre_inter1[n * L + l] * X[idx5(bi, eh, d_, ci, l, EH, D, C, L)];
                        inter_bmm1[idx3(ci, n, d_, N, D)] = acc;
                    }

                // state propagation
                if(ci == 0)
                {
                    for(int n = 0; n < N; ++n)
                        for(int d_ = 0; d_ < D; ++d_)
                            state[idx3(ci, n, d_, N, D)] = 0.0f;
                }
                else
                {
                    float el = expf(last_per_chunk[ci - 1]);
                    for(int n = 0; n < N; ++n)
                        for(int d_ = 0; d_ < D; ++d_)
                            state[idx3(ci, n, d_, N, D)] = inter_bmm1[idx3(ci - 1, n, d_, N, D)] +
                                                           el * state[idx3(ci - 1, n, d_, N, D)];
                }

                // InterBMM2
                std::fill(inter_bmm2.begin(), inter_bmm2.end(), 0.0f);
                for(int l = 0; l < L; ++l)
                    for(int d_ = 0; d_ < D; ++d_)
                    {
                        float acc = 0.0f;
                        for(int n = 0; n < N; ++n)
                            acc += C_mat[idx5(bi, g, n, ci, l, G, N, C, L)] *
                                   state[idx3(ci, n, d_, N, D)];
                        inter_bmm2[l * D + d_] = acc;
                    }

                // epilogue
                for(int l = 0; l < L; ++l)
                    for(int d_ = 0; d_ < D; ++d_)
                    {
                        float y = cum_exp[l] * inter_bmm2[l * D + d_] + intra_bmm2[l * D + d_];
                        y += D_param[idx2(eh, d_, D)] * X[idx5(bi, eh, d_, ci, l, EH, D, C, L)];
                        if(Z != nullptr)
                        {
                            float zv = Z[idx5(bi, eh, d_, ci, l, EH, D, C, L)];
                            y *= zv / (1.0f + expf(-zv)); // Y * silu(Z)
                        }
                        Y[idx5(bi, eh, d_, ci, l, EH, D, C, L)] = y;
                    }
            }

            // final state
            float el = expf(last_per_chunk[C - 1]);
            for(int d_ = 0; d_ < D; ++d_)
                for(int n = 0; n < N; ++n)
                    Fstate[idx4(bi, eh, d_, n, EH, D, N)] =
                        inter_bmm1[idx3(C - 1, n, d_, N, D)] + el * state[idx3(C - 1, n, d_, N, D)];
        }
    }
}

// ---- Argument parsing (ck_tile style) ----
auto create_ssd_args(int argc, char* argv[])
{
    ck_tile::ArgParser ap;
    ap.insert("B", "2", "Batch size")
        .insert("G", "1", "Groups (must be 1)")
        .insert("E", "2", "Expansion factor")
        .insert("H", "2", "Heads per group")
        .insert("Z", "0", "0=no Z gating, 1=enable HAS_Z (Y*silu(Z))")
        .insert("v", "1", "0=no verify, 1=verify on CPU")
        .insert("warmup", "3", "Warmup iterations")
        .insert("repeat", "10", "Benchmark iterations");
    bool ok = ap.parse(argc, argv);
    return std::make_tuple(ok, ap);
}

// ---- Main ----
int main(int argc, char* argv[])
{
    auto [ok, ap] = create_ssd_args(argc, argv);
    if(!ok)
        return -1;

    const int B  = ap.get_int("B");
    const int G  = ap.get_int("G");
    const int E  = ap.get_int("E");
    const int H  = ap.get_int("H");
    const int EH = E * H;
    const int C = 8, L = 128, D = 64, N = 128;
    const int has_z     = ap.get_int("Z");
    const int n_warmup  = ap.get_int("warmup");
    const int n_repeat  = ap.get_int("repeat");
    const int do_verify = ap.get_int("v");

    std::cout << "=== Mamba-2 SSD ck_tile Example ===" << std::endl;
    std::cout << "B=" << B << " G=" << G << " E=" << E << " H=" << H << " EH=" << EH << " C=" << C
              << " L=" << L << " D=" << D << " N=" << N << " HAS_Z=" << has_z << std::endl;

    // Sizes
    size_t sz_x  = static_cast<size_t>(B) * EH * D * C * L;
    size_t sz_da = static_cast<size_t>(B) * EH * C * L;
    size_t sz_b  = static_cast<size_t>(B) * G * N * C * L;
    size_t sz_y  = static_cast<size_t>(B) * EH * D * C * L;
    size_t sz_f  = static_cast<size_t>(B) * EH * D * N;
    size_t sz_dp = static_cast<size_t>(EH) * D;

    // Host tensors
    std::vector<float> h_x(sz_x), h_da(sz_da), h_delta(sz_da);
    std::vector<float> h_bm(sz_b), h_cm(sz_b), h_dp(sz_dp);
    std::vector<float> h_z;
    std::vector<float> h_y_gpu(sz_y, 0.0f), h_f_gpu(sz_f, 0.0f);

    // Init random (same seed as ssd_amd for comparison)
    std::mt19937 rng(2024);
    auto fill_u = [&](float* p, size_t n, float lo, float hi) {
        std::uniform_real_distribution<float> d(lo, hi);
        for(size_t i = 0; i < n; ++i)
            p[i] = d(rng);
    };
    auto fill_g = [&](float* p, size_t n, float mean, float std) {
        std::normal_distribution<float> d(mean, std);
        for(size_t i = 0; i < n; ++i)
            p[i] = d(rng);
    };

    fill_u(h_x.data(), sz_x, -2.0f, 2.0f);
    fill_g(h_da.data(), sz_da, 0.0f, 0.05f);
    fill_g(h_delta.data(), sz_da, 0.0f, 0.05f);
    fill_u(h_bm.data(), sz_b, -2.0f, 2.0f);
    fill_u(h_cm.data(), sz_b, -2.0f, 2.0f);
    fill_u(h_dp.data(), sz_dp, -2.0f, 2.0f);
    if(has_z)
    {
        h_z.resize(sz_x);
        fill_u(h_z.data(), sz_x, -2.0f, 2.0f);
    }

    // Device memory
    ck_tile::DeviceMem d_x(sz_x * sizeof(float));
    ck_tile::DeviceMem d_da(sz_da * sizeof(float));
    ck_tile::DeviceMem d_delta(sz_da * sizeof(float));
    ck_tile::DeviceMem d_bm(sz_b * sizeof(float));
    ck_tile::DeviceMem d_cm(sz_b * sizeof(float));
    ck_tile::DeviceMem d_dp(sz_dp * sizeof(float));
    ck_tile::DeviceMem d_z(has_z ? sz_x * sizeof(float) : 0);
    ck_tile::DeviceMem d_y(sz_y * sizeof(float));
    ck_tile::DeviceMem d_f(sz_f * sizeof(float));

    d_x.ToDevice(h_x.data());
    d_da.ToDevice(h_da.data());
    d_delta.ToDevice(h_delta.data());
    d_bm.ToDevice(h_bm.data());
    d_cm.ToDevice(h_cm.data());
    d_dp.ToDevice(h_dp.data());
    if(has_z)
        d_z.ToDevice(h_z.data());
    d_y.SetZero();
    d_f.SetZero();

    // Build host args
    ck_tile::SsdHostArgs ssd_args{d_x.GetDeviceBuffer(),
                                  d_da.GetDeviceBuffer(),
                                  d_delta.GetDeviceBuffer(),
                                  d_bm.GetDeviceBuffer(),
                                  d_cm.GetDeviceBuffer(),
                                  d_dp.GetDeviceBuffer(),
                                  has_z ? d_z.GetDeviceBuffer() : nullptr,
                                  d_y.GetDeviceBuffer(),
                                  d_f.GetDeviceBuffer(),
                                  B,
                                  G,
                                  EH,
                                  C,
                                  L,
                                  D,
                                  N};

    // Warmup
    for(int i = 0; i < n_warmup; ++i)
        ck_tile::ssd_fwd(ssd_args);

    // Benchmark
    hipEvent_t ev_start, ev_stop;
    (void)hipEventCreate(&ev_start);
    (void)hipEventCreate(&ev_stop);
    (void)hipEventRecord(ev_start);

    for(int i = 0; i < n_repeat; ++i)
        ck_tile::ssd_fwd(ssd_args);

    (void)hipEventRecord(ev_stop);
    (void)hipEventSynchronize(ev_stop);
    float ms = 0;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
    ms /= n_repeat;

    std::cout << "GPU time: " << ms << " ms (avg over " << n_repeat << " iters)" << std::endl;

    // Download
    d_y.FromDevice(h_y_gpu.data());
    d_f.FromDevice(h_f_gpu.data());

    // Verify
    if(do_verify)
    {
        std::cout << "\nRunning CPU reference..." << std::endl;
        std::vector<float> h_y_cpu(sz_y, 0.0f), h_f_cpu(sz_f, 0.0f);

        auto t0 = std::chrono::high_resolution_clock::now();
        ssd_forward_cpu_ref(h_y_cpu.data(),
                            h_f_cpu.data(),
                            h_x.data(),
                            h_da.data(),
                            h_delta.data(),
                            h_bm.data(),
                            h_cm.data(),
                            h_dp.data(),
                            has_z ? h_z.data() : nullptr,
                            B,
                            G,
                            EH,
                            C,
                            L,
                            D,
                            N);
        auto t1       = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "CPU reference time: " << cpu_ms << " ms" << std::endl;

        // Check Y
        int y_mismatch = 0;
        for(size_t i = 0; i < sz_y; ++i)
        {
            float r = h_y_cpu[i], t = h_y_gpu[i];
            float ae = std::fabs(t - r);
            float re = ae / (std::max(std::fabs(t), std::fabs(r)) + 1e-5f);
            if(std::isnan(ae) || std::isnan(re) || std::min(re, ae) > 0.05f)
            {
                if(y_mismatch < 5)
                    printf("  [Y] mismatch [%zu]: ref=%.6f got=%.6f\n", i, r, t);
                y_mismatch++;
            }
        }
        // Check F
        int f_mismatch = 0;
        for(size_t i = 0; i < sz_f; ++i)
        {
            float r = h_f_cpu[i], t = h_f_gpu[i];
            float ae = std::fabs(t - r);
            float re = ae / (std::max(std::fabs(t), std::fabs(r)) + 1e-5f);
            if(std::isnan(ae) || std::isnan(re) || std::min(re, ae) > 0.05f)
            {
                if(f_mismatch < 5)
                    printf("  [Fstate] mismatch [%zu]: ref=%.6f got=%.6f\n", i, r, t);
                f_mismatch++;
            }
        }

        bool pass = (y_mismatch == 0 && f_mismatch == 0);
        std::cout << "[Y]      " << (y_mismatch == 0 ? "PASSED" : "FAILED") << " (" << y_mismatch
                  << "/" << sz_y << " mismatches)" << std::endl;
        std::cout << "[Fstate] " << (f_mismatch == 0 ? "PASSED" : "FAILED") << " (" << f_mismatch
                  << "/" << sz_f << " mismatches)" << std::endl;
        std::cout << "\n" << (pass ? "ALL PASSED" : "VERIFICATION FAILED") << std::endl;

        (void)hipEventDestroy(ev_start);
        (void)hipEventDestroy(ev_stop);
        return pass ? 0 : 1;
    }

    (void)hipEventDestroy(ev_start);
    (void)hipEventDestroy(ev_stop);
    return 0;
}
