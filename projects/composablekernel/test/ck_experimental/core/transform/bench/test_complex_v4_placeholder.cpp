// Workload C (complex) -- V4 experimental, placeholder.
// Mirrors test_complex_v3_placeholder.cpp exactly.
//
// Buffer layout (14 ints, one per placeholder):
//   [0] d0,  [1] d1,  [2] d2,  [3] d3,  [4] d4,  [5] d5,
//   [6] s0,  [7] s1,  [8] s2,  [9] s3,  [10] s4,
//   [11] M0_div_mpair, [12] USER_M, [13] USER_K

#include "ck_experimental/core/transform/v4_experimental.hpp"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace {
using ck_tile::index_t;
using ck_tile::static_array;
namespace v4 = ck_tile::core::transform::v4;

constexpr auto make_complex_v4_runtime()
{
    using namespace v4;
    constexpr placeholder<0>  d0_p{};
    constexpr placeholder<1>  d1_p{};
    constexpr placeholder<2>  d2_p{};
    constexpr placeholder<3>  d3_p{};
    constexpr placeholder<4>  d4_p{};
    constexpr placeholder<5>  d5_p{};
    constexpr placeholder<6>  s0_p{};
    constexpr placeholder<7>  s1_p{};
    constexpr placeholder<8>  s2_p{};
    constexpr placeholder<9>  s3_p{};
    constexpr placeholder<10> s4_p{};
    constexpr placeholder<11> M0_div_mpair_p{};
    constexpr placeholder<12> USER_M_p{};
    constexpr placeholder<13> USER_K_p{};

    constexpr index_t KTRPerm  = 4;
    constexpr index_t M1       = 4;
    constexpr index_t kfold    = 2;
    constexpr index_t mpair    = 2;
    constexpr index_t AK1      = 1;
    constexpr index_t K0PerTW  = 1;

    constexpr index_t PH0=0, PH1=1, PH2=2, PH3=3, PH4=4, PH5=5;
    constexpr index_t OFF=6, XR0=7, XR1=8;
    constexpr index_t UM_KTRP=9, UM_M1=10, UM_KFOLD=11, UM_M0MP=12;
    constexpr index_t U_M=13, U_K=14;

    return make_transform_graph(
        outputs(read(OFF)),
        make_embed(dims(d0_p, d1_p, d2_p, d3_p, d4_p, d5_p),
                             strides(s0_p, s1_p, s2_p, s3_p, s4_p, 1), read(PH0, PH1, PH2, PH3, PH4, PH5), write(OFF)),
        make_xor(read(XR0, XR1), write(PH2, PH3)),
        make_unmerge(dims(KTRPerm, M1), read(UM_KTRP, UM_M1), write(XR0)),
        make_unmerge(dims(kfold, M0_div_mpair_p), read(UM_KFOLD, UM_M0MP), write(XR1)),
        make_merge(dims(KTRPerm, d0_p, kfold, K0PerTW, AK1), read(U_K), write(UM_KTRP, PH0, UM_KFOLD, PH1, PH5)),
        make_merge(dims(M0_div_mpair_p, mpair, M1), read(U_M), write(UM_M0MP, PH4, UM_M1)),
        inputs(dims(USER_M_p, USER_K_p), write(U_M, U_K)));
}

__global__ void test_kernel(const index_t* m_in, const index_t* k_in,
                             index_t* out, const index_t* runtime_args,
                             const index_t* n_iters_ptr)
{
    constexpr auto g = make_complex_v4_runtime();
    const auto gb = v4::make_graph_bindings<g>(
        runtime_args[0],  runtime_args[1],  runtime_args[2],  runtime_args[3],
        runtime_args[4],  runtime_args[5],  runtime_args[6],  runtime_args[7],
        runtime_args[8],  runtime_args[9],  runtime_args[10], runtime_args[11],
        runtime_args[12], runtime_args[13]);

    const index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t m_base  = m_in[tid];
    const index_t k_base  = k_in[tid];
    const index_t n_iters = *n_iters_ptr;   // runtime -- opaque

    index_t s = 0;
    for(index_t i = 0; i < n_iters; ++i)
    {
        const index_t m = m_base + (i & 0xff);
        const index_t k = k_base + ((i >> 4) & 0xff);

        s += v4::calculateOffset<g>(static_array<index_t, 2>{m,     k},     gb);
        s += v4::calculateOffset<g>(static_array<index_t, 2>{m + 1, k},     gb);
        s += v4::calculateOffset<g>(static_array<index_t, 2>{m + 2, k},     gb);
        s += v4::calculateOffset<g>(static_array<index_t, 2>{m + 3, k},     gb);
        s += v4::calculateOffset<g>(static_array<index_t, 2>{m + 4, k},     gb);
        s += v4::calculateOffset<g>(static_array<index_t, 2>{m,     k + 1}, gb);
        s += v4::calculateOffset<g>(static_array<index_t, 2>{m,     k + 2}, gb);
        s += v4::calculateOffset<g>(static_array<index_t, 2>{m,     k + 3}, gb);
        s += v4::calculateOffset<g>(static_array<index_t, 2>{m,     k + 4}, gb);
        s += v4::calculateOffset<g>(static_array<index_t, 2>{m + 1, k + 1}, gb);
    }
    out[tid] = s;
}
} // namespace

int main()
{
    std::mt19937 rng{42};
    constexpr int choices[]   = {32, 64, 128, 256};
    constexpr int k_choices[] = {32, 64, 128};
    std::uniform_int_distribution<int> dist_m{0, 3};
    std::uniform_int_distribution<int> dist_k{0, 2};
    const index_t MPerBlock = choices[dist_m(rng)];
    const index_t KPerBlock = k_choices[dist_k(rng)];

    constexpr index_t NA = 14;
    index_t h_args[NA];
    h_args[0]  = KPerBlock / 32;          // d0
    h_args[1]  = 1;                        // d1
    h_args[2]  = 16;                       // d2
    h_args[3]  = MPerBlock / 4;            // d3
    h_args[4]  = 2;                        // d4
    h_args[5]  = 1;                        // d5
    h_args[6]  = 8 * MPerBlock;            // s0
    h_args[7]  = 8 * MPerBlock;            // s1
    h_args[8]  = MPerBlock / 2;            // s2
    h_args[9]  = 2;                        // s3
    h_args[10] = 1;                        // s4
    h_args[11] = MPerBlock / 8;            // M0_div_mpair
    h_args[12] = MPerBlock;                // USER_M
    h_args[13] = KPerBlock / 4;            // USER_K

    constexpr index_t N = 1024;
    index_t* h_m = new index_t[N];
    index_t* h_k = new index_t[N];
    std::uniform_int_distribution<int> dist_coord{0, 31};
    for(index_t i = 0; i < N; ++i) { h_m[i] = dist_coord(rng); h_k[i] = dist_coord(rng); }

    const char* env_loop = std::getenv("LOOP_ITERS");
    const index_t loop_iters = env_loop ? static_cast<index_t>(std::atoi(env_loop)) : 10000;

    index_t *d_m=nullptr, *d_k=nullptr, *d_out=nullptr, *d_args=nullptr, *d_iters=nullptr;
    (void)hipMalloc(&d_m, N * sizeof(index_t));
    (void)hipMalloc(&d_k, N * sizeof(index_t));
    (void)hipMalloc(&d_out, N * sizeof(index_t));
    (void)hipMalloc(&d_args, NA * sizeof(index_t));
    (void)hipMemcpy(d_m, h_m, N * sizeof(index_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_k, h_k, N * sizeof(index_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_args, h_args, NA * sizeof(index_t), hipMemcpyHostToDevice);
    (void)hipMalloc(&d_iters, sizeof(index_t));
    (void)hipMemcpy(d_iters, &loop_iters, sizeof(index_t), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(test_kernel, dim3(4), dim3(256), 0, nullptr, d_m, d_k, d_out, d_args, d_iters);
    (void)hipDeviceSynchronize();

    const char* env_n     = std::getenv("N_TRIALS");
    const char* env_b     = std::getenv("TRIAL_BASE");
    const int   n_trials   = env_n ? std::atoi(env_n) : 100;
    const int   trial_base = env_b ? std::atoi(env_b) : 0;

    hipEvent_t start, stop;
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);
    for(int trial = 1; trial <= n_trials; ++trial)
    {
        (void)hipEventRecord(start, nullptr);
        hipLaunchKernelGGL(test_kernel, dim3(4), dim3(256), 0, nullptr, d_m, d_k, d_out, d_args, d_iters);
        (void)hipEventRecord(stop, nullptr);
        (void)hipEventSynchronize(stop);
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, start, stop);
        std::fprintf(stderr, "complex v4 trial %d: %.4f ms\n", trial_base + trial, ms);
    }
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);

    index_t* h_out = new index_t[N];
    (void)hipMemcpy(h_out, d_out, N * sizeof(index_t), hipMemcpyDeviceToHost);
    int rc = static_cast<int>(h_out[0]);
    (void)hipFree(d_m); (void)hipFree(d_k); (void)hipFree(d_out); (void)hipFree(d_args); (void)hipFree(d_iters);
    delete[] h_m; delete[] h_k; delete[] h_out;
    return rc;
}
