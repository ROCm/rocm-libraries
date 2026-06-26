// Workload C (complex) -- V4 experimental, literal.
// Mirrors test_complex_v3_literal.cpp exactly.
// 6-transform production-complexity GEMM universal LDS chain (6D base).

#include "ck_experimental/core/transform/v4_experimental.hpp"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace {
using ck_tile::index_t;
using ck_tile::static_array;
namespace v4 = ck_tile::core::transform::v4;

template <index_t MPerBlock, index_t KPerBlock>
constexpr auto make_complex_v4()
{
    constexpr index_t KThreadWrite = KPerBlock / 4;
    constexpr index_t K0PerTW = 1;
    constexpr index_t KTRPerm = 4;
    constexpr index_t M0 = MPerBlock / 4;
    constexpr index_t M1 = 4;
    constexpr index_t kfold = 2;
    constexpr index_t mpair = 2;
    constexpr index_t AK1 = 1;

    constexpr index_t d0 = KThreadWrite / kfold / KTRPerm;
    constexpr index_t d1 = K0PerTW;
    constexpr index_t d2 = KTRPerm * M1;
    constexpr index_t d3 = kfold * M0 / mpair;
    constexpr index_t d4 = mpair;
    constexpr index_t d5 = AK1;

    constexpr index_t USER_M = (M0 / mpair) * mpair * M1;
    constexpr index_t USER_K = KTRPerm * d0 * kfold * K0PerTW * AK1;

    constexpr index_t s0 = d1*d2*d3*d4*d5;
    constexpr index_t s1 = d2*d3*d4*d5;
    constexpr index_t s2 = d3*d4*d5;
    constexpr index_t s3 = d4*d5;
    constexpr index_t s4 = d5;

    // Slot ids
    constexpr index_t PH0=0, PH1=1, PH2=2, PH3=3, PH4=4, PH5=5;
    constexpr index_t OFF=6, XR0=7, XR1=8;
    constexpr index_t UM_KTRP=9, UM_M1=10, UM_KFOLD=11, UM_M0MP=12;
    constexpr index_t U_M=13, U_K=14;

    using namespace v4;
    return make_transform_graph(
        outputs(read(OFF)),
        make_embed(dims(d0, d1, d2, d3, d4, d5),
                             strides(s0, s1, s2, s3, s4, 1), read(PH0, PH1, PH2, PH3, PH4, PH5), write(OFF)),
        make_xor(read(XR0, XR1), write(PH2, PH3)),
        make_unmerge(dims(KTRPerm, M1), read(UM_KTRP, UM_M1), write(XR0)),
        make_unmerge(dims(kfold, M0/mpair), read(UM_KFOLD, UM_M0MP), write(XR1)),
        make_merge(dims(KTRPerm, d0, kfold, K0PerTW, AK1), read(U_K), write(UM_KTRP, PH0, UM_KFOLD, PH1, PH5)),
        make_merge(dims(M0/mpair, mpair, M1), read(U_M), write(UM_M0MP, PH4, UM_M1)),
        inputs(dims(USER_M, USER_K), write(U_M, U_K)));
}

template <index_t MPerBlock, index_t KPerBlock>
CK_TILE_HOST_DEVICE index_t use_v4(index_t m, index_t k)
{
    constexpr auto g = make_complex_v4<MPerBlock, KPerBlock>();
    return v4::calculateOffset<g>(static_array<index_t, 2>{m, k});
}

__global__ void test_kernel(const index_t* m_in, const index_t* k_in, index_t* out,
                             const index_t* n_iters_ptr)
{

    const index_t tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t m_base  = m_in[tid];
    const index_t k_base  = k_in[tid];
    const index_t n_iters = *n_iters_ptr;   // runtime -- opaque

    index_t s = 0;
    for(index_t i = 0; i < n_iters; ++i)
    {
        const index_t m = m_base + (i & 0xff);
        const index_t k = k_base + ((i >> 4) & 0xff);

        s += use_v4< 32,   32>(m, k);
        s += use_v4< 32,   64>(m, k);
        s += use_v4< 64,   32>(m, k);
        s += use_v4< 64,   64>(m, k);
        s += use_v4<128,   32>(m, k);
        s += use_v4<128,   64>(m, k);
        s += use_v4<128,  128>(m, k);
        s += use_v4<256,   32>(m, k);
        s += use_v4<256,   64>(m, k);
        s += use_v4<256,  128>(m, k);
    }
    out[tid] = s;
}
} // namespace

int main()
{
    std::mt19937 rng{42};
    constexpr index_t N = 1024;
    index_t* h_m = new index_t[N];
    index_t* h_k = new index_t[N];
    std::uniform_int_distribution<int> dist_coord{0, 31};
    for(index_t i = 0; i < N; ++i) { h_m[i] = dist_coord(rng); h_k[i] = dist_coord(rng); }
    const char* env_loop = std::getenv("LOOP_ITERS");
    const index_t loop_iters = env_loop ? static_cast<index_t>(std::atoi(env_loop)) : 10000;

    index_t *d_m=nullptr, *d_k=nullptr, *d_out=nullptr, *d_iters=nullptr;
    (void)hipMalloc(&d_m, N * sizeof(index_t));
    (void)hipMalloc(&d_k, N * sizeof(index_t));
    (void)hipMalloc(&d_out, N * sizeof(index_t));
    (void)hipMemcpy(d_m, h_m, N * sizeof(index_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_k, h_k, N * sizeof(index_t), hipMemcpyHostToDevice);
    (void)hipMalloc(&d_iters, sizeof(index_t));
    (void)hipMemcpy(d_iters, &loop_iters, sizeof(index_t), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(test_kernel, dim3(4), dim3(256), 0, nullptr, d_m, d_k, d_out, d_iters);
    (void)hipDeviceSynchronize();

    hipEvent_t start, stop;
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);
    const char* env_n     = std::getenv("N_TRIALS");
    const char* env_b     = std::getenv("TRIAL_BASE");
    const int   n_trials   = env_n ? std::atoi(env_n) : 100;
    const int   trial_base = env_b ? std::atoi(env_b) : 0;
    for(int trial = 1; trial <= n_trials; ++trial)
    {
        (void)hipEventRecord(start, nullptr);
        hipLaunchKernelGGL(test_kernel, dim3(4), dim3(256), 0, nullptr, d_m, d_k, d_out, d_iters);
        (void)hipEventRecord(stop, nullptr);
        (void)hipEventSynchronize(stop);
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, start, stop);
        std::fprintf(stderr, "complex v4 literal trial %d: %.4f ms\n", trial_base + trial, ms);
    }
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);

    index_t* h_out = new index_t[N];
    (void)hipMemcpy(h_out, d_out, N * sizeof(index_t), hipMemcpyDeviceToHost);
    int rc = static_cast<int>(h_out[0]);
    (void)hipFree(d_m); (void)hipFree(d_k); (void)hipFree(d_out); (void)hipFree(d_iters);
    delete[] h_m; delete[] h_k; delete[] h_out;
    return rc;
}
