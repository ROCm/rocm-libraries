// Workload A (simple) -- V4 experimental, literal.
// Mirrors test_simple_v3_literal.cpp exactly. Same kernel shape:
//   * 1024 threads (4 blocks x 256), 10 (M,K) template specializations,
//     LOOP_ITERS=10000 default, N_TRIALS=100 default.
//
// V4 vs V3 surface mapping:
//   v3::calculateOffset<g>(coords)  ->  v4::applyBindings<g>(empty) + v4::mapCoord<g>
//   v3::make_embed(dims(...), dims(...))  ->  v4::make_embed(dims(...), strides(...))
//   v3::make_merge(dims(...))  ->  v4::make_merge(...)
// Everything else (graph shape, edge ids, kernel/launcher) is identical to V3.

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
constexpr auto make_a_lds_graph_v4()
{
    constexpr index_t K_DIV8 = KPerBlock / 8;
    constexpr index_t K_MOD8 = 8;
    constexpr index_t OFF=0, S_KD=1, S_MP=2, S_KM=3, S_K=4;

    using namespace v4;
    return make_transform_graph(
        outputs(read(OFF)),
        make_embed(dims(K_DIV8, MPerBlock, K_MOD8),
                             strides((MPerBlock + 1) * 8, 8, 1), read(S_KD, S_MP, S_KM), write(OFF)),
        make_merge(dims(K_DIV8, K_MOD8), read(S_K), write(S_KD, S_KM)),
        inputs(dims(MPerBlock, KPerBlock), write(S_MP, S_K)));
}

template <index_t MPerBlock, index_t KPerBlock>
CK_TILE_HOST_DEVICE index_t use_v4(index_t m, index_t k)
{
    constexpr auto g = make_a_lds_graph_v4<MPerBlock, KPerBlock>();
    return v4::calculateOffset<g>(static_array<index_t, 2>{m, k});  // no-bindings overload
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
        std::fprintf(stderr, "simple v4 literal trial %d: %.4f ms\n", trial_base + trial, ms);
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
