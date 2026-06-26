// Workload HIGHARITY arity-64 (auto-generated)
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/core/container/container_helper.hpp"
namespace {
using namespace ck_tile;
CK_TILE_HOST_DEVICE constexpr auto make_higharity_v1()
{{
    constexpr auto d0 = make_naive_tensor_descriptor(
        make_tuple(number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}), make_tuple(number<524288>{}, number<262144>{}, number<131072>{}, number<65536>{}, number<32768>{}, number<16384>{}, number<8192>{}, number<4096>{}, number<2048>{}, number<1024>{}, number<512>{}, number<256>{}, number<128>{}, number<64>{}, number<32>{}, number<16>{}, number<8>{}, number<4>{}, number<2>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}), number<1>{}, number<1>{});
    return transform_tensor_descriptor(d0,
        make_tuple(make_merge_transform(make_tuple(number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}))),
        make_tuple(sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63>{}),
        make_tuple(sequence<0>{}));
}}
constexpr auto desc_lit = make_higharity_v1();

__global__ void test_kernel(const index_t* k_in, index_t* out, const index_t* n_iters_ptr)
{
    const index_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    const index_t kb=k_in[tid]; const index_t ni=*n_iters_ptr;
    index_t s=0;
    for(index_t i=0;i<ni;++i){
        const index_t k = kb + (i & 0xff);
        s += desc_lit.calculate_offset(make_multi_index(k)); s += desc_lit.calculate_offset(make_multi_index(k + 1)); s += desc_lit.calculate_offset(make_multi_index(k + 2));
    }
    out[tid]=s;
}
} // namespace

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
int main()
{
    std::mt19937 rng{42};
    constexpr index_t NN = 1024;
    index_t* h_k = new index_t[NN];
    std::uniform_int_distribution<int> dc{0, 31};
    for(index_t i=0;i<NN;++i) h_k[i]=dc(rng);
    const char* el=std::getenv("LOOP_ITERS");
    const index_t li = el?static_cast<index_t>(std::atoi(el)):10000;
    index_t *d_k=nullptr,*d_out=nullptr,*d_iters=nullptr;
    
    (void)hipMalloc(&d_k,NN*sizeof(index_t)); (void)hipMalloc(&d_out,NN*sizeof(index_t));
    (void)hipMalloc(&d_iters,sizeof(index_t));
    (void)hipMemcpy(d_k,h_k,NN*sizeof(index_t),hipMemcpyHostToDevice);
    (void)hipMemcpy(d_iters,&li,sizeof(index_t),hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_kernel,dim3(4),dim3(256),0,nullptr,d_k,d_out,d_iters);
    (void)hipDeviceSynchronize();
    hipEvent_t a,b; (void)hipEventCreate(&a); (void)hipEventCreate(&b);
    const char* en=std::getenv("N_TRIALS"); const char* eb=std::getenv("TRIAL_BASE");
    const int nt=en?std::atoi(en):100; const int tb=eb?std::atoi(eb):0;
    for(int t=1;t<=nt;++t){
        (void)hipEventRecord(a,nullptr);
        hipLaunchKernelGGL(test_kernel,dim3(4),dim3(256),0,nullptr,d_k,d_out,d_iters);
        (void)hipEventRecord(b,nullptr); (void)hipEventSynchronize(b);
        float ms=0.0f; (void)hipEventElapsedTime(&ms,a,b);
        std::fprintf(stderr,"higharity v1 literal trial %d: %.4f ms\n",tb+t,ms);
    }
    (void)hipEventDestroy(a); (void)hipEventDestroy(b);
    index_t* h_out=new index_t[NN];
    (void)hipMemcpy(h_out,d_out,NN*sizeof(index_t),hipMemcpyDeviceToHost);
    int rc=static_cast<int>(h_out[0]);
    (void)hipFree(d_k);(void)hipFree(d_out);(void)hipFree(d_iters);
    delete[] h_k; delete[] h_out;
    return rc;
}
