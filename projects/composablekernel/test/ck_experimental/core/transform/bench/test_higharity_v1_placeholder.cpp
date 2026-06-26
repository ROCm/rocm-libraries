// Workload HIGHARITY arity-64 (auto-generated)
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/core/container/container_helper.hpp"
namespace {
using namespace ck_tile;
CK_TILE_HOST_DEVICE auto make_higharity_desc_v1(const index_t* s)
{{
    const auto d0 = make_naive_tensor_descriptor(make_tuple(number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}), make_tuple(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15], number<8>{}, number<4>{}, number<2>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}));
    return transform_tensor_descriptor(d0,
        make_tuple(make_merge_transform(make_tuple(number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<2>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}, number<1>{}))),
        make_tuple(sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63>{}),
        make_tuple(sequence<0>{}));
}}

__global__ void test_kernel(const index_t* k_in, index_t* out,
                            const index_t* runtime_args, const index_t* n_iters_ptr)
{
    const auto desc = make_higharity_desc_v1(runtime_args);
    const index_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    const index_t kb=k_in[tid]; const index_t ni=*n_iters_ptr;
    index_t s=0;
    for(index_t i=0;i<ni;++i){
        const index_t k = kb + (i & 0xff);
        s += desc.calculate_offset(make_multi_index(k)); s += desc.calculate_offset(make_multi_index(k + 1)); s += desc.calculate_offset(make_multi_index(k + 2));
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
    index_t *d_k=nullptr,*d_out=nullptr,*d_iters=nullptr; index_t* d_args=nullptr;
    index_t h_args[16]={ 524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16 };
    (void)hipMalloc(&d_args,16*sizeof(index_t));
    (void)hipMemcpy(d_args,h_args,16*sizeof(index_t),hipMemcpyHostToDevice);
    (void)hipMalloc(&d_k,NN*sizeof(index_t)); (void)hipMalloc(&d_out,NN*sizeof(index_t));
    (void)hipMalloc(&d_iters,sizeof(index_t));
    (void)hipMemcpy(d_k,h_k,NN*sizeof(index_t),hipMemcpyHostToDevice);
    (void)hipMemcpy(d_iters,&li,sizeof(index_t),hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_kernel,dim3(4),dim3(256),0,nullptr,d_k,d_out,d_args,d_iters);
    (void)hipDeviceSynchronize();
    hipEvent_t a,b; (void)hipEventCreate(&a); (void)hipEventCreate(&b);
    const char* en=std::getenv("N_TRIALS"); const char* eb=std::getenv("TRIAL_BASE");
    const int nt=en?std::atoi(en):100; const int tb=eb?std::atoi(eb):0;
    for(int t=1;t<=nt;++t){
        (void)hipEventRecord(a,nullptr);
        hipLaunchKernelGGL(test_kernel,dim3(4),dim3(256),0,nullptr,d_k,d_out,d_args,d_iters);
        (void)hipEventRecord(b,nullptr); (void)hipEventSynchronize(b);
        float ms=0.0f; (void)hipEventElapsedTime(&ms,a,b);
        std::fprintf(stderr,"higharity v1 placeholder trial %d: %.4f ms\n",tb+t,ms);
    }
    (void)hipEventDestroy(a); (void)hipEventDestroy(b);
    index_t* h_out=new index_t[NN];
    (void)hipMemcpy(h_out,d_out,NN*sizeof(index_t),hipMemcpyDeviceToHost);
    int rc=static_cast<int>(h_out[0]);
    (void)hipFree(d_k);(void)hipFree(d_out);(void)hipFree(d_iters);
    delete[] h_k; delete[] h_out;
    return rc;
}
