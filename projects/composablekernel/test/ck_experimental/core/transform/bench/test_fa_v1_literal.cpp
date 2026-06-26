// Workload fa (flash-attn-shaped) -- regenerated single-spec for 4-way equivalence
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/core/container/container_helper.hpp"
namespace {
using namespace ck_tile;
CK_TILE_HOST_DEVICE constexpr auto make_fa_v1()
{
    return make_naive_tensor_descriptor(
        make_tuple(number<2>{}, number<8>{}, number<128>{}, number<64>{}),
        make_tuple(number<65536>{}, number<8192>{}, number<64>{}, number<1>{}),
        number<1>{}, number<1>{});
}
constexpr auto desc_lit = make_fa_v1();

__global__ void test_kernel(const index_t* k_in, index_t* out,
                            const index_t* n_iters_ptr)
{
    
    const index_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    const index_t kb=k_in[tid]; const index_t ni=*n_iters_ptr;
    index_t s=0;
    for(index_t i=0;i<ni;++i){
        const index_t b  = (kb + i) & 1;
        const index_t h  = (kb + (i>>1)) & 7;
        const index_t sx = (kb + (i>>2)) & 127;
        const index_t d  = (kb + (i>>3)) & 63;
        s += desc_lit.calculate_offset(make_multi_index(b, h, sx, d)); s += desc_lit.calculate_offset(make_multi_index(b, h, sx, d + 1)); s += desc_lit.calculate_offset(make_multi_index(b, h + 1, sx, d)); s += desc_lit.calculate_offset(make_multi_index(b + 1, h, sx, d)); s += desc_lit.calculate_offset(make_multi_index(b, h, sx + 1, d));
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
    constexpr index_t NN=1024;
    index_t* h_k=new index_t[NN];
    std::uniform_int_distribution<int> dc{0,31};
    for(index_t i=0;i<NN;++i) h_k[i]=dc(rng);
    const char* el=std::getenv("LOOP_ITERS"); const index_t li=el?static_cast<index_t>(std::atoi(el)):10000;
    index_t *d_k=nullptr,*d_out=nullptr,*d_iters=nullptr;
    (void)hipMalloc(&d_k,NN*sizeof(index_t)); (void)hipMalloc(&d_out,NN*sizeof(index_t)); (void)hipMalloc(&d_iters,sizeof(index_t));
    (void)hipMemcpy(d_k,h_k,NN*sizeof(index_t),hipMemcpyHostToDevice);
    (void)hipMemcpy(d_iters,&li,sizeof(index_t),hipMemcpyHostToDevice);
    
    hipLaunchKernelGGL(test_kernel,dim3(4),dim3(256),0,nullptr,d_k,d_out,d_iters);
    (void)hipDeviceSynchronize();
    hipEvent_t a,bb; (void)hipEventCreate(&a); (void)hipEventCreate(&bb);
    const char* en=std::getenv("N_TRIALS"); const char* ebs=std::getenv("TRIAL_BASE");
    const int nt=en?std::atoi(en):100; const int tb=ebs?std::atoi(ebs):0;
    for(int t=1;t<=nt;++t){
        (void)hipEventRecord(a,nullptr);
        hipLaunchKernelGGL(test_kernel,dim3(4),dim3(256),0,nullptr,d_k,d_out,d_iters);
        (void)hipEventRecord(bb,nullptr); (void)hipEventSynchronize(bb);
        float ms=0.0f; (void)hipEventElapsedTime(&ms,a,bb);
        std::fprintf(stderr,"fa v1 literal trial %d: %.4f ms\n",tb+t,ms);
    }
    (void)hipEventDestroy(a); (void)hipEventDestroy(bb);
    index_t* h_out=new index_t[NN];
    (void)hipMemcpy(h_out,d_out,NN*sizeof(index_t),hipMemcpyDeviceToHost);
    int rc=static_cast<int>(h_out[0]);
    (void)hipFree(d_k);(void)hipFree(d_out);(void)hipFree(d_iters);
    delete[] h_k; delete[] h_out;
    return rc;
}
