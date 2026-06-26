// Workload fa (flash-attn-shaped) -- regenerated single-spec for 4-way equivalence
#include "ck_experimental/core/transform/v4_experimental.hpp"
namespace {
using ck_tile::index_t; using ck_tile::static_array;
namespace v4 = ck_tile::core::transform::v4;
constexpr auto make_fa_v4_rt()
{
    using namespace v4;
    return make_transform_graph(
        outputs(read(4)),
        make_embed(dims(2, 8, 128, 64), strides(placeholder<0>{}, placeholder<1>{}, placeholder<2>{}, placeholder<3>{}), read(0, 8, 2, 3), write(4)),
        make_unmerge(dims(1, 2, 4), read(5, 6, 7), write(8)),
        make_merge(dims(1, 2, 4), read(1), write(5, 6, 7)),
        inputs(dims(2, 8, 128, 64), write(0, 1, 2, 3)));
}

__global__ void test_kernel(const index_t* k_in, index_t* out, const index_t* runtime_args,
                            const index_t* n_iters_ptr)
{
    constexpr auto g = make_fa_v4_rt();
    const auto gb = v4::make_graph_bindings<g>(runtime_args[0], runtime_args[1], runtime_args[2], runtime_args[3]);
    const index_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    const index_t kb=k_in[tid]; const index_t ni=*n_iters_ptr;
    index_t s=0;
    for(index_t i=0;i<ni;++i){
        const index_t b  = (kb + i) & 1;
        const index_t h  = (kb + (i>>1)) & 7;
        const index_t sx = (kb + (i>>2)) & 127;
        const index_t d  = (kb + (i>>3)) & 63;
        s += v4::calculateOffset<g>(static_array<index_t,4>{b, h, sx, d}, gb); s += v4::calculateOffset<g>(static_array<index_t,4>{b, h, sx, d + 1}, gb); s += v4::calculateOffset<g>(static_array<index_t,4>{b, h + 1, sx, d}, gb); s += v4::calculateOffset<g>(static_array<index_t,4>{b + 1, h, sx, d}, gb); s += v4::calculateOffset<g>(static_array<index_t,4>{b, h, sx + 1, d}, gb);
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
    index_t *d_k=nullptr,*d_out=nullptr,*d_iters=nullptr; index_t* d_args=nullptr;
    (void)hipMalloc(&d_k,NN*sizeof(index_t)); (void)hipMalloc(&d_out,NN*sizeof(index_t)); (void)hipMalloc(&d_iters,sizeof(index_t));
    (void)hipMemcpy(d_k,h_k,NN*sizeof(index_t),hipMemcpyHostToDevice);
    (void)hipMemcpy(d_iters,&li,sizeof(index_t),hipMemcpyHostToDevice);
    index_t h_args[4]={ 65536, 8192, 64, 1 };
    (void)hipMalloc(&d_args,4*sizeof(index_t));
    (void)hipMemcpy(d_args,h_args,4*sizeof(index_t),hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_kernel,dim3(4),dim3(256),0,nullptr,d_k,d_out,d_args,d_iters);
    (void)hipDeviceSynchronize();
    hipEvent_t a,bb; (void)hipEventCreate(&a); (void)hipEventCreate(&bb);
    const char* en=std::getenv("N_TRIALS"); const char* ebs=std::getenv("TRIAL_BASE");
    const int nt=en?std::atoi(en):100; const int tb=ebs?std::atoi(ebs):0;
    for(int t=1;t<=nt;++t){
        (void)hipEventRecord(a,nullptr);
        hipLaunchKernelGGL(test_kernel,dim3(4),dim3(256),0,nullptr,d_k,d_out,d_args,d_iters);
        (void)hipEventRecord(bb,nullptr); (void)hipEventSynchronize(bb);
        float ms=0.0f; (void)hipEventElapsedTime(&ms,a,bb);
        std::fprintf(stderr,"fa v4 placeholder trial %d: %.4f ms\n",tb+t,ms);
    }
    (void)hipEventDestroy(a); (void)hipEventDestroy(bb);
    index_t* h_out=new index_t[NN];
    (void)hipMemcpy(h_out,d_out,NN*sizeof(index_t),hipMemcpyDeviceToHost);
    int rc=static_cast<int>(h_out[0]);
    (void)hipFree(d_k);(void)hipFree(d_out);(void)hipFree(d_iters);
    delete[] h_k; delete[] h_out;
    return rc;
}
