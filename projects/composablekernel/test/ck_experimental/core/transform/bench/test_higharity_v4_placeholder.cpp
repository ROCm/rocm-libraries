// Workload HIGHARITY arity-64 (auto-generated)
#include "ck_experimental/core/transform/v4_experimental.hpp"
namespace {
using ck_tile::index_t; using ck_tile::static_array;
namespace v4 = ck_tile::core::transform::v4;
constexpr auto make_higharity_v4_rt()
{{
    using namespace v4;
    return make_transform_graph(
        outputs(read(1)),
        make_embed(dims(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), strides(placeholder<0>{}, placeholder<1>{}, placeholder<2>{}, placeholder<3>{}, placeholder<4>{}, placeholder<5>{}, placeholder<6>{}, placeholder<7>{}, placeholder<8>{}, placeholder<9>{}, placeholder<10>{}, placeholder<11>{}, placeholder<12>{}, placeholder<13>{}, placeholder<14>{}, placeholder<15>{}, 8, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), read(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65), write(1)),
        make_merge(dims(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), read(0), write(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65)),
        inputs(dims(1048576), write(0)));
}}

__global__ void test_kernel(const index_t* k_in, index_t* out,
                            const index_t* runtime_args, const index_t* n_iters_ptr)
{
    constexpr auto g = make_higharity_v4_rt();
    const auto gb = v4::make_graph_bindings<g>(runtime_args[0], runtime_args[1], runtime_args[2], runtime_args[3], runtime_args[4], runtime_args[5], runtime_args[6], runtime_args[7], runtime_args[8], runtime_args[9], runtime_args[10], runtime_args[11], runtime_args[12], runtime_args[13], runtime_args[14], runtime_args[15]);
    const index_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    const index_t kb=k_in[tid]; const index_t ni=*n_iters_ptr;
    index_t s=0;
    for(index_t i=0;i<ni;++i){
        const index_t k = kb + (i & 0xff);
        s += v4::calculateOffset<g>(static_array<index_t, 1>{k}, gb); s += v4::calculateOffset<g>(static_array<index_t, 1>{k + 1}, gb); s += v4::calculateOffset<g>(static_array<index_t, 1>{k + 2}, gb);
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
        std::fprintf(stderr,"higharity v4 placeholder trial %d: %.4f ms\n",tb+t,ms);
    }
    (void)hipEventDestroy(a); (void)hipEventDestroy(b);
    index_t* h_out=new index_t[NN];
    (void)hipMemcpy(h_out,d_out,NN*sizeof(index_t),hipMemcpyDeviceToHost);
    int rc=static_cast<int>(h_out[0]);
    (void)hipFree(d_k);(void)hipFree(d_out);(void)hipFree(d_iters);
    delete[] h_k; delete[] h_out;
    return rc;
}
