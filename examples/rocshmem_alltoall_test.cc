/*
hipcc -c -fgpu-rdc -x hip rocshmem_alltoall_test.cc \
  -I/opt/rocm/include \
  -I$ROCSHMEM_INSTALL_DIR/include \
  -I$OPENMPI_UCX_INSTALL_DIR/include/

hipcc -fgpu-rdc --hip-link rocshmem_alltoall_test.o -o rocshmem_alltoall_test \
  $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a \
  $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so \
  -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64

ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 8 ./rocshmem_alltoall_test
*/

#include <iostream>

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>

#define CHECK_HIP(condition) {                                            \
        hipError_t error = condition;                                     \
        if(error != hipSuccess){                                          \
            fprintf(stderr,"HIP error: %d line: %d\n", error,  __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, error);                             \
        }                                                                 \
    }

using namespace rocshmem;

__global__ void alltoall_test(int *source, int *dest, size_t nelem,
        rocshmem_team_t team) {
    __shared__ rocshmem_ctx_t ctx;
    int64_t ctx_type = 0;

    rocshmem_wg_init();
    rocshmem_wg_ctx_create(ctx_type, &ctx);
    int num_pes = rocshmem_ctx_n_pes(ctx);

    rocshmem_ctx_int_alltoall_wg(ctx, team, dest, source, nelem);

    rocshmem_ctx_quiet(ctx);
    __syncthreads();

    rocshmem_wg_ctx_destroy(&ctx);
    rocshmem_wg_finalize();
}

static void init_sendbuf (int *source, int nelem, int my_pe, int npes)
{
    for (int pe = 0; pe < npes; pe++) {
        for (int i = 0; i < nelem; i++) {
            int idx = (pe * nelem) + i;
            source[idx] = my_pe + pe;
        }
    }
}

static bool check_recvbuf(int *dest, int nelem, int my_pe, int npes)
{
    bool res=true;

    for(int pe = 0; pe < npes; pe++) {
        for(int i = 0; i < nelem; i++) {
            int idx = (pe * nelem) + i;
            int result = my_pe + pe;
            if (dest[idx] != result) {
                res = false;
#ifdef VERBOSE
                printf("recvbuf[%d] = %d expected %d \n", i, dest[i], result);
#endif
            }
        }
    }

    return res;
}

#define MAX_ELEM 256

int main (int argc, char **argv)
{
    int nelem = MAX_ELEM;

    if (argc > 1) {
        nelem = atoi(argv[1]);
    }

    int my_pe = rocshmem_my_pe();
    int npes =  rocshmem_n_pes();

    int ndevices, my_device = 0;
    CHECK_HIP(hipGetDeviceCount(&ndevices));
    my_device = my_pe % ndevices;
    CHECK_HIP(hipSetDevice(my_device));

    rocshmem_init();

    int *source = (int *)rocshmem_malloc(nelem * npes * sizeof(int));
    int *dest = (int *)rocshmem_malloc(nelem * npes * sizeof(int));
    if (NULL == source || NULL == dest) {
        std::cout << "Error allocating memory from symmetric heap" << std::endl;
        std::cout << "source: " << source << ", dest: " << dest << ", size: "
          << sizeof(int) * nelem * npes << std::endl;
        rocshmem_global_exit(1);
    }

    init_sendbuf(source, nelem, my_pe, npes);
    for (int i = 0; i < nelem * npes; i++) {
        dest[i] = -1;
    }

    rocshmem_team_t team_reduce_world_dup;
    team_reduce_world_dup = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, npes, nullptr, 0,
                               &team_reduce_world_dup);

    CHECK_HIP(hipDeviceSynchronize());

    int threadsPerBlock=256;
    alltoall_test<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(source, dest,
                        nelem, team_reduce_world_dup);
    CHECK_HIP(hipDeviceSynchronize());

    bool pass = check_recvbuf(dest, nelem, my_pe, npes);

    printf("Test %s \t nelem %d %s\n", argv[0], nelem, pass ? "[PASS]" : "[FAIL]");
    
    rocshmem_free(source);
    rocshmem_free(dest);

    rocshmem_finalize();
    return 0;
}
