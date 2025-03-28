/*
hipcc -c -fgpu-rdc -x hip rocshmem_init_attr_test.cc \
  -I/opt/rocm/include \
  -I$ROCSHMEM_INSTALL_DIR/include \
  -I$OPENMPI_UCX_INSTALL_DIR/include/

hipcc -fgpu-rdc --hip-link rocshmem_init_attr_test.o -o rocshmem_init_attr_test \
  $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a \
  $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so \
  -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64

ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 ./rocshmem_init_attr_test
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

int main (int argc, char **argv)
{
    int rank, nranks;
    int ret;
    rocshmem_uniqueid_t uid;
    rocshmem_init_attr_t attr;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &nranks);

    if (rank == 0) {
      ret = rocshmem_get_uniqueid (&uid);
      if (ret != ROCSHMEM_SUCCESS) {
	std::cout << rank << ": Error in rocshmem_get_uniqueid. Aborting.\n";
	MPI_Abort (MPI_COMM_WORLD, ret);
      }
    }

    MPI_Bcast (&uid, sizeof(rocshmem_uniqueid_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    ret = rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << rank << ": Error in rocshmem_set_attr_uniqueid_args. Aborting.\n";
      MPI_Abort (MPI_COMM_WORLD, ret);
    }
    
    ret = rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << rank << ": Error in rocshmem_init_attr. Aborting.\n";
      MPI_Abort (MPI_COMM_WORLD, ret);
    }

    std::cout << rank << ": rocshmem_init_attr SUCCESS\n";

    rocshmem_finalize();
    MPI_Finalize();
    return 0;
}
