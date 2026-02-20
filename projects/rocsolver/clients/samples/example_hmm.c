#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <stdio.h>   // for printf
#include <stdlib.h>  // for malloc

// Example: Compute the QR Factorization of a matrix on the GPU
// using unified memory (via hipMallocManaged)

double* create_example_matrix(rocblas_int *M_out,
                              rocblas_int *N_out,
                              rocblas_int *lda_out) {
  // a *very* small example input; not a very efficient use of the API
  const double A_source[3][3] = { {  12, -51,   4},
                           {   6, 167, -68},
                           {  -4,  24, -41} };
  const rocblas_int M = 3;
  const rocblas_int N = 3;
  const rocblas_int lda = 3;
  *M_out = M;
  *N_out = N;
  *lda_out = lda;
  // note: rocsolver matrices must be stored in column major format,
  //       i.e. entry (i,j) should be accessed by hA[i + j*lda]
  double* A;
  hipMallocManaged((void**)&A, sizeof(double)*lda*N, hipMemAttachGlobal);
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      // copy A (2D array) into hA (1D array, column-major)
      A[i + j*lda] = A_source[i][j];
    }
  }
  return A;
}

// We use rocsolver_dgeqrf to factor a real M-by-N matrix, A.
// See https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/api/lapack.html#rocsolver-type-geqrf
int main() {
  rocblas_int M;          // rows
  rocblas_int N;          // cols
  rocblas_int lda;        // leading dimension
  double* A = create_example_matrix(&M, &N, &lda); // input matrix on CPU

  // let's print the input matrix, just to see it
  printf("A = [\n");
  for (size_t i = 0; i < M; ++i) {
    printf("  ");
    for (size_t j = 0; j < N; ++j) {
      printf("% .3f ", A[i + j*lda]);
    }
    printf(";\n");
  }
  printf("]\n");

  // initialization
  rocblas_handle handle;
  rocblas_create_handle(&handle);

  // check for enablement of managed memory
  int deviceID, hmm_enabled;
  hipGetDevice(&deviceID);
  hipDeviceGetAttribute(&hmm_enabled,
                        hipDeviceAttributeManagedMemory,
                        deviceID);
  if (!hmm_enabled)
  {
    printf("Managed memory not enabled on device %i\n", deviceID);
    rocblas_destroy_handle(handle);
    return 0;
  }

  // calculate the sizes of our arrays
  size_t size_piv = (M < N) ? M : N; // count of Householder scalars

  // allocate memory
  double *ipiv;
  hipMallocManaged((void**)&ipiv, sizeof(double)*size_piv, hipMemAttachGlobal);

  // determine workspace size
  size_t size_W;
  rocblas_start_device_memory_size_query(handle);
  rocsolver_dgetrf(handle, M, N, NULL, lda, NULL, NULL);
  rocblas_stop_device_memory_size_query(handle, &size_W);

  // create custom workspace
  double *work;
  hipMallocManaged((void**)&work, size_W, hipMemAttachGlobal);
  rocblas_set_workspace(handle, work, size_W);

  // compute the QR factorization on the GPU
  hipStream_t stream;
  rocblas_get_stream(handle, &stream);
  rocsolver_dgeqrf(handle, M, N, A, lda, ipiv);
  hipStreamSynchronize(stream);

  // the results are now in A and ipiv
  // we can print some of the results if we want to see them
  printf("R = [\n");
  for (size_t i = 0; i < M; ++i) {
    printf("  ");
    for (size_t j = 0; j < N; ++j) {
      printf("% .3f ", (i <= j) ? A[i + j*lda] : 0);
    }
    printf(";\n");
  }
  printf("]\n");

  // clean up
  hipFree(A);
  hipFree(ipiv);
  hipFree(work);
  rocblas_destroy_handle(handle);
}
