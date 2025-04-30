#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <stdio.h>   // for printf
#include <stdlib.h>  // for malloc

// Example: Compute the QR Factorization of a matrix on the GPU

double *create_example_matrix(rocblas_int *M_out,
                              rocblas_int *N_out,
                              rocblas_int *lda_out) {
  // a *very* small example input; not a very efficient use of the API
  const double A[3][3] = { {  12, -51,   4},
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
  double *hA = (double*)malloc(sizeof(double)*lda*N);
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      // copy A (2D array) into hA (1D array, column-major)
      hA[i + j*lda] = A[i][j];
    }
  }
  return hA;
}

// We use rocsolver_dgeqrf to factor a real M-by-N matrix, A.
// See https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/api/lapack.html#rocsolver-type-geqrf
int main() {
  rocblas_int M;          // rows
  rocblas_int N;          // cols
  rocblas_int lda;        // leading dimension
  double *hA = create_example_matrix(&M, &N, &lda); // input matrix on CPU

  // let's print the input matrix, just to see it
  printf("A = [\n");
  for (size_t i = 0; i < M; ++i) {
    printf("  ");
    for (size_t j = 0; j < N; ++j) {
      printf("% .3f ", hA[i + j*lda]);
    }
    printf(";\n");
  }
  printf("]\n");

  // initialization
  rocblas_handle handle;
  rocblas_create_handle(&handle);

  // Some rocsolver functions may trigger rocblas to load its GEMM kernels.
  // You can preload the kernels by explicitly invoking rocblas_initialize
  // (e.g., to exclude one-time initialization overhead from benchmarking).

  // preload rocBLAS GEMM kernels (optional)
  // rocblas_initialize();

  // calculate the sizes of our arrays
  size_t size_A = lda * (size_t)N;   // count of elements in matrix A
  size_t size_piv = (M < N) ? M : N; // count of Householder scalars

  // allocate memory on GPU
  double *dA, *dIpiv;
  hipMalloc((void**)&dA, sizeof(double)*size_A);
  hipMalloc((void**)&dIpiv, sizeof(double)*size_piv);

  // copy data to GPU
  hipMemcpy(dA, hA, sizeof(double)*size_A, hipMemcpyHostToDevice);

  // compute the QR factorization on the GPU
  rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv);

  // copy the results back to CPU
  double *hIpiv = (double*)malloc(sizeof(double)*size_piv); // householder scalars on CPU
  hipMemcpy(hA, dA, sizeof(double)*size_A, hipMemcpyDeviceToHost);
  hipMemcpy(hIpiv, dIpiv, sizeof(double)*size_piv, hipMemcpyDeviceToHost);

  // the results are now in hA and hIpiv
  // we can print some of the results if we want to see them
  printf("R = [\n");
  for (size_t i = 0; i < M; ++i) {
    printf("  ");
    for (size_t j = 0; j < N; ++j) {
      printf("% .3f ", (i <= j) ? hA[i + j*lda] : 0);
    }
    printf(";\n");
  }
  printf("]\n");

  // clean up
  free(hIpiv);
  hipFree(dA);
  hipFree(dIpiv);
  free(hA);
  rocblas_destroy_handle(handle);
}
