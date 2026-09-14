#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <stdio.h> // for printf
#include <stdlib.h> // for malloc

// Example: Compute the QR Factorizations of a batch of matrices on the GPU

double **create_example_matrices(rocblas_int *M_out,
                                 rocblas_int *N_out,
                                 rocblas_int *lda_out,
                                 rocblas_int *batch_count_out) {
  // a small example input
  const double A[2][3][3] = {
    // First input matrix
    { { 12, -51,   4},
      {  6, 167, -68},
      { -4,  24, -41} },

    // Second input matrix
    { {  3, -12,  11},
      {  4, -46,  -2},
      {  0,   5,  15} } };

  const rocblas_int M = 3;
  const rocblas_int N = 3;
  const rocblas_int lda = 3;
  const rocblas_int batch_count = 2;
  *M_out = M;
  *N_out = N;
  *lda_out = lda;
  *batch_count_out = batch_count;

  // allocate space for input matrix data on CPU
  double **hA = (double**)malloc(sizeof(double*)*batch_count);
  hA[0] = (double*)malloc(sizeof(double)*lda*N);
  hA[1] = (double*)malloc(sizeof(double)*lda*N);

  for (size_t b = 0; b < batch_count; ++b)
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j)
        hA[b][i + j*lda] = A[b][i][j];

  return hA;
}

// Use rocsolver_dgeqrf_batched to factor a batch of real M-by-N matrices.
int main() {
  rocblas_int M;           // rows
  rocblas_int N;           // cols
  rocblas_int lda;         // leading dimension
  rocblas_int batch_count; // number of matricies
  double **hA = create_example_matrices(&M, &N, &lda, &batch_count);

  // print the input matrices
  for (size_t b = 0; b < batch_count; ++b) {
    printf("A[%zu] = [\n", b);
    for (size_t i = 0; i < M; ++i) {
      printf("  ");
      for (size_t j = 0; j < N; ++j) {
        printf("% 4.f ", hA[b][i + j*lda]);
      }
      printf(";\n");
    }
    printf("]\n");
  }

  // initialization
  rocblas_handle handle;
  rocblas_create_handle(&handle);

  // preload rocBLAS GEMM kernels (optional)
  // rocblas_initialize();

  // calculate the sizes of the arrays
  size_t size_A = lda * (size_t)N;          // count of elements in each matrix A
  rocblas_stride strideP = (M < N) ? M : N; // stride of Householder scalar sets
  size_t size_piv = strideP * (size_t)batch_count; // elements in array for Householder scalars

  // allocate memory on the CPU for an array of pointers,
  // then allocate memory for each matrix on the GPU.
  double **A = (double**)malloc(sizeof(double*)*batch_count);
  for (rocblas_int b = 0; b < batch_count; ++b)
    hipMalloc((void**)&A[b], sizeof(double)*size_A);

  // allocate memory on GPU for the array of pointers and Householder scalars
  double **dA, *dIpiv;
  hipMalloc((void**)&dA, sizeof(double*)*batch_count);
  hipMalloc((void**)&dIpiv, sizeof(double)*size_piv);

  // copy each matrix to the GPU
  for (rocblas_int b = 0; b < batch_count; ++b)
    hipMemcpy(A[b], hA[b], sizeof(double)*size_A, hipMemcpyHostToDevice);

  // copy the array of pointers to the GPU
  hipMemcpy(dA, A, sizeof(double*)*batch_count, hipMemcpyHostToDevice);

  // compute the QR factorizations on the GPU
  rocsolver_dgeqrf_batched(handle, M, N, dA, lda, dIpiv, strideP, batch_count);

  // copy the results back to CPU
  double *hIpiv = (double*)malloc(sizeof(double)*size_piv); // householder scalars on CPU
  hipMemcpy(hIpiv, dIpiv, sizeof(double)*size_piv, hipMemcpyDeviceToHost);
  for (rocblas_int b = 0; b < batch_count; ++b)
    hipMemcpy(hA[b], A[b], sizeof(double)*size_A, hipMemcpyDeviceToHost);

  // the results are now in hA and hIpiv
  // print some of the results
  for (size_t b = 0; b < batch_count; ++b) {
    printf("R[%zu] = [\n", b);
    for (size_t i = 0; i < M; ++i) {
      printf("  ");
      for (size_t j = 0; j < N; ++j) {
        printf("% 4.f ", (i <= j) ? hA[b][i + j*lda] : 0);
      }
      printf(";\n");
    }
    printf("]\n");
  }

  // clean up
  free(hIpiv);
  for (rocblas_int b = 0; b < batch_count; ++b)
    free(hA[b]);
  free(hA);
  for (rocblas_int b = 0; b < batch_count; ++b)
    hipFree(A[b]);
  free(A);
  hipFree(dA);
  hipFree(dIpiv);
  rocblas_destroy_handle(handle);
}
