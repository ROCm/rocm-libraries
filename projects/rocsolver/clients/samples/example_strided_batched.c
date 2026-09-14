#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <stdio.h>   // for printf
#include <stdlib.h> // for malloc

// Example: Compute the QR Factorizations of an array of matrices on the GPU

double *create_example_matrices(rocblas_int *M_out,
                                rocblas_int *N_out,
                                rocblas_int *lda_out,
                                rocblas_stride *strideA_out,
                                rocblas_int *batch_count_out) {
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
  const rocblas_stride strideA = lda * N;
  const rocblas_int batch_count = 2;
  *M_out = M;
  *N_out = N;
  *lda_out = lda;
  *strideA_out = strideA;
  *batch_count_out = batch_count;

  // allocate space for input matrix data on CPU
  double *hA = (double*)malloc(sizeof(double)*strideA*batch_count);

  // copy A (3D array) into hA (1D array, column-major)
  for (size_t b = 0; b < batch_count; ++b)
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j)
        hA[i + j*lda + b*strideA] = A[b][i][j];

  return hA;
}

// Use rocsolver_dgeqrf_strided_batched to factor an array of real M-by-N matrices.
int main() {
  rocblas_int M;           // rows
  rocblas_int N;           // cols
  rocblas_int lda;         // leading dimension
  rocblas_stride strideA;  // stride from start of one matrix to the next
  rocblas_int batch_count; // number of matricies
  double *hA = create_example_matrices(&M, &N, &lda, &strideA, &batch_count);

  // print the input matrices
  for (size_t b = 0; b < batch_count; ++b) {
    printf("A[%zu] = [\n", b);
    for (size_t i = 0; i < M; ++i) {
      printf("  ");
      for (size_t j = 0; j < N; ++j) {
        printf("% 4.f ", hA[i + j*lda + strideA*b]);
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

  // calculate the sizes of our arrays
  size_t size_A = strideA * (size_t)batch_count;   // elements in array for matrices
  rocblas_stride strideP = (M < N) ? M : N;        // stride of Householder scalar sets
  size_t size_piv = strideP * (size_t)batch_count; // elements in array for Householder scalars

  // allocate memory on GPU
  double *dA, *dIpiv;
  hipMalloc((void**)&dA, sizeof(double)*size_A);
  hipMalloc((void**)&dIpiv, sizeof(double)*size_piv);

  // copy data to GPU
  hipMemcpy(dA, hA, sizeof(double)*size_A, hipMemcpyHostToDevice);

  // compute the QR factorizations on the GPU
  rocsolver_dgeqrf_strided_batched(handle, M, N, dA, lda, strideA, dIpiv, strideP, batch_count);

  // copy the results back to CPU
  double *hIpiv = (double*)malloc(sizeof(double)*size_piv); // householder scalars on CPU
  hipMemcpy(hA, dA, sizeof(double)*size_A, hipMemcpyDeviceToHost);
  hipMemcpy(hIpiv, dIpiv, sizeof(double)*size_piv, hipMemcpyDeviceToHost);

  // the results are now in hA and hIpiv
  // print some of the results
  for (size_t b = 0; b < batch_count; ++b) {
    printf("R[%zu] = [\n", b);
    for (size_t i = 0; i < M; ++i) {
      printf("  ");
      for (size_t j = 0; j < N; ++j) {
        printf("% 4.f ", (i <= j) ? hA[i + j*lda + strideA*b] : 0);
      }
      printf(";\n");
    }
    printf("]\n");
  }

  // clean up
  free(hIpiv);
  hipFree(dA);
  hipFree(dIpiv);
  free(hA);
  rocblas_destroy_handle(handle);
}
