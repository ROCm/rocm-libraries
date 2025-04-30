#include <algorithm> // for std::min
#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <stdio.h>   // for size_t, printf
#include <vector>

// Example: Compute the QR Factorization of a matrix on the GPU

void get_example_matrix(std::vector<double>& hA,
                        rocblas_int& M,
                        rocblas_int& N,
                        rocblas_int& lda) {
  // a *very* small example input; not a very efficient use of the API
  const double A[3][3] = { {  12, -51,   4},
                           {   6, 167, -68},
                           {  -4,  24, -41} };
  M = 3;
  N = 3;
  lda = 3;
  // note: rocsolver matrices must be stored in column major format,
  //       i.e. entry (i,j) should be accessed by hA[i + j*lda]
  hA.resize(size_t(lda) * N);
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      // copy A (2D array) into hA (1D array, column-major)
      hA[i + j*lda] = A[i][j];
    }
  }
}

// We use rocsolver_dgeqrf to factor a real M-by-N matrix, A.
// See https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/api/lapack.html#rocsolver-type-geqrf
// and https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/userguide/logging.html
int main() {
  rocblas_int M;          // rows
  rocblas_int N;          // cols
  rocblas_int lda;        // leading dimension
  std::vector<double> hA; // input matrix on CPU
  get_example_matrix(hA, M, N, lda);

  // initialization
  rocblas_handle handle;
  rocblas_create_handle(&handle);
  rocsolver_log_begin();

  // calculate the sizes of our arrays
  size_t size_A = size_t(lda) * N;          // count of elements in matrix A
  size_t size_piv = size_t(std::min(M, N)); // count of Householder scalars

  // allocate memory on GPU
  double *dA, *dIpiv;
  hipMalloc(&dA, sizeof(double)*size_A);
  hipMalloc(&dIpiv, sizeof(double)*size_piv);


  // copy data to GPU
  hipMemcpy(dA, hA.data(), sizeof(double)*size_A, hipMemcpyHostToDevice);

  // begin trace logging and profile logging (max depth = 4)
  rocsolver_log_set_layer_mode(rocblas_layer_mode_log_trace | rocblas_layer_mode_log_profile | rocblas_layer_mode_ex_log_kernel);
  rocsolver_log_set_max_levels(4);

  // compute the QR factorization on the GPU
  rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv);

  // stop logging, print profile results, and clear the profile data
  rocsolver_log_flush_profile();
  rocsolver_log_restore_defaults();


  // copy data to GPU
  hipMemcpy(dA, hA.data(), sizeof(double)*size_A, hipMemcpyHostToDevice);

  // begin bench logging and profile logging (max depth = 1)
  rocsolver_log_set_layer_mode(rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile);

  // compute the QR factorization on the GPU
  rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv);

  // stop logging and print profile results
  rocsolver_log_write_profile();
  rocsolver_log_restore_defaults();


  // clean up
  hipFree(dA);
  hipFree(dIpiv);
  rocsolver_log_end();
  rocblas_destroy_handle(handle);
}
