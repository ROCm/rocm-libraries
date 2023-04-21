// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "hip/hip_runtime.h"
#include <cstdlib>
#include <unistd.h>
#include <random>
#include <cuda/std/type_traits>
#include <cuda/std/limits>

#define ENABLE_IF(...) cuda::std::enable_if_t<(__VA_ARGS__)>* = nullptr

#define HIP_CHECK(command)                                                     \
{                                                                            \
  if (command != hipSuccess) {                                               \
    printf("HIP ERROR: %s", hipGetErrorString(command));                     \
    std::abort();                                                            \
  }                                                                          \
}

#define divideAndRoundUp(x, y) ((x) / (y) + ((x) % (y) != 0))

constexpr int N_VALS = 1; 

template <typename T, ENABLE_IF(cuda::std::numeric_limits<T>::is_signed)>
__device__ T add_safe(T const& value, T const& delta)
{
  return (value < 0 || (cuda::std::numeric_limits<T>::max() - value) >= delta)
           ? (value + delta)
           : cuda::std::numeric_limits<T>::max();
}

template <typename T, ENABLE_IF(not cuda::std::numeric_limits<T>::is_signed)>
__device__ T add_safe(T const& value, T const& delta)
{
  return ((cuda::std::numeric_limits<T>::max() - value) >= delta)
           ? (value + delta)
           : cuda::std::numeric_limits<T>::max();
}

template <typename T, ENABLE_IF(cuda::std::numeric_limits<T>::is_signed)>
__device__ T subtract_safe(T const& value, T const& delta)
{
  return (value >= 0 || (value - cuda::std::numeric_limits<T>::min()) >= delta)
           ? (value - delta)
           : cuda::std::numeric_limits<T>::min();
}

template <typename T, ENABLE_IF(not cuda::std::numeric_limits<T>::is_signed)>
__device__ T subtract_safe(T const& value, T const& delta)
{
  return ((value - cuda::std::numeric_limits<T>::min()) >= delta)
           ? (value - delta)
           : cuda::std::numeric_limits<T>::min();
}


template<typename T, ENABLE_IF(cuda::std::is_floating_point<T>::value)>
__device__ void printResults(const char *op_name, const char *op, T first, T second, T result, T expected, T unsafe) {
   printf("\e[1;35m\t\t\t %s: %.6ef %s %.6ef, \e[1;32m actual result (safe operation with numerics) = %.6ef, expected = %.6ef, \e[1;33m unsafe = %.6ef\n", op_name, first, op, second, result, expected, unsafe);
}

template<void* = nullptr>
__device__ void printResults(const char *op_name, const char *op, unsigned int first, unsigned int second, unsigned int result, unsigned int expected, unsigned int unsafe) {
   printf("\e[1;35m\t\t\t %s: %u %s %u \e[1;32m actual result (safe operation with numerics) = %u, expected = %u, \e[1;33m unsafe = %u\n", op_name, first, op, second, result, expected, unsafe);
}

template<void* = nullptr>
__device__ void printResults(const char *op_name, const char *op, int first, int second, int result, int expected, int unsafe) {
   printf("\e[1;35m\t\t\t %s: %d %s %d \e[1;32m actual result (safe operation with numerics) = %d, expected = %d, \e[1;33m unsafe = %d\n", op_name, first, op, second, result, expected, unsafe);
}

template<typename T>
__device__
void test_safe_operations(T first_operand) {
  T second_operand_add_overflow = cuda::std::numeric_limits<T>::max() - first_operand + 1;
  T second_operand_add_ok = second_operand_add_overflow - second_operand_add_overflow/2;

  printResults("add", "+", first_operand, second_operand_add_overflow, add_safe<T>(first_operand, second_operand_add_overflow), cuda::std::numeric_limits<T>::max(), first_operand+second_operand_add_overflow);
  assert(add_safe<T>(first_operand, second_operand_add_overflow)==cuda::std::numeric_limits<T>::max());
  
  printResults("add", "+", first_operand, second_operand_add_ok, add_safe<T>(first_operand, second_operand_add_ok), first_operand+second_operand_add_ok, first_operand+second_operand_add_ok);
  assert(add_safe<T>(first_operand, second_operand_add_ok)<cuda::std::numeric_limits<T>::max());

  if constexpr(cuda::std::is_signed_v<T>)
    first_operand = (first_operand>0)? -first_operand : first_operand; 

  T second_operand_sub_underflow = first_operand + cuda::std::numeric_limits<T>::min() + 1;
  T second_operand_sub_ok = second_operand_sub_underflow -100;

  printResults("sub", "-", first_operand, second_operand_sub_underflow, subtract_safe<T>(first_operand, second_operand_sub_underflow), cuda::std::numeric_limits<T>::min(),
                                                                        first_operand-second_operand_sub_underflow);
  assert(subtract_safe<T>(first_operand, second_operand_sub_underflow)==cuda::std::numeric_limits<T>::min());
  
  printResults("sub", "-", first_operand, second_operand_sub_ok, subtract_safe<T>(first_operand, second_operand_sub_ok), first_operand-second_operand_sub_ok, first_operand-second_operand_sub_ok);
  assert(subtract_safe<T>(first_operand, second_operand_sub_ok)>cuda::std::numeric_limits<T>::min());
}

template<typename T> 
__global__ 
void test_safe_operations_kernel(T *d_first_operands) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if(id<N_VALS) {
    test_safe_operations(d_first_operands[id]);      
  }
}

template<typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
void initialize_array(T *arr) {
  std::random_device r;
  std::uniform_real_distribution<T> unif;
  std::default_random_engine re(r());
  
  for(int i=0; i<N_VALS; i++) {
    arr[i] = unif(re);
  }
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
void initialize_array(T *arr) {
  std::random_device r;
  std::uniform_int_distribution<T> unif;
  std::default_random_engine re(r());
  
  for(int i=0; i<N_VALS; i++) {
    arr[i] = unif(re);
  }
}

template<typename T>
void run_test() {
  T *h_input = static_cast<T*>(malloc(sizeof(T)*N_VALS));
  initialize_array<T>(h_input); 
  T *d_input = nullptr;
  HIP_CHECK(hipMalloc(&d_input, sizeof(T)*N_VALS));
  HIP_CHECK(hipMemcpy(d_input, h_input, sizeof(T)*N_VALS, hipMemcpyHostToDevice));
  
  dim3 block(N_VALS, 1, 1);
  dim3 grid(divideAndRoundUp(N_VALS, block.x), 1, 1);
  hipLaunchKernelGGL(test_safe_operations_kernel, grid, block, 0, 0, d_input);
  
}

int main(int argc, char **argv) {
  printf("\e[0;97m\n\n\tA program to test the functionality of LIBHIPCXX numerics library "
         "in the context of cuDF\e[0m\n\n");
  sleep(1);
  
  int deviceID;
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDevice(&deviceID));
  HIP_CHECK(hipGetDeviceProperties(&props, deviceID));

  printf("\e[0;97m\t\t(Operations are running on GPU id=%d %s)\e[0m\n", deviceID, props.name);

  printf("\e[0;97m\n\n\tTesting safe arithmetic operations for type unsigned int\e[0m\n\n");
  run_test<unsigned int>();
  HIP_CHECK(hipDeviceSynchronize());
  
  printf("\e[0;97m\n\n\tTesting safe arithmetic operations for type int\e[0m\n\n");
  run_test<int>();
  HIP_CHECK(hipDeviceSynchronize());
}

