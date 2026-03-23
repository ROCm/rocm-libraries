// Compile for a target architecture with:
//   hipcc --cuda-device-only -S --offload-arch=<arch> -O2 simple_adder_0.cpp -o
//   ../<arch>/simple_adder_0.s
#include <hip/hip_runtime.h>

template <int WG_SIZE>
__launch_bounds__(WG_SIZE) __global__
    void adder_kernel(float bias, const double *in0, float *out,
                      const float *in1) {
  unsigned i = WG_SIZE * blockIdx.x + threadIdx.x;
  out[i] = in0[i] + in1[i] + bias;
}

template __global__ void adder_kernel<256>(float, const double *, float *,
                                           const float *);
