// libmxfp6gemm implementation (HIP). Defines the public type-erased entry points and
// instantiates the internal templated launcher for each output type.
#include "mxfp6/gemm.hpp"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "dispatch.hpp"  // mxfp6::detail::dispatch_gemm<OutT>

namespace mxfp6 {

TileChoice choose_tile(int M, int N) {
    constexpr int CU = 256;  // MI350X (gfx950)
    int wg256 = (M / 256) * (N / 256);
    if (wg256 < CU && (M % 128) == 0 && (N % 256) == 0)
        return {128, 256, 2, 4};  // WG-starved small-M: fill CUs
    return {256, 256, 4, 4};      // workhorse: 16-acc sweet spot
}

void gemm(OutType ot, int M, int N, int Kp, const void* dA, const void* dBsh, const uint8_t* dsA,
          const uint8_t* dsB, void* dD, int A_row_bytes, int B_row_bytes) {
    switch (ot) {
        case OutType::F32:
            detail::dispatch_gemm<float>(M, N, Kp, dA, dBsh, dsA, dsB, static_cast<float*>(dD),
                                         A_row_bytes, B_row_bytes);
            break;
        case OutType::F16:
            detail::dispatch_gemm<__half>(M, N, Kp, dA, dBsh, dsA, dsB, static_cast<__half*>(dD),
                                          A_row_bytes, B_row_bytes);
            break;
        case OutType::BF16:
            detail::dispatch_gemm<__hip_bfloat16>(M, N, Kp, dA, dBsh, dsA, dsB,
                                                  static_cast<__hip_bfloat16*>(dD), A_row_bytes,
                                                  B_row_bytes);
            break;
    }
}

}  // namespace mxfp6
