#pragma once

#include <hip/hip_fp16.h>

/// WMMA result fragment layout for wave32.
///
/// D = A × B + C where D is M×N, distributed across 32 lanes × ELEMS elements.
/// Two lanes share each M-row; ELEMS elements per lane cover all N columns
/// via an interleaved pattern.
///
///   outer(lane)    → M-row   = lane >> 1
///   inner(lane, e) → N-col   = (e >> 1) * 4 + (lane & 1) * 2 + (e & 1)
template <int M_, int N_, int ELEMS_>
struct WmmaResultLayout
{
    static constexpr int M     = M_;
    static constexpr int N     = N_;
    static constexpr int ELEMS = ELEMS_;

    __host__ __device__ static int outer(int lane) { return lane >> 1; }

    __host__ __device__ static int inner(int lane, int e)
    {
        return (e >> 1) * 4 + (lane & 1) * 2 + (e & 1);
    }
};

/// Matrix layout for MFMA operations.
///
/// The lane-encoded dimension (outer) and the register-packed dimension (inner) follow the
/// CDNA ISA operand layout:
///   outer(lane) = lane % M  -- the outer-product dimension:
///                              row of A (M-dimension) or column of B/C/D (N-dimension,
///                              when N is passed as the first template argument)
///   inner(lane, idx)        -- the inner-product (K-reduction) dimension for operands;
///                              for result matrices C/D (where the first template arg is N),
///                              this gives the row of C/D (M-dimension)
template <int M, int K, int B = 1, typename T = __half>
struct MatrixLayout
{
    __host__ __device__ static constexpr int items_per_register()
    {
        if constexpr(sizeof(T) == 1)
        {
            return 4;
        }
        else if constexpr(sizeof(T) == 2)
        {
            return 2;
        }
        else if constexpr(sizeof(T) == 4 || sizeof(T) == 8)
        {
            return 1;
        }
        else
        {
            static_assert(sizeof(T) == 0, "Unsupported type size");
        }
    }

    static constexpr int K_L = K / (64 / (M * B));

    // Returns the outer-product dimension index for a given lane:
    // row of A (M-dimension) for operand A; column of B or C/D (N-dimension) for operand B
    // or result matrices (when N is passed as the first template argument).
    __host__ __device__ static int outer(int lane) { return lane % M; }

    // Returns the inner-product (K-reduction) dimension index for a given lane and register index.
    // For result matrices C/D (first template arg = N), this gives the M-row instead.
    __host__ __device__ static int inner(int lane, int idx = 0)
    {
        return idx * items_per_register() + (lane / (M * B)) * K_L;
    }

    // Returns the batch index corresponding to a given lane.
    __host__ __device__ static int batch(int lane) { return (lane / M) % B; }
};
