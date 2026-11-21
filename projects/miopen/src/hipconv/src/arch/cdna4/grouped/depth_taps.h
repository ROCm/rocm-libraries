#pragma once

#include "hipconv/conv_params.hpp"

#include <hip/hip_runtime.h>

namespace hipconv::cdna4
{

// Depth geometry of a conv3d as one output plane sees it; a conv2d is a single plane.
//
// Output plane od reads virtual input plane od * stride - pad + i * dilation through
// filter tap i. Dgrad runs over dY upsampled by ups with the filter flipped, so a
// virtual plane is real only on a multiple of ups and reads tap kd - 1 - i.
struct DepthParams
{
    int kd       = 1;
    int d_in     = 1;
    int d_out    = 1;
    int pad      = 0;
    int stride   = 1;
    int dilation = 1;
    int ups      = 1;

    static DepthParams of(const ConvParams& par)
    {
        // A conv2d may not have had normalize_unused_axes() zero its depth padding.
        if(par.dims != 3)
            return {};
        if(par.direction != Direction::Dgrad)
            return {par.kd, par.d, par.e, par.pad_d, par.stride_d, par.dilation_d, 1};
        return {par.kd,
                par.e,
                par.d,
                (par.kd - 1) * par.dilation_d - par.pad_d,
                1,
                par.dilation_d,
                par.stride_d};
    }
};

// The smallest tap capacity KD of a fprop/dgrad config that holds this layer's depth
// taps: 1 for a conv2d, 0 when no config does.
inline int depth_tap_capacity(const ConvParams& par)
{
    if(par.dims != 3)
        return 1;
    return par.kd <= 3 ? 3 : par.kd <= 5 ? 5 : par.kd <= 7 ? 7 : 0;
}

// Whether output plane od reads an input plane through depth tap i, and which (u).
__device__ __forceinline__ bool depth_tap_input(int od, int i, const DepthParams& dp, int& u)
{
    const int v = od * dp.stride - dp.pad + i * dp.dilation;
    // Scalar division is slow and ups is almost always 1 or 2.
    u = dp.ups == 1 ? v : dp.ups == 2 ? v >> 1 : v / dp.ups;
    return i < dp.kd && v >= 0 && u * dp.ups == v && u < dp.d_in;
}

// The depth taps that reach one output plane, live ones packed to the front.
template <int KD>
struct DepthTaps
{
    int count     = 0;
    int plane[KD] = {};
    int tap[KD]   = {};
};

template <int KD, bool flip>
__device__ __forceinline__ DepthTaps<KD> live_depth_taps(int od, const DepthParams& dp)
{
    DepthTaps<KD> taps;
    if constexpr(KD == 1)
    {
        taps.count = 1;
        return taps;
    }
#pragma unroll
    for(int i = 0; i < KD; i++)
    {
        int u;
        const bool live = depth_tap_input(od, i, dp, u);
        // Constant indices keep the arrays in registers.
#pragma unroll
        for(int j = 0; j < KD; j++)
        {
            if(live && taps.count == j)
            {
                taps.plane[j] = u;
                taps.tap[j]   = flip ? dp.kd - 1 - i : i;
            }
        }
        taps.count += live ? 1 : 0;
    }
    return taps;
}

// HIP's limit on grid z, where the grouped kernels put batch and depth.
constexpr unsigned max_grid_z = 65535;

// The delta plane and depth tap a wgrad block owns.
//
// blockIdx.z enumerates (delta plane, depth tap); a conv2d is one plane per image, so z is
// the plane and the tap is 0. A block whose tap falls in the depth padding is not live.
struct WgradDepthBlock
{
    int tap         = 0;
    int delta_plane = 0; // Of the N * d_out delta planes.
    int input_plane = 0; // Of the N * d_in input planes.
    bool live       = true;
};

__device__ __forceinline__ WgradDepthBlock wgrad_depth_block(int z, const DepthParams& dp)
{
    if(dp.kd == 1 && dp.d_in == 1 && dp.d_out == 1 && dp.pad == 0)
        return {0, z, z, true};
    WgradDepthBlock b;
    b.tap         = z % dp.kd;
    b.delta_plane = z / dp.kd;
    const int d   = b.delta_plane % dp.d_out * dp.stride - dp.pad + b.tap * dp.dilation;
    b.live        = d >= 0 && d < dp.d_in;
    b.input_plane = b.delta_plane / dp.d_out * dp.d_in + d;
    return b;
}

// Index in a [k][kd][TAP_SIZE] dW of element i of one depth tap's [k][TAP_SIZE] tile.
template <int TAP_SIZE>
__device__ __forceinline__ int dw_tap_index(int i, int tap, int kd)
{
    return (i / TAP_SIZE * kd + tap) * TAP_SIZE + i % TAP_SIZE;
}

// Index of the j-th uint4 of one depth tap's weight slice within a [k][kd][tap_uint4]
// tensor.
template <int TAP_UINT4, int KD>
__device__ __forceinline__ int weight_tap_offset(int j, int tap, int kd)
{
    if constexpr(KD == 1)
        return j;
    else
        return (j / TAP_UINT4 * kd + tap) * TAP_UINT4 + j % TAP_UINT4;
}

} // namespace hipconv::cdna4
