#pragma once

#include <hip/hip_runtime.h>

#include "layer_pars.h"

namespace hipconv::cdna4
{

// Strides of the channels-last input layout (N x H x W x C, C innermost; elements).
struct ChannelsLastInputStrides
{
    using Pars = InputPars;
    const Pars& pars;

    constexpr ChannelsLastInputStrides(const Pars& pars_in) : pars(pars_in) {}

    constexpr int c() const { return 1; }

    __device__ __host__ int w() const { return pars.C * c(); }

    __device__ __host__ int h() const { return pars.W * w(); }

    __device__ __host__ int n() const { return pars.H * h(); }
};

// Maps channels-last input dimensions (N x H x W x C, C innermost) to an offset.
//
// e.g. ChannelsLastInputLayout(4, 32, 64, 16) for N(4) x H(32) x W(64) x C(16).
// offset is in elements.
struct ChannelsLastInputLayout
{
    using Pars    = InputPars;
    using Strides = ChannelsLastInputStrides;

    const Pars pars;
    const int offset;

    __device__ __host__ ChannelsLastInputLayout(const Pars& pars_in, int offset_in = 0)
        : pars(pars_in)
        , offset(offset_in)
    {
    }

    __device__ __host__ ChannelsLastInputLayout(int N, int H, int W, int C, int offset_in = 0)
        : pars{N, H, W, C}
        , offset(offset_in)
    {
    }

    __device__ __host__ Strides strides() const { return Strides{pars}; }

    __device__ __host__ auto step(int delta) const
    {
        return ChannelsLastInputLayout(pars, offset + delta);
    }

    __device__ __host__ auto n(int x) const { return step(x * strides().n()); }

    __device__ __host__ auto h(int x) const { return step(x * strides().h()); }

    __device__ __host__ auto w(int x) const { return step(x * strides().w()); }

    __device__ __host__ auto c(int x) const { return step(x * strides().c()); }

    __device__ __host__ int size() const { return pars.N * strides().n(); }

    __device__ __host__ bool is_inbounds() const { return 0 <= offset && offset < size(); }
};


// Strides of the channels-last weights layout (K x Kh x Kw x C, C innermost; elements).
template <int Kh, int Kw>
struct ChannelsLastWeightsStrides
{
    using Pars = WeightPars<Kh, Kw>;
    const Pars& pars;

    constexpr ChannelsLastWeightsStrides(const Pars& pars_in) : pars(pars_in) {}

    constexpr int c() const { return 1; }

    __device__ __host__ int kw() const { return pars.C * c(); }

    __device__ __host__ int kh() const { return Kw * kw(); }

    __device__ __host__ int k() const { return Kh * kh(); }
};

// Maps channels-last weights dimensions (K x Kh x Kw x C, C innermost) to an offset.
//
// e.g. ChannelsLastWeightsLayout<3, 3>(64, 32) for a 3x3 filter, K=64, C=32.
// offset is in elements.
template <int Kh, int Kw>
struct ChannelsLastWeightsLayout
{
    using Pars    = WeightPars<Kh, Kw>;
    using Strides = ChannelsLastWeightsStrides<Kh, Kw>;

    const Pars pars;
    const int offset;

    __device__ __host__ ChannelsLastWeightsLayout(const Pars& pars_in, int offset_in = 0)
        : pars(pars_in)
        , offset(offset_in)
    {
    }

    __device__ __host__ ChannelsLastWeightsLayout(int K, int C, int offset_in = 0)
        : pars{K, C}
        , offset(offset_in)
    {
    }

    __device__ __host__ Strides strides() const { return Strides{pars}; }

    __device__ __host__ auto step(int delta) const
    {
        return ChannelsLastWeightsLayout(pars, offset + delta);
    }

    __device__ __host__ auto k(int x) const { return step(x * strides().k()); }

    __device__ __host__ auto kh(int x) const { return step(x * strides().kh()); }

    __device__ __host__ auto kw(int x) const { return step(x * strides().kw()); }

    __device__ __host__ auto c(int x) const { return step(x * strides().c()); }

    __device__ __host__ int size() const { return pars.K * strides().k(); }

    __device__ __host__ bool is_inbounds() const { return 0 <= offset && offset < size(); }
};

} // namespace hipconv::cdna4
