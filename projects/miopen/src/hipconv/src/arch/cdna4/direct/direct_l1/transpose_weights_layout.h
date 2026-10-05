#pragma once

// Compound-index structures for the transpose_weights grid and block.
//
// blockIdx.x maps to (kq, kw, kh, k16, c64) (c64 fastest); threadIdx.x to
// (c8, k1, c_last) (c_last fastest). The Shape/Strides/Index/Layout pattern
// mirrors weights_layout.h. Grid: c64 x k16 x kh x kw x kq. Workgroup:
// c_last (8/Veclen) x k1 (16) x c8 (4).

#include <hip/hip_runtime.h>

namespace hipconv::cdna4::direct_transpose_weights
{

// ---- Workgroup ----

template <int Kh, int Kw>
struct WorkgroupPars
{
    // K_padded here is one wavegroup's K share (Kwg * Kq), not the full K.
    const int K_padded;
    const int C_padded;
};

template <int Kh, int Kw, int Kwg>
struct GridShape
{
    using Pars = WorkgroupPars<Kh, Kw>;
    const Pars& pars;

    constexpr GridShape(const Pars& pars_in) : pars(pars_in) {}

    __device__ __host__ int c64() const { return pars.C_padded / 64; }

    constexpr int k16() const { return Kwg / 16; }

    constexpr int kh() const { return Kh; }

    constexpr int kw() const { return Kw; }

    __device__ __host__ int kq() const { return pars.K_padded / Kwg; }
};

template <int Kh, int Kw, int Kwg>
struct GridStrides
{
    using Pars  = WorkgroupPars<Kh, Kw>;
    using Shape = GridShape<Kh, Kw, Kwg>;

    const Pars& pars;

    constexpr GridStrides(const Pars& pars_in) : pars(pars_in) {}

    constexpr Shape shape() const { return Shape(pars); }

    constexpr int c64() const { return 1; }

    __device__ __host__ int k16() const { return shape().c64() * c64(); }

    __device__ __host__ int kh() const { return shape().k16() * k16(); }

    __device__ __host__ int kw() const { return shape().kh() * kh(); }

    __device__ __host__ int kq() const { return shape().kw() * kw(); }
};

template <int Kh, int Kw, int Kwg>
struct GridIndex
{
    using Pars    = WorkgroupPars<Kh, Kw>;
    using Shape   = GridShape<Kh, Kw, Kwg>;
    using Strides = GridStrides<Kh, Kw, Kwg>;

    const Pars pars;
    const int offset;

    __device__ __host__ GridIndex(const Pars& pars_in, int offset_in = 0)
        : pars(pars_in)
        , offset(offset_in)
    {
    }

    __device__ __host__ GridIndex(int K_padded, int C_padded, int offset_in = 0)
        : pars{K_padded, C_padded}
        , offset(offset_in)
    {
    }

    __device__ __host__ Strides strides() const { return Strides{pars}; }

    __device__ __host__ Shape shape() const { return Shape(pars); }

    constexpr int extract(int stride, int size) const { return (offset / stride) % size; }

    constexpr int extract(int stride) const { return offset / stride; }

    __device__ __host__ int c64() const { return extract(strides().c64(), shape().c64()); }

    __device__ __host__ int k16() const { return extract(strides().k16(), shape().k16()); }

    __device__ __host__ int kh() const { return extract(strides().kh(), shape().kh()); }

    __device__ __host__ int kw() const { return extract(strides().kw(), shape().kw()); }

    // Outermost: extracted without bound.
    __device__ __host__ int kq() const { return extract(strides().kq()); }

    // Unfold the K-base of the workgroup's 16xC tile within this wavegroup's sub-tensor.
    __device__ __host__ int k() const { return kq() * Kwg + k16() * 16; }

    // Unfold the C-base of the workgroup's Kx64 tile.
    __device__ __host__ int c() const { return c64() * 64; }

    __device__ __host__ int size() const { return shape().kq() * strides().kq(); }

    __device__ __host__ bool is_inbounds() const { return 0 <= offset && offset < size(); }
};

template <int Kh, int Kw, int Kwg>
struct GridLayout
{
    using Pars    = WorkgroupPars<Kh, Kw>;
    using Strides = GridStrides<Kh, Kw, Kwg>;

    const Pars pars;
    const int offset;

    __device__ __host__ GridLayout(const Pars& pars_in, int offset_in = 0)
        : pars(pars_in)
        , offset(offset_in)
    {
    }

    __device__ __host__ GridLayout(int K_padded, int C_padded, int offset_in = 0)
        : pars{K_padded, C_padded}
        , offset(offset_in)
    {
    }

    __device__ __host__ Strides strides() const { return Strides{pars}; }

    __device__ __host__ auto step(int delta) const { return GridLayout(pars, offset + delta); }

    __device__ __host__ auto c64(int x) const { return step(x * strides().c64()); }

    __device__ __host__ auto k16(int x) const { return step(x * strides().k16()); }

    __device__ __host__ auto kh(int x) const { return step(x * strides().kh()); }

    __device__ __host__ auto kw(int x) const { return step(x * strides().kw()); }

    __device__ __host__ auto kq(int x) const { return step(x * strides().kq()); }
};

// ---- Thread ----

template <int Veclen>
struct WorkgroupShape
{
    constexpr int c_last() const { return 8 / Veclen; }

    constexpr int k1() const { return 16; }

    constexpr int c8() const { return 4; }
};

template <int Veclen>
struct WorkgroupStrides
{
    using Shape = WorkgroupShape<Veclen>;

    constexpr Shape shape() const { return Shape{}; }

    constexpr int c_last() const { return 1; }

    constexpr int k1() const { return shape().c_last() * c_last(); }

    constexpr int c8() const { return shape().k1() * k1(); }
};

template <int Veclen>
struct WorkgroupIndex
{
    using Shape   = WorkgroupShape<Veclen>;
    using Strides = WorkgroupStrides<Veclen>;

    const int offset;

    constexpr WorkgroupIndex(int offset_in) : offset(offset_in) {}

    constexpr Strides strides() const { return Strides{}; }

    constexpr Shape shape() const { return Shape{}; }

    constexpr int extract(int stride, int size) const { return (offset / stride) % size; }

    constexpr int c_last() const { return extract(strides().c_last(), shape().c_last()); }

    constexpr int k1() const { return extract(strides().k1(), shape().k1()); }

    constexpr int c8() const { return extract(strides().c8(), shape().c8()); }

    // Unfold the K coordinate this thread targets within the 16-K tile.
    constexpr int k() const { return k1(); }

    // Unfold the C coordinate this thread targets within a 32-C strip.
    constexpr int c() const { return c8() * 8 + c_last() * Veclen; }

    constexpr int size() const { return shape().c8() * strides().c8(); }

    constexpr bool is_inbounds() const { return 0 <= offset && offset < size(); }
};

template <int Veclen>
struct WorkgroupLayout
{
    using Strides = WorkgroupStrides<Veclen>;

    const int offset;

    constexpr WorkgroupLayout(int offset_in = 0) : offset(offset_in) {}

    constexpr Strides strides() const { return Strides{}; }

    constexpr auto step(int delta) const { return WorkgroupLayout(offset + delta); }

    constexpr auto c_last(int x) const { return step(x * strides().c_last()); }

    constexpr auto k1(int x) const { return step(x * strides().k1()); }

    constexpr auto c8(int x) const { return step(x * strides().c8()); }
};

} // namespace hipconv::cdna4::direct_transpose_weights
