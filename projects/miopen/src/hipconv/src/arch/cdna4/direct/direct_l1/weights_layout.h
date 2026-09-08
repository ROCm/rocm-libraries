#pragma once

// Tensor layout of the DirectL1 weights workspace.
//
// The workspace holds waves_k shape-identical sub-tensors back-to-back, one per
// K-partition wave, so the partition is the outermost axis (stride = one
// sub-tensor). wave_group(i) addresses sub-tensor i; the count lives in
// Config::waves_k, not here. One partition's slice is:
//
//   (K_wavegroup / Kwg) × (C / 64) × Kw × (C / 32 % 2) × Kh × (K_wavegroup / 16 % (Kwg / 16))
//   × [(C / 8 % 4) × (K % 16) × (C % 8)] × f16
//
// where [ ] is a 16 × 32 f16 matrix in native MFMA layout (matrix_layout.h). The
// Kw × (C/32%2) factors are the Kw*2 main-loop steps; Kh × (Kwg/16) is the K(16) ×
// C(32) matrices per step; C/64 the channel chunks; K_wavegroup/Kwg the K workgroup.
// Kwg in {32,48,64}; K_wavegroup = Kwg × Kq. Per-thread the inner (C%8) splits into
// (8/Veclen) × Veclen, the trailing Veclen being one offset step.
#include <hip/hip_runtime.h>

#include "channels_last_layouts.h"

namespace hipconv::cdna4
{

template <int Kh, int Kw, int Kwg, int Veclen = 8>
struct WeightsShape
{
    using Pars = WeightPars<Kh, Kw>;
    const Pars& pars;

    constexpr WeightsShape(const Pars& pars_in) : pars(pars_in) {}

    constexpr int c_last() const { return 8 / Veclen; }

    constexpr int k1() const { return 16; }

    constexpr int c8() const { return 4; }

    constexpr int k16() const { return Kwg / 16; }

    constexpr int kh() const { return Kh; }

    constexpr int c32() const { return 2; }

    constexpr int kw() const { return Kw; }

    __device__ __host__ int c64() const { return pars.C / 64; }

    __device__ __host__ int kq() const { return pars.K / Kwg; }

    // Nominal partition extent only (real count is Config::waves_k).
    constexpr int wave_group() const { return 2; }
};

// Define the strides of one wavegroup's DirectL1 sub-tensor.
template <int Kh, int Kw, int Kwg, int Veclen = 8>
struct WeightsStrides
{
    using Pars  = WeightPars<Kh, Kw>;
    using Shape = WeightsShape<Kh, Kw, Kwg, Veclen>;

    const Pars& pars;

    constexpr WeightsStrides(const Pars& pars_in) : pars(pars_in) {}

    constexpr Shape shape() const { return Shape(pars); }

    constexpr int c_last() const { return 1; }

    constexpr int k1() const { return shape().c_last(); }

    constexpr int c8() const { return shape().k1() * k1(); }

    constexpr int k16() const { return shape().c8() * c8(); }

    constexpr int kh() const { return shape().k16() * k16(); }

    constexpr int c32() const { return shape().kh() * kh(); }

    constexpr int kw() const { return shape().c32() * c32(); }

    constexpr int c64() const { return shape().kw() * kw(); }

    __device__ __host__ int kq() const { return shape().c64() * c64(); }

    // One sub-tensor's element count: the stride between adjacent partitions.
    __device__ __host__ int wave_group() const { return shape().kq() * kq(); }
};


// Compound-index for one wavegroup's DirectL1 sub-tensor (inverse of WeightsLayout).
//
// Maps an offset to its individual dimension values. Kwg in {32,48,64}; Veclen is
// f16 elements per offset step (default 8). K must be divisible by Kwg, C by 64.
template <int Kh, int Kw, int Kwg, int Veclen = 8>
struct WeightsIndex
{
    using Pars    = WeightPars<Kh, Kw>;
    using Shape   = WeightsShape<Kh, Kw, Kwg, Veclen>;
    using Strides = WeightsStrides<Kh, Kw, Kwg, Veclen>;
    const Pars pars;
    const int offset;

    __device__ __host__ WeightsIndex(const Pars& pars_in, int offset_in = 0)
        : pars(pars_in)
        , offset(offset_in)
    {
    }

    __device__ __host__ WeightsIndex(int K, int C, int offset_in = 0)
        : pars{K, C}
        , offset(offset_in)
    {
    }

    __device__ __host__ Strides strides() const { return Strides{pars}; }

    __device__ __host__ Shape shape() const { return Shape(pars); }

    constexpr int extract(int stride, int size) const { return (offset / stride) % size; }

    constexpr int extract(int stride) const { return offset / stride; }

    __device__ __host__ int c_last() const { return extract(strides().c_last(), shape().c_last()); }

    __device__ __host__ int k1() const { return extract(strides().k1(), shape().k1()); }

    __device__ __host__ int c8() const { return extract(strides().c8(), shape().c8()); }

    __device__ __host__ int k16() const { return extract(strides().k16(), shape().k16()); }

    __device__ __host__ int kh() const { return extract(strides().kh(), shape().kh()); }

    __device__ __host__ int c32() const { return extract(strides().c32(), shape().c32()); }

    __device__ __host__ int kw() const { return extract(strides().kw(), shape().kw()); }

    __device__ __host__ int c64() const { return extract(strides().c64(), shape().c64()); }

    // This will go out of bounds by design for large offsets.
    __device__ __host__ int kq() const { return extract(strides().kq()); }

    // Return the unfolded k-dimension within this wavegroup's sub-tensor.
    __device__ __host__ int k() const { return kq() * Kwg + k16() * 16 + k1(); }

    // Return the unfolded c-dimension.
    __device__ __host__ int c() const
    {
        return c64() * 64 + c32() * 32 + c8() * 8 + c_last() * Veclen;
    }

    __device__ __host__ int size() const { return (pars.K / Kwg) * strides().kq(); }

    __device__ __host__ bool is_inbounds() const { return 0 <= offset && offset < size(); }
};

// Maps layout dimensions to an address offset for one wavegroup's sub-tensor.
//
// e.g. WeightsLayout<3, 3, 32>(64, 32) for a 3x3 filter, Kwg=32, K=64, C=32. Kwg in
// {32,48,64}; K divisible by Kwg, C by 64. offset is in Veclen f16 units (default
// Veclen 8); one MFMA 16x32 tile is 512/Veclen units = 1024 bytes.
template <int Kh, int Kw, int Kwg, int Veclen = 8>
struct WeightsLayout
{
    using Pars    = WeightPars<Kh, Kw>;
    using Strides = WeightsStrides<Kh, Kw, Kwg, Veclen>;

    const Pars pars;
    const int offset;


    __device__ __host__ WeightsLayout(const Pars& pars_in, int offset_in = 0)
        : pars(pars_in)
        , offset(offset_in)
    {
    }

    __device__ __host__ WeightsLayout(int K, int C, int offset_in = 0)
        : pars{K, C}
        , offset(offset_in)
    {
    }

    __device__ __host__ Strides strides() const { return Strides{pars}; }

    __device__ __host__ bool is_valid() const { return pars.K % Kwg == 0 && pars.C % 64 == 0; }

    __device__ __host__ auto step(int delta) const { return WeightsLayout(pars, offset + delta); }

    __device__ __host__ auto wave_group(int x) const { return step(x * strides().wave_group()); }

    __device__ __host__ auto kq(int x) const { return step(x * strides().kq()); }

    __device__ __host__ auto c64(int x) const { return step(x * strides().c64()); }

    __device__ __host__ auto kw(int x) const { return step(x * strides().kw()); }

    __device__ __host__ auto c32(int x) const { return step(x * strides().c32()); }

    __device__ __host__ auto kh(int x) const { return step(x * strides().kh()); }

    __device__ __host__ auto k16(int x) const { return step(x * strides().k16()); }

    __device__ __host__ auto c8(int x) const { return step(x * strides().c8()); }

    __device__ __host__ auto k1(int x) const { return step(x * strides().k1()); }

    __device__ __host__ auto c_last(int x) const { return step(x * strides().c_last()); }

    __device__ __host__ int byte_offset() const { return offset * Veclen * 2; }
};

} // namespace hipconv::cdna4
