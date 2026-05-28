// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Implementation of the header-only HIP/GPU mxDataGenerator backend.
// Included from `DataGeneratorGPU.hpp`. Do NOT include directly.

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

// When including hip/hip_ext_ocp.h __half* types are not defined.
// This is a workaround for the include bug in HIP headers, please remove when fixed.
#include <hip/hip_fp16.h>

// Provides __amd_cvt_floatx{2,8,32}_to_fp{4,6,8}*_scale and __amd_scale_t.
// Hits the gfx950 hardware MX convert instructions where available
// (v_cvt_scalef32_pk_fp4_f32 / pk32_f32_fp6 / pk_fp8_f32 etc.); falls back to
// the canonical fcbx software path on other architectures. Same wrapper
// hipblaslt's hipblaslt_float4 / _float6 / _float8 types use, so generator
// output is bit-identical to what those types produce.
#include <hip/hip_ext_ocp.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataTypeInfo.hpp"
#include "ocp_e2m1_mxfp4.hpp"
#include "ocp_e2m3_mxfp6.hpp"
#include "ocp_e3m2_mxfp6.hpp"
#include "ocp_e4m3_mxfp8.hpp"
#include "ocp_e5m2_mxfp8.hpp"

namespace DGen
{
    namespace gpu_detail
    {
        // ----------------------------------------------------------------------
        // HIP error helpers
        // ----------------------------------------------------------------------
        inline void checkHipStatus(hipError_t status, char const* what)
        {
            if(status != hipSuccess)
            {
                throw std::runtime_error(std::string("DataGeneratorGPU: ") + what + ": "
                                         + hipGetErrorString(status));
            }
        }

        // hipLaunchKernelGGL takes `dim3` (uint32 components); if our
        // computed gridDimX exceeds 2^32-1 the static_cast<unsigned> at the
        // launch site silently truncates and the kernel covers only part of
        // the work. Call this from every host-side launch site before the
        // cast so we fail loudly instead. `what` is the caller name used in
        // the error message.
        inline void checkGridDimX(size_t gridDimX, char const* what)
        {
            if(gridDimX > static_cast<size_t>(std::numeric_limits<unsigned>::max()))
                throw std::invalid_argument(
                    std::string("DataGeneratorGPU::") + what
                    + ": gridDim.x exceeds the 2^32-1 thread-block limit; "
                      "split the launch or use a 2D grid");
        }

        // Common threads-per-block for every kernel launch in this backend.
        // 256 is a good fit for the work pattern (one thread per MX scale
        // block / one thread per output byte), gives full 4-wave occupancy
        // on every supported arch, and keeps launch arithmetic uniform
        // across the generation + pre-swizzle kernels.
        constexpr int kThreadsPerBlock = 256;

// Macros aren't namespaced even when defined inside a namespace block. This
// detail name is `#undef`-ed at the bottom of this file so it doesn't leak
// to includers; trailing underscore signals "private to this TU".
#define DGEN_DETAIL_CHECK_HIP_(call) \
    ::DGen::gpu_detail::checkHipStatus((call), #call)

        // RAII guard for a hipMalloc'd device buffer. Releases on
        // destruction unless `release()` has been called to transfer
        // ownership. Used by the pre-swizzle paths so a launch failure
        // after hipMalloc doesn't leak the temporaries.
        class DeviceBuffer
        {
        public:
            DeviceBuffer() = default;
            explicit DeviceBuffer(size_t bytes)
            {
                DGEN_DETAIL_CHECK_HIP_(hipMalloc(&m_ptr, bytes));
            }
            ~DeviceBuffer()
            {
                if(m_ptr)
                    (void)hipFree(m_ptr);
            }
            DeviceBuffer(DeviceBuffer const&)            = delete;
            DeviceBuffer& operator=(DeviceBuffer const&) = delete;
            DeviceBuffer(DeviceBuffer&& other) noexcept
                : m_ptr(other.m_ptr)
            {
                other.m_ptr = nullptr;
            }
            DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
            {
                if(this != &other)
                {
                    if(m_ptr)
                        (void)hipFree(m_ptr);
                    m_ptr       = other.m_ptr;
                    other.m_ptr = nullptr;
                }
                return *this;
            }

            uint8_t* get() const noexcept
            {
                return m_ptr;
            }
            uint8_t* release() noexcept
            {
                uint8_t* p = m_ptr;
                m_ptr      = nullptr;
                return p;
            }

        private:
            uint8_t* m_ptr = nullptr;
        };

        // ----------------------------------------------------------------------
        // Device-side init mode tags
        //
        // We can't pass the host-side `std::variant<...>` to a kernel, so we
        // mirror the variant alternatives as a plain enum and dispatch on the
        // host before launching the kernel.
        // ----------------------------------------------------------------------
        enum class DeviceInitMode : int
        {
            Bounded                = 0,
            BoundedAlternatingSign = 1,
            Unbounded              = 2,
            Identity               = 3,
            Ones                   = 4,
            Zeros                  = 5,
            Sequential             = 6,
            RowIndex               = 7,
            ColIndex               = 8,
            Checkerboard           = 9,
            ScaledDiagonal         = 10,
            TrigonometricFromFloat = 11,
            NormalFromFloat        = 12,
            // Constant element value carried by `cfg.constValue`. Backs
            // Twos / NegOnes / MaxVals on the device: every element is the
            // same float, then the standard quantize-derive-scale pipeline
            // produces the right dequantised value (the byte pattern is
            // re-encoded relative to the CPU's "literal data byte +
            // scale=1.0" form, but the dequant float matches).
            ConstantValue          = 13,
            // Uniform-int sampler in [randIntLo, randIntHi]. As with the
            // constant-fill modes, the per-block scale derivation re-encodes
            // the byte pattern relative to the CPU's "literal int +
            // scale=1.0" form; the dequantised value is still an integer in
            // the requested range (saturation behaviour can differ from
            // CPU when |hi| exceeds the dtype's max normal -- both paths
            // satisfy the in-range invariant the mode promises).
            RandInt                = 14,
        };

        struct DeviceInitConfig
        {
            DeviceInitMode mode       = DeviceInitMode::Bounded;
            float          minVal     = -1.0f;
            float          maxVal     = 1.0f;
            float          mean       = 0.0f;
            float          stdDev     = 1.0f;
            // `dim0` is the size of the fastest-varying dimension (row count
            // for column-major matrices). Used by RowIndex/ColIndex/Identity.
            // `dim1` is the next dimension (column count). Used by Sequential
            // to compute the same row-major linearisation the CPU does.
            uint64_t       dim0       = 0;
            uint64_t       dim1       = 0;
            // Used by DeviceInitMode::ConstantValue (Twos / NegOnes /
            // MaxVals). Filled in by makeConfig<DTYPE> so MaxVals picks up
            // the dtype-specific max normal.
            float          constValue = 0.0f;
            // Used by DeviceInitMode::RandInt. Inclusive range for the
            // device PRNG sampler. A degenerate range (lo == hi) collapses
            // to a constant.
            int            randIntLo  = 0;
            int            randIntHi  = 0;
        };

        // Forward-declare GpuTraits so makeConfig<DTYPE> below can reference
        // GpuTraits<DTYPE>::maxNormal in the MaxVals branch. The full
        // specialisations live further down in this file (and are visible
        // at every instantiation site, which is `generateInto<DTYPE>` at
        // the bottom).
        template <typename DTYPE>
        struct GpuTraits;

        // Templated on DTYPE because MaxVals carries no value of its own -
        // its constant element value is the per-DTYPE max normal, which we
        // pull from GpuTraits<DTYPE>::maxNormal.
        template <typename DTYPE>
        inline DeviceInitConfig makeConfig(DataGeneratorOptions const& options,
                                           std::vector<index_t> const& sizes)
        {
            DeviceInitConfig cfg;
            cfg.minVal = static_cast<float>(options.min);
            cfg.maxVal = static_cast<float>(options.max);
            cfg.dim0   = sizes.empty() ? 0 : static_cast<uint64_t>(sizes[0]);
            cfg.dim1   = sizes.size() < 2 ? 0 : static_cast<uint64_t>(sizes[1]);

            std::visit(
                [&](auto const& mode) {
                    using T = std::decay_t<decltype(mode)>;
                    if constexpr(std::is_same_v<T, Bounded>)
                        cfg.mode = DeviceInitMode::Bounded;
                    else if constexpr(std::is_same_v<T, BoundedAlternatingSign>)
                        cfg.mode = DeviceInitMode::BoundedAlternatingSign;
                    else if constexpr(std::is_same_v<T, Unbounded>)
                        cfg.mode = DeviceInitMode::Unbounded;
                    else if constexpr(std::is_same_v<T, Identity>)
                        cfg.mode = DeviceInitMode::Identity;
                    else if constexpr(std::is_same_v<T, Ones>)
                        cfg.mode = DeviceInitMode::Ones;
                    else if constexpr(std::is_same_v<T, Zeros>)
                        cfg.mode = DeviceInitMode::Zeros;
                    else if constexpr(std::is_same_v<T, Sequential>)
                        cfg.mode = DeviceInitMode::Sequential;
                    else if constexpr(std::is_same_v<T, RowIndex>)
                        cfg.mode = DeviceInitMode::RowIndex;
                    else if constexpr(std::is_same_v<T, ColIndex>)
                        cfg.mode = DeviceInitMode::ColIndex;
                    else if constexpr(std::is_same_v<T, Checkerboard>)
                        cfg.mode = DeviceInitMode::Checkerboard;
                    else if constexpr(std::is_same_v<T, ScaledDiagonal>)
                        cfg.mode = DeviceInitMode::ScaledDiagonal;
                    else if constexpr(std::is_same_v<T, TrigonometricFromFloat>)
                        cfg.mode = DeviceInitMode::TrigonometricFromFloat;
                    else if constexpr(std::is_same_v<T, NormalFromFloat>)
                    {
                        cfg.mode   = DeviceInitMode::NormalFromFloat;
                        cfg.mean   = static_cast<float>(mode.mean);
                        cfg.stdDev = static_cast<float>(mode.std_dev);
                    }
                    else if constexpr(std::is_same_v<T, Twos>)
                    {
                        cfg.mode       = DeviceInitMode::ConstantValue;
                        cfg.constValue = 2.0f;
                    }
                    else if constexpr(std::is_same_v<T, NegOnes>)
                    {
                        cfg.mode       = DeviceInitMode::ConstantValue;
                        cfg.constValue = -1.0f;
                    }
                    else if constexpr(std::is_same_v<T, MaxVals>)
                    {
                        // Use the dtype's max normal so the per-block
                        // scale derivation picks scaleExp=0 (scale=1.0)
                        // and the data byte encodes the max-normal
                        // pattern - same dequantised value the CPU's
                        // setDataMax(positive=true, subNormal=false)
                        // produces.
                        cfg.mode       = DeviceInitMode::ConstantValue;
                        cfg.constValue = GpuTraits<DTYPE>::maxNormal;
                    }
                    else if constexpr(std::is_same_v<T, RandInt>)
                    {
                        // Mirror the CPU's invalid-argument check; defer
                        // it to here (rather than the kernel) so callers
                        // see the same exception type they would on CPU.
                        if(mode.lo > mode.hi)
                            throw std::invalid_argument(
                                "DataGeneratorGPU::makeConfig: RandInt lo "
                                "must be <= hi");
                        cfg.mode      = DeviceInitMode::RandInt;
                        cfg.randIntLo = mode.lo;
                        cfg.randIntHi = mode.hi;
                    }
                    else
                    {
                        // Only DenormMins, DenormMaxs, NaNs and Infs reach
                        // this branch, and they're all routed through the
                        // host fallback in `generateInto` so we never
                        // actually instantiate makeConfig for them.
                        //
                        // Compile-time enforcement isn't an option here:
                        // std::visit deduces the visitor's return type via
                        // std::invoke_result, which instantiates the lambda
                        // body for every variant alternative regardless of
                        // `if constexpr`. A static_assert in this branch
                        // therefore fires for the host-fallback types too.
                        //
                        // Runtime throw instead - if a new MXInitMethod
                        // alternative is added and someone forgets to
                        // extend either requiresHostFallback (so it stays
                        // device-side) or this visitor (so it gets a real
                        // mapping), we hit this on the first generate
                        // rather than silently behaving as Bounded[-1, 1].
                        throw std::logic_error(
                            std::string("DataGeneratorGPU::makeConfig: unhandled "
                                        "MXInitMethod alternative '")
                            + mode.toString()
                            + "'; either add it to gpu_detail::requiresHostFallback "
                              "or extend this visitor.");
                    }
                },
                options.initMode);

            return cfg;
        }

        // The on-device pipeline quantises a float-per-element through the
        // block-max scale derivation. That's fine for any mode whose
        // contract is the *dequantised float value* (Twos, NegOnes, MaxVals,
        // RandInt all flow through ConstantValue/RandInt device modes
        // above), but it can't reproduce the *byte patterns* these modes
        // require:
        //   * DenormMins / DenormMaxs need a specific subnormal data-byte
        //     encoding; the float pipeline can renormalise the encoding
        //     into a normal one with the same numeric value.
        //   * NaNs / Infs need NaN/Inf in *both* the data and scale bytes;
        //     the scale-byte derivation is undefined for NaN inputs and
        //     `__amd_cvt_*_scale` doesn't expose a way to pick a specific
        //     NaN payload or the Inf encoding.
        // Route those four through the host CPU `DataGenerator<DTYPE>` and
        // then hipMemcpy the bytes to device.
        inline bool requiresHostFallback(DataInitMode const& initMode)
        {
            return std::visit(
                [](auto const& mode) -> bool {
                    using T = std::decay_t<decltype(mode)>;
                    return std::is_same_v<T, DenormMins> || std::is_same_v<T, DenormMaxs>
                           || std::is_same_v<T, NaNs> || std::is_same_v<T, Infs>;
                },
                initMode);
        }

        // ----------------------------------------------------------------------
        // Per-DTYPE compile-time traits
        //
        // Centralises the dtype-dependent constants used by the generic
        // device-side quantizer so we don't have to specialise every kernel.
        // (`GpuTraits` itself is forward-declared above so the makeConfig
        // template can refer to it; the specialisations below provide the
        // actual constants.)
        // ----------------------------------------------------------------------

        template <>
        struct GpuTraits<ocp_e2m1_mxfp4>
        {
            static constexpr int       signBits     = 1;
            static constexpr int       expBits      = 2;
            static constexpr int       mantBits     = 1;
            static constexpr int       bias         = 1;
            static constexpr float     maxNormal    = 6.0f;
            static constexpr float     minSubnormal = 0.5f;
            static constexpr int       bitsPerElem  = 4;
            static constexpr ScaleType scaleKind    = ScaleType::E8M0;
        };

        template <>
        struct GpuTraits<ocp_e2m1_mxfp4_e4m3>
        {
            static constexpr int       signBits     = 1;
            static constexpr int       expBits      = 2;
            static constexpr int       mantBits     = 1;
            static constexpr int       bias         = 1;
            static constexpr float     maxNormal    = 6.0f;
            static constexpr float     minSubnormal = 0.5f;
            static constexpr int       bitsPerElem  = 4;
            static constexpr ScaleType scaleKind    = ScaleType::E4M3;
        };

        template <>
        struct GpuTraits<ocp_e2m1_mxfp4_e5m3>
        {
            static constexpr int       signBits     = 1;
            static constexpr int       expBits      = 2;
            static constexpr int       mantBits     = 1;
            static constexpr int       bias         = 1;
            static constexpr float     maxNormal    = 6.0f;
            static constexpr float     minSubnormal = 0.5f;
            static constexpr int       bitsPerElem  = 4;
            static constexpr ScaleType scaleKind    = ScaleType::E5M3;
        };

        template <>
        struct GpuTraits<ocp_e2m3_mxfp6>
        {
            static constexpr int       signBits     = 1;
            static constexpr int       expBits      = 2;
            static constexpr int       mantBits     = 3;
            static constexpr int       bias         = 1;
            static constexpr float     maxNormal    = 7.5f;
            static constexpr float     minSubnormal = 0.125f;
            static constexpr int       bitsPerElem  = 6;
            static constexpr ScaleType scaleKind    = ScaleType::E8M0;
        };

        template <>
        struct GpuTraits<ocp_e3m2_mxfp6>
        {
            static constexpr int       signBits     = 1;
            static constexpr int       expBits      = 3;
            static constexpr int       mantBits     = 2;
            static constexpr int       bias         = 3;
            static constexpr float     maxNormal    = 28.0f;
            static constexpr float     minSubnormal = 0.0625f;
            static constexpr int       bitsPerElem  = 6;
            static constexpr ScaleType scaleKind    = ScaleType::E8M0;
        };

        template <>
        struct GpuTraits<ocp_e4m3_mxfp8>
        {
            static constexpr int       signBits     = 1;
            static constexpr int       expBits      = 4;
            static constexpr int       mantBits     = 3;
            static constexpr int       bias         = 7;
            static constexpr float     maxNormal    = 448.0f;
            static constexpr float     minSubnormal = 0.001953125f;
            static constexpr int       bitsPerElem  = 8;
            static constexpr ScaleType scaleKind    = ScaleType::E8M0;
        };

        template <>
        struct GpuTraits<ocp_e5m2_mxfp8>
        {
            static constexpr int       signBits     = 1;
            static constexpr int       expBits      = 5;
            static constexpr int       mantBits     = 2;
            static constexpr int       bias         = 15;
            static constexpr float     maxNormal    = 57344.0f;
            static constexpr float     minSubnormal = 1.52587890625e-05f;
            static constexpr int       bitsPerElem  = 8;
            static constexpr ScaleType scaleKind    = ScaleType::E8M0;
        };

        // ----------------------------------------------------------------------
        // Device PRNG (xorshift32 seeded per element)
        //
        // The seeding pattern intentionally mirrors the host CPU `m_seed + tid`
        // convention: `(seed, idx)` -> a 32-bit state with reasonable bit
        // diffusion. Sufficient for statistical tests; not cryptographic.
        // ----------------------------------------------------------------------
        __device__ __forceinline__ uint32_t seedMix(uint32_t seed, uint64_t idx)
        {
            uint32_t s = seed ^ static_cast<uint32_t>(idx * 2654435761u)
                         ^ static_cast<uint32_t>((idx >> 32) * 40503u);
            // A few xorshift rounds to spread out bits before sampling.
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            if(s == 0)
                s = 0x9E3779B9u;
            return s;
        }

        __device__ __forceinline__ uint32_t xorshift32(uint32_t& s)
        {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            return s;
        }

        __device__ __forceinline__ float prngFloat01(uint32_t& s)
        {
            // 24-bit mantissa worth of randomness, scaled to [0, 1).
            uint32_t r = xorshift32(s) >> 8;
            return static_cast<float>(r) * (1.0f / 16777216.0f);
        }

        // ----------------------------------------------------------------------
        // Device-side raw value generation per init mode
        //
        // `dataIdx` is the linear element index (0 ... totalElements-1).
        // For 2D matrices column-major, row = dataIdx % dim0, col = dataIdx / dim0.
        // ----------------------------------------------------------------------
        __device__ __forceinline__ float
            genValue(DeviceInitMode mode, uint32_t seed, uint64_t dataIdx, DeviceInitConfig const& cfg)
        {
            uint64_t const dim0 = cfg.dim0 ? cfg.dim0 : 1;
            uint64_t const row  = dataIdx % dim0;
            uint64_t const col  = dataIdx / dim0;

            switch(mode)
            {
            case DeviceInitMode::Zeros:
                return 0.0f;
            case DeviceInitMode::Ones:
                return 1.0f;
            case DeviceInitMode::Identity:
                return (row == col) ? 1.0f : 0.0f;
            case DeviceInitMode::Sequential:
            {
                // Match CPU `generate_data_sequential`: value at storage
                // position (row, col) is `(row * cols + col) % 256`.
                // `cfg.dim1` is plumbed for exactly this purpose; falls back
                // to 1 for the (uncommon) 1-D case so dataIdx*1 still
                // produces a sensible monotone sequence.
                uint64_t const cols = cfg.dim1 ? cfg.dim1 : 1;
                return static_cast<float>((row * cols + col) % 256u);
            }
            case DeviceInitMode::RowIndex:
                // Match CPU `generate_data_row_index`: row index modulo 256
                // (CPU stores the value through `satConvertToType`, which
                // wraps via the `% 256` it does first).
                return static_cast<float>(row % 256u);
            case DeviceInitMode::ColIndex:
                // Match CPU `generate_data_col_index`: col index modulo 256.
                return static_cast<float>(col % 256u);
            case DeviceInitMode::Checkerboard:
                // Match CPU `generate_data_checkerboard`: `(row + col) % 2`,
                // 1.0 on the even-parity squares, 0.0 elsewhere. (The old
                // GPU implementation used {-1, 1}, which produced a valid
                // checkerboard but not the same one.)
                return ((row + col) & 1ull) ? 0.0f : 1.0f;
            case DeviceInitMode::ScaledDiagonal:
                return (row == col) ? static_cast<float>(row + 1) : 0.0f;
            case DeviceInitMode::TrigonometricFromFloat:
            {
                // Match the CPU path's `generate_data_trigonometric_from_float`:
                // draw a uniform random angle in [0, 2pi) and return its
                // cosine (so the values are uniformly distributed on [-1, 1]).
                // The deterministic `sinf(dataIdx)` we used here previously
                // produced a smooth oscillating pattern with no PRNG mixing,
                // which exposed pathological cancellations in tests that
                // multiply A*B by per-row vectors (e.g. scaleAlphaVec) and
                // then norm-check the activated/biased result -- the
                // resulting reference values were small enough that the
                // relative error denominator blew up.
                uint32_t       s     = seedMix(seed, dataIdx);
                float          u     = prngFloat01(s);
                constexpr float kTwoPi = 6.283185307179586f;
                return __builtin_cosf(kTwoPi * u);
            }
            case DeviceInitMode::Bounded:
            {
                uint32_t s = seedMix(seed, dataIdx);
                float    u = prngFloat01(s);
                return cfg.minVal + (cfg.maxVal - cfg.minVal) * u;
            }
            case DeviceInitMode::BoundedAlternatingSign:
            {
                uint32_t s = seedMix(seed, dataIdx);
                float    u = prngFloat01(s);
                float    v = cfg.minVal + (cfg.maxVal - cfg.minVal) * u;
                return ((dataIdx & 1ull) ? -__builtin_fabsf(v) : __builtin_fabsf(v));
            }
            case DeviceInitMode::Unbounded:
            {
                uint32_t s   = seedMix(seed, dataIdx);
                float    u   = prngFloat01(s);
                // Map to a wide log-spaced range, sign random.
                float    sign = (xorshift32(s) & 1u) ? -1.0f : 1.0f;
                float    mag  = __builtin_ldexpf(u + 1.0f, static_cast<int>((xorshift32(s) % 16) - 8));
                return sign * mag;
            }
            case DeviceInitMode::NormalFromFloat:
            {
                // Box-Muller transform from two uniforms.
                uint32_t s  = seedMix(seed, dataIdx);
                float    u1 = __builtin_fmaxf(prngFloat01(s), 1.0e-7f);
                float    u2 = prngFloat01(s);
                float    r  = __builtin_sqrtf(-2.0f * __builtin_logf(u1));
                float    z0 = r * __builtin_cosf(6.2831853f * u2);
                return cfg.mean + cfg.stdDev * z0;
            }
            case DeviceInitMode::ConstantValue:
                // Plumbed by makeConfig: 2.0 (Twos), -1.0 (NegOnes), or
                // GpuTraits<DTYPE>::maxNormal (MaxVals).
                return cfg.constValue;
            case DeviceInitMode::RandInt:
            {
                // Uniform integer in [randIntLo, randIntHi]. Modulo into a
                // 64-bit range so a full int range (lo=INT_MIN, hi=INT_MAX)
                // doesn't overflow; bias from xorshift32 % range is
                // negligible for the small ranges this mode is exercised
                // with (typically < 256). Degenerate range (lo == hi)
                // collapses to `lo` because range = 1 makes the modulo 0.
                uint32_t       s     = seedMix(seed, dataIdx);
                uint32_t const r     = xorshift32(s);
                uint64_t const range = static_cast<uint64_t>(cfg.randIntHi)
                                       - static_cast<uint64_t>(cfg.randIntLo) + 1ull;
                int64_t const v
                    = static_cast<int64_t>(cfg.randIntLo) + static_cast<int64_t>(r % range);
                return static_cast<float>(v);
            }
            }
            return 0.0f;
        }

        // ----------------------------------------------------------------------
        // Device-side scale encoding
        //
        // For E8M0: stored as a single byte = unbiased_exp + bias (127).
        // For E4M3 / E5M3: stored as a byte (S?EEEE.MMM) with the unbiased
        // exponent we want and a normalised (1.0) mantissa.
        // ----------------------------------------------------------------------
        // Per-scale-format clamp: returns the unbiased exponent we will
        // actually encode in the scale byte. Mirrors the saturation rules
        // baked into encodeScale<ST> below; exposed separately so the data
        // convert path passes the same clamped value to the hardware MX
        // convert (keeping the kernel-visible scale and the scale used for
        // data quantisation in lockstep).
        //
        // Both clamp and encoder source their bounds from `ScaleInfo<ST>`
        // (the same `ScaleFmt` used by the rest of the codebase) so the two
        // can't drift, and so a new scale format added to `ScaleInfoFor<>`
        // is picked up automatically. `ScaleInfo<ST>::unBiasedE{Min,Max}`
        // are already biased-corrected for the NaN/Inf reserved slots:
        //   * E8M0 (HasInf=true, HasNan=true): emin/emax = [-127, 127]
        //   * E4M3 (HasInf=false, HasNan=true, HasZero=true): [-6, 8]
        //   * E5M3 (HasInf=false, HasNan=true, HasZero=true): [-14, 16]
        // For E4M3 / E5M3 the all-ones biased exponent is a valid finite
        // normal (only the (biased=all-ones, mantissa=all-ones) byte is
        // reserved for NaN), so unBiasedEMax is the largest exponent we can
        // legitimately encode with mantissa = 0.
        template <ScaleType ST>
        __device__ __forceinline__ int clampScaleExp(int unbiasedExp)
        {
            using SI                 = ScaleInfo<ST>;
            constexpr int kMinExp    = SI::unBiasedEMin;
            constexpr int kMaxExp    = SI::unBiasedEMax;
            if(unbiasedExp < kMinExp)
                return kMinExp;
            if(unbiasedExp > kMaxExp)
                return kMaxExp;
            return unbiasedExp;
        }

        template <ScaleType ST>
        __device__ __forceinline__ uint8_t encodeScale(int unbiasedExp);

        template <>
        __device__ __forceinline__ uint8_t encodeScale<ScaleType::E8M0>(int unbiasedExp)
        {
            using SI               = ScaleInfo<ScaleType::E8M0>;
            constexpr int kMinExp  = SI::unBiasedEMin;
            constexpr int kMaxExp  = SI::unBiasedEMax;
            constexpr int kBias    = SI::bias;
            int           v = unbiasedExp + kBias;
            if(unbiasedExp < kMinExp)
                v = kMinExp + kBias;
            if(unbiasedExp > kMaxExp)
                v = kMaxExp + kBias;
            return static_cast<uint8_t>(v);
        }

        template <>
        __device__ __forceinline__ uint8_t encodeScale<ScaleType::E4M3>(int unbiasedExp)
        {
            // Byte layout S=1 EEEE MMM. Mantissa = 0 (normalised 1.0), so the
            // encoded scale value is exactly 1.0 * 2^unbiasedExp.
            using SI               = ScaleInfo<ScaleType::E4M3>;
            constexpr int kMinExp  = SI::unBiasedEMin;
            constexpr int kMaxExp  = SI::unBiasedEMax;
            constexpr int kBias    = SI::bias;
            int           biased   = unbiasedExp + kBias;
            if(unbiasedExp > kMaxExp)
                biased = kMaxExp + kBias;
            if(unbiasedExp < kMinExp)
                biased = 1;
            return static_cast<uint8_t>(biased << SI::mantissaBits);
        }

        template <>
        __device__ __forceinline__ uint8_t encodeScale<ScaleType::E5M3>(int unbiasedExp)
        {
            // Byte layout S? EEEEE MMM. Mantissa = 0; encoded scale = 2^unbiasedExp.
            using SI               = ScaleInfo<ScaleType::E5M3>;
            constexpr int kMinExp  = SI::unBiasedEMin;
            constexpr int kMaxExp  = SI::unBiasedEMax;
            constexpr int kBias    = SI::bias;
            int           biased   = unbiasedExp + kBias;
            if(unbiasedExp > kMaxExp)
                biased = kMaxExp + kBias;
            if(unbiasedExp < kMinExp)
                biased = 1;
            return static_cast<uint8_t>(biased << SI::mantissaBits);
        }

        template <ScaleType ST>
        __device__ __forceinline__ float decodeScale(uint8_t scale);

        template <>
        __device__ __forceinline__ float decodeScale<ScaleType::E8M0>(uint8_t scale)
        {
            int unbiased = static_cast<int>(scale) - 127;
            return __builtin_ldexpf(1.0f, unbiased);
        }

        template <>
        __device__ __forceinline__ float decodeScale<ScaleType::E4M3>(uint8_t scale)
        {
            int biasedExp = (scale >> 3) & 0x0f;
            int mant      = scale & 0x07;
            float m       = 1.0f + static_cast<float>(mant) / 8.0f;
            int unbiased  = biasedExp - 7;
            if(biasedExp == 0)
            {
                // Subnormal: implicit 0.M, exponent = 1 - bias.
                m        = static_cast<float>(mant) / 8.0f;
                unbiased = 1 - 7;
            }
            return m * __builtin_ldexpf(1.0f, unbiased);
        }

        template <>
        __device__ __forceinline__ float decodeScale<ScaleType::E5M3>(uint8_t scale)
        {
            int   biasedExp = (scale >> 3) & 0x1f;
            int   mant      = scale & 0x07;
            float m         = 1.0f + static_cast<float>(mant) / 8.0f;
            int   unbiased  = biasedExp - 15;
            if(biasedExp == 0)
            {
                m        = static_cast<float>(mant) / 8.0f;
                unbiased = 1 - 15;
            }
            return m * __builtin_ldexpf(1.0f, unbiased);
        }

        // ----------------------------------------------------------------------
        // Device-side data quantisation + packing.
        //
        // Each per-DTYPE specialisation takes one MX scale block worth of
        // float values, applies the supplied unbiased scale exponent, RNE
        // quantises to the target MX format, and writes the packed bytes
        // directly into `outBytes`.
        //
        // Implementation defers to `__amd_cvt_*_scale` from <hip/hip_ext_ocp.h>
        // - the same wrapper hipblaslt's hipblaslt_float4 / _float6 / _float8
        // types use. On gfx950 these emit the hardware MX convert
        // instructions (v_cvt_scalef32_pk_fp4_f32, pk32_f32_fp6, pk_fp8_f32);
        // off gfx950 they fall back to the canonical fcbx software path so
        // generator output is bit-identical to those production types.
        //
        // `blockSize` must equal `Tr::elementsPerScaleBlock`-style power-of-two
        // (16 or 32 in current use). FP6 always converts a full 32-element
        // group at once via __amd_cvt_floatx32_to_fp6x32_scale (the only
        // F32->FP6 wrapper); for blockSize<32 the trailing inputs are
        // zero-padded and the trailing output bytes discarded.
        // ----------------------------------------------------------------------
        template <typename DTYPE>
        __device__ __forceinline__ void convertBlockScaledRNE(uint8_t*      outBytes,
                                                              float const*  values,
                                                              int           blockSize,
                                                              __amd_scale_t scaleExp);

        template <>
        __device__ __forceinline__ void
            convertBlockScaledRNE<ocp_e2m1_mxfp4>(uint8_t*      outBytes,
                                                  float const*  values,
                                                  int           blockSize,
                                                  __amd_scale_t scaleExp)
        {
            int const pairs = blockSize / 2;
            for(int i = 0; i < pairs; ++i)
            {
                __amd_floatx2_storage_t pair{values[2 * i], values[2 * i + 1]};
                outBytes[i] = static_cast<uint8_t>(
                    __amd_cvt_floatx2_to_fp4x2_scale(pair, __AMD_OCP_E2M1, scaleExp));
            }
        }

        // FP4 with non-E8M0 scale formats (E4M3, E5M3) shares the data path -
        // only the scale byte's encoding differs and that's handled by
        // encodeScale<scaleKind> at the call site.
        template <>
        __device__ __forceinline__ void
            convertBlockScaledRNE<ocp_e2m1_mxfp4_e4m3>(uint8_t*      outBytes,
                                                       float const*  values,
                                                       int           blockSize,
                                                       __amd_scale_t scaleExp)
        {
            convertBlockScaledRNE<ocp_e2m1_mxfp4>(outBytes, values, blockSize, scaleExp);
        }

        template <>
        __device__ __forceinline__ void
            convertBlockScaledRNE<ocp_e2m1_mxfp4_e5m3>(uint8_t*      outBytes,
                                                       float const*  values,
                                                       int           blockSize,
                                                       __amd_scale_t scaleExp)
        {
            convertBlockScaledRNE<ocp_e2m1_mxfp4>(outBytes, values, blockSize, scaleExp);
        }

        // FP6 / BF6: one __amd_cvt_floatx32_to_fp6x32_scale call covers a full
        // 32-element group. Pad inputs and trim outputs for blockSize < 32.
        template <__amd_fp6_interpretation_t Interp>
        __device__ __forceinline__ void convertBlockScaledRNEFp6(uint8_t*      outBytes,
                                                                 float const*  values,
                                                                 int           blockSize,
                                                                 __amd_scale_t scaleExp)
        {
            __amd_floatx32_storage_t in;
            for(int i = 0; i < 32; ++i)
                in[i] = (i < blockSize) ? values[i] : 0.0f;
            __amd_fp6x32_storage_t out
                = __amd_cvt_floatx32_to_fp6x32_scale(in, Interp, scaleExp);
            auto const* bytes  = reinterpret_cast<uint8_t const*>(&out);
            int const   outLen = blockSize * 6 / 8;
            for(int b = 0; b < outLen; ++b)
                outBytes[b] = bytes[b];
        }

        template <>
        __device__ __forceinline__ void
            convertBlockScaledRNE<ocp_e2m3_mxfp6>(uint8_t*      outBytes,
                                                  float const*  values,
                                                  int           blockSize,
                                                  __amd_scale_t scaleExp)
        {
            convertBlockScaledRNEFp6<__AMD_OCP_E2M3>(outBytes, values, blockSize, scaleExp);
        }

        template <>
        __device__ __forceinline__ void
            convertBlockScaledRNE<ocp_e3m2_mxfp6>(uint8_t*      outBytes,
                                                  float const*  values,
                                                  int           blockSize,
                                                  __amd_scale_t scaleExp)
        {
            convertBlockScaledRNEFp6<__AMD_OCP_E3M2>(outBytes, values, blockSize, scaleExp);
        }

        // FP8 / BF8: pk2 convert per pair of elements; each call returns a
        // 16-bit storage holding two FP8 bytes.
        template <__amd_fp8_interpretation_t Interp>
        __device__ __forceinline__ void convertBlockScaledRNEFp8(uint8_t*      outBytes,
                                                                 float const*  values,
                                                                 int           blockSize,
                                                                 __amd_scale_t scaleExp)
        {
            int const pairs = blockSize / 2;
            for(int i = 0; i < pairs; ++i)
            {
                __amd_floatx2_storage_t pair{values[2 * i], values[2 * i + 1]};
                __amd_fp8x2_storage_t   p
                    = __amd_cvt_floatx2_to_fp8x2_scale(pair, Interp, scaleExp);
                outBytes[2 * i + 0] = static_cast<uint8_t>(p & 0xffu);
                outBytes[2 * i + 1] = static_cast<uint8_t>((p >> 8) & 0xffu);
            }
        }

        template <>
        __device__ __forceinline__ void
            convertBlockScaledRNE<ocp_e4m3_mxfp8>(uint8_t*      outBytes,
                                                  float const*  values,
                                                  int           blockSize,
                                                  __amd_scale_t scaleExp)
        {
            convertBlockScaledRNEFp8<__AMD_OCP_E4M3>(outBytes, values, blockSize, scaleExp);
        }

        template <>
        __device__ __forceinline__ void
            convertBlockScaledRNE<ocp_e5m2_mxfp8>(uint8_t*      outBytes,
                                                  float const*  values,
                                                  int           blockSize,
                                                  __amd_scale_t scaleExp)
        {
            convertBlockScaledRNEFp8<__AMD_OCP_E5M2>(outBytes, values, blockSize, scaleExp);
        }

        // ----------------------------------------------------------------------
        // Main generation kernel
        //
        // Launch with one thread per MX scale block (`numBlocks` threads
        // total). Each thread:
        //   1. Generates `blockSize` raw float values.
        //   2. Computes the per-block max abs.
        //   3. Picks an unbiased scale exponent so block_max / 2^scale fits
        //      in the dtype's max normal magnitude.
        //   4. Encodes the scale and writes it to `scaleOut`.
        //   5. Quantises each scaled element to the dtype bit pattern.
        //   6. Packs the result into `dataOut`.
        // ----------------------------------------------------------------------
        template <typename DTYPE>
        __global__ void generateMXBlocksKernel(uint8_t*         dataOut,
                                               uint8_t*         scaleOut,
                                               uint64_t         numBlocks,
                                               int              blockSize,
                                               uint32_t         seed,
                                               DeviceInitConfig cfg)
        {
            using Tr = GpuTraits<DTYPE>;

            uint64_t blockIdx_ = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if(blockIdx_ >= numBlocks)
                return;

            // Up to 32 elements per block; matches the only sizes used by
            // current callers (16 or 32). Stack allocation is fine.
            float values[32];
            float maxAbs = 0.0f;

            for(int i = 0; i < blockSize; ++i)
            {
                uint64_t dataIdx = blockIdx_ * static_cast<uint64_t>(blockSize) + i;
                float    v       = genValue(cfg.mode, seed, dataIdx, cfg);
                values[i]        = v;
                float a          = __builtin_fabsf(v);
                if(a > maxAbs)
                    maxAbs = a;
            }

            // Derive scale: choose unbiasedExp such that block_max <= maxNormal * 2^scaleExp.
            // i.e. scaleExp = ceil(log2(block_max / maxNormal)). Round up so
            // that division by 2^scaleExp brings every element within range.
            // For an all-zero block we pick a neutral scale of 2^0 = 1.
            int scaleExp = 0;
            if(maxAbs > 0.0f)
            {
                float ratio = maxAbs / Tr::maxNormal;
                float l2    = __builtin_log2f(ratio);
                scaleExp    = static_cast<int>(__builtin_ceilf(l2));
                while(maxAbs > Tr::maxNormal * __builtin_ldexpf(1.0f, scaleExp))
                    ++scaleExp;
            }

            // Saturate to the encodable range for this scale format BEFORE
            // touching either the scale byte or the data convert, so both
            // sides agree on the actual scale value the kernel will see.
            int const     clampedExp = clampScaleExp<Tr::scaleKind>(scaleExp);
            __amd_scale_t scaleArg   = static_cast<__amd_scale_t>(clampedExp);

            scaleOut[blockIdx_] = encodeScale<Tr::scaleKind>(clampedExp);

            uint64_t bytesPerBlock
                = static_cast<uint64_t>(blockSize) * Tr::bitsPerElem / 8u;
            uint8_t* outRow = dataOut + blockIdx_ * bytesPerBlock;
            convertBlockScaledRNE<DTYPE>(outRow, values, blockSize, scaleArg);
        }

        // ----------------------------------------------------------------------
        // Pre-swizzle scale kernel: applies the same algorithm as the host
        // `preSwizzleScalesGFX950` to a device-resident scale buffer.
        //
        // For an `(numRows, numCols)` row-major scale tensor (rounded up to
        // (paddedRows=mult32, paddedCols=mult8)), the algorithm views
        // the buffer as 6D (paddedRows/32, 2, 16, paddedCols/8, 2, 4) and
        // permutes (0,3,5,2,4,1). Each thread writes one output byte by
        // computing its source coordinates from the inverse permutation.
        // ----------------------------------------------------------------------
        __global__ void preSwizzleScalesKernel(uint8_t*       dst,
                                               uint8_t const* src,
                                               size_t         paddedRows,
                                               size_t         paddedCols)
        {
            size_t total = paddedRows * paddedCols;
            size_t tid   = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if(tid >= total)
                return;

            // Decompose tid into 6D output coords matching the host-side permutation.
            // dimOrder = {1, 4, 2, 5, 3, 0} (row-major fastest-varying first).
            // Sizes (in dimOrder order): {2, 2, 16, 4, paddedCols/8, paddedRows/32}.
            size_t s1 = 2;
            size_t s4 = 2;
            size_t s2 = 16;
            size_t s5 = 4;
            size_t s3 = paddedCols / 8;
            size_t s0 = paddedRows / 32;

            size_t cur = tid;
            size_t i1  = cur % s1; cur /= s1;
            size_t i4  = cur % s4; cur /= s4;
            size_t i2  = cur % s2; cur /= s2;
            size_t i5  = cur % s5; cur /= s5;
            size_t i3  = cur % s3; cur /= s3;
            size_t i0  = cur % s0;

            // Source row/col in the original 2D layout:
            //   row = i0*32 + i1*16 + i2,  col = i3*8 + i4*4 + i5
            size_t srcRow = i0 * 32 + i1 * 16 + i2;
            size_t srcCol = i3 * 8 + i4 * 4 + i5;
            size_t srcIdx = srcRow * paddedCols + srcCol;

            dst[tid] = src[srcIdx];
        }

        // ----------------------------------------------------------------------
        // gfx1250 dimk swizzle kernel: mirrors the host
        // `preSwizzleScalesGFX1250` algorithm.
        //
        // Input view:  {slowDim, fastDim} (natural-layout column-major scales).
        // Output view: {numTiles, slowDim, dimk}, where
        //   dimk       = 128 / mxBlock,
        //   numTiles   = ceil(fastDim / dimk),
        //   paddedFast = numTiles * dimk.
        // For each output index (tile, s, j) in [0, numTiles) x [0, slowDim) x
        // [0, dimk):
        //   srcFast = tile * dimk + j
        //   out[tile, s, j] = (srcFast < fastDim) ? src[s * fastDim + srcFast]
        //                                         : 0   // padded slot
        // One thread per output byte.
        // ----------------------------------------------------------------------
        __global__ void preSwizzleScalesGFX1250Kernel(uint8_t*       dst,
                                                      uint8_t const* src,
                                                      size_t         slowDim,
                                                      size_t         fastDim,
                                                      size_t         dimk,
                                                      size_t         numTiles)
        {
            size_t total = numTiles * slowDim * dimk;
            size_t tid   = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if(tid >= total)
                return;

            size_t j    = tid % dimk;
            size_t rest = tid / dimk;
            size_t s    = rest % slowDim;
            size_t tile = rest / slowDim;

            size_t srcFast = tile * dimk + j;
            dst[tid]       = (srcFast < fastDim) ? src[s * fastDim + srcFast]
                                                 : static_cast<uint8_t>(0);
        }

        // ----------------------------------------------------------------------
        // Generic host-side build helpers
        // ----------------------------------------------------------------------
        template <typename DTYPE>
        size_t computeArraySize(std::vector<index_t> const& sizes,
                                std::vector<index_t> const& strides)
        {
            // Mirror DataGenerator::generate logic: array_size =
            // strides[N-1] * sizes[N-1] after sorting dims by stride.
            std::vector<size_t> perm(sizes.size());
            for(size_t i = 0; i < perm.size(); ++i)
                perm[i] = i;
            std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
                return strides[a] < strides[b];
            });
            std::vector<size_t> ss(sizes.size()), st(strides.size());
            for(size_t i = 0; i < perm.size(); ++i)
            {
                ss[i] = static_cast<size_t>(sizes[perm[i]]);
                st[i] = static_cast<size_t>(strides[perm[i]]);
            }
            size_t n = ss.size();
            return st[n - 1] * ss[n - 1];
        }
    } // namespace gpu_detail

    // --------------------------------------------------------------------------
    // DataGeneratorGPU public-method definitions
    // --------------------------------------------------------------------------
    template <typename DTYPE>
    DataGeneratorGPU<DTYPE>::~DataGeneratorGPU()
    {
        if(m_ownsBuffers)
        {
            if(m_dataDevice)
                (void)hipFree(m_dataDevice);
            if(m_scaleDevice)
                (void)hipFree(m_scaleDevice);
        }
    }

    template <typename DTYPE>
    size_t DataGeneratorGPU<DTYPE>::getDataBufferBytes(std::vector<index_t> const& sizes,
                                                       std::vector<index_t> const& strides,
                                                       DataGeneratorOptions const& options)
    {
        (void)options;
        // computeArraySize is the same arithmetic the kernel-launch path
        // uses (`strides[N-1] * sizes[N-1]` after sorting dims by stride),
        // so the byte count below covers every element the kernel will
        // touch -- including any padding the strides imply.
        size_t arraySize = gpu_detail::computeArraySize<DTYPE>(sizes, strides);
        constexpr int bitsPerElem = gpu_detail::GpuTraits<DTYPE>::bitsPerElem;
        return (arraySize * bitsPerElem + 7) / 8;
    }

    template <typename DTYPE>
    size_t DataGeneratorGPU<DTYPE>::getScaleBufferBytes(std::vector<index_t> const& sizes,
                                                        std::vector<index_t> const& strides,
                                                        DataGeneratorOptions const& options)
    {
        size_t arraySize = gpu_detail::computeArraySize<DTYPE>(sizes, strides);
        if(options.blockScaling <= 0)
            return 0;
        return arraySize / static_cast<size_t>(options.blockScaling);
    }

    namespace gpu_detail
    {
        // Contiguous column-major strides for the no-strides sizer overloads.
        // Pulled into a helper so the two overloads can't drift.
        inline std::vector<index_t> defaultColumnMajorStrides(std::vector<index_t> const& sizes)
        {
            std::vector<index_t> strides(sizes.size(), 1);
            for(size_t i = 1; i < sizes.size(); ++i)
                strides[i] = strides[i - 1] * sizes[i - 1];
            return strides;
        }
    } // namespace gpu_detail

    template <typename DTYPE>
    size_t DataGeneratorGPU<DTYPE>::getDataBufferBytes(std::vector<index_t> const& sizes,
                                                       DataGeneratorOptions const& options)
    {
        return getDataBufferBytes(sizes, gpu_detail::defaultColumnMajorStrides(sizes), options);
    }

    template <typename DTYPE>
    size_t DataGeneratorGPU<DTYPE>::getScaleBufferBytes(std::vector<index_t> const& sizes,
                                                        DataGeneratorOptions const& options)
    {
        return getScaleBufferBytes(sizes, gpu_detail::defaultColumnMajorStrides(sizes), options);
    }

    template <typename DTYPE>
    DataGeneratorGPU<DTYPE>& DataGeneratorGPU<DTYPE>::generate(std::vector<index_t>        sizes,
                                                               std::vector<index_t>        strides,
                                                               DataGeneratorOptions const& options,
                                                               hipStream_t                 stream)
    {
        m_options = options;
        m_sizes   = sizes;
        m_strides = strides;

        // Stride-aware sizing -- the kernel launch uses computeArraySize
        // which is stride-aware, so a padded layout (e.g. strides imply
        // 80*2 elements while sizes={64,2} would only need 64*2 dense)
        // needs the larger allocation. Using the no-strides overload here
        // would under-allocate and the kernel would write OOB.
        m_dataBufferBytes  = getDataBufferBytes(sizes, strides, options);
        m_scaleBufferBytes = getScaleBufferBytes(sizes, strides, options);

        // Free any prior allocation (size could change between calls).
        if(m_ownsBuffers)
        {
            if(m_dataDevice)
                (void)hipFree(m_dataDevice);
            if(m_scaleDevice)
                (void)hipFree(m_scaleDevice);
            m_dataDevice  = nullptr;
            m_scaleDevice = nullptr;
        }

        DGEN_DETAIL_CHECK_HIP_(hipMalloc(&m_dataDevice, m_dataBufferBytes));
        if(m_scaleBufferBytes > 0)
            DGEN_DETAIL_CHECK_HIP_(hipMalloc(&m_scaleDevice, m_scaleBufferBytes));
        m_ownsBuffers = true;

        generateInto(m_dataDevice, m_scaleDevice, sizes, strides, options, stream);
        return *this;
    }

    template <typename DTYPE>
    void DataGeneratorGPU<DTYPE>::generateInto(void*                       devData,
                                               void*                       devScale,
                                               std::vector<index_t>        sizes,
                                               std::vector<index_t>        strides,
                                               DataGeneratorOptions const& options,
                                               hipStream_t                 stream)
    {
        if(sizes.size() != strides.size())
            throw std::invalid_argument(
                "DataGeneratorGPU::generateInto: size and stride vectors must have the same size");
        if(sizes.empty())
            throw std::invalid_argument(
                "DataGeneratorGPU::generateInto: size vector must not be empty");
        // The OCP MX spec defines only three valid block sizes; the kernel's
        // per-block `float values[32]` stack buffer is sized for the largest.
        // Reject anything else up front so we don't quietly mis-encode (e.g.
        // a caller that passed `blockScaling = 8` would get a kernel that
        // reads 8 elements but a scale layout the dequantiser expects to be
        // 16-aligned).
        switch(options.blockScaling)
        {
        case 1:
        case 16:
        case 32:
            break;
        default:
            throw std::invalid_argument(
                "DataGeneratorGPU::generateInto: blockScaling must be one of "
                "{1, 16, 32} (OCP MX block sizes)");
        }
        // Every DTYPE the GPU backend supports is an MX type, so we always
        // emit a per-block scale byte (either from the device kernel or the
        // host-fallback memcpy below). Refuse to launch without somewhere to
        // put them; otherwise the host-fallback branch silently drops the
        // scale bytes and dequant downstream sees garbage.
        if(devScale == nullptr)
            throw std::invalid_argument(
                "DataGeneratorGPU::generateInto: devScale must not be null "
                "for MX DTYPEs (one scale byte per block is required)");

        size_t arraySize = gpu_detail::computeArraySize<DTYPE>(sizes, strides);
        if(arraySize % static_cast<size_t>(options.blockScaling) != 0)
            throw std::invalid_argument(
                "DataGeneratorGPU::generateInto: array size must be a multiple of blockScaling");

        size_t numBlocks = arraySize / static_cast<size_t>(options.blockScaling);

        // Pathological modes whose byte-level encoding the on-device
        // pipeline can't reproduce (denorm bit patterns get renormalised,
        // NaN/Inf scale bytes need special encodings the magnitude-derived
        // path doesn't emit). See `requiresHostFallback` for the full
        // breakdown. Generate on the host CPU and hipMemcpy the bytes.
        if(gpu_detail::requiresHostFallback(options.initMode))
        {
            DataGenerator<DTYPE> hostGen;
            hostGen.setSeed(m_seed);
            hostGen.generate(sizes, strides, options);
            auto hostData  = hostGen.getDataBytes();
            auto hostScale = hostGen.getScaleBytes();
            DGEN_DETAIL_CHECK_HIP_(hipMemcpyAsync(
                devData, hostData.data(), hostData.size(), hipMemcpyHostToDevice, stream));
            if(devScale != nullptr && !hostScale.empty())
            {
                DGEN_DETAIL_CHECK_HIP_(hipMemcpyAsync(devScale,
                                                  hostScale.data(),
                                                  hostScale.size(),
                                                  hipMemcpyHostToDevice,
                                                  stream));
            }
            DGEN_DETAIL_CHECK_HIP_(hipStreamSynchronize(stream));
            return;
        }

        // Configuration for the kernel.
        gpu_detail::DeviceInitConfig cfg = gpu_detail::makeConfig<DTYPE>(options, sizes);

        // One thread per MX scale block. See gpu_detail::kThreadsPerBlock
        // for the rationale on 256.
        size_t gridDimX = (numBlocks + gpu_detail::kThreadsPerBlock - 1)
                          / gpu_detail::kThreadsPerBlock;
        gpu_detail::checkGridDimX(gridDimX, "generateInto");

        hipLaunchKernelGGL(gpu_detail::generateMXBlocksKernel<DTYPE>,
                           dim3(static_cast<unsigned>(gridDimX)),
                           dim3(gpu_detail::kThreadsPerBlock),
                           0,
                           stream,
                           static_cast<uint8_t*>(devData),
                           static_cast<uint8_t*>(devScale),
                           static_cast<uint64_t>(numBlocks),
                           static_cast<int>(options.blockScaling),
                           m_seed,
                           cfg);
        DGEN_DETAIL_CHECK_HIP_(hipGetLastError());
    }

    template <typename DTYPE>
    std::vector<uint8_t> DataGeneratorGPU<DTYPE>::getDataBytes() const
    {
        std::vector<uint8_t> host(m_dataBufferBytes);
        if(m_dataBufferBytes == 0 || m_dataDevice == nullptr)
            return host;
        DGEN_DETAIL_CHECK_HIP_(
            hipMemcpy(host.data(), m_dataDevice, m_dataBufferBytes, hipMemcpyDeviceToHost));
        return host;
    }

    template <typename DTYPE>
    std::vector<uint8_t> DataGeneratorGPU<DTYPE>::getScaleBytes() const
    {
        std::vector<uint8_t> host(m_scaleBufferBytes);
        if(m_scaleBufferBytes == 0 || m_scaleDevice == nullptr)
            return host;
        DGEN_DETAIL_CHECK_HIP_(
            hipMemcpy(host.data(), m_scaleDevice, m_scaleBufferBytes, hipMemcpyDeviceToHost));
        return host;
    }

    template <typename DTYPE>
    std::vector<float> DataGeneratorGPU<DTYPE>::getReferenceFloat() const
    {
        // Materialise the reference floats on the host using the existing
        // CPU `toFloat<DTYPE>` helper - that's the bit-exact converter the
        // CPU backend uses and shares with all validation paths.
        auto dataHost  = getDataBytes();
        auto scaleHost = getScaleBytes();

        // Use the strides the caller actually passed to `generate` (cached
        // in m_strides) so getReferenceFloat reflects what the kernel really
        // wrote, including padded layouts. Fall back to a contiguous
        // column-major assumption only if m_strides is empty (a defensive
        // path -- in practice m_strides is populated by `generate` before
        // m_dataDevice has anything readable in it).
        std::vector<index_t> const strides
            = !m_strides.empty() ? m_strides
                                 : gpu_detail::defaultColumnMajorStrides(m_sizes);
        size_t arraySize = gpu_detail::computeArraySize<DTYPE>(m_sizes, strides);

        std::vector<float> ret(arraySize);
        size_t blockSize = static_cast<size_t>(m_options.blockScaling);
        for(size_t i = 0; i < arraySize; ++i)
        {
            size_t scaleIdx = i / blockSize;
            ret[i]
                = toFloatPacked<DTYPE>(scaleHost.data(), dataHost.data(),
                                       static_cast<index_t>(scaleIdx), static_cast<index_t>(i));
        }
        return ret;
    }

    template <typename DTYPE>
    void DataGeneratorGPU<DTYPE>::preSwizzleScalesGFX950Device(
        std::vector<size_t> const& scaleSizes, hipStream_t stream)
    {
        if(scaleSizes.size() != 2)
            throw std::invalid_argument(
                "DataGeneratorGPU::preSwizzleScalesGFX950Device: scaleSizes must have 2 elements");
        if(m_scaleDevice == nullptr)
            throw std::runtime_error(
                "DataGeneratorGPU::preSwizzleScalesGFX950Device: no scale buffer allocated");

        size_t numRows = scaleSizes[0];
        size_t numCols = scaleSizes[1];
        // Mirror the host helper's `input.size() == product(sizes)` check.
        // Without it a wrong (rows, cols) pair would still pass to the
        // padder + kernel but read out of bounds from m_scaleDevice.
        if(numRows * numCols != m_scaleBufferBytes)
            throw std::invalid_argument(
                "DataGeneratorGPU::preSwizzleScalesGFX950Device: "
                "numRows * numCols does not match the current scale buffer size");
        size_t paddedRows = ((numRows + 31) / 32) * 32;
        size_t paddedCols = ((numCols + 7) / 8) * 8;
        size_t paddedSize = paddedRows * paddedCols;

        // Source buffer needs padded dimensions. If the existing scale buffer
        // is already padded, alias it; otherwise pad into a temporary
        // (RAII-owned so a later throw doesn't leak it).
        gpu_detail::DeviceBuffer srcBuf;
        uint8_t*                 src = nullptr;
        if(paddedRows == numRows && paddedCols == numCols)
        {
            src = m_scaleDevice;
        }
        else
        {
            srcBuf = gpu_detail::DeviceBuffer(paddedSize);
            src    = srcBuf.get();
            DGEN_DETAIL_CHECK_HIP_(hipMemsetAsync(src, 0, paddedSize, stream));
            // One 2-D copy replaces N row-by-row hipMemcpyAsyncs.
            DGEN_DETAIL_CHECK_HIP_(hipMemcpy2DAsync(src,
                                                    paddedCols,
                                                    m_scaleDevice,
                                                    numCols,
                                                    numCols,
                                                    numRows,
                                                    hipMemcpyDeviceToDevice,
                                                    stream));
        }

        gpu_detail::DeviceBuffer dstBuf(paddedSize);

        size_t gridDimX = (paddedSize + gpu_detail::kThreadsPerBlock - 1)
                          / gpu_detail::kThreadsPerBlock;
        gpu_detail::checkGridDimX(gridDimX, "preSwizzleScalesGFX950Device");

        hipLaunchKernelGGL(gpu_detail::preSwizzleScalesKernel,
                           dim3(static_cast<unsigned>(gridDimX)),
                           dim3(gpu_detail::kThreadsPerBlock),
                           0,
                           stream,
                           dstBuf.get(),
                           src,
                           paddedRows,
                           paddedCols);
        DGEN_DETAIL_CHECK_HIP_(hipGetLastError());

        // Sync before any hipFree (the one below for the old m_scaleDevice
        // and the implicit one in srcBuf's destructor on scope exit). The
        // kernel still has both as live source pointers; freeing either
        // before completion is a use-after-free even if the in-order stream
        // would eventually serialise it. Past this sync we own dstBuf
        // exclusively.
        DGEN_DETAIL_CHECK_HIP_(hipStreamSynchronize(stream));

        // Past the throw points: take ownership of the new buffer.
        if(m_ownsBuffers && m_scaleDevice)
            (void)hipFree(m_scaleDevice);
        m_scaleDevice      = dstBuf.release();
        m_scaleBufferBytes = paddedSize;
        m_ownsBuffers      = true;
    }

    template <typename DTYPE>
    void DataGeneratorGPU<DTYPE>::preSwizzleScalesGFX1250Device(size_t      slowDim,
                                                                size_t      fastDim,
                                                                size_t      mxBlock,
                                                                hipStream_t stream)
    {
        if(mxBlock == 0)
            throw std::invalid_argument(
                "DataGeneratorGPU::preSwizzleScalesGFX1250Device: mxBlock must be > 0");
        size_t const dimk = 128 / mxBlock;
        if(dimk == 0)
            throw std::invalid_argument(
                "DataGeneratorGPU::preSwizzleScalesGFX1250Device: dimk = 128/mxBlock must be > 0");
        if(m_scaleDevice == nullptr)
            throw std::runtime_error(
                "DataGeneratorGPU::preSwizzleScalesGFX1250Device: no scale buffer allocated");
        // Mirror the host helper's `input.size() == slowDim*fastDim` check
        // so wrong dims don't translate into an OOB read of m_scaleDevice.
        if(slowDim * fastDim != m_scaleBufferBytes)
            throw std::invalid_argument(
                "DataGeneratorGPU::preSwizzleScalesGFX1250Device: "
                "slowDim * fastDim does not match the current scale buffer size");

        size_t const paddedFast = ((fastDim + dimk - 1) / dimk) * dimk;
        size_t const numTiles   = paddedFast / dimk;
        size_t const paddedSize = slowDim * paddedFast;

        gpu_detail::DeviceBuffer dstBuf(paddedSize);

        size_t gridDimX = (paddedSize + gpu_detail::kThreadsPerBlock - 1)
                          / gpu_detail::kThreadsPerBlock;
        gpu_detail::checkGridDimX(gridDimX, "preSwizzleScalesGFX1250Device");

        hipLaunchKernelGGL(gpu_detail::preSwizzleScalesGFX1250Kernel,
                           dim3(static_cast<unsigned>(gridDimX)),
                           dim3(gpu_detail::kThreadsPerBlock),
                           0,
                           stream,
                           dstBuf.get(),
                           m_scaleDevice,
                           slowDim,
                           fastDim,
                           dimk,
                           numTiles);
        DGEN_DETAIL_CHECK_HIP_(hipGetLastError());

        // Sync before freeing the old m_scaleDevice -- the kernel above
        // reads from it as the swizzle source. See the matching note in
        // preSwizzleScalesGFX950Device.
        DGEN_DETAIL_CHECK_HIP_(hipStreamSynchronize(stream));

        if(m_ownsBuffers && m_scaleDevice)
            (void)hipFree(m_scaleDevice);
        m_scaleDevice      = dstBuf.release();
        m_scaleBufferBytes = paddedSize;
        m_ownsBuffers      = true;
    }

} // namespace DGen

// Tear down the file-scoped detail macro so includers don't see it.
#undef DGEN_DETAIL_CHECK_HIP_

#endif // HIP guard
