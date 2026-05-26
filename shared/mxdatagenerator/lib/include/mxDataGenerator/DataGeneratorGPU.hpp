// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Header-only HIP/GPU backend mirroring DataGenerator<DTYPE> (CPU). The body
// is HIP-guarded so non-HIP TUs can include the rest of mxDataGenerator
// without dragging in HIP. Gated by CMake option MXDATAGENERATOR_ENABLE_GPU.

#include "DataGenerator.hpp"
#include "PreSwizzle.hpp"
#include "dataTypeInfo.hpp"

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace DGen
{
    /**
     * @brief HIP/GPU backend for MX data generation.
     *
     * Mirrors the public interface of DGen::DataGenerator but uses a separate
     * device PRNG: bytes are deterministic per-seed within this backend but
     * will not match the CPU backend bit-for-bit (statistics match).
     *
     * Supported data types: ocp_e2m1_mxfp4 (+ e4m3/e5m3 scale variants),
     * ocp_e2m3_mxfp6, ocp_e3m2_mxfp6, ocp_e4m3_mxfp8, ocp_e5m2_mxfp8.
     * Scale formats: E8M0, E4M3, E5M3. Init modes generated entirely on
     * device: Bounded, BoundedAlternatingSign, Unbounded, Identity, Ones,
     * Zeros, Sequential, RowIndex, ColIndex, Checkerboard, ScaledDiagonal,
     * TrigonometricFromFloat, NormalFromFloat, Twos, NegOnes, MaxVals,
     * RandInt. The remaining `DataInitMode`s (DenormMins, DenormMaxs, NaNs,
     * Infs) need byte-pattern semantics that the device pipeline can't
     * reproduce, so they are computed on host CPU and copied to device.
     *
     * Notes on CPU/GPU parity for the deterministic modes (Identity, Ones,
     * Zeros, Sequential, RowIndex, ColIndex, Checkerboard, ScaledDiagonal):
     * CPU and GPU produce the same dequantised floats only when
     * `DataGeneratorOptions::forceDenorm == false`. With `forceDenorm = true`
     * (the CPU default), the CPU path encodes via a (subnormal-data,
     * large-scale) pair while the GPU derives a per-block scale from data
     * magnitudes; the resulting dequantised values match for `Ones` /
     * `Identity` / `Zeros` but can disagree at quantisation-precision
     * granularity for `Sequential` / `RowIndex` / `ColIndex`. Use
     * `forceDenorm = false` if you depend on byte-equivalent dequant across
     * backends.
     *
     * Notes on CPU/GPU parity for the constant-fill / RandInt modes
     * generated on device (Twos, NegOnes, MaxVals, RandInt): the CPU writes
     * a literal data byte with scale = 1.0 encoding; the GPU re-encodes the
     * same value through its derived per-block scale. Dequantised floats
     * match for Twos / NegOnes / MaxVals; for RandInt the in-range integer
     * contract holds on both backends but the *sequence* differs (GPU uses
     * xorshift32, CPU uses mt19937) and CPU saturates to dtype maxNormal
     * while GPU saturates to scale * maxNormal -- both still satisfy
     * "integer in [lo, hi]".
     */
    template <typename DTYPE>
    class DataGeneratorGPU
    {
    public:
        DataGeneratorGPU() = default;
        ~DataGeneratorGPU();

        DataGeneratorGPU(DataGeneratorGPU const&)            = delete;
        DataGeneratorGPU& operator=(DataGeneratorGPU const&) = delete;
        DataGeneratorGPU(DataGeneratorGPU&&)                 = delete;
        DataGeneratorGPU& operator=(DataGeneratorGPU&&)      = delete;

        /// Set the seed used by the device PRNG.
        void setSeed(uint32_t seed)
        {
            m_seed = seed;
        }

        /**
         * @brief Allocate device buffers and generate MX data.
         *
         * `sizes` must be 2D (the only shape used by current callers). The
         * fastest-varying dimension (sizes[0]) must be a multiple of
         * `options.blockScaling`.
         */
        DataGeneratorGPU& generate(std::vector<index_t>        sizes,
                                   std::vector<index_t>        strides,
                                   DataGeneratorOptions const& options,
                                   hipStream_t                 stream = nullptr);

        /**
         * @brief Generate MX data into caller-owned device buffers.
         *
         * `devData` must point to at least `getDataBufferBytes(sizes,
         * strides, options)` bytes of device memory; `devScale` must point
         * to at least `getScaleBufferBytes(sizes, strides, options)` bytes
         * of device memory.
         *
         * Note: pass the same `strides` to both the sizers and `generateInto`
         * -- the kernel launch arithmetic is stride-aware, so a padded
         * layout (e.g. `sizes={64,2}, strides={1,80}`) needs the larger
         * stride-aware allocation; the no-strides sizer overloads below
         * assume contiguous column-major and will under-allocate for
         * padded layouts.
         */
        void generateInto(void*                       devData,
                          void*                       devScale,
                          std::vector<index_t>        sizes,
                          std::vector<index_t>        strides,
                          DataGeneratorOptions const& options,
                          hipStream_t                 stream = nullptr);

        /// Device pointer to the packed data buffer (valid after `generate`).
        uint8_t const* getDataBytesDevice() const
        {
            return m_dataDevice;
        }

        /// Device pointer to the packed scale buffer (valid after `generate`).
        uint8_t const* getScaleBytesDevice() const
        {
            return m_scaleDevice;
        }

        /// Copy the packed data buffer back to host memory.
        std::vector<uint8_t> getDataBytes() const;

        /// Copy the packed scale buffer back to host memory.
        std::vector<uint8_t> getScaleBytes() const;

        /// Materialise the reference float vector on the host (for validation).
        std::vector<float> getReferenceFloat() const;

        /**
         * @brief Apply `preSwizzleScalesGFX950` to the scale buffer.
         *
         * `scaleSizes` is `{numScaleRows, numScaleCols}` (the same arguments
         * as the host helper of the same name). NOT in-place: the scale
         * buffer is replaced with a fresh allocation of the swizzled
         * (padded) layout. After this call:
         *   - `getScaleBytesDevice()` points at a new device pointer.
         *   - `m_scaleBufferBytes` may be larger than the value
         *     `getScaleBufferBytes(sizes, options)` would return for the
         *     pre-swizzle (unpadded) layout.
         * Callers that cached the pre-call pointer must re-query it.
         */
        void preSwizzleScalesGFX950Device(std::vector<size_t> const& scaleSizes,
                                          hipStream_t                stream = nullptr);

        /**
         * @brief Apply `preSwizzleScalesGFX1250` to the scale buffer.
         *
         * Same buffer-replacement semantics as `preSwizzleScalesGFX950Device`
         * (NOT in-place). `slowDim` and `fastDim` are the natural-layout
         * scale dimensions (see the host `preSwizzleScalesGFX1250` doc for
         * how these map to A / B and trans flags); `mxBlock` is the MX
         * block size (16 or 32).
         */
        void preSwizzleScalesGFX1250Device(size_t      slowDim,
                                           size_t      fastDim,
                                           size_t      mxBlock,
                                           hipStream_t stream = nullptr);

        /// Stride-aware byte size of the packed data buffer.
        ///
        /// Mirrors the kernel-launch arithmetic used by `generateInto`
        /// (`computeArraySize` = `strides[N-1] * sizes[N-1]` after sorting
        /// dims by stride), so the allocation covers every element the
        /// kernel will touch -- including any padding the strides imply.
        /// Always prefer this overload when you have the strides; the
        /// no-strides overload below assumes contiguous column-major.
        static size_t getDataBufferBytes(std::vector<index_t> const& sizes,
                                         std::vector<index_t> const& strides,
                                         DataGeneratorOptions const& options);

        /// Stride-aware byte size of the packed scale buffer.
        /// See `getDataBufferBytes` (stride-aware) for the rationale.
        static size_t getScaleBufferBytes(std::vector<index_t> const& sizes,
                                          std::vector<index_t> const& strides,
                                          DataGeneratorOptions const& options);

        /// Contiguous column-major byte size of the packed data buffer.
        ///
        /// Under-allocates for padded layouts (e.g. `sizes={64,2},
        /// strides={1,80}` allocates dense `64*2` while the kernel writes
        /// `80*2`); use the stride-aware overload above if your layout
        /// may be padded.
        static size_t getDataBufferBytes(std::vector<index_t> const& sizes,
                                         DataGeneratorOptions const& options);

        /// Contiguous column-major byte size of the packed scale buffer.
        /// See the no-strides `getDataBufferBytes` overload for the caveat.
        static size_t getScaleBufferBytes(std::vector<index_t> const& sizes,
                                          DataGeneratorOptions const& options);

    private:
        uint32_t m_seed = kDefaultSeed;

        DataGeneratorOptions m_options;
        std::vector<index_t> m_sizes;
        // Cached so getReferenceFloat() reproduces the exact array the
        // kernel actually wrote (rather than the contiguous-column-major
        // assumption it used to make).
        std::vector<index_t> m_strides;

        size_t m_dataBufferBytes  = 0;
        size_t m_scaleBufferBytes = 0;

        // Owned device buffers (populated by `generate`, not by `generateInto`).
        uint8_t* m_dataDevice  = nullptr;
        uint8_t* m_scaleDevice = nullptr;
        bool     m_ownsBuffers = false;
    };

} // namespace DGen

#include "DataGeneratorGPU_impl.hpp"

#endif // HIP guard
