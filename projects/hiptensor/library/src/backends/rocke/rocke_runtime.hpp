/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// The Python-free runtime of the rocKE backend: the kernel pack reader, the code object loader
// and the launch program a plan holds.
//
// A kernel pack (rocke.htpk, written by pack/rocke_pack_builder.cpp at build time) holds the code
// objects of one GPU target, the argument layout of every kernel (read from the code object
// metadata), kernel attributes the planner selects by, and a route table of tuning constants.
// The runtime maps a pack once per target, loads a code object with hipModuleLoadData the first
// time a plan needs one of its kernels (after checking its SHA-256), and launches kernels with
// hipModuleLaunchKernel and an argument buffer laid out from the pack.

#ifndef HIPTENSOR_BACKENDS_ROCKE_ROCKE_RUNTIME_HPP
#define HIPTENSOR_BACKENDS_ROCKE_ROCKE_RUNTIME_HPP

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hiptensor/hiptensor_types.h>

#include "../../backend/backend.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        // One kernel argument as the code object metadata describes it.
        struct KernelArg
        {
            std::string name;
            uint32_t    offset  = 0;
            uint32_t    size    = 0;
            bool        pointer = false;
        };

        // A kernel of a pack.
        struct PackKernel
        {
            uint32_t                           index = 0;
            std::string                        name;
            std::string                        family;
            std::string                        specKey;
            std::vector<KernelArg>             args;
            std::map<std::string, int64_t>     intAttrs;
            std::map<std::string, std::string> strAttrs;
            uint32_t                           kernargSize    = 0;
            uint32_t                           groupSegment   = 0;
            uint32_t                           privateSegment = 0;
            uint32_t                           maxWorkgroup   = 0;
            uint32_t                           wavefrontSize  = 0;
            uint64_t                           codeOffset     = 0;
            uint64_t                           codeSize       = 0;
            std::array<uint8_t, 32>            codeSha256{};

            int64_t     attr(std::string_view key, int64_t fallback = -1) const;
            std::string attrStr(std::string_view key) const;
            int         argIndex(std::string_view name) const;
        };

        // A validated kernel pack of one GPU target.
        class Pack
        {
        public:
            // Reads and validates a pack file: HIPTENSOR_STATUS_SUCCESS, or the status with `reason`
            // set (a missing file is HIPTENSOR_STATUS_ARCH_MISMATCH, a malformed or incompatible
            // one HIPTENSOR_STATUS_ARCH_MISMATCH with an install-defect reason).
            static hiptensorStatus_t
                load(std::string const& path, std::unique_ptr<Pack>& out, std::string& reason);

            std::string const& arch() const
            {
                return mArch;
            }
            std::string const& path() const
            {
                return mPath;
            }
            std::string const& packId() const
            {
                return mPackId;
            }
            std::string const& compilerId() const
            {
                return mCompilerId;
            }
            std::vector<PackKernel> const& kernels() const
            {
                return mKernels;
            }

            // The route value of `key`, or `fallback`.
            std::string route(std::string const& key, std::string const& fallback = "") const;
            double      routeDouble(std::string const& key, double fallback) const;

            // The first kernel of `family` whose integer and string attributes match.
            PackKernel const* find(std::string_view                                        family,
                                   std::vector<std::pair<std::string, int64_t>> const&     ints,
                                   std::vector<std::pair<std::string, std::string>> const& strs
                                   = {}) const;

            // The kernel named `name`, or nullptr.
            PackKernel const* byName(std::string const& name) const;

            // Loads the kernel's code object on the current device (once per device) and returns
            // its function.
            hiptensorStatus_t function(PackKernel const& kernel, hipFunction_t& out) const;

        private:
            std::string                        mPath;
            std::string                        mArch;
            std::string                        mPackId;
            std::string                        mCompilerId;
            std::string                        mRockeVersion;
            std::vector<uint8_t>               mBytes;
            uint64_t                           mBlobOffset = 0;
            uint64_t                           mBlobSize   = 0;
            std::vector<PackKernel>            mKernels;
            std::map<std::string, std::string> mRoutes;

            mutable std::mutex                                        mMutex;
            mutable std::map<std::pair<uint32_t, int>, hipFunction_t> mFunctions;
        };

        // The pack of a GPU target (base name, e.g. "gfx950"), mapped once per process. nullptr
        // with `reason` set when there is none.
        Pack const* packFor(std::string const& arch, std::string& reason);

        // floor(x / d) == umulhi(x, m) >> s for 0 <= x < 2^31; (0, 0) for d == 1 (the kernels
        // then use x itself). Matches hiptensor_aot.common.magic_u31.
        std::pair<uint32_t, uint32_t> magicU31(uint32_t d);

        // SHA-256 of a byte range.
        std::array<uint8_t, 32> sha256(uint8_t const* data, size_t size);

        // Where a pointer argument points at execution.
        enum class BufferRole : uint8_t
        {
            None,
            A,
            B,
            C,
            D, // the additive input of a trinary contraction (Buffers::D)
            Out,
            Workspace,
        };

        // Which host scalar a scalar argument takes.
        enum class ScalarRole : uint8_t
        {
            Alpha,
            Beta,
            Gamma,
            One,
            Zero,
        };

        // One kernel launch of a prepared program: a kernel argument template plus the
        // arguments execution fills in (buffer pointers and host scalars).
        struct LaunchStep
        {
            PackKernel const*    kernel   = nullptr;
            hipFunction_t        function = nullptr;
            dim3                 grid{1, 1, 1};
            dim3                 block{1, 1, 1};
            std::vector<uint8_t> args;

            struct PointerPatch
            {
                uint32_t   offset = 0;
                BufferRole role   = BufferRole::None;
                uint64_t   bytes  = 0; // offset into the buffer
                // the address this kernel argument needs to be a multiple of (0: none); a launch
                // with a misaligned address takes the fallback
                uint32_t align = 0;
            };
            struct ScalarPatch
            {
                uint32_t   offset = 0;
                ScalarRole role   = ScalarRole::One;
                bool       imag   = false; // the imaginary part of a complex scalar
                bool       f64    = false;
                // the scalar negated: a NEG operator of a real input folded into its scale
                // (alpha * -x == -alpha * x, bit for bit; the kernel takes the identity)
                bool negate = false;
            };
            std::vector<PointerPatch> pointers;
            std::vector<ScalarPatch>  scalars;
            // The launches this one replaces (a specialized JIT launch: the pack launch; a tiled
            // pack or epilogue of a contraction: the generic launches, one per piece), which
            // execution takes instead when a pointer misses its alignment, or, for a launch whose
            // code exists only on the device the plan was created on (`deviceBound`: a JIT
            // kernel), when the plan runs on another device. A program that moves a launch's
            // buffers (the steps of a trinary contraction) moves its fallback's the same way.
            std::shared_ptr<std::vector<LaunchStep> const> fallback;
            bool                                           deviceBound = false;
        };

        // Zero-fills a strided region of the workspace (padding of packed contraction operands,
        // stream-K flags): `height` rows of `width` bytes, `pitch` apart, before launch
        // `beforeStep` of the program (0: before every launch). A program that reuses a region
        // of the workspace for several steps (the two steps of a trinary contraction, sliced
        // launches) fills it right before the launches that need it.
        struct MemsetStep
        {
            uint64_t offset     = 0;
            uint64_t pitch      = 0;
            uint64_t width      = 0;
            uint64_t height     = 0;
            uint32_t beforeStep = 0;
        };

        // The launches of a plan, in order. Built at plan creation; execution only patches
        // pointers and scalars and launches (no allocation, no module load, no synchronization).
        struct Program
        {
            std::vector<MemsetStep> memsets;
            std::vector<LaunchStep> steps;
            uint64_t                workspaceBytes = 0;
            std::string             description;
            // How execution reads the host scalars.
            hiptensorDataType_t scalarType     = HIPTENSOR_R_32F;
            bool                complexScalars = false;
            // The pack the kernels come from and the device their functions were loaded on.
            Pack const* pack   = nullptr;
            int         device = -1;
            // Execution checks of the API: host scalars that must be given (0 alpha, 1 beta,
            // 2 gamma), and buffers that must be given, always (scale -1) or when that scale is
            // not zero; a violation returns nullStatus before anything is launched.
            std::vector<int>                        requiredScalars;
            std::vector<std::pair<BufferRole, int>> requiredBuffers;
            hiptensorStatus_t                       nullStatus = HIPTENSOR_STATUS_NOT_INITIALIZED;
            // Another program of the same problem, with no more workspace, that execution runs
            // instead when a host scalar of `fallbackOnZero` is zero (a kernel that combines the
            // terms of zero scales would let a NaN or Inf of a tensor it should not read reach
            // the result) or a buffer of `fallbackAlign` is not a multiple of its alignment.
            std::shared_ptr<Program>                     fallback;
            std::vector<ScalarRole>                      fallbackOnZero;
            std::vector<std::pair<BufferRole, uint32_t>> fallbackAlign;
            // The program's kernels exist only on the device it was created on (JIT kernels):
            // on another device execution runs the fallback.
            bool deviceBound = false;
            // The launches are not built yet (a JIT candidate before buildJitCandidate): never
            // executed.
            bool pending = false;
        };

        // Builds argument buffers of one kernel by argument name.
        class ArgWriter
        {
        public:
            explicit ArgWriter(LaunchStep& step);

            // Set an integer argument (the size the kernel declares: 4 or 8 bytes). False when the
            // kernel has no such argument or the value does not fit.
            bool set(std::string_view name, int64_t value);
            // Set a 32-bit argument to the bits of an unsigned value (magic numbers).
            bool setU32(std::string_view name, uint32_t value);
            bool setF32(std::string_view name, float value);
            bool setF64(std::string_view name, double value);
            bool pointer(std::string_view name, BufferRole role, uint64_t bytes = 0);
            bool scalar(std::string_view name,
                        ScalarRole       role,
                        bool             imag   = false,
                        bool             negate = false);

            // Whether every argument was written.
            bool complete(std::string& missing) const;

        private:
            KernelArg const*  arg(std::string_view name);
            LaunchStep&       mStep;
            std::vector<bool> mWritten;
        };

        // Executes a program.
        hiptensorStatus_t runProgram(Program const&          program,
                                     backend::Scalars const& scalars,
                                     backend::Buffers const& buffers,
                                     void*                   workspace,
                                     uint64_t                workspaceSize,
                                     hipStream_t             stream);
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_ROCKE_ROCKE_RUNTIME_HPP
