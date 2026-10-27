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

// The file formats of the rocKE kernel pack builder (rocke_pack_builder): the code object
// metadata it reads (NT_AMDGPU_METADATA), the binary pack it writes (rocke.htpk, read by
// rocke_runtime.cpp) and its JSON index (rocke.index.json, tooling only). Host-only: no HIP.

#ifndef HIPTENSOR_BACKENDS_ROCKE_PACK_ROCKE_PACK_FORMAT_HPP
#define HIPTENSOR_BACKENDS_ROCKE_PACK_ROCKE_PACK_FORMAT_HPP

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace hiptensor
{
    namespace rockepack
    {
        // Must match hiptensor::rockebackend::kKernelAbi and the reader in rocke_runtime.cpp.
        constexpr uint32_t kKernelAbi         = 1;
        constexpr uint32_t kPackFormatVersion = 1;
        constexpr uint32_t kPackMinReader     = 1;

        constexpr uint32_t kArgKindPointer = 0;
        constexpr uint32_t kArgKindValue   = 1;

        // SHA-256 (FIPS 180-4).
        class Sha256
        {
        public:
            Sha256();
            void                    update(void const* data, size_t size);
            std::array<uint8_t, 32> digest();

        private:
            void                    block(uint8_t const* p);
            std::array<uint32_t, 8> mH;
            std::array<uint8_t, 64> mBuf{};
            size_t                  mBufLen = 0;
            uint64_t                mBits   = 0;
        };

        std::array<uint8_t, 32> sha256(void const* data, size_t size);
        std::string             hex(uint8_t const* data, size_t size);

        // A kernel attribute: an integer or a string.
        using AttrValue = std::variant<int64_t, std::string>;
        using Attrs     = std::map<std::string, AttrValue>;

        // One kernel argument of the code object metadata (hidden arguments dropped).
        struct MdArg
        {
            std::string name;
            uint32_t    offset = 0;
            uint32_t    size   = 0;
            uint32_t    kind   = kArgKindValue;
        };

        // The kernel description the pack records (make_rocke_pack.py _md_subset).
        struct KernelMd
        {
            std::vector<MdArg> args;
            uint32_t           kernargSize    = 0;
            uint32_t           groupSegment   = 0;
            uint32_t           privateSegment = 0;
            uint32_t           maxWorkgroup   = 0;
            uint32_t           wavefrontSize  = 64;
            uint32_t           sgpr           = 0;
            uint32_t           vgpr           = 0;
            uint32_t           agpr           = 0;
            std::string        symbol;
        };

        // The description of kernel `name` in an AMDGPU code object. False with `reason` set
        // when the code object is not an ELF64 file, has no AMDGPU metadata note or no such
        // kernel.
        bool readKernelMd(std::vector<uint8_t> const& co,
                          std::string const&          name,
                          KernelMd&                   out,
                          std::string&                reason);

        // One compiled kernel of a pack.
        struct PackEntry
        {
            std::string          name;
            Attrs                attrs; // must hold "family"
            std::vector<uint8_t> co;
            KernelMd             md;
        };

        // The pack's key of a kernel: "hiptensor-rocke/v<abi>|<family>|<arch>|<name>".
        std::string specKey(std::string const& arch, Attrs const& attrs, std::string const& name);

        // Writes <dir>/rocke.htpk and <dir>/rocke.index.json, byte for byte the files
        // make_rocke_pack.py writes for the same inputs. `kernels` sorted by name. Returns the
        // pack id (hex) and the htpk size; false with `reason` set on an I/O error.
        bool writePack(std::string const&                        dir,
                       std::string const&                        arch,
                       std::vector<PackEntry> const&             kernels,
                       std::map<std::string, std::string> const& routes,
                       std::string const&                        compilerId,
                       std::string const&                        rockeVersion,
                       std::string&                              packId,
                       size_t&                                   size,
                       std::string&                              reason);
    } // namespace rockepack
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_ROCKE_PACK_ROCKE_PACK_FORMAT_HPP
