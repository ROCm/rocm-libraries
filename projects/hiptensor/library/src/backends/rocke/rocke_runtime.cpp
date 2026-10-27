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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>

#include <hip/hip_complex.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "data_types.hpp"
#include "logger.hpp"
#include "rocke_backend.hpp"
#include "rocke_runtime.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        namespace
        {
            // Pack format, see pack/make_rocke_pack.py.
            constexpr char     kPackMagic[8]       = {'H', 'T', 'P', 'K', 'I', 'D', 'X', '\0'};
            constexpr uint32_t kPackReaderVersion  = 1u;
            constexpr size_t   kHeaderSize         = 160;
            constexpr size_t   kKernelRecordSize   = 96;
            constexpr size_t   kArgRecordSize      = 16;
            constexpr size_t   kAttrRecordSize     = 16;
            constexpr size_t   kRouteRecordSize    = 8;
            constexpr uint32_t kArgKindPointer     = 0u;
            constexpr uint32_t kAttrKindString     = 1u;
            constexpr char     kPackFileName[]     = "rocke.htpk";
            constexpr char     kPackPathVariable[] = "HIPTENSOR_ROCKE_PACK_PATH";

            template <typename T>
            T readLe(std::vector<uint8_t> const& bytes, size_t offset)
            {
                T value;
                std::memcpy(&value, bytes.data() + offset, sizeof(T));
                return value;
            }

            // Bounds-checked view of the string table.
            class Strings
            {
            public:
                Strings(std::vector<uint8_t> const& bytes, uint64_t offset, uint64_t size)
                    : mBytes(bytes)
                    , mOffset(offset)
                    , mSize(size)
                {
                }

                bool get(uint32_t at, std::string& out) const
                {
                    if(at >= mSize)
                    {
                        return false;
                    }
                    auto const* begin = reinterpret_cast<char const*>(mBytes.data() + mOffset);
                    auto const* end
                        = static_cast<char const*>(std::memchr(begin + at, '\0', mSize - at));
                    if(end == nullptr)
                    {
                        return false;
                    }
                    out.assign(begin + at, end);
                    return true;
                }

            private:
                std::vector<uint8_t> const& mBytes;
                uint64_t                    mOffset;
                uint64_t                    mSize;
            };

            bool inRange(uint64_t offset, uint64_t size, uint64_t total)
            {
                return offset <= total && size <= total - offset;
            }

            std::string toHex(std::array<uint8_t, 32> const& digest)
            {
                static constexpr char hexDigits[] = "0123456789abcdef";
                std::string           out;
                for(auto byte : digest)
                {
                    out += hexDigits[byte >> 4];
                    out += hexDigits[byte & 15];
                }
                return out;
            }

            // The directory of the loaded hipTensor library.
            std::string libraryDirectory()
            {
#ifdef _WIN32
                HMODULE module = nullptr;
                if(!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                                           | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                       reinterpret_cast<LPCSTR>(&packFor),
                                       &module))
                {
                    return {};
                }
                char path[MAX_PATH];
                auto size = GetModuleFileNameA(module, path, MAX_PATH);
                if(size == 0 || size >= MAX_PATH)
                {
                    return {};
                }
                std::string file(path, size);
                auto        slash = file.find_last_of("\\/");
#else
                Dl_info info;
                if(dladdr(reinterpret_cast<void*>(&packFor), &info) == 0
                   || info.dli_fname == nullptr)
                {
                    return {};
                }
                std::string file(info.dli_fname);
                auto        slash = file.find_last_of('/');
#endif
                return slash == std::string::npos ? std::string(".") : file.substr(0, slash);
            }

            std::vector<std::string> packCandidates(std::string const& arch)
            {
                std::vector<std::string> out;
                auto                     add = [&](std::string const& dir) {
                    if(!dir.empty())
                    {
                        out.push_back(dir + "/" + arch + "/" + kPackFileName);
                    }
                };
                if(char const* env = std::getenv(kPackPathVariable); env != nullptr && *env)
                {
                    std::string list(env);
                    size_t      start = 0;
                    while(start <= list.size())
                    {
                        auto end = list.find(':', start);
                        if(end == std::string::npos)
                        {
                            end = list.size();
                        }
                        add(list.substr(start, end - start));
                        start = end + 1;
                    }
                    // An explicit path is the only place looked at.
                    return out;
                }
#ifdef HIPTENSOR_ROCKE_PACK_RELATIVE_DIR
                if(auto dir = libraryDirectory(); !dir.empty())
                {
                    add(dir + "/" + HIPTENSOR_ROCKE_PACK_RELATIVE_DIR);
                }
#endif
#ifdef HIPTENSOR_ROCKE_PACK_INSTALL_DIR
                add(HIPTENSOR_ROCKE_PACK_INSTALL_DIR);
#endif
#ifdef HIPTENSOR_ROCKE_PACK_BUILD_DIR
                add(HIPTENSOR_ROCKE_PACK_BUILD_DIR);
#endif
                return out;
            }

            // SHA-256 (FIPS 180-4).
            class Sha256
            {
            public:
                void update(uint8_t const* data, size_t size)
                {
                    for(size_t i = 0; i < size; i++)
                    {
                        mBlock[mFill++] = data[i];
                        if(mFill == 64)
                        {
                            compress();
                            mFill = 0;
                        }
                    }
                    mBits += static_cast<uint64_t>(size) * 8;
                }

                std::array<uint8_t, 32> finish()
                {
                    uint64_t bits = mBits;
                    uint8_t  pad  = 0x80;
                    update(&pad, 1);
                    uint8_t zero = 0;
                    while(mFill != 56)
                    {
                        update(&zero, 1);
                    }
                    for(int i = 7; i >= 0; i--)
                    {
                        uint8_t byte = static_cast<uint8_t>(bits >> (8 * i));
                        update(&byte, 1);
                    }
                    std::array<uint8_t, 32> out{};
                    for(int i = 0; i < 8; i++)
                    {
                        for(int j = 0; j < 4; j++)
                        {
                            out[4 * i + j] = static_cast<uint8_t>(mState[i] >> (24 - 8 * j));
                        }
                    }
                    return out;
                }

            private:
                static uint32_t rotr(uint32_t x, int n)
                {
                    return (x >> n) | (x << (32 - n));
                }

                void compress()
                {
                    static constexpr uint32_t k[64]
                        = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
                           0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
                           0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
                           0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
                           0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
                           0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
                           0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
                           0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
                           0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
                           0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
                           0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
                    uint32_t w[64];
                    for(int i = 0; i < 16; i++)
                    {
                        w[i] = (uint32_t(mBlock[4 * i]) << 24) | (uint32_t(mBlock[4 * i + 1]) << 16)
                               | (uint32_t(mBlock[4 * i + 2]) << 8) | uint32_t(mBlock[4 * i + 3]);
                    }
                    for(int i = 16; i < 64; i++)
                    {
                        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                        w[i]        = w[i - 16] + s0 + w[i - 7] + s1;
                    }
                    uint32_t a = mState[0], b = mState[1], c = mState[2], d = mState[3];
                    uint32_t e = mState[4], f = mState[5], g = mState[6], h = mState[7];
                    for(int i = 0; i < 64; i++)
                    {
                        uint32_t s1  = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
                        uint32_t ch  = (e & f) ^ (~e & g);
                        uint32_t t1  = h + s1 + ch + k[i] + w[i];
                        uint32_t s0  = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
                        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                        uint32_t t2  = s0 + maj;
                        h            = g;
                        g            = f;
                        f            = e;
                        e            = d + t1;
                        d            = c;
                        c            = b;
                        b            = a;
                        a            = t1 + t2;
                    }
                    mState[0] += a;
                    mState[1] += b;
                    mState[2] += c;
                    mState[3] += d;
                    mState[4] += e;
                    mState[5] += f;
                    mState[6] += g;
                    mState[7] += h;
                }

                uint32_t mState[8] = {0x6a09e667,
                                      0xbb67ae85,
                                      0x3c6ef372,
                                      0xa54ff53a,
                                      0x510e527f,
                                      0x9b05688c,
                                      0x1f83d9ab,
                                      0x5be0cd19};
                uint8_t  mBlock[64]{};
                size_t   mFill = 0;
                uint64_t mBits = 0;
            };
        } // namespace

        std::array<uint8_t, 32> sha256(uint8_t const* data, size_t size)
        {
            Sha256 hash;
            hash.update(data, size);
            return hash.finish();
        }

        std::pair<uint32_t, uint32_t> magicU31(uint32_t d)
        {
            if(d <= 1u)
            {
                return {0u, 0u};
            }
            constexpr uint64_t limit = uint64_t(1) << 31;
            for(uint32_t s = 32; s < 64; s++)
            {
                // m = ceil(2^s / d), exact in 128-bit arithmetic
                unsigned __int128 p = static_cast<unsigned __int128>(1) << s;
                unsigned __int128 m = (p + d - 1) / d;
                if(m >= (static_cast<unsigned __int128>(1) << 32))
                {
                    break;
                }
                unsigned __int128 e = m * d - p;
                if(e * (limit - 1) < p)
                {
                    return {static_cast<uint32_t>(m), s - 32};
                }
            }
            return {0u, 0u}; // unreachable for 1 < d < 2^31
        }

        // ------------------------------------------------------------------------- PackKernel

        int64_t PackKernel::attr(std::string_view key, int64_t fallback) const
        {
            auto found = intAttrs.find(std::string(key));
            return found == intAttrs.end() ? fallback : found->second;
        }

        std::string PackKernel::attrStr(std::string_view key) const
        {
            auto found = strAttrs.find(std::string(key));
            return found == strAttrs.end() ? std::string() : found->second;
        }

        int PackKernel::argIndex(std::string_view argName) const
        {
            for(size_t i = 0; i < args.size(); i++)
            {
                if(args[i].name == argName)
                {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

        // ------------------------------------------------------------------------------- Pack

        hiptensorStatus_t
            Pack::load(std::string const& path, std::unique_ptr<Pack>& out, std::string& reason)
        {
            std::ifstream file(path, std::ios::binary);
            if(!file)
            {
                reason = "no rocKE pack at " + path;
                return HIPTENSOR_STATUS_ARCH_MISMATCH;
            }
            auto pack   = std::make_unique<Pack>();
            pack->mPath = path;
            pack->mBytes.assign(std::istreambuf_iterator<char>(file),
                                std::istreambuf_iterator<char>());
            auto const& bytes = pack->mBytes;
            auto        bad   = [&](std::string const& what) {
                reason = "rocKE pack " + path + " is not usable (install defect): " + what;
                return HIPTENSOR_STATUS_ARCH_MISMATCH;
            };
            if(bytes.size() < kHeaderSize || std::memcmp(bytes.data(), kPackMagic, 8) != 0)
            {
                return bad("bad magic");
            }
            auto formatVersion = readLe<uint32_t>(bytes, 8);
            auto minReader     = readLe<uint32_t>(bytes, 12);
            auto kernelAbi     = readLe<uint32_t>(bytes, 16);
            auto kernelCount   = readLe<uint32_t>(bytes, 20);
            auto argCount      = readLe<uint32_t>(bytes, 24);
            auto attrCount     = readLe<uint32_t>(bytes, 28);
            auto routeCount    = readLe<uint32_t>(bytes, 32);
            auto kernelTable   = readLe<uint64_t>(bytes, 40);
            auto argTable      = readLe<uint64_t>(bytes, 48);
            auto attrTable     = readLe<uint64_t>(bytes, 56);
            auto routeTable    = readLe<uint64_t>(bytes, 64);
            auto stringTable   = readLe<uint64_t>(bytes, 72);
            auto stringSize    = readLe<uint64_t>(bytes, 80);
            pack->mBlobOffset  = readLe<uint64_t>(bytes, 88);
            pack->mBlobSize    = readLe<uint64_t>(bytes, 96);
            if(minReader > kPackReaderVersion || formatVersion < 1u)
            {
                return bad("format version " + std::to_string(formatVersion) + " needs reader "
                           + std::to_string(minReader));
            }
            if(kernelAbi != kKernelAbi)
            {
                return bad("kernel ABI " + std::to_string(kernelAbi) + ", this library uses "
                           + std::to_string(kKernelAbi));
            }
            uint64_t total = bytes.size();
            if(!inRange(kernelTable, uint64_t(kernelCount) * kKernelRecordSize, total)
               || !inRange(argTable, uint64_t(argCount) * kArgRecordSize, total)
               || !inRange(attrTable, uint64_t(attrCount) * kAttrRecordSize, total)
               || !inRange(routeTable, uint64_t(routeCount) * kRouteRecordSize, total)
               || !inRange(stringTable, stringSize, total)
               || !inRange(pack->mBlobOffset, pack->mBlobSize, total))
            {
                return bad("a table lies outside the file");
            }
            Strings strings(bytes, stringTable, stringSize);
            if(!strings.get(readLe<uint32_t>(bytes, 104), pack->mArch)
               || !strings.get(readLe<uint32_t>(bytes, 108), pack->mCompilerId)
               || !strings.get(readLe<uint32_t>(bytes, 112), pack->mRockeVersion)
               || !strings.get(readLe<uint32_t>(bytes, 116), pack->mPackId))
            {
                return bad("bad header strings");
            }
            for(uint32_t r = 0; r < routeCount; r++)
            {
                auto        at = routeTable + uint64_t(r) * kRouteRecordSize;
                std::string key, value;
                if(!strings.get(readLe<uint32_t>(bytes, at), key)
                   || !strings.get(readLe<uint32_t>(bytes, at + 4), value))
                {
                    return bad("bad route record");
                }
                pack->mRoutes[key] = value;
            }
            pack->mKernels.resize(kernelCount);
            for(uint32_t i = 0; i < kernelCount; i++)
            {
                auto  at         = kernelTable + uint64_t(i) * kKernelRecordSize;
                auto& k          = pack->mKernels[i];
                k.index          = i;
                auto argFirst    = readLe<uint32_t>(bytes, at + 12);
                auto argN        = readLe<uint32_t>(bytes, at + 16);
                auto attrFirst   = readLe<uint32_t>(bytes, at + 20);
                auto attrN       = readLe<uint32_t>(bytes, at + 24);
                k.kernargSize    = readLe<uint32_t>(bytes, at + 28);
                k.groupSegment   = readLe<uint32_t>(bytes, at + 32);
                k.privateSegment = readLe<uint32_t>(bytes, at + 36);
                k.maxWorkgroup   = readLe<uint32_t>(bytes, at + 40);
                k.wavefrontSize  = readLe<uint32_t>(bytes, at + 44);
                k.codeOffset     = readLe<uint64_t>(bytes, at + 48);
                k.codeSize       = readLe<uint64_t>(bytes, at + 56);
                std::memcpy(k.codeSha256.data(), bytes.data() + at + 64, 32);
                if(!strings.get(readLe<uint32_t>(bytes, at), k.name)
                   || !strings.get(readLe<uint32_t>(bytes, at + 4), k.family)
                   || !strings.get(readLe<uint32_t>(bytes, at + 8), k.specKey))
                {
                    return bad("bad kernel record strings");
                }
                if(uint64_t(argFirst) + argN > argCount || uint64_t(attrFirst) + attrN > attrCount
                   || !inRange(k.codeOffset, k.codeSize, pack->mBlobSize))
                {
                    return bad("kernel " + k.name + " refers outside the tables");
                }
                for(uint32_t a = 0; a < argN; a++)
                {
                    auto      aat = argTable + uint64_t(argFirst + a) * kArgRecordSize;
                    KernelArg arg;
                    if(!strings.get(readLe<uint32_t>(bytes, aat), arg.name))
                    {
                        return bad("bad argument name");
                    }
                    arg.offset  = readLe<uint32_t>(bytes, aat + 4);
                    arg.size    = readLe<uint32_t>(bytes, aat + 8);
                    arg.pointer = readLe<uint32_t>(bytes, aat + 12) == kArgKindPointer;
                    if(uint64_t(arg.offset) + arg.size > k.kernargSize)
                    {
                        return bad("argument " + arg.name + " of " + k.name
                                   + " lies outside the kernel arguments");
                    }
                    k.args.push_back(arg);
                }
                for(uint32_t a = 0; a < attrN; a++)
                {
                    auto        tat = attrTable + uint64_t(attrFirst + a) * kAttrRecordSize;
                    std::string key;
                    if(!strings.get(readLe<uint32_t>(bytes, tat), key))
                    {
                        return bad("bad attribute name");
                    }
                    auto kind  = readLe<uint32_t>(bytes, tat + 4);
                    auto value = readLe<int64_t>(bytes, tat + 8);
                    if(kind == kAttrKindString)
                    {
                        std::string text;
                        if(value < 0 || !strings.get(static_cast<uint32_t>(value), text))
                        {
                            return bad("bad attribute value");
                        }
                        k.strAttrs[key] = text;
                    }
                    else
                    {
                        k.intAttrs[key] = value;
                    }
                }
            }
            out = std::move(pack);
            return HIPTENSOR_STATUS_SUCCESS;
        }

        std::string Pack::route(std::string const& key, std::string const& fallback) const
        {
            auto found = mRoutes.find(key);
            return found == mRoutes.end() ? fallback : found->second;
        }

        double Pack::routeDouble(std::string const& key, double fallback) const
        {
            auto value = route(key);
            if(value.empty())
            {
                return fallback;
            }
            char* end    = nullptr;
            auto  parsed = std::strtod(value.c_str(), &end);
            return end == value.c_str() ? fallback : parsed;
        }

        PackKernel const*
            Pack::find(std::string_view                                        family,
                       std::vector<std::pair<std::string, int64_t>> const&     ints,
                       std::vector<std::pair<std::string, std::string>> const& strs) const
        {
            for(auto const& kernel : mKernels)
            {
                if(kernel.family != family)
                {
                    continue;
                }
                bool match = true;
                for(auto const& [key, value] : ints)
                {
                    match = match && kernel.attr(key, INT64_MIN) == value;
                }
                for(auto const& [key, value] : strs)
                {
                    match = match && kernel.attrStr(key) == value;
                }
                if(match)
                {
                    return &kernel;
                }
            }
            return nullptr;
        }

        PackKernel const* Pack::byName(std::string const& name) const
        {
            for(auto const& kernel : mKernels)
            {
                if(kernel.name == name)
                {
                    return &kernel;
                }
            }
            return nullptr;
        }

        hiptensorStatus_t Pack::function(PackKernel const& kernel, hipFunction_t& out) const
        {
            int device = 0;
            if(hipGetDevice(&device) != hipSuccess)
            {
                return HIPTENSOR_STATUS_HIP_ERROR;
            }
            std::lock_guard<std::mutex> lock(mMutex);
            auto                        key   = std::make_pair(kernel.index, device);
            auto                        found = mFunctions.find(key);
            if(found != mFunctions.end())
            {
                out = found->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            auto const* code = mBytes.data() + mBlobOffset + kernel.codeOffset;
            if(sha256(code, kernel.codeSize) != kernel.codeSha256)
            {
                char msg[512];
                snprintf(msg,
                         sizeof(msg),
                         "rocKE pack %s: code object of %s does not match its SHA-256 %s",
                         mPath.c_str(),
                         kernel.name.c_str(),
                         toHex(kernel.codeSha256).c_str());
                Logger::instance()->logError("hiptensorCreatePlan", msg);
                return HIPTENSOR_STATUS_INTERNAL_ERROR;
            }
            // Code objects are loaded once per device and stay loaded for the process lifetime.
            hipModule_t module = nullptr;
            auto        status = hipModuleLoadData(&module, code);
            if(status == hipErrorOutOfMemory)
            {
                return HIPTENSOR_STATUS_ALLOC_FAILED;
            }
            if(status != hipSuccess)
            {
                return HIPTENSOR_STATUS_HIP_ERROR;
            }
            hipFunction_t function = nullptr;
            if(hipModuleGetFunction(&function, module, kernel.name.c_str()) != hipSuccess)
            {
                static_cast<void>(hipModuleUnload(module));
                return HIPTENSOR_STATUS_HIP_ERROR;
            }
            mFunctions[key] = function;
            out             = function;
            return HIPTENSOR_STATUS_SUCCESS;
        }

        Pack const* packFor(std::string const& arch, std::string& reason)
        {
            struct Entry
            {
                std::unique_ptr<Pack> pack;
                std::string           reason;
            };
            static std::mutex                   mutex;
            static std::map<std::string, Entry> packs;
            std::lock_guard<std::mutex>         lock(mutex);
            auto                                found = packs.find(arch);
            if(found == packs.end())
            {
                Entry       entry;
                std::string tried;
                for(auto const& path : packCandidates(arch))
                {
                    std::string why;
                    auto        status = Pack::load(path, entry.pack, why);
                    if(status == HIPTENSOR_STATUS_SUCCESS)
                    {
                        if(entry.pack->arch() == arch)
                        {
                            break;
                        }
                        why = "rocKE pack " + path + " is for " + entry.pack->arch();
                        entry.pack.reset();
                    }
                    tried += (tried.empty() ? "" : "; ") + why;
                }
                if(entry.pack == nullptr)
                {
                    entry.reason = "no usable rocKE pack for " + arch
                                   + (tried.empty() ? std::string(" (no pack path configured)")
                                                    : " (" + tried + ")");
                }
                else
                {
                    char msg[1024];
                    snprintf(msg,
                             sizeof(msg),
                             "rocKE pack %s: %zu kernels, pack id %.16s, compiler %s",
                             entry.pack->path().c_str(),
                             entry.pack->kernels().size(),
                             entry.pack->packId().c_str(),
                             entry.pack->compilerId().c_str());
                    Logger::instance()->logHeuristics("hiptensorCreatePlan", msg);
                }
                found = packs.emplace(arch, std::move(entry)).first;
            }
            reason = found->second.reason;
            return found->second.pack.get();
        }

        // --------------------------------------------------------------------------- ArgWriter

        ArgWriter::ArgWriter(LaunchStep& step)
            : mStep(step)
            , mWritten(step.kernel->args.size(), false)
        {
            mStep.args.assign(step.kernel->kernargSize, 0);
        }

        KernelArg const* ArgWriter::arg(std::string_view name)
        {
            auto index = mStep.kernel->argIndex(name);
            if(index < 0)
            {
                return nullptr;
            }
            mWritten[index] = true;
            return &mStep.kernel->args[index];
        }

        bool ArgWriter::set(std::string_view name, int64_t value)
        {
            auto const* a = arg(name);
            if(a == nullptr || a->pointer)
            {
                return false;
            }
            if(a->size == 4)
            {
                if(value < INT32_MIN || value > INT32_MAX)
                {
                    return false;
                }
                auto v = static_cast<int32_t>(value);
                std::memcpy(mStep.args.data() + a->offset, &v, 4);
                return true;
            }
            if(a->size == 8)
            {
                std::memcpy(mStep.args.data() + a->offset, &value, 8);
                return true;
            }
            return false;
        }

        bool ArgWriter::setU32(std::string_view name, uint32_t value)
        {
            auto const* a = arg(name);
            if(a == nullptr || a->pointer || a->size != 4)
            {
                return false;
            }
            std::memcpy(mStep.args.data() + a->offset, &value, 4);
            return true;
        }

        bool ArgWriter::setF32(std::string_view name, float value)
        {
            auto const* a = arg(name);
            if(a == nullptr || a->size != 4)
            {
                return false;
            }
            std::memcpy(mStep.args.data() + a->offset, &value, 4);
            return true;
        }

        bool ArgWriter::setF64(std::string_view name, double value)
        {
            auto const* a = arg(name);
            if(a == nullptr || a->size != 8)
            {
                return false;
            }
            std::memcpy(mStep.args.data() + a->offset, &value, 8);
            return true;
        }

        bool ArgWriter::pointer(std::string_view name, BufferRole role, uint64_t bytes)
        {
            auto const* a = arg(name);
            if(a == nullptr || !a->pointer || a->size != 8)
            {
                return false;
            }
            mStep.pointers.push_back({a->offset, role, bytes});
            return true;
        }

        bool ArgWriter::scalar(std::string_view name, ScalarRole role, bool imag)
        {
            auto const* a = arg(name);
            if(a == nullptr || a->pointer || (a->size != 4 && a->size != 8))
            {
                return false;
            }
            mStep.scalars.push_back({a->offset, role, imag, a->size == 8});
            return true;
        }

        bool ArgWriter::complete(std::string& missing) const
        {
            for(size_t i = 0; i < mWritten.size(); i++)
            {
                if(!mWritten[i])
                {
                    missing = mStep.kernel->args[i].name;
                    return false;
                }
            }
            return true;
        }

        // --------------------------------------------------------------------------- execution

        namespace
        {
            struct HostScalar
            {
                double re = 0.0;
                double im = 0.0;
            };

            HostScalar readScalar(void const* value, hiptensorDataType_t type, bool complexValue)
            {
                HostScalar out;
                if(value == nullptr)
                {
                    return out;
                }
                if(complexValue)
                {
                    if(type == HIPTENSOR_C_64F)
                    {
                        auto v = *static_cast<hipDoubleComplex const*>(value);
                        out.re = hipCreal(v);
                        out.im = hipCimag(v);
                    }
                    else
                    {
                        auto v = *static_cast<hipFloatComplex const*>(value);
                        out.re = hipCrealf(v);
                        out.im = hipCimagf(v);
                    }
                    return out;
                }
                out.re = hiptensor::readVal<double>(value, type);
                return out;
            }

            void const* bufferOf(BufferRole role, backend::Buffers const& buffers, void* workspace)
            {
                switch(role)
                {
                case BufferRole::A:
                    return buffers.A;
                case BufferRole::B:
                    return buffers.B;
                case BufferRole::C:
                    return buffers.C;
                case BufferRole::D:
                    return buffers.D;
                case BufferRole::Out:
                    return buffers.out;
                case BufferRole::Workspace:
                    return workspace;
                case BufferRole::None:
                    break;
                }
                return nullptr;
            }

            constexpr size_t kMaxKernargBytes = 4096;
            constexpr size_t kMaxKernelArgs   = 512;
        } // namespace

        hiptensorStatus_t runProgram(Program const&          program,
                                     backend::Scalars const& scalars,
                                     backend::Buffers const& buffers,
                                     void*                   workspace,
                                     uint64_t                workspaceSize,
                                     hipStream_t             stream)
        {
            void const* const scalarPtrs[] = {scalars.alpha, scalars.beta, scalars.gamma};
            for(auto s : program.requiredScalars)
            {
                if(scalarPtrs[s] == nullptr)
                {
                    return program.nullStatus;
                }
            }
            if(program.workspaceBytes > 0
               && (workspace == nullptr || workspaceSize < program.workspaceBytes))
            {
                return HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE;
            }
            HostScalar const values[] = {
                readScalar(scalars.alpha, program.scalarType, program.complexScalars),
                readScalar(scalars.beta, program.scalarType, program.complexScalars),
                readScalar(scalars.gamma, program.scalarType, program.complexScalars),
                {1.0, 0.0},
                {0.0, 0.0},
            };
            for(auto const& [role, scale] : program.requiredBuffers)
            {
                bool const needed = scale < 0 || values[scale].re != 0.0 || values[scale].im != 0.0;
                if(needed && bufferOf(role, buffers, workspace) == nullptr)
                {
                    return program.nullStatus;
                }
            }
            // the zero fills that precede launch `index`
            auto memsetsBefore = [&](size_t index) {
                for(auto const& m : program.memsets)
                {
                    if(m.beforeStep != index)
                    {
                        continue;
                    }
                    auto* base   = static_cast<uint8_t*>(workspace) + m.offset;
                    auto  status = m.height == 1 ? hipMemsetAsync(base, 0, m.width, stream)
                                                 : hipMemset2DAsync(
                                                    base, m.pitch, 0, m.width, m.height, stream);
                    if(status != hipSuccess)
                    {
                        return HIPTENSOR_STATUS_HIP_ERROR;
                    }
                }
                return HIPTENSOR_STATUS_SUCCESS;
            };

            int device = 0;
            if(hipGetDevice(&device) != hipSuccess)
            {
                return HIPTENSOR_STATUS_HIP_ERROR;
            }
            for(size_t index = 0; index < program.steps.size(); index++)
            {
                if(auto status = memsetsBefore(index); status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
                auto const&       planned = program.steps[index];
                LaunchStep const* chosen  = &planned;
                if(planned.fallback != nullptr)
                {
                    bool use = device != program.device;
                    for(auto const& p : planned.pointers)
                    {
                        auto const* base = bufferOf(p.role, buffers, workspace);
                        use              = use
                              || (p.align != 0 && base != nullptr
                                  && (reinterpret_cast<uint64_t>(base) + p.bytes) % p.align != 0);
                    }
                    if(use)
                    {
                        chosen = planned.fallback.get();
                    }
                }
                LaunchStep const& step = *chosen;
                if(step.args.size() > kMaxKernargBytes || step.kernel->args.size() > kMaxKernelArgs)
                {
                    return HIPTENSOR_STATUS_INTERNAL_ERROR;
                }
                alignas(16) uint8_t args[kMaxKernargBytes];
                std::memcpy(args, step.args.data(), step.args.size());
                for(auto const& p : step.pointers)
                {
                    auto const* base
                        = static_cast<uint8_t const*>(bufferOf(p.role, buffers, workspace));
                    uint64_t address
                        = base == nullptr ? 0 : reinterpret_cast<uint64_t>(base) + p.bytes;
                    std::memcpy(args + p.offset, &address, 8);
                }
                for(auto const& s : step.scalars)
                {
                    auto const& v     = values[static_cast<int>(s.role)];
                    double      value = s.imag ? v.im : v.re;
                    if(s.f64)
                    {
                        std::memcpy(args + s.offset, &value, 8);
                    }
                    else
                    {
                        float f = static_cast<float>(value);
                        std::memcpy(args + s.offset, &f, 4);
                    }
                }
                void* params[kMaxKernelArgs];
                for(size_t i = 0; i < step.kernel->args.size(); i++)
                {
                    params[i] = args + step.kernel->args[i].offset;
                }
                hipFunction_t function = step.function;
                if(device != program.device && program.pack != nullptr)
                {
                    // Executed on another device than the plan was created on: use (and load
                    // once) the kernel's code object there.
                    if(auto status = program.pack->function(*step.kernel, function);
                       status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        return status;
                    }
                }
                if(function == nullptr)
                {
                    return HIPTENSOR_STATUS_INTERNAL_ERROR;
                }
                auto status = hipModuleLaunchKernel(function,
                                                    step.grid.x,
                                                    step.grid.y,
                                                    step.grid.z,
                                                    step.block.x,
                                                    step.block.y,
                                                    step.block.z,
                                                    0,
                                                    stream,
                                                    params,
                                                    nullptr);
                if(status != hipSuccess)
                {
                    return HIPTENSOR_STATUS_HIP_ERROR;
                }
            }
            return HIPTENSOR_STATUS_SUCCESS;
        }
    } // namespace rockebackend
} // namespace hiptensor
