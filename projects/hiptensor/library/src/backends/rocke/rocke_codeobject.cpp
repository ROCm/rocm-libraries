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

#include <cstring>
#include <map>
#include <memory>

#include "rocke_codeobject.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        namespace
        {
            constexpr uint32_t kShtNote            = 7;
            constexpr uint32_t kNtAmdgpuMetadata   = 32;
            constexpr int      kMaxMsgPackDepth    = 32;
            constexpr size_t   kMaxMsgPackElements = 1u << 20;

            // A decoded msgpack value (the subset AMDGPU metadata uses).
            struct MpValue
            {
                enum class Kind : uint8_t
                {
                    Nil,
                    Bool,
                    Int,
                    Float,
                    String,
                    Array,
                    Map,
                };
                Kind                     kind = Kind::Nil;
                int64_t                  i    = 0;
                double                   f    = 0.0;
                std::string              s;
                std::vector<MpValue>     a; // array elements, or map values
                std::vector<std::string> keys; // map keys

                MpValue const* get(char const* key) const
                {
                    for(size_t k = 0; k < keys.size(); k++)
                    {
                        if(keys[k] == key)
                        {
                            return &a[k];
                        }
                    }
                    return nullptr;
                }

                int64_t intOf(char const* key, int64_t fallback) const
                {
                    auto const* v = get(key);
                    return v != nullptr && v->kind == Kind::Int ? v->i : fallback;
                }

                std::string stringOf(char const* key) const
                {
                    auto const* v = get(key);
                    return v != nullptr && v->kind == Kind::String ? v->s : std::string();
                }
            };

            class MsgPack
            {
            public:
                MsgPack(uint8_t const* data, size_t size)
                    : mData(data)
                    , mSize(size)
                {
                }

                bool value(MpValue& out, int depth = 0)
                {
                    if(depth > kMaxMsgPackDepth)
                    {
                        return false;
                    }
                    uint64_t t = 0;
                    if(!take(1, t))
                    {
                        return false;
                    }
                    if(t <= 0x7F)
                    {
                        return setInt(out, int64_t(t));
                    }
                    if(t >= 0xE0)
                    {
                        return setInt(out, int64_t(t) - 0x100);
                    }
                    if(t >= 0x80 && t <= 0x8F)
                    {
                        return map(out, t & 0x0F, depth);
                    }
                    if(t >= 0x90 && t <= 0x9F)
                    {
                        return array(out, t & 0x0F, depth);
                    }
                    if(t >= 0xA0 && t <= 0xBF)
                    {
                        return string(out, t & 0x1F);
                    }
                    uint64_t n = 0;
                    switch(t)
                    {
                    case 0xC0:
                        out.kind = MpValue::Kind::Nil;
                        return true;
                    case 0xC2:
                    case 0xC3:
                        out.kind = MpValue::Kind::Bool;
                        out.i    = t == 0xC3;
                        return true;
                    case 0xC4:
                    case 0xC5:
                    case 0xC6:
                        return take(t == 0xC4 ? 1 : t == 0xC5 ? 2 : 4, n) && string(out, n);
                    case 0xCA:
                    {
                        uint64_t bits = 0;
                        float    f;
                        uint32_t b32;
                        if(!take(4, bits))
                        {
                            return false;
                        }
                        b32 = uint32_t(bits);
                        std::memcpy(&f, &b32, 4);
                        out.kind = MpValue::Kind::Float;
                        out.f    = f;
                        return true;
                    }
                    case 0xCB:
                    {
                        uint64_t bits = 0;
                        if(!take(8, bits))
                        {
                            return false;
                        }
                        out.kind = MpValue::Kind::Float;
                        std::memcpy(&out.f, &bits, 8);
                        return true;
                    }
                    case 0xCC:
                    case 0xCD:
                    case 0xCE:
                    case 0xCF:
                        return take(size_t(1) << (t - 0xCC), n) && setInt(out, int64_t(n));
                    case 0xD0:
                    case 0xD1:
                    case 0xD2:
                    case 0xD3:
                    {
                        size_t const bytes = size_t(1) << (t - 0xD0);
                        if(!take(bytes, n))
                        {
                            return false;
                        }
                        int64_t v = int64_t(n);
                        if(bytes < 8 && (n >> (bytes * 8 - 1)) != 0)
                        {
                            v = int64_t(n) - (int64_t(1) << (bytes * 8));
                        }
                        return setInt(out, v);
                    }
                    case 0xD9:
                    case 0xDA:
                    case 0xDB:
                        return take(t == 0xD9 ? 1 : t == 0xDA ? 2 : 4, n) && string(out, n);
                    case 0xDC:
                    case 0xDD:
                        return take(t == 0xDC ? 2 : 4, n) && array(out, n, depth);
                    case 0xDE:
                    case 0xDF:
                        return take(t == 0xDE ? 2 : 4, n) && map(out, n, depth);
                    default:
                        return false;
                    }
                }

            private:
                bool take(size_t n, uint64_t& out)
                {
                    if(n > mSize - mAt)
                    {
                        return false;
                    }
                    out = 0;
                    for(size_t i = 0; i < n; i++)
                    {
                        out = (out << 8) | mData[mAt + i];
                    }
                    mAt += n;
                    return true;
                }

                static bool setInt(MpValue& out, int64_t v)
                {
                    out.kind = MpValue::Kind::Int;
                    out.i    = v;
                    return true;
                }

                bool string(MpValue& out, uint64_t n)
                {
                    if(n > mSize - mAt)
                    {
                        return false;
                    }
                    out.kind = MpValue::Kind::String;
                    out.s.assign(reinterpret_cast<char const*>(mData + mAt), size_t(n));
                    mAt += size_t(n);
                    return true;
                }

                bool array(MpValue& out, uint64_t n, int depth)
                {
                    if(n > kMaxMsgPackElements || n > mSize - mAt)
                    {
                        return false;
                    }
                    out.kind = MpValue::Kind::Array;
                    out.a.resize(size_t(n));
                    for(auto& v : out.a)
                    {
                        if(!value(v, depth + 1))
                        {
                            return false;
                        }
                    }
                    return true;
                }

                bool map(MpValue& out, uint64_t n, int depth)
                {
                    if(n > kMaxMsgPackElements || n > mSize - mAt)
                    {
                        return false;
                    }
                    out.kind = MpValue::Kind::Map;
                    out.a.resize(size_t(n));
                    out.keys.resize(size_t(n));
                    for(size_t k = 0; k < out.a.size(); k++)
                    {
                        MpValue key;
                        if(!value(key, depth + 1) || key.kind != MpValue::Kind::String
                           || !value(out.a[k], depth + 1))
                        {
                            return false;
                        }
                        out.keys[k] = std::move(key.s);
                    }
                    return true;
                }

                uint8_t const* mData;
                size_t         mSize;
                size_t         mAt = 0;
            };

            template <typename T>
            bool readAt(uint8_t const* code, size_t size, uint64_t offset, T& out)
            {
                if(offset > size || sizeof(T) > size - offset)
                {
                    return false;
                }
                std::memcpy(&out, code + offset, sizeof(T));
                return true;
            }

            // The NT_AMDGPU_METADATA note of an ELF64 file.
            bool metadataNote(uint8_t const*  code,
                              size_t          size,
                              uint8_t const*& desc,
                              size_t&         descSize)
            {
                if(size < 64
                   || std::memcmp(code,
                                  "\x7f"
                                  "ELF",
                                  4)
                          != 0
                   || code[4] != 2)
                {
                    return false;
                }
                uint64_t shoff     = 0;
                uint16_t shentsize = 0, shnum = 0;
                if(!readAt(code, size, 0x28, shoff) || !readAt(code, size, 0x3A, shentsize)
                   || !readAt(code, size, 0x3C, shnum) || shentsize < 0x28)
                {
                    return false;
                }
                for(uint32_t i = 0; i < shnum; i++)
                {
                    uint64_t const base   = shoff + uint64_t(i) * shentsize;
                    uint32_t       shType = 0;
                    uint64_t       off = 0, len = 0;
                    if(!readAt(code, size, base + 4, shType)
                       || !readAt(code, size, base + 0x18, off)
                       || !readAt(code, size, base + 0x20, len))
                    {
                        return false;
                    }
                    if(shType != kShtNote || off > size || len > size - off)
                    {
                        continue;
                    }
                    uint64_t p = off;
                    while(p + 12 <= off + len)
                    {
                        uint32_t namesz = 0, descsz = 0, ntype = 0;
                        readAt(code, size, p, namesz);
                        readAt(code, size, p + 4, descsz);
                        readAt(code, size, p + 8, ntype);
                        uint64_t const dp = p + 12 + ((uint64_t(namesz) + 3) & ~uint64_t(3));
                        if(dp > off + len || descsz > off + len - dp)
                        {
                            break;
                        }
                        bool const amdgpu
                            = namesz >= 6 && std::memcmp(code + p + 12, "AMDGPU", 6) == 0;
                        if(amdgpu && ntype == kNtAmdgpuMetadata)
                        {
                            desc     = code + dp;
                            descSize = descsz;
                            return true;
                        }
                        p = dp + ((uint64_t(descsz) + 3) & ~uint64_t(3));
                    }
                }
                return false;
            }
        } // namespace

        bool readKernelMetadata(uint8_t const*     code,
                                size_t             size,
                                std::string const& name,
                                PackKernel&        out,
                                std::string&       reason)
        {
            uint8_t const* desc     = nullptr;
            size_t         descSize = 0;
            if(!metadataNote(code, size, desc, descSize))
            {
                reason = "the code object is not an AMDGPU ELF file with a metadata note";
                return false;
            }
            MpValue root;
            MsgPack reader(desc, descSize);
            if(!reader.value(root) || root.kind != MpValue::Kind::Map)
            {
                reason = "the code object's metadata note is malformed";
                return false;
            }
            auto const* kernels = root.get("amdhsa.kernels");
            if(kernels == nullptr || kernels->kind != MpValue::Kind::Array)
            {
                reason = "the code object's metadata lists no kernels";
                return false;
            }
            for(auto const& k : kernels->a)
            {
                if(k.kind != MpValue::Kind::Map || k.stringOf(".name") != name)
                {
                    continue;
                }
                out.name = name;
                out.args.clear();
                if(auto const* args = k.get(".args");
                   args != nullptr && args->kind == MpValue::Kind::Array)
                {
                    for(auto const& a : args->a)
                    {
                        auto const kind = a.stringOf(".value_kind");
                        if(kind.rfind("hidden_", 0) == 0)
                        {
                            continue;
                        }
                        KernelArg arg;
                        arg.name    = a.stringOf(".name");
                        arg.offset  = static_cast<uint32_t>(a.intOf(".offset", -1));
                        arg.size    = static_cast<uint32_t>(a.intOf(".size", 0));
                        arg.pointer = kind == "global_buffer";
                        if(a.intOf(".offset", -1) < 0 || arg.size == 0 || arg.name.empty())
                        {
                            reason = "kernel " + name + ": an argument has no name, offset or size";
                            return false;
                        }
                        out.args.push_back(std::move(arg));
                    }
                }
                out.kernargSize  = static_cast<uint32_t>(k.intOf(".kernarg_segment_size", 0));
                out.groupSegment = static_cast<uint32_t>(k.intOf(".group_segment_fixed_size", 0));
                out.privateSegment
                    = static_cast<uint32_t>(k.intOf(".private_segment_fixed_size", 0));
                out.maxWorkgroup  = static_cast<uint32_t>(k.intOf(".max_flat_workgroup_size", 0));
                out.wavefrontSize = static_cast<uint32_t>(k.intOf(".wavefront_size", 64));
                for(auto const& a : out.args)
                {
                    if(a.offset + uint64_t(a.size) > out.kernargSize)
                    {
                        reason = "kernel " + name + ": argument " + a.name
                                 + " lies past the kernel argument segment";
                        return false;
                    }
                }
                return true;
            }
            reason = "the code object has no kernel " + name;
            return false;
        }
    } // namespace rockebackend
} // namespace hiptensor
