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

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "rocke_pack_format.hpp"

namespace hiptensor
{
    namespace rockepack
    {
        // ----------------------------------------------------------------------------------------
        // SHA-256
        // ----------------------------------------------------------------------------------------

        namespace
        {
            constexpr uint32_t kSha256K[64] = {
                0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
                0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
                0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
                0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
                0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
                0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
                0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
                0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
                0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
                0xc67178f2};

            inline uint32_t rotr(uint32_t x, int n)
            {
                return (x >> n) | (x << (32 - n));
            }
        } // namespace

        Sha256::Sha256()
            : mH{0x6a09e667,
                 0xbb67ae85,
                 0x3c6ef372,
                 0xa54ff53a,
                 0x510e527f,
                 0x9b05688c,
                 0x1f83d9ab,
                 0x5be0cd19}
        {
        }

        void Sha256::block(uint8_t const* p)
        {
            uint32_t w[64];
            for(int i = 0; i < 16; ++i)
            {
                w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16)
                       | (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
            }
            for(int i = 16; i < 64; ++i)
            {
                uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i]        = w[i - 16] + s0 + w[i - 7] + s1;
            }
            uint32_t a = mH[0], b = mH[1], c = mH[2], d = mH[3], e = mH[4], f = mH[5], g = mH[6],
                     h = mH[7];
            for(int i = 0; i < 64; ++i)
            {
                uint32_t s1  = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
                uint32_t ch  = (e & f) ^ (~e & g);
                uint32_t t1  = h + s1 + ch + kSha256K[i] + w[i];
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
            mH[0] += a;
            mH[1] += b;
            mH[2] += c;
            mH[3] += d;
            mH[4] += e;
            mH[5] += f;
            mH[6] += g;
            mH[7] += h;
        }

        void Sha256::update(void const* data, size_t size)
        {
            auto const* p = static_cast<uint8_t const*>(data);
            mBits += uint64_t(size) * 8;
            while(size > 0)
            {
                size_t n = std::min(size, mBuf.size() - mBufLen);
                std::memcpy(mBuf.data() + mBufLen, p, n);
                mBufLen += n;
                p += n;
                size -= n;
                if(mBufLen == mBuf.size())
                {
                    block(mBuf.data());
                    mBufLen = 0;
                }
            }
        }

        std::array<uint8_t, 32> Sha256::digest()
        {
            uint64_t bits = mBits;
            uint8_t  pad  = 0x80;
            update(&pad, 1);
            uint8_t zero = 0;
            while(mBufLen != 56)
            {
                update(&zero, 1);
            }
            uint8_t len[8];
            for(int i = 0; i < 8; ++i)
            {
                len[i] = uint8_t(bits >> (56 - 8 * i));
            }
            update(len, 8);
            std::array<uint8_t, 32> out{};
            for(int i = 0; i < 8; ++i)
            {
                out[4 * i]     = uint8_t(mH[i] >> 24);
                out[4 * i + 1] = uint8_t(mH[i] >> 16);
                out[4 * i + 2] = uint8_t(mH[i] >> 8);
                out[4 * i + 3] = uint8_t(mH[i]);
            }
            return out;
        }

        std::array<uint8_t, 32> sha256(void const* data, size_t size)
        {
            Sha256 h;
            h.update(data, size);
            return h.digest();
        }

        std::string hex(uint8_t const* data, size_t size)
        {
            static char const digits[] = "0123456789abcdef";
            std::string       out;
            out.reserve(2 * size);
            for(size_t i = 0; i < size; ++i)
            {
                out += digits[data[i] >> 4];
                out += digits[data[i] & 15];
            }
            return out;
        }

        // ----------------------------------------------------------------------------------------
        // Code object metadata: the NT_AMDGPU_METADATA note (msgpack) of an ELF64 code object
        // ----------------------------------------------------------------------------------------

        namespace
        {
            struct MpValue;
            using MpPtr = std::shared_ptr<MpValue>;

            struct MpValue
            {
                enum Kind
                {
                    Nil,
                    Bool,
                    Int,
                    UInt,
                    Float,
                    Str,
                    Bin,
                    Array,
                    Map
                } kind
                    = Nil;
                bool                                 b = false;
                int64_t                              i = 0;
                uint64_t                             u = 0;
                double                               f = 0.0;
                std::string                          s;
                std::vector<MpPtr>                   arr;
                std::vector<std::pair<MpPtr, MpPtr>> map;

                MpValue const* get(std::string const& key) const
                {
                    if(kind != Map)
                    {
                        return nullptr;
                    }
                    for(auto const& kv : map)
                    {
                        if(kv.first->kind == Str && kv.first->s == key)
                        {
                            return kv.second.get();
                        }
                    }
                    return nullptr;
                }

                bool isInt() const
                {
                    return kind == Int || kind == UInt;
                }

                int64_t asInt() const
                {
                    return kind == UInt ? int64_t(u) : i;
                }
            };

            class MsgPack
            {
            public:
                MsgPack(uint8_t const* data, size_t size)
                    : mD(data)
                    , mN(size)
                {
                }

                MpPtr value()
                {
                    auto     v = std::make_shared<MpValue>();
                    uint64_t t = take(1);
                    if(t <= 0x7F)
                    {
                        v->kind = MpValue::UInt;
                        v->u    = t;
                    }
                    else if(t >= 0xE0)
                    {
                        v->kind = MpValue::Int;
                        v->i    = int64_t(t) - 0x100;
                    }
                    else if(t >= 0x80 && t <= 0x8F)
                    {
                        readMap(*v, t & 0x0F);
                    }
                    else if(t >= 0x90 && t <= 0x9F)
                    {
                        readArray(*v, t & 0x0F);
                    }
                    else if(t >= 0xA0 && t <= 0xBF)
                    {
                        readStr(*v, t & 0x1F);
                    }
                    else if(t == 0xC0)
                    {
                        v->kind = MpValue::Nil;
                    }
                    else if(t == 0xC2 || t == 0xC3)
                    {
                        v->kind = MpValue::Bool;
                        v->b    = t == 0xC3;
                    }
                    else if(t == 0xC4 || t == 0xC5 || t == 0xC6)
                    {
                        size_t n = take(t == 0xC4 ? 1 : t == 0xC5 ? 2 : 4);
                        v->kind  = MpValue::Bin;
                        v->s     = bytes(n);
                    }
                    else if(t == 0xCA)
                    {
                        uint32_t bits = uint32_t(take(4));
                        float    f;
                        std::memcpy(&f, &bits, 4);
                        v->kind = MpValue::Float;
                        v->f    = f;
                    }
                    else if(t == 0xCB)
                    {
                        uint64_t bits = take(8);
                        std::memcpy(&v->f, &bits, 8);
                        v->kind = MpValue::Float;
                    }
                    else if(t >= 0xCC && t <= 0xCF)
                    {
                        v->kind = MpValue::UInt;
                        v->u    = take(size_t(1) << (t - 0xCC));
                    }
                    else if(t >= 0xD0 && t <= 0xD3)
                    {
                        size_t   n   = size_t(1) << (t - 0xD0);
                        uint64_t raw = take(n);
                        v->kind      = MpValue::Int;
                        if(n < 8 && (raw >> (8 * n - 1)) & 1)
                        {
                            raw |= ~uint64_t(0) << (8 * n);
                        }
                        v->i = int64_t(raw);
                    }
                    else if(t == 0xD9 || t == 0xDA || t == 0xDB)
                    {
                        readStr(*v, take(t == 0xD9 ? 1 : t == 0xDA ? 2 : 4));
                    }
                    else if(t == 0xDC || t == 0xDD)
                    {
                        readArray(*v, take(t == 0xDC ? 2 : 4));
                    }
                    else if(t == 0xDE || t == 0xDF)
                    {
                        readMap(*v, take(t == 0xDE ? 2 : 4));
                    }
                    else
                    {
                        char msg[64];
                        std::snprintf(msg, sizeof msg, "unsupported msgpack type 0x%02x", int(t));
                        throw std::runtime_error(msg);
                    }
                    return v;
                }

            private:
                uint64_t take(size_t n)
                {
                    if(mI + n > mN)
                    {
                        throw std::runtime_error("truncated msgpack");
                    }
                    uint64_t v = 0;
                    for(size_t k = 0; k < n; ++k)
                    {
                        v = (v << 8) | mD[mI + k];
                    }
                    mI += n;
                    return v;
                }

                std::string bytes(size_t n)
                {
                    if(mI + n > mN)
                    {
                        throw std::runtime_error("truncated msgpack");
                    }
                    std::string s(reinterpret_cast<char const*>(mD + mI), n);
                    mI += n;
                    return s;
                }

                void readStr(MpValue& v, size_t n)
                {
                    v.kind = MpValue::Str;
                    v.s    = bytes(n);
                }

                void readArray(MpValue& v, size_t n)
                {
                    v.kind = MpValue::Array;
                    for(size_t k = 0; k < n; ++k)
                    {
                        v.arr.push_back(value());
                    }
                }

                void readMap(MpValue& v, size_t n)
                {
                    v.kind = MpValue::Map;
                    for(size_t k = 0; k < n; ++k)
                    {
                        MpPtr key = value();
                        MpPtr val = value();
                        v.map.emplace_back(key, val);
                    }
                }

                uint8_t const* mD;
                size_t         mN;
                size_t         mI = 0;
            };

            template <typename T>
            T rd(std::vector<uint8_t> const& co, size_t off)
            {
                if(off + sizeof(T) > co.size())
                {
                    throw std::runtime_error("truncated ELF file");
                }
                T v;
                std::memcpy(&v, co.data() + off, sizeof(T));
                return v;
            }

            MpPtr codeObjectMetadata(std::vector<uint8_t> const& co)
            {
                if(co.size() < 64
                   || std::memcmp(co.data(),
                                  "\x7f"
                                  "ELF",
                                  4)
                          != 0
                   || co[4] != 2)
                {
                    throw std::runtime_error("not an ELF64 code object");
                }
                uint64_t shoff     = rd<uint64_t>(co, 0x28);
                uint16_t shentsize = rd<uint16_t>(co, 0x3A);
                uint16_t shnum     = rd<uint16_t>(co, 0x3C);
                for(uint32_t i = 0; i < shnum; ++i)
                {
                    uint64_t base = shoff + uint64_t(i) * shentsize;
                    if(rd<uint32_t>(co, base + 4) != 7) // SHT_NOTE
                    {
                        continue;
                    }
                    uint64_t off  = rd<uint64_t>(co, base + 0x18);
                    uint64_t size = rd<uint64_t>(co, base + 0x20);
                    uint64_t p    = off;
                    while(p + 12 <= off + size)
                    {
                        uint32_t    namesz = rd<uint32_t>(co, p);
                        uint32_t    descsz = rd<uint32_t>(co, p + 4);
                        uint32_t    ntype  = rd<uint32_t>(co, p + 8);
                        std::string name(reinterpret_cast<char const*>(co.data() + p + 12),
                                         std::min<size_t>(namesz, co.size() - (p + 12)));
                        while(!name.empty() && name.back() == '\0')
                        {
                            name.pop_back();
                        }
                        uint64_t dp = p + 12 + ((uint64_t(namesz) + 3) & ~uint64_t(3));
                        if(name == "AMDGPU" && ntype == 32)
                        {
                            if(dp + descsz > co.size())
                            {
                                throw std::runtime_error("truncated metadata note");
                            }
                            MsgPack mp(co.data() + dp, descsz);
                            return mp.value();
                        }
                        p = dp + ((uint64_t(descsz) + 3) & ~uint64_t(3));
                    }
                }
                throw std::runtime_error("code object has no NT_AMDGPU_METADATA note");
            }

            uint32_t intOr(MpValue const* m, char const* key, uint32_t fallback)
            {
                MpValue const* v = m->get(key);
                if(v == nullptr)
                {
                    return fallback;
                }
                if(!v->isInt())
                {
                    throw std::runtime_error(std::string("metadata ") + key + " is not an integer");
                }
                return uint32_t(v->asInt());
            }

            std::string strOr(MpValue const* m, char const* key)
            {
                MpValue const* v = m->get(key);
                return v != nullptr && v->kind == MpValue::Str ? v->s : std::string();
            }
        } // namespace

        bool readKernelMd(std::vector<uint8_t> const& co,
                          std::string const&          name,
                          KernelMd&                   out,
                          std::string&                reason)
        {
            try
            {
                MpPtr          md      = codeObjectMetadata(co);
                MpValue const* kernels = md->get("amdhsa.kernels");
                if(kernels != nullptr && kernels->kind == MpValue::Array)
                {
                    for(auto const& k : kernels->arr)
                    {
                        MpValue const* kn = k->get(".name");
                        if(kn == nullptr || kn->kind != MpValue::Str || kn->s != name)
                        {
                            continue;
                        }
                        out                 = KernelMd{};
                        MpValue const* args = k->get(".args");
                        if(args != nullptr && args->kind == MpValue::Array)
                        {
                            for(auto const& a : args->arr)
                            {
                                std::string kind = strOr(a.get(), ".value_kind");
                                if(kind.rfind("hidden_", 0) == 0)
                                {
                                    continue;
                                }
                                MpValue const* offset = a->get(".offset");
                                MpValue const* size   = a->get(".size");
                                if(offset == nullptr || size == nullptr || !offset->isInt()
                                   || !size->isInt())
                                {
                                    throw std::runtime_error(
                                        "kernel argument without offset / size");
                                }
                                MdArg arg;
                                arg.name   = strOr(a.get(), ".name");
                                arg.offset = uint32_t(offset->asInt());
                                arg.size   = uint32_t(size->asInt());
                                arg.kind
                                    = kind == "global_buffer" ? kArgKindPointer : kArgKindValue;
                                out.args.push_back(arg);
                            }
                        }
                        out.kernargSize    = intOr(k.get(), ".kernarg_segment_size", 0);
                        out.groupSegment   = intOr(k.get(), ".group_segment_fixed_size", 0);
                        out.privateSegment = intOr(k.get(), ".private_segment_fixed_size", 0);
                        out.maxWorkgroup   = intOr(k.get(), ".max_flat_workgroup_size", 0);
                        out.wavefrontSize  = intOr(k.get(), ".wavefront_size", 64);
                        out.sgpr           = intOr(k.get(), ".sgpr_count", 0);
                        out.vgpr           = intOr(k.get(), ".vgpr_count", 0);
                        out.agpr           = intOr(k.get(), ".agpr_count", 0);
                        out.symbol         = strOr(k.get(), ".symbol");
                        return true;
                    }
                }
                reason = "kernel " + name + " not in the code object metadata";
                return false;
            }
            catch(std::exception const& e)
            {
                reason = e.what();
                return false;
            }
        }

        // ----------------------------------------------------------------------------------------
        // Pack writer
        // ----------------------------------------------------------------------------------------

        namespace
        {
            constexpr char     kPackMagic[8]     = {'H', 'T', 'P', 'K', 'I', 'D', 'X', '\0'};
            constexpr size_t   kHeaderSize       = 160;
            constexpr size_t   kKernelRecordSize = 96;
            constexpr uint32_t kAttrKindInt      = 0;
            constexpr uint32_t kAttrKindString   = 1;

            class Strings
            {
            public:
                Strings()
                {
                    mBuf.push_back('\0');
                    mIndex[""] = 0;
                }
                uint32_t add(std::string const& s)
                {
                    auto it = mIndex.find(s);
                    if(it != mIndex.end())
                    {
                        return it->second;
                    }
                    uint32_t off = uint32_t(mBuf.size());
                    mIndex[s]    = off;
                    mBuf.insert(mBuf.end(), s.begin(), s.end());
                    mBuf.push_back('\0');
                    return off;
                }
                std::vector<uint8_t> const& buf() const
                {
                    return mBuf;
                }

            private:
                std::vector<uint8_t>            mBuf;
                std::map<std::string, uint32_t> mIndex;
            };

            template <typename T>
            void put(std::vector<uint8_t>& out, T v)
            {
                uint8_t b[sizeof(T)];
                std::memcpy(b, &v, sizeof(T)); // little-endian hosts only (checked in main)
                out.insert(out.end(), b, b + sizeof(T));
            }

            void align16(std::vector<uint8_t>& v)
            {
                while(v.size() % 16)
                {
                    v.push_back(0);
                }
            }

            // json.dump(..., indent=1, sort_keys=True) of rocKE's Python reference writer
            // (ensure_ascii).
            class Json
            {
            public:
                std::string out;

                void str(std::string const& s)
                {
                    out += '"';
                    size_t i = 0;
                    while(i < s.size())
                    {
                        uint32_t cp = decode(s, i);
                        switch(cp)
                        {
                        case '"':
                            out += "\\\"";
                            break;
                        case '\\':
                            out += "\\\\";
                            break;
                        case '\n':
                            out += "\\n";
                            break;
                        case '\r':
                            out += "\\r";
                            break;
                        case '\t':
                            out += "\\t";
                            break;
                        case '\b':
                            out += "\\b";
                            break;
                        case '\f':
                            out += "\\f";
                            break;
                        default:
                            if(cp >= 0x20 && cp <= 0x7E)
                            {
                                out += char(cp);
                            }
                            else if(cp < 0x10000)
                            {
                                u16(cp);
                            }
                            else
                            {
                                cp -= 0x10000;
                                u16(0xD800 | (cp >> 10));
                                u16(0xDC00 | (cp & 0x3FF));
                            }
                        }
                    }
                    out += '"';
                }

                void num(int64_t v)
                {
                    out += std::to_string(v);
                }

                void nl(int level)
                {
                    out += '\n';
                    out.append(size_t(level), ' ');
                }

            private:
                void u16(uint32_t u)
                {
                    char buf[16];
                    std::snprintf(buf, sizeof buf, "\\u%04x", unsigned(u & 0xFFFF));
                    out += buf;
                }

                static uint32_t decode(std::string const& s, size_t& i)
                {
                    auto     c  = uint8_t(s[i++]);
                    int      n  = c < 0x80 ? 0 : c < 0xE0 ? 1 : c < 0xF0 ? 2 : 3;
                    uint32_t cp = n == 0   ? c
                                  : n == 1 ? (c & 0x1F)
                                  : n == 2 ? (c & 0x0F)
                                           : (c & 0x07);
                    for(int k = 0; k < n && i < s.size(); ++k)
                    {
                        cp = (cp << 6) | (uint8_t(s[i++]) & 0x3F);
                    }
                    return cp;
                }
            };

            // A JSON object writer with sorted keys: the caller adds members in sorted order.
            struct Obj
            {
                Json& j;
                int   level;
                bool  first = true;
                Obj(Json& json, int lvl)
                    : j(json)
                    , level(lvl)
                {
                    j.out += '{';
                }
                void key(std::string const& k)
                {
                    j.out += first ? "" : ",";
                    first = false;
                    j.nl(level + 1);
                    j.str(k);
                    j.out += ": ";
                }
                void close()
                {
                    if(!first)
                    {
                        j.nl(level);
                    }
                    j.out += '}';
                }
            };

            struct Arr
            {
                Json& j;
                int   level;
                bool  first = true;
                Arr(Json& json, int lvl)
                    : j(json)
                    , level(lvl)
                {
                    j.out += '[';
                }
                void item()
                {
                    j.out += first ? "" : ",";
                    first = false;
                    j.nl(level + 1);
                }
                void close()
                {
                    if(!first)
                    {
                        j.nl(level);
                    }
                    j.out += ']';
                }
            };

            void attrValue(Json& j, AttrValue const& v)
            {
                if(std::holds_alternative<int64_t>(v))
                {
                    j.num(std::get<int64_t>(v));
                }
                else
                {
                    j.str(std::get<std::string>(v));
                }
            }

            bool writeFile(std::string const& path,
                           void const*        data,
                           size_t             size,
                           std::string&       reason)
            {
                std::string   tmp = path + ".tmp";
                std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
                f.write(static_cast<char const*>(data), std::streamsize(size));
                f.close();
                if(!f)
                {
                    reason = "cannot write " + tmp;
                    return false;
                }
                if(std::rename(tmp.c_str(), path.c_str()) != 0)
                {
                    std::remove(path.c_str());
                    if(std::rename(tmp.c_str(), path.c_str()) != 0)
                    {
                        reason = "cannot rename " + tmp + " to " + path;
                        return false;
                    }
                }
                return true;
            }
        } // namespace

        std::string specKey(std::string const& arch, Attrs const& attrs, std::string const& name)
        {
            return "hiptensor-rocke/v" + std::to_string(kKernelAbi) + "|"
                   + std::get<std::string>(attrs.at("family")) + "|" + arch + "|" + name;
        }

        bool writePack(std::string const&                        dir,
                       std::string const&                        arch,
                       std::vector<PackEntry> const&             kernels,
                       std::map<std::string, std::string> const& routes,
                       std::string const&                        compilerId,
                       std::string const&                        rockeVersion,
                       std::string&                              packId,
                       size_t&                                   size,
                       std::string&                              reason)
        {
            Strings              st;
            std::vector<uint8_t> krec, args, attrs, blob;
            uint32_t             nArgs = 0, nAttrs = 0;
            Json                 ij;
            ij.out.reserve(1 << 20);
            // The kernels list of the index is written into its own buffer, then spliced in.
            Json kj;
            Arr  karr(kj, 1);
            for(auto const& k : kernels)
            {
                auto     sha       = sha256(k.co.data(), k.co.size());
                uint32_t argFirst  = nArgs;
                uint32_t attrFirst = nAttrs;
                for(auto const& a : k.md.args)
                {
                    put<uint32_t>(args, st.add(a.name));
                    put<uint32_t>(args, a.offset);
                    put<uint32_t>(args, a.size);
                    put<uint32_t>(args, a.kind);
                    ++nArgs;
                }
                for(auto const& kv : k.attrs) // std::map: sorted keys
                {
                    uint32_t keyOff = st.add(kv.first);
                    put<uint32_t>(attrs, keyOff);
                    if(std::holds_alternative<int64_t>(kv.second))
                    {
                        put<uint32_t>(attrs, kAttrKindInt);
                        put<int64_t>(attrs, std::get<int64_t>(kv.second));
                    }
                    else
                    {
                        put<uint32_t>(attrs, kAttrKindString);
                        put<int64_t>(attrs, st.add(std::get<std::string>(kv.second)));
                    }
                    ++nAttrs;
                }
                align16(blob);
                uint64_t    off = blob.size();
                std::string sk  = specKey(arch, k.attrs, k.name);
                blob.insert(blob.end(), k.co.begin(), k.co.end());
                uint32_t nameOff   = st.add(k.name);
                uint32_t familyOff = st.add(std::get<std::string>(k.attrs.at("family")));
                uint32_t keyOff    = st.add(sk);
                for(uint32_t v : {nameOff,
                                  familyOff,
                                  keyOff,
                                  argFirst,
                                  uint32_t(k.md.args.size()),
                                  attrFirst,
                                  uint32_t(k.attrs.size()),
                                  k.md.kernargSize,
                                  k.md.groupSegment,
                                  k.md.privateSegment,
                                  k.md.maxWorkgroup,
                                  k.md.wavefrontSize})
                {
                    put<uint32_t>(krec, v);
                }
                put<uint64_t>(krec, off);
                put<uint64_t>(krec, uint64_t(k.co.size()));
                krec.insert(krec.end(), sha.begin(), sha.end());
                if(krec.size() % kKernelRecordSize)
                {
                    reason = "kernel record size";
                    return false;
                }

                karr.item();
                Obj ko(kj, 2);
                ko.key("args");
                {
                    Arr aa(kj, 3);
                    for(auto const& a : k.md.args)
                    {
                        aa.item();
                        Obj ao(kj, 4);
                        ao.key("kind");
                        kj.num(a.kind);
                        ao.key("name");
                        kj.str(a.name);
                        ao.key("offset");
                        kj.num(a.offset);
                        ao.key("size");
                        kj.num(a.size);
                        ao.close();
                    }
                    aa.close();
                }
                ko.key("attrs");
                {
                    Obj ao(kj, 3);
                    for(auto const& kv : k.attrs)
                    {
                        ao.key(kv.first);
                        attrValue(kj, kv.second);
                    }
                    ao.close();
                }
                ko.key("code_sha256");
                kj.str(hex(sha.data(), sha.size()));
                ko.key("code_size");
                kj.num(int64_t(k.co.size()));
                ko.key("group_segment");
                kj.num(k.md.groupSegment);
                ko.key("kernarg_size");
                kj.num(k.md.kernargSize);
                ko.key("max_flat_workgroup_size");
                kj.num(k.md.maxWorkgroup);
                ko.key("name");
                kj.str(k.name);
                ko.key("private_segment");
                kj.num(k.md.privateSegment);
                ko.key("registers");
                {
                    Obj ro(kj, 3);
                    ro.key("agpr");
                    kj.num(k.md.agpr);
                    ro.key("sgpr");
                    kj.num(k.md.sgpr);
                    ro.key("vgpr");
                    kj.num(k.md.vgpr);
                    ro.close();
                }
                ko.key("spec_key");
                kj.str(sk);
                ko.key("wavefront_size");
                kj.num(k.md.wavefrontSize);
                ko.close();
            }
            karr.close();

            std::vector<uint8_t> rrec;
            for(auto const& kv : routes) // std::map: sorted keys
            {
                uint32_t keyOff = st.add(kv.first);
                put<uint32_t>(rrec, keyOff);
                put<uint32_t>(rrec, st.add(kv.second));
            }
            Sha256 ph;
            ph.update(krec.data(), krec.size());
            ph.update(blob.data(), blob.size());
            auto pid         = ph.digest();
            packId           = hex(pid.data(), pid.size());
            uint32_t archOff = st.add(arch);
            uint32_t cidOff  = st.add(compilerId);
            uint32_t rvOff   = st.add(rockeVersion);
            uint32_t pidOff  = st.add(packId);

            std::vector<std::vector<uint8_t> const*> data
                = {&krec, &args, &attrs, &rrec, &st.buf()};
            std::vector<uint64_t> secOff;
            uint64_t              offset = kHeaderSize;
            for(auto const* d : data)
            {
                offset = (offset + 15) & ~uint64_t(15);
                secOff.push_back(offset);
                offset += d->size();
            }
            offset           = (offset + 15) & ~uint64_t(15);
            uint64_t blobOff = offset;

            std::vector<uint8_t> out(kPackMagic, kPackMagic + 8);
            for(uint32_t v : {kPackFormatVersion,
                              kPackMinReader,
                              kKernelAbi,
                              uint32_t(kernels.size()),
                              nArgs,
                              nAttrs,
                              uint32_t(routes.size()),
                              0u})
            {
                put<uint32_t>(out, v);
            }
            for(int i = 0; i < 5; ++i)
            {
                put<uint64_t>(out, secOff[i]);
            }
            put<uint64_t>(out, uint64_t(st.buf().size()));
            put<uint64_t>(out, blobOff);
            put<uint64_t>(out, uint64_t(blob.size()));
            for(uint32_t v : {archOff, cidOff, rvOff, pidOff})
            {
                put<uint32_t>(out, v);
            }
            if(out.size() > kHeaderSize)
            {
                reason = "header size";
                return false;
            }
            out.resize(kHeaderSize, 0);
            for(int i = 0; i < 5; ++i)
            {
                out.resize(secOff[i], 0);
                out.insert(out.end(), data[i]->begin(), data[i]->end());
            }
            out.resize(blobOff, 0);
            out.insert(out.end(), blob.begin(), blob.end());
            size = out.size();
            if(!writeFile(dir + "/rocke.htpk", out.data(), out.size(), reason))
            {
                return false;
            }

            Obj top(ij, 0);
            top.key("arch");
            ij.str(arch);
            top.key("compiler_id");
            ij.str(compilerId);
            top.key("format_version");
            ij.num(kPackFormatVersion);
            top.key("kernel_abi");
            ij.num(kKernelAbi);
            top.key("kernels");
            ij.out += kj.out;
            top.key("min_reader");
            ij.num(kPackMinReader);
            top.key("pack_id");
            ij.str(packId);
            top.key("rocke_version");
            ij.str(rockeVersion);
            top.key("routes");
            {
                Obj ro(ij, 1);
                for(auto const& kv : routes)
                {
                    ro.key(kv.first);
                    ij.str(kv.second);
                }
                ro.close();
            }
            top.close();
            return writeFile(dir + "/rocke.index.json", ij.out.data(), ij.out.size(), reason);
        }
    } // namespace rockepack
} // namespace hiptensor
