// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_bwd_refusals.py: spec_digest
 * and the recorded resource-gate refusals of the attention backward main
 * kernel (RESOURCE_REFUSED) and of its split dQ kernel (DQ_RESOURCE_REFUSED,
 * attention_bwd_dq.py). Both tables are generated from the Python dicts into
 * _attention_bwd_refusals_table.inc; see rocke/instance_attention_bwd.h.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h"
#include "rocke/instance_attention_bwd.h"
#include "rocke/instance_attention_bwd_dq.h"

namespace rocke_attn_bwd
{
namespace
{
/* One row of a refusal table: (arch, spec_digest) -> the recorded failure. */
struct RefusalRow
{
    const char* arch;
    const char* digest;
    const char* reason;
};

#include "_attention_bwd_refusals_table.inc"

template <size_t N>
const char* lookup(const RefusalRow (&rows)[N], const std::string& arch, const std::string& digest)
{
    /* the generator sorts the rows by (arch, digest), bytewise */
    auto less
        = [](const RefusalRow& r, const std::pair<const std::string*, const std::string*>& k) {
              int c = strcmp(r.arch, k.first->c_str());
              return c < 0 || (c == 0 && strcmp(r.digest, k.second->c_str()) < 0);
          };
    auto key = std::make_pair(&arch, &digest);
    const RefusalRow* it = std::lower_bound(rows, rows + N, key, less);
    if(it != rows + N && arch == it->arch && digest == it->digest)
        return it->reason;
    return NULL;
}

/* hashlib.sha256(blob).hexdigest()[:16] with blob = f"{arch}|{sorted(items)!r}",
 * items the (name, repr(value)) of every non-default field plus head_size
 * (no declared default, so always present); name is never in items. */
std::string digest_of(const std::string& arch,
                      std::vector<std::pair<std::string, std::string>> items,
                      int head_size)
{
    items.emplace_back("head_size", std::to_string(head_size));
    std::sort(
        items.begin(), items.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::string r = arch + "|[";
    for(size_t i = 0; i < items.size(); ++i)
    {
        if(i)
            r += ", ";
        r += "(" + py_repr(items[i].first) + ", " + items[i].second + ")";
    }
    r += "]";
    return sha256_hex(r).substr(0, 16);
}
} // namespace

/* SHA-256 (FIPS 180-4) of a byte string, lower-case hex. */
std::string sha256_hex(const std::string& msg)
{
    static const uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
        0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
        0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
        0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
        0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
        0xc67178f2u};
    uint32_t h[8] = {0x6a09e667u,
                     0xbb67ae85u,
                     0x3c6ef372u,
                     0xa54ff53au,
                     0x510e527fu,
                     0x9b05688cu,
                     0x1f83d9abu,
                     0x5be0cd19u};
    std::string m = msg;
    uint64_t bits = (uint64_t)msg.size() * 8u;
    m += (char)0x80;
    while(m.size() % 64 != 56)
        m += (char)0;
    for(int i = 7; i >= 0; --i)
        m += (char)((bits >> (8 * i)) & 0xff);
    auto ror = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for(size_t off = 0; off < m.size(); off += 64)
    {
        uint32_t w[64];
        for(int i = 0; i < 16; ++i)
            w[i] = ((uint32_t)(uint8_t)m[off + 4 * i] << 24)
                   | ((uint32_t)(uint8_t)m[off + 4 * i + 1] << 16)
                   | ((uint32_t)(uint8_t)m[off + 4 * i + 2] << 8)
                   | (uint32_t)(uint8_t)m[off + 4 * i + 3];
        for(int i = 16; i < 64; ++i)
        {
            uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
        for(int i = 0; i < 64; ++i)
        {
            uint32_t s1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = k + s1 + ch + K[i] + w[i];
            uint32_t s0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = s0 + mj;
            k = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += k;
    }
    char out[65];
    for(int i = 0; i < 8; ++i)
        snprintf(out + 8 * i, 9, "%08x", h[i]);
    return std::string(out, 64);
}

std::string spec_digest(const rocke_attn_bwd_spec_t& spec, const std::string& arch)
{
    return digest_of(arch, non_default_fields(spec), spec.head_size);
}

std::string dq_spec_digest(const rocke_attn_bwd_dq_spec_t& spec, const std::string& arch)
{
    return digest_of(arch, dq_non_default_fields(spec), spec.head_size);
}

const char* resource_refusal(const rocke_attn_bwd_spec_t& spec, const std::string& arch)
{
    return lookup(RESOURCE_REFUSED, arch, spec_digest(spec, arch));
}

const char* dq_resource_refusal(const rocke_attn_bwd_dq_spec_t& spec, const std::string& arch)
{
    return lookup(DQ_RESOURCE_REFUSED, arch, dq_spec_digest(spec, arch));
}
} // namespace rocke_attn_bwd
