// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * attention_dense_spec.cpp -- C port of library/kernels/common/
 * attention_dense_spec.py (see rocke/instance_attention_dense_spec.h).
 *
 * Every reject reason is the Python exception text byte for byte. Python ints
 * are unbounded, so extents and products are computed in 64 or 128 bits and a
 * C remainder is used only where its zero-ness matches Python's for every
 * operand sign (x % y == 0 holds in both languages for the same x, y != 0).
 */
#include "rocke/instance_attention_dense_spec.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <initializer_list>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */

const rocke_dense_tile_geometry_t ROCKE_DENSE_TILE_GEOMETRIES[ROCKE_DENSE_TILE_GEOMETRY_COUNT] = {
    {"default", ROCKE_DENSE_DEFAULT_BLOCK_M, ROCKE_DENSE_DEFAULT_BLOCK_N},
    {"bm128", 128, 64},
};

namespace
{

const char* const k_base_persist_decodes[] = {
    "auto", "bt_hkv_minor", "gqa_pair", "gqa_pair_2phase", "hkv_major", "hq_minor_swz", "qb_major"};
const char* const k_base_nonpersist_decodes[]
    = {"auto", "bt_hkv_minor", "hq_minor_swz", "qb_minor"};

/* Write the formatted reason (if a buffer was given) and return false. */
bool reject(char* reason, size_t cap, const char* fmt, ...)
{
    if(reason != NULL && cap > 0)
    {
        va_list ap;
        va_start(ap, fmt);
        (void)vsnprintf(reason, cap, fmt, ap);
        va_end(ap);
    }
    return false;
}

bool streq(const char* a, const char* b)
{
    return a != NULL && b != NULL && strcmp(a, b) == 0;
}

/* Python repr() of an Optional[str]: None, or the quoted string with the
 * escapes repr() emits for the characters a spec value can carry. */
void py_repr(const char* s, char* out, size_t cap)
{
    if(s == NULL)
    {
        snprintf(out, cap, "None");
        return;
    }
    const char quote = (strchr(s, '\'') != NULL && strchr(s, '"') == NULL) ? '"' : '\'';
    size_t n = 0;
    auto put = [&](const char* piece) {
        size_t len = strlen(piece);
        if(n + len < cap)
        {
            memcpy(out + n, piece, len);
            n += len;
        }
    };
    char q[2] = {quote, 0};
    put(q);
    for(const unsigned char* p = (const unsigned char*)s; *p; ++p)
    {
        char buf[8];
        if(*p == '\\')
            put("\\\\");
        else if(*p == (unsigned char)quote)
        {
            buf[0] = '\\';
            buf[1] = quote;
            buf[2] = 0;
            put(buf);
        }
        else if(*p == '\n')
            put("\\n");
        else if(*p == '\r')
            put("\\r");
        else if(*p == '\t')
            put("\\t");
        else if(*p < 0x20 || *p == 0x7f)
        {
            snprintf(buf, sizeof(buf), "\\x%02x", *p);
            put(buf);
        }
        else
        {
            buf[0] = (char)*p;
            buf[1] = 0;
            put(buf);
        }
    }
    put(q);
    out[n < cap ? n : cap - 1] = '\0';
}

/* Python str(list_of_str), e.g. "['auto', 'qb_major']". */
void py_str_list(const char* const* items, size_t count, char* out, size_t cap)
{
    size_t n = 0;
    out[0] = '\0';
    auto put = [&](const char* piece) {
        size_t len = strlen(piece);
        if(n + len < cap)
        {
            memcpy(out + n, piece, len + 1);
            n += len;
        }
    };
    put("[");
    for(size_t i = 0; i < count; ++i)
    {
        char item[ROCKE_ATTENTION_DENSE_NAME_PART_CAP * 2];
        py_repr(items[i], item, sizeof(item));
        if(i)
            put(", ");
        put(item);
    }
    put("]");
}

/* Exact products of spec factors. Python ints are unbounded, so products and
 * error messages cannot narrow to 64 bits. Magnitude uses base-10**9 limbs. */
struct ExactProduct
{
    bool negative = false;
    bool zero = false;
    uint32_t limb[6] = {1};
    int n = 1;

    explicit ExactProduct(std::initializer_list<int64_t> factors)
    {
        for(int64_t f : factors)
        {
            if(f == 0)
                zero = true;
            if(f < 0)
                negative = !negative;
            const uint64_t m = f < 0 ? (uint64_t)(-(f + 1)) + 1u : (uint64_t)f;
            uint64_t carry = 0;
            for(int i = 0; i < n; ++i)
            {
                const uint64_t v = (uint64_t)limb[i] * m + carry;
                limb[i] = (uint32_t)(v % 1000000000u);
                carry = v / 1000000000u;
            }
            while(carry != 0 && n < 6)
            {
                limb[n++] = (uint32_t)(carry % 1000000000u);
                carry /= 1000000000u;
            }
        }
        if(zero)
        {
            negative = false;
            n = 1;
            limb[0] = 0;
        }
    }

    /* this >= t, for a non-negative t below 10**18. */
    bool at_least(uint64_t t) const
    {
        if(negative)
            return false;
        if(n >= 3)
            return true;
        const uint64_t v = (n == 2 ? (uint64_t)limb[1] * 1000000000u : 0) + limb[0];
        return v >= t;
    }

    /* Compare exact signed products without narrowing Python's unbounded ints. */
    int compare(const ExactProduct& other) const
    {
        if(negative != other.negative)
            return negative ? -1 : 1;
        if(n != other.n)
        {
            const int magnitude = n < other.n ? -1 : 1;
            return negative ? -magnitude : magnitude;
        }
        for(int i = n - 1; i >= 0; --i)
            if(limb[i] != other.limb[i])
            {
                const int magnitude = limb[i] < other.limb[i] ? -1 : 1;
                return negative ? -magnitude : magnitude;
            }
        return 0;
    }

    void str(char* out, size_t cap) const
    {
        size_t k = (size_t)snprintf(out, cap, "%s%u", negative ? "-" : "", limb[n - 1]);
        for(int i = n - 2; i >= 0 && k < cap; --i)
            k += (size_t)snprintf(out + k, cap - k, "%09u", limb[i]);
    }
};

/* Python a // b: floors, where C's / truncates toward zero. b != 0. */
int64_t py_floordiv(int64_t a, int64_t b)
{
    int64_t q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        --q;
    return q;
}

bool contains(const char* const* items, size_t count, const char* s)
{
    for(size_t i = 0; i < count; ++i)
        if(streq(items[i], s))
            return true;
    return false;
}

int64_t dense_nqb(const rocke_attention_dense_spec_t* s)
{
    return py_floordiv((int64_t)s->seqlen_q + s->block_m - 1, s->block_m);
}

int64_t dense_gqa(const rocke_attention_dense_spec_t* s)
{
    return py_floordiv(s->num_query_heads, s->num_kv_heads);
}

bool balance_causal(const rocke_attention_dense_spec_t* s)
{
    return s->causal && !(s->sliding_window > 0 && s->sliding_window < s->seqlen_kv);
}

bool aligned_causal(const rocke_attention_dense_spec_t* s, bool has_right_bound)
{
    const bool moving_diagonal = s->causal_bottom_right && s->seqlen_q != s->seqlen_kv;
    return balance_causal(s) && !moving_diagonal && !s->ragged && !s->varlen && !s->paged
           && !has_right_bound;
}

bool check_nonpersist_decode(const rocke_attention_dense_spec_t* s, char* reason, size_t cap)
{
    if(streq(s->nonpersist_decode, "hq_minor_swz") && s->num_query_heads % s->chiplet_num_xcds != 0)
        return reject(reason,
                      cap,
                      "hq_minor_swz needs num_query_heads (%d) divisible by chiplet_num_xcds (%d)",
                      s->num_query_heads,
                      s->chiplet_num_xcds);
    return true;
}

bool check_persist_decode(const rocke_attention_dense_spec_t* s,
                          const rocke_attention_dense_traits_t* t,
                          char* reason,
                          size_t cap)
{
    if(streq(s->persist_decode, "hq_minor_swz"))
    {
        if(s->num_query_heads % s->chiplet_num_xcds != 0
           || s->num_persistent % s->chiplet_num_xcds != 0)
            return reject(reason,
                          cap,
                          "hq_minor_swz needs num_query_heads and num_persistent divisible by %d",
                          s->chiplet_num_xcds);
        if(balance_causal(s) && dense_nqb(s) % 2 != 0)
            return reject(reason,
                          cap,
                          "hq_minor_swz needs an even number of query blocks under causal "
                          "masking without a sliding window");
        return true;
    }

    const bool gqa_pair = streq(s->persist_decode, "gqa_pair");
    const bool gqa_pair_2phase = streq(s->persist_decode, "gqa_pair_2phase");
    if(!gqa_pair && !gqa_pair_2phase)
        return true;

    const char* name = gqa_pair ? "gqa_pair" : "gqa_pair_2phase";
    if(!s->persistent || !s->causal || t->has_right_bound)
        return reject(reason, cap, "%s requires persistent causal attention", name);
    if(s->ragged || s->varlen || s->paged)
        return reject(reason, cap, "%s is validated only for aligned dense attention", name);

    const int64_t nqb = dense_nqb(s);
    const int64_t gqa = dense_gqa(s);
    if(gqa_pair)
    {
        if(nqb % 2 != 0 || gqa % 2 != 0)
            return reject(reason, cap, "gqa_pair requires even NQB and even GQA ratio");
        const ExactProduct expected({nqb, s->num_kv_heads, s->batch});
        const ExactProduct got({s->num_persistent});
        if(expected.compare(got) != 0)
        {
            char expected_text[64];
            expected.str(expected_text, sizeof(expected_text));
            return reject(reason,
                          cap,
                          "gqa_pair requires num_persistent == NQB*Hkv*B (%s), got %d",
                          expected_text,
                          s->num_persistent);
        }
        return true;
    }

    if(nqb % 2 != 0 || gqa < 2)
        return reject(reason, cap, "gqa_pair_2phase requires even NQB and GQA ratio >= 2");
    const ExactProduct expected({nqb / 2, s->num_kv_heads, s->batch, gqa});
    const ExactProduct got({s->num_persistent});
    if(expected.compare(got) != 0)
    {
        char expected_text[64];
        expected.str(expected_text, sizeof(expected_text));
        return reject(reason,
                      cap,
                      "gqa_pair_2phase requires num_persistent == W/2 (%s), got %d",
                      expected_text,
                      s->num_persistent);
    }
    return true;
}

const char* persist_decode_tag(const char* decode)
{
    if(streq(decode, "hkv_major"))
        return "hkvmaj";
    if(streq(decode, "bt_hkv_minor"))
        return "bthkvmin";
    if(streq(decode, "hq_minor_swz"))
        return "hqminswz";
    if(streq(decode, "gqa_pair"))
        return "gqapair";
    if(streq(decode, "gqa_pair_2phase"))
        return "gqapair2";
    return "";
}

const char* nonpersist_decode_tag(const char* decode)
{
    if(streq(decode, "bt_hkv_minor"))
        return "npbthkvmin";
    if(streq(decode, "hq_minor_swz"))
        return "nphqminswz";
    return "";
}

} // namespace

rocke_attention_dense_spec_t rocke_attention_dense_spec_default(void)
{
    rocke_attention_dense_spec_t s;
    memset(&s, 0, sizeof(s));
    s.causal = true;
    s.dtype = "bf16";
    s.sliding_window = 0;
    s.ragged = false;
    s.varlen = false;
    s.block_m = ROCKE_DENSE_DEFAULT_BLOCK_M;
    s.block_n = ROCKE_DENSE_DEFAULT_BLOCK_N;
    s.waves_per_eu = 2;
    s.lds_k_group_pad = 8;
    s.persistent = false;
    s.num_persistent = 256;
    s.interleave = false;
    s.persist_decode = "auto";
    s.lazy_rescale = true;
    s.paged = false;
    s.block_size = 0;
    s.num_kv_blocks = 0;
    s.use_sinks = false;
    s.causal_bottom_right = false;
    s.nonpersist_decode = "auto";
    s.chiplet_num_xcds = 8;
    return s;
}

rocke_attention_dense_traits_t rocke_attention_dense_traits_base(void)
{
    rocke_attention_dense_traits_t t;
    t.type_name = "AttentionDenseSpec";
    t.supports_noncausal_window = false;
    t.supports_noncausal_cross_length_ragged = false;
    t.has_right_bound = false;
    t.persist_decodes = k_base_persist_decodes;
    t.num_persist_decodes = sizeof(k_base_persist_decodes) / sizeof(k_base_persist_decodes[0]);
    t.nonpersist_decodes = k_base_nonpersist_decodes;
    t.num_nonpersist_decodes
        = sizeof(k_base_nonpersist_decodes) / sizeof(k_base_nonpersist_decodes[0]);
    return t;
}

bool rocke_attention_dense_spec_validate(const rocke_attention_dense_spec_t* s,
                                         const rocke_attention_dense_traits_t* t,
                                         char* reason,
                                         size_t cap)
{
    if(s == NULL || t == NULL)
        return reject(reason, cap, "attention dense spec is NULL");

    if(!streq(s->dtype, "bf16") && !streq(s->dtype, "fp16"))
        return reject(reason,
                      cap,
                      "dtype must be one of ['bf16', 'fp16'], got %s",
                      s->dtype != NULL ? s->dtype : "None");
    if(s->block_m <= 0)
        return reject(reason, cap, "block_m must be positive, got %d", s->block_m);
    if(s->block_n <= 0 || s->block_n % 32 != 0)
        return reject(reason, cap, "block_n must be a positive multiple of 32, got %d", s->block_n);
    if(s->head_size != 64 && s->head_size != 128)
        return reject(reason, cap, "head_size must be 64 or 128, got %d", s->head_size);
    if(s->lds_k_group_pad < 0 || s->lds_k_group_pad % 8 != 0)
        return reject(reason,
                      cap,
                      "lds_k_group_pad must be a non-negative multiple of 8 bf16 elements (16 "
                      "bytes) so the K group pitch stays ds_read_b128-aligned, got %d",
                      s->lds_k_group_pad);
    if(s->causal_bottom_right)
    {
        if(!s->causal)
            return reject(reason, cap, "causal_bottom_right requires causal=True");
        if(s->seqlen_q > s->seqlen_kv)
            return reject(reason,
                          cap,
                          "causal_bottom_right requires seqlen_q <= seqlen_kv, got %d > %d",
                          s->seqlen_q,
                          s->seqlen_kv);
    }

    if(s->ragged)
    {
        if(s->seqlen_q <= 0 || s->seqlen_kv <= 0)
            return reject(reason, cap, "ragged requires positive seqlen_q/seqlen_kv");
        if(s->seqlen_q != s->seqlen_kv && !s->causal && !t->supports_noncausal_cross_length_ragged)
            return reject(reason,
                          cap,
                          "ragged is self-attention only (seqlen_q == seqlen_kv) unless causal "
                          "(top-left or bottom-right) is set, got %d != %d",
                          s->seqlen_q,
                          s->seqlen_kv);
        if(s->varlen)
            return reject(reason, cap, "ragged is not supported with varlen");
        if(s->sliding_window > 0)
            return reject(reason, cap, "ragged is not supported with sliding_window");
    }
    else
    {
        if(s->seqlen_q % s->block_m != 0)
            return reject(reason,
                          cap,
                          "seqlen_q must be a multiple of block_m=%d, got %d",
                          s->block_m,
                          s->seqlen_q);
        if(s->seqlen_kv % s->block_n != 0)
            return reject(reason,
                          cap,
                          "seqlen_kv must be a multiple of block_n=%d, got %d",
                          s->block_n,
                          s->seqlen_kv);
    }

    /* 64-bit remainder: INT_MIN % -1 is undefined in 32 bits. */
    if(s->num_kv_heads == 0 || (int64_t)s->num_query_heads % (int64_t)s->num_kv_heads != 0)
        return reject(reason,
                      cap,
                      "num_query_heads (%d) must be a positive multiple of num_kv_heads (%d)",
                      s->num_query_heads,
                      s->num_kv_heads);
    if(!contains(t->persist_decodes, t->num_persist_decodes, s->persist_decode))
    {
        char allowed[ROCKE_ATTENTION_DENSE_REASON_CAP];
        char got[ROCKE_ATTENTION_DENSE_NAME_PART_CAP * 2];
        py_str_list(t->persist_decodes, t->num_persist_decodes, allowed, sizeof(allowed));
        py_repr(s->persist_decode, got, sizeof(got));
        return reject(reason, cap, "persist_decode must be one of %s, got %s", allowed, got);
    }
    if(!contains(t->nonpersist_decodes, t->num_nonpersist_decodes, s->nonpersist_decode))
    {
        char allowed[ROCKE_ATTENTION_DENSE_REASON_CAP];
        char got[ROCKE_ATTENTION_DENSE_NAME_PART_CAP * 2];
        py_str_list(t->nonpersist_decodes, t->num_nonpersist_decodes, allowed, sizeof(allowed));
        py_repr(s->nonpersist_decode, got, sizeof(got));
        return reject(reason, cap, "nonpersist_decode must be one of %s, got %s", allowed, got);
    }
    if(s->chiplet_num_xcds <= 0)
        return reject(
            reason, cap, "chiplet_num_xcds must be positive, got %d", s->chiplet_num_xcds);
    if(s->persistent)
    {
        if(s->num_persistent <= 0)
            return reject(
                reason, cap, "num_persistent must be positive, got %d", s->num_persistent);
        if(!streq(s->persist_decode, "auto") && !check_persist_decode(s, t, reason, cap))
            return false;
    }
    else if(!streq(s->nonpersist_decode, "auto") && !check_nonpersist_decode(s, reason, cap))
        return false;
    if(s->sliding_window < 0)
        return reject(reason, cap, "sliding_window must be >= 0, got %d", s->sliding_window);
    if(s->sliding_window > 0)
    {
        if(!s->causal && !t->supports_noncausal_window)
            return reject(reason, cap, "sliding_window>0 requires causal=True");
        if(s->sliding_window % s->block_n != 0)
            return reject(reason,
                          cap,
                          "sliding_window (%d) must be a multiple of block_n=%d",
                          s->sliding_window,
                          s->block_n);
    }
    if(s->varlen)
    {
        if(s->persistent)
            return reject(reason, cap, "varlen is not supported with persistent=True");
        if(!s->causal)
            return reject(reason, cap, "varlen requires causal=True");
    }
    if(s->waves_per_eu < 1 || s->waves_per_eu > 8)
        return reject(reason, cap, "waves_per_eu must be in [1, 8], got %d", s->waves_per_eu);

    if(s->paged)
    {
        if(s->block_size <= 0)
            return reject(reason, cap, "paged=True requires block_size > 0");
        if((s->block_size & (s->block_size - 1)) != 0)
            return reject(
                reason, cap, "paged block_size (%d) must be a power of two", s->block_size);
        if(s->block_n % s->block_size != 0)
            return reject(reason,
                          cap,
                          "block_n (%d) must be a multiple of page block_size (%d)",
                          s->block_n,
                          s->block_size);
        /* Python: rows_per_wave = block_n // num_waves. A block_m below 32 makes
         * num_waves 0, and a num_waves above block_n makes rows_per_wave 0 (then
         * block_size % 0); both raise ZeroDivisionError. Its text is the CPython
         * 3.14 one. */
        const int num_waves = s->block_m / 32;
        if(num_waves == 0)
            return reject(reason, cap, "division by zero");
        const int rows_per_wave = s->block_n / num_waves;
        if(s->block_size < rows_per_wave)
            return reject(reason,
                          cap,
                          "paged block_size (%d) must be >= and a multiple of ROWS_PER_WAVE (%d)",
                          s->block_size,
                          rows_per_wave);
        if(rows_per_wave == 0)
            return reject(reason, cap, "division by zero");
        if(s->block_size % rows_per_wave != 0)
            return reject(reason,
                          cap,
                          "paged block_size (%d) must be >= and a multiple of ROWS_PER_WAVE (%d)",
                          s->block_size,
                          rows_per_wave);
        if(s->num_kv_blocks <= 0)
            return reject(reason, cap, "paged=True requires num_kv_blocks > 0");
        const ExactProduct cache_bytes(
            {s->num_kv_blocks, s->block_size, s->num_kv_heads, s->head_size, 2});
        if(cache_bytes.at_least(2147483648u))
        {
            char num[64];
            cache_bytes.str(num, sizeof(num));
            return reject(reason, cap, "paged cache %s B exceeds i32 addressing (2 GiB)", num);
        }
        if(s->batch != 1)
            return reject(reason, cap, "paged multi-sequence (batch>1) not yet implemented");
        if(s->varlen)
            return reject(reason, cap, "paged varlen not yet implemented (single-seq only)");
        if(s->persistent)
            return reject(reason,
                          cap,
                          "paged + persistent not yet implemented (persistent builder is "
                          "contiguous-only)");
        if(s->head_size != 128)
            return reject(reason, cap, "paged not yet implemented for head_size != 128");
        if(!streq(s->dtype, "fp16") && !streq(s->dtype, "bf16"))
            return reject(reason, cap, "paged not yet implemented for dtype=%s", s->dtype);
        if(s->sliding_window <= 0)
            return reject(
                reason, cap, "paged not yet implemented for plain-causal (sliding_window>0 only)");
    }
    if(s->use_sinks && s->paged)
        return reject(reason, cap, "use_sinks is not yet supported with paged KV");
    if(s->use_sinks && s->varlen)
        return reject(reason, cap, "use_sinks is not yet supported with varlen");
    return true;
}

int rocke_attention_dense_spec_num_waves(const rocke_attention_dense_spec_t* s)
{
    return (int)py_floordiv(s->block_m, 32);
}

int rocke_attention_dense_spec_num_queries_per_kv(const rocke_attention_dense_spec_t* s)
{
    return (int)py_floordiv(s->num_query_heads, s->num_kv_heads);
}

const char*
    rocke_attention_dense_spec_resolved_persist_decode(const rocke_attention_dense_spec_t* s,
                                                       const rocke_attention_dense_traits_t* t)
{
    if(!streq(s->persist_decode, "auto"))
        return s->persist_decode;

    const int64_t gqa = dense_gqa(s);
    const int64_t nqb = dense_nqb(s);
    if(aligned_causal(s, t->has_right_bound))
    {
        if(s->interleave)
            return "qb_major";
        if(gqa == 1 && s->num_query_heads % s->chiplet_num_xcds == 0
           && s->num_persistent % s->chiplet_num_xcds == 0 && (!balance_causal(s) || nqb % 2 == 0))
        {
            const ExactProduct work({nqb, s->num_query_heads, s->batch});
            const ExactProduct threshold({8, s->num_persistent});
            if(work.compare(threshold) > 0)
                return "hq_minor_swz";
        }
        if(s->batch < s->chiplet_num_xcds)
            return "bt_hkv_minor";
    }

    const ExactProduct per_hkv({gqa, nqb, s->batch});
    const ExactProduct threshold({2, s->num_persistent});
    if(gqa > 1 && per_hkv.compare(threshold) >= 0)
        return "hkv_major";
    return "qb_major";
}

const char*
    rocke_attention_dense_spec_resolved_nonpersist_decode(const rocke_attention_dense_spec_t* s,
                                                          const rocke_attention_dense_traits_t* t)
{
    if(!streq(s->nonpersist_decode, "auto"))
        return s->nonpersist_decode;
    if(!aligned_causal(s, t->has_right_bound))
        return "qb_minor";

    const int64_t nqb = dense_nqb(s);
    if(s->num_query_heads == s->num_kv_heads && s->num_query_heads % s->chiplet_num_xcds == 0
       && nqb >= 16)
    {
        const ExactProduct work({s->batch, s->num_query_heads, nqb});
        if(work.at_least(8192))
            return "hq_minor_swz";
    }
    return "bt_hkv_minor";
}

void rocke_attention_dense_name_parts_base(const rocke_attention_dense_spec_t* s,
                                           const rocke_attention_dense_traits_t* t,
                                           rocke_attention_dense_name_parts_t* out)
{
    memset(out, 0, sizeof(*out));
    out->head_parts = true;
    out->shape_parts = true;
    if(s->lazy_rescale)
        snprintf(
            out->algorithm[out->num_algorithm++], ROCKE_ATTENTION_DENSE_NAME_PART_CAP, "lazyrs");
    snprintf(out->persist_decode,
             sizeof(out->persist_decode),
             "%s",
             persist_decode_tag(rocke_attention_dense_spec_resolved_persist_decode(s, t)));
    snprintf(out->nonpersist_decode,
             sizeof(out->nonpersist_decode),
             "%s",
             nonpersist_decode_tag(rocke_attention_dense_spec_resolved_nonpersist_decode(s, t)));
}

rocke_status_t rocke_attention_dense_kernel_name(const rocke_attention_dense_spec_t* s,
                                                 const rocke_attention_dense_name_parts_t* np,
                                                 char* out,
                                                 size_t out_cap)
{
    if(s == NULL || np == NULL || out == NULL)
        return ROCKE_ERR_VALUE;

    enum
    {
        CAP = ROCKE_ATTENTION_DENSE_NAME_PART_CAP
    };
    /* Python kernel_name()'s parts list, in order. */
    char buf[48][CAP];
    const char* parts[48];
    size_t n = 0;
    auto add = [&](const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        (void)vsnprintf(buf[n], CAP, fmt, ap);
        va_end(ap);
        parts[n] = buf[n];
        ++n;
    };
    auto add_list = [&](const char(*items)[CAP], int count) {
        for(int i = 0; i < count; ++i)
            add("%s", items[i]);
    };

    add("d%d", s->head_size);
    if(np->head_parts)
    {
        add("hq%d", s->num_query_heads);
        add("kv%d", s->num_kv_heads);
    }
    add("bn%d", s->block_n);
    add("%s", s->dtype);
    if(s->block_m != ROCKE_DENSE_DEFAULT_BLOCK_M)
        add("bm%d", s->block_m);
    if(128 / s->head_size > 1)
        add("kpad%d", s->lds_k_group_pad);
    add_list(np->layout, np->num_layout);
    if(np->shape_parts)
    {
        add("sq%d", s->seqlen_q);
        add("sk%d", s->seqlen_kv);
    }
    add("%s", s->causal ? "causal" : "full");
    if(s->causal_bottom_right)
        add("br");
    add_list(np->band, np->num_band);
    if(s->ragged)
        add("ragged");
    if(s->sliding_window > 0)
        add("swa%d", s->sliding_window);
    if(s->use_sinks)
        add("sinks");
    if(s->varlen)
        add("varlen");
    if(s->paged)
    {
        add("pgd%d", s->block_size);
        add("nb%d", s->num_kv_blocks);
    }
    add_list(np->algorithm, np->num_algorithm);
    if(s->persistent)
    {
        add("persist%d", s->num_persistent);
        if(np->persist_decode[0] != '\0')
            add("%s", np->persist_decode);
        if(s->interleave)
            add("intl");
    }
    else if(np->nonpersist_decode[0] != '\0')
        add("%s", np->nonpersist_decode);
    if(s->chiplet_num_xcds != 8)
        add("xcd%d", s->chiplet_num_xcds);
    return rocke_kernel_name_join(
        "rocke_attention_dense", parts, n, NULL, NULL, 0, out, out_cap, NULL);
}

bool rocke_attention_dense_spec_preflight(const rocke_attention_dense_spec_t* s,
                                          const char* type_name,
                                          rocke_attention_dense_validate_fn validate,
                                          const void* ctx,
                                          char* reason,
                                          size_t cap)
{
    if(s == NULL || validate == NULL)
        return reject(reason, cap, "attention dense spec is NULL");

    /* 1. Re-run the concrete type's validators. */
    char why[ROCKE_ATTENTION_DENSE_REASON_CAP];
    why[0] = '\0';
    if(!validate(ctx, why, sizeof(why)))
        return reject(reason, cap, "invalid %s: %s", type_name, why);

    /* 2. Positive extents. */
    const struct
    {
        const char* name;
        int value;
    } extents[] = {
        {"batch", s->batch},
        {"seqlen_q", s->seqlen_q},
        {"seqlen_kv", s->seqlen_kv},
        {"num_query_heads", s->num_query_heads},
        {"num_kv_heads", s->num_kv_heads},
        {"head_size", s->head_size},
    };
    for(const auto& e : extents)
        if(e.value <= 0)
            return reject(reason, cap, "%s must be positive, got %d", e.name, e.value);

    /* 3. block_n divides the query tile. */
    if(s->block_m % s->block_n != 0)
        return reject(reason,
                      cap,
                      "block_n must divide the %d-row query tile (got block_n=%d; the spec also "
                      "requires block_n %% 32 == 0, so use 32, 64, 128 or 256). Load-bearing for "
                      "causal=True, where n_per = %d // block_n floors and drops keys",
                      s->block_m,
                      s->block_n,
                      s->block_m);

    /* 4. 32-bit addressing. */
    const ExactProduct kv_bytes({s->batch, s->seqlen_kv, s->num_kv_heads, s->head_size, 2});
    if(kv_bytes.at_least((uint64_t)ROCKE_ATTENTION_DENSE_INT32_LIMIT))
    {
        char num[64];
        kv_bytes.str(num, sizeof(num));
        return reject(reason,
                      cap,
                      "K/V extent is %s B, at or past the 32-bit buffer-resource limit (%lld B)",
                      num,
                      ROCKE_ATTENTION_DENSE_INT32_LIMIT);
    }
    const ExactProduct qo_elems({s->batch, s->seqlen_q, s->num_query_heads, s->head_size});
    if(qo_elems.at_least((uint64_t)ROCKE_ATTENTION_DENSE_INT32_LIMIT))
    {
        char num[64];
        qo_elems.str(num, sizeof(num));
        return reject(reason,
                      cap,
                      "Q/O extent is %s elements, at or past the 32-bit addressing limit (%lld)",
                      num,
                      ROCKE_ATTENTION_DENSE_INT32_LIMIT);
    }
    if(reason != NULL && cap > 0)
        reason[0] = '\0';
    return true;
}
