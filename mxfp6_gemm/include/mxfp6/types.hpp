#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace mxfp6 {

// ---- FP6 E2M3 ----
// 1 sign + 2 exponent + 3 mantissa, bias=1, no Inf/NaN
// Range: ±7.5 (max), ±0.125 (min subnormal), ±1.0 (min normal)

inline float fp6_e2m3_to_float(uint8_t bits6) {
    uint8_t S = (bits6 >> 5) & 1;
    uint8_t E = (bits6 >> 3) & 0x3;
    uint8_t M = bits6 & 0x7;
    float val;
    if (E == 0) {
        val = M * 0.125f;
    } else {
        val = ldexpf(1.0f + M * 0.125f, E - 1);
    }
    return S ? -val : val;
}

inline uint8_t float_to_fp6_e2m3(float v) {
    uint8_t sign = 0;
    if (v < 0.0f) {
        sign = 1;
        v = -v;
    }
    if (v == 0.0f) return sign << 5;
    if (v > 7.5f) v = 7.5f;

    if (v < 1.0f) {
        int M = (int)roundf(v * 8.0f);
        float residual = v * 8.0f - (float)(int)(v * 8.0f);
        if (residual == 0.5f) {
            int m_floor = (int)(v * 8.0f);
            M = (m_floor % 2 == 0) ? m_floor : m_floor + 1;
        }
        if (M <= 0) return sign << 5;
        if (M >= 8) return (sign << 5) | (1 << 3);
        return (sign << 5) | (uint8_t)M;
    }

    int E;
    float frac;
    if (v < 2.0f) {
        E = 1;
        frac = v / 1.0f - 1.0f;
    } else if (v < 4.0f) {
        E = 2;
        frac = v / 2.0f - 1.0f;
    } else {
        E = 3;
        frac = v / 4.0f - 1.0f;
    }

    int M = (int)roundf(frac * 8.0f);
    float residual = frac * 8.0f - (float)(int)(frac * 8.0f);
    if (residual == 0.5f) {
        int m_floor = (int)(frac * 8.0f);
        M = (m_floor % 2 == 0) ? m_floor : m_floor + 1;
    }
    if (M >= 8) {
        E += 1;
        M = 0;
        if (E > 3) return (sign << 5) | (3 << 3) | 7;
    }
    return (sign << 5) | (E << 3) | (uint8_t)M;
}

// ---- E8M0 ----
// 8-bit exponent only, bias=127, no sign/mantissa
// value = 2^(code - 127),  code=0 → 2^-127,  code=255 → NaN

inline float e8m0_to_float(uint8_t code) {
    if (code == 255) return NAN;
    return ldexpf(1.0f, (int)code - 127);
}

inline uint8_t float_to_e8m0(float v) {
    if (std::isnan(v)) return 255;
    if (v <= 0.0f) return 0;
    int exp;
    float mant = frexpf(v, &exp);
    exp -= 1;
    if (mant * 2.0f >= 1.41421356f) exp += 1;
    int code = exp + 127;
    return (uint8_t)std::clamp(code, 0, 254);
}

// ---- FP6 dense packing ----
// 4 FP6 values (6 bits each) ↔ 3 bytes, bit-contiguous LSB-first

inline void pack_fp6x4(const uint8_t* in, uint8_t* out) {
    out[0] = (in[0] & 0x3F) | ((in[1] & 0x03) << 6);
    out[1] = ((in[1] >> 2) & 0x0F) | ((in[2] & 0x0F) << 4);
    out[2] = ((in[2] >> 4) & 0x03) | ((in[3] & 0x3F) << 2);
}

inline void unpack_fp6x4(const uint8_t* in, uint8_t* out) {
    out[0] = in[0] & 0x3F;
    out[1] = ((in[0] >> 6) & 0x03) | ((in[1] & 0x0F) << 2);
    out[2] = ((in[1] >> 4) & 0x0F) | ((in[2] & 0x03) << 4);
    out[3] = (in[2] >> 2) & 0x3F;
}

inline void pack_fp6(const uint8_t* vals, int n, uint8_t* packed) {
    assert(n % 4 == 0);
    for (int i = 0; i < n; i += 4) pack_fp6x4(vals + i, packed + (i / 4) * 3);
}

inline void unpack_fp6(const uint8_t* packed, int n, uint8_t* vals) {
    assert(n % 4 == 0);
    for (int i = 0; i < n; i += 4) unpack_fp6x4(packed + (i / 4) * 3, vals + i);
}

inline int fp6_packed_bytes(int n) {
    assert(n % 4 == 0);
    return n * 3 / 4;
}

}  // namespace mxfp6
