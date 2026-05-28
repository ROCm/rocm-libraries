/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
 * ************************************************************************ */

#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

#include <cstdlib>

namespace stinkytofu {

// --- DPP control conversion utilities ---

std::string dppCtrlToAsmStr(DppCtrl ctrl) {
    auto v = static_cast<uint16_t>(ctrl);

    // quad_perm: 0x000-0x0FF
    if (v <= static_cast<uint16_t>(DppCtrl::QUAD_PERM_LAST)) {
        uint8_t p0 = v & 3, p1 = (v >> 2) & 3, p2 = (v >> 4) & 3, p3 = (v >> 6) & 3;
        return "quad_perm:[" + std::to_string(p0) + "," + std::to_string(p1) + "," +
               std::to_string(p2) + "," + std::to_string(p3) + "]";
    }
    // row_shl: 0x101-0x10F
    if (v >= static_cast<uint16_t>(DppCtrl::ROW_SHL_FIRST) &&
        v <= static_cast<uint16_t>(DppCtrl::ROW_SHL_LAST))
        return "row_shl:" + std::to_string(v - static_cast<uint16_t>(DppCtrl::ROW_SHL0));
    // row_shr: 0x111-0x11F
    if (v >= static_cast<uint16_t>(DppCtrl::ROW_SHR_FIRST) &&
        v <= static_cast<uint16_t>(DppCtrl::ROW_SHR_LAST))
        return "row_shr:" + std::to_string(v - static_cast<uint16_t>(DppCtrl::ROW_SHR0));
    // row_ror: 0x121-0x12F
    if (v >= static_cast<uint16_t>(DppCtrl::ROW_ROR_FIRST) &&
        v <= static_cast<uint16_t>(DppCtrl::ROW_ROR_LAST))
        return "row_ror:" + std::to_string(v - static_cast<uint16_t>(DppCtrl::ROW_ROR0));
    // wave ops (GFX9)
    if (v == static_cast<uint16_t>(DppCtrl::WAVE_SHL1)) return "wave_shl:1";
    if (v == static_cast<uint16_t>(DppCtrl::WAVE_ROL1)) return "wave_rol:1";
    if (v == static_cast<uint16_t>(DppCtrl::WAVE_SHR1)) return "wave_shr:1";
    if (v == static_cast<uint16_t>(DppCtrl::WAVE_ROR1)) return "wave_ror:1";
    // mirror
    if (v == static_cast<uint16_t>(DppCtrl::ROW_MIRROR)) return "row_mirror";
    if (v == static_cast<uint16_t>(DppCtrl::ROW_HALF_MIRROR)) return "row_half_mirror";
    // bcast (GFX9)
    if (v == static_cast<uint16_t>(DppCtrl::BCAST15)) return "row_bcast:15";
    if (v == static_cast<uint16_t>(DppCtrl::BCAST31)) return "row_bcast:31";
    // row_share: 0x150-0x15F
    if (v >= static_cast<uint16_t>(DppCtrl::ROW_SHARE_FIRST) &&
        v <= static_cast<uint16_t>(DppCtrl::ROW_SHARE_LAST))
        return "row_share:" + std::to_string(v - static_cast<uint16_t>(DppCtrl::ROW_SHARE_FIRST));
    // row_xmask: 0x160-0x16F
    if (v >= static_cast<uint16_t>(DppCtrl::ROW_XMASK_FIRST) &&
        v <= static_cast<uint16_t>(DppCtrl::ROW_XMASK_LAST))
        return "row_xmask:" + std::to_string(v - static_cast<uint16_t>(DppCtrl::ROW_XMASK_FIRST));

    return {};
}

// Helper: parse trailing integer after "prefix:" (e.g. "row_shr:3" -> 3).
static int parseTrailingInt(std::string_view s, std::string_view prefix) {
    if (s.size() <= prefix.size()) return -1;
    auto numStr = s.substr(prefix.size());
    char* end = nullptr;
    long val = std::strtol(std::string(numStr).c_str(), &end, 10);
    return (end && *end == '\0') ? static_cast<int>(val) : -1;
}

DppCtrl parseDppCtrlFromAsm(std::string_view s) {
    // quad_perm:[a,b,c,d]
    if (s.substr(0, 10) == "quad_perm:") {
        auto bracket = s.substr(10);
        if (bracket.size() >= 9 && bracket.front() == '[' && bracket.back() == ']') {
            auto inner = bracket.substr(1, bracket.size() - 2);
            uint8_t vals[4] = {0, 0, 0, 0};
            int idx = 0;
            for (size_t i = 0; i < inner.size() && idx < 4; ++i) {
                if (inner[i] >= '0' && inner[i] <= '3') {
                    vals[idx++] = static_cast<uint8_t>(inner[i] - '0');
                }
            }
            if (idx == 4) return dppQuadPerm(vals[0], vals[1], vals[2], vals[3]);
        }
        return DppCtrl::NONE;
    }

    // Parameterized modes — validate using FIRST/LAST enum range
    constexpr int kRowShiftMax =
        static_cast<uint16_t>(DppCtrl::ROW_SHL_LAST) - static_cast<uint16_t>(DppCtrl::ROW_SHL0);
    constexpr int kRowShareMax = static_cast<uint16_t>(DppCtrl::ROW_SHARE_LAST) -
                                 static_cast<uint16_t>(DppCtrl::ROW_SHARE_FIRST);
    constexpr int kRowXmaskMax = static_cast<uint16_t>(DppCtrl::ROW_XMASK_LAST) -
                                 static_cast<uint16_t>(DppCtrl::ROW_XMASK_FIRST);

    if (s.substr(0, 8) == "row_shl:") {
        int n = parseTrailingInt(s, "row_shl:");
        if (n >= 1 && n <= kRowShiftMax) return dppRowShl(n);
    }
    if (s.substr(0, 8) == "row_shr:") {
        int n = parseTrailingInt(s, "row_shr:");
        if (n >= 1 && n <= kRowShiftMax) return dppRowShr(n);
    }
    if (s.substr(0, 8) == "row_ror:") {
        int n = parseTrailingInt(s, "row_ror:");
        if (n >= 1 && n <= kRowShiftMax) return dppRowRor(n);
    }
    if (s.substr(0, 10) == "row_share:") {
        int n = parseTrailingInt(s, "row_share:");
        if (n >= 0 && n <= kRowShareMax) return dppRowShare(n);
    }
    if (s.substr(0, 10) == "row_xmask:") {
        int n = parseTrailingInt(s, "row_xmask:");
        if (n >= 0 && n <= kRowXmaskMax) return dppRowXmask(n);
    }
    if (s.substr(0, 10) == "row_bcast:") {
        int n = parseTrailingInt(s, "row_bcast:");
        if (n == 15) return DppCtrl::BCAST15;
        if (n == 31) return DppCtrl::BCAST31;
    }

    // Fixed modes
    if (s == "row_mirror") return DppCtrl::ROW_MIRROR;
    if (s == "row_half_mirror") return DppCtrl::ROW_HALF_MIRROR;
    if (s == "wave_shl:1") return DppCtrl::WAVE_SHL1;
    if (s == "wave_rol:1") return DppCtrl::WAVE_ROL1;
    if (s == "wave_shr:1") return DppCtrl::WAVE_SHR1;
    if (s == "wave_ror:1") return DppCtrl::WAVE_ROR1;

    return DppCtrl::NONE;
}

// --- MFMA matrix format conversion utilities ---

std::string matrixFmtToStr(MatrixFmt fmt) {
    switch (fmt) {
        case MatrixFmt::FP8:
            return "MATRIX_FMT_FP8";
        case MatrixFmt::BF8:
            return "MATRIX_FMT_BF8";
        case MatrixFmt::FP6:
            return "MATRIX_FMT_FP6";
        case MatrixFmt::BF6:
            return "MATRIX_FMT_BF6";
        case MatrixFmt::FP4:
            return "MATRIX_FMT_FP4";
        default:
            return {};
    }
}

MatrixFmt parseMatrixFmt(std::string_view s) {
    if (s == "MATRIX_FMT_FP8") return MatrixFmt::FP8;
    if (s == "MATRIX_FMT_BF8") return MatrixFmt::BF8;
    if (s == "MATRIX_FMT_FP6") return MatrixFmt::FP6;
    if (s == "MATRIX_FMT_BF6") return MatrixFmt::BF6;
    if (s == "MATRIX_FMT_FP4") return MatrixFmt::FP4;
    return MatrixFmt::NONE;
}

std::string matrixScaleFmtToStr(MatrixScaleFmt fmt) {
    switch (fmt) {
        case MatrixScaleFmt::E8:
            return "MATRIX_SCALE_FMT_E8";
        case MatrixScaleFmt::E5M3:
            return "MATRIX_SCALE_FMT_E5M3";
        case MatrixScaleFmt::E4M3:
            return "MATRIX_SCALE_FMT_E4M3";
        default:
            return {};
    }
}

MatrixScaleFmt parseMatrixScaleFmt(std::string_view s) {
    if (s == "MATRIX_SCALE_FMT_E8") return MatrixScaleFmt::E8;
    if (s == "MATRIX_SCALE_FMT_E5M3") return MatrixScaleFmt::E5M3;
    if (s == "MATRIX_SCALE_FMT_E4M3") return MatrixScaleFmt::E4M3;
    // Also accept raw integer values (used by rocisa scale_fmt:N syntax)
    if (s == "0") return MatrixScaleFmt::E8;
    if (s == "1") return MatrixScaleFmt::E5M3;
    if (s == "2") return MatrixScaleFmt::E4M3;
    return MatrixScaleFmt::NONE;
}

}  // namespace stinkytofu
