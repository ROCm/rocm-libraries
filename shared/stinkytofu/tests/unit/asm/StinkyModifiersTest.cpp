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
#include <gtest/gtest.h>

#include <string>

#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

using namespace stinkytofu;

// ---------------------------------------------------------------------------
// dppCtrlToAsmStr
// ---------------------------------------------------------------------------

TEST(DppCtrlToAsmStr, QuadPerm) {
    // quad_perm:[0,1,2,3] → identity
    EXPECT_EQ(dppCtrlToAsmStr(dppQuadPerm(0, 1, 2, 3)), "quad_perm:[0,1,2,3]");
    // quad_perm:[3,2,1,0] → reverse
    EXPECT_EQ(dppCtrlToAsmStr(dppQuadPerm(3, 2, 1, 0)), "quad_perm:[3,2,1,0]");
    // all lanes to lane 0
    EXPECT_EQ(dppCtrlToAsmStr(dppQuadPerm(0, 0, 0, 0)), "quad_perm:[0,0,0,0]");
}

TEST(DppCtrlToAsmStr, RowShl) {
    EXPECT_EQ(dppCtrlToAsmStr(dppRowShl(1)), "row_shl:1");
    EXPECT_EQ(dppCtrlToAsmStr(dppRowShl(8)), "row_shl:8");
    EXPECT_EQ(dppCtrlToAsmStr(dppRowShl(15)), "row_shl:15");
}

TEST(DppCtrlToAsmStr, RowShr) {
    EXPECT_EQ(dppCtrlToAsmStr(dppRowShr(1)), "row_shr:1");
    EXPECT_EQ(dppCtrlToAsmStr(dppRowShr(7)), "row_shr:7");
    EXPECT_EQ(dppCtrlToAsmStr(dppRowShr(15)), "row_shr:15");
}

TEST(DppCtrlToAsmStr, RowRor) {
    EXPECT_EQ(dppCtrlToAsmStr(dppRowRor(1)), "row_ror:1");
    EXPECT_EQ(dppCtrlToAsmStr(dppRowRor(15)), "row_ror:15");
}

TEST(DppCtrlToAsmStr, WaveOps) {
    EXPECT_EQ(dppCtrlToAsmStr(DppCtrl::WAVE_SHL1), "wave_shl:1");
    EXPECT_EQ(dppCtrlToAsmStr(DppCtrl::WAVE_ROL1), "wave_rol:1");
    EXPECT_EQ(dppCtrlToAsmStr(DppCtrl::WAVE_SHR1), "wave_shr:1");
    EXPECT_EQ(dppCtrlToAsmStr(DppCtrl::WAVE_ROR1), "wave_ror:1");
}

TEST(DppCtrlToAsmStr, RowMirror) {
    EXPECT_EQ(dppCtrlToAsmStr(DppCtrl::ROW_MIRROR), "row_mirror");
    EXPECT_EQ(dppCtrlToAsmStr(DppCtrl::ROW_HALF_MIRROR), "row_half_mirror");
}

TEST(DppCtrlToAsmStr, Bcast) {
    EXPECT_EQ(dppCtrlToAsmStr(DppCtrl::BCAST15), "row_bcast:15");
    EXPECT_EQ(dppCtrlToAsmStr(DppCtrl::BCAST31), "row_bcast:31");
}

TEST(DppCtrlToAsmStr, RowShare) {
    EXPECT_EQ(dppCtrlToAsmStr(dppRowShare(0)), "row_share:0");
    EXPECT_EQ(dppCtrlToAsmStr(dppRowShare(7)), "row_share:7");
    EXPECT_EQ(dppCtrlToAsmStr(dppRowShare(15)), "row_share:15");
}

TEST(DppCtrlToAsmStr, RowXmask) {
    EXPECT_EQ(dppCtrlToAsmStr(dppRowXmask(0)), "row_xmask:0");
    EXPECT_EQ(dppCtrlToAsmStr(dppRowXmask(15)), "row_xmask:15");
}

TEST(DppCtrlToAsmStr, NoneReturnsEmpty) {
    EXPECT_TRUE(dppCtrlToAsmStr(DppCtrl::NONE).empty());
}

// ---------------------------------------------------------------------------
// parseDppCtrlFromAsm — round-trips and edge cases
// ---------------------------------------------------------------------------

TEST(ParseDppCtrlFromAsm, QuadPermRoundTrip) {
    auto ctrl = dppQuadPerm(1, 2, 3, 0);
    EXPECT_EQ(parseDppCtrlFromAsm(dppCtrlToAsmStr(ctrl)), ctrl);
}

TEST(ParseDppCtrlFromAsm, RowShlRoundTrip) {
    for (int n = 1; n <= 15; ++n) {
        auto ctrl = dppRowShl(n);
        EXPECT_EQ(parseDppCtrlFromAsm(dppCtrlToAsmStr(ctrl)), ctrl) << "row_shl:" << n;
    }
}

TEST(ParseDppCtrlFromAsm, RowShrRoundTrip) {
    for (int n = 1; n <= 15; ++n) {
        auto ctrl = dppRowShr(n);
        EXPECT_EQ(parseDppCtrlFromAsm(dppCtrlToAsmStr(ctrl)), ctrl) << "row_shr:" << n;
    }
}

TEST(ParseDppCtrlFromAsm, RowRorRoundTrip) {
    for (int n = 1; n <= 15; ++n) {
        auto ctrl = dppRowRor(n);
        EXPECT_EQ(parseDppCtrlFromAsm(dppCtrlToAsmStr(ctrl)), ctrl) << "row_ror:" << n;
    }
}

TEST(ParseDppCtrlFromAsm, FixedModes) {
    EXPECT_EQ(parseDppCtrlFromAsm("row_mirror"), DppCtrl::ROW_MIRROR);
    EXPECT_EQ(parseDppCtrlFromAsm("row_half_mirror"), DppCtrl::ROW_HALF_MIRROR);
    EXPECT_EQ(parseDppCtrlFromAsm("wave_shl:1"), DppCtrl::WAVE_SHL1);
    EXPECT_EQ(parseDppCtrlFromAsm("wave_rol:1"), DppCtrl::WAVE_ROL1);
    EXPECT_EQ(parseDppCtrlFromAsm("wave_shr:1"), DppCtrl::WAVE_SHR1);
    EXPECT_EQ(parseDppCtrlFromAsm("wave_ror:1"), DppCtrl::WAVE_ROR1);
}

TEST(ParseDppCtrlFromAsm, RowBcast) {
    EXPECT_EQ(parseDppCtrlFromAsm("row_bcast:15"), DppCtrl::BCAST15);
    EXPECT_EQ(parseDppCtrlFromAsm("row_bcast:31"), DppCtrl::BCAST31);
}

TEST(ParseDppCtrlFromAsm, RowShareRoundTrip) {
    for (int n = 0; n <= 15; ++n) {
        auto ctrl = dppRowShare(n);
        EXPECT_EQ(parseDppCtrlFromAsm(dppCtrlToAsmStr(ctrl)), ctrl) << "row_share:" << n;
    }
}

TEST(ParseDppCtrlFromAsm, RowXmaskRoundTrip) {
    for (int n = 0; n <= 15; ++n) {
        auto ctrl = dppRowXmask(n);
        EXPECT_EQ(parseDppCtrlFromAsm(dppCtrlToAsmStr(ctrl)), ctrl) << "row_xmask:" << n;
    }
}

TEST(ParseDppCtrlFromAsm, InvalidReturnsNone) {
    EXPECT_EQ(parseDppCtrlFromAsm(""), DppCtrl::NONE);
    EXPECT_EQ(parseDppCtrlFromAsm("not_a_ctrl"), DppCtrl::NONE);
    EXPECT_EQ(parseDppCtrlFromAsm("row_shl:0"), DppCtrl::NONE);   // 0 out of range [1..15]
    EXPECT_EQ(parseDppCtrlFromAsm("row_shl:16"), DppCtrl::NONE);  // 16 out of range
    EXPECT_EQ(parseDppCtrlFromAsm("row_bcast:7"), DppCtrl::NONE); // only 15/31 valid
    EXPECT_EQ(parseDppCtrlFromAsm("quad_perm:[]"), DppCtrl::NONE);
}

// ---------------------------------------------------------------------------
// matrixFmtToStr / parseMatrixFmt
// ---------------------------------------------------------------------------

TEST(MatrixFmt, ToStrAndParse) {
    EXPECT_EQ(matrixFmtToStr(MatrixFmt::FP8), "MATRIX_FMT_FP8");
    EXPECT_EQ(matrixFmtToStr(MatrixFmt::BF8), "MATRIX_FMT_BF8");
    EXPECT_EQ(matrixFmtToStr(MatrixFmt::FP6), "MATRIX_FMT_FP6");
    EXPECT_EQ(matrixFmtToStr(MatrixFmt::BF6), "MATRIX_FMT_BF6");
    EXPECT_EQ(matrixFmtToStr(MatrixFmt::FP4), "MATRIX_FMT_FP4");
    EXPECT_TRUE(matrixFmtToStr(MatrixFmt::NONE).empty());
}

TEST(MatrixFmt, ParseRoundTrip) {
    for (auto fmt : {MatrixFmt::FP8, MatrixFmt::BF8, MatrixFmt::FP6, MatrixFmt::BF6, MatrixFmt::FP4}) {
        EXPECT_EQ(parseMatrixFmt(matrixFmtToStr(fmt)), fmt);
    }
}

TEST(MatrixFmt, ParseInvalidReturnsNone) {
    EXPECT_EQ(parseMatrixFmt(""), MatrixFmt::NONE);
    EXPECT_EQ(parseMatrixFmt("FP8"), MatrixFmt::NONE);
    EXPECT_EQ(parseMatrixFmt("MATRIX_FMT_INT8"), MatrixFmt::NONE);
}

// ---------------------------------------------------------------------------
// matrixScaleFmtToStr / parseMatrixScaleFmt
// ---------------------------------------------------------------------------

TEST(MatrixScaleFmt, ToStrAndParse) {
    EXPECT_EQ(matrixScaleFmtToStr(MatrixScaleFmt::E8), "MATRIX_SCALE_FMT_E8");
    EXPECT_EQ(matrixScaleFmtToStr(MatrixScaleFmt::E5M3), "MATRIX_SCALE_FMT_E5M3");
    EXPECT_EQ(matrixScaleFmtToStr(MatrixScaleFmt::E4M3), "MATRIX_SCALE_FMT_E4M3");
    EXPECT_TRUE(matrixScaleFmtToStr(MatrixScaleFmt::NONE).empty());
}

TEST(MatrixScaleFmt, ParseSymbolicRoundTrip) {
    for (auto fmt : {MatrixScaleFmt::E8, MatrixScaleFmt::E5M3, MatrixScaleFmt::E4M3}) {
        EXPECT_EQ(parseMatrixScaleFmt(matrixScaleFmtToStr(fmt)), fmt);
    }
}

TEST(MatrixScaleFmt, ParseIntegerForms) {
    EXPECT_EQ(parseMatrixScaleFmt("0"), MatrixScaleFmt::E8);
    EXPECT_EQ(parseMatrixScaleFmt("1"), MatrixScaleFmt::E5M3);
    EXPECT_EQ(parseMatrixScaleFmt("2"), MatrixScaleFmt::E4M3);
}

TEST(MatrixScaleFmt, ParseInvalidReturnsNone) {
    EXPECT_EQ(parseMatrixScaleFmt(""), MatrixScaleFmt::NONE);
    EXPECT_EQ(parseMatrixScaleFmt("3"), MatrixScaleFmt::NONE);
    EXPECT_EQ(parseMatrixScaleFmt("E8"), MatrixScaleFmt::NONE);
}
