/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <cmath>
#include <migraphx/float_equal.hpp>
#include <migraphx/float8.hpp>
#include <migraphx/half.hpp>
#include <migraphx/ranges.hpp>
#include "test.hpp"

#include <limits>

static float fp8e4m3fn_to_fp32_value(uint8_t input)
{
    constexpr std::array<float, 256> e4m3fnuz_lut = {
        0.0,        0.001953125,  0.00390625,  0.005859375,
        0.0078125,  0.009765625,  0.01171875,  0.013671875,
        0.015625,   0.017578125,  0.01953125,  0.021484375,
        0.0234375,  0.025390625,  0.02734375,  0.029296875,
        0.03125,    0.03515625,   0.0390625,   0.04296875,
        0.046875,   0.05078125,   0.0546875,   0.05859375,
        0.0625,     0.0703125,    0.078125,    0.0859375,
        0.09375,    0.1015625,    0.109375,    0.1171875,
        0.125,      0.140625,     0.15625,     0.171875,
        0.1875,     0.203125,     0.21875,     0.234375,
        0.25,       0.28125,      0.3125,      0.34375,
        0.375,      0.40625,      0.4375,      0.46875,
        0.5,        0.5625,       0.625,       0.6875,
        0.75,       0.8125,       0.875,       0.9375,
        1.0,        1.125,        1.25,        1.375,
        1.5,        1.625,        1.75,        1.875,
        2.0,        2.25,         2.5,         2.75,
        3.0,        3.25,         3.5,         3.75,
        4.0,        4.5,          5.0,         5.5,
        6.0,        6.5,          7.0,         7.5,
        8.0,        9.0,          10.0,        11.0,
        12.0,       13.0,         14.0,        15.0,
        16.0,       18.0,         20.0,        22.0,
        24.0,       26.0,         28.0,        30.0,
        32.0,       36.0,         40.0,        44.0,
        48.0,       52.0,         56.0,        60.0,
        64.0,       72.0,         80.0,        88.0,
        96.0,       104.0,        112.0,       120.0,
        128.0,      144.0,        160.0,       176.0,
        192.0,      208.0,        224.0,       240.0,
        256.0,      288.0,        320.0,       352.0,
        384.0,      416.0,        448.0,       std::numeric_limits<float>::quiet_NaN(),
        -0.0,       -0.001953125, -0.00390625, -0.005859375,
        -0.0078125, -0.009765625, -0.01171875, -0.013671875,
        -0.015625,  -0.017578125, -0.01953125, -0.021484375,
        -0.0234375, -0.025390625, -0.02734375, -0.029296875,
        -0.03125,   -0.03515625,  -0.0390625,  -0.04296875,
        -0.046875,  -0.05078125,  -0.0546875,  -0.05859375,
        -0.0625,    -0.0703125,   -0.078125,   -0.0859375,
        -0.09375,   -0.1015625,   -0.109375,   -0.1171875,
        -0.125,     -0.140625,    -0.15625,    -0.171875,
        -0.1875,    -0.203125,    -0.21875,    -0.234375,
        -0.25,      -0.28125,     -0.3125,     -0.34375,
        -0.375,     -0.40625,     -0.4375,     -0.46875,
        -0.5,       -0.5625,      -0.625,      -0.6875,
        -0.75,      -0.8125,      -0.875,      -0.9375,
        -1.0,       -1.125,       -1.25,       -1.375,
        -1.5,       -1.625,       -1.75,       -1.875,
        -2.0,       -2.25,        -2.5,        -2.75,
        -3.0,       -3.25,        -3.5,        -3.75,
        -4.0,       -4.5,         -5.0,        -5.5,
        -6.0,       -6.5,         -7.0,        -7.5,
        -8.0,       -9.0,         -10.0,       -11.0,
        -12.0,      -13.0,        -14.0,       -15.0,
        -16.0,      -18.0,        -20.0,       -22.0,
        -24.0,      -26.0,        -28.0,       -30.0,
        -32.0,      -36.0,        -40.0,       -44.0,
        -48.0,      -52.0,        -56.0,       -60.0,
        -64.0,      -72.0,        -80.0,       -88.0,
        -96.0,      -104.0,       -112.0,      -120.0,
        -128.0,     -144.0,       -160.0,      -176.0,
        -192.0,     -208.0,       -224.0,      -240.0,
        -256.0,     -288.0,       -320.0,      -352.0,
        -384.0,     -416.0,       -448.0,      std::numeric_limits<float>::quiet_NaN(),

    };

    return e4m3fnuz_lut[input];
}

TEST_CASE(test_fp8_cast_to_float)
{
    std::vector<uint8_t> bit_vals(256);
    std::iota(bit_vals.begin(), bit_vals.end(), 0);
    EXPECT(bool{std::all_of(bit_vals.begin(), bit_vals.end(), [](uint8_t bit_val) {
        migraphx::fp8::fp8e4m3fn fp8_val(bit_val, migraphx::fp8::fp8e4m3fn::from_bits());
        if(std::isnan(float(fp8_val)) and std::isnan(fp8e4m3fn_to_fp32_value(bit_val)))
        {
            return true;
        }
        return migraphx::float_equal(float(fp8_val), fp8e4m3fn_to_fp32_value(bit_val));
    })});
}

TEST_CASE(test_fp8_cast_from_float)
{
    std::unordered_map<float, uint8_t> test_vals = {
        {{512, 0x7e},        {-512, 0xfe},        {448, 0x7e},        {-448, 0xfe},
         {256, 0x78},        {-256, 0xf8},        {240, 0x77},        {-240, 0xf7},
         {1e-07, 0x0},       {1e+07, 0x7e},       {1, 0x38},          {-1, 0xb8},
         {0.1, 0x1d},        {0.11, 0x1e},        {0.111, 0x1e},      {0.1111, 0x1e},
         {-0.1, 0x9d},       {-0.11, 0x9e},       {-0.111, 0x9e},     {-0.1111, 0x9e},
         {0.2, 0x25},        {2, 0x40},           {20, 0x5a},         {200, 0x74},
         {-0.2, 0xa5},       {-2, 0xc0},          {-20, 0xda},        {-200, 0xf4},
         {0.5, 0x30},        {-0.5, 0xb0},        {1.17549e-38, 0x0}, {1.4013e-45, 0x0},
         {0.0078125, 0x4},   {-0.0078125, 0x84},  {0.000976562, 0x0}, {-0.000976562, 0x80},
         {0.000488281, 0x0}, {-0.000488281, 0x80}}};

    EXPECT(bool{std::all_of(test_vals.begin(), test_vals.end(), [](const auto sample) {
        return migraphx::float_equal(
            migraphx::fp8::fp8e4m3fn(sample.first),
            migraphx::fp8::fp8e4m3fn(sample.second, migraphx::fp8::fp8e4m3fn::from_bits()));
    })});
}

TEST_CASE(test_positive_zero)
{
    float zero = 0.0;
    migraphx::fp8::fp8e4m3fn fp8_zero(zero);
    EXPECT(fp8_zero.is_zero());
    EXPECT(migraphx::float_equal(zero, float(fp8_zero)));
}

TEST_CASE(test_negative_zero)
{
    float nzero = -0.0;
    migraphx::fp8::fp8e4m3fn fp8_nzero(nzero);
    EXPECT(fp8_nzero.is_zero());
    //  negative zero is preserved for fp8e4m3fn
    EXPECT(migraphx::float_equal(nzero, float(fp8_nzero)));
}

TEST_CASE(test_pos_zero_eq_neg_zero)
{
    float nzero = -0.0;
    float pzero = 0.0;
    migraphx::fp8::fp8e5m2 fp8_nzero(nzero);
    migraphx::fp8::fp8e5m2 fp8_pzero(pzero);
    EXPECT(fp8_nzero == fp8_pzero);
}

TEST_CASE(test_nan_1)
{
    float fnan = std::numeric_limits<float>::quiet_NaN();
    migraphx::fp8::fp8e4m3fn fp8_nan(fnan);
    EXPECT(fp8_nan.is_nan());
    EXPECT(std::isnan(fp8_nan));
}

TEST_CASE(test_nan_2)
{
    auto fnan = std::numeric_limits<migraphx::fp8::fp8e4m3fn>::quiet_NaN();
    migraphx::fp8::fp8e4m3fn fp8_nan(fnan.data, migraphx::fp8::fp8e4m3fn::from_bits());
    EXPECT(fp8_nan.is_nan());
    EXPECT(std::isnan(fp8_nan));
    EXPECT(std::isnan(float(fp8_nan)));
}

TEST_CASE(test_bool)
{
    float zero  = 0.0;
    float two   = 2.0;
    float other = -0.375;
    migraphx::fp8::fp8e4m3fn fp8_zero(zero);
    migraphx::fp8::fp8e4m3fn fp8_two(two);
    migraphx::fp8::fp8e4m3fn fp8_other(other);
    EXPECT(not static_cast<bool>(fp8_zero));
    EXPECT(static_cast<bool>(fp8_two));
    EXPECT(static_cast<bool>(fp8_other));
}

TEST_CASE(test_infinity_1)
{
    float finf = std::numeric_limits<float>::infinity();
    // no inf in fp8e4m3fn, it gets clipped to max()
    migraphx::fp8::fp8e4m3fn fp8_max(finf);
    EXPECT(fp8_max == std::numeric_limits<migraphx::fp8::fp8e4m3fn>::max());
}

TEST_CASE(test_infinity_2)
{
    // neg inf
    float finf = -1.0 * std::numeric_limits<float>::infinity();
    // no inf in fp8e4m3fn, it gets clipped to lowest
    migraphx::fp8::fp8e4m3fn fp8_lowest(finf);
    EXPECT(bool{fp8_lowest == std::numeric_limits<migraphx::fp8::fp8e4m3fn>::lowest()});
}

TEST_CASE(test_numeric_max_1)
{
    float fmax = std::numeric_limits<float>::max();
    migraphx::fp8::fp8e4m3fn fp8_max(fmax);
    EXPECT(fp8_max == std::numeric_limits<migraphx::fp8::fp8e4m3fn>::max());
}

TEST_CASE(test_numeric_max_2)
{
    // gets clipped to max
    float fmax = 2 * std::numeric_limits<migraphx::fp8::fp8e4m3fn>::max();
    migraphx::fp8::fp8e4m3fn fp8_max(fmax);
    EXPECT(fp8_max == std::numeric_limits<migraphx::fp8::fp8e4m3fn>::max());
}

TEST_CASE(test_numeric_lowest_1)
{
    float flowest = std::numeric_limits<float>::lowest();
    migraphx::fp8::fp8e4m3fn fp8_lowest(flowest);
    EXPECT(fp8_lowest == std::numeric_limits<migraphx::fp8::fp8e4m3fn>::lowest());
}

TEST_CASE(test_numeric_lowest_2)
{
    // gets clipped to lowest
    float fmin = 2.0 * std::numeric_limits<migraphx::fp8::fp8e4m3fn>::lowest();
    migraphx::fp8::fp8e4m3fn fp8_lowest(fmin);
    EXPECT(fp8_lowest == std::numeric_limits<migraphx::fp8::fp8e4m3fn>::lowest());
}

TEST_CASE(test_max_eq_lowest)
{
    EXPECT(migraphx::float_equal(std::numeric_limits<migraphx::fp8::fp8e4m3fn>::lowest(),
                                 -1 * std::numeric_limits<migraphx::fp8::fp8e4m3fn>::max()));
}

TEST_CASE(test_isfinite)
{
    EXPECT(std::isfinite(migraphx::fp8::fp8e4m3fn(0.0)));
    EXPECT(std::isfinite(migraphx::fp8::fp8e4m3fn(-0.0)));
    EXPECT(not std::isfinite(
        migraphx::fp8::fp8e4m3fn(std::numeric_limits<migraphx::fp8::fp8e4m3fn>::quiet_NaN())));
}

TEST_CASE(test_no_infinity)
{
    EXPECT(not bool{std::numeric_limits<migraphx::fp8::fp8e4m3fn>::has_infinity});
}

TEST_CASE(test_binary_ops)
{
    auto a = migraphx::fp8::fp8e4m3fn(-1.0);
    auto b = migraphx::fp8::fp8e4m3fn(1.0);
    auto c = migraphx::fp8::fp8e4m3fn(0.0);
    auto d = migraphx::fp8::fp8e4m3fn(-0.0);
    EXPECT(migraphx::float_equal((c + d), c));
    EXPECT(migraphx::float_equal((c + d), d));
    EXPECT(migraphx::float_equal((a + b), c));
    EXPECT(migraphx::float_equal((a + b), d));

    auto e = migraphx::fp8::fp8e4m3fn(10.0);
    auto f = migraphx::fp8::fp8e4m3fn(-10.0);
    EXPECT(bool{e > f});
    EXPECT(bool{f < e});
    EXPECT(bool{f <= e});
    EXPECT(bool{e >= f});
    EXPECT(bool{e <= e});
    EXPECT(bool{f >= f});
    EXPECT(not migraphx::float_equal(f, e));
}

TEST_CASE(test_fabs)
{
    auto a = migraphx::fp8::fp8e4m3fn(-1.0);
    auto b = migraphx::fp8::fp8e4m3fn(1.0);
    EXPECT(migraphx::float_equal(b, migraphx::fp8::fabs(a)));
}

TEST_CASE(test_stream_op)
{
    auto a = migraphx::fp8::fp8e4m3fn(-1.0);
    std::stringstream ss;
    ss << a;
    EXPECT(std::string("-1") == ss.str());
    ss     = std::stringstream();
    auto b = std::numeric_limits<migraphx::fp8::fp8e4m3fn>::quiet_NaN();
    ss << b;
    EXPECT(std::string("nan") == ss.str());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
