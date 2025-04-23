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
#include <sstream>

static float fp8e5m2_to_fp32_value(uint8_t input)
{
    constexpr std::array<float, 256> e4m3fnuz_lut = {
        0.0,
        1.52587890625e-05,
        3.0517578125e-05,
        4.57763671875e-05,
        6.103515625e-05,
        7.62939453125e-05,
        9.1552734375e-05,
        0.0001068115234375,
        0.0001220703125,
        0.000152587890625,
        0.00018310546875,
        0.000213623046875,
        0.000244140625,
        0.00030517578125,
        0.0003662109375,
        0.00042724609375,
        0.00048828125,
        0.0006103515625,
        0.000732421875,
        0.0008544921875,
        0.0009765625,
        0.001220703125,
        0.00146484375,
        0.001708984375,
        0.001953125,
        0.00244140625,
        0.0029296875,
        0.00341796875,
        0.00390625,
        0.0048828125,
        0.005859375,
        0.0068359375,
        0.0078125,
        0.009765625,
        0.01171875,
        0.013671875,
        0.015625,
        0.01953125,
        0.0234375,
        0.02734375,
        0.03125,
        0.0390625,
        0.046875,
        0.0546875,
        0.0625,
        0.078125,
        0.09375,
        0.109375,
        0.125,
        0.15625,
        0.1875,
        0.21875,
        0.25,
        0.3125,
        0.375,
        0.4375,
        0.5,
        0.625,
        0.75,
        0.875,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        10.0,
        12.0,
        14.0,
        16.0,
        20.0,
        24.0,
        28.0,
        32.0,
        40.0,
        48.0,
        56.0,
        64.0,
        80.0,
        96.0,
        112.0,
        128.0,
        160.0,
        192.0,
        224.0,
        256.0,
        320.0,
        384.0,
        448.0,
        512.0,
        640.0,
        768.0,
        896.0,
        1024.0,
        1280.0,
        1536.0,
        1792.0,
        2048.0,
        2560.0,
        3072.0,
        3584.0,
        4096.0,
        5120.0,
        6144.0,
        7168.0,
        8192.0,
        10240.0,
        12288.0,
        14336.0,
        16384.0,
        20480.0,
        24576.0,
        28672.0,
        32768.0,
        40960.0,
        49152.0,
        57344.0,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        -0.0,
        -1.52587890625e-05,
        -3.0517578125e-05,
        -4.57763671875e-05,
        -6.103515625e-05,
        -7.62939453125e-05,
        -9.1552734375e-05,
        -0.0001068115234375,
        -0.0001220703125,
        -0.000152587890625,
        -0.00018310546875,
        -0.000213623046875,
        -0.000244140625,
        -0.00030517578125,
        -0.0003662109375,
        -0.00042724609375,
        -0.00048828125,
        -0.0006103515625,
        -0.000732421875,
        -0.0008544921875,
        -0.0009765625,
        -0.001220703125,
        -0.00146484375,
        -0.001708984375,
        -0.001953125,
        -0.00244140625,
        -0.0029296875,
        -0.00341796875,
        -0.00390625,
        -0.0048828125,
        -0.005859375,
        -0.0068359375,
        -0.0078125,
        -0.009765625,
        -0.01171875,
        -0.013671875,
        -0.015625,
        -0.01953125,
        -0.0234375,
        -0.02734375,
        -0.03125,
        -0.0390625,
        -0.046875,
        -0.0546875,
        -0.0625,
        -0.078125,
        -0.09375,
        -0.109375,
        -0.125,
        -0.15625,
        -0.1875,
        -0.21875,
        -0.25,
        -0.3125,
        -0.375,
        -0.4375,
        -0.5,
        -0.625,
        -0.75,
        -0.875,
        -1.0,
        -1.25,
        -1.5,
        -1.75,
        -2.0,
        -2.5,
        -3.0,
        -3.5,
        -4.0,
        -5.0,
        -6.0,
        -7.0,
        -8.0,
        -10.0,
        -12.0,
        -14.0,
        -16.0,
        -20.0,
        -24.0,
        -28.0,
        -32.0,
        -40.0,
        -48.0,
        -56.0,
        -64.0,
        -80.0,
        -96.0,
        -112.0,
        -128.0,
        -160.0,
        -192.0,
        -224.0,
        -256.0,
        -320.0,
        -384.0,
        -448.0,
        -512.0,
        -640.0,
        -768.0,
        -896.0,
        -1024.0,
        -1280.0,
        -1536.0,
        -1792.0,
        -2048.0,
        -2560.0,
        -3072.0,
        -3584.0,
        -4096.0,
        -5120.0,
        -6144.0,
        -7168.0,
        -8192.0,
        -10240.0,
        -12288.0,
        -14336.0,
        -16384.0,
        -20480.0,
        -24576.0,
        -28672.0,
        -32768.0,
        -40960.0,
        -49152.0,
        -57344.0,
        -1.0f * std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),

    };

    return e4m3fnuz_lut[input];
}

TEST_CASE(test_fp8_cast_to_float)
{
    std::vector<uint8_t> bit_vals(256);
    std::iota(bit_vals.begin(), bit_vals.end(), 0);
    EXPECT(bool{std::all_of(bit_vals.begin(), bit_vals.end(), [](uint8_t bit_val) {
        migraphx::fp8::fp8e5m2 fp8_val(bit_val, migraphx::fp8::fp8e5m2::from_bits());
        if(std::isnan(float(fp8_val)) and std::isnan(fp8e5m2_to_fp32_value(bit_val)))
        {
            return true;
        }
        else if(std::isinf(float(fp8_val)) and std::isinf(fp8e5m2_to_fp32_value(bit_val)))
        {
            return true;
        }
        return migraphx::float_equal(float(fp8_val), fp8e5m2_to_fp32_value(bit_val));
    })});
}

TEST_CASE(test_fp8_cast_from_float)
{
    std::unordered_map<float, uint8_t> test_vals = {
        {-60000, 0xfb},
        {-57344, 0xfb},
        {-448, 0xdf},
        {-256, 0xdc},
        {-240, 0xdc},
        {-200, 0xda},
        {-20, 0xcd},
        {-2, 0xc0},
        {-1, 0xbc},
        {-0.5, 0xb8},
        {-0.2, 0xb2},
        {-0.1111, 0xaf},
        {-0.111, 0xaf},
        {-0.11, 0xaf},
        {-0.1, 0xae},
        {6.10351e-05, 0x4},
        {-6.10351e-05, 0x84},
        {3.05176e-05, 0x2},
        {-3.05176e-05, 0x82},
        {1.52588e-05, 0x1},
        {-1.52588e-05, 0x81},
        {7.62939e-06, 0x0},
        {-7.62939e-06, 0x80},
        {0.1, 0x2e},
        {0.11, 0x2f},
        {0.111, 0x2f},
        {0.1111, 0x2f},
        {0.2, 0x32},
        {0.5, 0x38},
        {1, 0x3c},
        {2, 0x40},
        {20, 0x4d},
        {200, 0x5a},
        {240, 0x5c},
        {256, 0x5c},
        {448, 0x5f},
        {57344, 0x7b},
        {60000, 0x7b},
        {1e+07, 0x7b},
    };

    EXPECT(bool{std::all_of(test_vals.begin(), test_vals.end(), [](const auto sample) {
        return migraphx::float_equal(
            migraphx::fp8::fp8e5m2(sample.first),
            migraphx::fp8::fp8e5m2(sample.second, migraphx::fp8::fp8e5m2::from_bits()));
    })});
}

TEST_CASE(test_positive_zero)
{
    float zero = 0.0;
    migraphx::fp8::fp8e5m2 fp8_zero(zero);
    EXPECT(fp8_zero.is_zero());
    EXPECT(migraphx::float_equal(zero, float(fp8_zero)));
}

TEST_CASE(test_negative_zero)
{
    float nzero = -0.0;
    migraphx::fp8::fp8e5m2 fp8_nzero(nzero);
    EXPECT(fp8_nzero.is_zero());
    //  negative zero is preserved for fp8e5m2
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
    migraphx::fp8::fp8e5m2 fp8_nan(fnan);
    EXPECT(fp8_nan.is_nan());
    EXPECT(std::isnan(fp8_nan));
}

TEST_CASE(test_nan_2)
{
    auto fnan = std::numeric_limits<migraphx::fp8::fp8e5m2>::quiet_NaN();
    migraphx::fp8::fp8e5m2 fp8_nan(fnan.data, migraphx::fp8::fp8e5m2::from_bits());
    EXPECT(fp8_nan.is_nan());
    EXPECT(std::isnan(fp8_nan));
    EXPECT(std::isnan(float(fp8_nan)));
}

TEST_CASE(test_bool)
{
    float zero  = 0.0;
    float two   = 2.0;
    float other = -0.375;
    migraphx::fp8::fp8e5m2 fp8_zero(zero);
    migraphx::fp8::fp8e5m2 fp8_two(two);
    migraphx::fp8::fp8e5m2 fp8_other(other);
    EXPECT(not static_cast<bool>(fp8_zero));
    EXPECT(static_cast<bool>(fp8_two));
    EXPECT(static_cast<bool>(fp8_other));
}

TEST_CASE(test_infinity_1)
{
    // float infinity should get clipped to max
    float finf = std::numeric_limits<float>::infinity();
    migraphx::fp8::fp8e5m2 fp8_max(finf);
    EXPECT(fp8_max == std::numeric_limits<migraphx::fp8::fp8e5m2>::max());
}

TEST_CASE(test_infinity_2)
{
    // neg inf
    float finf = -1.0 * std::numeric_limits<float>::infinity();
    // no inf in fp8e5m2, it gets clipped to lowest
    migraphx::fp8::fp8e5m2 fp8_lowest(finf);
    EXPECT(bool{fp8_lowest == std::numeric_limits<migraphx::fp8::fp8e5m2>::lowest()});
}

TEST_CASE(test_numeric_max_1)
{
    float fmax = std::numeric_limits<float>::max();
    migraphx::fp8::fp8e5m2 fp8_max(fmax);
    EXPECT(fp8_max == std::numeric_limits<migraphx::fp8::fp8e5m2>::max());
}

TEST_CASE(test_numeric_max_2)
{
    // gets clipped to max
    float fmax = 2 * std::numeric_limits<migraphx::fp8::fp8e5m2>::max();
    migraphx::fp8::fp8e5m2 fp8_max(fmax);
    EXPECT(fp8_max == std::numeric_limits<migraphx::fp8::fp8e5m2>::max());
}

TEST_CASE(test_numeric_lowest_1)
{
    float flowest = std::numeric_limits<float>::lowest();
    migraphx::fp8::fp8e5m2 fp8_lowest(flowest);
    EXPECT(fp8_lowest == std::numeric_limits<migraphx::fp8::fp8e5m2>::lowest());
}

TEST_CASE(test_numeric_lowest_2)
{
    // gets clipped to lowest
    float fmin = 2.0 * std::numeric_limits<migraphx::fp8::fp8e5m2>::lowest();
    migraphx::fp8::fp8e5m2 fp8_lowest(fmin);
    EXPECT(fp8_lowest == std::numeric_limits<migraphx::fp8::fp8e5m2>::lowest());
}

TEST_CASE(test_max_eq_lowest)
{
    EXPECT(migraphx::float_equal(std::numeric_limits<migraphx::fp8::fp8e5m2>::lowest(),
                                 -1 * std::numeric_limits<migraphx::fp8::fp8e5m2>::max()));
}

TEST_CASE(test_isfinite)
{
    EXPECT(std::isfinite(migraphx::fp8::fp8e5m2(0.0)));
    EXPECT(std::isfinite(migraphx::fp8::fp8e5m2(-0.0)));
    EXPECT(not std::isfinite(
        migraphx::fp8::fp8e5m2(std::numeric_limits<migraphx::fp8::fp8e5m2>::quiet_NaN())));
    EXPECT(not std::isfinite(std::numeric_limits<migraphx::fp8::fp8e5m2>::infinity()));
    // -1.0 * inf  is float(-inf) which with clipping/saturation gets converted into fp8::lowest()
    EXPECT(std::isfinite(
        migraphx::fp8::fp8e5m2(-1.0 * std::numeric_limits<migraphx::fp8::fp8e5m2>::infinity())));
    EXPECT(not std::isfinite(migraphx::fp8::fp8e5m2(0xFC, migraphx::fp8::fp8e5m2::from_bits())));
}

TEST_CASE(test_binary_ops)
{
    auto a = migraphx::fp8::fp8e5m2(-1.0);
    auto b = migraphx::fp8::fp8e5m2(1.0);
    auto c = migraphx::fp8::fp8e5m2(0.0);
    auto d = migraphx::fp8::fp8e5m2(-0.0);
    EXPECT(migraphx::float_equal((c + d), c));
    EXPECT(migraphx::float_equal((c + d), d));
    EXPECT(migraphx::float_equal((a + b), c));
    EXPECT(migraphx::float_equal((a + b), d));

    auto e = migraphx::fp8::fp8e5m2(10.0);
    auto f = migraphx::fp8::fp8e5m2(-10.0);
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
    auto a = migraphx::fp8::fp8e5m2(-1.0);
    auto b = migraphx::fp8::fp8e5m2(1.0);
    EXPECT(migraphx::float_equal(b, migraphx::fp8::fabs(a)));
}

TEST_CASE(test_stream_op)
{
    auto a = migraphx::fp8::fp8e5m2(-1.0);
    std::stringstream ss;
    ss << a;
    EXPECT(std::string("-1") == ss.str());
    ss     = std::stringstream();
    auto b = std::numeric_limits<migraphx::fp8::fp8e5m2>::quiet_NaN();
    ss << b;
    EXPECT(std::string("nan") == ss.str());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
