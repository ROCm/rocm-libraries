// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "../../shared/fft_hash.h"
#include "../../shared/params_gen.h"
#include "../../shared/rocfft_params.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <vector>

static void set_params(const fft_precision precision, fft_params& param)
{
    std::vector<size_t> blengths = {131072};

    std::vector<size_t> unit_strides = {1};

    size_t nbatch = 1;

    std::vector<size_t> zero_offsets = {0};

    auto btype = fft_array_type::fft_array_type_complex_interleaved;

    param.length    = blengths;
    param.istride   = unit_strides;
    param.ostride   = unit_strides;
    param.nbatch    = nbatch;
    param.precision = precision;

    param.idist = blengths[0];
    param.odist = blengths[0];

    param.isize = {blengths[0]};
    param.osize = {blengths[0]};

    param.itype = btype;
    param.otype = btype;

    param.ioffset = zero_offsets;
    param.ooffset = zero_offsets;

    param.placement = fft_placement_inplace;
}

// Create an fft params struct for a contiguous input/output buffer.
// Purpose of the unit tests here is only to test the hashing strategy,
// i.e., to reduce multiple floating point values to a single 64 bit
// identifier. The strategy for hashing a non-contiguous buffer is
// essentially the same, only the data access pattern is changed.
static void validate_buffer_params(const fft_params& param)
{
    ASSERT_EQ(param.length.size() == 1, true);

    ASSERT_EQ(param.istride.size() == 1, true);
    ASSERT_EQ(param.istride[0] == 1, true);

    ASSERT_EQ(param.ostride.size() == 1, true);
    ASSERT_EQ(param.ostride[0] == 1, true);

    ASSERT_EQ(param.ioffset.size() == 1, true);
    ASSERT_EQ(param.ioffset[0] == 0, true);

    ASSERT_EQ(param.ooffset.size() == 1, true);
    ASSERT_EQ(param.ooffset[0] == 0, true);

    ASSERT_EQ(param.isize.size() == 1, true);
    ASSERT_EQ(param.isize[0] == param.length[0], true);

    ASSERT_EQ(param.osize.size() == 1, true);
    ASSERT_EQ(param.osize[0] == param.length[0], true);

    ASSERT_EQ(param.nbatch == 1, true);

    ASSERT_EQ(param.itype == fft_array_type_complex_interleaved, true);
    ASSERT_EQ(param.otype == fft_array_type_complex_interleaved, true);

    ASSERT_EQ(param.placement == fft_placement_inplace, true);
}

static unsigned int gen_seed()
{
    auto seed = static_cast<unsigned int>(time(NULL));

    return seed;
}

template <typename Tfloat>
static void shuffle_buffer(const size_t N, const size_t seed, std::vector<hostbuf>& buffer)
{
    auto idata = (rocfft_complex<Tfloat>*)buffer[0].data();

    std::random_device rd;
    std::mt19937       g(rd());

    std::shuffle(idata, idata + N, g);
}

static void shuffle_buffer(const fft_params& param, const size_t seed, std::vector<hostbuf>& buffer)
{
    validate_buffer_params(param);

    auto N = param.length[0];

    switch(param.precision)
    {
    case fft_precision_half:
        shuffle_buffer<rocfft_fp16>(N, seed, buffer);
        break;
    case fft_precision_double:
        shuffle_buffer<double>(N, seed, buffer);
        break;
    case fft_precision_single:
        shuffle_buffer<float>(N, seed, buffer);
        break;
    default:
        abort();
    }
}

template <typename Tfloat>
static void corrupt_buffer_single(const size_t N, const size_t seed, std::vector<hostbuf>& buffer)
{
    auto idata = (rocfft_complex<Tfloat>*)buffer[0].data();

    std::minstd_rand                       gen(seed);
    std::uniform_real_distribution<double> dist1(0.0f, 1.0f);
    std::uniform_real_distribution<double> dist2(-1.0f, 1.0f);

    auto random_id = static_cast<size_t>(dist1(gen) * static_cast<double>(N - 1));

    auto real = idata[random_id].real();
    auto imag = idata[random_id].imag();

    idata[random_id].real(real + dist2(gen));
    idata[random_id].imag(imag + dist2(gen));
}

static void
    corrupt_buffer_single(const fft_params& param, const size_t seed, std::vector<hostbuf>& buffer)
{
    validate_buffer_params(param);

    auto N = param.length[0];

    switch(param.precision)
    {
    case fft_precision_half:
        corrupt_buffer_single<rocfft_fp16>(N, seed, buffer);
        break;
    case fft_precision_double:
        corrupt_buffer_single<double>(N, seed, buffer);
        break;
    case fft_precision_single:
        corrupt_buffer_single<float>(N, seed, buffer);
        break;
    default:
        abort();
    }
}

template <typename Tfloat>
static void corrupt_buffer_full(const size_t N, const size_t seed, std::vector<hostbuf>& buffer)
{
    auto idata = (rocfft_complex<Tfloat>*)buffer[0].data();

    std::minstd_rand                       gen(seed);
    std::uniform_real_distribution<double> dist(-1.0f, 1.0f);

    for(size_t i = 0; i < N; i++)
    {
        auto real = idata[i].real();
        auto imag = idata[i].imag();

        idata[i].real(real + dist(gen));
        idata[i].imag(imag + dist(gen));
    }
}

static void
    corrupt_buffer_full(const fft_params& param, const size_t seed, std::vector<hostbuf>& buffer)
{
    validate_buffer_params(param);

    auto N = param.length[0];

    switch(param.precision)
    {
    case fft_precision_half:
        corrupt_buffer_full<rocfft_fp16>(N, seed, buffer);
        break;
    case fft_precision_double:
        corrupt_buffer_full<double>(N, seed, buffer);
        break;
    case fft_precision_single:
        corrupt_buffer_full<float>(N, seed, buffer);
        break;
    default:
        abort();
    }
}

template <typename Tfloat>
static void init_buffer(const size_t N, const size_t seed, std::vector<hostbuf>& buffer)
{
    auto idata = (rocfft_complex<Tfloat>*)buffer[0].data();

    std::minstd_rand                       gen(seed);
    std::uniform_real_distribution<double> dist(-1.0f, 1.0f);

    for(size_t i = 0; i < N; i++)
    {
        idata[i].real(dist(gen));
        idata[i].imag(dist(gen));
    }
}

static void init_buffer(const fft_params& params, const size_t seed, std::vector<hostbuf>& buffer)
{
    validate_buffer_params(params);

    auto N = params.length[0];

    switch(params.precision)
    {
    case fft_precision_half:
        init_buffer<rocfft_fp16>(N, seed, buffer);
        break;
    case fft_precision_double:
        init_buffer<double>(N, seed, buffer);
        break;
    case fft_precision_single:
        init_buffer<float>(N, seed, buffer);
        break;
    default:
        abort();
    }
}

static void run_test(const rocfft_params& params)
{
    auto hash_in    = hash_input(rocfft_precision_from_fftparams(params.precision),
                              params.ilength(),
                              params.istride,
                              params.idist,
                              rocfft_array_type_from_fftparams(params.itype),
                              params.nbatch);
    auto hash_out_1 = hash_output<size_t>();
    auto hash_out_2 = hash_output<size_t>();

    auto seed = gen_seed();

    std::vector<hostbuf> buffer1, buffer2;
    buffer1 = allocate_host_buffer(params.precision, params.itype, params.ibuffer_sizes());
    buffer2 = allocate_host_buffer(params.precision, params.itype, params.ibuffer_sizes());

    init_buffer(params, seed, buffer1);
    compute_hash(buffer1, hash_in, hash_out_1);

    copy_buffers(buffer1,
                 buffer2,
                 params.ilength(),
                 params.nbatch,
                 params.precision,
                 params.itype,
                 params.istride,
                 params.idist,
                 params.itype,
                 params.istride,
                 params.idist,
                 params.ioffset,
                 params.ioffset);
    compute_hash(buffer2, hash_in, hash_out_2);
    ASSERT_EQ(hash_out_1.buffer_real == hash_out_2.buffer_real, true)
        << "random seed: " << seed << std::endl;

    ASSERT_EQ(hash_out_1.buffer_imag == hash_out_2.buffer_imag, true)
        << "random seed: " << seed << std::endl;

    copy_buffers(buffer1,
                 buffer2,
                 params.ilength(),
                 params.nbatch,
                 params.precision,
                 params.itype,
                 params.istride,
                 params.idist,
                 params.itype,
                 params.istride,
                 params.idist,
                 params.ioffset,
                 params.ioffset);
    corrupt_buffer_full(params, seed, buffer2);
    compute_hash(buffer2, hash_in, hash_out_2);
    ASSERT_EQ(hash_out_1.buffer_real != hash_out_2.buffer_real, true)
        << "random seed: " << seed << std::endl;
    ASSERT_EQ(hash_out_1.buffer_imag != hash_out_2.buffer_imag, true)
        << "random seed: " << seed << std::endl;

    copy_buffers(buffer1,
                 buffer2,
                 params.ilength(),
                 params.nbatch,
                 params.precision,
                 params.itype,
                 params.istride,
                 params.idist,
                 params.itype,
                 params.istride,
                 params.idist,
                 params.ioffset,
                 params.ioffset);
    corrupt_buffer_single(params, seed, buffer2);
    compute_hash(buffer2, hash_in, hash_out_2);
    ASSERT_EQ(hash_out_1.buffer_real != hash_out_2.buffer_real, true)
        << "random seed: " << seed << std::endl;
    ASSERT_EQ(hash_out_1.buffer_imag != hash_out_2.buffer_imag, true)
        << "random seed: " << seed << std::endl;

    copy_buffers(buffer1,
                 buffer2,
                 params.ilength(),
                 params.nbatch,
                 params.precision,
                 params.itype,
                 params.istride,
                 params.idist,
                 params.itype,
                 params.istride,
                 params.idist,
                 params.ioffset,
                 params.ioffset);
    shuffle_buffer(params, seed, buffer2);
    compute_hash(buffer2, hash_in, hash_out_2);
    ASSERT_EQ(hash_out_1.buffer_real != hash_out_2.buffer_real, true)
        << "random seed: " << seed << std::endl;
    ASSERT_EQ(hash_out_1.buffer_imag != hash_out_2.buffer_imag, true)
        << "random seed: " << seed << std::endl;
}

TEST(rocfft_UnitTest, buffer_hashing_half)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    rocfft_params params;
    set_params(fft_precision_half, params);

    try
    {
        run_test(params);
    }
    catch(const HOSTBUF_MEM_USAGE& e)
    {
        GTEST_SKIP() << e.what();
    }
    catch(const DEVICEBUF_MEM_USAGE& e)
    {
        GTEST_SKIP() << e.what();
    }
}

TEST(rocfft_UnitTest, buffer_hashing_single)
{

    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    rocfft_params params;
    set_params(fft_precision_single, params);

    try
    {
        run_test(params);
    }
    catch(const HOSTBUF_MEM_USAGE& e)
    {
        GTEST_SKIP() << e.what();
    }
    catch(const DEVICEBUF_MEM_USAGE& e)
    {
        GTEST_SKIP() << e.what();
    }
}

TEST(rocfft_UnitTest, buffer_hashing_double)
{

    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    rocfft_params params;
    set_params(fft_precision_double, params);

    try
    {
        run_test(params);
    }
    catch(const HOSTBUF_MEM_USAGE& e)
    {
        GTEST_SKIP() << e.what();
    }
    catch(const DEVICEBUF_MEM_USAGE& e)
    {
        GTEST_SKIP() << e.what();
    }
}

// Hash a host buffer laid out exactly as given (strides/dist describe memory).
template <typename Telem>
static hash_output<size_t> hash_host_buffer(rocfft_precision           precision,
                                            rocfft_array_type          type,
                                            const std::vector<Telem>&  mem,
                                            const std::vector<size_t>& length,
                                            const std::vector<size_t>& stride,
                                            size_t                     dist,
                                            size_t                     nbatch)
{
    std::vector<hostbuf> buf(1);
    buf[0].alloc(mem.size() * sizeof(Telem));
    std::copy(mem.begin(), mem.end(), static_cast<Telem*>(buf[0].data()));
    auto hash_in  = hash_input(precision, length, stride, dist, type, nbatch);
    auto hash_out = hash_output<size_t>();
    compute_hash(buf, hash_in, hash_out);
    return hash_out;
}

template <typename Tfloat>
static hash_output<size_t> hash_real_1234(rocfft_precision precision)
{
    std::vector<Tfloat> mem;
    for(size_t i = 0; i < 4; ++i)
        mem.push_back(static_cast<Tfloat>(i + 1)); // {1, 2, 3, 4}
    return hash_host_buffer(precision, rocfft_array_type_real, mem, {4}, {1}, 4, 1);
}

// The buffer hash must be a fixed function of the data (no std::hash), so a
// known buffer always produces the same fingerprint on every toolchain.
TEST(rocfft_UnitTest, buffer_hash_golden_values)
{
    PROB_SKIP_UNITTEST();
    auto h_half = hash_real_1234<rocfft_fp16>(rocfft_precision_half);
    EXPECT_EQ(h_half.buffer_real, 6010172260665940913ULL);
    EXPECT_EQ(h_half.buffer_imag, 0u);

    auto h_single = hash_real_1234<float>(rocfft_precision_single);
    EXPECT_EQ(h_single.buffer_real, 13029437891923012638ULL);
    EXPECT_EQ(h_single.buffer_imag, 0u);

    auto h_double = hash_real_1234<double>(rocfft_precision_double);
    EXPECT_EQ(h_double.buffer_real, 4918654574207189942ULL);
    EXPECT_EQ(h_double.buffer_imag, 0u);
}

template <typename Tfloat>
static void expect_real_transpose_detected(rocfft_precision precision)
{
    std::vector<Tfloat> a(16), t(16);
    for(size_t r = 0; r < 4; ++r)
        for(size_t c = 0; c < 4; ++c)
            a[r * 4 + c] = static_cast<Tfloat>(r * 4 + c + 1);
    for(size_t r = 0; r < 4; ++r)
        for(size_t c = 0; c < 4; ++c)
            t[r * 4 + c] = a[c * 4 + r];
    auto ha = hash_host_buffer(precision, rocfft_array_type_real, a, {4, 4}, {4, 1}, 16, 1);
    auto ht = hash_host_buffer(precision, rocfft_array_type_real, t, {4, 4}, {4, 1}, 16, 1);
    EXPECT_NE(ha.buffer_real, ht.buffer_real);
}

// Elements on an anti-diagonal must not share a position, so a transposed
// square output hashes differently.
TEST(rocfft_UnitTest, buffer_hash_detects_transpose)
{
    PROB_SKIP_UNITTEST();
    expect_real_transpose_detected<float>(rocfft_precision_single);
    expect_real_transpose_detected<double>(rocfft_precision_double);

    std::vector<rocfft_complex<float>> a(16), t(16);
    for(size_t r = 0; r < 4; ++r)
        for(size_t c = 0; c < 4; ++c)
        {
            const float v = static_cast<float>(r * 4 + c + 1);
            a[r * 4 + c]  = rocfft_complex<float>(v, -v);
        }
    for(size_t r = 0; r < 4; ++r)
        for(size_t c = 0; c < 4; ++c)
            t[r * 4 + c] = a[c * 4 + r];
    const auto type = rocfft_array_type_complex_interleaved;
    auto       ha   = hash_host_buffer(rocfft_precision_single, type, a, {4, 4}, {4, 1}, 16, 1);
    auto       ht   = hash_host_buffer(rocfft_precision_single, type, t, {4, 4}, {4, 1}, 16, 1);
    EXPECT_NE(ha.buffer_real, ht.buffer_real);
    EXPECT_NE(ha.buffer_imag, ht.buffer_imag);
}

// A 1-ULP change in any element must change the hash.
TEST(rocfft_UnitTest, buffer_hash_detects_one_ulp)
{
    PROB_SKIP_UNITTEST();
    std::vector<float> a = {1.0f, 2.0f, 1.7f, 4.0f};
    std::vector<float> b = a;
    b[2]                 = std::nextafter(1.7f, 2.0f);
    auto ha = hash_host_buffer(rocfft_precision_single, rocfft_array_type_real, a, {4}, {1}, 4, 1);
    auto hb = hash_host_buffer(rocfft_precision_single, rocfft_array_type_real, b, {4}, {1}, 4, 1);
    EXPECT_NE(ha.buffer_real, hb.buffer_real);
}

// Swapping two batches must change the hash.
TEST(rocfft_UnitTest, buffer_hash_detects_batch_swap)
{
    PROB_SKIP_UNITTEST();
    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> b = {5, 6, 7, 8, 1, 2, 3, 4};
    auto ha = hash_host_buffer(rocfft_precision_single, rocfft_array_type_real, a, {4}, {1}, 4, 2);
    auto hb = hash_host_buffer(rocfft_precision_single, rocfft_array_type_real, b, {4}, {1}, 4, 2);
    EXPECT_NE(ha.buffer_real, hb.buffer_real);
}

// The hash depends on logical data only: strides, dist and padding contents
// must not change it.
TEST(rocfft_UnitTest, buffer_hash_ignores_padding)
{
    PROB_SKIP_UNITTEST();
    const size_t       nbatch = 2, rows = 3, cols = 4;
    std::vector<float> contig(nbatch * rows * cols);
    std::vector<float> padded(nbatch * 17, -999.0f);
    for(size_t b = 0; b < nbatch; ++b)
        for(size_t r = 0; r < rows; ++r)
            for(size_t c = 0; c < cols; ++c)
            {
                const float v              = static_cast<float>(b * 100 + r * 10 + c);
                contig[b * 12 + r * 4 + c] = v;
                padded[b * 17 + r * 5 + c] = v;
            }
    const auto prec = rocfft_precision_single;
    const auto type = rocfft_array_type_real;
    auto       hc   = hash_host_buffer(prec, type, contig, {rows, cols}, {4, 1}, 12, nbatch);
    auto       hp   = hash_host_buffer(prec, type, padded, {rows, cols}, {5, 1}, 17, nbatch);
    EXPECT_EQ(hc.buffer_real, hp.buffer_real);
}
