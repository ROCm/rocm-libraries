// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../shared/accuracy_test.h"
#include "../../shared/gpubuf.h"
#include "../../shared/rocfft_params.h"
#include "../samples/rocfft/examplekernels.h"
#include "../samples/rocfft/exampleutils.h"
#include "rocfft/rocfft.h"
#include <functional>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <random>
#include <thread>
#include <vector>

void run_1D_hermitian_test(size_t length)
{
    // Run two 1D C2R transforms, on:
    // * random input
    // * identical random input, but modified to be Hermitian-symmetric
    // We should tolerate the input being having non-zero imaginary part in the DC mode
    // and the Nyquist frequency (of the length is even).

    rocfft_params p;
    p.length         = {length};
    p.precision      = fft_precision_double;
    p.transform_type = fft_transform_type_real_inverse;
    p.placement      = fft_placement_notinplace;
    p.validate();

    if(verbose)
    {
        std::cout << p.str("\n\t") << std::endl;
    }

    ASSERT_TRUE(p.valid(verbose));

    std::vector<hipDoubleComplex> h_input(p.isize[0]);

    std::random_device                     rd;
    std::mt19937                           gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for(auto& val : h_input)
    {
        val.x = dis(gen);
        val.y = dis(gen);
    }

    if(verbose > 2)
    {
        std::cout << "non-Hermitian input:";
        for(const auto& val : h_input)
        {
            std::cout << " "
                      << "(" << val.x << ", " << val.y << ")";
        }
        std::cout << std::endl;
    }

    gpubuf ibuf;
    ASSERT_TRUE(ibuf.alloc(p.ibuffer_sizes()[0]) == hipSuccess);
    ASSERT_TRUE(hipMemcpy(ibuf.data(), h_input.data(), ibuf.size(), hipMemcpyHostToDevice)
                == hipSuccess);

    gpubuf obuf;
    ASSERT_TRUE(obuf.alloc(p.obuffer_sizes()[0]) == hipSuccess);

    ASSERT_TRUE(p.create_plan() == fft_status_success);

    std::vector<void*> pibuf = {ibuf.data()};
    std::vector<void*> pobuf = {obuf.data()};
    ASSERT_TRUE(p.execute(pibuf.data(), pobuf.data()) == fft_status_success);

    std::vector<double> h_output(p.osize[0]);
    ASSERT_TRUE(hipMemcpy(h_output.data(), obuf.data(), obuf.size(), hipMemcpyDeviceToHost)
                == hipSuccess);

    ASSERT_TRUE(hipDeviceSynchronize() == hipSuccess);

    if(verbose > 2)
    {
        std::cout << "output:";
        for(const auto& val : h_output)
        {
            std::cout << " " << val;
        }
        std::cout << std::endl;
    }

    std::vector<hipDoubleComplex> h_input1(p.isize[0]);
    std::copy(h_input.begin(), h_input.end(), h_input1.begin());

    // Impose Hermitian symmetry on the input:
    h_input1[0].y = 0.0;

    if(p.length[0] % 2 == 0)
    {
        h_input1.back().y = 0.0;
    }
    if(verbose > 2)
    {
        std::cout << "Hermitian input:";
        for(const auto& val : h_input1)
        {
            std::cout << " "
                      << "(" << val.x << ", " << val.y << ")";
        }
        std::cout << std::endl;
    }

    double maxdiff = 0.0;
    for(unsigned int i = 0; i < h_input.size(); ++i)
    {
        auto val = std::abs(
            rocfft_complex<double>(h_input[i].x - h_input1[i].x, h_input[i].y - h_input1[i].y));
        if(val > maxdiff)
            maxdiff = val;
    }
    ASSERT_TRUE(maxdiff > 0.0);

    ASSERT_TRUE(hipMemcpy(ibuf.data(), h_input1.data(), ibuf.size(), hipMemcpyHostToDevice)
                == hipSuccess);
    ASSERT_TRUE(p.execute(pibuf.data(), pobuf.data()) == fft_status_success);
    std::vector<double> h_output1(p.osize[0]);
    ASSERT_TRUE(hipMemcpy(h_output1.data(), obuf.data(), obuf.size(), hipMemcpyDeviceToHost)
                == hipSuccess);

    if(verbose > 2)
    {
        std::cout << "output:";
        for(const auto& val : h_output1)
        {
            std::cout << " " << val;
        }
        std::cout << std::endl;
    }

    double maxerr = 0;
    for(unsigned int i = 0; i < h_output.size(); ++i)
    {
        auto val = std::abs(h_output[i] - h_output1[i]);
        if(val > maxerr)
            maxerr = val;
    }

    if(verbose)
        std::cout << maxerr << std::endl;

    EXPECT_TRUE(maxerr == 0.0);
}

// test a case that's small enough that it only needs one kernel
TEST(rocfft_UnitTest, 1D_hermitian_single_small)
{
    run_1D_hermitian_test(8);
}

// test a case that's big enough that it needs multiple kernels
TEST(rocfft_UnitTest, 1D_hermitian_single_large)
{
    run_1D_hermitian_test(8192);
}

template <typename T>
std::string str(T begin, T end)
{
    std::stringstream ss;
    bool              first = true;
    for(; begin != end; begin++)
    {
        if(!first)
            ss << ", ";
        ss << *begin;
        first = false;
    }
    return ss.str();
}

// Test that the GPU Hermitian symmetrizer code produces the correct results.
TEST(rocfft_UnitTest, gpu_symmetrizer)
{
    std::vector<std::vector<size_t>> lengths = {{4, 4, 3},
                                                {5},
                                                {8},
                                                {5, 5},
                                                {5, 8},
                                                {8, 5},
                                                {8, 8},
                                                {5, 5, 5},
                                                {8, 5, 5},
                                                {5, 8, 5},
                                                {5, 5, 8},
                                                {5, 8, 8},
                                                {8, 5, 8},
                                                {8, 8, 5},
                                                {8, 8, 8}};

    for(const auto& length : lengths)
    {
        // Symmetrize complex data and ensure that the checker sees that it's symmetric.

        // Use the params class to set up strides and lengths:
        rocfft_params p;
        p.length         = length;
        p.precision      = fft_precision_double;
        p.transform_type = fft_transform_type_real_inverse;
        p.placement      = fft_placement_notinplace;
        p.validate();
        if(verbose)
        {
            std::cout << "\t" << p.str("\n\t") << std::endl;
        }
        ASSERT_TRUE(p.valid(verbose));

        // Data buffers:
        gpubuf buf;
        ASSERT_TRUE(buf.alloc(sizeof(hipDoubleComplex) * p.isize[0]) == hipSuccess);
        std::vector<hipDoubleComplex> hbuf(p.isize[0]);

        // Initialize a Hermitian-symmetric array; it should be symmetric.
        init_hermitiancomplex_cm(p.length_cm(), p.ilength_cm(), p.istride_cm(), buf.data());
        ASSERT_TRUE(hipMemcpy(hbuf.data(), buf.data(), buf.size(), hipMemcpyDeviceToHost)
                    == hipSuccess);
        if(verbose > 1)
        {
            printbuffer_cm(hbuf, p.ilength_cm(), p.istride_cm(), p.nbatch, p.idist);
        }
        EXPECT_TRUE(
            check_symmetry_cm(hbuf, p.length_cm(), p.istride_cm(), p.nbatch, p.idist, verbose > 0))
            << "length: " << str(length.begin(), length.end());

        // This should not be symmetric:
        std::mt19937_64 rng;
        std::seed_seq   ss{uint32_t(10)};
        rng.seed(ss);
        std::uniform_real_distribution<double> unif(0, 1);
        for(auto& v : hbuf)
        {
            v.x = unif(rng);
            v.y = unif(rng);
        }
        if(verbose > 2)
        {
            printbuffer_cm(hbuf, p.ilength_cm(), p.istride_cm(), p.nbatch, p.idist);
        }
        EXPECT_TRUE(
            !check_symmetry_cm(hbuf, p.length_cm(), p.istride_cm(), p.nbatch, p.idist, false))
            << "length: " << str(length.begin(), length.end());
    }

    for(const auto& length : lengths)
    {
        // Generate Hermitian-symmetric data and ensure that applying the symmetrizer has no effect.

        rocfft_params p;
        p.length         = length;
        p.precision      = fft_precision_double;
        p.transform_type = fft_transform_type_real_forward;
        p.placement      = fft_placement_notinplace;
        p.validate();
        if(verbose)
        {
            std::cout << "\t" << p.str("\n\t") << std::endl;
        }
        ASSERT_TRUE(p.valid(verbose));
        ASSERT_TRUE(p.create_plan() == fft_status_success);

        gpubuf ibuf, obuf;
        ASSERT_TRUE(ibuf.alloc(p.ibuffer_sizes()[0]) == hipSuccess);
        ASSERT_TRUE(obuf.alloc(p.obuffer_sizes()[0]) == hipSuccess);

        initreal_cm(p.length_cm(), p.istride_cm(), ibuf.data());

        std::vector<void*> pibuf = {ibuf.data()};
        std::vector<void*> pobuf = {obuf.data()};

        ASSERT_TRUE(p.execute(pibuf.data(), pobuf.data()) == fft_status_success);

        std::vector<hipDoubleComplex> h_output(p.osize[0]);
        std::fill(h_output.begin(), h_output.end(), hipDoubleComplex{0.0, 0.0});

        ASSERT_TRUE(
            hipMemcpy(h_output.data(), obuf.data(), p.obuffer_sizes()[0], hipMemcpyDeviceToHost)
            == hipSuccess);

        impose_hermitian_symmetry_cm(p.length_cm(), p.olength_cm(), p.ostride_cm(), obuf.data());

        std::vector<hipDoubleComplex> h_output_resym(p.osize[0]);
        std::fill(h_output_resym.begin(), h_output_resym.end(), hipDoubleComplex{0.0, 0.0});

        ASSERT_TRUE(
            hipMemcpy(
                h_output_resym.data(), obuf.data(), p.obuffer_sizes()[0], hipMemcpyDeviceToHost)
            == hipSuccess);

        double maxdiff = 0;
        for(unsigned int i = 0; i < h_output.size(); ++i)
        {
            auto rdiff = std::abs(h_output[i].x - h_output_resym[i].x);
            auto idiff = std::abs(h_output[i].y - h_output_resym[i].y);
            maxdiff    = std::max({maxdiff, rdiff, idiff});
        }

        if(verbose)
        {
            std::cout << "maxdiff: " << maxdiff << std::endl;
        }

        if(verbose > 2)
        {
            std::cout << "before symmetrization:\n";
            printbuffer_cm(h_output, p.olength_cm(), p.ostride_cm(), p.nbatch, p.odist);
            std::cout << "after symmetrization:\n";
            printbuffer_cm(h_output_resym, p.olength_cm(), p.ostride_cm(), p.nbatch, p.odist);
        }

        EXPECT_TRUE(maxdiff < 1e-13) << maxdiff << "\n" << p.str() << "\n";
    }
}
