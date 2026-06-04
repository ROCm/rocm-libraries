// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Parameterized GoogleTest for the rocFFT device-side, header-only FFT library
// (<rocfft/device/fft.hpp>).  Each block performs one single-block transform;
// correctness is validated across the full supported (length, precision) matrix
// using analytic cases (impulse, DC, single-bin sinusoid), a host O(N^2) DFT
// reference, and a forward/inverse round-trip.
//
// This target deliberately depends only on HIP and GoogleTest: it does not link
// roc::rocfft, FFTW, OpenMP, or SQLite.

#include <rocfft/device/fft.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

using rocfft::device::Direction;
using rocfft::device::FFT;
using rocfft::device::Precision;
using rocfft::device::precision_traits;

namespace
{

    constexpr double PI = 3.14159265358979323846;

    // Kernel: a single thread block performs one in-place FFT.  Each thread loads
    // elements_per_thread complex values into registers, cooperates on the
    // transform via shared memory, then writes the result back.
    template <typename FFTType>
    __global__ void fft_kernel(typename FFTType::complex_type* data)
    {
        constexpr unsigned ept  = FFTType::elements_per_thread;
        constexpr unsigned smem = FFTType::shared_memory_size;

        __shared__ char shared_mem[smem];

        typename FFTType::complex_type thread_data[ept];
        for(unsigned i = 0; i < ept; ++i)
            thread_data[i] = data[threadIdx.x + i * FFTType::workgroup_size];

        FFTType::execute(thread_data, shared_mem, threadIdx.x);

        for(unsigned i = 0; i < ept; ++i)
            data[threadIdx.x + i * FFTType::workgroup_size] = thread_data[i];
    }

    // Run one in-place transform on the device for the given compile-time config.
    template <unsigned Length, Precision P, Direction D>
    void execute_on_device(std::vector<typename precision_traits<P>::complex_type>& data)
    {
        using FFT_t   = FFT<Length, P, D>;
        using complex = typename FFT_t::complex_type;

        const size_t bytes = Length * sizeof(complex);

        complex* d_data = nullptr;
        ASSERT_EQ(hipMalloc(&d_data, bytes), hipSuccess);
        ASSERT_EQ(hipMemcpy(d_data, data.data(), bytes, hipMemcpyHostToDevice), hipSuccess);

        hipLaunchKernelGGL(HIP_KERNEL_NAME(fft_kernel<FFT_t>),
                           dim3(1),
                           dim3(FFT_t::workgroup_size),
                           0, // sharedMemBytes (kernel uses static __shared__)
                           0, // stream
                           d_data);
        ASSERT_EQ(hipGetLastError(), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        ASSERT_EQ(hipMemcpy(data.data(), d_data, bytes, hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipFree(d_data), hipSuccess);
    }

    // Naive O(N^2) DFT computed in double precision.  sign = -1 for the forward
    // transform (matching rocFFT's convention), +1 for the inverse.
    std::vector<std::complex<double>> naive_dft(const std::vector<std::complex<double>>& in,
                                                int                                      sign)
    {
        const size_t                      N = in.size();
        std::vector<std::complex<double>> out(N);
        for(size_t j = 0; j < N; ++j)
        {
            std::complex<double> acc(0.0, 0.0);
            for(size_t n = 0; n < N; ++n)
            {
                const double angle = sign * 2.0 * PI * double(j) * double(n) / double(N);
                acc += in[n] * std::complex<double>(std::cos(angle), std::sin(angle));
            }
            out[j] = acc;
        }
        return out;
    }

    // Relative tolerance: errors accumulate in double, single-precision transforms
    // are far looser than double-precision ones.
    template <Precision P>
    constexpr double rel_tol()
    {
        return P == Precision::Single ? 3e-3 : 1e-9;
    }

    // Compare a device result against a double-precision reference using a relative
    // max-norm error.
    template <typename Complex>
    void expect_close(const std::vector<Complex>&              got,
                      const std::vector<std::complex<double>>& ref,
                      double                                   tol,
                      const char*                              what)
    {
        ASSERT_EQ(got.size(), ref.size());

        double max_ref = 0.0;
        for(const auto& r : ref)
            max_ref = std::max(max_ref, std::abs(r));
        if(max_ref == 0.0)
            max_ref = 1.0;

        double max_err = 0.0;
        for(size_t i = 0; i < got.size(); ++i)
        {
            const std::complex<double> g(static_cast<double>(got[i].x),
                                         static_cast<double>(got[i].y));
            max_err = std::max(max_err, std::abs(g - ref[i]));
        }

        EXPECT_LT(max_err / max_ref, tol)
            << what << ": relative error too high (max_err=" << max_err << ", max_ref=" << max_ref
            << ")";
    }

    // ---- analytic and reference-based test bodies ----

    // Impulse at index 0 -> flat unit-magnitude spectrum.
    template <unsigned Length, Precision P>
    void run_impulse()
    {
        using complex = typename precision_traits<P>::complex_type;
        using real    = typename precision_traits<P>::real_type;

        std::vector<complex> data(Length, complex(real(0), real(0)));
        data[0] = complex(real(1), real(0));

        execute_on_device<Length, P, Direction::Forward>(data);

        std::vector<std::complex<double>> ref(Length, std::complex<double>(1.0, 0.0));
        expect_close(data, ref, rel_tol<P>(), "Impulse");
    }

    // All-ones input -> energy only in bin 0 (= N).
    template <unsigned Length, Precision P>
    void run_dc()
    {
        using complex = typename precision_traits<P>::complex_type;
        using real    = typename precision_traits<P>::real_type;

        std::vector<complex> data(Length, complex(real(1), real(0)));

        execute_on_device<Length, P, Direction::Forward>(data);

        std::vector<std::complex<double>> ref(Length, std::complex<double>(0.0, 0.0));
        ref[0] = std::complex<double>(double(Length), 0.0);
        expect_close(data, ref, rel_tol<P>(), "DC");
    }

    // Complex exponential at frequency k -> energy only in bin k (= N).
    template <unsigned Length, Precision P>
    void run_single_bin_sinusoid()
    {
        using complex = typename precision_traits<P>::complex_type;
        using real    = typename precision_traits<P>::real_type;

        constexpr unsigned k = 3; // < min supported length (32)

        std::vector<complex> data(Length);
        for(unsigned n = 0; n < Length; ++n)
        {
            const double angle = 2.0 * PI * double(k) * double(n) / double(Length);
            data[n]            = complex(real(std::cos(angle)), real(std::sin(angle)));
        }

        execute_on_device<Length, P, Direction::Forward>(data);

        std::vector<std::complex<double>> ref(Length, std::complex<double>(0.0, 0.0));
        ref[k] = std::complex<double>(double(Length), 0.0);
        expect_close(data, ref, rel_tol<P>(), "SingleBinSinusoid");
    }

    // Random input compared against a host double-precision DFT.
    template <unsigned Length, Precision P>
    void run_random_vs_naive_dft()
    {
        using complex = typename precision_traits<P>::complex_type;
        using real    = typename precision_traits<P>::real_type;

        std::mt19937                           gen(12345u + Length + unsigned(P));
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        std::vector<complex>              data(Length);
        std::vector<std::complex<double>> in(Length);
        for(unsigned n = 0; n < Length; ++n)
        {
            const double re = dist(gen);
            const double im = dist(gen);
            in[n]           = std::complex<double>(re, im);
            data[n]         = complex(real(re), real(im));
        }

        const auto ref = naive_dft(in, -1);

        execute_on_device<Length, P, Direction::Forward>(data);

        expect_close(data, ref, rel_tol<P>(), "RandomVsNaiveDFT");
    }

    // Forward then inverse recovers the original input scaled by N.
    template <unsigned Length, Precision P>
    void run_roundtrip()
    {
        using complex = typename precision_traits<P>::complex_type;
        using real    = typename precision_traits<P>::real_type;

        std::mt19937                           gen(98765u + Length + unsigned(P));
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        std::vector<complex>              data(Length);
        std::vector<std::complex<double>> ref(Length);
        for(unsigned n = 0; n < Length; ++n)
        {
            const double re = dist(gen);
            const double im = dist(gen);
            data[n]         = complex(real(re), real(im));
            ref[n]          = std::complex<double>(re * double(Length), im * double(Length));
        }

        execute_on_device<Length, P, Direction::Forward>(data);
        if(::testing::Test::HasFatalFailure())
            return;
        execute_on_device<Length, P, Direction::Inverse>(data);

        expect_close(data, ref, rel_tol<P>(), "RoundTrip");
    }

    // ---- parameterization over the supported matrix ----

    struct DeviceFFTConfig
    {
        unsigned  length;
        Precision prec;
    };

    // Dispatch a runtime (length, precision) config to the matching compile-time
    // FFT<...> instantiation by invoking the supplied callable's templated
    // operator()<Length, Precision>().
    template <typename F>
    void dispatch_config(const DeviceFFTConfig& cfg, F f)
    {
#define ROCFFT_DEVICE_TEST_DISPATCH_CASE(LEN)                \
    case LEN:                                                \
        if(cfg.prec == Precision::Single)                    \
            f.template operator()<LEN, Precision::Single>(); \
        else                                                 \
            f.template operator()<LEN, Precision::Double>(); \
        break

        switch(cfg.length)
        {
            ROCFFT_DEVICE_TEST_DISPATCH_CASE(32);
            ROCFFT_DEVICE_TEST_DISPATCH_CASE(64);
            ROCFFT_DEVICE_TEST_DISPATCH_CASE(128);
            ROCFFT_DEVICE_TEST_DISPATCH_CASE(256);
            ROCFFT_DEVICE_TEST_DISPATCH_CASE(512);
        default:
            FAIL() << "unsupported transform length " << cfg.length;
        }

#undef ROCFFT_DEVICE_TEST_DISPATCH_CASE
    }

    std::string config_name(const ::testing::TestParamInfo<DeviceFFTConfig>& info)
    {
        return std::to_string(info.param.length) + "_"
               + (info.param.prec == Precision::Single ? "single" : "double");
    }

    class DeviceFFTTest : public ::testing::TestWithParam<DeviceFFTConfig>
    {
    protected:
        void SetUp() override
        {
            int count = 0;
            if(hipGetDeviceCount(&count) != hipSuccess || count == 0)
                GTEST_SKIP() << "No HIP device available";
        }
    };

    TEST_P(DeviceFFTTest, Impulse)
    {
        dispatch_config(GetParam(), []<unsigned L, Precision P>() { run_impulse<L, P>(); });
    }

    TEST_P(DeviceFFTTest, DC)
    {
        dispatch_config(GetParam(), []<unsigned L, Precision P>() { run_dc<L, P>(); });
    }

    TEST_P(DeviceFFTTest, SingleBinSinusoid)
    {
        dispatch_config(GetParam(),
                        []<unsigned L, Precision P>() { run_single_bin_sinusoid<L, P>(); });
    }

    TEST_P(DeviceFFTTest, RandomVsNaiveDFT)
    {
        dispatch_config(GetParam(),
                        []<unsigned L, Precision P>() { run_random_vs_naive_dft<L, P>(); });
    }

    TEST_P(DeviceFFTTest, RoundTrip)
    {
        dispatch_config(GetParam(), []<unsigned L, Precision P>() { run_roundtrip<L, P>(); });
    }

    INSTANTIATE_TEST_SUITE_P(Supported,
                             DeviceFFTTest,
                             ::testing::Values(DeviceFFTConfig{32, Precision::Single},
                                               DeviceFFTConfig{32, Precision::Double},
                                               DeviceFFTConfig{64, Precision::Single},
                                               DeviceFFTConfig{64, Precision::Double},
                                               DeviceFFTConfig{128, Precision::Single},
                                               DeviceFFTConfig{128, Precision::Double},
                                               DeviceFFTConfig{256, Precision::Single},
                                               DeviceFFTConfig{256, Precision::Double},
                                               DeviceFFTConfig{512, Precision::Single},
                                               DeviceFFTConfig{512, Precision::Double}),
                             config_name);

} // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
