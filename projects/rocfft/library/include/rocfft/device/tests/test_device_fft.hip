// Standalone HIP test for the rocFFT device-side header-only library.
// Validates FFT<Length, Precision, Direction>::execute() for all supported
// configurations by computing forward FFT of a known input and checking
// the output against NumPy-equivalent expected values.
//
// Build:
//   hipcc -std=c++17 -I<rocfft>/library/include test_device_fft.hip -o test_device_fft
//
// Run:
//   ./test_device_fft

#include <rocfft/device/fft.hpp>

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(expr)                                                         \
    do {                                                                         \
        hipError_t err = (expr);                                                 \
        if(err != hipSuccess) {                                                  \
            fprintf(stderr, "HIP error %d at %s:%d: %s\n",                      \
                    err, __FILE__, __LINE__, hipGetErrorString(err));             \
            exit(1);                                                             \
        }                                                                        \
    } while(0)

// Kernel: each block performs one FFT.
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

template <typename Real>
struct TestResult
{
    bool   pass;
    double max_err;
};

// Run a forward FFT of all-ones input.  Expected output: [N, 0, 0, ..., 0].
template <unsigned Length, rocfft::device::Precision P>
TestResult<typename rocfft::device::precision_traits<P>::real_type>
test_fft_ones()
{
    using namespace rocfft::device;
    using FFTFwd  = FFT<Length, P, Direction::Forward>;
    using FFTInv  = FFT<Length, P, Direction::Inverse>;
    using complex = typename FFTFwd::complex_type;
    using real    = typename FFTFwd::real_type;

    std::vector<complex> h_data(Length);
    for(unsigned i = 0; i < Length; ++i)
        h_data[i] = complex(real(1), real(0));

    complex* d_data = nullptr;
    HIP_CHECK(hipMalloc(&d_data, Length * sizeof(complex)));
    HIP_CHECK(hipMemcpy(d_data, h_data.data(), Length * sizeof(complex), hipMemcpyHostToDevice));

    fft_kernel<FFTFwd><<<1, FFTFwd::workgroup_size>>>(d_data);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<complex> h_fwd(Length);
    HIP_CHECK(hipMemcpy(h_fwd.data(), d_data, Length * sizeof(complex), hipMemcpyDeviceToHost));

    // Verify forward: h_fwd[0] should be (N, 0), rest should be (0, 0).
    double max_err = 0.0;
    double tol     = std::is_same<real, float>::value ? 1e-4 : 1e-10;

    max_err = std::max(max_err, std::abs(static_cast<double>(h_fwd[0].x) - static_cast<double>(Length)));
    max_err = std::max(max_err, std::abs(static_cast<double>(h_fwd[0].y)));
    for(unsigned i = 1; i < Length; ++i)
    {
        max_err = std::max(max_err, std::abs(static_cast<double>(h_fwd[i].x)));
        max_err = std::max(max_err, std::abs(static_cast<double>(h_fwd[i].y)));
    }

    // Now test round-trip: inverse FFT should recover original (scaled by N).
    fft_kernel<FFTInv><<<1, FFTInv::workgroup_size>>>(d_data);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<complex> h_inv(Length);
    HIP_CHECK(hipMemcpy(h_inv.data(), d_data, Length * sizeof(complex), hipMemcpyDeviceToHost));

    for(unsigned i = 0; i < Length; ++i)
    {
        double expected_real = static_cast<double>(Length);
        max_err = std::max(max_err, std::abs(static_cast<double>(h_inv[i].x) - expected_real));
        max_err = std::max(max_err, std::abs(static_cast<double>(h_inv[i].y)));
    }

    HIP_CHECK(hipFree(d_data));

    return {max_err < tol, max_err};
}

template <unsigned Length, rocfft::device::Precision P>
void run_test(const char* label)
{
    auto result = test_fft_ones<Length, P>();
    printf("  %-40s %s  (max_err=%.2e)\n",
           label,
           result.pass ? "PASS" : "FAIL",
           result.max_err);
    if(!result.pass)
        exit(1);
}

int main()
{
    printf("rocFFT device-side library tests:\n\n");

    using namespace rocfft::device;

    // Configurations matching the validated Warp FFT matrix.
    run_test<32,  Precision::Double>("FFT<32,  Double>");
    run_test<64,  Precision::Single>("FFT<64,  Single>");
    run_test<64,  Precision::Double>("FFT<64,  Double>");
    run_test<128, Precision::Double>("FFT<128, Double>");
    run_test<256, Precision::Single>("FFT<256, Single>");
    run_test<256, Precision::Double>("FFT<256, Double>");
    run_test<512, Precision::Single>("FFT<512, Single>");
    run_test<512, Precision::Double>("FFT<512, Double>");

    printf("\nAll tests passed.\n");
    return 0;
}
