// -*- C++ -*-

// Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <hip/thread>
#include <thrust/unique_ptr.h>
#include <thrust/copy.h>
#include <vector>
#include <thread>

#define N 0x10000000U
#define NUM_ITERATIONS 512

int main() {
    // Create and initialize vectors on the host
    std::vector<float> x_host(N, 1.0F);
    std::vector<float> y_host(N, 2.0F);
    const float alpha = 2.0F;

    // Allocate device memory and copy data to GPU
    thrust::unique_ptr<float[]> x = thrust::make_unique<float[]>(N);
    thrust::unique_ptr<float[]> y = thrust::make_unique<float[]>(N);
    thrust::copy(x_host.begin(), x_host.end(), x.get());
    thrust::copy(y_host.begin(), y_host.end(), y.get());

    decltype(std::chrono::steady_clock::now()) start;
    { // hip::wthread scope starts here
        std::vector<hip::wthread> threads(hip::wthread::hardware_concurrency());

        // Make sure any async work is done before we 'start the clock'
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        start = std::chrono::steady_clock::now();
        for (unsigned int i = 0; i < threads.size(); ++i) {
            size_t chunk_size = (i < N % threads.size()) ? (N / threads.size() + 1) : (N / threads.size());
            size_t offset =     (i < N % threads.size()) ? (i * chunk_size)         : (i * chunk_size + N % threads.size());
            
            threads[i] = hip::wthread(hip::wthread::max_width(),
                [] __device__(uint32_t n, float a, const float *x, float *y) {
                    for (uint32_t i = hip::this_thread::get_fiber_id(); i < n; i += hip::this_thread::get_width()) {
                        float t = x[i];
                        for (int j = 0; j < NUM_ITERATIONS; ++j) {
                            t = a * t + y[i];
                        }
                        y[i] = t;
                    }
                },
                chunk_size, alpha, x.get_raw() + offset, y.get_raw() + offset);
        }
        
        for (auto &t : threads) {
            t.join();
        }
    } // hip::wthread scope ends here
    auto finished = std::chrono::steady_clock::now();
    
    // Copy results back from GPU to host if necessary
    thrust::copy(y.get(), y.get() + N, y_host.begin());
    
    // Check results:
    float t = 1.0F;
    for (int j = 0; j < NUM_ITERATIONS; ++j) {
        t = alpha * t + 2.0F;
    }
    for (uint32_t i = 0; i < N; ++i) {
        if (std::abs(y_host[i] - t) > 0.0000001F) {
            std::cerr << "Error: y_host[" << i << "] = " << y_host[i] << " (Expected " << t << ").\n";
            return 1;
        }
    }
    std::chrono::nanoseconds body = finished - start;
    std::cout.imbue(std::locale(""));
    std::cout << "Time to run saxpy(" << N << ") = " << body.count() << "ns\n";
    return 0;
}
