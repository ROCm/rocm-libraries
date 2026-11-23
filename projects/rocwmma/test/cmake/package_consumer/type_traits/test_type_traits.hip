/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <cstdio>

template <typename T>
constexpr bool owned_traits()
{
    return rocwmma::is_arithmetic<T>::value && rocwmma::is_arithmetic_v<T>
           && rocwmma::is_floating_point<T>::value && rocwmma::is_floating_point_v<T>
           && rocwmma::is_signed<T>::value && rocwmma::is_signed_v<T>;
}

static_assert(owned_traits<rocwmma::float8_t>());
static_assert(owned_traits<rocwmma::bfloat8_t>());
static_assert(owned_traits<rocwmma::float8_fnuz_t>());
static_assert(owned_traits<rocwmma::bfloat8_fnuz_t>());
static_assert(owned_traits<rocwmma::xfloat32_t>());
static_assert(rocwmma::is_signed_v<const int>);
static_assert(!rocwmma::is_signed_v<unsigned int>);
static_assert(!rocwmma::is_floating_point_v<float&>);
static_assert(!rocwmma::is_arithmetic_v<void*>);

__global__ void traits_gemm(const rocwmma::float16_t* a, const rocwmma::float16_t* b, float* c)
{
    using namespace rocwmma;
    fragment<matrix_a, 16, 16, 16, float16_t, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, float16_t, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float>             c_frag;
    load_matrix_sync(a_frag, a, 16);
    load_matrix_sync(b_frag, b, 16);
    fill_fragment(c_frag, 0.0f);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(c, c_frag, 16, mem_row_major);
}

int main(int argc, char**)
{
    if(argc == 1)
    {
        std::printf("Owned trait checks passed; legacy setting = %d\n",
                    ROCWMMA_LEGACY_STD_TRAIT_SPECIALIZATIONS);
        return 0;
    }

    int  count  = 0;
    auto status = hipGetDeviceCount(&count);
    if(status == hipErrorNoDevice || status == hipErrorInsufficientDriver
       || (status == hipSuccess && count == 0))
    {
        std::puts("No ROCm device available for the default-mode GEMM");
        return 77;
    }
    if(status != hipSuccess)
    {
        std::fprintf(stderr, "hipGetDeviceCount: %s\n", hipGetErrorString(status));
        return 1;
    }
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
        return 1;

    // Managed storage keeps allocation/cleanup identical on every error path.
    rocwmma::float16_t* a  = nullptr;
    rocwmma::float16_t* b  = nullptr;
    float*              c  = nullptr;
    bool                ok = hipMallocManaged(&a, 256 * sizeof(*a)) == hipSuccess
              && hipMallocManaged(&b, 256 * sizeof(*b)) == hipSuccess
              && hipMallocManaged(&c, 256 * sizeof(*c)) == hipSuccess;
    if(ok)
    {
        for(int i = 0; i < 256; ++i)
        {
            a[i] = b[i] = static_cast<rocwmma::float16_t>(1.0f);
            c[i]        = 0.0f;
        }
        traits_gemm<<<1, props.warpSize>>>(a, b, c);
        ok = hipGetLastError() == hipSuccess && hipDeviceSynchronize() == hipSuccess;
        for(int i = 0; ok && i < 256; ++i)
            ok = c[i] == 16.0f;
    }
    if(c)
        ok = hipFree(c) == hipSuccess && ok;
    if(b)
        ok = hipFree(b) == hipSuccess && ok;
    if(a)
        ok = hipFree(a) == hipSuccess && ok;
    return ok ? 0 : 1;
}
