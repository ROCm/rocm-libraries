/* ************************************************************************
 * Copyright (C) 2020-2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HIP_RAND_UNIFORM_HPP_
#define ROCALUTION_HIP_RAND_UNIFORM_HPP_

#include "hip_rand.hpp"
#include <hip/hip_runtime.h>
#include <rocrand.hpp>

namespace rocalution
{
    //
    // Implementation of a uniform distribution using rocRAND.
    //
    template <typename T>
    class HIPRandUniform_rocRAND;

    template <typename T>
    struct CRTP_HIPRand_Traits<HIPRandUniform_rocRAND<T>>
    {
        using data_t = T;
    };

    template <typename T>
    class HIPRandUniform_rocRAND : public CRTP_HIPRand<HIPRandUniform_rocRAND<T>>
    {
    protected:
        using value_type = typename numeric_traits<T>::value_type;
        value_type                                         m_a, m_b;
        rocrand_cpp::mtgp32_engine<0UL>                    m_engine;
        rocrand_cpp::uniform_real_distribution<value_type> m_distribution;
        const Rocalution_Backend_Descriptor*               m_backend;

    public:
        inline HIPRandUniform_rocRAND(unsigned long long                   seed,
                                      value_type                           a,
                                      value_type                           b,
                                      const Rocalution_Backend_Descriptor* backend)
            : m_a(a)
            , m_b(b)
            , m_engine(seed)
            , m_backend(backend){};

        inline void Generate(T* data, size_t size)
        {
            if(size > 0)
            {
                assert(0 == sizeof(T) % sizeof(value_type));
                const int n = sizeof(T) / sizeof(value_type);

                //
                // Get the uniform distribution between 0 and 1.
                //
                this->m_distribution(this->m_engine, ((value_type*)data), size * n);

                //
                // Apply the affine transformation.
                //
                if((this->m_a != static_cast<value_type>(0))
                   || (this->m_b != static_cast<value_type>(1)))
                {
                    dim3 BlockSize(m_backend->HIP_block_size);
                    dim3 GridSize((size * n) / m_backend->HIP_block_size + 1);

                    kernel_affine_transform<<<GridSize,
                                              BlockSize,
                                              0,
                                              HIPSTREAM(
                                                  _get_backend_descriptor()->HIP_stream_current)>>>(
                        size * n, this->m_a, this->m_b, (value_type*)data);

                    CHECK_HIP_ERROR(__FILE__, __LINE__);
                }
            }
        };
    };

} // namespace rocalution

#endif // ROCALUTION_BASE_VECTOR_HPP_
