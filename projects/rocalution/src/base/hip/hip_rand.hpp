/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HIP_RAND_HPP_
#define ROCALUTION_HIP_RAND_HPP_

namespace rocalution
{
    template <typename IMPL>
    struct CRTP_HIPRand_Traits;

    //
    // Base class of the random number functionality.
    //
    template <typename IMPL>
    class CRTP_HIPRand
    {

    protected:
        inline CRTP_HIPRand(){};
        using traits_t = CRTP_HIPRand_Traits<IMPL>;

    public:
        inline CRTP_HIPRand(const CRTP_HIPRand& that) = delete;
        using data_t                                  = typename traits_t::data_t;

        inline void Generate(data_t* data, size_t size)
        {
            static_cast<IMPL&>(*this).Generate(data, size);
        };
    };

} // namespace rocalution

#endif // ROCALUTION_BASE_VECTOR_HPP_
