/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#pragma once

#include <vector>

namespace matxu
{
    template <
        typename T_,
        typename I_,
        /* typename = typename std::enable_if<std::is_arithmetic<std::decay_t<T_>>::value>::type, */
        typename = typename std::enable_if<std::is_integral<std::decay_t<I_>>::value
                                           && std::is_signed<std::decay_t<I_>>::value>::type>
    class MatrixInterface
    {
    public:
        using T = T_;
        using I = I_;
        using S = decltype(std::real(T{}));

        virtual ~MatrixInterface() = default;

        virtual T const* data() const
        {
            return nullptr;
        }

        virtual T* data()
        {
            return nullptr;
        }

        [[maybe_unused]] virtual T* copy_to(T*) const
        {
            return nullptr;
        }

        [[maybe_unused]] virtual bool copy_to(std::vector<T>&) const
        {
            return false;
        }

        [[maybe_unused]] virtual bool copy_data_from(const MatrixInterface<T_, I_>& source)
        {
            return false;
        }

        [[maybe_unused]] virtual bool set_data_from(const MatrixInterface<T_, I_>& source)
        {
            return false;
        }

        virtual void set_to_zero() = 0;

        virtual I nrows() const = 0;

        virtual I ncols() const = 0;

        virtual I ld() const = 0;

        virtual I size() const = 0;

        virtual I num_bytes() const = 0;

        virtual bool empty() const
        {
            return true;
        }

        virtual bool reshape(I /* nrows */, I /* ncols */)
        {
            return false;
        }

        virtual T operator()(I, I) const = 0;

        virtual T& operator()(I, I) = 0;

        virtual T operator[](I) const = 0;

        virtual T& operator[](I) = 0;

        virtual S norm() const = 0;

        virtual S max_coeff_norm() const = 0;

        virtual S max_col_norm() const = 0;

    protected:
    };

} // namespace matxu
