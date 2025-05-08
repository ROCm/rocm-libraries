/* **************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <vector>

namespace matxu
{
template <typename T_,
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
