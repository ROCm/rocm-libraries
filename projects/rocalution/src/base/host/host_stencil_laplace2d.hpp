/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HOST_STENCIL_LAPLACE2D_HPP_
#define ROCALUTION_HOST_STENCIL_LAPLACE2D_HPP_

#include "../base_stencil.hpp"
#include "../base_vector.hpp"
#include "../stencil_types.hpp"

namespace rocalution
{

    template <typename ValueType>
    class HostStencilLaplace2D : public HostStencil<ValueType>
    {
    public:
        HostStencilLaplace2D();
        explicit HostStencilLaplace2D(const Rocalution_Backend_Descriptor& local_backend);
        virtual ~HostStencilLaplace2D();

        virtual int64_t      GetNnz(void) const;
        virtual void         Info(void) const;
        virtual unsigned int GetStencilId(void) const
        {
            return Laplace2D;
        }

        virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
        virtual void ApplyAdd(const BaseVector<ValueType>& in,
                              ValueType                    scalar,
                              BaseVector<ValueType>*       out) const;

    private:
        friend class BaseVector<ValueType>;
        friend class HostVector<ValueType>;

        // friend class HIPAcceleratorStencilLaplace2D<ValueType>;
    };

} // namespace rocalution

#endif // ROCALUTION_HOST_STENCIL_LAPLACE2D_HPP_
