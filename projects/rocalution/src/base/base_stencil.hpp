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

#ifndef ROCALUTION_BASE_STENCIL_HPP_
#define ROCALUTION_BASE_STENCIL_HPP_

#include "base_rocalution.hpp"

namespace rocalution
{

    template <typename ValueType>
    class BaseVector;
    template <typename ValueType>
    class HostVector;
    template <typename ValueType>
    class HIPAcceleratorVector;

    template <typename ValueType>
    class HostStencilLaplace2D;
    template <typename ValueType>
    class HIPAcceleratorStencil;
    template <typename ValueType>
    class HIPAcceleratorStencilLaplace2D;

    /// Base class for all host/accelerator stencils
    template <typename ValueType>
    class BaseStencil
    {
    public:
        BaseStencil();
        virtual ~BaseStencil();

        /** \brief Return the number of rows in the stencil */
        int GetM(void) const;
        /** \brief Return the number of columns in the stencil */
        int GetN(void) const;
        /** \brief Return the dimension of the stencil */
        int GetNDim(void) const;
        /** \brief Return the nnz per row */
        virtual int64_t GetNnz(void) const = 0;

        /** \brief Shows simple info about the object */
        virtual void Info(void) const = 0;
        /** \brief Return the stencil format id (see stencil_formats.hpp) */
        virtual unsigned int GetStencilId(void) const = 0;
        /** \brief Copy the backend descriptor information */
        virtual void set_backend(const Rocalution_Backend_Descriptor& local_backend);
        /** \brief Set the grid size */
        virtual void SetGrid(int size);

        /** \brief Apply the stencil to vector, out = this*in; */
        virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const = 0;
        /** \brief Apply and add the stencil to vector, out = out + scalar*this*in; */
        virtual void ApplyAdd(const BaseVector<ValueType>& in,
                              ValueType                    scalar,
                              BaseVector<ValueType>*       out) const = 0;

    protected:
        /** \brief Number of rows */
        int ndim_;
        /** \brief Number of columns */
        int size_;

        /** \brief Backend descriptor (local copy) */
        Rocalution_Backend_Descriptor local_backend_;

        friend class BaseVector<ValueType>;
        friend class HostVector<ValueType>;
        friend class AcceleratorVector<ValueType>;
        friend class HIPAcceleratorVector<ValueType>;
    };

    template <typename ValueType>
    class HostStencil : public BaseStencil<ValueType>
    {
    public:
        HostStencil();
        virtual ~HostStencil();
    };

    template <typename ValueType>
    class AcceleratorStencil : public BaseStencil<ValueType>
    {
    public:
        AcceleratorStencil();
        virtual ~AcceleratorStencil();

        /** \brief Copy (accelerator stencil) from host stencil */
        virtual void CopyFromHost(const HostStencil<ValueType>& src) = 0;

        /** \brief Copy (accelerator stencil) to host stencil */
        virtual void CopyToHost(HostStencil<ValueType>* dst) const = 0;
    };

    template <typename ValueType>
    class HIPAcceleratorStencil : public AcceleratorStencil<ValueType>
    {
    public:
        HIPAcceleratorStencil();
        virtual ~HIPAcceleratorStencil();
    };

} // namespace rocalution

#endif // ROCALUTION_BASE_STENCIL_HPP_
