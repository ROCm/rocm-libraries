/* ************************************************************************
 * Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HIP_MATRIX_DENSE_HPP_
#define ROCALUTION_HIP_MATRIX_DENSE_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace rocalution
{

    template <typename ValueType>
    class HIPAcceleratorMatrixDENSE : public HIPAcceleratorMatrix<ValueType>
    {
    public:
        HIPAcceleratorMatrixDENSE();
        explicit HIPAcceleratorMatrixDENSE(const Rocalution_Backend_Descriptor& local_backend);
        virtual ~HIPAcceleratorMatrixDENSE();

        virtual void         Info(void) const;
        virtual unsigned int GetMatFormat(void) const
        {
            return DENSE;
        }

        virtual void Clear(void);
        virtual void AllocateDENSE(int nrow, int ncol);
        virtual void SetDataPtrDENSE(ValueType** val, int nrow, int ncol);
        virtual void LeaveDataPtrDENSE(ValueType** val);

        virtual bool ConvertFrom(const BaseMatrix<ValueType>& mat);

        virtual void CopyFrom(const BaseMatrix<ValueType>& src);
        virtual void CopyFromAsync(const BaseMatrix<ValueType>& src);
        virtual void CopyTo(BaseMatrix<ValueType>* dst) const;
        virtual void CopyToAsync(BaseMatrix<ValueType>* dst) const;

        virtual void CopyFromHost(const HostMatrix<ValueType>& src);
        virtual void CopyFromHostAsync(const HostMatrix<ValueType>& src);
        virtual void CopyToHost(HostMatrix<ValueType>* dst) const;
        virtual void CopyToHostAsync(HostMatrix<ValueType>* dst) const;

        virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
        virtual void ApplyAdd(const BaseVector<ValueType>& in,
                              ValueType                    scalar,
                              BaseVector<ValueType>*       out) const;

        virtual bool MatMatMult(const BaseMatrix<ValueType>& A, const BaseMatrix<ValueType>& B);

        virtual bool ReplaceColumnVector(int idx, const BaseVector<ValueType>& vec);
        virtual bool ReplaceRowVector(int idx, const BaseVector<ValueType>& vec);
        virtual bool ExtractColumnVector(int idx, BaseVector<ValueType>* vec) const;
        virtual bool ExtractRowVector(int idx, BaseVector<ValueType>* vec) const;

    private:
        MatrixDENSE<ValueType> mat_;

        friend class HIPAcceleratorMatrixCSR<ValueType>;

        friend class BaseVector<ValueType>;
        friend class AcceleratorVector<ValueType>;
        friend class HIPAcceleratorVector<ValueType>;
    };

} // namespace rocalution

#endif // ROCALUTION_HIP_MATRIX_DENSE_HPP_
