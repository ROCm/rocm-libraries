/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HOST_MATRIX_DENSE_HPP_
#define ROCALUTION_HOST_MATRIX_DENSE_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace rocalution
{

    template <typename ValueType>
    class HostMatrixDENSE : public HostMatrix<ValueType>
    {
    public:
        HostMatrixDENSE();
        explicit HostMatrixDENSE(const Rocalution_Backend_Descriptor& local_backend);
        virtual ~HostMatrixDENSE();

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

        virtual void CopyFrom(const BaseMatrix<ValueType>& mat);
        virtual void CopyTo(BaseMatrix<ValueType>* mat) const;

        virtual bool ReadFileRSIO(const std::string& filename);
        virtual bool WriteFileRSIO(const std::string& filename) const;

        virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
        virtual void ApplyAdd(const BaseVector<ValueType>& in,
                              ValueType                    scalar,
                              BaseVector<ValueType>*       out) const;

        virtual bool MatMatMult(const BaseMatrix<ValueType>& A, const BaseMatrix<ValueType>& B);

        virtual bool Householder(int idx, ValueType& beta, BaseVector<ValueType>* vec) const;
        virtual bool QRDecompose(void);
        virtual bool QRSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

        virtual bool LUFactorize(void);
        virtual bool LUSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

        virtual bool Invert(void);

        virtual bool ReplaceColumnVector(int idx, const BaseVector<ValueType>& vec);
        virtual bool ReplaceRowVector(int idx, const BaseVector<ValueType>& vec);
        virtual bool ExtractColumnVector(int idx, BaseVector<ValueType>* vec) const;
        virtual bool ExtractRowVector(int idx, BaseVector<ValueType>* vec) const;

    private:
        MatrixDENSE<ValueType> mat_;

        friend class BaseVector<ValueType>;
        friend class HostVector<ValueType>;
        friend class HostMatrixCSR<ValueType>;
        friend class HostMatrixCOO<ValueType>;
        friend class HostMatrixELL<ValueType>;
        friend class HostMatrixHYB<ValueType>;
        friend class HostMatrixDIA<ValueType>;

        friend class HIPAcceleratorMatrixDENSE<ValueType>;
    };

} // namespace rocalution

#endif // ROCALUTION_HOST_MATRIX_DENSE_HPP_
