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

#ifndef ROCALUTION_HIP_MATRIX_COO_HPP_
#define ROCALUTION_HIP_MATRIX_COO_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

#include <rocsparse/rocsparse.h>

namespace rocalution
{

    template <typename ValueType>
    class HIPAcceleratorMatrixCOO : public HIPAcceleratorMatrix<ValueType>
    {
    public:
        HIPAcceleratorMatrixCOO();
        explicit HIPAcceleratorMatrixCOO(const Rocalution_Backend_Descriptor& local_backend);
        virtual ~HIPAcceleratorMatrixCOO();

        virtual void         Info(void) const;
        virtual unsigned int GetMatFormat(void) const
        {
            return COO;
        }

        virtual void Clear(void);
        virtual void AllocateCOO(int64_t nnz, int nrow, int ncol);

        virtual void
            SetDataPtrCOO(int** row, int** col, ValueType** val, int64_t nnz, int nrow, int ncol);
        virtual void LeaveDataPtrCOO(int** row, int** col, ValueType** val);

        virtual bool ConvertFrom(const BaseMatrix<ValueType>& mat);

        virtual bool Permute(const BaseVector<int>& permutation);
        virtual bool PermuteBackward(const BaseVector<int>& permutation);

        virtual void CopyFrom(const BaseMatrix<ValueType>& src);
        virtual void CopyFromAsync(const BaseMatrix<ValueType>& src);
        virtual void CopyTo(BaseMatrix<ValueType>* dst) const;
        virtual void CopyToAsync(BaseMatrix<ValueType>* dst) const;

        virtual void CopyFromHost(const HostMatrix<ValueType>& src);
        virtual void CopyFromHostAsync(const HostMatrix<ValueType>& src);
        virtual void CopyToHost(HostMatrix<ValueType>* dst) const;
        virtual void CopyToHostAsync(HostMatrix<ValueType>* dst) const;

        virtual void CopyFromCOO(const int* row, const int* col, const ValueType* val);
        virtual void CopyToCOO(int* row, int* col, ValueType* val) const;

        virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
        virtual void ApplyAdd(const BaseVector<ValueType>& in,
                              ValueType                    scalar,
                              BaseVector<ValueType>*       out) const;
        virtual bool Sort(void);

    private:
        MatrixCOO<ValueType, int> mat_;

        rocsparse_mat_descr mat_descr_;

        friend class HIPAcceleratorMatrixCSR<ValueType>;

        friend class BaseVector<ValueType>;
        friend class AcceleratorVector<ValueType>;
        friend class HIPAcceleratorVector<ValueType>;
    };

} // namespace rocalution

#endif // ROCALUTION_HIP_MATRIX_COO_HPP_
