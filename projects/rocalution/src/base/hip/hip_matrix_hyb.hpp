/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HIP_MATRIX_HYB_HPP_
#define ROCALUTION_HIP_MATRIX_HYB_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"
#include "hip_sparse.hpp"

#include <rocsparse/rocsparse.h>

namespace rocalution
{

    template <typename ValueType>
    class HIPAcceleratorMatrixHYB : public HIPAcceleratorMatrix<ValueType>
    {
    public:
        HIPAcceleratorMatrixHYB();
        explicit HIPAcceleratorMatrixHYB(const Rocalution_Backend_Descriptor& local_backend);
        virtual ~HIPAcceleratorMatrixHYB();

        inline int GetELLMaxRow(void) const
        {
            return this->mat_.ELL.max_row;
        }
        inline int64_t GetELLNnz(void) const
        {
            return this->ell_nnz_;
        }
        inline int64_t GetCOONnz(void) const
        {
            return this->coo_nnz_;
        }

        virtual void         Info(void) const;
        virtual unsigned int GetMatFormat(void) const
        {
            return HYB;
        }

        virtual void Clear(void);
        virtual void
            AllocateHYB(int64_t ell_nnz, int64_t coo_nnz, int ell_max_row, int nrow, int ncol);

        virtual bool ConvertFrom(const BaseMatrix<ValueType>& mat);

        virtual void CopyFrom(const BaseMatrix<ValueType>& src);
        virtual void CopyFromAsync(const BaseMatrix<ValueType>& src);
        virtual void CopyTo(BaseMatrix<ValueType>* dst) const;
        virtual void CopyToAsync(BaseMatrix<ValueType>* dst) const;

        virtual void CopyFromHost(const HostMatrix<ValueType>& src);
        virtual void CopyFromHostAsync(const HostMatrix<ValueType>& src);
        virtual void CopyToHost(HostMatrix<ValueType>* dst) const;
        virtual void CopyToHostAsync(HostMatrix<ValueType>* dst) const;

        virtual void ApplyAnalyse(void) const;
        virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
        virtual void ApplyAdd(const BaseVector<ValueType>& in,
                              ValueType                    scalar,
                              BaseVector<ValueType>*       out) const;

    private:
        // rocSPARSE keeps the COO SpMV analysis in the sparse matrix descriptor, so every
        // write to the arrays needs a new descriptor
        void CreateSpMatDescr_(void);
        void DestroySpMatDescr_(void);
        void ApplyAnalyseClear_(void);

        MatrixHYB<ValueType, int> mat_;

        int64_t ell_nnz_;
        int64_t coo_nnz_;

        rocsparse_spmat_descr ell_spmat_descr_;
        rocsparse_spmat_descr coo_spmat_descr_;

        mutable HIPSpMV<ValueType> ell_spmv_;
        mutable HIPSpMV<ValueType> coo_spmv_;

        friend class BaseVector<ValueType>;
        friend class AcceleratorVector<ValueType>;
        friend class HIPAcceleratorVector<ValueType>;
    };

} // namespace rocalution

#endif // ROCALUTION_HIP_MATRIX_HYB_HPP_
