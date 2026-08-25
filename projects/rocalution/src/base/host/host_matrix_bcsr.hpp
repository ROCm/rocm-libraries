/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HOST_MATRIX_BCSR_HPP_
#define ROCALUTION_HOST_MATRIX_BCSR_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace rocalution
{

    template <typename ValueType>
    class HostMatrixBCSR : public HostMatrix<ValueType>
    {
    public:
        HostMatrixBCSR();
        explicit HostMatrixBCSR(const Rocalution_Backend_Descriptor& local_backend, int blockdim);
        virtual ~HostMatrixBCSR();

        virtual void         Info(void) const;
        virtual unsigned int GetMatFormat(void) const
        {
            return BCSR;
        }

        virtual int GetMatBlockDimension(void) const
        {
            return this->mat_.blockdim;
        }

        virtual void Clear(void);
        virtual void AllocateBCSR(int64_t nnzb, int nrowb, int ncolb, int blockdim);
        virtual void SetDataPtrBCSR(int**       row_offset,
                                    int**       col,
                                    ValueType** val,
                                    int64_t     nnzb,
                                    int         nrowb,
                                    int         ncolb,
                                    int         blockdim);
        virtual void LeaveDataPtrBCSR(int** row_offset, int** col, ValueType** val, int& blockdim);

        virtual bool ConvertFrom(const BaseMatrix<ValueType>& mat);

        virtual void CopyFrom(const BaseMatrix<ValueType>& mat);
        virtual void CopyTo(BaseMatrix<ValueType>* mat) const;

        virtual bool ReadFileRSIO(const std::string& filename);
        virtual bool WriteFileRSIO(const std::string& filename) const;

        virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
        virtual void ApplyAdd(const BaseVector<ValueType>& in,
                              ValueType                    scalar,
                              BaseVector<ValueType>*       out) const;

    private:
        MatrixBCSR<ValueType, int> mat_;

        friend class BaseVector<ValueType>;
        friend class HostVector<ValueType>;
        friend class HostMatrixCSR<ValueType>;
        friend class HostMatrixCOO<ValueType>;
        friend class HostMatrixHYB<ValueType>;
        friend class HostMatrixDENSE<ValueType>;

        friend class HIPAcceleratorMatrixBCSR<ValueType>;
    };

} // namespace rocalution

#endif // ROCALUTION_HOST_MATRIX_BCSR_HPP_
