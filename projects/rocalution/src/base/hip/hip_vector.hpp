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

#ifndef ROCALUTION_HIP_VECTOR_HPP_
#define ROCALUTION_HIP_VECTOR_HPP_

#include "../../utils/log.hpp"
#include "../backend_manager.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"

#include <hip/hip_runtime.h>

#include <complex>

#include "hip_rand.hpp"

namespace rocalution
{
    template <typename ValueType>
    class HIPAcceleratorVector : public AcceleratorVector<ValueType>
    {
    public:
        HIPAcceleratorVector();
        explicit HIPAcceleratorVector(const Rocalution_Backend_Descriptor& local_backend);
        virtual ~HIPAcceleratorVector();

        virtual void Info(void) const;

        virtual void Allocate(int64_t n);
        virtual void SetDataPtr(ValueType** ptr, int64_t size);
        virtual void LeaveDataPtr(ValueType** ptr);
        virtual void Clear(void);
        virtual void Zeros(void);
        virtual void Ones(void);
        virtual void SetValues(ValueType val);

        virtual void CopyFrom(const BaseVector<ValueType>& src);
        virtual void CopyFromAsync(const BaseVector<ValueType>& src);
        virtual void CopyFrom(const BaseVector<ValueType>& src,
                              int64_t                      src_offset,
                              int64_t                      dst_offset,
                              int64_t                      size);

        virtual void CopyTo(BaseVector<ValueType>* dst) const;
        virtual void CopyToAsync(BaseVector<ValueType>* dst) const;
        virtual void CopyFromFloat(const BaseVector<float>& src);
        virtual void CopyFromDouble(const BaseVector<double>& src);

        virtual void CopyFromHostAsync(const HostVector<ValueType>& src);
        virtual void CopyFromHost(const HostVector<ValueType>& src);
        virtual void CopyToHostAsync(HostVector<ValueType>* dst) const;
        virtual void CopyToHost(HostVector<ValueType>* dst) const;

        virtual void CopyFromData(const ValueType* data);
        virtual void CopyFromHostData(const ValueType* data);
        virtual void CopyToData(ValueType* data) const;
        virtual void CopyToHostData(ValueType* data) const;

        virtual void CopyFromPermute(const BaseVector<ValueType>& src,
                                     const BaseVector<int>&       permutation);
        virtual void CopyFromPermuteBackward(const BaseVector<ValueType>& src,
                                             const BaseVector<int>&       permutation);

        virtual void Permute(const BaseVector<int>& permutation);
        virtual void PermuteBackward(const BaseVector<int>& permutation);

        // this = this + alpha*x
        virtual void AddScale(const BaseVector<ValueType>& x, ValueType alpha);
        // this = alpha*this + x
        virtual void ScaleAdd(ValueType alpha, const BaseVector<ValueType>& x);
        // this = alpha*this + x*beta
        virtual void ScaleAddScale(ValueType alpha, const BaseVector<ValueType>& x, ValueType beta);
        virtual void ScaleAddScale(ValueType                    alpha,
                                   const BaseVector<ValueType>& x,
                                   ValueType                    beta,
                                   int64_t                      src_offset,
                                   int64_t                      dst_offset,
                                   int64_t                      size);
        // this = alpha*this + x*beta + y*gamma
        virtual void ScaleAdd2(ValueType                    alpha,
                               const BaseVector<ValueType>& x,
                               ValueType                    beta,
                               const BaseVector<ValueType>& y,
                               ValueType                    gamma);
        // this = alpha*this
        virtual void Scale(ValueType alpha);

        // this^T x
        virtual ValueType Dot(const BaseVector<ValueType>& x) const;
        // this^T x
        virtual ValueType DotNonConj(const BaseVector<ValueType>& x) const;
        // srqt(this^T this)
        virtual ValueType Norm(void) const;
        // reduce
        virtual ValueType Reduce(void) const;
        // Compute out-of-place inclusive sum
        virtual ValueType InclusiveSum(const BaseVector<ValueType>& vec);
        // Compute out-of-place exclusive sum
        virtual ValueType ExclusiveSum(const BaseVector<ValueType>& vec);
        // Compute sum of absolute values of this
        virtual ValueType Asum(void) const;
        // Compute absolute value of this
        virtual int64_t Amax(ValueType& value) const;
        // point-wise multiplication
        virtual void PointWiseMult(const BaseVector<ValueType>& x);
        virtual void PointWiseMult(const BaseVector<ValueType>& x, const BaseVector<ValueType>& y);
        virtual void Power(double power);

        // get index values
        virtual void GetIndexValues(const BaseVector<int>& index,
                                    BaseVector<ValueType>* values) const;
        // set index values
        virtual void SetIndexValues(const BaseVector<int>&       index,
                                    const BaseVector<ValueType>& values);
        // add index values
        virtual void AddIndexValues(const BaseVector<int>&       index,
                                    const BaseVector<ValueType>& values);
        // get continuous index values
        virtual void GetContinuousValues(int64_t start, int64_t end, ValueType* values) const;
        // set continuous index values
        virtual void SetContinuousValues(int64_t start, int64_t end, const ValueType* values);

        // update cf map
        virtual void RSPMISUpdateCFmap(const BaseVector<int>& index, BaseVector<ValueType>* values);
        // get coarse boundary mapping
        virtual void ExtractCoarseMapping(
            int64_t start, int64_t end, const int* index, int nc, int* size, int* map) const;
        // get coarse boundary index
        virtual void ExtractCoarseBoundary(
            int64_t start, int64_t end, const int* index, int nc, int* size, int* boundary) const;

        // set random values from a uniform distribution.
        virtual void SetRandomUniform(unsigned long long seed, ValueType a, ValueType b);
        // set random values from a normal distribution.
        virtual void SetRandomNormal(unsigned long long seed, ValueType mean, ValueType var);

        // set random values with an implementation of CRTP_HIPRand.
        template <typename RANDIMPL>
        inline void SetRandom(CRTP_HIPRand<RANDIMPL>& rand_engine)
        {
            if(this->size_ > 0)
            {
                rand_engine.Generate(this->vec_, this->size_);
            };
        };

        // out of place sort with permutation
        virtual void Sort(BaseVector<ValueType>* sorted, BaseVector<int>* perm) const;

    private:
        ValueType* vec_;

        friend class HIPAcceleratorVector<float>;
        friend class HIPAcceleratorVector<double>;
        friend class HIPAcceleratorVector<std::complex<float>>;
        friend class HIPAcceleratorVector<std::complex<double>>;
        friend class HIPAcceleratorVector<bool>;
        friend class HIPAcceleratorVector<int>;
        friend class HIPAcceleratorVector<int64_t>;

        friend class HostVector<ValueType>;
        friend class AcceleratorMatrix<ValueType>;

        friend class HIPAcceleratorMatrixCSR<ValueType>;
        friend class HIPAcceleratorMatrixMCSR<ValueType>;
        friend class HIPAcceleratorMatrixBCSR<ValueType>;
        friend class HIPAcceleratorMatrixCOO<ValueType>;
        friend class HIPAcceleratorMatrixDIA<ValueType>;
        friend class HIPAcceleratorMatrixELL<ValueType>;
        friend class HIPAcceleratorMatrixDENSE<ValueType>;
        friend class HIPAcceleratorMatrixHYB<ValueType>;

        friend class HIPAcceleratorMatrixCOO<double>;
        friend class HIPAcceleratorMatrixCOO<float>;
        friend class HIPAcceleratorMatrixCOO<std::complex<double>>;
        friend class HIPAcceleratorMatrixCOO<std::complex<float>>;

        friend class HIPAcceleratorMatrixCSR<double>;
        friend class HIPAcceleratorMatrixCSR<float>;
        friend class HIPAcceleratorMatrixCSR<std::complex<double>>;
        friend class HIPAcceleratorMatrixCSR<std::complex<float>>;
    };

} // namespace rocalution

#endif // ROCALUTION_BASE_VECTOR_HPP_
