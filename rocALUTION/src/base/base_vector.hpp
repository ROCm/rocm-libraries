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

#ifndef ROCALUTION_BASE_VECTOR_HPP_
#define ROCALUTION_BASE_VECTOR_HPP_

#include "backend_manager.hpp"

namespace rocalution
{

    // Forward declarations
    template <typename ValueType>
    class HostVector;

    template <typename ValueType>
    class AcceleratorVector;

    template <typename ValueType>
    class HIPAcceleratorVector;

    /// Base class for all host/accelerator vectors
    template <typename ValueType>
    class BaseVector
    {
    public:
        BaseVector();
        virtual ~BaseVector();

        /** \brief Shows info about the object */
        virtual void Info(void) const = 0;

        /** \brief Returns the size of the vector */
        int64_t GetSize(void) const;

        /** \brief Copy the backend descriptor information */
        void set_backend(const Rocalution_Backend_Descriptor& local_backend);

        /** \brief Check if everything is ok */
        virtual bool Check(void) const;

        /** \brief Allocate a local vector with name and size */
        virtual void Allocate(int64_t n) = 0;

        /** \brief Initialize a vector with externally allocated data */
        virtual void SetDataPtr(ValueType** ptr, int64_t size) = 0;
        /** \brief Get a pointer from the vector data and free the vector object */
        virtual void LeaveDataPtr(ValueType** ptr) = 0;

        /** \brief Clear (free) the vector */
        virtual void Clear(void) = 0;
        /** \brief Set the values of the vector to zero */
        virtual void Zeros(void) = 0;
        /** \brief Set the values of the vector to one */
        virtual void Ones(void) = 0;
        /** \brief Set the values of the vector to given argument */
        virtual void SetValues(ValueType val) = 0;

        /** \brief Perform inplace permutation (forward) of the vector */
        virtual void Permute(const BaseVector<int>& permutation) = 0;
        /** \brief Perform inplace permutation (backward) of the vector */
        virtual void PermuteBackward(const BaseVector<int>& permutation) = 0;

        /** \brief Copy values from another vector */
        virtual void CopyFrom(const BaseVector<ValueType>& vec) = 0;
        /** \brief Async copy values from another vector */
        virtual void CopyFromAsync(const BaseVector<ValueType>& vec);
        /** \brief Copy values from another (float) vector */
        virtual void CopyFromFloat(const BaseVector<float>& vec);
        /** \brief Copy values from another (double) vector */
        virtual void CopyFromDouble(const BaseVector<double>& vec);
        /** \brief Copy values to another vector */
        virtual void CopyTo(BaseVector<ValueType>* vec) const = 0;
        /** \brief Async copy values to another vector */
        virtual void CopyToAsync(BaseVector<ValueType>* vec) const;
        /** \brief Copy data (not entire vector) from another vector with specified
        * src/dst offsets and size */
        virtual void CopyFrom(const BaseVector<ValueType>& src,
                              int64_t                      src_offset,
                              int64_t                      dst_offset,
                              int64_t                      size)
            = 0;

        /** \brief Copy a vector under specified permutation (forward permutation) */
        virtual void CopyFromPermute(const BaseVector<ValueType>& src,
                                     const BaseVector<int>&       permutation)
            = 0;
        /** \brief Copy a vector under specified permutation (backward permutation) */
        virtual void CopyFromPermuteBackward(const BaseVector<ValueType>& src,
                                             const BaseVector<int>&       permutation)
            = 0;

        virtual void CopyFromData(const ValueType* data);
        virtual void CopyFromHostData(const ValueType* data);
        virtual void CopyToData(ValueType* data) const;
        virtual void CopyToHostData(ValueType* data) const;

        /** \brief Restriction operator based on restriction mapping vector */
        virtual bool Restriction(const BaseVector<ValueType>& vec_fine, const BaseVector<int>& map);

        /** \brief Prolongation operator based on restriction(!) mapping vector */
        virtual bool Prolongation(const BaseVector<ValueType>& vec_coarse,
                                  const BaseVector<int>&       map);

        /** \brief Perform vector update of type this = this + alpha*x */
        virtual void AddScale(const BaseVector<ValueType>& x, ValueType alpha) = 0;
        /** \brief Perform vector update of type this = alpha*this + x */
        virtual void ScaleAdd(ValueType alpha, const BaseVector<ValueType>& x) = 0;
        /** \brief Perform vector update of type this = alpha*this + x*beta */
        virtual void ScaleAddScale(ValueType alpha, const BaseVector<ValueType>& x, ValueType beta)
            = 0;

        virtual void ScaleAddScale(ValueType                    alpha,
                                   const BaseVector<ValueType>& x,
                                   ValueType                    beta,
                                   int64_t                      src_offset,
                                   int64_t                      dst_offset,
                                   int64_t                      size)
            = 0;

        /** \brief Perform vector update of type this = alpha*this + x*beta + y*gamma */
        virtual void ScaleAdd2(ValueType                    alpha,
                               const BaseVector<ValueType>& x,
                               ValueType                    beta,
                               const BaseVector<ValueType>& y,
                               ValueType                    gamma)
            = 0;
        /** \brief Perform vector scaling this = alpha*this */
        virtual void Scale(ValueType alpha) = 0;
        /** \brief Compute dot (scalar) product, return this^T y */
        virtual ValueType Dot(const BaseVector<ValueType>& x) const = 0;
        /** \brief Compute non-conjugated dot (scalar) product, return this^T y */
        virtual ValueType DotNonConj(const BaseVector<ValueType>& x) const = 0;
        /** \brief Compute L2 norm of the vector, return =  srqt(this^T this) */
        virtual ValueType Norm(void) const = 0;
        /** \brief Reduce vector */
        virtual ValueType Reduce(void) const = 0;
        /** \brief Compute out-of-place inclusive sum */
        virtual ValueType InclusiveSum(const BaseVector<ValueType>& vec) = 0;
        /** \brief Compute out-of-place exclusive sum */
        virtual ValueType ExclusiveSum(const BaseVector<ValueType>& vec) = 0;
        /** \brief Compute sum of absolute values of the vector (L1 norm), return =  sum(|this|) */
        virtual ValueType Asum(void) const = 0;
        /** \brief Compute the absolute max value of the vector, return =  max(|this|) */
        virtual int64_t Amax(ValueType& value) const = 0;
        /** \brief Perform point-wise multiplication (element-wise) of type this = this * x */
        virtual void PointWiseMult(const BaseVector<ValueType>& x) = 0;
        /** \brief Perform point-wise multiplication (element-wise) of type this = x*y */
        virtual void PointWiseMult(const BaseVector<ValueType>& x, const BaseVector<ValueType>& y)
            = 0;
        /** \brief Compute power of each vector component*/
        virtual void Power(double power) = 0;

        /** \brief Gets index values */
        virtual void GetIndexValues(const BaseVector<int>& index,
                                    BaseVector<ValueType>* values) const = 0;
        /** \brief Sets index values */
        virtual void SetIndexValues(const BaseVector<int>&       index,
                                    const BaseVector<ValueType>& values)
            = 0;
        /** \brief Adds index values */
        virtual void AddIndexValues(const BaseVector<int>&       index,
                                    const BaseVector<ValueType>& values)
            = 0;
        /** \brief Gets continuous index values */
        virtual void GetContinuousValues(int64_t start, int64_t end, ValueType* values) const = 0;
        /** \brief Sets continuous index values */
        virtual void SetContinuousValues(int64_t start, int64_t end, const ValueType* values) = 0;
        /** \brief Updates the CF map */
        virtual void RSPMISUpdateCFmap(const BaseVector<int>& index, BaseVector<ValueType>* values)
            = 0;
        /** \brief Extract coarse boundary mapping */
        virtual void ExtractCoarseMapping(
            int64_t start, int64_t end, const int* index, int nc, int* size, int* map) const = 0;
        /** \brief Extract coarse boundary index */
        virtual void ExtractCoarseBoundary(int64_t    start,
                                           int64_t    end,
                                           const int* index,
                                           int        nc,
                                           int*       size,
                                           int*       boundary) const = 0;

        virtual void SetRandomUniform(unsigned long long seed, ValueType a, ValueType b)     = 0;
        virtual void SetRandomNormal(unsigned long long seed, ValueType mean, ValueType var) = 0;

        virtual void Sort(BaseVector<ValueType>* sorted, BaseVector<int>* perm) const = 0;

    protected:
        /** \brief The size of the vector */
        int64_t size_;

        /** \brief Backend descriptor (local copy) */
        Rocalution_Backend_Descriptor local_backend_;
    };

    template <typename ValueType>
    class AcceleratorVector : public BaseVector<ValueType>
    {
    public:
        AcceleratorVector();
        virtual ~AcceleratorVector();

        /** \brief Copy (accelerator vector) from host vector */
        virtual void CopyFromHost(const HostVector<ValueType>& src) = 0;
        /** \brief Copy (host vector) from accelerator vector */
        virtual void CopyToHost(HostVector<ValueType>* dst) const = 0;

        /** \brief Async copy (accelerator vector) from host vector */
        virtual void CopyFromHostAsync(const HostVector<ValueType>& src);
        /** \brief Async copy (host vector) from accelerator vector */
        virtual void CopyToHostAsync(HostVector<ValueType>* dst) const;
    };

} // namespace rocalution

#endif // ROCALUTION_BASE_VECTOR_HPP_
