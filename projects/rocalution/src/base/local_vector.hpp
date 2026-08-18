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

#ifndef ROCALUTION_LOCAL_VECTOR_HPP_
#define ROCALUTION_LOCAL_VECTOR_HPP_

#include "rocalution/export.hpp"
#include "vector.hpp"

namespace rocalution
{

    template <typename ValueType>
    class BaseVector;
    template <typename ValueType>
    class HostVector;

    template <typename ValueType>
    class LocalMatrix;
    template <typename ValueType>
    class GlobalMatrix;

    template <typename ValueType>
    class LocalStencil;

    /** \ingroup op_vec_module
  * \class LocalVector
  * \brief LocalVector class
  * \details
  * A LocalVector is called local, because it will always stay on a single system. The
  * system can contain several CPUs via UMA or NUMA memory system or it can contain an
  * accelerator.
  *
  * \tparam ValueType - can be int, float, double, std::complex<float> and
  *                     std::complex<double>
  */
    template <typename ValueType>
    class LocalVector : public Vector<ValueType>
    {
    public:
        ROCALUTION_EXPORT
        LocalVector();
        ROCALUTION_EXPORT
        virtual ~LocalVector();

        /** \brief Move all data (i.e. move the vector) to the accelerator */
        ROCALUTION_EXPORT
        virtual void MoveToAccelerator(void);
        /** \brief Move all data (i.e. move the vector) to the accelerator asynchronously */
        ROCALUTION_EXPORT
        virtual void MoveToAcceleratorAsync(void);
        /** \brief Move all data (i.e. move the vector) to the host */
        ROCALUTION_EXPORT
        virtual void MoveToHost(void);
        /** \brief Move all data (i.e. move the vector) to the host asynchronously */
        ROCALUTION_EXPORT
        virtual void MoveToHostAsync(void);
        /** \brief Synchronize the vector */
        ROCALUTION_EXPORT
        virtual void Sync(void);

        /** \brief Shows simple info about the vector. */
        ROCALUTION_EXPORT
        virtual void Info(void) const;
        /** \brief Return the size of the vector. */
        ROCALUTION_EXPORT
        virtual int64_t GetSize(void) const;

        /** \private */
        ROCALUTION_EXPORT
        const LocalVector<ValueType>& GetInterior() const;
        /** \private */
        ROCALUTION_EXPORT
        LocalVector<ValueType>& GetInterior();

        /** \brief Perform a sanity check of the vector
        * \details
        * Checks, if the vector contains valid data, i.e. if the values are not infinity
        * and not NaN (not a number).
        *
        * \retval true if the vector is ok (empty vector is also ok).
        * \retval false if there is something wrong with the values.
        */
        ROCALUTION_EXPORT
        virtual bool Check(void) const;

        /** \brief Allocate a local vector with name and size
      * \details
      * The local vector allocation function requires a name of the object (this is only
      * for information purposes) and corresponding size description for vector objects.
      *
      * @param[in]
      * name    object name
      * @param[in]
      * size    number of elements in the vector
      *
      * \par Example
      * \code{.cpp}
      *   LocalVector<ValueType> vec;
      *
      *   vec.Allocate("my vector", 100);
      *   vec.Clear();
      * \endcode
      */
        ROCALUTION_EXPORT
        void Allocate(std::string name, int64_t size);

        /** \brief Initialize a LocalVector on the host with externally allocated data
      * \details
      * \p SetDataPtr has direct access to the raw data via pointers. Already allocated
      * data can be set by passing the pointer.
      *
      * \note
      * Setting data pointer will leave the original pointer empty (set to \p NULL).
      *
      * \par Example
      * \code{.cpp}
      *   // Allocate vector
      *   ValueType* ptr_vec = new ValueType[200];
      *
      *   // Fill vector
      *   // ...
      *
      *   // rocALUTION local vector object
      *   LocalVector<ValueType> vec;
      *
      *   // Set the vector data, ptr_vec will become invalid
      *   vec.SetDataPtr(&ptr_vec, "my_vector", 200);
      * \endcode
      */
        ROCALUTION_EXPORT
        void SetDataPtr(ValueType** ptr, std::string name, int64_t size);

        /** \brief Leave a LocalVector to host pointers
      * \details
      * \p LeaveDataPtr has direct access to the raw data via pointers. A LocalVector
      * object can leave its raw data to a host pointer. This will leave the LocalVector
      * empty.
      *
      * \par Example
      * \code{.cpp}
      *   // rocALUTION local vector object
      *   LocalVector<ValueType> vec;
      *
      *   // Allocate the vector
      *   vec.Allocate("my_vector", 100);
      *
      *   // Fill vector
      *   // ...
      *
      *   ValueType* ptr_vec = NULL;
      *
      *   // Get (steal) the data from the vector, this will leave the local vector object empty
      *   vec.LeaveDataPtr(&ptr_vec);
      * \endcode
      */
        ROCALUTION_EXPORT
        void LeaveDataPtr(ValueType** ptr);

        /** \brief Clear (free) the vector */
        ROCALUTION_EXPORT
        virtual void Clear();
        /** \brief Set the values of the vector to zero */
        ROCALUTION_EXPORT
        virtual void Zeros();
        /** \brief Set the values of the vector to one */
        ROCALUTION_EXPORT
        virtual void Ones();
        /** \brief Set the values of the vector to given argument */
        ROCALUTION_EXPORT
        virtual void SetValues(ValueType val);
        /** \brief Set the values of the vector to random uniformly distributed values (between -1 and 1) */
        ROCALUTION_EXPORT
        virtual void SetRandomUniform(unsigned long long seed,
                                      ValueType          a = static_cast<ValueType>(-1),
                                      ValueType          b = static_cast<ValueType>(1));
        /** \brief Set the values of the vector to random normally distributed values (between 0 and 1) */
        ROCALUTION_EXPORT
        virtual void SetRandomNormal(unsigned long long seed,
                                     ValueType          mean = static_cast<ValueType>(0),
                                     ValueType          var  = static_cast<ValueType>(1));

        /** \brief Access operator (only for host data)
      * \details
      * The elements in the vector can be accessed via [] operators, when the vector is
      * allocated on the host.
      *
      * @param[in]
      * i   access data at index \p i
      *
      * \returns    value at index \p i
      *
      * \par Example
      * \code{.cpp}
      *   // rocALUTION local vector object
      *   LocalVector<ValueType> vec;
      *
      *   // Allocate vector
      *   vec.Allocate("my_vector", 100);
      *
      *   // Initialize vector with 1
      *   vec.Ones();
      *
      *   // Set even elements to -1
      *   for(int64_t i = 0; i < vec.GetSize(); i += 2)
      *   {
      *     vec[i] = -1;
      *   }
      * \endcode
      */
        /**@{*/
        ROCALUTION_EXPORT
        ValueType& operator[](int64_t i);
        ROCALUTION_EXPORT
        const ValueType& operator[](int64_t i) const;
        /**@}*/

        /** \brief Read LocalVector from ASCII file. */
        ROCALUTION_EXPORT
        virtual void ReadFileASCII(const std::string& filename);
        /** \brief Write LocalVector to ASCII file. */
        ROCALUTION_EXPORT
        virtual void WriteFileASCII(const std::string& filename) const;
        /** \brief Read LocalVector from binary file. */
        ROCALUTION_EXPORT
        virtual void ReadFileBinary(const std::string& filename);
        /** \brief Write LocalVector to binary file. */
        ROCALUTION_EXPORT
        virtual void WriteFileBinary(const std::string& filename) const;

        /** \brief Clone the entire vector (values,structure+backend descr) from another
        * LocalVector
        */
        ROCALUTION_EXPORT
        virtual void CopyFrom(const LocalVector<ValueType>& src);
        /** \brief Async copy from another local vector */
        ROCALUTION_EXPORT
        virtual void CopyFromAsync(const LocalVector<ValueType>& src);
        /** \brief Copy values from another local float vector */
        ROCALUTION_EXPORT
        virtual void CopyFromFloat(const LocalVector<float>& src);
        /** \brief Copy values from another local double vector */
        ROCALUTION_EXPORT
        virtual void CopyFromDouble(const LocalVector<double>& src);
        /** \brief Copy from another vector */
        ROCALUTION_EXPORT
        virtual void CopyFrom(const LocalVector<ValueType>& src,
                              int64_t                       src_offset,
                              int64_t                       dst_offset,
                              int64_t                       size);

        /** \brief Copy a vector under permutation (forward permutation) */
        ROCALUTION_EXPORT
        void CopyFromPermute(const LocalVector<ValueType>& src,
                             const LocalVector<int>&       permutation);

        /** \brief Copy a vector under permutation (backward permutation) */
        ROCALUTION_EXPORT
        void CopyFromPermuteBackward(const LocalVector<ValueType>& src,
                                     const LocalVector<int>&       permutation);

        /** \brief Clone from another vector */
        ROCALUTION_EXPORT
        virtual void CloneFrom(const LocalVector<ValueType>& src);

        /** \brief Copy (import) vector
      * \details
      * Copy (import) vector data that is described in one array (values). The object
      * data has to be allocated with Allocate(), using the corresponding size of the
      * data, first.
      *
      * @param[in]
      * data    data to be imported.
      */
        ROCALUTION_EXPORT
        void CopyFromData(const ValueType* data);
        /** \brief Copy (import) vector from host data
      * \details
      * Copy (import) vector data that is described in one host array (values). The object
      * data has to be allocated with Allocate(), using the corresponding size of the
      * data, first.
      *
      * @param[in]
      * data    data to be imported from host.
      */
        ROCALUTION_EXPORT
        void CopyFromHostData(const ValueType* data);

        /** \brief Copy (export) vector
      * \details
      * Copy (export) vector data that is described in one array (values). The output
      * array has to be allocated, using the corresponding size of the data, first.
      * Size can be obtain by GetSize().
      *
      * @param[out]
      * data    exported data.
      */
        ROCALUTION_EXPORT
        void CopyToData(ValueType* data) const;
        /** \brief Copy (export) vector to host data
      * \details
      * Copy (export) vector data that is described in one array (values). The output
      * array has to be allocated on the host, using the corresponding size of the data, first.
      * Size can be obtain by GetSize().
      *
      * @param[out]
      * data    exported data on host.
      */
        ROCALUTION_EXPORT
        void CopyToHostData(ValueType* data) const;

        /** \brief Perform in-place permutation (forward) of the vector */
        ROCALUTION_EXPORT
        void Permute(const LocalVector<int>& permutation);

        /** \brief Perform in-place permutation (backward) of the vector */
        ROCALUTION_EXPORT
        void PermuteBackward(const LocalVector<int>& permutation);

        /** \brief Restriction operator based on restriction mapping vector */
        ROCALUTION_EXPORT
        void Restriction(const LocalVector<ValueType>& vec_fine, const LocalVector<int>& map);

        /** \brief Prolongation operator based on restriction mapping vector */
        ROCALUTION_EXPORT
        void Prolongation(const LocalVector<ValueType>& vec_coarse, const LocalVector<int>& map);

        /** \brief Perform scalar-vector multiplication and add it to another vector, this = this + alpha * x;
      * \par Example
      * \code{.cpp}
      * // rocALUTION structures
      * LocalVector<T> x;
      * LocalVector<T> y;
      *
      * // Allocate vectors
      * x.Allocate("x", 100);
      * y.Allocate("y", 100);
      *
      * x.Ones();
      * y.Ones();
      *
      * T alpha = 2.0;
      * y.AddScale(x, alpha);
      * \endcode
      */
        ROCALUTION_EXPORT
        virtual void AddScale(const LocalVector<ValueType>& x, ValueType alpha);

        /** \brief Perform scalar-vector multiplication and add another vector, this = alpha * this + x;
      * \par Example
      * \code{.cpp}
      * // rocALUTION structures
      * LocalVector<T> x;
      * LocalVector<T> y;
      *
      * // Allocate vectors
      * x.Allocate("x", 100);
      * y.Allocate("y", 100);
      *
      * x.Ones();
      * y.Ones();
      *
      * T alpha = 2.0;
      * y.ScaleAdd(alpha, x);
      * \endcode
      */
        ROCALUTION_EXPORT
        virtual void ScaleAdd(ValueType alpha, const LocalVector<ValueType>& x);

        /** \brief Perform scalar-vector multiplication and add another scaled vector (i.e. axpby), this = alpha * this + beta * x;
      * \par Example
      * \code{.cpp}
      * // rocALUTION structures
      * LocalVector<T> x;
      * LocalVector<T> y;
      *
      * // Allocate vectors
      * x.Allocate("x", 100);
      * y.Allocate("y", 100);
      *
      * x.Ones();
      * y.Ones();
      *
      * T alpha = 2.0;
      * T beta = -1.0;
      * y.ScaleAddScale(alpha, x, beta);
      * \endcode
      */
        /**@{*/
        ROCALUTION_EXPORT
        virtual void
            ScaleAddScale(ValueType alpha, const LocalVector<ValueType>& x, ValueType beta);
        ROCALUTION_EXPORT
        virtual void ScaleAddScale(ValueType                     alpha,
                                   const LocalVector<ValueType>& x,
                                   ValueType                     beta,
                                   int64_t                       src_offset,
                                   int64_t                       dst_offset,
                                   int64_t                       size);

        /** \brief Perform vector update of type this = alpha*this + x*beta + y*gamma */
        ROCALUTION_EXPORT
        virtual void ScaleAdd2(ValueType                     alpha,
                               const LocalVector<ValueType>& x,
                               ValueType                     beta,
                               const LocalVector<ValueType>& y,
                               ValueType                     gamma);
        /**@}*/

        /** \brief Scale vector, this = alpha * this;
      * \par Example
      * \code{.cpp}
      * // rocALUTION structures
      * LocalVector<T> y;
      *
      * // Allocate vectors
      * y.Allocate("y", 100);
      *
      * y.Ones();
      *
      * T alpha = 2.0;
      * y.Scale(alpha);
      * \endcode
      */
        ROCALUTION_EXPORT
        virtual void Scale(ValueType alpha);

        /** \brief Perform dot product
      * \details
      * Perform dot product of 'this' vector and the vector x. In the case of complex types, this performs
      * conjugate dot product.
      *
      * \par Example
      * \code{.cpp}
      * // rocALUTION structures
      * LocalVector<T> x;
      * LocalVector<T> y;
      *
      * // Allocate vectors
      * y.Allocate("x", 100);
      * y.Allocate("y", 100);
      *
      * x.Ones();
      * y.Ones();
      *
      * y.Dot(x);
      * \endcode
      */
        ROCALUTION_EXPORT
        virtual ValueType Dot(const LocalVector<ValueType>& x) const;

        /** \brief Perform dot product
      * \par Example
      * \code{.cpp}
      * // rocALUTION structures
      * LocalVector<T> x;
      * LocalVector<T> y;
      *
      * // Allocate vectors
      * y.Allocate("x", 100);
      * y.Allocate("y", 100);
      *
      * x.Ones();
      * y.Ones();
      *
      * y.Dot(x);
      * \endcode
      */
        ROCALUTION_EXPORT
        virtual ValueType DotNonConj(const LocalVector<ValueType>& x) const;

        /** \brief Compute L2 (Euclidean) norm of vector
      * \par Example
      * \code{.cpp}
      * // rocALUTION structures
      * LocalVector<T> y;
      *
      * // Allocate vectors
      * y.Allocate("y", 100);
      *
      * y.Ones();
      *
      * T norm2 = y.Norm();
      * \endcode
      */
        ROCALUTION_EXPORT
        virtual ValueType Norm(void) const;

        /** \brief Reduce (sum) the vector components
      * \par Example
      * \code{.cpp}
      * // rocALUTION structures
      * LocalVector<T> y;
      *
      * // Allocate vectors
      * y.Allocate("y", 100);
      *
      * y.Ones();
      *
      * T sum = y.Reduce();
      * \endcode
      */
        ROCALUTION_EXPORT
        virtual ValueType Reduce(void) const;

        /** \brief Compute inclsuive sum of vector
      * \par Example
      * Given starting vector:
      *    this = [1, 1, 1, 1]
      * After performing inclusive sum out vector will be:
      *    this = [1, 2, 3, 4]
      * The function returns 4.
      *
      * \code{.cpp}
      * // rocALUTION structures
      * LocalVector<T> y;
      *
      * // Allocate vectors
      * y.Allocate("y", 100);
      *
      * y.Ones();
      *
      * T sum = y.InclusiveSum();
      * \endcode
      */
        /**@{*/
        ROCALUTION_EXPORT
        virtual ValueType InclusiveSum(void);
        ROCALUTION_EXPORT
        virtual ValueType InclusiveSum(const LocalVector<ValueType>& vec);
        /**@}*/

        /** \brief Compute exclusive sum of vector
      * \par Example
      * Given starting vector:
      *    this = [1, 1, 1, 1]
      * After performing exclusive sum out vector will be:
      *    this = [0, 1, 2, 3]
      * The function returns 3.
      *
      * \code{.cpp}
      * // rocALUTION structures
      * LocalVector<T> y;
      *
      * // Allocate vectors
      * y.Allocate("y", 100);
      *
      * y.Ones();
      *
      * T sum = y.ExclusiveSum();
      * \endcode
      */
        /**@{*/
        ROCALUTION_EXPORT
        virtual ValueType ExclusiveSum(void);
        ROCALUTION_EXPORT
        virtual ValueType ExclusiveSum(const LocalVector<ValueType>& vec);
        /**@}*/

        /** \brief Compute absolute value sum of vector components
      * \par Example
      * \code{.cpp}
      * // rocALUTION structures
      * LocalVector<T> y;
      *
      * // Allocate vectors
      * y.Allocate("y", 100);
      *
      * y.Ones();
      *
      * y.Scale(-1.0);
      * T sum = y.Asum();
      * \endcode
      */
        ROCALUTION_EXPORT
        virtual ValueType Asum(void) const;

        /** \brief Compute maximum absolute value component of vector
      * \par Example
      * \code{.cpp}
      * // rocALUTION structures
      * LocalVector<T> y;
      *
      * // Allocate vectors
      * y.Allocate("y", 100);
      *
      * y.Ones();
      *
      * T max = y.Amax();
      * \endcode
      */
        ROCALUTION_EXPORT
        virtual int64_t Amax(ValueType& value) const;

        /** \brief Perform pointwise multiplication of vector
      * \details
      * Perform pointwise multiplication of vector components with the vector components of x, this = this * x.
      * Alternatively, one can also perform pointwise multiplication of vector components of x with
      * vector components of y and set that to the current 'this' vector, this = x * y.
      *
      * \par Example
      * \code{.cpp}
      * // rocALUTION structures
      * LocalVector<T> x;
      * LocalVector<T> y;
      *
      * // Allocate vectors
      * y.Allocate("x", 100);
      * y.Allocate("y", 100);
      *
      * x.Ones();
      * y.Ones();
      *
      * y.PointWiseMult(x);
      * \endcode
      */
        /**@{*/
        ROCALUTION_EXPORT
        virtual void PointWiseMult(const LocalVector<ValueType>& x);
        ROCALUTION_EXPORT
        virtual void PointWiseMult(const LocalVector<ValueType>& x,
                                   const LocalVector<ValueType>& y);
        /**@}*/

        /** \brief Take the power of each vector component */
        ROCALUTION_EXPORT
        virtual void Power(double power);

        /** \brief Get indexed values */
        ROCALUTION_EXPORT
        void GetIndexValues(const LocalVector<int>& index, LocalVector<ValueType>* values) const;
        /** \brief Set indexed values */
        ROCALUTION_EXPORT
        void SetIndexValues(const LocalVector<int>& index, const LocalVector<ValueType>& values);
        /** \brief Add indexed values */
        ROCALUTION_EXPORT
        void AddIndexValues(const LocalVector<int>& index, const LocalVector<ValueType>& values);
        /** \brief Get continuous indexed values */
        ROCALUTION_EXPORT
        void GetContinuousValues(int64_t start, int64_t end, ValueType* values) const;
        /** \brief Set continuous indexed values */
        ROCALUTION_EXPORT
        void SetContinuousValues(int64_t start, int64_t end, const ValueType* values);
        /** \brief Extract coarse boundary mapping */
        ROCALUTION_EXPORT
        void ExtractCoarseMapping(
            int64_t start, int64_t end, const int* index, int nc, int* size, int* map) const;
        /** \brief Extract coarse boundary index */
        ROCALUTION_EXPORT
        void ExtractCoarseBoundary(
            int64_t start, int64_t end, const int* index, int nc, int* size, int* boundary) const;

        /** \brief Out-of-place radix sort that can also obtain the permutation */
        ROCALUTION_EXPORT
        void Sort(LocalVector<ValueType>* sorted, LocalVector<int>* perm = NULL) const;

    protected:
        /** \brief Return true if the object is on the host */
        virtual bool is_host_(void) const;
        /** \brief Return true if the object is on the accelerator */
        virtual bool is_accel_(void) const;

    private:
        // Pointer from the base vector class to the current allocated vector (host_ or accel_)
        BaseVector<ValueType>* vector_;

        // Host Vector
        HostVector<ValueType>* vector_host_;

        // Accelerator Vector
        AcceleratorVector<ValueType>* vector_accel_;

        friend class LocalVector<bool>;
        friend class LocalVector<double>;
        friend class LocalVector<float>;
        friend class LocalVector<std::complex<double>>;
        friend class LocalVector<std::complex<float>>;
        friend class LocalVector<int>;
        friend class LocalVector<int64_t>;

        friend class LocalMatrix<double>;
        friend class LocalMatrix<float>;
        friend class LocalMatrix<std::complex<double>>;
        friend class LocalMatrix<std::complex<float>>;

        friend class GlobalMatrix<double>;
        friend class GlobalMatrix<float>;
        friend class GlobalMatrix<std::complex<double>>;
        friend class GlobalMatrix<std::complex<float>>;

        friend class LocalStencil<double>;
        friend class LocalStencil<float>;
        friend class LocalStencil<std::complex<double>>;
        friend class LocalStencil<std::complex<float>>;

        friend class GlobalVector<ValueType>;
        friend class LocalMatrix<ValueType>;
        friend class GlobalMatrix<ValueType>;
    };

} // namespace rocalution

#endif // ROCALUTION_LOCAL_VECTOR_HPP_
