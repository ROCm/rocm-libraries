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

#ifndef ROCALUTION_GLOBAL_VECTOR_HPP_
#define ROCALUTION_GLOBAL_VECTOR_HPP_

#include "parallel_manager.hpp"
#include "vector.hpp"

namespace rocalution
{

    template <typename ValueType>
    class LocalVector;
    template <typename ValueType>
    class LocalMatrix;
    template <typename ValueType>
    class GlobalMatrix;
    struct MRequest;

    /** \ingroup op_vec_module
  * \class GlobalVector
  * \brief GlobalVector class
  * \details
  * A GlobalVector is called global, because it can stay on a single or on multiple nodes
  * in a network. For this type of communication, MPI is used.
  *
  * \tparam ValueType - can be int, float, double, std::complex<float> and
  *                     std::complex<double>
  */
    template <typename ValueType>
    class GlobalVector : public Vector<ValueType>
    {
    public:
        GlobalVector();
        /** \brief Initialize a global vector with a parallel manager */
        explicit GlobalVector(const ParallelManager& pm);
        virtual ~GlobalVector();

        /** \brief Move all data (i.e. move the part of the global vector stored on this rank) to the accelerator */
        virtual void MoveToAccelerator(void);
        /** \brief Move all data (i.e. move the part of the global vector stored on this rank) to the host */
        virtual void MoveToHost(void);

        /** \brief Shows simple info about the matrix. */
        virtual void Info(void) const;

        /** \brief Perform a sanity check of the vector
        * \details
        * Checks, if the vector contains valid data, i.e. if the values are not infinity
        * and not NaN (not a number).
        *
        * \retval true if the vector is ok (empty vector is also ok).
        * \retval false if there is something wrong with the values.
        */
        virtual bool Check(void) const;

        /** \brief Return the size of the global vector. */
        virtual int64_t GetSize(void) const;
        /** \brief Return the size of the interior part of the global vector. */
        virtual int64_t GetLocalSize(void) const;

        /** \private */
        const LocalVector<ValueType>& GetInterior() const;
        /** \private */
        LocalVector<ValueType>& GetInterior();

        /** \brief Allocate a global vector with name and size */
        virtual void Allocate(std::string name, int64_t size);

        /** \brief Clear (free) the vector */
        virtual void Clear(void);

        /** \brief Set the parallel manager of a global vector */
        void SetParallelManager(const ParallelManager& pm);

        /** \brief Set all vector interior values to zero */
        virtual void Zeros(void);
        /** \brief Set all vector interior values to ones */
        virtual void Ones(void);
        /** \brief Set the values of the interior vector to given argument */
        virtual void SetValues(ValueType val);
        /** \brief Set the values of the interior vector to random uniformly distributed values (between -1 and 1) */
        virtual void SetRandomUniform(unsigned long long seed,
                                      ValueType          a = static_cast<ValueType>(-1),
                                      ValueType          b = static_cast<ValueType>(1));
        /** \brief Set the values of the interior vector to random normally distributed values (between 0 and 1) */
        virtual void SetRandomNormal(unsigned long long seed,
                                     ValueType          mean = static_cast<ValueType>(0),
                                     ValueType          var  = static_cast<ValueType>(1));
        /** \brief Clone the entire vector (values,structure+backend descr) from another
        * GlobalVector
        */
        void CloneFrom(const GlobalVector<ValueType>& src);

        /** \brief Access operator (only for host data) */
        ValueType& operator[](int64_t i);
        /** \brief Access operator (only for host data) */
        const ValueType& operator[](int64_t i) const;

        /** \brief Initialize the local part of a global vector with externally allocated
      * data
      */
        void SetDataPtr(ValueType** ptr, std::string name, int64_t size);
        /** \brief Get a pointer to the data from the local part of a global vector and free
      * the global vector object
      */
        void LeaveDataPtr(ValueType** ptr);

        /** \brief Copy vector (values and structure) from another GlobalVector */
        virtual void CopyFrom(const GlobalVector<ValueType>& src);
        /** \brief Read GlobalVector from ASCII file. This method reads the current ranks interior vector from the file */
        virtual void ReadFileASCII(const std::string& filename);
        /** \brief Write GlobalVector to ASCII file. This method writes the current ranks interior vector to the file */
        virtual void WriteFileASCII(const std::string& filename) const;
        /** \brief Read GlobalVector from binary file. This method reads the current ranks interior vector from the file */
        virtual void ReadFileBinary(const std::string& filename);
        /** \brief Write GlobalVector to binary file. This method writes the current ranks interior vector to the file */
        virtual void WriteFileBinary(const std::string& filename) const;

        /** \brief Perform scalar-vector multiplication and add it to another vector, this = this + alpha * x; */
        virtual void AddScale(const GlobalVector<ValueType>& x, ValueType alpha);
        /** \brief Perform scalar-vector multiplication and add another vector, this = alpha * this + x; */
        virtual void ScaleAdd(ValueType alpha, const GlobalVector<ValueType>& x);
        /** \brief Perform vector update of type this = alpha*this + x*beta + y*gamma */
        virtual void ScaleAdd2(ValueType                      alpha,
                               const GlobalVector<ValueType>& x,
                               ValueType                      beta,
                               const GlobalVector<ValueType>& y,
                               ValueType                      gamma);
        /** \brief Perform scalar-vector multiplication and add another scaled vector (i.e. axpby), this = alpha * this + beta * x; */
        virtual void
            ScaleAddScale(ValueType alpha, const GlobalVector<ValueType>& x, ValueType beta);
        /** \brief Scale vector, this = alpha * this; */
        virtual void Scale(ValueType alpha);
        /** \brief Perform dot product */
        virtual ValueType Dot(const GlobalVector<ValueType>& x) const;
        /** \brief Perform non conjugate (when T is complex) dot product */
        virtual ValueType DotNonConj(const GlobalVector<ValueType>& x) const;
        /** \brief Compute L2 (Euclidean) norm of vector */
        virtual ValueType Norm(void) const;
        /** \brief Reduce (sum) the vector components */
        virtual ValueType Reduce(void) const;
        /** \brief Compute inclsuive sum of vector */
        virtual ValueType InclusiveSum(void);
        /** \brief Compute inclsuive sum of vector */
        virtual ValueType InclusiveSum(const GlobalVector<ValueType>& vec);
        /** \brief Compute exclsuive sum of vector */
        virtual ValueType ExclusiveSum(void);
        /** \brief Compute exclsuive sum of vector */
        virtual ValueType ExclusiveSum(const GlobalVector<ValueType>& vec);
        /** \brief Compute absolute value sum of vector components */
        virtual ValueType Asum(void) const;
        /** \brief Compute maximum absolute value component of vector */
        virtual int64_t Amax(ValueType& value) const;
        /** \brief Perform pointwise multiplication of vector */
        virtual void PointWiseMult(const GlobalVector<ValueType>& x);
        /** \brief Perform pointwise multiplication of vector */
        virtual void PointWiseMult(const GlobalVector<ValueType>& x,
                                   const GlobalVector<ValueType>& y);
        /** \brief Take the power of each vector component */
        virtual void Power(double power);

        /** \brief Restriction operator based on restriction mapping vector */
        void Restriction(const GlobalVector<ValueType>& vec_fine, const LocalVector<int>& map);

        /** \brief Prolongation operator based on restriction mapping vector */
        void Prolongation(const GlobalVector<ValueType>& vec_coarse, const LocalVector<int>& map);

    protected:
        /** \brief Return true if the object is on the host */
        virtual bool is_host_(void) const;
        /** \brief Return true if the object is on the accelerator */
        virtual bool is_accel_(void) const;

    private:
        LocalVector<ValueType> vector_interior_;

        friend class LocalMatrix<ValueType>;
        friend class GlobalMatrix<ValueType>;

        friend class BaseRocalution<ValueType>;
    };

} // namespace rocalution

#endif // ROCALUTION_GLOBAL_VECTOR_HPP_
