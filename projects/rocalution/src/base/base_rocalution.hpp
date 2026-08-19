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

#ifndef ROCALUTION_BASE_HPP_
#define ROCALUTION_BASE_HPP_

#include "backend_manager.hpp"
#include "rocalution/export.hpp"

#include <complex>
#include <vector>

namespace rocalution
{
    template <typename ValueType>
    class GlobalVector;
    template <typename ValueType>
    class GlobalMatrix;
    class ParallelManager;

    /** \private */
    class RocalutionObj
    {
    public:
        RocalutionObj();
        virtual ~RocalutionObj();

        virtual void Clear() = 0;

    protected:
        size_t global_obj_id_;
    };

    // Global data for all ROCALUTION objects
    /** \private */
    struct Rocalution_Object_Data
    {
        std::vector<class RocalutionObj*> all_obj;
    };

    // Global obj tracking structure
    /** \private */
    extern struct Rocalution_Object_Data Rocalution_Object_Data_Tracking;

    /** \class BaseRocalution
  * \brief Base class for all operators and vectors
  *
  * \tparam ValueType - can be int, float, double, std::complex<float> and
  *                     std::complex<double>
  */
    template <typename ValueType>
    class BaseRocalution : public RocalutionObj
    {
    public:
        BaseRocalution();
        /** \private */
        BaseRocalution(const BaseRocalution<ValueType>& src);
        virtual ~BaseRocalution();

        /** \private */
        BaseRocalution<ValueType>& operator=(const BaseRocalution<ValueType>& src);

        /** \brief Move the object to the accelerator backend */
        virtual void MoveToAccelerator(void) = 0;

        /** \brief Move the object to the host backend */
        virtual void MoveToHost(void) = 0;

        /** \brief Move the object to the accelerator backend with async move */
        virtual void MoveToAcceleratorAsync(void);

        /** \brief Move the object to the host backend with async move */
        virtual void MoveToHostAsync(void);

        /** \brief Sync (the async move) */
        virtual void Sync(void);

        /** \brief Clone the Backend descriptor from another object
      * \details
      * With \p CloneBackend, the backend can be cloned without copying any data. This is
      * especially useful, if several objects should reside on the same backend, but keep
      * their original data.
      *
      * @param[in]
      * src Object, where the backend should be cloned from.
      *
      * \par Example
      * \code{.cpp}
      *   LocalVector<ValueType> vec;
      *   LocalMatrix<ValueType> mat;
      *
      *   // Allocate and initialize vec and mat
      *   // ...
      *
      *   LocalVector<ValueType> tmp;
      *   // By cloning backend, tmp and vec will have the same backend as mat
      *   tmp.CloneBackend(mat);
      *   vec.CloneBackend(mat);
      *
      *   // The following matrix vector multiplication will be performed on the backend
      *   // selected in mat
      *   mat.Apply(vec, &tmp);
      * \endcode
      */
        ROCALUTION_EXPORT
        virtual void CloneBackend(const BaseRocalution<ValueType>& src);

        // Clone the backend descriptor from another object with different
        // template ValueType
        template <typename ValueType2>
        ROCALUTION_EXPORT void CloneBackend(const BaseRocalution<ValueType2>& src); /**< \private */

        /** \brief Print object information
      * \details
      * \p Info can print object information about any rocALUTION object. This
      * information consists of object properties and backend data.
      *
      * \par Example
      * \code{.cpp}
      * mat.Info();
      * vec.Info();
      * \endcode
      */
        virtual void Info(void) const = 0;

        /** \brief Clear (free all data) the object */
        virtual void Clear(void) = 0;

    protected:
        /** \brief Name of the object */
        std::string object_name_;

        // Parallel Manager
        const ParallelManager* pm_; /**< \private */

        // Backend descriptor
        Rocalution_Backend_Descriptor local_backend_; /**< \private */

        /** \brief Return true if the object is on the host */
        virtual bool is_host_(void) const = 0;

        /** \brief Return true if the object is on the accelerator */
        virtual bool is_accel_(void) const = 0;

        // active async transfer
        bool asyncf_; /**< \private */

        friend class BaseRocalution<double>;
        friend class BaseRocalution<float>;
        friend class BaseRocalution<std::complex<double>>;
        friend class BaseRocalution<std::complex<float>>;

        friend class BaseRocalution<bool>;
        friend class BaseRocalution<int>;
        friend class BaseRocalution<int64_t>;

        friend class GlobalVector<int>;
        friend class GlobalVector<float>;
        friend class GlobalVector<double>;

        friend class GlobalMatrix<int>;
        friend class GlobalMatrix<float>;
        friend class GlobalMatrix<double>;
    };

} // namespace rocalution

#endif // ROCALUTION_LOCAL_BASE_HPP_
