/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_ILUT_DRIVER_CSR_HPP_
#define ROCALUTION_ILUT_DRIVER_CSR_HPP_

#include <cstdlib>

namespace rocalution
{

    template <typename T, typename J>
    class ILUTDriverCSR
    {
    private:
        T* w;
        J* jw;
        J* jr;

        J m;
        J row;
        J bw;
        J w_size;

        J diag_pos;
        J len_u;
        J len_l;

        J stored_size_l;
        J stored_size_u;
        J front_work_l;

        double norm;

        void add_element(J col, T val);
        void swap(J index1, J index2);
        void partition(J begin, J size, J p);

    public:
        // Delete default behaviour
        ILUTDriverCSR()                     = delete;
        ILUTDriverCSR(const ILUTDriverCSR&) = delete;
        ILUTDriverCSR& operator=(const ILUTDriverCSR&) = delete;

        ILUTDriverCSR(J m, J bw);

        size_t buffer_size();

        void set_buffer(void* buffer);

        /*!
        *  \brief Initialize data structure for each row.
        *
        *  @param[in]
        *  val  The sparse row's value array
        *  @param[in]
        *  col  The sparse row's position array
        *  @param[in]
        *  size The sparse row's size
        *  @param[in]
        *  base The sparse row's index base
        *  @param[in]
        *  row  The current row being processed
        */
        void initialize(const T* val, const J* col, J size, int base, J row);

        /*!
        *  \brief Returns the column index and value of the next
        *  element to eliminate. In other words, the row element
        *  with the smallest column index less than the diagonal.
        *  Additionally, drops the previously returned element, unless
        *  \p save_lower() is called previously.
        *
        *  @param[out]
        *  col  The column of the next lower element.
        *  @param[out]
        *  val  The value of the next lower element.
        *
        *  \retval false if no remaining lower element, true otherwise.
        */
        bool next_lower(J& col, T& val);

        /*!
        *  \brief Called after \p next_lower() to save the element
        *  returned previously by \p next_lower() with value set
        *  to the parameter \p val. Use to store the factor
        *  calculated to eliminate the current lower element.
        *  Otherwise, do not call to drop the element.
        *
        *  @param[in]
        *  val  The value of the factor.
        */
        void save_lower(T val);

        /*!
        *  \brief Update values in row.
        *  Add \p val_to_add to element in position \p col.
        *
        *  @param[in]
        *  col  Column of the element which is added to.
        *  @param[in]
        *  val  The value to add.
        */
        void add_to_element(J col, T val_to_add);

        /*!
        *  \brief Get the number of elements in the row.
        *
        *  \retval the number of elements in the row.
        */
        J row_size();

        /*!
        *  \brief Store the row's values and column position in
        *  the given arrays and writes the index of the diagonal
        *  element.
        *
        *  @param[out]
        *  val     The array to store the values of the row.
        *  @param[out]
        *  col     The array to store the column indeces of the row.
        *  @param[out]
        *  diag_in The index of the diagonal element in the array.
        *
        *  \retval false if zero diagonal is found, true otherwise.
        */
        bool store_row(T* val, J* col, J& diag_in);

        /*!
        *  \brief Drop elemenets based on ILUT drop paramaters.
        *
        *  @param[in]
        *  tol  element value tolerance parameter
        *  @param[in]
        *  p    the maximum elements each in the upper and lower
        *       part of a row.
        */
        void trim(double tol, J p);
    };

} // namespace rocalution

#endif // ROCALUTION_ILUT_DRIVER_CSR_HPP_
