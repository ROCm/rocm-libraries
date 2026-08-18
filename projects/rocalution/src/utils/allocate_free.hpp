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

#ifndef ROCALUTION_UTILS_ALLOCATE_FREE_HPP_
#define ROCALUTION_UTILS_ALLOCATE_FREE_HPP_

#include "rocalution/export.hpp"

#include <cstdint>

namespace rocalution
{
    /** \ingroup backend_module
  * \brief Allocate buffer on the host
  * \details
  * \p allocate_host allocates a buffer on the host.
  *
  * @param[in]
  * n       number of elements the buffer need to be allocated for
  * @param[out]
  * ptr     pointer to the position in memory where the buffer should be allocated,
  *         it is expected that \p *ptr == \p NULL
  *
  * \tparam DataType can be char, int, unsigned int, float, double, std::complex<float>
  *         or std::complex<double>.
  */
    template <typename DataType>
    ROCALUTION_EXPORT void allocate_host(int64_t n, DataType** ptr);

    /** \ingroup backend_module
  * \brief Allocate page-locked buffer on the host
  * \details
  * \p allocate_pinned allocates a page-locked buffer on the host.
  *
  * @param[in]
  * n       number of elements the buffer need to be allocated for
  * @param[out]
  * ptr     pointer to the position in memory where the buffer should be allocated,
  *         it is expected that \p *ptr == \p NULL
  *
  * \tparam DataType can be float, double, std::complex<float>
  *         or std::complex<double>.
  */
    template <typename DataType>
    ROCALUTION_EXPORT void allocate_pinned(int64_t n, DataType** ptr);

    /** \ingroup backend_module
  * \brief Free buffer on the host
  * \details
  * \p free_host deallocates a buffer on the host. \p *ptr will be set to NULL after
  * successful deallocation.
  *
  * @param[inout]
  * ptr     pointer to the position in memory where the buffer should be deallocated,
  *         it is expected that \p *ptr != \p NULL
  *
  * \tparam DataType can be char, int, unsigned int, float, double, std::complex<float>
  *         or std::complex<double>.
  */
    template <typename DataType>
    ROCALUTION_EXPORT void free_host(DataType** ptr);

    /** \ingroup backend_module
  * \brief Free page-locked buffer on the host
  * \details
  * \p free_host deallocates a page-locked buffer on the host. \p *ptr will be set to NULL after
  * successful deallocation.
  *
  * @param[inout]
  * ptr     pointer to the position in memory where the buffer should be deallocated,
  *         it is expected that \p *ptr != \p NULL
  *
  * \tparam DataType can be float, double, std::complex<float>
  *         or std::complex<double>.
  */
    template <typename DataType>
    ROCALUTION_EXPORT void free_pinned(DataType** ptr);

    /** \ingroup backend_module
  * \brief Set a host buffer to zero
  * \details
  * \p set_to_zero_host sets a host buffer to zero.
  *
  * @param[in]
  * n       number of elements
  * @param[inout]
  * ptr     pointer to the host buffer
  *
  * \tparam DataType can be char, int, unsigned int, float, double, std::complex<float>
  *         or std::complex<double>.
  */
    template <typename DataType>
    ROCALUTION_EXPORT void set_to_zero_host(int64_t n, DataType* ptr);

    /** \ingroup backend_module
  * \brief Copy a host buffer to another host buffer
  * \details
  * \p copy_h2h copies a host buffer to another host buffer.
  *
  * @param[in]
  * n       number of elements
  * @param[in]
  * src     pointer to the source host buffer
  * @param[out]
  * dst     pointer to the destination host buffer
  *
  * \tparam DataType can be bool, char, int, unsigned int, float, double,
  *         std::complex<float> or std::complex<double>
  */
    template <typename DataType>
    ROCALUTION_EXPORT void copy_h2h(int64_t n, const DataType* src, DataType* dst);
} // namespace rocalution

#endif // ROCALUTION_UTILS_ALLOCATE_FREE_HPP_
