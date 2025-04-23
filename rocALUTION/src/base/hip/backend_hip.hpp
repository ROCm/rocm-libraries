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

#ifndef ROCALUTION_BACKEND_HIP_HPP_
#define ROCALUTION_BACKEND_HIP_HPP_

#include "../backend_manager.hpp"

namespace rocalution
{

    template <typename ValueType>
    class AcceleratorVector;
    template <typename ValueType>
    class AcceleratorMatrix;
    template <typename ValueType>
    class HostMatrix;

    /** \brief Initialize HIP (rocBLAS, rocSPARSE) */
    bool rocalution_init_hip();
    /** \brief Release HIP resources (rocBLAS, rocSPARSE) */
    void rocalution_stop_hip();

    /** \brief Print information about the HIPs in the systems */
    void rocalution_info_hip(const struct Rocalution_Backend_Descriptor& backend_descriptor);

    /** \brief Sync the device (for async transfers) */
    void rocalution_hip_sync(void);

    /** \brief Sync the default stream */
    void rocalution_hip_sync_default(void);

    /** \brief Sync the interior stream */
    void rocalution_hip_sync_interior(void);

    /** \brief Sync the ghost stream */
    void rocalution_hip_sync_ghost(void);

    /** \brief Returns name of device architecture */
    std::string rocalution_get_arch_hip(void);

    /** \brief Set the rocsparse stream to 'interior' */
    void rocalution_hip_compute_interior(void);

    /** \brief Set the rocsparse stream to 'ghost' */
    void rocalution_hip_compute_ghost(void);

    /** \brief Set the rocsparse stream to default */
    void rocalution_hip_compute_default(void);

    /** \brief Build (and return) a vector on HIP */
    template <typename ValueType>
    AcceleratorVector<ValueType>* _rocalution_init_base_hip_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);

    /** \brief Build (and return) a matrix on HIP */
    template <typename ValueType>
    AcceleratorMatrix<ValueType>* _rocalution_init_base_hip_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim = 1);

} // namespace rocalution

#endif // ROCALUTION_BACKEND_HIP_HPP_
