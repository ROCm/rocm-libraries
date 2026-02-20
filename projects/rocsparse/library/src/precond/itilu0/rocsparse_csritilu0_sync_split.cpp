/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

#include "../conversion/rocsparse_csxsldu.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "common.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_csritilu0_driver.hpp"
#include "rocsparse_csritilu0x_buffer_size.hpp"
#include "rocsparse_csritilu0x_compute.hpp"
#include "rocsparse_csritilu0x_history.hpp"
#include "rocsparse_csritilu0x_preprocess.hpp"

template <>
struct rocsparse::csritilu0_driver_t<rocsparse_itilu0_alg_sync_split>
{

    //
    // HISTORY.
    //
    template <typename T, typename J>
    struct history
    {
        template <typename IMPL>
        static rocsparse_status run(buffer_layout_crtp_t<IMPL>& layout_,
                                    rocsparse_handle            handle_,
                                    rocsparse_itilu0_alg        alg_,
                                    J*                          niter_,
                                    T*                          data_,
                                    size_t                      buffer_size_,
                                    void*                       buffer_)
        {
            using layout_t = buffer_layout_crtp_t<IMPL>;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &layout_, buffer_, sizeof(IMPL), hipMemcpyDeviceToHost, handle_->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));
            void*  p_buffer      = layout_.get_pointer(layout_t::buffer);
            size_t p_buffer_size = layout_.get_size(layout_t::buffer);
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::csritilu0x_history_template<T, J>(
                handle_, alg_, niter_, data_, p_buffer_size, p_buffer)));
            return rocsparse_status_success;
        }

        static rocsparse_status run(rocsparse_handle     handle_,
                                    rocsparse_itilu0_alg alg_,
                                    J*                   niter_,
                                    T*                   data_,
                                    size_t               buffer_size_,
                                    void*                buffer_)
        {
            using layout_t = buffer_layout_contiguous_t;
            layout_t layout;
            RETURN_IF_ROCSPARSE_ERROR(
                run(layout, handle_, alg_, niter_, data_, buffer_size_, buffer_));
            return rocsparse_status_success;
        }
    };

    //
    // BUFFER SIZE.
    //
    template <typename I, typename J>
    struct buffer_size
    {
        static rocsparse_status run(rocsparse_handle     handle_,
                                    rocsparse_itilu0_alg alg_,
                                    J                    options_,
                                    J                    nsweeps_,
                                    J                    m_,
                                    I                    nnz_,
                                    const I* __restrict__ ptr_,
                                    const J* __restrict__ ind_,
                                    rocsparse_index_base base_,
                                    rocsparse_datatype   datatype_,
                                    size_t* __restrict__ buffer_size_)
        {
            size_t buffer_size = 0;
            buffer_size += buffer_layout_contiguous_t::get_sizeof_double() * sizeof(double);

            //
            // solution.
            //
            if(datatype_ == rocsparse_datatype_f32_r)
            {
                buffer_size += sizeof(float) * nnz_;
            }
            else if(datatype_ == rocsparse_datatype_f64_r)
            {
                buffer_size += sizeof(double) * nnz_;
            }
            else if(datatype_ == rocsparse_datatype_f32_c)
            {
                buffer_size += sizeof(rocsparse_float_complex) * nnz_;
            }
            else if(datatype_ == rocsparse_datatype_f64_c)
            {
                buffer_size += sizeof(rocsparse_double_complex) * nnz_;
            }

            buffer_size += sizeof(I) * 1; // lnnz
            buffer_size += sizeof(I) * (m_ + 1); // lptr
            buffer_size += sizeof(I) * 1; // unnz
            buffer_size += sizeof(I) * (m_ + 1); // uptr
            buffer_size += sizeof(J) * nnz_; // ind
            buffer_size += sizeof(I) * nnz_; // perm

            size_t buffer_size_csritilu0x = 0;
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csritilu0x_buffer_size_template(handle_,
                                                           alg_,
                                                           options_,
                                                           nsweeps_,
                                                           m_,
                                                           nnz_,
                                                           ptr_,
                                                           ptr_ + 1,
                                                           ind_,
                                                           base_,
                                                           rocsparse_diag_type_unit,
                                                           rocsparse_direction_row,
                                                           rocsparse_diag_type_unit,
                                                           rocsparse_direction_column,
                                                           datatype_,
                                                           &buffer_size_csritilu0x));

            //
            // buffer csxsldu
            //
            const size_t buffer_size_identity = sizeof(I) * nnz_;
            size_t       buffer_size_csxsldu  = 0;
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csxsldu_buffer_size_template<I, I, J>(handle_,
                                                                  rocsparse_direction_row,
                                                                  m_,
                                                                  m_,
                                                                  nnz_,
                                                                  ptr_,
                                                                  ind_,
                                                                  nullptr,
                                                                  base_,
                                                                  rocsparse_diag_type_unit,
                                                                  rocsparse_direction_row,
                                                                  rocsparse_diag_type_unit,
                                                                  rocsparse_direction_column,
                                                                  &buffer_size_csxsldu)));

            buffer_size_csxsldu += buffer_size_identity;
            //
            // buffer csritilu0x
            //
            buffer_size += rocsparse::max(buffer_size_csxsldu, buffer_size_csritilu0x);
            *buffer_size_ = buffer_size;
            return rocsparse_status_success;
        }
    };

    template <typename I, typename J>
    struct preprocess
    {
        template <typename LAYOUT_IMPL>
        static rocsparse_status run(rocsparse_handle     handle_,
                                    rocsparse_itilu0_alg alg_,
                                    J                    options_,
                                    J                    nsweeps_,
                                    J                    m_,
                                    I                    nnz_,
                                    const I* __restrict__ ptr_,
                                    const J* __restrict__ ind_,
                                    rocsparse_index_base base_,
                                    rocsparse_datatype   datatype_,
                                    size_t               buffer_size_,
                                    void* __restrict__ buffer__)

        {

            using layout_t             = LAYOUT_IMPL;
            void* __restrict__ buffer_ = buffer__;
            layout_t layout;
            layout.init(m_, nnz_, datatype_, buffer_size_, buffer_);

            I* p_lnnz = (I*)layout.get_pointer(layout_t::lnnz);
            I* p_unnz = (I*)layout.get_pointer(layout_t::unnz);
            I* p_perm = (I*)layout.get_pointer(layout_t::perm);
            I* p_lptr = (I*)layout.get_pointer(layout_t::lptr);
            I* p_uptr = (I*)layout.get_pointer(layout_t::uptr);
            J* p_ind  = (J*)layout.get_pointer(layout_t::ind);

            //
            // Cheat. (but x must be the last field before the buffer, since we don't use it at this stage )
            //
            void*  p_buffer      = layout.get_pointer(layout_t::x);
            size_t p_buffer_size = layout.get_size(layout_t::buffer) + layout.get_size(layout_t::x);

            //
            // compute lnnz and unnz.
            //
            I host_lnnz = 0;
            I host_unnz = 0;

            //
            // Set perm to identity.
            //
            I* identity = rocsparse::assign_b<I>(p_buffer_size, p_buffer, nnz_);
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::create_identity_permutation_core(handle_, nnz_, identity));

            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csxsldu_preprocess_template<I, I, J>(handle_,
                                                                 rocsparse_direction_row,
                                                                 m_,
                                                                 m_,
                                                                 nnz_,
                                                                 ptr_,
                                                                 ind_,
                                                                 identity,
                                                                 base_,
                                                                 rocsparse_diag_type_unit,
                                                                 rocsparse_direction_row,
                                                                 &host_lnnz,
                                                                 p_lptr,
                                                                 base_,
                                                                 rocsparse_diag_type_unit,
                                                                 rocsparse_direction_column,
                                                                 &host_unnz,
                                                                 p_uptr,
                                                                 base_,
                                                                 p_buffer)));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                p_lnnz, &host_lnnz, sizeof(I), hipMemcpyHostToDevice, handle_->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                p_unnz, &host_unnz, sizeof(I), hipMemcpyHostToDevice, handle_->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));

            if(nnz_ != m_ + host_lnnz + host_unnz)
            {
                std::cerr << "rocsparse_csritilu0_preprocess has detected "
                          << (m_ + host_lnnz + host_unnz - nnz_) << "/" << m_
                          << " non-existent diagonal element." << std::endl;
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_zero_pivot);
            }

            J* p_lind = p_ind;
            J* p_uind = p_ind + host_lnnz;

            //
            // Set the layout.
            //
            I* p_lval = p_perm;
            I* p_uval = p_perm + host_lnnz;
            I* p_diag = p_perm + host_lnnz + host_unnz;
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csxsldu_compute_template<I, I, J>(handle_,
                                                              rocsparse_direction_row,
                                                              m_,
                                                              m_,
                                                              nnz_,

                                                              ptr_,
                                                              ind_,
                                                              identity,
                                                              base_,
                                                              rocsparse_diag_type_unit,

                                                              rocsparse_direction_row,
                                                              host_lnnz,
                                                              p_lptr,
                                                              p_lind,
                                                              p_lval,

                                                              base_,
                                                              rocsparse_diag_type_unit,
                                                              rocsparse_direction_column,
                                                              host_unnz,
                                                              p_uptr,

                                                              p_uind,
                                                              p_uval,
                                                              base_,
                                                              p_diag,
                                                              p_buffer)));

            //
            // Free identity from the buffer.
            //
            identity = rocsparse::unassign_b<I>(p_buffer_size, p_buffer, nnz_);

            //
            // The buffer need to be kept. Get the buffer from layout since we need to preserve x.
            //
            p_buffer      = layout.get_pointer(layout_t::buffer);
            p_buffer_size = layout.get_size(layout_t::buffer);
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0x_preprocess_template(handle_,
                                                           alg_,
                                                           options_,
                                                           nsweeps_,
                                                           m_,
                                                           nnz_,
                                                           ptr_,
                                                           ptr_ + 1,
                                                           ind_,
                                                           base_,
                                                           rocsparse_diag_type_unit,
                                                           rocsparse_direction_row,
                                                           host_lnnz,
                                                           p_lptr,
                                                           p_lptr + 1,
                                                           p_lind,
                                                           base_,
                                                           rocsparse_diag_type_unit,
                                                           rocsparse_direction_column,
                                                           host_unnz,
                                                           p_uptr,
                                                           p_uptr + 1,
                                                           p_uind,
                                                           base_,
                                                           datatype_,
                                                           p_buffer_size,
                                                           p_buffer)));

            //
            // Copy the struct to device.
            //
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                buffer__, &layout, sizeof(layout_t), hipMemcpyHostToDevice, handle_->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));
            return rocsparse_status_success;
        }

        static rocsparse_status run(rocsparse_handle     handle_,
                                    rocsparse_itilu0_alg alg_,
                                    J                    options_,
                                    J                    nsweeps_,
                                    J                    m_,
                                    I                    nnz_,
                                    const I* __restrict__ ptr_,
                                    const J* __restrict__ ind_,
                                    rocsparse_index_base base_,
                                    rocsparse_datatype   datatype_,
                                    size_t               buffer_size_,
                                    void* __restrict__ buffer__)
        {
            RETURN_IF_ROCSPARSE_ERROR(run<buffer_layout_contiguous_t>(handle_,
                                                                      alg_,
                                                                      options_,
                                                                      nsweeps_,
                                                                      m_,
                                                                      nnz_,
                                                                      ptr_,
                                                                      ind_,
                                                                      base_,
                                                                      datatype_,
                                                                      buffer_size_,
                                                                      buffer__));
            return rocsparse_status_success;
        }
    };

    //
    // COMPUTE.
    //
    template <typename T, typename I, typename J>
    struct compute
    {
        static rocsparse_status run(rocsparse_handle     handle_,
                                    rocsparse_itilu0_alg alg_,
                                    rocsparse_int        options_,
                                    J*                   nsweeps_,
                                    J                    nfreeiter_,
                                    floating_data_t<T>   tol_,
                                    J                    m_,
                                    I                    nnz_,
                                    const I* __restrict__ ptr_,
                                    const J* __restrict__ ind_,
                                    const T* __restrict__ val_,
                                    T* __restrict__ sol_,
                                    rocsparse_index_base base_,
                                    size_t               buffer_size_,
                                    void* __restrict__ buffer_)
        {
            static constexpr int BLOCKSIZE_PERM = 1024;
            using layout_t                      = buffer_layout_contiguous_t;
            layout_t layout;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &layout, buffer_, sizeof(layout_t), hipMemcpyDeviceToHost, handle_->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));

            const I* p_lnnz        = (const I*)layout.get_pointer(layout_t::lnnz);
            const I* p_unnz        = (const I*)layout.get_pointer(layout_t::unnz);
            const I* p_perm        = (const I*)layout.get_pointer(layout_t::perm);
            const I* p_lptr        = (const I*)layout.get_pointer(layout_t::lptr);
            const I* p_uptr        = (const I*)layout.get_pointer(layout_t::uptr);
            const J* p_ind         = (const J*)layout.get_pointer(layout_t::ind);
            T*       p_x           = (T*)layout.get_pointer(layout_t::x);
            void*    p_buffer      = layout.get_pointer(layout_t::buffer);
            size_t   p_buffer_size = layout.get_size(layout_t::buffer);

            I host_lnnz = -1;
            I host_unnz = -1;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &host_lnnz, p_lnnz, sizeof(I), hipMemcpyDeviceToHost, handle_->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &host_unnz, p_unnz, sizeof(I), hipMemcpyDeviceToHost, handle_->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));

            const J* p_lind = p_ind;
            const J* p_uind = p_ind + host_lnnz;

            //
            // Here is the arrangement in the X vector.
            //
            T* p_lval = p_x;
            T* p_uval = p_x + host_lnnz;
            T* p_dval = p_x + host_lnnz + host_unnz;

            //
            // Copy solution_ to X, solution and X are of the same size.
            //
            rocsparse::get_permuted_array<BLOCKSIZE_PERM>(handle_, nnz_, sol_, p_x, p_perm);

            //
            // Compute expert routine.
            //
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csritilu0x_compute_template(handle_,
                                                       alg_,
                                                       options_,
                                                       nsweeps_,
                                                       nfreeiter_,
                                                       tol_,
                                                       m_,
                                                       nnz_,
                                                       ptr_,
                                                       ptr_ + 1,
                                                       ind_,
                                                       val_,
                                                       base_,
                                                       rocsparse_diag_type_unit,
                                                       rocsparse_direction_row,
                                                       host_lnnz,
                                                       p_lptr,
                                                       p_lptr + 1,
                                                       p_lind,
                                                       p_lval,
                                                       base_,
                                                       rocsparse_diag_type_unit,
                                                       rocsparse_direction_column,
                                                       host_unnz,
                                                       p_uptr,
                                                       p_uptr + 1,
                                                       p_uind,
                                                       p_uval,
                                                       base_,
                                                       p_dval,
                                                       p_buffer_size,
                                                       p_buffer));

            //
            // Move factorization to matrix.
            //
            rocsparse::set_permuted_array<BLOCKSIZE_PERM>(handle_, nnz_, sol_, p_x, p_perm);
            return rocsparse_status_success;
        }
    };
};

#define INSTANTIATE(T, I, J) \
    template struct rocsparse::csritilu0_driver_t<rocsparse_itilu0_alg_sync_split>::compute<T, I, J>

INSTANTIATE(float, rocsparse_int, rocsparse_int);
INSTANTIATE(double, rocsparse_int, rocsparse_int);
INSTANTIATE(rocsparse_float_complex, rocsparse_int, rocsparse_int);
INSTANTIATE(rocsparse_double_complex, rocsparse_int, rocsparse_int);

#undef INSTANTIATE

#define INSTANTIATE(T, J) \
    template struct rocsparse::csritilu0_driver_t<rocsparse_itilu0_alg_sync_split>::history<T, J>

INSTANTIATE(float, rocsparse_int);
INSTANTIATE(double, rocsparse_int);

#undef INSTANTIATE

#define INSTANTIATE(I, J)                                                                         \
    template struct rocsparse::csritilu0_driver_t<                                                \
        rocsparse_itilu0_alg_sync_split>::buffer_size<I, J>;                                      \
    template struct rocsparse::csritilu0_driver_t<rocsparse_itilu0_alg_sync_split>::preprocess<I, \
                                                                                               J>;

INSTANTIATE(rocsparse_int, rocsparse_int);

#undef INSTANTIATE
