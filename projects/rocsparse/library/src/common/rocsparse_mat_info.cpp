/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_mat_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include <type_traits>

namespace
{
    template <typename INFO>
    std::shared_ptr<INFO> make_trm_info(_rocsparse_mat_info* info)
    {
        if constexpr(std::is_constructible<INFO,
                                           const rocsparse::triangular_data_t*,
                                           rocsparse::pivot_numeric_info_t*,
                                           rocsparse::pivot_numeric_info_t*>::value)
        {
            return std::make_shared<INFO>(info->get_triangular_data(),
                                          info->get_singularity_numeric_exact(),
                                          info->get_singularity_numeric_near());
        }
        else
        {
            return std::make_shared<INFO>(info->get_triangular_data(),
                                          info->get_singularity_numeric_exact());
        }
    }
}

const rocsparse::numeric_boost* _rocsparse_mat_info::get_boost() const
{
    return &this->m_boost;
}

rocsparse::numeric_boost* _rocsparse_mat_info::get_boost()
{
    return &this->m_boost;
}

const rocsparse::triangular_data_t* _rocsparse_mat_info::get_triangular_data() const
{
    return &this->m_triangular_data;
}

rocsparse::triangular_data_t* _rocsparse_mat_info::get_triangular_data()
{
    return &this->m_triangular_data;
}

rocsparse::pivot_numeric_info_t* _rocsparse_mat_info::get_singularity_numeric_exact()
{
    return this->m_singularity_numeric_exact;
}

rocsparse::pivot_numeric_info_t* _rocsparse_mat_info::get_singularity_numeric_near()
{
    return this->m_singularity_numeric_near;
}

rocsparse_csrsm_info _rocsparse_mat_info::get_csrsm_info()
{
    return this->get_shared_csrsm_info().get();
}
rocsparse_csrsv_info _rocsparse_mat_info::get_csrsv_info()
{
    return this->get_shared_csrsv_info().get();
}
rocsparse_csrilu0_info _rocsparse_mat_info::get_csrilu0_info()
{
    return this->get_shared_csrilu0_info().get();
}
rocsparse_csric0_info _rocsparse_mat_info::get_csric0_info()
{
    return this->get_shared_csric0_info().get();
}
rocsparse_csrildlt0_info _rocsparse_mat_info::get_csrildlt0_info()
{
    return this->get_shared_csrildlt0_info().get();
}
rocsparse_bsrsm_info _rocsparse_mat_info::get_bsrsm_info()
{
    return this->get_shared_bsrsm_info().get();
}
rocsparse_bsrsv_info _rocsparse_mat_info::get_bsrsv_info()
{
    return this->get_shared_bsrsv_info().get();
}
rocsparse_bsrilu0_info _rocsparse_mat_info::get_bsrilu0_info()
{
    return this->get_shared_bsrilu0_info().get();
}
rocsparse_bsric0_info _rocsparse_mat_info::get_bsric0_info()
{
    return this->get_shared_bsric0_info().get();
}

rocsparse::triangular_info_t* _rocsparse_mat_info::get_csrsm_info(rocsparse_operation operation,
                                                                  rocsparse_fill_mode fill_mode)
{
    return this->m_triangular_data.get(operation, fill_mode);
}

rocsparse::triangular_info_t* _rocsparse_mat_info::get_csrsv_info(rocsparse_operation operation,
                                                                  rocsparse_fill_mode fill_mode)
{
    return this->m_triangular_data.get(operation, fill_mode);
}

rocsparse::triangular_info_t* _rocsparse_mat_info::get_csrilu0_info(rocsparse_operation operation,
                                                                    rocsparse_fill_mode fill_mode)
{
    return this->m_triangular_data.get(operation, fill_mode);
}

rocsparse::triangular_info_t* _rocsparse_mat_info::get_csric0_info(rocsparse_operation operation,
                                                                   rocsparse_fill_mode fill_mode)
{
    return this->m_triangular_data.get(operation, fill_mode);
}

rocsparse::triangular_info_t* _rocsparse_mat_info::get_csrildlt0_info(rocsparse_operation operation,
                                                                      rocsparse_fill_mode fill_mode)
{
    return this->m_triangular_data.get(operation, fill_mode);
}

rocsparse::triangular_info_t* _rocsparse_mat_info::get_bsrsm_info(rocsparse_operation operation,
                                                                  rocsparse_fill_mode fill_mode)
{
    return this->m_triangular_data.get(operation, fill_mode);
}

rocsparse::triangular_info_t* _rocsparse_mat_info::get_bsrsv_info(rocsparse_operation operation,
                                                                  rocsparse_fill_mode fill_mode)
{
    return this->m_triangular_data.get(operation, fill_mode);
}

rocsparse::triangular_info_t* _rocsparse_mat_info::get_bsrilu0_info(rocsparse_operation operation,
                                                                    rocsparse_fill_mode fill_mode)
{
    return this->m_triangular_data.get(operation, fill_mode);
}

rocsparse::triangular_info_t* _rocsparse_mat_info::get_bsric0_info(rocsparse_operation operation,
                                                                   rocsparse_fill_mode fill_mode)
{
    return this->m_triangular_data.get(operation, fill_mode);
}

void _rocsparse_mat_info::set_bsrsm_info(rocsparse_operation           operation,
                                         rocsparse_fill_mode           fill_mode,
                                         rocsparse::triangular_info_t* trm)
{
    this->m_triangular_data.set(operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_bsrsv_info(rocsparse_operation           operation,
                                         rocsparse_fill_mode           fill_mode,
                                         rocsparse::triangular_info_t* trm)
{
    this->m_triangular_data.set(operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_bsrilu0_info(rocsparse_operation           operation,
                                           rocsparse_fill_mode           fill_mode,
                                           rocsparse::triangular_info_t* trm)
{
    this->m_triangular_data.set(operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_bsric0_info(rocsparse_operation           operation,
                                          rocsparse_fill_mode           fill_mode,
                                          rocsparse::triangular_info_t* trm)
{
    this->m_triangular_data.set(operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_csrsm_info(rocsparse_operation           operation,
                                         rocsparse_fill_mode           fill_mode,
                                         rocsparse::triangular_info_t* trm)
{
    this->m_triangular_data.set(operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_csrsv_info(rocsparse_operation           operation,
                                         rocsparse_fill_mode           fill_mode,
                                         rocsparse::triangular_info_t* trm)
{
    this->m_triangular_data.set(operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_csrilu0_info(rocsparse_operation           operation,
                                           rocsparse_fill_mode           fill_mode,
                                           rocsparse::triangular_info_t* trm)
{
    this->m_triangular_data.set(operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_csric0_info(rocsparse_operation           operation,
                                          rocsparse_fill_mode           fill_mode,
                                          rocsparse::triangular_info_t* trm)
{
    this->m_triangular_data.set(operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_csrildlt0_info(rocsparse_operation           operation,
                                             rocsparse_fill_mode           fill_mode,
                                             rocsparse::triangular_info_t* trm)
{
    this->m_triangular_data.set(operation, fill_mode, trm);
}

rocsparse_ellsv_info _rocsparse_mat_info::get_ellsv_info()
{
    return this->get_shared_ellsv_info().get();
}

rocsparse::triangular_info_t* _rocsparse_mat_info::get_ellsv_info(rocsparse_operation operation,
                                                                  rocsparse_fill_mode fill_mode)
{
    return this->m_triangular_data.get(operation, fill_mode);
}

void _rocsparse_mat_info::set_ellsv_info(rocsparse_operation           operation,
                                         rocsparse_fill_mode           fill_mode,
                                         rocsparse::triangular_info_t* trm)
{
    this->m_triangular_data.set(operation, fill_mode, trm);
}

std::shared_ptr<_rocsparse_csrsv_info> _rocsparse_mat_info::get_shared_csrsv_info()
{
    if(this->m_csrsv_info == nullptr)
    {
        this->m_csrsv_info = make_trm_info<_rocsparse_csrsv_info>(this);
    }
    return this->m_csrsv_info;
}
std::shared_ptr<_rocsparse_csrsm_info> _rocsparse_mat_info::get_shared_csrsm_info()
{
    if(this->m_csrsm_info == nullptr)
    {
        this->m_csrsm_info = make_trm_info<_rocsparse_csrsm_info>(this);
    }
    return this->m_csrsm_info;
}
std::shared_ptr<_rocsparse_csrilu0_info> _rocsparse_mat_info::get_shared_csrilu0_info()
{
    if(this->m_csrilu0_info == nullptr)
    {
        this->m_csrilu0_info = make_trm_info<_rocsparse_csrilu0_info>(this);
    }
    return this->m_csrilu0_info;
}
std::shared_ptr<_rocsparse_csric0_info> _rocsparse_mat_info::get_shared_csric0_info()
{
    if(this->m_csric0_info == nullptr)
    {
        this->m_csric0_info = make_trm_info<_rocsparse_csric0_info>(this);
    }
    return this->m_csric0_info;
}
std::shared_ptr<_rocsparse_csrildlt0_info> _rocsparse_mat_info::get_shared_csrildlt0_info()
{
    if(this->m_csrildlt0_info == nullptr)
    {
        this->m_csrildlt0_info = make_trm_info<_rocsparse_csrildlt0_info>(this);
    }
    return this->m_csrildlt0_info;
}

std::shared_ptr<_rocsparse_bsrsv_info> _rocsparse_mat_info::get_shared_bsrsv_info()
{
    if(this->m_bsrsv_info == nullptr)
    {
        this->m_bsrsv_info = make_trm_info<_rocsparse_bsrsv_info>(this);
    }
    return this->m_bsrsv_info;
}
std::shared_ptr<_rocsparse_bsrsm_info> _rocsparse_mat_info::get_shared_bsrsm_info()
{
    if(this->m_bsrsm_info == nullptr)
    {
        this->m_bsrsm_info = make_trm_info<_rocsparse_bsrsm_info>(this);
    }
    return this->m_bsrsm_info;
}
std::shared_ptr<_rocsparse_bsrilu0_info> _rocsparse_mat_info::get_shared_bsrilu0_info()
{
    if(this->m_bsrilu0_info == nullptr)
    {
        this->m_bsrilu0_info = make_trm_info<_rocsparse_bsrilu0_info>(this);
    }
    return this->m_bsrilu0_info;
}
std::shared_ptr<_rocsparse_bsric0_info> _rocsparse_mat_info::get_shared_bsric0_info()
{
    if(this->m_bsric0_info == nullptr)
    {
        this->m_bsric0_info = make_trm_info<_rocsparse_bsric0_info>(this);
    }
    return this->m_bsric0_info;
}

void _rocsparse_mat_info::clear_csrsv_info()
{
    this->m_csrsv_info.reset();
    this->release_trm_data_if_unused();
}
void _rocsparse_mat_info::clear_csrsm_info()
{
    this->m_csrsm_info.reset();
    this->release_trm_data_if_unused();
}
void _rocsparse_mat_info::clear_csrilu0_info()
{
    this->m_csrilu0_info.reset();
    this->release_trm_data_if_unused();
}
void _rocsparse_mat_info::clear_csric0_info()
{
    this->m_csric0_info.reset();
    this->release_trm_data_if_unused();
}
void _rocsparse_mat_info::clear_csrildlt0_info()
{
    this->m_csrildlt0_info.reset();
    this->release_trm_data_if_unused();
}
void _rocsparse_mat_info::clear_bsrsv_info()
{
    this->m_bsrsv_info.reset();
    this->release_trm_data_if_unused();
}
void _rocsparse_mat_info::clear_bsrsm_info()
{
    this->m_bsrsm_info.reset();
    this->release_trm_data_if_unused();
}
void _rocsparse_mat_info::clear_bsrilu0_info()
{
    this->m_bsrilu0_info.reset();
    this->release_trm_data_if_unused();
}
void _rocsparse_mat_info::clear_bsric0_info()
{
    this->m_bsric0_info.reset();
    this->release_trm_data_if_unused();
}
void _rocsparse_mat_info::clear_ellsv_info()
{
    this->m_ellsv_info.reset();
    this->release_trm_data_if_unused();
}

void _rocsparse_mat_info::release_trm_data_if_unused()
{
    if(this->m_csrsv_info == nullptr && this->m_csrsm_info == nullptr
       && this->m_csrilu0_info == nullptr && this->m_csric0_info == nullptr
       && this->m_csrildlt0_info == nullptr && this->m_bsrsv_info == nullptr
       && this->m_bsrsm_info == nullptr && this->m_bsrilu0_info == nullptr
       && this->m_bsric0_info == nullptr && this->m_ellsv_info == nullptr)
    {
        this->m_triangular_data.clear();
        this->m_diag_ind->free();
        this->m_transposed->free();
    }
}

//
// Duplicate all the triangular_info_t.
//
void _rocsparse_mat_info::duplicate_trdata(rocsparse_mat_info src, hipStream_t stream)
{
    this->m_diag_ind->copy(*src->m_diag_ind);
    this->m_transposed->copy(*src->m_transposed);
    this->m_triangular_data.copy(&src->m_triangular_data);
    THROW_IF_ROCSPARSE_ERROR(this->m_pivot_info->copy_pivot_info_async(src->m_pivot_info, stream));
    THROW_IF_ROCSPARSE_ERROR(this->m_singularity_numeric_exact->copy_singular_info_async(
        src->m_singularity_numeric_exact, stream));
    THROW_IF_ROCSPARSE_ERROR(this->m_singularity_numeric_near->copy_singular_info_async(
        src->m_singularity_numeric_near, stream));
    THROW_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));
    if(src->m_csrsv_info != nullptr)
    {
        this->get_csrsv_info();
    }
    if(src->m_csrsm_info != nullptr)
    {
        this->get_csrsm_info();
    }
    if(src->m_csrilu0_info != nullptr)
    {
        this->get_csrilu0_info();
    }
    if(src->m_csric0_info != nullptr)
    {
        this->get_csric0_info();
    }
    if(src->m_csrildlt0_info != nullptr)
    {
        this->get_csrildlt0_info();
    }
    if(src->m_bsrsv_info != nullptr)
    {
        this->get_bsrsv_info();
    }
    if(src->m_bsrsm_info != nullptr)
    {
        this->get_bsrsm_info();
    }
    if(src->m_bsrilu0_info != nullptr)
    {
        this->get_bsrilu0_info();
    }
    if(src->m_bsric0_info != nullptr)
    {
        this->get_bsric0_info();
    }
    if(src->m_ellsv_info != nullptr)
    {
        this->get_ellsv_info();
    }
}

rocsparse_csrmv_info _rocsparse_mat_info::get_csrmv_info()
{
    return this->csrmv_info;
}

void _rocsparse_mat_info::set_csrmv_info(rocsparse_csrmv_info value)
{
    this->csrmv_info = value;
}

rocsparse_bsrmv_info _rocsparse_mat_info::get_bsrmv_info()
{
    return this->bsrmv_info;
}

void _rocsparse_mat_info::set_bsrmv_info(rocsparse_bsrmv_info value)
{
    this->bsrmv_info = value;
}

rocsparse_coomv_info _rocsparse_mat_info::get_coomv_info()
{
    return this->coomv_info;
}

void _rocsparse_mat_info::set_coomv_info(rocsparse_coomv_info value)
{
    this->coomv_info = value;
}

void _rocsparse_mat_info::set_sorted_coo2csr_info(rocsparse::sorted_coo2csr_info_t* value)
{
    this->m_sorted_coo2csr_info = value;
}

rocsparse::sorted_coo2csr_info_t* _rocsparse_mat_info::get_sorted_coo2csr_info()
{
    return this->m_sorted_coo2csr_info;
}

std::shared_ptr<_rocsparse_ellsv_info> _rocsparse_mat_info::get_shared_ellsv_info()
{
    if(this->m_ellsv_info == nullptr)
    {
        this->m_ellsv_info = make_trm_info<_rocsparse_ellsv_info>(this);
    }
    return this->m_ellsv_info;
}

_rocsparse_mat_info::_rocsparse_mat_info()
    : m_singularity_numeric_exact(new rocsparse::pivot_numeric_info_t())
    , m_singularity_numeric_near(new rocsparse::pivot_numeric_info_t())
    , m_diag_ind(new rocsparse::idnvec_descr_t())
    , m_transposed(new rocsparse::transposed_sparsity_pattern_t())
    , m_pivot_info(new rocsparse::pivot_symbolic_info_t())
{
    this->m_triangular_data.set_storage(this->m_pivot_info, this->m_diag_ind, this->m_transposed);
}

rocsparse::transposed_sparsity_pattern_t* _rocsparse_mat_info::get_transposed_sparsity_pattern(
    const rocsparse::transposed_sparsity_pattern_t* transposed)
{
    return (transposed == this->m_transposed) ? this->m_transposed : nullptr;
}

_rocsparse_mat_info::~_rocsparse_mat_info()
{

    // Clear csrgemm info struct
    WARNING_IF_ROCSPARSE_ERROR(rocsparse::destroy_csrgemm_info(this->csrgemm_info));

    if(this->csritsv_info != nullptr)
    {
        delete this->csritsv_info;
        this->csritsv_info = nullptr;
    }

    // Due to the changes in the hipFree introduced in HIP 7.0
    // https://rocm.docs.amd.com/projects/HIP/en/latest/hip-7-changes.html#update-hipfree
    // we need to introduce a device synchronize here as the below hipFree calls are now asynchronous.
    // hipFree() previously had an implicit wait for synchronization purpose which is applicable for all memory allocations.
    // This wait has been disabled in the HIP 7.0 runtime for allocations made with hipMallocAsync and hipMallocFromPoolAsync.
    WARNING_IF_HIP_ERROR(rocsparse_hipDeviceSynchronize());

    //
    // TRM_INFO data are automatically destroyed.
    //
    if(this->csrmv_info != nullptr)
    {
        delete this->csrmv_info;
        this->csrmv_info = nullptr;
    }

    if(this->bsrmv_info != nullptr)
    {
        delete this->bsrmv_info;
        this->bsrmv_info = nullptr;
    }

    if(this->coomv_info != nullptr)
    {
        delete this->coomv_info;
        this->coomv_info = nullptr;
    }

    rocsparse::sorted_coo2csr_info_t* sorted_coo2csr_info = this->get_sorted_coo2csr_info();
    if(sorted_coo2csr_info != nullptr)
    {
        hipStream_t default_stream = 0;
        std::ignore                = sorted_coo2csr_info->free_memory(default_stream);

        delete sorted_coo2csr_info;
        this->set_sorted_coo2csr_info(nullptr);
    }

    delete this->m_diag_ind;
    this->m_diag_ind = nullptr;

    delete this->m_singularity_numeric_exact;
    this->m_singularity_numeric_exact = nullptr;

    delete this->m_singularity_numeric_near;
    this->m_singularity_numeric_near = nullptr;

    delete this->m_transposed;
    this->m_transposed = nullptr;

    delete this->m_pivot_info;
    this->m_pivot_info = nullptr;
}
