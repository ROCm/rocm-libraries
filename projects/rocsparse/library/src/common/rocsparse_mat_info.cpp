/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

rocsparse_csrsm_info _rocsparse_mat_info::get_csrsm_info()
{
    return this->m_trm.create_csrsm_info();
}
rocsparse_csrsv_info _rocsparse_mat_info::get_csrsv_info()
{
    return this->m_trm.create_csrsv_info();
}
rocsparse_csrilu0_info _rocsparse_mat_info::get_csrilu0_info()
{
    return this->m_trm.create_csrilu0_info();
}
rocsparse_csric0_info _rocsparse_mat_info::get_csric0_info()
{
    return this->m_trm.create_csric0_info();
}
rocsparse_bsrsm_info _rocsparse_mat_info::get_bsrsm_info()
{
    return this->m_trm.create_bsrsm_info();
}
rocsparse_bsrsv_info _rocsparse_mat_info::get_bsrsv_info()
{
    return this->m_trm.create_bsrsv_info();
}
rocsparse_bsrilu0_info _rocsparse_mat_info::get_bsrilu0_info()
{
    return this->m_trm.create_bsrilu0_info();
}
rocsparse_bsric0_info _rocsparse_mat_info::get_bsric0_info()
{
    return this->m_trm.create_bsric0_info();
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_csrsm_info(rocsparse_operation operation,
                                                           rocsparse_fill_mode fill_mode)
{
    return this->get_csrsm_info()->get(operation, fill_mode);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_csrsv_info(rocsparse_operation operation,
                                                           rocsparse_fill_mode fill_mode)
{
    return this->get_csrsv_info()->get(operation, fill_mode);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_csrilu0_info(rocsparse_operation operation,
                                                             rocsparse_fill_mode fill_mode)
{
    return this->get_csrilu0_info()->get(operation, fill_mode);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_csric0_info(rocsparse_operation operation,
                                                            rocsparse_fill_mode fill_mode)
{
    return this->get_csric0_info()->get(operation, fill_mode);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_bsrsm_info(rocsparse_operation operation,
                                                           rocsparse_fill_mode fill_mode)
{
    return this->get_bsrsm_info()->get(operation, fill_mode);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_bsrsv_info(rocsparse_operation operation,
                                                           rocsparse_fill_mode fill_mode)
{
    return this->get_bsrsv_info()->get(operation, fill_mode);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_bsrilu0_info(rocsparse_operation operation,
                                                             rocsparse_fill_mode fill_mode)
{
    return this->get_bsrilu0_info()->get(operation, fill_mode);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_bsric0_info(rocsparse_operation operation,
                                                            rocsparse_fill_mode fill_mode)
{
    return this->get_bsric0_info()->get(operation, fill_mode);
}

void _rocsparse_mat_info::set_bsrsm_info(rocsparse_operation    operation,
                                         rocsparse_fill_mode    fill_mode,
                                         rocsparse::trm_info_t* trm)
{
    this->m_trm.create_bsrsm_info()->set(operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_bsrsv_info(rocsparse_operation    operation,
                                         rocsparse_fill_mode    fill_mode,
                                         rocsparse::trm_info_t* trm)
{
    this->m_trm.create_bsrsv_info()->set(operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_bsrilu0_info(rocsparse_operation    operation,
                                           rocsparse_fill_mode    fill_mode,
                                           rocsparse::trm_info_t* trm)
{
    this->m_trm.create_bsrilu0_info()->set(operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_bsric0_info(rocsparse_operation    operation,
                                          rocsparse_fill_mode    fill_mode,
                                          rocsparse::trm_info_t* trm)
{
    this->m_trm.create_bsric0_info()->set(operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_csrsm_info(rocsparse_operation    operation,
                                         rocsparse_fill_mode    fill_mode,
                                         rocsparse::trm_info_t* trm)
{
    this->m_trm.create_csrsm_info()->set(operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_csrsv_info(rocsparse_operation    operation,
                                         rocsparse_fill_mode    fill_mode,
                                         rocsparse::trm_info_t* trm)
{
    this->m_trm.create_csrsv_info()->set(operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_csrilu0_info(rocsparse_operation    operation,
                                           rocsparse_fill_mode    fill_mode,
                                           rocsparse::trm_info_t* trm)
{
    this->m_trm.create_csrilu0_info()->set(operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_csric0_info(rocsparse_operation    operation,
                                          rocsparse_fill_mode    fill_mode,
                                          rocsparse::trm_info_t* trm)
{
    this->m_trm.create_csric0_info()->set(operation, fill_mode, trm);
}

std::shared_ptr<_rocsparse_csrsv_info> _rocsparse_mat_info::get_shared_csrsv_info()
{
    return this->m_trm.get_shared_csrsv_info();
}
std::shared_ptr<_rocsparse_csrsm_info> _rocsparse_mat_info::get_shared_csrsm_info()
{
    return this->m_trm.get_shared_csrsm_info();
}
std::shared_ptr<_rocsparse_csrilu0_info> _rocsparse_mat_info::get_shared_csrilu0_info()
{
    return this->m_trm.get_shared_csrilu0_info();
}
std::shared_ptr<_rocsparse_csric0_info> _rocsparse_mat_info::get_shared_csric0_info()
{
    return this->m_trm.get_shared_csric0_info();
}

std::shared_ptr<_rocsparse_bsrsv_info> _rocsparse_mat_info::get_shared_bsrsv_info()
{
    return this->m_trm.get_shared_bsrsv_info();
}
std::shared_ptr<_rocsparse_bsrsm_info> _rocsparse_mat_info::get_shared_bsrsm_info()
{
    return this->m_trm.get_shared_bsrsm_info();
}
std::shared_ptr<_rocsparse_bsrilu0_info> _rocsparse_mat_info::get_shared_bsrilu0_info()
{
    return this->m_trm.get_shared_bsrilu0_info();
}
std::shared_ptr<_rocsparse_bsric0_info> _rocsparse_mat_info::get_shared_bsric0_info()
{
    return this->m_trm.get_shared_bsric0_info();
}

void _rocsparse_mat_info::clear_csrsv_info()
{
    this->m_trm.clear_csrsv_info();
}
void _rocsparse_mat_info::clear_csrsm_info()
{
    this->m_trm.clear_csrsm_info();
}
void _rocsparse_mat_info::clear_csrilu0_info()
{
    this->m_trm.clear_csrilu0_info();
}
void _rocsparse_mat_info::clear_csric0_info()
{
    this->m_trm.clear_csric0_info();
}
void _rocsparse_mat_info::clear_bsrsv_info()
{
    this->m_trm.clear_bsrsv_info();
}
void _rocsparse_mat_info::clear_bsrsm_info()
{
    this->m_trm.clear_bsrsm_info();
}
void _rocsparse_mat_info::clear_bsrilu0_info()
{
    this->m_trm.clear_bsrilu0_info();
}
void _rocsparse_mat_info::clear_bsric0_info()
{
    this->m_trm.clear_bsric0_info();
}

//
// Duplicate all the trm_info_t.
//
void _rocsparse_mat_info::duplicate_trdata(rocsparse_mat_info src, hipStream_t stream)
{
    this->m_trm.copy(src->m_trm, stream);
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

void _rocsparse_mat_info::set_sorted_coo2csr_info(rocsparse::sorted_coo2csr_info_t* value)
{
    this->m_sorted_coo2csr_info = value;
}

rocsparse::sorted_coo2csr_info_t* _rocsparse_mat_info::get_sorted_coo2csr_info()
{
    return this->m_sorted_coo2csr_info;
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
    WARNING_IF_HIP_ERROR(hipDeviceSynchronize());

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

    rocsparse::sorted_coo2csr_info_t* sorted_coo2csr_info = this->get_sorted_coo2csr_info();
    if(sorted_coo2csr_info != nullptr)
    {
        hipStream_t default_stream = 0;
        std::ignore                = sorted_coo2csr_info->free_memory(default_stream);

        delete sorted_coo2csr_info;
        this->set_sorted_coo2csr_info(nullptr);
    }
}
