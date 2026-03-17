/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_spvec_descr.hpp"
#include "rocsparse_utility.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief rocsparse_create_spvec_descr creates a descriptor holding the sparse
 * vector data, sizes and properties. It must be called prior to all subsequent
 * library function calls that involve sparse vectors. It should be destroyed at
 * the end using rocsparse_destroy_spvec_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_spvec_descr(rocsparse_spvec_descr* descr,
                                              int64_t                size,
                                              int64_t                nnz,
                                              void*                  indices,
                                              void*                  values,
                                              rocsparse_indextype    idx_type,
                                              rocsparse_index_base   idx_base,
                                              rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, size);
    ROCSPARSE_CHECKARG_SIZE(2, nnz);
    ROCSPARSE_CHECKARG(2, nnz, (nnz > size), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(3, nnz, indices);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, values);
    ROCSPARSE_CHECKARG_ENUM(5, idx_type);
    ROCSPARSE_CHECKARG_ENUM(6, idx_base);
    ROCSPARSE_CHECKARG_ENUM(7, data_type);

    *descr = new _rocsparse_spvec_descr;

    (*descr)->set_init(true);

    (*descr)->set_size(size);
    (*descr)->set_nnz(nnz);

    (*descr)->set_idx_data(indices);
    (*descr)->set_val_data(values);

    (*descr)->set_const_idx_data(indices);
    (*descr)->set_const_val_data(values);

    (*descr)->set_idx_type(idx_type);
    (*descr)->set_data_type(data_type);

    (*descr)->set_idx_base(idx_base);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_create_const_spvec_descr(rocsparse_const_spvec_descr* descr,
                                                    int64_t                      size,
                                                    int64_t                      nnz,
                                                    const void*                  indices,
                                                    const void*                  values,
                                                    rocsparse_indextype          idx_type,
                                                    rocsparse_index_base         idx_base,
                                                    rocsparse_datatype           data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, size);
    ROCSPARSE_CHECKARG_SIZE(2, nnz);
    ROCSPARSE_CHECKARG(2, nnz, (nnz > size), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(3, nnz, indices);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, values);
    ROCSPARSE_CHECKARG_ENUM(5, idx_type);
    ROCSPARSE_CHECKARG_ENUM(6, idx_base);
    ROCSPARSE_CHECKARG_ENUM(7, data_type);

    rocsparse_spvec_descr new_descr = new _rocsparse_spvec_descr;

    new_descr->set_init(true);

    new_descr->set_size(size);
    new_descr->set_nnz(nnz);

    new_descr->set_idx_data(nullptr);
    new_descr->set_val_data(nullptr);

    new_descr->set_const_idx_data(indices);
    new_descr->set_const_val_data(values);

    new_descr->set_idx_type(idx_type);
    new_descr->set_data_type(data_type);

    new_descr->set_idx_base(idx_base);

    *descr = new_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_destroy_spvec_descr destroys a sparse vector descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_spvec_descr(rocsparse_const_spvec_descr descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);

    if(descr->get_init() == false)
    {
        // Do nothing
        return rocsparse_status_success;
    }

    delete descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spvec_get returns the sparse vector matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_spvec_get(const rocsparse_spvec_descr descr,
                                     int64_t*                    size,
                                     int64_t*                    nnz,
                                     void**                      indices,
                                     void**                      values,
                                     rocsparse_indextype*        idx_type,
                                     rocsparse_index_base*       idx_base,
                                     rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, size);
    ROCSPARSE_CHECKARG_POINTER(2, nnz);
    ROCSPARSE_CHECKARG_POINTER(3, indices);
    ROCSPARSE_CHECKARG_POINTER(4, values);
    ROCSPARSE_CHECKARG_POINTER(5, idx_type);
    ROCSPARSE_CHECKARG_POINTER(6, idx_base);
    ROCSPARSE_CHECKARG_POINTER(7, data_type);

    *size = descr->get_size();
    *nnz  = descr->get_nnz();

    *indices = descr->get_idx_data();
    *values  = descr->get_val_data();

    *idx_type  = descr->get_idx_type();
    *idx_base  = descr->get_idx_base();
    *data_type = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_spvec_get(rocsparse_const_spvec_descr descr,
                                           int64_t*                    size,
                                           int64_t*                    nnz,
                                           const void**                indices,
                                           const void**                values,
                                           rocsparse_indextype*        idx_type,
                                           rocsparse_index_base*       idx_base,
                                           rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, size);
    ROCSPARSE_CHECKARG_POINTER(2, nnz);
    ROCSPARSE_CHECKARG_POINTER(3, indices);
    ROCSPARSE_CHECKARG_POINTER(4, values);
    ROCSPARSE_CHECKARG_POINTER(5, idx_type);
    ROCSPARSE_CHECKARG_POINTER(6, idx_base);
    ROCSPARSE_CHECKARG_POINTER(7, data_type);
    *size = descr->get_size();
    *nnz  = descr->get_nnz();

    *indices = descr->get_const_idx_data();
    *values  = descr->get_const_val_data();

    *idx_type  = descr->get_idx_type();
    *idx_base  = descr->get_idx_base();
    *data_type = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spvec_get_index_base returns the sparse vector index base.
 *******************************************************************************/
rocsparse_status rocsparse_spvec_get_index_base(rocsparse_const_spvec_descr descr,
                                                rocsparse_index_base*       idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, idx_base);

    *idx_base = descr->get_idx_base();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spvec_get_values returns the sparse vector value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_spvec_get_values(const rocsparse_spvec_descr descr, void** values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);
    *values = descr->get_val_data();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_spvec_get_values(rocsparse_const_spvec_descr descr,
                                                  const void**                values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);
    *values = descr->get_const_val_data();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spvec_set_values sets the sparse vector value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_spvec_set_values(rocsparse_spvec_descr descr, void* values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);
    descr->set_val_data(values);
    descr->set_const_val_data(values);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

#ifdef __cplusplus
}
#endif
