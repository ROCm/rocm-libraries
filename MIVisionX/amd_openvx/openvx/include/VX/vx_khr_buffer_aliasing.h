/*
 * Copyright (c) 2012-2020 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _OPENVX_BUFFER_ALIASING_H_
#define _OPENVX_BUFFER_ALIASING_H_

/*!
 * \file
 * \brief The OpenVX User Kernel Buffer Aliasing extension API.
 */

#define OPENVX_KHR_BUFFER_ALIASING  "vx_khr_buffer_aliasing"

#include <VX/vx.h>

#ifdef  __cplusplus
extern "C" {
#endif

/*! \brief Extra enums.
 *
 * \ingroup group_buffer_aliasing
 */
enum vx_buffer_aliasing_enum_e
{
    VX_ENUM_BUFFER_ALIASING_TYPE     = 0x1F, /*!< \brief Buffer aliasing type enumeration. */
};

/*! \brief Type of processing the kernel will perform on the buffer
 *
 * Indicates what type of processing the kernel will perform on the buffer. The framework may use this information to
 * arbitrate between requests.  For example, if there are two or three conflicting requests for buffer aliasing,
 * then the framework may choose to prioritize a request which gives a performance improvement as compared with one
 * that only saves memory but otherwise doesn't give a performance improvement.  For example, a kernel which performs
 * sparse processing may need to first do a buffer copy before processing if the buffers are not aliased. However a kernel
 * which performs dense processing will not need to do this.  So priority of the alias request may be given to the
 * kernel which performs sparse processing.
 *
 * \ingroup group_buffer_aliasing
 */
enum vx_buffer_aliasing_processing_type_e {

    /*! \brief Dense processing on the buffer that can be aliased
     */
    VX_BUFFER_ALIASING_PROCESSING_TYPE_DENSE = ((( VX_ID_KHRONOS ) << 20) | ( VX_ENUM_BUFFER_ALIASING_TYPE << 12)) + 0x0,

    /*! \brief Sparse processing on the buffer that can be aliased
     */
    VX_BUFFER_ALIASING_PROCESSING_TYPE_SPARSE = ((( VX_ID_KHRONOS ) << 20) | ( VX_ENUM_BUFFER_ALIASING_TYPE << 12)) + 0x1

};

/*! \brief Notifies framework that the kernel supports buffer aliasing of specified parameters
 *
 * This is intended to be called from within the vx_publish_kernels_f callback, for applicable
 * kernels in between the call to the <tt>\ref vxAddUserKernel</tt> function and the <tt>\ref vxFinalizeKernel(kernel)</tt>
 * function for the corresponding kernel.
 *
 * If a kernel can not support buffer aliasing of its parameters (for in-place processing),
 * then it should not call this function.  However, if a kernel can support buffer aliasing of
 * a pair of its parameters, then it may call this function with the appropriate parameter indices and
 * priority value.
 *
 * Note that calling this function does not guarantee that the buffers will ultimatly be aliased by
 * the framework. The framework may consider this hint as part of performance or memory optimization
 * logic along with other factors such as graph topology, other competing hints, and if the parameters
 * are virtual objects or not.
 *
 * \param [in] kernel Kernel reference
 * \param [in] parameter_index_a Index of a kernel parameter to request for aliasing
 * \param [in] parameter_index_b Index of another kernel paramter to request to alias with parameter_index_a
 * \param [in] processing_type Indicate the type of processing on this buffer from the kernel
 *              (See <tt>\ref vx_buffer_aliasing_processing_type_e</tt>)
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE kernel is not a valid reference
 * \retval VX_ERROR_INVALID_PARAMETERS parameter_index_a or parameter_index_b is NOT a valid kernel parameter index
 * \retval VX_FAILURE priority is not a supported enumeration value.
 *
 * \ingroup group_buffer_aliasing
 */
VX_API_ENTRY vx_status VX_API_CALL vxAliasParameterIndexHint(vx_kernel kernel,
                vx_uint32 parameter_index_a,
                vx_uint32 parameter_index_b,
                vx_enum processing_type);


/*! \brief Query framework if the specified parameters are aliased
 *
 * This is intended to be called from the vx_kernel_initialize_f or vx_kernel_f callback functions.
 *
 * If a kernel has called the vxAliasParameterIndexHint function during the vx_publish_kernels_f callback,
 * then vxIsParameterAliased is the function that can be called in the init or processing function to
 * query if the framework was able to alias the buffers specified.  Based on this information, the kernel
 * may execute the kernel differently.
 *
 * \param [in] node Node reference
 * \param [in] parameter_index_a Index of a kernel parameter to query for aliasing
 * \param [in] parameter_index_b Index of another kernel paramter to query to alias with parameter_index_a
 *
 * \return A <tt>\ref vx_bool</tt> value.
 * \retval vx_true_e The parameters are aliased.
 * \retval vx_false_e The parameters are not aliased.
 *
 * \ingroup group_buffer_aliasing
 */
VX_API_ENTRY vx_bool VX_API_CALL vxIsParameterAliased(vx_node node,
                vx_uint32 parameter_index_a,
                vx_uint32 parameter_index_b);


#ifdef  __cplusplus
}
#endif

#endif
