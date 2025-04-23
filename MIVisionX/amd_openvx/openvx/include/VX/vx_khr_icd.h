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

/*! \file
 * \defgroup group_icd OpenVX ICD Loader API
 * \brief The OpenVX Installable Client Driver (ICD) Loader API.
 * \details The vx_khr_icd extension provides a mechanism for vendors to implement Installable Client Driver (ICD) for OpenVX. The OpenVX ICD Loader API provides a mechanism for applications to access these vendor implementations.
 */

#ifndef _VX_KHR_ICD_H_
#define _VX_KHR_ICD_H_

#include <VX/vx.h>
#include <VX/vxu.h>

/*! \brief Platform handle of an implementation.
 *  \ingroup group_icd
 */
typedef struct _vx_platform * vx_platform;

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Queries list of available platforms.
 * \param [in] capacity Maximum number of items that platform[] can hold.
 * \param [out] platform[] List of platform handles.
 * \param [out] pNumItems Number of platform handles returned.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_FAILURE If no platforms are found.
 * \ingroup group_icd
 */
VX_API_ENTRY vx_status VX_API_CALL vxIcdGetPlatforms(vx_size capacity, vx_platform platform[], vx_size * pNumItems);

/*! \brief Queries the platform for some specific information.
 * \param [in] platform The platform handle.
 * \param [in] attribute The attribute to query. Use one of the following:
 *               <tt>\ref VX_CONTEXT_VENDOR_ID</tt>,
 *               <tt>\ref VX_CONTEXT_VERSION</tt>,
 *               <tt>\ref VX_CONTEXT_EXTENSIONS_SIZE</tt>,
 *               <tt>\ref VX_CONTEXT_EXTENSIONS</tt>.
 * \param [out] ptr The location at which to store the resulting value.
 * \param [in] size The size in bytes of the container to which \a ptr points.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE If the platform is not a <tt>\ref vx_platform</tt>.
 * \retval VX_ERROR_INVALID_PARAMETERS If any of the other parameters are incorrect.
 * \retval VX_ERROR_NOT_SUPPORTED If the attribute is not supported on this implementation.
 * \ingroup group_icd
 */
VX_API_ENTRY vx_status VX_API_CALL vxQueryPlatform(vx_platform platform, vx_enum attribute, void *ptr, vx_size size);

/*! \brief Creates a <tt>\ref vx_context</tt> from a <tt>\ref vx_platform</tt>.
 * \details This creates a top-level object context for OpenVX from a platform handle.
 * \returns The reference to the implementation context <tt>\ref vx_context</tt>. Any possible errors
 * preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_icd
 */
VX_API_ENTRY vx_context VX_API_CALL vxCreateContextFromPlatform(vx_platform platform);

#ifdef __cplusplus
}
#endif

#endif
