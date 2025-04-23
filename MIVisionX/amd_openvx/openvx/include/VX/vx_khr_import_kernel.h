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

#ifndef _OPENVX_IMPORT_KERNEL_H_
#define _OPENVX_IMPORT_KERNEL_H_

#include <VX/vx.h>

/*!
 * \file
 * \brief The OpenVX import kernel extension API.
 */
#define OPENVX_KHR_IMPORT_KERNEL  "vx_khr_import_kernel"


#ifdef  __cplusplus
extern "C" {
#endif

/*! \brief Import a kernel from binary specified by URL.
 *
 * The name of kernel parameters can be queried using the vxQueryReference API
 * with vx_parameter as ref and VX_REFERENCE_NAME as attribute.
 *
 * \param context [in] The OpenVX context
 * \param type [in] Vendor-specific identifier that indicates to the implementation
 *   how to interpret the url. For example, if an implementation can interpret the url
 *   as a file, a folder a symbolic label, or a pointer, then a vendor may choose
 *   to use "vx_<vendor>_file", "vx_<vendor>_folder", "vx_<vendor>_label", and
 *   "vx_<vendor>_pointer", respectively for this field. Container types starting
 *   with "vx_khr_" are reserved. Refer to vendor documentation for list of
 *   container types supported
 * \param url [in] URL to binary container.
 *
 * \retval On success, a valid vx_kernel object. Calling vxGetStatus with the return value
 *   as a parameter will return VX_SUCCESS if the function was successful.
 *
 * \ingroup group_import_kernel
 */
VX_API_ENTRY vx_kernel VX_API_CALL vxImportKernelFromURL(
        vx_context context,
        const vx_char * type,
        const vx_char * url
    );

#ifdef  __cplusplus
}
#endif

#endif
