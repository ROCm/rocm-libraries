/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#include "ago_internal.h"
#define VX_MAX_TENSOR_DIMENSIONS 6

static inline vx_uint32 vxComputePatchOffset(vx_uint32 x, vx_uint32 y, const vx_imagepatch_addressing_t *addr)
{
#if VX_SCALE_UNITY == (1024u)
    return ((addr->stride_y * ((addr->scale_y * y) >> 10)) +
        (addr->stride_x * ((addr->scale_x * x) >> 10)));
#else
    return ((addr->stride_y * ((addr->scale_y * y) / VX_SCALE_UNITY)) +
        (addr->stride_x * ((addr->scale_x * x) / VX_SCALE_UNITY)));
#endif
}

/*! \brief Creates a <tt>\ref vx_context</tt>.
* \details This creates a top-level object context for OpenVX.
* \note This is required to do anything else.
* \returns The reference to the implementation context.
* \retval 0 No context was created.
* \retval * A context reference.
* \ingroup group_context
* \post <tt>\ref vxReleaseContext</tt>
*/
VX_API_ENTRY vx_context VX_API_CALL vxCreateContext()
{
    vx_context context = agoCreateContextFromPlatform(nullptr);
    return context;
}

/*! \brief Releases the OpenVX object context.
* \details All reference counted objects are garbage-collected by the return of this call.
* No calls are possible using the parameter context after the context has been
* released until a new reference from <tt>\ref vxCreateContext</tt> is returned.
* All outstanding references to OpenVX objects from this context are invalid
* after this call.
* \param [in] context The pointer to the reference to the context.
* \post After returning from this function the reference is zeroed.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
* \ingroup group_context
* \pre <tt>\ref vxCreateContext</tt>
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseContext(vx_context *context)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (context && !agoReleaseContext(*context)) {
        *context = NULL;
        status = VX_SUCCESS;
    }
    return status;
}

/**
* \brief Set custom image format description.
* \ingroup vx_framework_reference
*
* This function is used to support custom image formats with single-plane by ISVs. Should be called from vxPublishKernels().
*
* \param [in] context The context.
* \param [in] format The image format.
* \param [in] desc The image format description.
* \return A \ref vx_status_e enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE if reference is not valid.
* \retval VX_ERROR_INVALID_FORMAT if format is already in use.
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetContextImageFormatDescription(vx_context context, vx_df_image format, const AgoImageFormatDescription * desc)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidContext(context)) {
        status = VX_ERROR_INVALID_FORMAT;
        if (desc->planes == 1 && !agoSetImageComponentsAndPlanes(context, format, desc->components, desc->planes, (vx_uint32)desc->pixelSizeInBitsNum, (vx_uint32)(desc->pixelSizeInBitsDenom ? desc->pixelSizeInBitsDenom : 1), desc->colorSpace, desc->channelRange)) {
            status = VX_SUCCESS;
        }
    }
    return status;
}

/**
* \brief Get custom image format description.
* \ingroup vx_framework_reference
* \param [in] context The context.
* \param [in] format The image format.
* \param [out] desc The image format description.
* \return A \ref vx_status_e enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE if reference is not valid.
* \retval VX_ERROR_INVALID_FORMAT if format is already in use.
*/
VX_API_ENTRY vx_status VX_API_CALL vxGetContextImageFormatDescription(vx_context context, vx_df_image format, AgoImageFormatDescription * desc)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidContext(context)) {
        status = VX_ERROR_INVALID_FORMAT;
        vx_uint32 pixelSizeInBitsNum, pixelSizeInBitsDenom;
        if (!agoGetImageComponentsAndPlanes(context, format, &desc->components, &desc->planes, &pixelSizeInBitsNum, &pixelSizeInBitsDenom, &desc->colorSpace, &desc->channelRange)) {
            desc->pixelSizeInBitsNum = pixelSizeInBitsNum;
            desc->pixelSizeInBitsDenom = pixelSizeInBitsDenom;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Retrieves the context from any reference from within a context.
* \param [in] reference The reference from which to extract the context.
* \ingroup group_context
* \return The overall context that created the particular
* reference.
*/
VX_API_ENTRY vx_context VX_API_CALL vxGetContext(vx_reference reference)
{
    vx_context context = NULL;
    if (agoIsValidReference(reference)) {
        context = reference->context;
    }
    return context;
}

/*! \brief Queries the context for some specific information.
* \param [in] context The reference to the context.
* \param [in] attribute The attribute to query. Use a <tt>\ref vx_context_attribute_e</tt>.
* \param [out] ptr The location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If the context is not a <tt>\ref vx_context</tt>.
* \retval VX_ERROR_INVALID_PARAMETERS If any of the other parameters are incorrect.
* \retval VX_ERROR_NOT_SUPPORTED If the attribute is not supported on this implementation.
* \ingroup group_context
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryContext(vx_context context, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidContext(context)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            CAgoLock lock(context->cs);
            switch (attribute)
            {
            case VX_CONTEXT_ATTRIBUTE_VENDOR_ID:
                if (size == sizeof(vx_uint16)) {
                    *(vx_uint16 *)ptr = VX_ID_AMD;
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONTEXT_ATTRIBUTE_VERSION:
                if (size == sizeof(vx_uint16)) {
                    *(vx_uint16 *)ptr = (vx_uint16)VX_VERSION;
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONTEXT_ATTRIBUTE_MODULES:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = (vx_uint32)context->num_active_modules;
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONTEXT_ATTRIBUTE_REFERENCES:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = (vx_uint32)context->num_active_references;
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONTEXT_ATTRIBUTE_IMPLEMENTATION:
                if (size <= VX_MAX_IMPLEMENTATION_NAME) {
                    strncpy((char *)ptr, "AMD OpenVX " AGO_VERSION, VX_MAX_IMPLEMENTATION_NAME);
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONTEXT_ATTRIBUTE_EXTENSIONS_SIZE:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = strlen(context->extensions) + 1;
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONTEXT_ATTRIBUTE_EXTENSIONS:
                if (size >= strlen(context->extensions) + 1) {
                    strcpy((char *)ptr, context->extensions);
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONTEXT_ATTRIBUTE_CONVOLUTION_MAXIMUM_DIMENSION:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = AGO_MAX_CONVOLUTION_DIM;
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONTEXT_ATTRIBUTE_NONLINEAR_MAX_DIMENSION:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = AGO_MAX_NONLINEAR_FILTER_DIM;
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONTEXT_ATTRIBUTE_OPTICAL_FLOW_WINDOW_MAXIMUM_DIMENSION:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = AGO_OPTICALFLOWPYRLK_MAX_DIM;
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONTEXT_ATTRIBUTE_IMMEDIATE_BORDER_MODE:
                if (size == sizeof(vx_border_mode_t)) {
                    *(vx_border_mode_t *)ptr = context->immediate_border_mode;
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONTEXT_ATTRIBUTE_IMMEDIATE_BORDER_POLICY:
            if (size == sizeof(vx_border_mode_policy_e)) {
                    *(vx_border_mode_policy_e *)ptr = context->immediate_border_policy;
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONTEXT_ATTRIBUTE_UNIQUE_KERNELS:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = (vx_uint32)context->kernelList.count;
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONTEXT_ATTRIBUTE_UNIQUE_KERNEL_TABLE:
                if (size == (context->kernelList.count * sizeof(vx_kernel_info_t))) {
                    vx_kernel_info_t * table = (vx_kernel_info_t *)ptr;
                    for (AgoKernel * kernel = context->kernelList.head; kernel; kernel = kernel->next, table++) {
                        table->enumeration = kernel->id;
                        strncpy(table->name, kernel->name, VX_MAX_KERNEL_NAME);
                    }
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY:
                if (size == sizeof(AgoTargetAffinityInfo_)) {
                    *(AgoTargetAffinityInfo_ *)ptr = context->attr_affinity;
                    status = VX_SUCCESS;
                }
                break;
#if ENABLE_OPENCL
            case VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT:
                if (size == sizeof(cl_context)) {
                    if (!context->opencl_context && agoGpuOclCreateContext(context, nullptr) != VX_SUCCESS) {
                        status = VX_FAILURE;
                    }
                    else {
                        *(cl_context *)ptr = context->opencl_context;
                        status = VX_SUCCESS;
                    }
                }
                else if (size == 0) {
                    // special case to just request internal context without creating one
                    // when not available
                    *(cl_context *)ptr = context->opencl_context;
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONTEXT_CL_QUEUE_PROPERTIES:
                if (size == sizeof(cl_command_queue_properties)) {
                    *(cl_command_queue_properties *)ptr = context->opencl_cmdq_properties;
                    status = VX_SUCCESS;
                }
                break;
#elif ENABLE_HIP
            case VX_CONTEXT_ATTRIBUTE_AMD_HIP_DEVICE:
                if (size == sizeof(int)) {
                    if (context->hip_device_id < 0 && agoGpuHipCreateContext(context, -1) != VX_SUCCESS) {
                        status = VX_FAILURE;
                    }
                    else {
                        *(int *)ptr = context->hip_device_id;
                        status = VX_SUCCESS;
                    }
                }
                else if (size == 0) {
                    // special case to just request internal context without creating one
                    // when not available
                    *(int *)ptr = context->hip_device_id;
                    status = VX_SUCCESS;
                }
                break;
#endif
            case VX_CONTEXT_MAX_TENSOR_DIMS:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = AGO_MAX_TENSOR_DIMENSIONS;
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Sets an attribute on the context.
* \param [in] context The handle to the overall context.
* \param [in] attribute The attribute to set from <tt>\ref vx_context_attribute_e</tt>.
* \param [in] ptr The pointer to the data to which to set the attribute.
* \param [in] size The size in bytes of the data to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If the context is not a <tt>\ref vx_context</tt>.
* \retval VX_ERROR_INVALID_PARAMETERS If any of the other parameters are incorrect.
* \retval VX_ERROR_NOT_SUPPORTED If the attribute is not settable.
* \ingroup group_context
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetContextAttribute(vx_context context, vx_enum attribute, const void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidContext(context)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        CAgoLock lock(context->cs);
        switch (attribute)
        {
        case VX_CONTEXT_ATTRIBUTE_IMMEDIATE_BORDER_MODE:
            if(!ptr) return VX_ERROR_INVALID_PARAMETERS;
            if (size == sizeof(vx_border_mode_t)) {
                vx_border_mode_t immediate_border_mode = *(vx_border_mode_t *)ptr;
                if (immediate_border_mode.mode == VX_BORDER_MODE_UNDEFINED || immediate_border_mode.mode == VX_BORDER_MODE_CONSTANT || immediate_border_mode.mode == VX_BORDER_MODE_REPLICATE) {
                    context->immediate_border_mode = immediate_border_mode;
                    if (immediate_border_mode.mode == VX_BORDER_MODE_UNDEFINED || immediate_border_mode.mode == VX_BORDER_MODE_REPLICATE)
                        memset(&context->immediate_border_mode.constant_value, 0, sizeof(context->immediate_border_mode.constant_value));
                    status = VX_SUCCESS;
                }
                else
                    status = VX_ERROR_INVALID_VALUE;
            }
            break;
        case VX_CONTEXT_ATTRIBUTE_AMD_SET_TEXT_MACRO:
            if(!ptr) return VX_ERROR_INVALID_PARAMETERS;
            if (size == sizeof(AgoContextTextMacroInfo)) {
                status = VX_SUCCESS;
                AgoContextTextMacroInfo * info = (AgoContextTextMacroInfo *)ptr;
                for (auto it = context->macros.begin(); it != context->macros.end(); ++it) {
                    if (!strcmp(it->name, info->macroName)) {
                        status = VX_FAILURE;
                        agoAddLogEntry(&context->ref, status, "ERROR: vxSetContextAttribute: macro already exists: %s\n", info->macroName);
                        break;
                    }
                }
                if (status == VX_SUCCESS) {
                    MacroData macro;
                    macro.text = macro.text_allocated = (char *)calloc(1, strlen(info->text) + 1);
                    if (!macro.text) {
                        status = VX_ERROR_NO_MEMORY;
                    }
                    else {
                        strncpy(macro.name, info->macroName, sizeof(macro.name) - 1);
                        strcpy(macro.text, info->text);
                        context->macros.push_back(macro);
                    }
                }
            }
            break;
        case VX_CONTEXT_ATTRIBUTE_AMD_SET_MERGE_RULE:
            if(!ptr) return VX_ERROR_INVALID_PARAMETERS;
            if (size == sizeof(AgoNodeMergeRule)) {
                status = VX_SUCCESS;
                context->merge_rules.push_back(*(AgoNodeMergeRule *)ptr);
            }
            break;
        case VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY:
            if(!ptr) return VX_ERROR_INVALID_PARAMETERS;
            if (size == sizeof(AgoTargetAffinityInfo_)) {
                status = VX_SUCCESS;
                context->attr_affinity = *(AgoTargetAffinityInfo_ *)ptr;
            }
            break;
#if ENABLE_OPENCL
        case VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT:
            if(!ptr) return VX_ERROR_INVALID_PARAMETERS;
            if (size == sizeof(cl_context)) {
                if (!context->opencl_context) {
                    status = agoGpuOclCreateContext(context, *(cl_context *)ptr);
                }
                else {
                    status = VX_FAILURE;
                }
            }
            break;
        case VX_CONTEXT_CL_QUEUE_PROPERTIES:
            if(!ptr) return VX_ERROR_INVALID_PARAMETERS;
            if (size == sizeof(cl_command_queue_properties)) {
                context->opencl_cmdq_properties = *(cl_command_queue_properties *)ptr;
                status = VX_SUCCESS;
            }
            break;
#elif ENABLE_HIP
        case VX_CONTEXT_ATTRIBUTE_AMD_HIP_DEVICE:
            if (size == sizeof(hipDevice_t)) {
                if (*(int *)ptr >= 0 && agoGpuHipCreateContext(context, *(int *)ptr) == VX_SUCCESS) {
                    status = VX_SUCCESS;
                }
                else {
                    status = VX_FAILURE;
                }
            }
            break;
#endif
        default:
            status = VX_ERROR_NOT_SUPPORTED;
            break;
        }
    }
    return status;
}

/*! \brief Provides a generic API to give platform-specific hints to the implementation.
* \param [in] reference The reference to the object to hint at.
* This could be <tt>\ref vx_context</tt>, <tt>\ref vx_graph</tt>, <tt>\ref vx_node</tt>, <tt>\ref vx_image</tt>, <tt>\ref vx_array</tt>, or any other reference.
* \param [in] hint A <tt>\ref vx_hint_e</tt> \a hint to give to a \ref vx_context. This is a platform-specific optimization or implementation mechanism.
* \param [in] data Optional vendor specific data.
* \param [in] data_size Size of the data structure \p data.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No error.
* \retval VX_ERROR_INVALID_REFERENCE If context or reference is invalid.
* \retval VX_ERROR_NOT_SUPPORTED If the hint is not supported.
* \ingroup group_hint
*/
VX_API_ENTRY vx_status VX_API_CALL vxHint(vx_reference reference, vx_enum hint, const void* data, vx_size data_size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidReference(reference)) {
        vx_context context = reference->context;
        if (agoIsValidContext(context)) {
            CAgoLock lock(context->cs);
            status = VX_SUCCESS;
            switch (hint)
            {
            case VX_HINT_SERIALIZE:
                reference->hint_serialize = true;
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Provides a generic API to give platform-specific directives to the implementations.
* \param [in] context The reference to the implementation context.
* \param [in] reference The reference to the object to set the directive on.
* This could be <tt>\ref vx_context</tt>, <tt>\ref vx_graph</tt>, <tt>\ref vx_node</tt>, <tt>\ref vx_image</tt>, <tt>\ref vx_array</tt>, or any other reference.
* \param [in] directive The directive to set.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No error.
* \retval VX_ERROR_INVALID_REFERENCE If context or reference is invalid.
* \retval VX_ERROR_NOT_SUPPORTED If the directive is not supported.
* \ingroup group_directive
*/
VX_API_ENTRY vx_status VX_API_CALL vxDirective(vx_reference reference, vx_enum directive)
{
    return agoDirective(reference, directive);
}

/*! \brief Provides a generic API to return status values from Object constructors if they
* fail.
* \note Users do not need to strictly check every object creator as the errors
* should properly propogate and be detected during verification time or run-time.
* \code
* vx_image img = vxCreateImage(context, 639, 480, VX_DF_IMAGE_UYVY);
* vx_status status = vxGetStatus((vx_reference)img);
* // status == VX_ERROR_INVALID_DIMENSIONS
* vxReleaseImage(&img);
* \endcode
* \pre Appropriate Object Creator function.
* \post Appropriate Object Release function.
* \param [in] reference The reference to check for construction errors.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No error.
* \retval * Some error occurred, please check enumeration list and constructor.
* \ingroup group_basic_features
*/
VX_API_ENTRY vx_status VX_API_CALL vxGetStatus(vx_reference reference)
{
    if(reference == NULL)
        return VX_ERROR_NO_RESOURCES;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidReference(reference)) {
        status = reference->status;
    }
    return status;
}

/*!
* \brief Registers user-defined structures to the context.
* \param [in] context  The reference to the implementation context.
* \param [in] size     The size of user struct in bytes.
* \return A <tt>\ref vx_enum</tt> value that is a type given to the User
* to refer to their custom structure when declaring a <tt>\ref vx_array</tt>
* of that structure.
* \retval VX_TYPE_INVALID If the namespace of types has been exhausted.
* \note This call should only be used once within the lifetime of a context for
* a specific structure.
*
* \snippet vx_arrayrange.c array define
* \ingroup group_adv_array
*/
VX_API_ENTRY vx_enum VX_API_CALL vxRegisterUserStruct(vx_context context, vx_size size)
{
    vx_enum type = VX_TYPE_INVALID;
    if (agoIsValidContext(context) && (size > 0)) {
        CAgoLock lock(context->cs);
        type = agoAddUserStruct(context, size, NULL);
    }
    return type;
}

/*!
 * \brief Registers user-defined structures to the context, and associates a name to it.
 * \param [in] context     The reference to the implementation context.
 * \param [in] size        The size of user struct in bytes.
 * \param [in] *type_name  Pointer to the '\0' terminated string that identifies the
 *                         user struct type. The string is copied by the function so
 *                         that it stays the property of the caller. NULL means that
 *                         the user struct is not named. The length of the string
 *                         shall be lower than VX_MAX_REFERENCE_NAME bytes.
 * \return A <tt>\ref vx_enum</tt> value that is a type given to the User
 * to refer to their custom structure when declaring a <tt>\ref vx_array</tt>
 * of that structure.
 * \retval VX_TYPE_INVALID If the namespace of types has been exhausted.
 * \note This call should only be used once within the lifetime of a context for
 * a specific structure.
 * \ingroup group_adv_array
 */
VX_API_ENTRY vx_enum VX_API_CALL vxRegisterUserStructWithName(vx_context context, vx_size size,  const vx_char* type_name)
{
    vx_enum type = VX_TYPE_INVALID;
    if (agoIsValidContext(context) && (size > 0)) {
        CAgoLock lock(context->cs);
        if((sizeof(char)*strlen(type_name)) < VX_MAX_REFERENCE_NAME || type_name == NULL)
            type = agoAddUserStruct(context, size, (vx_char*)type_name);
    }
    return type;
}

/*!
 * \brief Returns the name of the user-defined structure associated with the enumeration given.
 * \param [in] context     The reference to the implementation context.
 * \param [in] type_name   The enumeration value of the user struct
 * \param [out] name_size  Name of the user struct
 * \param [in] name_size   The size of allocated buffer pointed to by type_name
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS user_struct_type was valid, and name was found and returned
 * \retval VX_ERROR_INVALID_PARAMETERS user_struct_type was not a valid user struct enumeration.
 * \retval VX_ERROR_NO_MEMORY name_size is too small to hold the name of the user struct type.
 * \retval VX_FAILURE user_struct_type does not have an associated type name.
 * \pre <tt>\ref vxRegisterUserStructWithName</tt> should be called for this user struct.
 * \ingroup group_adv_array
 */
VX_API_ENTRY vx_status VX_API_CALL vxGetUserStructNameByEnum(vx_context context, vx_enum user_struct_type, vx_char* type_name, vx_size name_size)
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    if (agoIsValidContext(context) && (user_struct_type!= VX_TYPE_INVALID)) {
        status = VX_FAILURE;
        const char* temp_type_name = (agoGetUserStructName(context, user_struct_type));
        if(temp_type_name != NULL) {
            status = VX_ERROR_NO_MEMORY;
            if((strlen(temp_type_name) + 1) <= name_size) {
                memcpy(type_name, temp_type_name, strlen(temp_type_name) + 1);
                status = VX_SUCCESS;
            }
        }
    }
    return status;
}

/*!
 * \brief Returns the enum of the user-defined structure associated with the name given
 * \param [in] context           The reference to the implementation context.
 * \param [in] type_name         Pointer to the '\0' terminated string that identifies the user
 *                               struct type. The length of the string shall be lower than VX_MAX_REFERENCE_NAME bytes.
 * \param [out] user_struct_type The enumeration value of the user struct
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS type_name was valid, and enumeration was found and returned
 * \retval VX_FAILURE type_name does not match any user struct enumeration.
* \pre <tt>\ref vxRegisterUserStructWithName</tt> should be called for this user struct.
 * \ingroup group_adv_array
 */
VX_API_ENTRY vx_status VX_API_CALL vxGetUserStructEnumByName(vx_context context, const vx_char* type_name, vx_enum *user_struct_type)
{
    vx_status status = VX_FAILURE;
    if (agoIsValidContext(context) && type_name) {
        if((strlen(type_name)) < VX_MAX_REFERENCE_NAME) {
            *user_struct_type = agoGetUserStructType(context, (vx_char*)type_name);
            if(*user_struct_type != 0){
                status = VX_SUCCESS;
            }
        }
    }
    return status;
}

/*!
* \brief Allocates and registers user-defined kernel enumeration to a context.
* The allocated enumeration is from available pool of 4096 enumerations reserved
* for dynamic allocation from VX_KERNEL_BASE(VX_ID_USER,0).
* \param [in] context  The reference to the implementation context.
* \param [out] pKernelEnumId  pointer to return <tt>\ref vx_enum</tt> for user-defined kernel.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_NO_RESOURCES The enumerations has been exhausted.
* \ingroup group_user_kernels
*/
VX_API_ENTRY vx_status VX_API_CALL vxAllocateUserKernelId(vx_context context, vx_enum * pKernelEnumId)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidContext(context) && pKernelEnumId)
    {
        status = VX_ERROR_NO_RESOURCES;
        if (context->nextUserKernelId <= VX_KERNEL_MASK)
        {
            *pKernelEnumId = VX_KERNEL_BASE(VX_ID_USER, 0) + context->nextUserKernelId++;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*!
* \brief Allocates and registers user-defined kernel library ID to a context.
*
* The allocated library ID is from available pool of library IDs (1..255)
* reserved for dynamic allocation. The returned libraryId can be used by
* user-kernel library developer to specify individual kernel enum IDs in
* a header file, shown below:
* \code
* #define MY_KERNEL_ID1(libraryId) (VX_KERNEL_BASE(VX_ID_USER,libraryId) + 0);
* #define MY_KERNEL_ID2(libraryId) (VX_KERNEL_BASE(VX_ID_USER,libraryId) + 1);
* #define MY_KERNEL_ID3(libraryId) (VX_KERNEL_BASE(VX_ID_USER,libraryId) + 2);
* \endcode
* \param [in] context  The reference to the implementation context.
* \param [out] pLibraryId  pointer to <tt>\ref vx_enum</tt> for user-kernel libraryId.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_NO_RESOURCES The enumerations has been exhausted.
* \ingroup group_user_kernels
*/
VX_API_ENTRY vx_status VX_API_CALL vxAllocateUserKernelLibraryId(vx_context context, vx_enum * pLibraryId)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidContext(context) && pLibraryId)
    {
        status = VX_ERROR_NO_RESOURCES;
        if (context->nextUserLibraryId <= VX_LIBRARY(VX_LIBRARY_MASK))
        {
            *pLibraryId = context->nextUserLibraryId++;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Sets the default target of the immediate mode. Upon successful execution of this
* function any future execution of immediate mode function is attempted on the new default
* target of the context.
* \param [in] context  The reference to the implementation context.
* \param [in] target_enum  The default immediate mode target enum to be set
* to the <tt>\ref vx_context</tt> object. Use a <tt>\ref vx_target_e</tt>.
* \param [in] target_string  The target name ASCII string. This contains a valid value
* when target_enum is set to <tt>\ref VX_TARGET_STRING</tt>, otherwise it is ignored.
* \ingroup group_context
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Default target set.
* \retval VX_ERROR_INVALID_REFERENCE If the context is not a <tt>\ref vx_context</tt>.
* \retval VX_ERROR_NOT_SUPPORTED If the specified target is not supported in this context.
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetImmediateModeTarget(vx_context context, vx_enum target_enum, const char* target_string)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidContext(context))
    {
        status = VX_ERROR_NOT_SUPPORTED;
        if (target_enum == VX_TARGET_ANY) {
            status = VX_SUCCESS;
        }
        else if (target_enum == VX_TARGET_STRING) {
            if (!target_string) {
                status = VX_ERROR_INVALID_REFERENCE;
            }
            else if (!_stricmp(target_string, "any") || !_stricmp(target_string, "cpu")) {
                // only cpu mode is supported as immediate mode target
                status = VX_SUCCESS;
            }
        }
    }
    return status;
}

/*==============================================================================
IMAGE
=============================================================================*/

/*! \brief Creates an opaque reference to an image buffer.
* \details Not guaranteed to exist until the <tt>\ref vx_graph</tt> containing it has been verified.
* \param [in] context The reference to the implementation context.
* \param [in] width The image width in pixels.
* \param [in] height The image height in pixels.
* \param [in] color The VX_DF_IMAGE (<tt>\ref vx_df_image_e</tt>) code that represents the format of the image and the color space.
* \return An image reference or zero when an error is encountered.
* \see vxAccessImagePatch to obtain direct memory access to the image data.
* \ingroup group_image
*/
VX_API_ENTRY vx_image VX_API_CALL vxCreateImage(vx_context context, vx_uint32 width, vx_uint32 height, vx_df_image color)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context)) {
        CAgoLock lock(context->cs);
        char desc[128]; snprintf(desc, sizeof(desc), "image:%4.4s,%d,%d", FORMAT_STR(color), width, height);
        data = agoCreateDataFromDescription(context, NULL, desc, true);
        if (data) {
            agoGenerateDataName(context, "image", data->name);
            agoAddData(&context->dataList, data);
            // if data has children, add them too
            if (data->children) {
                for (vx_uint32 i = 0; i < data->numChildren; i++) {
                    agoAddData(&context->dataList, data->children[i]);
                }
            }
        }
    }
    return (vx_image)data;
}

/*! \brief Creates an image from another image given a rectangle. This second
* reference refers to the data in the original image. Updates to this image
* updates the parent image. The rectangle must be defined within the pixel space
* of the parent image.
* \param [in] img The reference to the parent image.
* \param [in] rect The region of interest rectangle. Must contain points within
* the parent image pixel space.
* \return The reference to the sub-image or zero if the rectangle is invalid.
* \ingroup group_image
*/
VX_API_ENTRY vx_image VX_API_CALL vxCreateImageFromROI(vx_image img, const vx_rectangle_t *rect)
{
    AgoData * master_img = (AgoData *)img;
    AgoData * data = NULL;
    if (agoIsValidData(master_img, VX_TYPE_IMAGE)) {
        vx_context context = master_img->ref.context;
        CAgoLock lock(context->cs);
        char desc[128]; snprintf(desc, sizeof(desc), "image-roi:%s,%d,%d,%d,%d", master_img->name.c_str(), rect->start_x, rect->start_y, rect->end_x, rect->end_y);
        data = agoCreateDataFromDescription(context, NULL, desc, true);
        if (data) {
            agoGenerateDataName(context, "image-roi", data->name);
            agoAddData(&context->dataList, data);
            // if data has children, add them too
            if (data->children) {
                for (vx_uint32 i = 0; i < data->numChildren; i++) {
                    agoAddData(&context->dataList, data->children[i]);
                }
            }
        }
    }
    return (vx_image)data;
}

/*! \brief Creates a reference to an image object that has a singular,
* uniform value in all pixels.
* \details The value pointer must reflect the specific format of the desired
* image. For example:
* | Color       | Value Ptr  |
* |:------------|:-----------|
* | <tt>\ref VX_DF_IMAGE_U8</tt>   | vx_uint8 * |
* | <tt>\ref VX_DF_IMAGE_S16</tt>  | vx_int16 * |
* | <tt>\ref VX_DF_IMAGE_U16</tt>  | vx_uint16 *|
* | <tt>\ref VX_DF_IMAGE_S32</tt>  | vx_int32 * |
* | <tt>\ref VX_DF_IMAGE_U32</tt>  | vx_uint32 *|
* | <tt>\ref VX_DF_IMAGE_RGB</tt>  | vx_uint8 pixel[3] in R, G, B order |
* | <tt>\ref VX_DF_IMAGE_RGBX</tt> | vx_uint8 pixels[4] |
* | Any YUV     | vx_uint8 pixel[3] in Y, U, V order |
*
* \param [in] context The reference to the implementation context.
* \param [in] width The image width in pixels.
* \param [in] height The image height in pixels.
* \param [in] color The VX_DF_IMAGE (\ref vx_df_image_e) code that represents the format of the image and the color space.
* \param [in] value The pointer to the pixel value to which to set all pixels.
* \return An image reference or zero when an error is encountered.
* <tt>\see vxAccessImagePatch</tt> to obtain direct memory access to the image data.
* \note <tt>\ref vxAccessImagePatch</tt> and <tt>\ref vxCommitImagePatch</tt> may be called with
* a uniform image reference.
* \ingroup group_image
*/
VX_API_ENTRY vx_image VX_API_CALL vxCreateUniformImage(vx_context context, vx_uint32 width, vx_uint32 height, vx_df_image color, const vx_pixel_value_t *value)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context)) {
        CAgoLock lock(context->cs);
        char desc[128];
        if (color == VX_DF_IMAGE_U8) {
            snprintf(desc, sizeof(desc), "image-uniform:%4.4s,%d,%d,%d", FORMAT_STR(color), width, height, value->U8);
        }
        else if (color == VX_DF_IMAGE_S16) {
            snprintf(desc, sizeof(desc), "image-uniform:%4.4s,%d,%d,%d", FORMAT_STR(color), width, height, value->S16);
        }
        else if (color == VX_DF_IMAGE_U16) {
            snprintf(desc, sizeof(desc), "image-uniform:%4.4s,%d,%d,%d", FORMAT_STR(color), width, height, value->U16);
        }
        else if (color == VX_DF_IMAGE_S32) {
            snprintf(desc, sizeof(desc), "image-uniform:%4.4s,%d,%d,%d", FORMAT_STR(color), width, height, value->S32);
        }
        else if (color == VX_DF_IMAGE_U32) {
            snprintf(desc, sizeof(desc), "image-uniform:%4.4s,%d,%d,%u", FORMAT_STR(color), width, height, value->U32);
        }
        else {
            snprintf(desc, sizeof(desc), "image-uniform:%4.4s,%d,%d,%d,%d,%d,%d", FORMAT_STR(color), width, height, value->reserved[0], value->reserved[1], value->reserved[2], value->reserved[3]);
        }
        data = agoCreateDataFromDescription(context, NULL, desc, true);
        if (data) {
            agoGenerateDataName(context, "image-uniform", data->name);
            agoAddData(&context->dataList, data);
            // if data has children, add them too
            if (data->children) {
                for (vx_uint32 i = 0; i < data->numChildren; i++) {
                    agoAddData(&context->dataList, data->children[i]);
                }
            }
        }
    }
    return (vx_image)data;
}

/*! \brief Creates an opaque reference to an image buffer with no direct
* user access. This function allows setting the image width, height, or format.
* \details Virtual data objects allow users to connect various nodes within a
* graph via data references without access to that data, but they also permit the
* implementation to take maximum advantage of possible optimizations. Use this
* API to create a data reference to link two or more nodes together when the
* intermediate data are not required to be accessed by outside entities. This API
* in particular allows the user to define the image format of the data without
* requiring the exact dimensions. Virtual objects are scoped within the graph
* they are declared a part of, and can't be shared outside of this scope.
* All of the following constructions of virtual images are valid.
* \code
* vx_context context = vxCreateContext();
* vx_graph graph = vxCreateGraph(context);
* vx_image virt[] = {
*     vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8), // no specified dimension
*     vxCreateVirtualImage(graph, 320, 240, VX_DF_IMAGE_VIRT), // no specified format
*     vxCreateVirtualImage(graph, 640, 480, VX_DF_IMAGE_U8), // no user access
* };
* \endcode
* \param [in] graph The reference to the parent graph.
* \param [in] width The width of the image in pixels. A value of zero informs the interface that the value is unspecified.
* \param [in] height The height of the image in pixels. A value of zero informs the interface that the value is unspecified.
* \param [in] color The VX_DF_IMAGE (<tt>\ref vx_df_image_e</tt>) code that represents the format of the image and the color space. A value of <tt>\ref VX_DF_IMAGE_VIRT</tt> informs the interface that the format is unspecified.
* \return An image reference or zero when an error is encountered.
* \note Passing this reference to <tt>\ref vxAccessImagePatch</tt> will return an error.
* \ingroup group_image
*/
VX_API_ENTRY vx_image VX_API_CALL vxCreateVirtualImage(vx_graph graph, vx_uint32 width, vx_uint32 height, vx_df_image color)
{
    AgoData * data = NULL;
    if (agoIsValidGraph(graph)) {
        vx_context context = graph->ref.context;
        CAgoLock lock(graph->cs);
        char desc[128]; snprintf(desc, sizeof(desc), "image-virtual:%4.4s,%d,%d", FORMAT_STR(color), width, height);
        data = agoCreateDataFromDescription(context, graph, desc, true);
        if (data) {
            agoGenerateVirtualDataName(graph, "image", data->name);
            agoAddData(&graph->dataList, data);
            // if data has children, add them too
            if (data->children) {
                for (vx_uint32 i = 0; i < data->numChildren; i++) {
                    agoAddData(&graph->dataList, data->children[i]);
                }
            }
        }
    }
    return (vx_image)data;
}

/*! \brief Creates a reference to an image object that was externally allocated.
* \param [in] context The reference to the implementation context.
* \param [in] color See the <tt>\ref vx_df_image_e</tt> codes. This mandates the
* number of planes needed to be valid in the \a addrs and \a ptrs arrays based on the format given.
* \param [in] addrs[] The array of image patch addressing structures that
* define the dimension and stride of the array of pointers. See note below.
* \param [in] ptrs[] The array of platform-defined references to each plane. See note below.
* \param [in] memory_type <tt>\ref vx_memory_type_e</tt>. When giving <tt>\ref VX_MEMORY_TYPE_HOST</tt>
* the \a ptrs array is assumed to be HOST accessible pointers to memory.
* \returns An image reference <tt>\ref vx_image</tt>. Any possible errors preventing a
* successful creation should be checked using <tt>\ref vxGetStatus</tt>.
* \note The user must call vxMapImagePatch prior to accessing the pixels of an image, even if the
* image was created via <tt>\ref vxCreateImageFromHandle</tt>. Reads or writes to memory referenced
* by ptrs[ ] after calling <tt>\ref vxCreateImageFromHandle</tt> without first calling
* <tt>\ref vxMapImagePatch</tt> will result in undefined behavior.
* The property of addr[] and ptrs[] arrays is kept by the caller (It means that the implementation will
* make an internal copy of the provided information. \a addr and \a ptrs can then simply be application's
* local variables).
* Only \a dim_x, \a dim_y, \a stride_x and \a stride_y fields of the <tt>\ref vx_imagepatch_addressing_t</tt> need to be
* provided by the application. Other fields (\a step_x, \a step_y, \a scale_x & \a scale_y) are ignored by this function.
* The layout of the imported memory must follow a row-major order. In other words, \a stride_x should be
* sufficiently large so that there is no overlap between data elements corresponding to different
* pixels, and \a stride_y >= \a stride_x * \a dim_x.
*
* In order to release the image back to the application we should use <tt>\ref vxSwapImageHandle</tt>.
*
* Import type of the created image is available via the image attribute <tt>\ref vx_image_attribute_e</tt> parameter.
*
* \ingroup group_image
*/
VX_API_ENTRY vx_image VX_API_CALL vxCreateImageFromHandle(vx_context context, vx_df_image color, const vx_imagepatch_addressing_t addrs[], void *const ptrs[], vx_enum memory_type)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context)) {
        if (memory_type == VX_MEMORY_TYPE_HOST) {
            char desc[128]; snprintf(desc, sizeof(desc), "image:%4.4s,%d,%d", FORMAT_STR(color), addrs[0].dim_x, addrs[0].dim_y);
            data = agoCreateDataFromDescription(context, NULL, desc, true);
            if (data) {
                agoGenerateDataName(context, "image-host", data->name);
                agoAddData(&context->dataList, data);
                // if data has children, add them too
                if (data->children) {
                    for (vx_uint32 i = 0; i < data->numChildren; i++) {
                        agoAddData(&context->dataList, data->children[i]);
                    }
                }
                // set host allocated pointers
                if (data->children) {
                    data->import_type = VX_MEMORY_TYPE_HOST;
                    for (vx_uint32 i = 0; i < data->numChildren; i++) {
                        data->children[i]->import_type = VX_MEMORY_TYPE_HOST;
                        data->children[i]->buffer = (vx_uint8 *)(ptrs ? ptrs[i] : nullptr);
                        data->children[i]->u.img.stride_in_bytes = addrs[i].stride_y;
                        data->children[i]->gpu_buffer_offset = 0;
#if (ENABLE_OPENCL || ENABLE_HIP)
                        data->children[i]->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                        data->children[i]->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
#endif
                    }
                }
                else {
                    data->import_type = VX_MEMORY_TYPE_HOST;
                    data->buffer = (vx_uint8 *)(ptrs ? ptrs[0] : nullptr);
                    data->u.img.stride_in_bytes = addrs[0].stride_y;
                    data->gpu_buffer_offset = 0;
#if (ENABLE_OPENCL || ENABLE_HIP)
                    data->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                    data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
#endif
                }
                data->u.img.mem_handle = vx_false_e;
            }
        }
#if ENABLE_OPENCL
        else if (memory_type == VX_MEMORY_TYPE_OPENCL) {
            char desc[128]; snprintf(desc, sizeof(desc), "image:%4.4s,%d,%d", FORMAT_STR(color), addrs[0].dim_x, addrs[0].dim_y);
            data = agoCreateDataFromDescription(context, NULL, desc, true);
            if (data) {
                agoGenerateDataName(context, "image-opencl", data->name);
                agoAddData(&context->dataList, data);
                // if data has children, add them too
                if (data->children) {
                    for (vx_uint32 i = 0; i < data->numChildren; i++) {
                        agoAddData(&context->dataList, data->children[i]);
                    }
                }
                // set host allocated pointers
                if (data->children) {
                    data->import_type = VX_MEMORY_TYPE_OPENCL;
                    for (vx_uint32 i = 0; i < data->numChildren; i++) {
                        data->children[i]->import_type = VX_MEMORY_TYPE_OPENCL;
                        data->children[i]->opencl_buffer = (cl_mem)(ptrs ? ptrs[i] : nullptr);
                        data->children[i]->gpu_buffer_offset = 0;
                        data->children[i]->u.img.stride_in_bytes = addrs[i].stride_y;
                        data->children[i]->gpu_buffer_offset = 0;
                    }
                }
                else {
                    data->import_type = VX_MEMORY_TYPE_OPENCL;
                    data->opencl_buffer = (cl_mem)(ptrs ? ptrs[0] : nullptr);
                    data->u.img.stride_in_bytes = addrs[0].stride_y;
                    data->gpu_buffer_offset = 0;
                }
                data->u.img.mem_handle = vx_false_e;
            }
        }
#elif ENABLE_HIP
        else if (memory_type == VX_MEMORY_TYPE_HIP) {
            char desc[128]; snprintf(desc, sizeof(desc), "image:%4.4s,%d,%d", FORMAT_STR(color), addrs[0].dim_x, addrs[0].dim_y);
            data = agoCreateDataFromDescription(context, NULL, desc, true);
            if (data) {
                agoGenerateDataName(context, "image-hip", data->name);
                agoAddData(&context->dataList, data);
                // if data has children, add them too
                if (data->children) {
                    for (vx_uint32 i = 0; i < data->numChildren; i++) {
                        agoAddData(&context->dataList, data->children[i]);
                    }
                }
                // set host allocated pointers
                if (data->children) {
                    data->import_type = VX_MEMORY_TYPE_HIP;
                    for (vx_uint32 i = 0; i < data->numChildren; i++) {
                        data->children[i]->import_type = VX_MEMORY_TYPE_HIP;
                        data->children[i]->hip_memory = (vx_uint8 *)(ptrs ? ptrs[i] : nullptr);
                        data->children[i]->gpu_buffer_offset = 0;
                        data->children[i]->u.img.stride_in_bytes = addrs[i].stride_y;
                        data->children[i]->gpu_buffer_offset = 0;
                    }
                }
                else {
                    data->import_type = VX_MEMORY_TYPE_HIP;
                    data->hip_memory = (vx_uint8 *)(ptrs ? ptrs[0] : nullptr);
                    data->u.img.stride_in_bytes = addrs[0].stride_y;
                    data->gpu_buffer_offset = 0;
                }
            }
        }
#endif
    }
    return (vx_image)data;
}

/*! \brief Swaps the image handle of an image previously created from handle.
*
* This function sets the new image handle (i.e. pointer to all image planes)
* and returns the previous one.
*
* Once this function call has completed, the application gets back the
* ownership of the memory referenced by the previous handle. This memory
* contains up-to-date pixel data, and the application can safely reuse or
* release it.
*
* The memory referenced by the new handle must have been allocated
* consistently with the image properties since the import type,
* memory layout and dimensions are unchanged (see addrs, color, and
* memory_type in <tt>\ref vxCreateImageFromHandle</tt>).
*
* All images created from ROI with this image as parent or ancestor
* will automatically use the memory referenced by the new handle.
*
* The behavior of SwapImageHandle when called from a user node is undefined.
* \param [in] image The reference to an image created from handle
* \param [in] new_ptrs[] pointer to a caller owned array that contains
* the new image handle (image plane pointers)
* \arg new_ptrs is non NULL. new_ptrs[i] must be non NULL for each i such as
* 0 < i < nbPlanes, otherwise, this is an error. The address of the storage memory
* for image plane i is set to new_ptrs[i]
* \arg new_ptrs is NULL: the previous image storage memory is reclaimed by the
* caller, while no new handle is provided.
* \param [out] prev_ptrs[] pointer to a caller owned array in which
* the application returns the previous image handle
* \arg prev_ptrs is non NULL. prev_ptrs must have at least as many
* elements as the number of image planes. For each i such as
* 0 < i < nbPlanes , prev_ptrs[i] is set to the address of the previous storage
* memory for plane i.
* \arg prev_ptrs NULL : the previous handle is not returned.
* \param [in] num_planes Number of planes in the image. This must be set equal to the number of planes of the input image.
*  The number of elements in new_ptrs and prev_ptrs arrays must be equal to or greater than num_planes.
* If either array has more than num_planes elements, the extra elements are ignored. If either array is smaller
* than num_planes, the results are undefined.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_INVALID_REFERENCE image is not a valid image
* reference.
* \retval VX_ERROR_INVALID_PARAMETERS The image was not created from handle or
* the content of new_ptrs is not valid.
* \retval VX_FAILURE The image was already being accessed.
* \ingroup group_image
*/
VX_API_ENTRY vx_status VX_API_CALL vxSwapImageHandle(vx_image image_, void* const new_ptrs[], void* prev_ptrs[], vx_size num_planes)
{
    AgoData * image = (AgoData *)image_;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(image, VX_TYPE_IMAGE)) {
        CAgoLock lock(image->ref.context->cs);
        status = VX_ERROR_INVALID_PARAMETERS;
        if(image->u.img.roiMasterImage) {
            return status;
        }
        if (image->import_type == VX_MEMORY_TYPE_HOST && num_planes == image->u.img.planes) {
            status = VX_SUCCESS;
            if (image->children) {
                for (vx_uint32 i = 0; i < image->numChildren; i++) {
                    if (prev_ptrs) prev_ptrs[i] = image->children[i]->buffer;
                    image->children[i]->buffer = (vx_uint8 *)(new_ptrs ? new_ptrs[i] : nullptr);
                    if (image->children[i]->buffer) {
                        image->children[i]->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                        image->children[i]->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
                    }
                    // propagate to ROIs
                    for (auto roi = image->children[i]->roiDepList.begin(); roi != image->children[i]->roiDepList.end(); roi++) {
                        if(image->buffer) {
                            (*roi)->buffer = image->children[i]->buffer +
                                (*roi)->children[i]->u.img.rect_roi.start_y * (*roi)->children[i]->u.img.stride_in_bytes +
                                ImageWidthInBytesFloor((*roi)->children[i]->u.img.rect_roi.start_x, (*roi)->children[i]);
                        }
                        else {
                            (*roi)->buffer = nullptr;
                        }
                    }
                    image->children[i]->u.img.mem_handle = (new_ptrs == NULL) ?  vx_true_e : vx_false_e;
                }
            }
            else {
                if (prev_ptrs) prev_ptrs[0] = image->buffer;
                image->buffer = (vx_uint8 *)(new_ptrs ? new_ptrs[0] : nullptr);
                if (image->buffer) {
                    image->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                    image->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
                }
                // propagate to ROIs
                for (auto roi = image->roiDepList.begin(); roi != image->roiDepList.end(); roi++) {
                    if(image->buffer) {
                        (*roi)->buffer = image->buffer +
                            (*roi)->u.img.rect_roi.start_y * image->u.img.stride_in_bytes +
                            ImageWidthInBytesFloor((*roi)->u.img.rect_roi.start_x, image);
                    }
                    else {
                        (*roi)->buffer = nullptr;
                    }
                }
                image->u.img.mem_handle = (new_ptrs == NULL) ?  vx_true_e : vx_false_e;
            }
        }
#if ENABLE_OPENCL
        else if (image->import_type == VX_MEMORY_TYPE_OPENCL && num_planes == image->u.img.planes) {
            status = VX_SUCCESS;
            if (image->children) {
                for (vx_uint32 i = 0; i < image->numChildren; i++) {
                    if (prev_ptrs) prev_ptrs[i] = image->children[i]->opencl_buffer;
                    image->children[i]->opencl_buffer = (cl_mem)(new_ptrs ? new_ptrs[i] : nullptr);
                    if (image->children[i]->opencl_buffer) {
                        image->children[i]->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                        image->children[i]->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL;
                    }
                    // propagate to ROIs
                    for (auto roi = image->children[i]->roiDepList.begin(); roi != image->children[i]->roiDepList.end(); roi++) {
                        (*roi)->opencl_buffer = image->children[i]->opencl_buffer;
                    }
                    image->children[i]->u.img.mem_handle = (new_ptrs == NULL) ?  vx_true_e : vx_false_e;
                }
            }
            else {
                if (prev_ptrs) prev_ptrs[0] = image->opencl_buffer;
                image->opencl_buffer = (cl_mem)(new_ptrs ? new_ptrs[0] : nullptr);
                if (image->opencl_buffer) {
                    image->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                    image->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL;
                }
                // propagate to ROIs
                for (auto roi = image->roiDepList.begin(); roi != image->roiDepList.end(); roi++) {
                    (*roi)->opencl_buffer = image->opencl_buffer;
                }
                image->u.img.mem_handle = (new_ptrs == NULL) ?  vx_true_e : vx_false_e;
            }
        }
#elif ENABLE_HIP
        else if (image->import_type == VX_MEMORY_TYPE_HIP && num_planes == image->u.img.planes) {
            status = VX_SUCCESS;
            if (image->children) {
                for (vx_uint32 i = 0; i < image->numChildren; i++) {
                    if (prev_ptrs) prev_ptrs[i] = (void *)image->children[i]->hip_memory;
                    image->children[i]->hip_memory = (vx_uint8 *)(new_ptrs ? new_ptrs[i] : nullptr);
                    if (image->children[i]->hip_memory) {
                        image->children[i]->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                        image->children[i]->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL;
                    }
                    // propagate to ROIs
                    for (auto roi = image->children[i]->roiDepList.begin(); roi != image->children[i]->roiDepList.end(); roi++) {
                        (*roi)->hip_memory = image->children[i]->hip_memory;
                    }
                }
            }
            else {
                if (prev_ptrs) prev_ptrs[0] = image->hip_memory;
                image->hip_memory = (vx_uint8 *)(new_ptrs ? new_ptrs[0] : nullptr);
                if (image->hip_memory) {
                    image->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                    image->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL;
                }
                // propagate to ROIs
                for (auto roi = image->roiDepList.begin(); roi != image->roiDepList.end(); roi++) {
                    (*roi)->hip_memory = image->hip_memory;
                }
            }
        }
#endif
    }
    return status;
}

/*! \brief Retrieves various attributes of an image.
* \param [in] image The reference to the image to query.
* \param [in] attribute The attribute to query. Use a <tt>\ref vx_image_attribute_e</tt>.
* \param [out] ptr The location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If the image is not a <tt>\ref vx_image</tt>.
* \retval VX_ERROR_INVALID_PARAMETERS If any of the other parameters are incorrect.
* \retval VX_ERROR_NOT_SUPPORTED If the attribute is not supported on this implementation.
* \ingroup group_image
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryImage(vx_image image_, vx_enum attribute, void *ptr, vx_size size)
{
    AgoData * image = (AgoData *)image_;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(image, VX_TYPE_IMAGE)) {
        CAgoLock lock(image->ref.context->cs);
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_IMAGE_WIDTH:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = image->u.img.width;
                    status = VX_SUCCESS;
                }
                break;
            case VX_IMAGE_HEIGHT:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = image->u.img.height;
                    status = VX_SUCCESS;
                }
                break;
            case VX_IMAGE_FORMAT:
                if (size == sizeof(vx_uint32)) {
                    *(vx_df_image *)ptr = image->u.img.format;
                    status = VX_SUCCESS;
                }
                break;
            case VX_IMAGE_PLANES:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = image->u.img.planes;
                    status = VX_SUCCESS;
                }
                break;
            case VX_IMAGE_SPACE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = image->u.img.color_space;
                    status = VX_SUCCESS;
                }
                break;
            case VX_IMAGE_RANGE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = image->u.img.channel_range;
                    status = VX_SUCCESS;
                }
                break;
            case VX_IMAGE_MEMORY_TYPE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = image->import_type;
                    status = VX_SUCCESS;
                }
                break;
            case VX_IMAGE_SIZE:
                if (size == sizeof(vx_size)) {
                    status = VX_SUCCESS;
                    if (image->numChildren) {
                        size = 0;
                        for (vx_uint32 plane = 0; plane < image->u.img.planes; plane++) {
                            if (!image->children[plane]->size) {
                                if (image->children[plane]->isNotFullyConfigured || agoDataSanityCheckAndUpdate(image->children[plane])) {
                                    status = VX_ERROR_INVALID_REFERENCE;
                                }
                            }
                            size += image->children[plane]->size;
                        }
                        if (status == VX_SUCCESS)
                            *(vx_size *)ptr = size;
                    }
                    else {
                        if (!image->size) {
                            if (image->isNotFullyConfigured || agoDataSanityCheckAndUpdate(image)) {
                                status = VX_ERROR_INVALID_REFERENCE;
                            }
                        }
                        if (status == VX_SUCCESS)
                            *(vx_size *)ptr = image->size;
                    }
                }
                break;
#if ENABLE_OPENCL
            case VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER:
                if (size == sizeof(cl_mem)) {
                    if (image->opencl_buffer) {
                        *(cl_mem *)ptr = image->opencl_buffer;
                    }
                    else {
#if defined(CL_VERSION_2_0)
                        *(vx_uint8 **)ptr = image->opencl_svm_buffer;
#else
                        *(vx_uint8 **)ptr = NULL;
#endif
                    }
                    status = VX_SUCCESS;
                }
                break;
            case VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_OFFSET:
                if (size == sizeof(cl_uint)) {
                    *(cl_uint *)ptr = image->gpu_buffer_offset;
                    status = VX_SUCCESS;
                }
                break;
            case VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_STRIDE:
                if (size == sizeof(cl_uint)) {
                    *(cl_uint *)ptr = image->u.img.stride_in_bytes;
                    status = VX_SUCCESS;
                }
                break;
#elif ENABLE_HIP
            case VX_IMAGE_ATTRIBUTE_AMD_HIP_BUFFER:
                if (image->hip_memory) {
                    *(vx_uint8 **)ptr = image->hip_memory;
                }
                else {
                    *(vx_uint8 **)ptr = NULL;
                }
                status = VX_SUCCESS;
                break;
            case VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_OFFSET:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = image->gpu_buffer_offset;
                    status = VX_SUCCESS;
                }
                break;
            case VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_STRIDE:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = image->u.img.stride_in_bytes;
                    status = VX_SUCCESS;
                }
                break;
#endif
#if (ENABLE_OPENCL || ENABLE_HIP)
            case VX_IMAGE_ATTRIBUTE_AMD_ENABLE_USER_BUFFER_GPU:
                if (size == sizeof(vx_bool)) {
                    *(vx_bool *)ptr = image->u.img.enableUserBufferGPU;
                    status = VX_SUCCESS;
                }
                break;
#endif
            case VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER:
                if (size == sizeof(vx_uint8)) {
                    if (image->buffer) {
                        vx_uint8** temp  = static_cast< vx_uint8 **>(ptr);
                        *temp = image->buffer;
                        status = VX_SUCCESS;
                    }
                }
                break;

            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Allows setting attributes on the image.
* \param [in] image The reference to the image on which to set the attribute.
* \param [in] attribute The attribute to set. Use a <tt>\ref vx_image_attribute_e</tt> enumeration.
* \param [in] out The pointer to the location from which to read the value.
* \param [in] size The size of the object pointed to by \a out.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If the image is not a <tt>\ref vx_image</tt>.
* \retval VX_ERROR_INVALID_PARAMETERS If any of the other parameters are incorrect.
* \ingroup group_image
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetImageAttribute(vx_image image_, vx_enum attribute, const void *ptr, vx_size size)
{
    AgoData * image = (AgoData *)image_;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(image, VX_TYPE_IMAGE)) {
        CAgoLock lock(image->ref.context->cs);
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_IMAGE_ATTRIBUTE_SPACE:
                if (size == sizeof(vx_enum)) {
                    image->u.img.color_space = *(vx_color_space_e *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_IMAGE_ATTRIBUTE_RANGE:
                if (size == sizeof(vx_enum)) {
                    image->u.img.channel_range = *(vx_channel_range_e *)ptr;
                    status = VX_SUCCESS;
                }
                break;
#if ENABLE_OPENCL
            case VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER:
                if (size == sizeof(cl_mem) && image->u.img.enableUserBufferGPU) {
                    image->opencl_buffer = *(cl_mem *)ptr;
                    if (image->opencl_buffer) {
                        image->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                        image->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL;
                    }
                    status = VX_SUCCESS;
                }
                break;

            case VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_OFFSET:
                if (size == sizeof(cl_uint) && image->u.img.enableUserBufferGPU) {
                    image->gpu_buffer_offset = *(cl_uint *)ptr;
                    status = VX_SUCCESS;
                }
                break;
#elif ENABLE_HIP
            case VX_IMAGE_ATTRIBUTE_AMD_HIP_BUFFER:
                if (image->u.img.enableUserBufferGPU) {
                    image->hip_memory = *(vx_uint8 **)ptr;
                    if (image->hip_memory) {
                        image->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                        image->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL;
                    }
                    status = VX_SUCCESS;
                }
                break;

            case VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_OFFSET:
                if (size == sizeof(vx_uint32) && image->u.img.enableUserBufferGPU) {
                    image->gpu_buffer_offset = *(vx_uint32 *)ptr;
                    status = VX_SUCCESS;
                }
                break;

#endif

            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Initialize an image with the given pixel value.
 * \param [in] image The reference to the image to initialize.
 * \param [in] pixel_value The pointer to the constant pixel value to initialize all image pixels. See <tt>\ref vx_pixel_value_t</tt>.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE If the image is a uniform image, a virtual image, or not a <tt>\ref vx_image</tt>.
 * \retval VX_ERROR_INVALID_PARAMETERS If any of the other parameters are incorrect.
 * \note All pixels of the entire image are initialized to the indicated pixel value, independently from the valid region.
 * The valid region of the image is unaffected by this function. The image remains mutable after the call to this function,
 * so its pixels and mutable attributes may be changed by subsequent functions.
 * \ingroup group_image
 */
VX_API_ENTRY vx_status VX_API_CALL vxSetImagePixelValues(vx_image image, const vx_pixel_value_t *pixel_value)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * image_data = (AgoData *)image;
    if (agoIsValidData(image_data, VX_TYPE_IMAGE))
    {
        CAgoLock lock(image_data->ref.context->cs);
           status = VX_ERROR_INVALID_PARAMETERS;
        vx_uint32 x, y, p, width, height;
        vx_size planes = 0;
        vx_rectangle_t rect = {0,0,0,0};
        vxGetValidRegionImage(image, &rect);
        vx_df_image format = 0;
        vxQueryImage(image, VX_IMAGE_FORMAT, &format, sizeof(format));
        vxQueryImage(image, VX_IMAGE_PLANES, &planes, sizeof(planes));
        for (p = 0; p < planes; p++)
        {
            vx_imagepatch_addressing_t addr;
            void *base = NULL;
            if (vxAccessImagePatch(image, &rect, p, &addr, &base, VX_WRITE_ONLY) == VX_SUCCESS)
            {
                status = VX_SUCCESS;
                width  = (format == VX_DF_IMAGE_U1) ? addr.dim_x - rect.start_x % 8 : addr.dim_x;
                height = addr.dim_y;
                for (y = 0; y < height; y+=addr.step_y)
                {
                    for (x = 0; x < width; x+=addr.step_x)
                    {
                        if (format == VX_DF_IMAGE_U1)
                        {
                            vx_uint32 xShftd = x + rect.start_x % 8;
                            vx_uint8 *ptr = (vx_uint8*)vxFormatImagePatchAddress2d(base, xShftd, y, &addr);
                            vx_uint8 offset = xShftd % 8;
                            vx_uint8 mask = 1 << offset;
                            *ptr = (*ptr & ~mask) | ((pixel_value->U1 ? 1 : 0) << offset);
                        }
                        else if (format == VX_DF_IMAGE_U8)
                        {
                            vx_uint8 *ptr = (vx_uint8*)vxFormatImagePatchAddress2d(base, x, y, &addr);
                            *ptr = pixel_value->U8;
                        }
                        else if (format == VX_DF_IMAGE_U16)
                        {
                            vx_uint16 *ptr = (vx_uint16*)vxFormatImagePatchAddress2d(base, x, y, &addr);
                            *ptr = pixel_value->U16;
                        }
                        else if (format == VX_DF_IMAGE_U32)
                        {
                            vx_uint32 *ptr = (vx_uint32*)vxFormatImagePatchAddress2d(base, x, y, &addr);
                            *ptr = pixel_value->U32;
                        }
                        else if (format == VX_DF_IMAGE_S16)
                        {
                            vx_int16 *ptr = (vx_int16*)vxFormatImagePatchAddress2d(base, x, y, &addr);
                            *ptr = pixel_value->S16;
                        }
                        else if (format == VX_DF_IMAGE_S32)
                        {
                            vx_int32 *ptr = (vx_int32*)vxFormatImagePatchAddress2d(base, x, y, &addr);
                            *ptr = pixel_value->S32;
                        }
                        else if ((format == VX_DF_IMAGE_RGB)  ||
                                 (format == VX_DF_IMAGE_RGBX))
                        {
                            vx_uint8 *ptr = (vx_uint8*)vxFormatImagePatchAddress2d(base, x, y, &addr);
                            ptr[0] = pixel_value->RGBX[0];
                            ptr[1] = pixel_value->RGBX[1];
                            ptr[2] = pixel_value->RGBX[2];
                            if (format == VX_DF_IMAGE_RGBX)
                                ptr[3] = pixel_value->RGBX[3];
                        }
                        else if ((format == VX_DF_IMAGE_YUV4) ||
                                 (format == VX_DF_IMAGE_IYUV))
                        {
                            vx_uint8 *pixel = (vx_uint8 *)&pixel_value->YUV;
                            vx_uint8 *ptr = (vx_uint8*)vxFormatImagePatchAddress2d(base, x, y, &addr);
                            *ptr = pixel[p];
                        }
                        else if ((p == 0) &&
                                 ((format == VX_DF_IMAGE_NV12) ||
                                  (format == VX_DF_IMAGE_NV21)))
                        {
                            vx_uint8 *pixel = (vx_uint8 *)&pixel_value->YUV;
                            vx_uint8 *ptr = (vx_uint8*)vxFormatImagePatchAddress2d(base, x, y, &addr);
                            *ptr = pixel[0];
                        }
                        else if ((p == 1) && (format == VX_DF_IMAGE_NV12))
                        {
                            vx_uint8 *pixel = (vx_uint8 *)&pixel_value->YUV;
                            vx_uint8 *ptr = (vx_uint8*)vxFormatImagePatchAddress2d(base, x, y, &addr);
                            ptr[0] = pixel[1];
                            ptr[1] = pixel[2];
                        }
                        else if ((p == 1) && (format == VX_DF_IMAGE_NV21))
                        {
                            vx_uint8 *pixel = (vx_uint8 *)&pixel_value->YUV;
                            vx_uint8 *ptr = (vx_uint8*)vxFormatImagePatchAddress2d(base, x, y, &addr);
                            ptr[0] = pixel[2];
                            ptr[1] = pixel[1];
                        }
                        else if (format == VX_DF_IMAGE_UYVY)
                        {
                            vx_uint8 *pixel = (vx_uint8 *)&pixel_value->YUV;
                            vx_uint8 *ptr = (vx_uint8*)vxFormatImagePatchAddress2d(base, x, y, &addr);
                            if (x % 2 == 0)
                            {
                                ptr[0] = pixel[1];
                                ptr[1] = pixel[0];
                            }
                            else
                            {
                                ptr[0] = pixel[2];
                                ptr[1] = pixel[0];
                            }
                        }
                        else if (format == VX_DF_IMAGE_YUYV)
                        {
                            vx_uint8 *pixel = (vx_uint8 *)&pixel_value->YUV;
                            vx_uint8 *ptr = (vx_uint8*)vxFormatImagePatchAddress2d(base, x, y, &addr);
                            if (x % 2 == 0)
                            {
                                ptr[0] = pixel[0];
                                ptr[1] = pixel[1];
                            }
                            else
                            {
                                ptr[0] = pixel[0];
                                ptr[1] = pixel[2];
                            }
                        }
                    }
                }
                if (vxCommitImagePatch(image, &rect, p, &addr, base) != VX_SUCCESS)
                {
                    status = VX_FAILURE;
                    break;
                }
            }
            else
            {
                status = VX_FAILURE;
                break;
            }
        }
    }
    return status;
}

/*! \brief Releases a reference to an image object.
* The object may not be garbage collected until its total reference count is zero.
* \param [in] image The pointer to the image to release.
* \post After returning from this function the reference is zeroed.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
* \ingroup group_image
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseImage(vx_image *image)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (image && agoIsValidData((AgoData*)*image, VX_TYPE_IMAGE)) {
        if (!agoReleaseData((AgoData*)*image, true)) {
            *image = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief This computes the size needed to retrieve an image patch from an image.
* \param [in] image The reference to the image from which to extract the patch.
* \param [in] rect The coordinates. Must be 0 <= start < end <= dimension where
* dimension is width for x and height for y.
* \param [in] plane_index The plane index from which to get the data.
* \return vx_size
* \ingroup group_image
*/
VX_API_ENTRY vx_size VX_API_CALL vxComputeImagePatchSize(vx_image image_,
    const vx_rectangle_t *rect,
    vx_uint32 plane_index)
{
    AgoData * image = (AgoData *)image_;
    vx_size size = 0;
    if (agoIsValidData(image, VX_TYPE_IMAGE) && !image->isVirtual && rect && (plane_index < image->u.img.planes)) {
        AgoData * img = image;
        if (image->children) {
            img = image->children[plane_index];
        }
        size = (vx_size)((unsigned long) ImageWidthInBytesFloor(((rect->end_x - rect->start_x) >> img->u.img.x_scale_factor_is_2), img) *
                ((rect->end_y - rect->start_y) >> img->u.img.y_scale_factor_is_2));
    }
    return size;
}

/*! \brief Allows the User to extract a rectangular patch (subset) of an image from a single plane.
* \param [in] image The reference to the image from which to extract the patch.
* \param [in] rect The coordinates from which to get the patch. Must be 0 <= start < end.
* \param [in] plane_index The plane index from which to get the data.
* \param [out] addr The addressing information for the image patch to be written into the data structure.
* \param [out] ptr The pointer to a pointer of a location to store the data.
* \arg If the user passes in a NULL, an error occurs.
* \arg If the user passes in a pointer to a NULL, the function returns internal memory, map, or allocates a buffer and returns it.
* \arg If the user passes in a pointer to a non-NULL pointer, the function attempts to
* copy to the location provided by the user.
*
* (*ptr) must be given to <tt>\ref vxCommitImagePatch</tt>.
* \param [in] usage This declares the intended usage of the pointer using the <tt>\ref vx_accessor_e</tt> enumeration.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_OPTIMIZED_AWAY The reference is a virtual image and cannot be accessed or committed.
* \retval VX_ERROR_INVALID_PARAMETERS The \a start, \a end, \a plane_index, \a stride_x, or \a stride_y pointer is incorrect.
* \retval VX_ERROR_INVALID_REFERENCE The image reference is not actually an image reference.
* \note The user may ask for data outside the bounds of the valid region, but
* such data has an undefined value.
* \note Users must be cautious to prevent passing in \e uninitialized pointers or
* addresses of uninitialized pointers to this function.
* \pre <tt>\ref vxComputeImagePatchSize</tt> if users wish to allocate their own memory.
* \post <tt>\ref vxCommitImagePatch</tt> with same (*ptr) value.
* \ingroup group_image
* \include vx_imagepatch.c
*/
VX_API_ENTRY vx_status VX_API_CALL vxAccessImagePatch(vx_image image_,
    const vx_rectangle_t *rect,
    vx_uint32 plane_index,
    vx_imagepatch_addressing_t *addr,
    void **ptr,
    vx_enum usage)
{
    AgoData * image = (AgoData *)image_;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(image, VX_TYPE_IMAGE)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (image->isVirtual && !image->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if ((plane_index < image->u.img.planes) && addr && ptr && rect &&
                 rect->start_x < rect->end_x && rect->start_y < rect->end_y &&
                 rect->end_x <= image->u.img.width && rect->end_y <= image->u.img.height &&
                 (!image->u.img.isUniform || usage == VX_READ_ONLY) && !image->isNotFullyConfigured)
        {
            AgoData * img = image;
            if (image->children) {
                img = image->children[plane_index];
            }
            if (!img->buffer) {
                CAgoLock lock(img->ref.context->cs);
                if (agoAllocData(img)) {
                    return VX_ERROR_NO_MEMORY;
                }
            }
            if (!*ptr) {
                addr->dim_x = (rect->end_x - rect->start_x);
                addr->dim_y = (rect->end_y - rect->start_y);
                addr->scale_x = VX_SCALE_UNITY >> img->u.img.x_scale_factor_is_2;
                addr->scale_y = VX_SCALE_UNITY >> img->u.img.y_scale_factor_is_2;
                addr->step_x = 1 << img->u.img.x_scale_factor_is_2;
                addr->step_y = 1 << img->u.img.y_scale_factor_is_2;
                addr->stride_x = ((img->u.img.pixel_size_in_bits_num & 7) || (img->u.img.pixel_size_in_bits_denom > 1)) ?
                    0 : (img->u.img.pixel_size_in_bits_num >> 3);
                addr->stride_y = img->u.img.stride_in_bytes;
            }
            vx_uint8 * ptr_internal = img->buffer +
                (rect->start_y >> img->u.img.y_scale_factor_is_2) * img->u.img.stride_in_bytes +
                ImageWidthInBytesFloor((rect->start_x >> img->u.img.x_scale_factor_is_2), img);
            vx_uint8 * ptr_returned = *ptr ? (vx_uint8 *)*ptr : ptr_internal;
            // save the pointer and usage for use in vxCommitImagePatch
            status = VX_SUCCESS;
            for (auto i = img->mapped.begin(); i != img->mapped.end(); i++) {
                if (i->ptr == ptr_returned) {
                    // can't support vxAccessImagePatch() more than once with same pointer
                    // the application needs to call vxCommitImagePatch() before calling vxAccessImagePatch()
                    status = VX_FAILURE;
                }
            }
            if (status == VX_SUCCESS) {
                MappedData item = { img->nextMapId++, ptr_returned, usage, (ptr_returned != ptr_internal) ? true : false };
                img->mapped.push_back(item);
                *ptr = ptr_returned;
                if (usage == VX_READ_ONLY || usage == VX_READ_AND_WRITE) {
#if ENABLE_OPENCL
                    auto dataToSync = img->u.img.isROI ? img->u.img.roiMasterImage : img;
                    if (dataToSync->opencl_buffer && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                        // make sure dirty OpenCL buffers are synched before giving access for read
                        if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                            cl_int err = clEnqueueReadBuffer(dataToSync->ref.context->opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->gpu_buffer_offset, dataToSync->size, dataToSync->buffer, 0, NULL, NULL);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&image->ref, status, "ERROR: vxAccessImagePatch: clEnqueueReadBuffer() => %d\n", err);
                                return status;
                            }
                            dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                        }
                    }
#elif ENABLE_HIP
                    auto dataToSync = img->u.img.isROI ? img->u.img.roiMasterImage : img;
                    if (dataToSync->hip_memory && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                        // make sure dirty OpenCL buffers are synched before giving access for read
                        // copy from Device to Host
                        if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                            hipError_t err = hipMemcpyDtoH((void *)dataToSync->buffer, (dataToSync->hip_memory + dataToSync->gpu_buffer_offset), dataToSync->size);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&image->ref, status, "ERROR: vxAccessImagePatch: hipMemcpyDtoH() => %d\n", err);
                                return status;
                            }
                            dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                        }
                    }
#endif
                    if (item.used_external_ptr) {
                        // copy if read is requested with explicit external buffer
                        if (addr->stride_x == 0 || ((addr->stride_x << 3) == img->u.img.pixel_size_in_bits_num && img->u.img.pixel_size_in_bits_denom == 1))
                            HafCpu_ChannelCopy_U8_U8(ImageWidthInBytesFloor((rect->end_x - rect->start_x) >> img->u.img.x_scale_factor_is_2, img),
                                ((rect->end_y - rect->start_y) >> img->u.img.y_scale_factor_is_2), ptr_returned, addr->stride_y, ptr_internal, img->u.img.stride_in_bytes);
                        else
                            HafCpu_BufferCopyDisperseInDst(((rect->end_x - rect->start_x) >> img->u.img.x_scale_factor_is_2), ((rect->end_y - rect->start_y) >> img->u.img.y_scale_factor_is_2),
                                (img->u.img.pixel_size_in_bits_num / img->u.img.pixel_size_in_bits_denom + 7) >> 3, ptr_returned, addr->stride_y, addr->stride_x, ptr_internal, img->u.img.stride_in_bytes);
                    }
                }
            }
        }
    }
    return status;
}

/*! \brief This allows the User to commit a rectangular patch (subset) of an image from a single plane.
* \param [in] image The reference to the image from which to extract the patch.
* \param [in] rect The coordinates to which to set the patch. Must be 0 <= start <= end.
* This may be 0 or a rectangle of zero area in order to indicate that the commit
* must only decrement the reference count.
* \param [in] plane_index The plane index to which to set the data.
* \param [in] addr The addressing information for the image patch.
* \param [in] ptr The pointer of a location from which to read the data. If the
* user allocated the pointer they must free it. If the pointer
* was set by <tt>\ref vxAccessImagePatch</tt>, the user may not access the pointer after
* this call completes.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_OPTIMIZED_AWAY The reference is a virtual image and cannot be accessed or committed.
* \retval VX_ERROR_INVALID_PARAMETERS The \a start, \a end, \a plane_index, \a stride_x, or \a stride_y pointer is incorrect.
* \retval VX_ERROR_INVALID_REFERENCE The image reference is not actually an image reference.
* \ingroup group_image
* \include vx_imagepatch.c
* \note If the implementation gives the client a pointer from
* <tt>\ref vxAccessImagePatch</tt> then implementation-specific behavior may occur.
* If not, then a copy occurs from the users pointer to the internal data of the object.
* \note If the rectangle intersects bounds of the current valid region, the
* valid region grows to the union of the two rectangles as long as they occur
* within the bounds of the original image dimensions.
*/
VX_API_ENTRY vx_status VX_API_CALL vxCommitImagePatch(vx_image image_,
    const vx_rectangle_t *rect,
    vx_uint32 plane_index,
    const vx_imagepatch_addressing_t *addr,
    const void *ptr)
{
    AgoData * image = (AgoData *)image_;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(image, VX_TYPE_IMAGE)) {
        // check for ZERO AREA and mark rect as NULL for ZERO AREA
        if (rect && ((rect->start_x == rect->end_x) || (rect->start_y == rect->end_y)))
            rect = NULL;
        // check for valid arguments
        status = VX_ERROR_INVALID_PARAMETERS;
        if (image->isVirtual && !image->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if ((plane_index < image->u.img.planes) && addr && ptr &&
                 (!rect || (rect->start_x < rect->end_x && rect->start_y < rect->end_y && rect->end_x <= image->u.img.width && rect->end_y <= image->u.img.height)))
        {
            status = VX_SUCCESS;
            AgoData * img = image;
            if (image->children) {
                img = image->children[plane_index];
            }
            if (!img->buffer) {
                status = VX_FAILURE;
            }
            else if (!img->mapped.empty()) {
                vx_enum usage = VX_READ_ONLY;
                bool used_external_ptr = false;
                for (auto i = img->mapped.begin(); i != img->mapped.end(); i++) {
                    if (i->ptr == ptr) {
                        if (rect) {
                            usage = i->usage;
                            used_external_ptr = i->used_external_ptr;
                        }
                        img->mapped.erase(i);
                        break;
                    }
                }
                if (usage == VX_WRITE_ONLY || usage == VX_READ_AND_WRITE) {
                    if (used_external_ptr) {
                        // copy from external buffer
                        vx_uint8 * buffer = img->buffer + (rect->start_y >> img->u.img.y_scale_factor_is_2) * img->u.img.stride_in_bytes +
                            ImageWidthInBytesFloor((rect->start_x >> img->u.img.x_scale_factor_is_2), img);

                        if (addr->stride_x == 0 || ((addr->stride_x << 3) == img->u.img.pixel_size_in_bits_num && img->u.img.pixel_size_in_bits_denom == 1))
                            HafCpu_ChannelCopy_U8_U8(ImageWidthInBytesFloor(((rect->end_x - rect->start_x) >> img->u.img.x_scale_factor_is_2), img),
                                ((rect->end_y - rect->start_y) >> img->u.img.y_scale_factor_is_2), buffer, img->u.img.stride_in_bytes, (vx_uint8 *)ptr, addr->stride_y);
                        else
                            HafCpu_BufferCopyDisperseInSrc(((rect->end_x - rect->start_x) >> img->u.img.x_scale_factor_is_2) * addr->stride_x, ((rect->end_y - rect->start_y) >> img->u.img.y_scale_factor_is_2),
                            (img->u.img.pixel_size_in_bits_num / img->u.img.pixel_size_in_bits_denom + 7) >> 3, buffer, img->u.img.stride_in_bytes, (vx_uint8 *)ptr, addr->stride_y, addr->stride_x);
                    }
                    // update sync flags
                    auto dataToSync = img->u.img.isROI ? img->u.img.roiMasterImage : img;
                    dataToSync->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
                }
            }
        }
    }
    return status;
}

/*!
* \brief Accesses a specific indexed pixel in an image patch.
* \param [in] ptr The base pointer of the patch as returned from <tt>\ref vxAccessImagePatch</tt>.
* \param [in] index The 0 based index of the pixel count in the patch. Indexes increase horizontally by 1 then wrap around to the next row.
* \param [in] addr The pointer to the addressing mode information returned from <tt>\ref vxAccessImagePatch</tt>.
* \return void * Returns the pointer to the specified pixel.
* \pre <tt>\ref vxAccessImagePatch</tt>
* \include vx_imagepatch.c
* \ingroup group_image
*/
VX_API_ENTRY void * VX_API_CALL vxFormatImagePatchAddress1d(void *ptr, vx_uint32 index, const vx_imagepatch_addressing_t *addr)
{
    vx_uint8 *new_ptr = NULL;
    if (ptr && index < addr->dim_x*addr->dim_y)
    {
        vx_uint32 x = index % addr->dim_x;
        vx_uint32 y = index / addr->dim_x;
        vx_uint32 offset = vxComputePatchOffset(x, y, addr);
        new_ptr = (vx_uint8 *)ptr;
        new_ptr = &new_ptr[offset];
    }
    return new_ptr;
}

/*!
* \brief Accesses a specific pixel at a 2d coordinate in an image patch.
* \param [in] ptr The base pointer of the patch as returned from <tt>\ref vxAccessImagePatch</tt>.
* \param [in] x The x dimension within the patch.
* \param [in] y The y dimension within the patch.
* \param [in] addr The pointer to the addressing mode information returned from <tt>\ref vxAccessImagePatch</tt>.
* \return void * Returns the pointer to the specified pixel.
* \pre <tt>\ref vxAccessImagePatch</tt>
* \include vx_imagepatch.c
* \ingroup group_image
*/
VX_API_ENTRY void * VX_API_CALL vxFormatImagePatchAddress2d(void *ptr, vx_uint32 x, vx_uint32 y, const vx_imagepatch_addressing_t *addr)
{
    vx_uint8 *new_ptr = NULL;
    if (ptr && x < addr->dim_x && y < addr->dim_y)
    {
        vx_uint32 offset = vxComputePatchOffset(x, y, addr);
        new_ptr = (vx_uint8 *)ptr;
        new_ptr = &new_ptr[offset];
    }
    return new_ptr;
}

/*! \brief Retrieves the valid region of the image as a rectangle.
* \details After the image is allocated but has not been written to this
* returns the full rectangle of the image so that functions do not have to manage
* a case for uninitialized data. The image still retains an uninitialized
* value, but once the image is written to via any means such as <tt>\ref vxCommitImagePatch</tt>,
* the valid region is altered to contain the maximum bounds of the written
* area.
* \param [in] image The image from which to retrieve the valid region.
* \param [out] rect The destination rectangle.
* \return vx_status
* \retval VX_ERROR_INVALID_REFERENCE Invalid image.
* \retval VX_ERROR_INVALID_PARAMETERS Invalid rect.
* \retval VX_STATUS Valid image.
* \note This rectangle can be passed directly to <tt>\ref vxAccessImagePatch</tt> to get
* the full valid region of the image. Modifications from <tt>\ref vxCommitImagePatch</tt>
* grows the valid region.
* \ingroup group_image
*/
VX_API_ENTRY vx_status VX_API_CALL vxGetValidRegionImage(vx_image image_, vx_rectangle_t *rect)
{
    AgoData * image = (AgoData *)image_;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(image, VX_TYPE_IMAGE))
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (rect) {
            *rect = image->u.img.rect_valid;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Allows the application to copy a rectangular patch from/into an image object plane.
* \param [in] image The reference to the image object that is the source or the
* destination of the copy.
* \param [in] image_rect The coordinates of the image patch. The patch must be within
* the bounds of the image. (start_x, start_y) gives the coordinates of the topleft
* pixel inside the patch, while (end_x, end_y) gives the coordinates of the bottomright
* element out of the patch. Must be 0 <= start < end <= number of pixels in the image dimension.
* \param [in] image_plane_index The plane index of the image object that is the source or the
* destination of the patch copy.
* \param [in] user_addr The address of a structure describing the layout of the
* user memory location pointed by user_ptr. In the structure, only dim_x, dim_y,
* stride_x and stride_y fields must be provided, other fields are ignored by the function.
* The layout of the user memory must follow a row major order:
* stride_x >= pixel size in bytes, and stride_y >= stride_x * dim_x.
* \param [in] user_ptr The address of the memory location where to store the requested data
* if the copy was requested in read mode, or from where to get the data to store into the image
* object if the copy was requested in write mode. The accessible memory must be large enough
* to contain the specified patch with the specified layout:
* accessible memory in bytes >= (end_y - start_y) * stride_y.
* \param [in] usage This declares the effect of the copy with regard to the image object
* using the <tt>\ref vx_accessor_e</tt> enumeration. For uniform images, only VX_READ_ONLY
* is supported. For other images, Only VX_READ_ONLY and VX_WRITE_ONLY are supported:
* \arg VX_READ_ONLY means that data is copied from the image object into the application memory
* \arg VX_WRITE_ONLY means that data is copied into the image object from the application memory
* \param [in] user_mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that specifies
* the memory type of the memory referenced by the user_addr.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_OPTIMIZED_AWAY This is a reference to a virtual image that cannot be
* accessed by the application.
* \retval VX_ERROR_INVALID_REFERENCE The image reference is not actually an image reference.
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \note The application may ask for data outside the bounds of the valid region, but
* such data has an undefined value.
* \ingroup group_image
*/
VX_API_ENTRY vx_status VX_API_CALL vxCopyImagePatch(vx_image image_, const vx_rectangle_t *image_rect, vx_uint32 image_plane_index, const vx_imagepatch_addressing_t *user_addr, void * user_ptr, vx_enum usage, vx_enum user_mem_type)
{
    AgoData * image = (AgoData *)image_;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(image, VX_TYPE_IMAGE))
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        if ((user_mem_type == VX_MEMORY_TYPE_HOST) && user_ptr && (usage == VX_READ_ONLY || usage == VX_WRITE_ONLY)) {
            vx_rectangle_t rect = *image_rect;
            vx_imagepatch_addressing_t addr = *user_addr;
            status = vxAccessImagePatch(image_, &rect, image_plane_index, &addr, &user_ptr, usage);
            if (status == VX_SUCCESS) {
                status = vxCommitImagePatch(image_, &rect, image_plane_index, &addr, user_ptr);
            }
        }
    }
    return status;
}

/*! \brief Allows the application to get direct access to a rectangular patch of an image object plane.
* \param [in] image The reference to the image object that contains the patch to map.
* \param [in] rect The coordinates of image patch. The patch must be within the
* bounds of the image. (start_x, start_y) gives the coordinate of the topleft
* element inside the patch, while (end_x, end_y) give the coordinate of
* the bottomright element out of the patch. Must be 0 <= start < end.
* \param [in] plane_index The plane index of the image object to be accessed.
* \param [out] map_id The address of a vx_map_id variable where the function
* returns a map identifier.
* \arg (*map_id) must eventually be provided as the map_id parameter of a call to
* <tt>\ref vxUnmapImagePatch</tt>.
* \param [out] addr The address of a structure describing the memory layout of the
* image patch to access. The function fills the structure pointed by addr with the
* layout information that the application must consult to access the pixel data
* at address (*ptr). The layout of the mapped memory follows a row-major order:
* stride_x>0, stride_y>0 and stride_y >= stride_x * dim_x.
* If the image object being accessed was created via
* <tt>\ref vxCreateImageFromHandle</tt>, then the returned memory layout will be
* the identical to that of the addressing structure provided when
* <tt>\ref vxCreateImageFromHandle</tt> was called.
* \param [out] ptr The address of a pointer that the function sets to the
* address where the requested data can be accessed. This returned (*ptr) address
* is only valid between the call to this function and the corresponding call to
* <tt>\ref vxUnmapImagePatch</tt>.
* If image was created via <tt>\ref vxCreateImageFromHandle</tt> then the returned
* address (*ptr) will be the address of the patch in the original pixel buffer
* provided when image was created.
* \param [in] usage This declares the access mode for the image patch, using
* the <tt>\ref vx_accessor_e</tt> enumeration. For uniform images, only VX_READ_ONLY
* is supported.
* \arg VX_READ_ONLY: after the function call, the content of the memory location
* pointed by (*ptr) contains the image patch data. Writing into this memory location
* is forbidden and its behavior is undefined.
* \arg VX_READ_AND_WRITE : after the function call, the content of the memory
* location pointed by (*ptr) contains the image patch data; writing into this memory
* is allowed only for the location of pixels only and will result in a modification
* of the written pixels in the image object once the patch is unmapped. Writing into
* a gap between pixels (when addr->stride_x > pixel size in bytes or addr->stride_y > addr->stride_x*addr->dim_x)
* is forbidden and its behavior is undefined.
* \arg VX_WRITE_ONLY: after the function call, the memory location pointed by (*ptr)
* contains undefined data; writing each pixel of the patch is required prior to
* unmapping. Pixels not written by the application before unmap will become
* undefined after unmap, even if they were well defined before map. Like for
* VX_READ_AND_WRITE, writing into a gap between pixels is forbidden and its behavior
* is undefined.
* \param [in] mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that
* specifies the type of the memory where the image patch is requested to be mapped.
* \param [in] flags An integer that allows passing options to the map operation.
* Use the <tt>\ref vx_map_flag_e</tt> enumeration.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_OPTIMIZED_AWAY This is a reference to a virtual image that cannot be
* accessed by the application.
* \retval VX_ERROR_INVALID_REFERENCE The image reference is not actually an image
* reference.
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \note The user may ask for data outside the bounds of the valid region, but
* such data has an undefined value.
* \ingroup group_image
* \post <tt>\ref vxUnmapImagePatch </tt> with same (*map_id) value.
*/
VX_API_ENTRY vx_status VX_API_CALL vxMapImagePatch(vx_image image_, const vx_rectangle_t *rect, vx_uint32 plane_index, vx_map_id *map_id, vx_imagepatch_addressing_t *addr, void **ptr, vx_enum usage, vx_enum mem_type, vx_uint32 flags)
{
    AgoData * image = (AgoData *)image_;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(image, VX_TYPE_IMAGE)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (image->isVirtual && !image->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if ((mem_type == VX_MEMORY_TYPE_HOST) && (plane_index < image->u.img.planes) && addr && ptr && rect &&
            rect->start_x < rect->end_x && rect->start_y < rect->end_y &&
            rect->end_x <= image->u.img.width && rect->end_y <= image->u.img.height &&
            (!image->u.img.isUniform || usage == VX_READ_ONLY) && !image->isNotFullyConfigured)
        {
            AgoData * img = image;
            if (image->children) {
                img = image->children[plane_index];
            }
            if (!img->buffer) {
                CAgoLock lock(img->ref.context->cs);
                if (agoAllocData(img) || img->u.img.mem_handle) {
                    return VX_ERROR_NO_MEMORY;
                }
            }
            vx_uint8 * ptr_returned = img->buffer +
                (rect->start_y >> img->u.img.y_scale_factor_is_2) * img->u.img.stride_in_bytes +
                ImageWidthInBytesFloor((rect->start_x >> img->u.img.x_scale_factor_is_2), img);
            // save the pointer and usage for use in vxCommitImagePatch
            status = VX_SUCCESS;
            for (auto i = img->mapped.begin(); i != img->mapped.end(); i++) {
                if (i->ptr == ptr_returned) {
                    // can't support vxAccessImagePatch() more than once with same pointer
                    // the application needs to call vxCommitImagePatch() before calling vxAccessImagePatch()
                    status = VX_FAILURE;
                }
            }
            if (status == VX_SUCCESS) {
#if ENABLE_OPENCL
                if (usage == VX_READ_ONLY || usage == VX_READ_AND_WRITE) {
                    auto dataToSync = img->u.img.isROI ? img->u.img.roiMasterImage : img;
                    if (dataToSync->opencl_buffer && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                        // make sure dirty OpenCL buffers are synched before giving access for read
                        if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                            cl_int err = clEnqueueReadBuffer(dataToSync->ref.context->opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->gpu_buffer_offset, dataToSync->size, dataToSync->buffer, 0, NULL, NULL);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&image->ref, status, "ERROR: vxMapImagePatch: clEnqueueReadBuffer() => %d\n", err);
                                return status;
                            }
                            dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                        }
                    }
                }
                else if (usage == VX_WRITE_ONLY)
                {
                    auto dataToSync = img->u.img.isROI ? img->u.img.roiMasterImage : img;
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_WRITE;
                }
#elif ENABLE_HIP
                    auto dataToSync = img->u.img.isROI ? img->u.img.roiMasterImage : img;
                    if (dataToSync->hip_memory && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                        // make sure dirty OpenCL buffers are synched before giving access for read
                        // copy from Device to Host
                        if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                            hipError_t err = hipMemcpyDtoH((void *)dataToSync->buffer, (dataToSync->hip_memory + dataToSync->gpu_buffer_offset), dataToSync->size);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&image->ref, status, "ERROR: vxAccessImagePatch: hipMemcpyDtoH() => %d\n", err);
                                return status;
                            }
                            dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                        }
                    }
                    if (usage == VX_WRITE_ONLY)
                    {
                        auto dataToSync = img->u.img.isROI ? img->u.img.roiMasterImage : img;
                        dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_WRITE;
                    }
#endif
                // get map id and set returned pointer
                MappedData item = { img->nextMapId++, ptr_returned, usage, false, 0, plane_index };
                image->mapped.push_back(item);
                *map_id = item.map_id;
                *ptr = ptr_returned;
                addr->dim_x = rect->end_x - rect->start_x;
                addr->dim_y = rect->end_y - rect->start_y;
                addr->scale_x = VX_SCALE_UNITY >> img->u.img.x_scale_factor_is_2;
                addr->scale_y = VX_SCALE_UNITY >> img->u.img.y_scale_factor_is_2;
                addr->step_x = 1 << img->u.img.x_scale_factor_is_2;
                addr->step_y = 1 << img->u.img.y_scale_factor_is_2;
                addr->stride_x = (img->u.img.pixel_size_in_bits_denom > 1 || (img->u.img.pixel_size_in_bits_num & 7)) ? 0 : (img->u.img.pixel_size_in_bits_num >> 3);
                addr->stride_y = img->u.img.stride_in_bytes;
            }
        }
    }
    return status;
}

/*! \brief Unmap and commit potential changes to a image object patch that were previously mapped.
* Unmapping an image patch invalidates the memory location from which the patch could
* be accessed by the application. Accessing this memory location after the unmap function
* completes has an undefined behavior.
* \param [in] image The reference to the image object to unmap.
* \param [out] map_id The unique map identifier that was returned by <tt>\ref vxMapImagePatch</tt> .
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_INVALID_REFERENCE The image reference is not actually an image reference.
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \ingroup group_image
* \pre <tt>\ref vxMapImagePatch</tt> with same map_id value
*/
VX_API_ENTRY vx_status VX_API_CALL vxUnmapImagePatch(vx_image image_, vx_map_id map_id)
{
    AgoData * image = (AgoData *)image_;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(image, VX_TYPE_IMAGE)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        for (auto i = image->mapped.begin(); i != image->mapped.end(); i++) {
            if (i->map_id == map_id) {
                vx_enum usage = i->usage;
                vx_uint32 plane = i->plane;
                image->mapped.erase(i);
                if (usage == VX_WRITE_ONLY || usage == VX_READ_AND_WRITE) {
                    // update sync flags
                    auto dataToSync = image->u.img.isROI ? image->u.img.roiMasterImage : image;
                    dataToSync->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
                    if (dataToSync->numChildren > 0 && plane < dataToSync->numChildren && dataToSync->children[plane]) {
                        dataToSync->children[plane]->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                        dataToSync->children[plane]->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
                    }
                }
                status = VX_SUCCESS;
                break;
            }
        }
    }
    return status;
}

/*! \brief Create a sub-image from a single plane channel of another image.
*
* The sub-image refers to the data in the original image. Updates to this image
* update the parent image and reversely.
*
* The function supports only channels that occupy an entire plane of a multi-planar
* images, as listed below. Other cases are not supported.
*     VX_CHANNEL_Y from YUV4, IYUV, NV12, NV21
*     VX_CHANNEL_U from YUV4, IYUV, NV12, NV21
*     VX_CHANNEL_V from YUV4, IYUV, NV12, NV21
*
* \param [in] img          The reference to the parent image.
* \param [in] channel      The <tt>\ref vx_channel_e</tt> channel to use.

* \returns An image reference <tt>\ref vx_image</tt> to the sub-image. Any possible errors preventing a
* successful creation should be checked using <tt>\ref vxGetStatus</tt>.
* \ingroup group_image
*/
VX_API_ENTRY vx_image VX_API_CALL vxCreateImageFromChannel(vx_image img, vx_enum channel)
{
    AgoData * image = (AgoData *)img, * subImage = nullptr;
    if (agoIsValidData(image, VX_TYPE_IMAGE))
    {
        if (image->numChildren > 0) {
            if (channel == VX_CHANNEL_Y &&
                (image->u.img.format == VX_DF_IMAGE_YUV4 || image->u.img.format == VX_DF_IMAGE_IYUV ||
                 image->u.img.format == VX_DF_IMAGE_NV12 || image->u.img.format == VX_DF_IMAGE_NV21))
            {
                subImage = image->children[0];
            }
            else if (channel == VX_CHANNEL_U && (image->u.img.format == VX_DF_IMAGE_YUV4 || image->u.img.format == VX_DF_IMAGE_IYUV))
            {
                subImage = image->children[1];
            }
            else if (channel == VX_CHANNEL_V && (image->u.img.format == VX_DF_IMAGE_YUV4 || image->u.img.format == VX_DF_IMAGE_IYUV))
            {
                subImage = image->children[2];
            }
            else if ((channel == VX_CHANNEL_U || channel == VX_CHANNEL_V) && (image->u.img.format == VX_DF_IMAGE_NV12 || image->u.img.format == VX_DF_IMAGE_NV21))
            {
                subImage = image->children[1];
            }
        }
    }
    if (subImage) {
        subImage->ref.external_count++;
        subImage->ref.context->num_active_references++;
    }
    return (vx_image)subImage;
}

/*! \brief Sets the valid rectangle for an image according to a supplied rectangle.
* \note Setting or changing the valid region from within a user node by means other than the call-back, for
* example by calling <tt>\ref vxSetImageValidRectangle</tt>, might result in an incorrect valid region calculation
* by the framework.
* \param [in] image  The reference to the image.
* \param [in] rect   The value to be set to the image valid rectangle. A NULL indicates that the valid region is the entire image.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE  The image is not a <tt>\ref vx_image</tt>.
* \retval VX_ERROR_INVALID_PARAMETERS The rect does not define a proper valid rectangle.
* \ingroup group_image
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetImageValidRectangle(vx_image image_, const vx_rectangle_t *rect)
{
    AgoData * image = (AgoData *)image_;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(image, VX_TYPE_IMAGE) && !image->isVirtual)
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (!rect) {
            image->u.img.rect_valid.start_x = 0;
            image->u.img.rect_valid.start_y = 0;
            image->u.img.rect_valid.end_x = image->u.img.width;
            image->u.img.rect_valid.end_y = image->u.img.height;
            status = VX_SUCCESS;
        }
        else if (rect->start_x < rect->end_x && rect->start_y < rect->end_y && rect->end_x <= image->u.img.width && rect->end_y <= image->u.img.height) {
            image->u.img.rect_valid = *rect;
            status = VX_SUCCESS;
        }
        if (status == VX_SUCCESS) {
            // TBD: inform graphs for take this image as input, to re-computed output valid regions
        }
    }
    return status;
}

/*==============================================================================
KERNEL
=============================================================================*/

/*! \brief Loads one or more kernels into the OpenVX context. This is the interface
* by which OpenVX is extensible. Once the set of kernels is loaded new kernels
* and their parameters can be queried.
* \note When all references to loaded kernels are released, the module
* may be automatically unloaded.
* \param [in] context The reference to the implementation context.
* \param [in] module The short name of the module to load. On systems where
* there are specific naming conventions for modules, the name passed
* should ignore such conventions. For example: \c libxyz.so should be
* passed as just \c xyz and the implementation will <i>do the right thing</i> that
* the platform requires.
* \note This API uses the system pre-defined paths for modules.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If the context is not a <tt>\ref vx_context</tt>.
* \retval VX_ERROR_INVALID_PARAMETERS If any of the other parameters are incorrect.
* \ingroup group_user_kernels
* \see vxGetKernelByName
*/
VX_API_ENTRY vx_status VX_API_CALL vxLoadKernels(vx_context context, const vx_char *module)
{
    return agoLoadModule(context, module);
}

/*! \brief Unloads all kernels from the OpenVX context that had been loaded from
* the module using the \ref vxLoadKernels function.
* \param [in] context The reference to the implementation context.
* \param [in] module The short name of the module to unload. On systems where
* there are specific naming conventions for modules, the name passed
* should ignore such conventions. For example: \c libxyz.so should be
* passed as just \c xyz and the implementation will <i>do the right thing</i>
* that the platform requires.
* \note This API uses the system pre-defined paths for modules.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If the context is not a <tt>\ref
vx_context</tt>.
* \retval VX_ERROR_INVALID_PARAMETERS If any of the other parameters are
incorrect.
* \ingroup group_user_kernels
* \see vxLoadKernels
*/
VX_API_ENTRY vx_status VX_API_CALL vxUnloadKernels(vx_context context, const vx_char *module)
{
    return agoUnloadModule(context, module);
}

/*! \brief Registers a module with kernels in a context.
 * \details This function registers the appropriate publish and unpublish functions
 * with the module name if the module is not a dynamic library, so <tt>\ref vxLoadKernels</tt> and
 * <tt>\ref vxUnloadKernels</tt> can be called.
 * \param [in] context The reference to the context the kernels must be added to.
 * \param [in] module The short name of the module to load.
 * \param [in] publish must add kernels to the context by calling <tt>\ref vxAddUserKernel</tt>
 * for each new kernel. It is called by <tt>\ref vxLoadKernels</tt>.
 * \param [in] unpublish must remove kernels from the context by calling <tt>\ref vxRemoveKernel</tt>
 * for each kernel the <tt>vxPublishKernels</tt> has added. It is called by <tt>\ref vxUnloadKernels</tt>.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 * \retval VX_ERROR_INVALID_REFERENCE context is not a valid <tt>\ref vx_context</tt> reference.
 * \retval VX_ERROR_INVALID_PARAMETERS If any of the other parameters are incorrect.
 * \ingroup group_user_kernels
 * \see vxLoadKernels
 */
VX_API_ENTRY vx_status VX_API_CALL vxRegisterKernelLibrary(vx_context context, const vx_char *module, vx_publish_kernels_f publish, vx_unpublish_kernels_f unpublish)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidContext(context))
    {
        // TBD: add support to load obj files or static libs
        status = VX_SUCCESS;
    }
    return status;
}

/*! \brief Obtains a reference to a kernel using a string to specify the name.
* \param [in] context The reference to the implementation context.
* \param [in] name The string of the name of the kernel to get.
* \return A kernel reference or zero if an error occurred.
* \retval 0 The kernel name is not found in the context.
* \ingroup group_kernel
* \pre <tt>\ref vxLoadKernels</tt> if the kernel is not provided by the
* OpenVX implementation.
* \note User Kernels should follow a "dotted" heirarchical syntax. For example:
* "com.company.example.xyz".
*/
VX_API_ENTRY vx_kernel VX_API_CALL vxGetKernelByName(vx_context context, const vx_char *name)
{
    vx_kernel akernel = NULL;
    if (agoIsValidContext(context)) {
        CAgoLock lock(context->cs);
        akernel = agoFindKernelByName(context, name);
        if (akernel) {
            akernel->ref.external_count++;
        }
    }
    return akernel;
}

/*! \brief Obtains a reference to the kernel using the <tt>\ref vx_kernel_e</tt> enumeration.
* \details Enum values above the standard set are assumed to apply to
* loaded libraries.
* \param [in] context The reference to the implementation context.
* \param [in] kernel A value from <tt>\ref vx_kernel_e</tt> or a vendor or client-defined value.
* \return A <tt>\ref vx_kernel</tt>.
* \retval 0 The kernel enumeration is not found in the context.
* \ingroup group_kernel
* \pre <tt>\ref vxLoadKernels</tt> if the kernel is not provided by the
* OpenVX implementation.
*/
VX_API_ENTRY vx_kernel VX_API_CALL vxGetKernelByEnum(vx_context context, vx_enum kernel)
{
    vx_kernel akernel = NULL;
    if (agoIsValidContext(context)) {
        CAgoLock lock(context->cs);
        akernel = agoFindKernelByEnum(context, kernel);
        if (akernel) {
            akernel->ref.external_count++;
        }
    }
    return akernel;
}

/*! \brief This allows the client to query the kernel to get information about
* the number of parameters, enum values, etc.
* \param [in] kernel The kernel reference to query.
* \param [in] attribute The attribute to query. Use a <tt>\ref vx_kernel_attribute_e</tt>.
* \param [out] ptr The pointer to the location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If the kernel is not a <tt>\ref vx_kernel</tt>.
* \retval VX_ERROR_INVALID_PARAMETERS If any of the other parameters are incorrect.
* \retval VX_ERROR_NOT_SUPPORTED If the attribute value is not supported in this implementation.
* \ingroup group_kernel
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryKernel(vx_kernel kernel, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidKernel(kernel)) {
        CAgoLock lock(kernel->ref.context->cs);
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_KERNEL_ATTRIBUTE_PARAMETERS:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = kernel->argCount;
                    status = VX_SUCCESS;
                }
                break;
            case VX_KERNEL_ATTRIBUTE_NAME:
                if (ptr != NULL && size >= VX_MAX_KERNEL_NAME) {
                    strncpy((char *)ptr, kernel->name, size);
                    status = VX_SUCCESS;
                }
                break;
            case VX_KERNEL_ATTRIBUTE_ENUM:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = kernel->id;
                    status = VX_SUCCESS;
                }
                break;
            case VX_KERNEL_ATTRIBUTE_LOCAL_DATA_SIZE:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = kernel->localDataSize;
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Release the reference to the kernel.
* The object may not be garbage collected until its total reference count is zero.
* \param [in] kernel The pointer to the kernel reference to release.
* \post After returning from this function the reference is zeroed.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
* \ingroup group_kernel
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseKernel(vx_kernel *kernel)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (kernel && agoIsValidKernel(*kernel)) {
        if (!agoReleaseKernel(*kernel, true)) {
            *kernel = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Allows users to add custom kernels to the known kernel
* database in OpenVX at run-time. This would primarily be used by the module function
* \c vxPublishKernels.
* \param [in] context The reference to the implementation context.
* \param [in] name The string to use to match the kernel.
* \param [in] enumeration The enumerated value of the kernel to be used by clients.
* \param [in] func_ptr The process-local function pointer to be invoked.
* \param [in] numParams The number of parameters for this kernel.
* \param [in] input The pointer to <tt>\ref vx_kernel_input_validate_f</tt>, which validates the
* input parameters to this kernel.
* \param [in] output The pointer to <tt>\ref vx_kernel_output_validate_f </tt>, which validates the
* output parameters to this kernel.
* \param [in] init The kernel initialization function.
* \param [in] deinit The kernel de-initialization function.
* \ingroup group_user_kernels
* \return <tt>\ref vx_kernel</tt>
* \retval 0 Indicates that an error occurred when adding the kernel.
* \retval * Kernel added to OpenVX.
*/
VX_API_ENTRY vx_kernel VX_API_CALL vxAddKernel(vx_context context,
    const vx_char name[VX_MAX_KERNEL_NAME],
    vx_enum enumeration,
    vx_kernel_f func_ptr,
    vx_uint32 numParams,
    vx_kernel_input_validate_f input,
    vx_kernel_output_validate_f output,
    vx_kernel_initialize_f init,
    vx_kernel_deinitialize_f deinit)
{
    vx_kernel kernel = NULL;
    if (agoIsValidContext(context) && numParams > 0 && numParams <= AGO_MAX_PARAMS && func_ptr && input && output) {
        CAgoLock lock(context->cs);
        // make sure there are no kernels with the same name
        if (!agoFindKernelByEnum(context, enumeration) && !agoFindKernelByName(context, name)) {
            kernel = new AgoKernel;
            // initialize references
            agoResetReference(&kernel->ref, VX_TYPE_KERNEL, context, NULL);
            for (vx_uint32 index = 0; index < AGO_MAX_PARAMS; index++) {
                agoResetReference(&kernel->parameters[index].ref, VX_TYPE_PARAMETER, kernel->ref.context, &kernel->ref);
                kernel->parameters[index].scope = &kernel->ref;
            }
            // add kernel object to context
            kernel->external_kernel = true;
            kernel->ref.external_count++;
            kernel->id = enumeration;
            kernel->flags = AGO_KERNEL_FLAG_GROUP_USER | AGO_KERNEL_FLAG_DEVICE_CPU | AGO_KERNEL_FLAG_VALID_RECT_RESET;
            strcpy(kernel->name, name);
            kernel->argCount = numParams;
            kernel->kernel_f = func_ptr;
            kernel->input_validate_f = input;
            kernel->output_validate_f = output;
            kernel->initialize_f = init;
            kernel->deinitialize_f = deinit;
            kernel->importing_module_index_plus1 = context->importing_module_index_plus1;
            kernel->user_kernel = vx_false_e;
            agoAddKernel(&context->kernelList, kernel);
        }
    }
    return kernel;
}

/*! \brief Allows users to add custom kernels to the known kernel
* database in OpenVX at run-time. This would primarily be used by the module function
* <tt>\ref vxPublishKernels</tt>.
* \param [in] context The reference to the implementation context.
* \param [in] name The string to use to match the kernel.
* \param [in] enumeration The enumerated value of the kernel to be used by clients.
* \param [in] func_ptr The process-local function pointer to be invoked.
* \param [in] numParams The number of parameters for this kernel.
* \param [in] validate The pointer to <tt>\ref vx_kernel_validate_f</tt>, which validates
* parameters to this kernel.
* \param [in] init The kernel initialization function.
* \param [in] deinit The kernel de-initialization function.
* \ingroup group_user_kernels
* \return <tt>\ref vx_kernel</tt>. Any possible errors
* preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
* \retval 0 Indicates that an error occurred when adding the kernel.
* \retval * Kernel added to OpenVX.
*/
VX_API_ENTRY vx_kernel VX_API_CALL vxAddUserKernel(vx_context context,
    const vx_char name[VX_MAX_KERNEL_NAME],
    vx_enum enumeration,
    vx_kernel_f func_ptr,
    vx_uint32 numParams,
    vx_kernel_validate_f validate,
    vx_kernel_initialize_f init,
    vx_kernel_deinitialize_f deinit)
{
    vx_kernel kernel = NULL;
    if (agoIsValidContext(context) && numParams > 0 && numParams <= AGO_MAX_PARAMS && func_ptr && validate) {
        CAgoLock lock(context->cs);
        // make sure there are no kernels with the same name
        if (!agoFindKernelByEnum(context, enumeration) && !agoFindKernelByName(context, name)) {
            kernel = new AgoKernel;
            // initialize references
            agoResetReference(&kernel->ref, VX_TYPE_KERNEL, context, NULL);
            for (vx_uint32 index = 0; index < AGO_MAX_PARAMS; index++) {
                agoResetReference(&kernel->parameters[index].ref, VX_TYPE_PARAMETER, kernel->ref.context, &kernel->ref);
                kernel->parameters[index].scope = &kernel->ref;
            }
            // add kernel object to context
            kernel->external_kernel = true;
            kernel->ref.external_count++;
            kernel->id = enumeration;
            kernel->flags = AGO_KERNEL_FLAG_GROUP_USER | AGO_KERNEL_FLAG_DEVICE_CPU | AGO_KERNEL_FLAG_VALID_RECT_RESET;
            strcpy(kernel->name, name);
            kernel->argCount = numParams;
            kernel->kernel_f = func_ptr;
            kernel->validate_f = validate;
            kernel->initialize_f = init;
            kernel->deinitialize_f = deinit;
            kernel->importing_module_index_plus1 = context->importing_module_index_plus1;
            kernel->user_kernel = vx_true_e;
            agoAddKernel(&context->kernelList, kernel);
        }
    }
    return kernel;
}

/*! \brief This API is called after all parameters have been added to the
* kernel and the kernel is \e ready to be used.
* \param [in] kernel The reference to the loaded kernel from <tt>\ref vxAddKernel</tt>.
* \return A <tt>\ref vx_status_e</tt> enumeration. If an error occurs, the kernel is not available
* for usage by the clients of OpenVX. Typically this is due to a mismatch
* between the number of parameters requested and given.
* \pre <tt>\ref vxAddKernel</tt> and <tt>\ref vxAddParameterToKernel</tt>
* \ingroup group_user_kernels
*/
VX_API_ENTRY vx_status VX_API_CALL vxFinalizeKernel(vx_kernel kernel)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidKernel(kernel)) {
        CAgoLock lock(kernel->ref.context->cs);
        if (kernel->external_kernel && !kernel->finalized && kernel->argCount > 0) {
            status = VX_SUCCESS;
            // check if kernel has been initialized properly
            for (vx_uint32 i = 0; i < kernel->argCount; i++) {
                if (!kernel->argType[i] || !kernel->argConfig[i] || !kernel->parameters[i].scope) {
                    status = VX_ERROR_INVALID_REFERENCE;
                    break;
                }
            }
            if (status == VX_SUCCESS) {
                // mark that kernel has been finalized
                kernel->finalized = true;
            }
        }
    }
    return status;
}

/*! \brief Allows users to set the signatures of the custom kernel.
* \param [in] kernel The reference to the kernel added with <tt>\ref vxAddKernel</tt>.
* \param [in] index The index of the parameter to add.
* \param [in] dir The direction of the parameter. This must be a value from <tt>\ref vx_direction_e</tt>.
* \param [in] data_type The type of parameter. This must be a value from <tt>\ref vx_type_e</tt>.
* \param [in] state The state of the parameter (required or not). This must be a value from <tt>\ref vx_parameter_state_e</tt>.
* \return A <tt>\ref vx_status_e</tt> enumerated value.
* \retval VX_SUCCESS Parameter is successfully set on kernel.
* \retval VX_ERROR_INVALID_REFERENCE The value passed as kernel was not a \c vx_kernel.
* \pre <tt>\ref vxAddKernel</tt>
* \ingroup group_user_kernels
*/
VX_API_ENTRY vx_status VX_API_CALL vxAddParameterToKernel(vx_kernel kernel, vx_uint32 index, vx_enum dir, vx_enum data_type, vx_enum state)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidKernel(kernel)) {
        CAgoLock lock(kernel->ref.context->cs);
        status = VX_ERROR_INVALID_PARAMETERS;
        // add parameter if the kernel is not finalized and not a built-in kernel and not initialized earlier
        if((data_type == VX_TYPE_DELAY && dir != VX_INPUT))
                return VX_ERROR_INVALID_PARAMETERS;
        if (kernel->external_kernel && !kernel->finalized &&
            index < AGO_MAX_PARAMS &&
            (dir == VX_INPUT || dir == VX_OUTPUT || dir == VX_BIDIRECTIONAL) &&
            (state == VX_PARAMETER_STATE_REQUIRED || state == VX_PARAMETER_STATE_OPTIONAL))
        {
            status = VX_SUCCESS;
            // save parameter details
            kernel->parameters[index].index = index;
            kernel->parameters[index].direction = (vx_direction_e)dir;
            kernel->argConfig[index] = (dir == VX_INPUT) ? AGO_KERNEL_ARG_INPUT_FLAG :
                ((dir == VX_OUTPUT) ? AGO_KERNEL_ARG_OUTPUT_FLAG : (AGO_KERNEL_ARG_INPUT_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG));
            kernel->parameters[index].type = data_type;
            kernel->argType[index] = data_type;
            kernel->parameters[index].state = (vx_parameter_state_e)state;
            if (state == VX_PARAMETER_STATE_OPTIONAL)
                kernel->argConfig[index] |= AGO_KERNEL_ARG_OPTIONAL_FLAG;
            kernel->parameters[index].scope = &kernel->ref;
            // update argument count
            if (index >= kernel->argCount)
                kernel->argCount = index + 1;
        }
    }
    return status;
}

/*! \brief Removes a non-finalized <tt>\ref vx_kernel</tt> from the <tt>\ref vx_context</tt>.
* Once a <tt>\ref vx_kernel</tt> has been finalized it cannot be removed.
* \param [in] kernel The reference to the kernel to remove. Returned from <tt>\ref vxAddKernel</tt>.
* \note Any kernel enumerated in the base standard
* cannot be removed; only kernels added through <tt>\ref vxAddKernel</tt> can
* be removed.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_INVALID_REFERENCE If an invalid kernel is passed in.
* \retval VX_ERROR_INVALID_PARAMETER If a base kernel is passed in.
* \ingroup group_user_kernels
*/
VX_API_ENTRY vx_status VX_API_CALL vxRemoveKernel(vx_kernel kernel)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidKernel(kernel)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        // release if the kernel is not finalized and not a built-in kernel or user kernel with validate_f without external references
        if (!kernel->finalized ||
            (kernel->validate_f && kernel->external_kernel && (kernel->flags & AGO_KERNEL_FLAG_GROUP_USER) /*&&
             kernel->ref.internal_count < 2 && kernel->ref.external_count == 0*/))
        {
            CAgoLock lock(kernel->ref.context->cs);
            kernel->finalized = false;
            if (!agoReleaseKernel(kernel, true)) {
                status = VX_SUCCESS;
            }
        }
    }
    return status;
}

/*! \brief Sets kernel attributes.
* \param [in] kernel The reference to the kernel.
* \param [in] attribute The enumeration of the attributes. See <tt>\ref vx_kernel_attribute_e</tt>.
* \param [in] ptr The pointer to the location from which to read the attribute.
* \param [in] size The size of the data area indicated by \a ptr in bytes.
* \note After a kernel has been passed to <tt>\ref vxFinalizeKernel</tt>, no attributes
* can be altered.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_user_kernels
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetKernelAttribute(vx_kernel kernel, vx_enum attribute, const void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidKernel(kernel)) {
        CAgoLock lock(kernel->ref.context->cs);
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_KERNEL_ATTRIBUTE_LOCAL_DATA_SIZE:
                if (size == sizeof(vx_size)) {
                    kernel->localDataSize = *(vx_size *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_KERNEL_ATTRIBUTE_AMD_NODE_REGEN_CALLBACK:
                if (size == sizeof(void *)) {
                    if (!kernel->finalized) {
                        *((void **)&kernel->regen_callback_f) = *(void **)ptr;
                        status = VX_SUCCESS;
                    }
                    else {
                        status = VX_ERROR_NOT_SUPPORTED;
                    }
                }
                break;
            case VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT:
                if (size == sizeof(void *)) {
                    if (!kernel->finalized) {
                        *((void **)&kernel->query_target_support_f) = *(void **)ptr;
                        status = VX_SUCCESS;
                    }
                    else {
                        status = VX_ERROR_NOT_SUPPORTED;
                    }
                }
                break;
#if (ENABLE_OPENCL || ENABLE_HIP)                    
            case VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK:
                if (size == sizeof(void *)) {
                    if (!kernel->finalized) {
                        *((void **)&kernel->opencl_codegen_callback_f) = *(void **)ptr;
                        status = VX_SUCCESS;
                    }
                    else {
                        status = VX_ERROR_NOT_SUPPORTED;
                    }
                }
                break;
            case VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK:
                if (size == sizeof(void *)) {
                    if (!kernel->finalized) {
                        *((void **)&kernel->opencl_global_work_update_callback_f) = *(void **)ptr;
                        status = VX_SUCCESS;
                    }
                    else {
                        status = VX_ERROR_NOT_SUPPORTED;
                    }
                }
                break;
            case VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE:
                if (size == sizeof(vx_bool)) {
                    if (!kernel->finalized && !kernel->gpu_buffer_update_callback_f) {
                        kernel->opencl_buffer_access_enable = *(vx_bool *)ptr;
                        status = VX_SUCCESS;
                    }
                    else {
                        status = VX_ERROR_NOT_SUPPORTED;
                    }
                }
                break;
            case VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_UPDATE_CALLBACK:
                if (size == sizeof(AgoKernelGpuBufferUpdateInfo)) {
                    if (!kernel->finalized) {
                        AgoKernelGpuBufferUpdateInfo * info = (AgoKernelGpuBufferUpdateInfo *)ptr;
                        if (info->gpu_buffer_update_param_index >= kernel->argCount ||
                            info->gpu_buffer_update_callback_f == nullptr ||
                            kernel->parameters[info->gpu_buffer_update_param_index].direction != VX_INPUT ||
                            kernel->parameters[info->gpu_buffer_update_param_index].type != VX_TYPE_IMAGE ||
                            kernel->parameters[info->gpu_buffer_update_param_index].state != VX_PARAMETER_STATE_REQUIRED)
                        {
                            // param index has to point to required input images only
                            status = VX_ERROR_INVALID_PARAMETERS;
                        }
                        else {
                            kernel->gpu_buffer_update_callback_f = info->gpu_buffer_update_callback_f;
                            kernel->gpu_buffer_update_param_index = info->gpu_buffer_update_param_index;
                            kernel->opencl_buffer_access_enable = vx_true_e;
                            status = VX_SUCCESS;
                        }
                    }
                    else {
                        status = VX_ERROR_NOT_SUPPORTED;
                    }
                }
                break;
#endif
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Retrieves a <tt>\ref vx_parameter</tt> from a <tt>\ref vx_kernel</tt>.
* \param [in] kernel The reference to the kernel.
* \param [in] index The index of the parameter.
* \return A <tt>\ref vx_parameter</tt>.
* \retval 0 Either the kernel or index is invalid.
* \retval * The parameter reference.
* \ingroup group_parameter
*/
VX_API_ENTRY vx_parameter VX_API_CALL vxGetKernelParameterByIndex(vx_kernel kernel, vx_uint32 index)
{
    vx_parameter parameter = NULL;
    if (agoIsValidKernel(kernel) && index < kernel->argCount) {
        parameter = &kernel->parameters[index];
        parameter->ref.external_count++;
    }
    return parameter;
}

/*==============================================================================
GRAPH
=============================================================================*/

/*! \brief Creates an empty graph.
* \param [in] context The reference to the implementation context.
* \return A graph reference.
* \retval 0 if an error occurred.
* \ingroup group_graph
*/
VX_API_ENTRY vx_graph VX_API_CALL vxCreateGraph(vx_context context)
{
    vx_graph graph = NULL;
    if (agoIsValidContext(context)) {
        graph = agoCreateGraph(context);
    }
    return graph;
}

/*! \brief Releases a reference to a graph.
* The object may not be garbage collected until its total reference count is zero.
* Once the reference count is zero, all node references in the graph are automatically
* released as well. Data referenced by those nodes may not be released as
* the user may have external references to the data.
* \param [in] graph The pointer to the graph to release.
* \post After returning from this function the reference is zeroed.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
* \ingroup group_graph
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseGraph(vx_graph *graph)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (graph && agoIsValidGraph(*graph)) {
        if (!agoReleaseGraph(*graph)) {
            *graph = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Verifies the state of the graph before it is executed.
* This is useful to catch programmer errors and contract errors. If not verified,
* the graph verifies before being processed.
* \pre Memory for data objects is not guarenteed to exist before
* this call. \post After this call data objects exist unless
* the implementation optimized them out.
* \param [in] graph The reference to the graph to verify.
* \return A status code for graphs with more than one error; it is
* undefined which error will be returned. Register a log callback using <tt>\ref vxRegisterLogCallback</tt>
* to receive each specific error in the graph.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
* \retval VX_ERROR_MULTIPLE_WRITERS If the graph contains more than one writer
* to any data object.
* \retval VX_ERROR_INVALID_NODE If a node in the graph is invalid or failed be created.
* \retval VX_ERROR_INVALID_GRAPH If the graph contains cycles or some other invalid topology.
* \retval VX_ERROR_INVALID_TYPE If any parameter on a node is given the wrong type.
* \retval VX_ERROR_INVALID_VALUE If any value of any parameter is out of bounds of specification.
* \retval VX_ERROR_INVALID_FORMAT If the image format is not compatible.
* \ingroup group_graph
* \see vxConvertReference
* \see vxProcessGraph
*/
VX_API_ENTRY vx_status VX_API_CALL vxVerifyGraph(vx_graph graph)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph)) {
        CAgoLock lock(graph->cs);
        CAgoLock lock2(graph->ref.context->cs);

        // mark that graph is not verified and can't be executed
        //graph->verified = vx_false_e;
        graph->isReadyToExecute = vx_false_e;
        graph->state = VX_GRAPH_STATE_UNVERIFIED;

        // check to see if user requested for graph dump
        vx_uint32 ago_graph_dump = 0;
        char textBuffer[256];
        if (agoGetEnvironmentVariable("AGO_DUMP_GRAPH", textBuffer, sizeof(textBuffer))) {
            ago_graph_dump = atoi(textBuffer);
        }
        if (ago_graph_dump) {
            agoWriteGraph(graph, NULL, 0, stdout, "*INPUT*");
        }

        // set the AGO_DEFAULT_TARGET if it is not set by the user
        if (!agoGetEnvironmentVariable("AGO_DEFAULT_TARGET", textBuffer, sizeof(textBuffer))) {
#if ENABLE_OPENCL || ENABLE_HIP
            agoSetEnvironmentVariable("AGO_DEFAULT_TARGET", "GPU");
#else
            agoSetEnvironmentVariable("AGO_DEFAULT_TARGET", "CPU");
#endif
        }

        // verify graph per OpenVX specification
        status = agoVerifyGraph(graph);
        if (status == VX_SUCCESS) {
            // run graph optimizer
            if (agoOptimizeGraph(graph)) {
                status = VX_FAILURE;
            }
            // initialize graph
            else if (agoInitializeGraph(graph)) {
                status = VX_FAILURE;
            }
            // graph is ready to execute
            else {
                graph->isReadyToExecute = vx_true_e;
            }
            graph->verified = vx_true_e;
            graph->state = VX_GRAPH_STATE_VERIFIED;
        }

        if (ago_graph_dump) {
            if (status == VX_SUCCESS) {
                agoWriteGraph(graph, NULL, 0, stdout, "*FINAL*");
            }
        }
    }
    if (status == VX_SUCCESS)
        graph->verified = vx_true_e;
    else
        graph->verified = vx_false_e;
    return status;
}

/*! \brief This function causes the synchronous processing of a graph. If the graph
* has not been verified, then the implementation verifies the graph
* immediately. If verification fails this function returns a status
* identical to what <tt>\ref vxVerifyGraph</tt> would return. After
* the graph verfies successfully then processing occurs. If the graph was
* previously verified via <tt>\ref vxVerifyGraph</tt> or <tt>\ref vxProcessGraph</tt>
* then the graph is processed. This function blocks until the graph is completed.
* \param [in] graph The graph to execute.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Graph has been processed.
* \retval VX_FAILURE A catastrophic error occurred during processing.
* \retval * See <tt>\ref vxVerifyGraph</tt>.
* \pre <tt>\ref vxVerifyGraph</tt> must return <tt>\ref VX_SUCCESS</tt> before this function will pass.
* \ingroup group_graph
* \see vxVerifyGraph
*/
VX_API_ENTRY vx_status VX_API_CALL vxProcessGraph(vx_graph graph)
{
    vx_status status = agoProcessGraph(graph);
    return status;
}

/*! \brief Schedules a graph for future execution.
* \param [in] graph The graph to schedule.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_NO_RESOURCES The graph cannot be scheduled now.
* \retval VX_ERROR_NOT_SUFFICIENT The graph is not verified and has failed
forced verification.
* \retval VX_SUCCESS The graph has been scheduled.
* \pre <tt>\ref vxVerifyGraph</tt> must return <tt>\ref VX_SUCCESS</tt> before this function will pass.
* \ingroup group_graph
*/
VX_API_ENTRY vx_status VX_API_CALL vxScheduleGraph(vx_graph graph)
{
    return agoScheduleGraph(graph);
}

/*! \brief Waits for a specific graph to complete.
* \param [in] graph The graph to wait on.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS The graph has completed.
* \retval VX_FAILURE The graph has not completed yet.
* \pre <tt>\ref vxScheduleGraph</tt>
* \ingroup group_graph
*/
VX_API_ENTRY vx_status VX_API_CALL vxWaitGraph(vx_graph graph)
{
    return agoWaitGraph(graph);
}

/*! \brief Allows the user to query attributes of the Graph.
* \param [in] graph The reference to the created graph.
* \param [in] attribute The <tt>\ref vx_graph_attribute_e</tt> type needed.
* \param [out] ptr The location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_graph
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryGraph(vx_graph graph, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            CAgoLock lock(graph->cs);
            switch (attribute)
            {
            case VX_GRAPH_ATTRIBUTE_NUMNODES:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = graph->nodeList.count;
                    status = VX_SUCCESS;
                }
                break;
            case VX_GRAPH_ATTRIBUTE_STATUS:
                if (size == sizeof(vx_status)) {
                    *(vx_status *)ptr = graph->status;
                    status = VX_SUCCESS;
                }
                break;
            case VX_GRAPH_ATTRIBUTE_PERFORMANCE:
                if (size == sizeof(vx_perf_t)) {
                    agoPerfCopyNormalize(graph->ref.context, (vx_perf_t *)ptr, &graph->perf);
                    status = VX_SUCCESS;
                }
                break;
            case VX_GRAPH_ATTRIBUTE_STATE:
                if (size == sizeof(vx_enum)) {
                    *(vx_status *)ptr = graph->state;
                    status = VX_SUCCESS;
                }
                break;
            case VX_GRAPH_ATTRIBUTE_NUMPARAMETERS:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = (vx_uint32)graph->parameters.size();
                    status = VX_SUCCESS;
                }
                break;
            case VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = graph->optimizer_flags;
                    status = VX_SUCCESS;
                }
                break;
            case VX_GRAPH_ATTRIBUTE_AMD_AFFINITY:
                if (size == sizeof(AgoTargetAffinityInfo_)) {
                    *(AgoTargetAffinityInfo_ *)ptr = graph->attr_affinity;
                    status = VX_SUCCESS;
                }
                break;
            case VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_LAST:
                if (size == sizeof(AgoGraphPerfInternalInfo)) {
#if ENABLE_OPENCL
                    // normalize all time units into nanoseconds
                    uint64_t num = 1000000000, denom = (uint64_t)agoGetClockFrequency();
                    ((AgoGraphPerfInternalInfo *)ptr)->kernel_enqueue = graph->gpu_perf.kernel_enqueue * num / denom;
                    ((AgoGraphPerfInternalInfo *)ptr)->kernel_wait = graph->gpu_perf.kernel_wait * num / denom;
                    ((AgoGraphPerfInternalInfo *)ptr)->buffer_read = graph->gpu_perf.buffer_read * num / denom;
                    ((AgoGraphPerfInternalInfo *)ptr)->buffer_write = graph->gpu_perf.buffer_write * num / denom;
#else
                    memset(ptr, 0, size);
#endif
                    status = VX_SUCCESS;
                }
                break;
            case VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_AVG:
                if (size == sizeof(AgoGraphPerfInternalInfo)) {
#if ENABLE_OPENCL
                    if (graph->perf.num > 0) {
                        // normalize all time units into nanoseconds
                        uint64_t num = 1000000000, denom = (uint64_t)agoGetClockFrequency();
                        ((AgoGraphPerfInternalInfo *)ptr)->kernel_enqueue = (graph->gpu_perf_total.kernel_enqueue / graph->perf.num) * num / denom;
                        ((AgoGraphPerfInternalInfo *)ptr)->kernel_wait = (graph->gpu_perf_total.kernel_wait / graph->perf.num) * num / denom;
                        ((AgoGraphPerfInternalInfo *)ptr)->buffer_read = (graph->gpu_perf_total.buffer_read / graph->perf.num) * num / denom;
                        ((AgoGraphPerfInternalInfo *)ptr)->buffer_write = (graph->gpu_perf_total.buffer_write / graph->perf.num) * num / denom;
                    }
                    else
#endif
                    {
                        memset(ptr, 0, size);
                    }
                    status = VX_SUCCESS;
                }
                break;
            case VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_PROFILE:
                status = agoGraphDumpPerformanceProfile(graph, (const char *)ptr);
                break;
#if ENABLE_OPENCL
            case VX_GRAPH_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE:
                if (size == sizeof(cl_command_queue)) {
                    *(cl_command_queue *)ptr = graph->opencl_cmdq;
                    status = VX_SUCCESS;
                }
                break;
#endif
            case VX_GRAPH_ATTRIBUTE_AMD_CPU_NUM_THREADS:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = (vx_uint32)graph->cpu_num_threads;
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Allows the set to attributes on the Graph.
* \param [in] graph The reference to the graph.
* \param [in] attribute The <tt>\ref vx_graph_attribute_e</tt> type needed.
* \param [in] ptr The location from which to read the value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_graph
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetGraphAttribute(vx_graph graph, vx_enum attribute, const void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            CAgoLock lock(graph->cs);
            switch (attribute)
            {
            case VX_GRAPH_ATTRIBUTE_AMD_IMPORT_FROM_TEXT:
                if (size == sizeof(AgoGraphImportInfo)) {
                    status = VX_SUCCESS;
                    AgoGraphImportInfo * info = (AgoGraphImportInfo *)ptr;
                    if (agoReadGraphFromString(graph, info->ref, info->num_ref, info->data_registry_callback_f, info->data_registry_callback_obj, info->text, info->dumpToConsole)) {
                        status = VX_FAILURE;
                    }
                }
                break;
            case VX_GRAPH_ATTRIBUTE_AMD_EXPORT_TO_TEXT:
                if (size == sizeof(AgoGraphExportInfo)) {
                    status = VX_SUCCESS;
                    AgoGraphExportInfo * info = (AgoGraphExportInfo *)ptr;
                    FILE * fp = stdout;
                    if (strcmp(info->fileName, "stdout") != 0) {
                        fp = fopen(info->fileName, "w");
                        if (!fp) {
                            status = VX_FAILURE;
                            agoAddLogEntry(&graph->ref, status, "ERROR: vxSetGraphAttribute: unable to create: %s\n", info->fileName);
                        }
                    }
                    else if (agoWriteGraph(graph, info->ref, info->num_ref, fp, info->comment)) {
                        status = VX_FAILURE;
                    }
                    if (fp && fp != stdout) {
                        fclose(fp);
                    }
                }
                break;
            case VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS:
                if (size == sizeof(vx_uint32)) {
                    graph->optimizer_flags = *(vx_uint32 *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_GRAPH_ATTRIBUTE_AMD_AFFINITY:
                if (size == sizeof(AgoTargetAffinityInfo_)) {
                    status = VX_SUCCESS;
                    graph->attr_affinity = *(AgoTargetAffinityInfo_ *)ptr;
                }
                break;
            case VX_GRAPH_ATTRIBUTE_AMD_CPU_NUM_THREADS:
                if (size == sizeof(vx_uint32)) {
                    graph->cpu_num_threads = *(vx_uint32 *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Adds the given parameter extracted from a <tt>\ref vx_node</tt> to the graph.
* \param [in] graph The graph reference that contains the node.
* \param [in] parameter The parameter reference to add to the graph from the node.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Parameter added to Graph.
* \retval VX_ERROR_INVALID_REFERENCE The parameter is not a valid <tt>\ref vx_parameter</tt>.
* \retval VX_ERROR_INVALID_PARAMETER The parameter is of a node not in this
* graph.
* \ingroup group_graph_parameters
*/
VX_API_ENTRY vx_status VX_API_CALL vxAddParameterToGraph(vx_graph graph, vx_parameter parameter)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph) && !graph->verified) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (!parameter || (agoIsValidParameter(parameter) && parameter->scope->type == VX_TYPE_NODE)) {
            graph->parameters.push_back(parameter);
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Sets a reference to the parameter on the graph. The implementation
* must set this parameter on the originating node as well.
* \param [in] graph The graph reference.
* \param [in] index The parameter index.
* \param [in] value The reference to set to the parameter.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Parameter set to Graph.
* \retval VX_ERROR_INVALID_REFERENCE The value is not a valid <tt>\ref vx_reference</tt>.
* \retval VX_ERROR_INVALID_PARAMETER The parameter index is out of bounds or the
* dir parameter is incorrect.
* \ingroup group_graph_parameters
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetGraphParameterByIndex(vx_graph graph, vx_uint32 index, vx_reference value)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph) && !graph->verified) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if ((index < graph->parameters.size()) && graph->parameters[index] && (!value || agoIsValidReference(value))) {
            vx_parameter parameter = graph->parameters[index];
            if (((vx_node)parameter->scope)->paramList[parameter->index]) {
                agoReleaseData(((vx_node)parameter->scope)->paramList[parameter->index], false);
            }
            ((vx_node)parameter->scope)->paramList[parameter->index] = (AgoData *)value;
            if (((vx_node)parameter->scope)->paramList[parameter->index]) {
                agoRetainData(graph, ((vx_node)parameter->scope)->paramList[parameter->index], false);
            }
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Retrieves a <tt>\ref vx_parameter</tt> from a <tt>\ref vx_graph</tt>.
* \param [in] graph The graph.
* \param [in] index The index of the parameter.
* \return <tt>\ref vx_parameter</tt> reference.
* \retval 0 if the index is out of bounds.
* \retval * The parameter reference.
* \ingroup group_graph_parameters
*/
VX_API_ENTRY vx_parameter VX_API_CALL vxGetGraphParameterByIndex(vx_graph graph, vx_uint32 index)
{
    vx_parameter parameter = NULL;
    if (agoIsValidGraph(graph) && (index < graph->parameters.size())) {
        parameter = graph->parameters[index];
        parameter->ref.external_count++;
    }
    return parameter;
}

/*! \brief Returns a Boolean to indicate the state of graph verification.
* \param [in] graph The reference to the graph to check.
* \return A <tt>\ref vx_bool</tt> value.
* \retval vx_true_e The graph is verified.
* \retval vx_false_e The graph is not verified. It must be verified before
* execution either through <tt>\ref vxVerifyGraph</tt> or automatically through
* <tt>\ref vxProcessGraph</tt> or <tt>\ref vxScheduleGraph</tt>.
* \ingroup group_graph
*/
VX_API_ENTRY vx_bool VX_API_CALL vxIsGraphVerified(vx_graph graph)
{
    vx_bool verified = vx_false_e;
    if (agoIsValidGraph(graph)) {
        verified = graph->verified ? vx_true_e : vx_false_e;
    }
    return verified;
}

/*==============================================================================
NODE
=============================================================================*/

/*! \brief Creates a reference to a node object for a given kernel.
* \details This node has no references assigned as parameters after completion.
* The client is then required to set these parameters manually by <tt>\ref vxSetParameterByIndex</tt>.
* When clients supply their own node creation functions (for use with User Kernels), this is the API
* to use along with the parameter setting API.
* \param [in] graph The reference to the graph in which this node exists.
* \param [in] kernel The kernel reference to associate with this new node.
* \return vx_node
* \retval 0 The node failed to create.
* \retval * A node was created.
* \ingroup group_adv_node
* \post Call <tt>\ref vxSetParameterByIndex</tt> for as many parameters as needed to be set.
*/
VX_API_ENTRY vx_node VX_API_CALL vxCreateGenericNode(vx_graph graph, vx_kernel kernel)
{
    vx_node node = NULL;
    if (agoIsValidGraph(graph) && agoIsValidKernel(kernel) && /*!graph->verified &&*/ kernel->finalized) {
        CAgoLock lock(graph->cs);
        // graph->reverify = graph->verified;
        // graph->verified = vx_false_e;
        // graph->state = VX_GRAPH_STATE_UNVERIFIED;
        node = agoCreateNode(graph, kernel);
        node->ref.external_count++;
    }
    return node;
}

/*! \brief Allows a user to query information out of a node.
* \param [in] node The reference to the node to query.
* \param [in] attribute Use <tt>\ref vx_node_attribute_e</tt> value to query for information.
* \param [out] ptr The location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Successful
* \retval VX_ERROR_INVALID_PARAMETERS The type or size is incorrect.
* \ingroup group_node
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryNode(vx_node node, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidNode(node)) {
        CAgoLock lock(((vx_graph)node->ref.scope)->cs);
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_NODE_STATUS:
                if (size == sizeof(vx_status)) {
                    *(vx_status *)ptr = node->status;
                    status = VX_SUCCESS;
                }
                break;
            case VX_NODE_PERFORMANCE:
                if (size == sizeof(vx_perf_t)) {
                    vx_perf_t * perf = &node->perf;
                    if (node->perf.num == 0) {
                        // TBD: need mapping of node performance into its subsets or superset
                        // For now, nodes that doesn't exist in the graph will report the overall graph
                        // performance because the nodes might have got morphed into other nodes have
                        // no accountability
                        perf = &((AgoGraph *)node->ref.scope)->perf;
                    }
                    agoPerfCopyNormalize(node->ref.context, (vx_perf_t *)ptr, perf);
                    status = VX_SUCCESS;
                }
                break;
            case VX_NODE_BORDER:
                if (size == sizeof(vx_border_mode_t) || size == sizeof(vx_border_t)) {
                    *(vx_border_mode_t *)ptr = node->attr_border_mode;
                    status = VX_SUCCESS;
                }
                break;
            case VX_NODE_LOCAL_DATA_SIZE:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = node->localDataSize;
                    status = VX_SUCCESS;
                }
                break;
            case VX_NODE_LOCAL_DATA_PTR:
                if (size == sizeof(void *)) {
                    *(void **)ptr = node->localDataPtr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_NODE_PARAMETERS:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = node->paramCount;
                    status = VX_SUCCESS;
                }
                break;
            case VX_NODE_VALID_RECT_RESET:
                if (size == sizeof(vx_bool)) {
                    *(vx_bool *)ptr = node->valid_rect_reset;
                    status = VX_SUCCESS;
                }
                break;
            case VX_NODE_ATTRIBUTE_AMD_AFFINITY:
                if (size == sizeof(AgoTargetAffinityInfo_)) {
                    *(AgoTargetAffinityInfo_ *)ptr = node->attr_affinity;
                    status = VX_SUCCESS;
                }
                break;
            case VX_NODE_ATTRIBUTE_AMD_CPU_NUM_THREADS:
                if (size == sizeof(vx_uint32)) {
                    AgoGraph * graph = (AgoGraph *)node->ref.scope;
                    *(vx_uint32 *)ptr = graph->cpu_num_threads;
                    status = VX_SUCCESS;
                }
                break;
#if ENABLE_OPENCL
            case VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE:
                if (size == sizeof(cl_command_queue)) {
                    AgoGraph * graph = (AgoGraph *)node->ref.scope;
                    *(cl_command_queue *)ptr = graph->opencl_cmdq;
                    status = VX_SUCCESS;
                }
                break;
#elif ENABLE_HIP
            case VX_NODE_ATTRIBUTE_AMD_HIP_STREAM:
                if (size == sizeof(hipStream_t)){
                    AgoGraph * graph = (AgoGraph *)node->ref.scope;
                    *(hipStream_t *)ptr = graph->hip_stream0;
                    status = VX_SUCCESS;
            }
            break;
#endif
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Allows a user to set attribute of a node before Graph Validation.
* \param [in] node The reference to the node to set.
* \param [in] attribute Use <tt>\ref vx_node_attribute_e</tt> value to query for information.
* \param [out] ptr The output pointer to where to send the value.
* \param [in] size The size of the objects to which \a ptr points.
* \note Some attributes are inherited from the <tt>\ref vx_kernel</tt>, which was used
* to create the node. Some of these can be overridden using this API, notably
* \ref VX_NODE_ATTRIBUTE_LOCAL_DATA_SIZE and \ref VX_NODE_ATTRIBUTE_LOCAL_DATA_PTR.
* \ingroup group_node
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS The attribute was set.
* \retval VX_ERROR_INVALID_REFERENCE node is not a vx_node.
* \retval VX_ERROR_INVALID_PARAMETER size is not correct for the type needed.
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetNodeAttribute(vx_node node, vx_enum attribute, const void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidNode(node)) {
        CAgoLock lock(((vx_graph)node->ref.scope)->cs);
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_NODE_ATTRIBUTE_BORDER_MODE:
                if (size == sizeof(vx_border_mode_t) || size == sizeof(vx_border_t)) {
                    node->attr_border_mode = *(vx_border_mode_t *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_NODE_ATTRIBUTE_LOCAL_DATA_SIZE:
                if (node->local_data_change_is_enabled) {
                    if (size == sizeof(vx_size)) {
                        node->localDataSize = *(vx_size *)ptr;
                        node->local_data_set_by_implementation = vx_false_e;
                        status = VX_SUCCESS;
                    }
                }
                else
                {
                    status = VX_ERROR_NOT_SUPPORTED;
                }
                break;
            case VX_NODE_ATTRIBUTE_LOCAL_DATA_PTR:
                if(node->local_data_change_is_enabled) {
                    if (size == sizeof(void *)) {
                        node->localDataPtr = *(vx_uint8 **)ptr;
                        node->local_data_set_by_implementation = vx_false_e;
                        status = VX_SUCCESS;
                    }
                }
                else
                {
                    status = VX_ERROR_NOT_SUPPORTED;
                }
                break;
            case VX_NODE_ATTRIBUTE_AMD_AFFINITY:
                if (size == sizeof(AgoTargetAffinityInfo_)) {
                    node->attr_affinity = *(AgoTargetAffinityInfo_ *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Releases a reference to a Node object.
* The object may not be garbage collected until its total reference count is zero.
* \param [in] node The pointer to the reference of the node to release.
* \ingroup group_node
* \post After returning from this function the reference is zeroed.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseNode(vx_node *node)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (node && agoIsValidNode(*node)) {
        if (!agoReleaseNode(*node)) {
            *node = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Removes a Node from its parent Graph and releases it.
* \param [in] node The pointer to the node to remove and release.
* \ingroup group_node
* \post After returning from this function the reference is zeroed.
*/
VX_API_ENTRY vx_status VX_API_CALL vxRemoveNode(vx_node *node)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (node && agoIsValidNode(*node)) {
        vx_node anode = *node;
        vx_graph graph = (vx_graph)anode->ref.scope;
        CAgoLock lock(graph->cs);
        if (!graph->verified && anode->ref.external_count == 1) {
            // only remove the kernels that are created externally
            if (agoRemoveNode(&graph->nodeList, anode, true)) {
                status = VX_FAILURE;
                agoAddLogEntry(&anode->ref, status, "ERROR: vxRemoveNode: failed for %s\n", anode->akernel->name);
            }
            else {
                *node = NULL;
                status = VX_SUCCESS;
            }
        }
    }
    return status;
}

/*! \brief Assigns a callback to a node.
* If a callback already exists in this node, this function must return an error
* and the user may clear the callback by passing a NULL pointer as the callback.
* \param [in] node The reference to the node.
* \param [in] callback The callback to associate with completion of this
* specific node.
* \warning This must be used with <b><i>extreme</i></b> caution as it can \e ruin
* optimizations in the power/performance efficiency of a graph.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Callback assigned.
* \retval VX_ERROR_INVALID_REFERENCE The value passed as node was not a <tt>\ref vx_node</tt>.
* \ingroup group_node_callback
*/
VX_API_ENTRY vx_status VX_API_CALL vxAssignNodeCallback(vx_node node, vx_nodecomplete_f callback)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidNode(node)) {
        node->callback = callback;
        if (node->newchildnode) {
            node->newchildnode->callback = callback;
        }
        status = VX_SUCCESS;
    }
    return status;
}

/*! \brief Retrieves the current node callback function pointer set on the node.
* \param [in] node The reference to the <tt>\ref vx_node</tt> object.
* \ingroup group_node_callback
* \return vx_nodecomplete_f The pointer to the callback function.
* \retval NULL No callback is set.
* \retval * The node callback function.
*/
VX_API_ENTRY vx_nodecomplete_f VX_API_CALL vxRetrieveNodeCallback(vx_node node)
{
    vx_nodecomplete_f callback = NULL;
    if (agoIsValidNode(node)) {
        callback = node->callback;
    }

    return callback;
}

/*! \brief Sets the node target to the provided value. A success invalidates the graph
* that the node belongs to (<tt>\ref vxVerifyGraph</tt> must be called before the next execution)
* \param [in] node  The reference to the <tt>\ref vx_node</tt> object.
* \param [in] target_enum  The target enum to be set to the <tt>\ref vx_node</tt> object.
* Use a <tt>\ref vx_target_e</tt>.
* \param [in] target_string  The target name ASCII string. This contains a valid value
* when target_enum is set to <tt>\ref VX_TARGET_STRING</tt>, otherwise it is ignored.
* \ingroup group_node
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Node target set.
* \retval VX_ERROR_INVALID_REFERENCE If node is not a <tt>\ref vx_node</tt>.
* \retval VX_ERROR_NOT_SUPPORTED If the node kernel is not supported by the specified target.
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetNodeTarget(vx_node node, vx_enum target_enum, const char* target_string)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidNode(node)) {
        status = VX_ERROR_NOT_SUPPORTED;
        if (target_enum == VX_TARGET_ANY) {
            status = VX_SUCCESS;
        }
        else if (target_enum == VX_TARGET_STRING) {
            if (!target_string) {
                status = VX_ERROR_INVALID_REFERENCE;
            }
            else if (!_stricmp(target_string, "any")) {
                status = VX_SUCCESS;
            }
            else if (!_stricmp(target_string, "cpu")) {
                if (node->attr_affinity.device_type == 0) {
                    node->attr_affinity.device_type = AGO_TARGET_AFFINITY_CPU;
                    status = VX_SUCCESS;
                }
            }
            else if (!_stricmp(target_string, "gpu")) {
                if (node->attr_affinity.device_type == 0) {
                    node->attr_affinity.device_type = AGO_TARGET_AFFINITY_GPU;
                    status = VX_SUCCESS;
                }
            }
        }
    }
    return status;
}

/*! \brief Creates replicas of the same node first_node to process a set of objects
* stored in <tt>\ref vx_pyramid</tt> or <tt>\ref vx_object_array</tt>.
* first_node needs to have as parameter levels 0 of a <tt>\ref vx_pyramid</tt> or the index 0 of a <tt>\ref vx_object_array</tt>.
* Replica nodes are not accessible by the application through any means. An application request for removal of
* first_node from the graph will result in removal of all replicas. Any change of parameter or attribute of
* first_node will be propagated to the replicas. <tt>\ref vxVerifyGraph</tt> shall enforce consistency of parameters and attributes
* in the replicas.
* \param [in] graph The reference to the graph.
* \param [in] first_node The reference to the node in the graph that will be replicated.
* \param [in] replicate an array of size equal to the number of node parameters, vx_true_e for the parameters
* that should be iterated over (should be a reference to a vx_pyramid or a vx_object_array),
* vx_false_e for the parameters that should be the same across replicated nodes and for optional
* parameters that are not used. Should be vx_true_e for all output and bidirectional parameters.
* \param [in] number_of_parameters number of elements in the replicate array
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If the first_node is not a <tt>\ref vx_node</tt>, or it is not the first child of a vx_pyramid.
* \retval VX_ERROR_NOT_COMPATIBLE At least one of replicated parameters is not of level 0 of a pyramid or at index 0 of an object array.
* \retval VX_FAILURE If the node does not belong to the graph, or the number of objects in the parent objects of inputs and output are not the same.
* \ingroup group_node
*/
VX_API_ENTRY vx_status VX_API_CALL vxReplicateNode(vx_graph graph, vx_node first_node, vx_bool replicate[], vx_uint32 number_of_parameters)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph) && agoIsValidNode(first_node)) {
        status = VX_FAILURE;
        if (first_node->ref.scope == &graph->ref && first_node->paramCount == number_of_parameters) {
            status = VX_SUCCESS;
            AgoData ** paramList = first_node->paramList;
            vx_uint32 num_levels = 0;
            for (vx_uint32 i = 0; i < number_of_parameters; i++) {
                if (replicate[i]) {
                    if (paramList[i] && paramList[i]->parent && paramList[i]->siblingIndex == 0 && (paramList[i]->parent->ref.type == VX_TYPE_PYRAMID || paramList[i]->parent->ref.type == VX_TYPE_OBJECT_ARRAY)) {
                        if (num_levels != paramList[i]->parent->numChildren) {
                            if (num_levels == 0) {
                                num_levels = paramList[i]->parent->numChildren;
                            }
                            else {
                                status = VX_FAILURE;
                                break;
                            }
                        }
                    }
                    else {
                        status = VX_ERROR_NOT_COMPATIBLE;
                        break;
                    }
                }
            }
            if (num_levels < 2)
                status = VX_ERROR_NOT_COMPATIBLE;
            for (vx_uint32 level = 1; level < num_levels && status == VX_SUCCESS; level++) {
                vx_node node = vxCreateGenericNode(graph, first_node->akernel);
                status = vxGetStatus((vx_reference)node);
                if (status == VX_SUCCESS) {
                    for (vx_uint32 i = 0; i < number_of_parameters && status == VX_SUCCESS; i++) {
                        if (replicate[i]) {
                            AgoData * param = paramList[i]->parent->children[level];
                            if (param) {
                                status = vxSetParameterByIndex(node, i, &paramList[i]->parent->children[level]->ref);
                            }
                            else {
                                status = VX_FAILURE;
                            }
                        }
                        else if (paramList[i]) {
                            status = vxSetParameterByIndex(node, i, &paramList[i]->ref);
                        }
                    }
                }
            }
        }
    }
    return status;
}

/*==============================================================================
PARAMETER
=============================================================================*/

/*! \brief Retrieves a <tt>\ref vx_parameter</tt> from a <tt>\ref vx_node</tt>.
* \param [in] node The node from which to extract the parameter.
* \param [in] index The index of the parameter to which to get a reference.
* \return <tt>\ref vx_parameter</tt>
* \ingroup group_parameter
*/
VX_API_ENTRY vx_parameter VX_API_CALL vxGetParameterByIndex(vx_node node, vx_uint32 index)
{
    vx_parameter parameter = NULL;
    if (agoIsValidNode(node) && (index < node->paramCount)) {
        if (agoUpdateDelaySlots(node) == VX_SUCCESS) {
            parameter = &node->parameters[index];
            parameter->ref.external_count++;
        }
    }
    return parameter;
}

/*! \brief Releases a reference to a parameter object.
* The object may not be garbage collected until its total reference count is zero.
* \param [in] param The pointer to the parameter to release.
* \ingroup group_parameter
* \post After returning from this function the reference is zeroed.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseParameter(vx_parameter *param)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (param && agoIsValidParameter(*param)) {
        if ((*param)->ref.external_count > 0) {
            (*param)->ref.external_count--;
            *param = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Sets the specified parameter data for a kernel on the node.
* \param [in] node The node that contains the kernel.
* \param [in] index The index of the parameter desired.
* \param [in] value The reference to the parameter.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_parameter
* \see vxSetParameterByReference
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetParameterByIndex(vx_node node, vx_uint32 index, vx_reference value)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidNode(node)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        vx_graph graph = (AgoGraph *)node->ref.scope;
        if (graph->verified) {
            status = VX_ERROR_NOT_SUPPORTED;
        }
        else if (node->parameters[index].state == VX_PARAMETER_STATE_REQUIRED && !value) {
            status = VX_ERROR_INVALID_REFERENCE;
        }
        else if ((index < node->paramCount) && (!node->parameters[index].type || !value || node->parameters[index].type == value->type || node->parameters[index].type == VX_TYPE_REFERENCE)) {
            if (node->paramList[index]) {
                agoReleaseData(node->paramList[index], false);
            }
            AgoData * data = (AgoData *)value;
            if (data && agoIsPartOfDelay(data)) {
                // get the trace to delay object from original node parameter without vxAgeDelay changes
                int siblingTrace[AGO_MAX_DEPTH_FROM_DELAY_OBJECT], siblingTraceCount = 0;
                AgoData * delay = agoGetSiblingTraceToDelayForInit(data, siblingTrace, siblingTraceCount);
                if (delay) {
                    // get the data
                    data = agoGetDataFromTrace(delay, siblingTrace, siblingTraceCount);
                }
            }
            node->paramList[index] = node->paramListForAgeDelay[index] = data;
            if (node->paramList[index]) {
                agoRetainData((AgoGraph *)node->ref.scope, node->paramList[index], false);
            }
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Associates a parameter reference and a data reference with a kernel
* on a node.
* \param [in] parameter The reference to the kernel parameter.
* \param [in] value The value to associate with the kernel parameter.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_parameter
* \see vxGetParameterByIndex
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetParameterByReference(vx_parameter parameter, vx_reference value)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidParameter(parameter) && parameter->scope->type == VX_TYPE_NODE && parameter->ref.external_count > 0) {
        vx_node node = (vx_node)parameter->scope;
        vx_uint32 index = parameter->index;
        status = vxSetParameterByIndex(node, index, value);
    }
    return status;
}

/*! \brief Allows the client to query a parameter to determine its meta-information.
* \param [in] param The reference to the parameter.
* \param [in] attribute The attribute to query. Use a <tt>\ref vx_parameter_attribute_e</tt>.
* \param [out] ptr The location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_parameter
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryParameter(vx_parameter param, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidParameter(param)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_PARAMETER_ATTRIBUTE_DIRECTION:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = param->direction;
                    status = VX_SUCCESS;
                }
                break;
            case VX_PARAMETER_ATTRIBUTE_INDEX:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = param->index;
                    status = VX_SUCCESS;
                }
                break;
            case VX_PARAMETER_ATTRIBUTE_TYPE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = param->type;
                    status = VX_SUCCESS;
                }
                break;
            case VX_PARAMETER_ATTRIBUTE_STATE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = param->state;
                    status = VX_SUCCESS;
                }
                break;
            case VX_PARAMETER_ATTRIBUTE_REF:
                if (size == sizeof(vx_reference)) {
                    vx_node node = (vx_node)param->scope;
                    if (agoIsValidNode(node)) {
                        if (param->index < node->paramCount) {
                            vx_reference ref = (vx_reference)node->paramList[param->index];
                            *(vx_reference *)ptr = ref;
                            // TBD: handle optimized buffers and kernels
                            if (ref) {
                                ref->external_count++;
                                ref->context->num_active_references++;
                            }
                            status = VX_SUCCESS;
                        }
                    }
                    else {
                        status = VX_ERROR_NOT_SUPPORTED;
                    }
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*==============================================================================
SCALAR
=============================================================================*/

/*! \brief Creates a reference to a scalar object. Also see \ref sub_node_parameters.
* \param [in] context The reference to the system context.
* \param [in] data_type The <tt>\ref vx_type_e</tt> of the scalar. Must be greater than
* <tt>\ref VX_TYPE_INVALID</tt> and less than <tt>\ref VX_TYPE_SCALAR_MAX</tt>.
* \param [in] ptr The pointer to the initial value of the scalar.
* \ingroup group_scalar
* \return A <tt>\ref vx_scalar</tt> reference.
* \retval 0 The scalar could not be created.
* \retval * The scalar was created. Check for further errors with <tt>\ref vxGetStatus</tt>.
*/
VX_API_ENTRY vx_scalar VX_API_CALL vxCreateScalar(vx_context context, vx_enum data_type, const void *ptr)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context)) {
        CAgoLock lock(context->cs);
        vx_size size = agoType2Size(context, data_type);
        if(size > 0 || data_type == VX_TYPE_STRING_AMD) {
            data = (AgoData *)vxCreateScalarWithSize(context, data_type, ptr, size);
        }
    }
    return (vx_scalar)data;
}

/*! \brief Creates an opaque reference to a scalar object with no direct user access.
* \param [in] graph The reference to the parent graph.
* \param [in] data_type The type of data to hold. Must be greater than VX_TYPE_INVALID and less than or equal to VX_TYPE_VENDOR_STRUCT_END.
* Or must be a vx_enum returned from vxRegisterUserStruct.
* \return A scalar reference vx_scalar. Any possible errors preventing a successful creation should be checked using vxGetStatus.
*/
VX_API_ENTRY vx_scalar VX_API_CALL vxCreateVirtualScalar(vx_graph graph, vx_enum data_type)
{
    AgoData * data = NULL;
    if (agoIsValidGraph(graph)) {
        CAgoLock lock(graph->cs);
        const char * desc_type = agoEnum2Name(data_type);
        if (data_type && !desc_type) {
            desc_type = agoGetUserStructName(graph->ref.context, data_type);
        }
        if(!data_type || desc_type) {
            char desc[512];
            if (desc_type) snprintf(desc, sizeof(desc), "scalar-virtual:%s,0", desc_type);
            else snprintf(desc, sizeof(desc), "scalar-virtual:0,0");
            data = agoCreateDataFromDescription(graph->ref.context, graph, desc, true);
            if (data) {
                agoGenerateVirtualDataName(graph, "scalar", data->name);
                agoAddData(&graph->dataList, data);
            }
        }
    }
    return (vx_scalar)data;
}

/*! \brief Creates a reference to a scalar object. Also see \ref sub_node_parameters.
 * \param [in] context The reference to the system context.
 * \param [in] data_type The type of data to hold. Must be greater than
 * <tt>\ref VX_TYPE_INVALID</tt> and less than or equal to <tt>\ref VX_TYPE_VENDOR_STRUCT_END</tt>.
 * Or must be a <tt>\ref vx_enum</tt> returned from <tt>\ref vxRegisterUserStruct</tt>.
 * \param [in] ptr The pointer to the initial value of the scalar.
 * \param [in] size Size of data at ptr in bytes.
 * \ingroup group_scalar
 * \returns A scalar reference <tt>\ref vx_scalar</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_scalar VX_API_CALL vxCreateScalarWithSize(vx_context context, vx_enum data_type, const void *ptr, vx_size size)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context)) {
        CAgoLock lock(context->cs);
        vx_size data_size = 0;
        if(data_type != VX_TYPE_STRING_AMD) {
            data_size = agoType2Size(context, data_type);
            if(data_size == 0 || data_size != size) {
                return NULL;
            }
        }
        data = agoCreateDataFromDescription(context, NULL, "scalar:UINT32,0", true);
        if (data) {
            agoAddData(&context->dataList, data);
            data->u.scalar.type = data_type;
            data->u.scalar.itemsize = size;
            switch (data_type) {
            case VX_TYPE_ENUM:
                if (ptr) data->u.scalar.u.e = *(vx_enum *)ptr;
                break;
            case VX_TYPE_UINT32:
                if (ptr) data->u.scalar.u.u = *(vx_uint32 *)ptr;
                break;
            case VX_TYPE_INT32:
                if (ptr) data->u.scalar.u.i = *(vx_int32 *)ptr;
                break;
            case VX_TYPE_UINT16:
                if (ptr) data->u.scalar.u.u = *(vx_uint16 *)ptr;
                break;
            case VX_TYPE_INT16:
                if (ptr) data->u.scalar.u.i = *(vx_int16 *)ptr;
                break;
            case VX_TYPE_UINT8:
                if (ptr) data->u.scalar.u.u = *(vx_uint8 *)ptr;
                break;
            case VX_TYPE_INT8:
                if (ptr) data->u.scalar.u.i = *(vx_int8 *)ptr;
                break;
            case VX_TYPE_CHAR:
                if (ptr) data->u.scalar.u.i = *(vx_char *)ptr;
                break;
            case VX_TYPE_FLOAT32:
                if (ptr) data->u.scalar.u.f = *(vx_float32 *)ptr;
                break;
            case VX_TYPE_FLOAT16:
                if (ptr) data->u.scalar.u.u = *(vx_uint16 *)ptr;
                break;
            case VX_TYPE_SIZE:
                if (ptr) data->u.scalar.u.s = *(vx_size *)ptr;
                break;
            case VX_TYPE_BOOL:
                if (ptr) data->u.scalar.u.u = *(vx_bool *)ptr;
                break;
            case VX_TYPE_DF_IMAGE:
                if (ptr) data->u.scalar.u.df = *(vx_df_image *)ptr;
                break;
            case VX_TYPE_FLOAT64:
                if (ptr) data->u.scalar.u.f64 = *(vx_float64 *)ptr;
                break;
            case VX_TYPE_INT64:
                if (ptr) data->u.scalar.u.i64 = *(vx_int64 *)ptr;
                break;
            case VX_TYPE_UINT64:
                if (ptr) data->u.scalar.u.u64 = *(vx_uint64 *)ptr;
                break;
            default:
                if(data_type == VX_TYPE_STRING_AMD) {
                    data->u.scalar.itemsize = sizeof(char *);
                    data->size = VX_MAX_STRING_BUFFER_SIZE_AMD;
                }
                else {
                    data->u.scalar.itemsize = data->size = data_size;
                }
                data->buffer_allocated = data->buffer = (vx_uint8 *)agoAllocMemory(data->size);
                if (data->buffer) {
                    memset(data->buffer, 0, data->size);
                    if (ptr) {
                        if(data_type == VX_TYPE_STRING_AMD) {
                            strncpy((char *)data->buffer, (const char *)ptr, VX_MAX_STRING_BUFFER_SIZE_AMD);
                            data->buffer[VX_MAX_STRING_BUFFER_SIZE_AMD - 1] = 0; // NUL terminate string in case of overflow
                        }
                        else {
                            memcpy(data->buffer, ptr, size);
                        }
                    }
                    data->isInitialized = vx_true_e;
                }
                else {
                    agoReleaseData(data, true);
                    data = NULL;
                }
                break;
            }
        }
    }
    return (vx_scalar)data;
}

/*! \brief Releases a reference to a scalar object.
* The object may not be garbage collected until its total reference count is zero.
* \param [in] scalar The pointer to the scalar to release.
* \ingroup group_scalar
* \post After returning from this function the reference is zeroed.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseScalar(vx_scalar *scalar)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (scalar && agoIsValidData((AgoData *)*scalar, VX_TYPE_SCALAR)) {
        if (!agoReleaseData((AgoData *)*scalar, true)) {
            *scalar = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Queries attributes from a scalar.
* \param [in] scalar The scalar object.
* \param [in] attribute The enumeration to query. Use a <tt>\ref vx_scalar_attribute_e</tt> enumeration.
* \param [out] ptr The location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_scalar
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryScalar(vx_scalar scalar, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)scalar;
    if (agoIsValidData(data, VX_TYPE_SCALAR)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_SCALAR_ATTRIBUTE_TYPE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = data->u.scalar.type;
                    status = VX_SUCCESS;
                }
                break;
            case VX_SCALAR_BUFFER:
                if (size == sizeof(vx_uint8 *)) {
                    if (data->buffer) {
                        *(vx_uint8 **)ptr = data->buffer;
                    } else {
                        *(vx_uint8 **)ptr = NULL;
                    }
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Gets the scalar value out of a reference.
* \note Use this in conjunction with Query APIs that return references which
* should be converted into values.
* \ingroup group_scalar
* \param [in] ref The reference from which to get the scalar value.
* \param [out] ptr An appropriate typed pointer that points to a location to which to copy
* the scalar value.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_INVALID_REFERENCE If the ref is not a valid
* reference.
* \retval VX_ERROR_INVALID_PARAMETERS If \a ptr is NULL.
* \retval VX_ERROR_INVALID_TYPE If the type does not match the type in the reference or is a bad value.
*/
VX_API_ENTRY vx_status VX_API_CALL vxReadScalarValue(vx_scalar ref, void *ptr)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)ref;
    if (agoIsValidData(data, VX_TYPE_SCALAR)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            status = VX_SUCCESS;
            switch (data->u.scalar.type)
            {
            case VX_TYPE_ENUM:
                *(vx_enum *)ptr = data->u.scalar.u.e;
                break;
            case VX_TYPE_UINT32:
                *(vx_uint32 *)ptr = data->u.scalar.u.u;
                break;
            case VX_TYPE_INT32:
                *(vx_int32 *)ptr = data->u.scalar.u.i;
                break;
            case VX_TYPE_UINT16:
                *(vx_uint16 *)ptr = data->u.scalar.u.u;
                break;
            case VX_TYPE_INT16:
                *(vx_int16 *)ptr = data->u.scalar.u.i;
                break;
            case VX_TYPE_UINT8:
                *(vx_uint8 *)ptr = data->u.scalar.u.u;
                break;
            case VX_TYPE_INT8:
                *(vx_int8 *)ptr = data->u.scalar.u.i;
                break;
            case VX_TYPE_CHAR:
                *(vx_char *)ptr = data->u.scalar.u.i;
                break;
            case VX_TYPE_FLOAT32:
                *(vx_float32 *)ptr = data->u.scalar.u.f;
                break;
            case VX_TYPE_SIZE:
                *(vx_size *)ptr = data->u.scalar.u.s;
                break;
            case VX_TYPE_BOOL:
                *(vx_bool *)ptr = data->u.scalar.u.u ? vx_true_e : vx_false_e;
                break;
            case VX_TYPE_DF_IMAGE:
                *(vx_df_image *)ptr = data->u.scalar.u.df;
                break;
            case VX_TYPE_FLOAT64:
                *(vx_float64 *)ptr = data->u.scalar.u.f64;
                break;
            case VX_TYPE_UINT64:
                *(vx_uint64 *)ptr = data->u.scalar.u.u64;
                break;
            case VX_TYPE_INT64:
                *(vx_int64 *)ptr = data->u.scalar.u.i64;
                break;
            case VX_TYPE_STRING_AMD:
                strcpy((char *)ptr, (const char *)data->buffer);
                break;
            default:
                if (data->buffer) {
                    memcpy(ptr, data->buffer, data->size);
                    break;
                }
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Sets the scalar value in a reference.
* \note Use this in conjunction with Parameter APIs that return references
* to parameters that need to be altered.
* \ingroup group_scalar
* \param [in] ref The reference from which to get the scalar value.
* \param [in] ptr An appropriately typed pointer that points to a location to which to copy
* the scalar value.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_INVALID_REFERENCE If the ref is not a valid
* reference.
* \retval VX_ERROR_INVALID_PARAMETERS If \a ptr is NULL.
* \retval VX_ERROR_INVALID_TYPE If the type does not match the type in the reference or is a bad value.
*/
VX_API_ENTRY vx_status VX_API_CALL vxWriteScalarValue(vx_scalar ref, const void *ptr)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)ref;
    if (agoIsValidData(data, VX_TYPE_SCALAR) && !data->isVirtual) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            // TBD: need sem-lock for thread safety
            status = VX_SUCCESS;
            switch (data->u.scalar.type)
            {
            case VX_TYPE_ENUM:
                data->u.scalar.u.e = *(vx_enum *)ptr;
                data->isInitialized = vx_true_e;
                break;
            case VX_TYPE_UINT32:
                data->u.scalar.u.u = *(vx_uint32 *)ptr;
                data->isInitialized = vx_true_e;
                break;
            case VX_TYPE_INT32:
                data->u.scalar.u.i = *(vx_int32 *)ptr;
                data->isInitialized = vx_true_e;
                break;
            case VX_TYPE_UINT16:
                data->u.scalar.u.u = *(vx_uint16 *)ptr;
                data->isInitialized = vx_true_e;
                break;
            case VX_TYPE_INT16:
                data->u.scalar.u.i = *(vx_int16 *)ptr;
                data->isInitialized = vx_true_e;
                break;
            case VX_TYPE_UINT8:
                data->u.scalar.u.u = *(vx_uint8 *)ptr;
                data->isInitialized = vx_true_e;
                break;
            case VX_TYPE_INT8:
                data->u.scalar.u.i = *(vx_int8 *)ptr;
                data->isInitialized = vx_true_e;
                break;
            case VX_TYPE_CHAR:
                data->u.scalar.u.i = *(vx_char *)ptr;
                data->isInitialized = vx_true_e;
                break;
            case VX_TYPE_FLOAT32:
                data->u.scalar.u.f = *(vx_float32 *)ptr;
                data->isInitialized = vx_true_e;
                break;
            case VX_TYPE_SIZE:
                data->u.scalar.u.s = *(vx_size *)ptr;
                data->isInitialized = vx_true_e;
                break;
            case VX_TYPE_BOOL:
                data->u.scalar.u.u = *(vx_bool *)ptr;
                data->isInitialized = vx_true_e;
                break;
            case VX_TYPE_DF_IMAGE:
                data->u.scalar.u.df = *(vx_df_image *)ptr;
                data->isInitialized = vx_true_e;
                break;
            case VX_TYPE_FLOAT64:
                data->u.scalar.u.f64 = *(vx_float64 *)ptr;
                data->isInitialized = vx_true_e;
                break;
            case VX_TYPE_UINT64:
                data->u.scalar.u.u64 = *(vx_uint64 *)ptr;
                data->isInitialized = vx_true_e;
                break;
            case VX_TYPE_INT64:
                data->u.scalar.u.i64 = *(vx_int64 *)ptr;
                data->isInitialized = vx_true_e;
                break;
            case VX_TYPE_STRING_AMD:
                strncpy((char *)data->buffer, (const char *)ptr, VX_MAX_STRING_BUFFER_SIZE_AMD);
                data->buffer[VX_MAX_STRING_BUFFER_SIZE_AMD - 1] = 0; // NUL terminate string in case of overflow
                data->isInitialized = vx_true_e;
                break;
            default:
                if (ptr) {
                    memcpy(data->buffer,ptr, data->size);
                    break;
                }
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Allows the application to copy from/into a scalar object.
* \param [in] scalar The reference to the scalar object that is the source or the
* destination of the copy.
* \param [in] user_ptr The address of the memory location where to store the requested data
* if the copy was requested in read mode, or from where to get the data to store into the
* scalar object if the copy was requested in write mode. In the user memory, the scalar is
* a variable of the type corresponding to <tt>\ref VX_SCALAR_TYPE</tt>.
* The accessible memory must be large enough to contain this variable.
* \param [in] usage This declares the effect of the copy with regard to the scalar object
* using the <tt>\ref vx_accessor_e</tt> enumeration. Only VX_READ_ONLY and VX_WRITE_ONLY
* are supported:
* \arg VX_READ_ONLY means that data are copied from the scalar object into the user memory.
* \arg VX_WRITE_ONLY means that data are copied into the scalar object from the user memory.
* \param [in] user_mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that specifies
* the memory type of the memory referenced by the user_addr.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_INVALID_REFERENCE The scalar reference is not actually a scalar reference.
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \ingroup group_scalar
*/
VX_API_ENTRY vx_status VX_API_CALL vxCopyScalar(vx_scalar scalar_, void *user_ptr, vx_enum usage, vx_enum user_mem_type)
{
    AgoData * scalar = (AgoData *)scalar_;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(scalar, VX_TYPE_SCALAR))
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        if ((user_mem_type == VX_MEMORY_TYPE_HOST) && user_ptr) {
            if (usage == VX_READ_ONLY)
                status = vxReadScalarValue(scalar_, user_ptr);
            else if (usage == VX_WRITE_ONLY)
                status = vxWriteScalarValue(scalar_, user_ptr);
        }
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxCopyScalarWithSize(vx_scalar scalar_, vx_size size, void *user_ptr, vx_enum usage, vx_enum user_mem_type)
{
    AgoData * scalar = (AgoData *)scalar_;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(scalar, VX_TYPE_SCALAR))
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        if ((user_mem_type == VX_MEMORY_TYPE_HOST) && user_ptr && (scalar->u.scalar.itemsize == size)) {
            if (usage == VX_READ_ONLY)
                status = vxReadScalarValue(scalar_, user_ptr);
            else if (usage == VX_WRITE_ONLY)
                status = vxWriteScalarValue(scalar_, user_ptr);
        }
    }
    return status;
}

/*==============================================================================
REFERENCE
=============================================================================*/

/*! \brief Queries any reference type for some basic information (count, type).
* \param [in] ref The reference to query.
* \param [in] attribute The value for which to query. Use <tt>\ref vx_reference_attribute_e</tt>.
* \param [out] ptr The location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_reference
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryReference(vx_reference ref, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidReference(ref)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_REFERENCE_COUNT:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = ref->external_count;
                    status = VX_SUCCESS;
                }
                break;
            case VX_REFERENCE_TYPE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = ref->type;
                    status = VX_SUCCESS;
                }
                break;
            case VX_REFERENCE_NAME:
                if (size == sizeof(vx_char*)) {
                    if(ref->type == VX_TYPE_GRAPH)
                    {
                        AgoGraph * graph = (AgoGraph *)ref;
                        //strncpy((char *)ptr, data->name.c_str(), size);
                        *(vx_char**)ptr = &graph->name[0];
                        status = VX_SUCCESS;
                    }
                    else
                    {
                        AgoData * data = (AgoData *)ref;
                        //strncpy((char *)ptr, data->name.c_str(), size);
                        *(vx_char**)ptr = &data->name[0];
                        status = VX_SUCCESS;
                    }
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Releases a reference. The reference may potentially refer to multiple OpenVX objects of different types.
* This function can be used instead of calling a specific release function for each individual object type
* (e.g. vxRelease<object>). The object will not be destroyed until its total reference count is zero.
* \note After returning from this function the reference is zeroed.
* \param [in] ref_ptr The pointer to the reference of the object to release.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If the reference is not valid.
* \ingroup group_reference
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseReference(vx_reference* ref_ptr)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (ref_ptr) {
        vx_reference ref = *ref_ptr;
        if (agoIsValidReference(ref)) {
            switch (ref->type) {
            case VX_TYPE_CONTEXT:
                status = vxReleaseContext((vx_context *)ref_ptr);
                break;
            case VX_TYPE_GRAPH:
                status = vxReleaseGraph((vx_graph *)ref_ptr);
                break;
            case VX_TYPE_NODE:
                status = vxReleaseNode((vx_node *)ref_ptr);
                break;
            case VX_TYPE_KERNEL:
                status = vxReleaseKernel((vx_kernel *)ref_ptr);
                break;
            case VX_TYPE_PARAMETER:
                status = vxReleaseParameter((vx_parameter *)ref_ptr);
                break;
            case VX_TYPE_DELAY:
                status = vxReleaseDelay((vx_delay *)ref_ptr);
                break;
            case VX_TYPE_LUT:
                status = vxReleaseLUT((vx_lut *)ref_ptr);
                break;
            case VX_TYPE_DISTRIBUTION:
                status = vxReleaseDistribution((vx_distribution *)ref_ptr);
                break;
            case VX_TYPE_PYRAMID:
                status = vxReleasePyramid((vx_pyramid *)ref_ptr);
                break;
            case VX_TYPE_THRESHOLD:
                status = vxReleaseThreshold((vx_threshold *)ref_ptr);
                break;
            case VX_TYPE_MATRIX:
                status = vxReleaseMatrix((vx_matrix *)ref_ptr);
                break;
            case VX_TYPE_CONVOLUTION:
                status = vxReleaseConvolution((vx_convolution *)ref_ptr);
                break;
            case VX_TYPE_SCALAR:
                status = vxReleaseScalar((vx_scalar *)ref_ptr);
                break;
            case VX_TYPE_ARRAY:
                status = vxReleaseArray((vx_array *)ref_ptr);
                break;
            case VX_TYPE_OBJECT_ARRAY:
                status = vxReleaseObjectArray((vx_object_array *)ref_ptr);
                break;
            case VX_TYPE_IMAGE:
                status = vxReleaseImage((vx_image *)ref_ptr);
                break;
            case VX_TYPE_REMAP:
                status = vxReleaseRemap((vx_remap *)ref_ptr);
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*!
* \brief Increments the reference counter of an object
* This function is used to express the fact that the OpenVX object is referenced
* multiple times by an application. Each time this function is called for
* an object, the application will need to release the object one additional
* time before it can be destructed
* \param [in] ref The reference to retain.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE if reference is not valid.
* \ingroup group_reference
*/
VX_API_ENTRY vx_status VX_API_CALL vxRetainReference(vx_reference ref)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidReference(ref)) {
        ref->external_count++;
        if (ref->type != VX_TYPE_KERNEL && ref->type != VX_TYPE_NODE && ref->type != VX_TYPE_PARAMETER)
            ref->context->num_active_references++;
        status = VX_SUCCESS;
    }
    return status;
}

/*! \brief Name a reference
* \ingroup group_reference
*
* This function is used to associate a name to a referenced object. This name
* can be used by the OpenVX implementation in log messages and any
* other reporting mechanisms.
*
* The OpenVX implementation will not check if the name is unique in
* the reference scope (context or graph). Several references can then
* have the same name.
*
* \param [in] ref The reference to the object to be named.
* \param [in] name Pointer to the '\0' terminated string that identifies
*             the referenced object.
*             The string is copied by the function so that it
*             stays the property of the caller.
*             NULL means that the reference is not named.
*             The length of the string shall be lower than VX_MAX_REFERENCE_NAME bytes.
* \return A \ref vx_status_e enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If reference is not valid.
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetReferenceName(vx_reference ref, const vx_char *name)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidReference(ref) && ((ref->type >= VX_TYPE_DELAY && ref->type <= VX_TYPE_REMAP) ||
        (ref->type == VX_TYPE_TENSOR) ||
        (ref->type >= VX_TYPE_VENDOR_OBJECT_START && ref->type <= VX_TYPE_VENDOR_OBJECT_END)))
    {
        AgoData * data = (AgoData *)ref;
        //printf("%s %s %lu\n", data->name.c_str(), name, strlen(name));
        //printf("before:::strlen(data name) = %lu\n", data->name.length());
        //data->name.assign(name, strlen(name));
        data->name = name;
        //std::copy(name, name + strlen(name), std::back_inserter(data->name));
        //strncpy((char *)data->name.c_str(), name, strnlen(name, VX_MAX_REFERENCE_NAME));
        //data->name.assign("name", 4);
        //sscanf(name, "%s", (char *)data->name.c_str());
        //printf("after:::strlen(data name) = %lu\n", data->name.length());
        //printf("after == %s\n", data->name.c_str());

        status = VX_SUCCESS;
    }
    else if(agoIsValidReference(ref) && (ref->type == VX_TYPE_GRAPH))
    {
        AgoGraph * graph = (AgoGraph *)ref;
        graph->name = name;
        status = VX_SUCCESS;
    }
    return status;
}

/*==============================================================================
DELAY
=============================================================================*/

/*! \brief Queries a <tt>\ref vx_delay</tt> object attribute.
* \param [in] delay The coordinates object to set.
* \param [in] attribute The attribute to query. Use a <tt>\ref vx_delay_attribute_e</tt> enumeration.
* \param [out] ptr The location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_delay
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryDelay(vx_delay delay, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)delay;
    if (agoIsValidData(data, VX_TYPE_DELAY)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_DELAY_ATTRIBUTE_TYPE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = data->u.delay.type;
                    status = VX_SUCCESS;
                }
                break;
            case VX_DELAY_ATTRIBUTE_SLOTS:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->u.delay.count;
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Releases a reference to a delay object.
* The object may not be garbage collected until its total reference count is zero.
* \param [in] delay The pointer to the delay to release.
* \post After returning from this function the reference is zeroed.
* \ingroup group_delay
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseDelay(vx_delay *delay)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (delay && agoIsValidData((AgoData*)*delay, VX_TYPE_DELAY)) {
        if (!agoReleaseData((AgoData*)*delay, true)) {
            *delay = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Creates a Delay object.
* \details This function  uses only the metadata from the exemplar, ignoring the object
* data. It does not alter the exemplar or keep or release the reference to the
* exemplar.
* \param [in] context The reference to the system context.
* \param [in] exemplar The exemplar object.
* \param [in] count The number of reference in the delay.
* \return <tt>\ref vx_delay</tt>
* \ingroup group_delay
*/
VX_API_ENTRY vx_delay VX_API_CALL vxCreateDelay(vx_context context,
    vx_reference exemplar,
    vx_size slots)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context) && agoIsValidReference(exemplar) && slots > 0) {
        CAgoLock lock(context->cs);
        char desc_exemplar[MAX_DESCRIPTION_DATA_SIZE]; agoGetDescriptionFromData(context, desc_exemplar, (AgoData *)exemplar);
        char desc[2048]; snprintf(desc, sizeof(desc), "delay:" VX_FMT_SIZE ",[%s]", slots, desc_exemplar);
        data = agoCreateDataFromDescription(context, NULL, desc, true);
        if (data) {
            agoGenerateDataName(context, "delay", data->name);
            agoAddData(&context->dataList, data);
            // add the children too
            for (vx_uint32 i = 0; i < data->numChildren; i++) {
                agoAddData(&context->dataList, data->children[i]);
                for (vx_uint32 j = 0; j < data->children[i]->numChildren; j++) {
                    if (data->children[i]->children[j]) {
                        agoAddData(&context->dataList, data->children[i]->children[j]);
                        for (vx_uint32 k = 0; k < data->children[i]->children[j]->numChildren; k++) {
                            if (data->children[i]->children[j]->children[k]) {
                                agoAddData(&context->dataList, data->children[i]->children[j]->children[k]);
                            }
                        }
                    }
                }
            }
        }
    }
    return (vx_delay)data;
}

/*! \brief Retrieves a reference from a delay object.
* \param [in] delay The reference to the delay object.
* \param [in] index An index into the delay from which to extract the
* reference.
* \return <tt>\ref vx_reference</tt>
* \note The delay index is in the range \f$ [-count+1,0] \f$. 0 is always the
* \e current object.
* \ingroup group_delay
* \note A reference from a delay object must not be given to its associated
* release API (e.g. <tt>\ref vxReleaseImage</tt>). Use the <tt>\ref vxReleaseDelay</tt> only.
*/
VX_API_ENTRY vx_reference VX_API_CALL vxGetReferenceFromDelay(vx_delay delay, vx_int32 index)
{
    AgoData * data = (AgoData *)delay;
    AgoData * item = NULL;
    if (agoIsValidData(data, VX_TYPE_DELAY)) {
        // convert the index from 0..-(N-1) to 0..N-1
        vx_uint32 index_inverted = (vx_uint32)-index;
        if (index_inverted < data->u.delay.count) {
            item = data->children[index_inverted];
        }
    }
    return (vx_reference)item;
}

/*! \brief Ages the internal delay ring by one. This means that once this API is
* called the reference from index 0 will go to index -1 and so forth until
* \f$ -count+1 \f$ is reached. This last object will become 0. Once the delay has
* been aged, it updates the reference in any associated nodes.
* \param [in] delay
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Delay was aged.
* \retval VX_ERROR_INVALID_REFERENCE The value passed as delay was not a <tt>\ref vx_delay</tt>.
* \ingroup group_delay
*/
VX_API_ENTRY vx_status VX_API_CALL vxAgeDelay(vx_delay delay)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)delay;
    if (agoIsValidData(data, VX_TYPE_DELAY)) {
        status = agoAgeDelay(data);
    }
    return status;
}

/*! \brief Register a delay for auto-aging.
*
* This function registers a delay object to be auto-aged by the graph.
* This delay object will be automatically aged after each successful completion of
* this graph. Aging of a delay object cannot be called during graph execution.
* A graph abandoned due to a node callback will trigger an auto-aging.
*
* If a delay is registered for auto-aging multiple times in a same graph,
* the delay will be only aged a single time at each graph completion.
* If a delay is registered for auto-aging in multiple graphs, this delay will
* aged automatically after each successful completion of any of these graphs.
*
* \param [in] graph The graph to which the delay is registered for auto-aging.
* \param [in] delay The delay to automatically age.
*
* \return A \ref vx_status_e enumeration.
* \retval VX_SUCCESS                   No errors.
* \retval VX_ERROR_INVALID_REFERENCE   If the \p graph or \p delay is not a valid reference
* \ingroup group_graph
*/
VX_API_ENTRY vx_status VX_API_CALL vxRegisterAutoAging(vx_graph graph, vx_delay delay_)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * delay = (AgoData *)delay_;
    if (agoIsValidGraph(graph) && agoIsValidData(delay, VX_TYPE_DELAY))
    {
        for (auto it = graph->autoAgeDelayList.begin(); it != graph->autoAgeDelayList.end(); it++) {
            if (*it == delay) {
                delay = nullptr;
                break;
            }
        }
        if (delay) {
            delay->ref.internal_count++;
            graph->autoAgeDelayList.push_back(delay);
        }
        status = VX_SUCCESS;
    }
    return status;
}

/*==============================================================================
LOGGING
=============================================================================*/

/*! \brief Adds a line to the log.
* \param [in] ref The reference to add the log entry against. Some valid value must be provided.
* \param [in] status The status code. <tt>\ref VX_SUCCESS</tt> status entries are ignored and not added.
* \param [in] message The human readable message to add to the log.
* \param [in] ... a list of variable arguments to the message.
* \note Messages may not exceed <tt>\ref VX_MAX_LOG_MESSAGE_LEN</tt> bytes and will be truncated in the log if they exceed this limit.
* \ingroup group_log
*/
VX_API_ENTRY void VX_API_CALL vxAddLogEntry(vx_reference ref, vx_status status, const char *message, ...)
{
    va_list ap;
    if (agoIsValidReference(ref) && ref->enable_logging && ref->context->callback_log) {
        vx_char string[VX_MAX_LOG_MESSAGE_LEN];
        va_start(ap, message);
        vsnprintf(string, VX_MAX_LOG_MESSAGE_LEN, message, ap);
        string[VX_MAX_LOG_MESSAGE_LEN - 1] = 0; // for MSVC which is not C99 compliant
        va_end(ap);
        if (!ref->context->callback_reentrant) {
            CAgoLock lock(ref->context->cs); // TBD: create a separate lock object for log_callback
            ref->context->callback_log(ref->context, ref, status, string);
        }
        else {
            ref->context->callback_log(ref->context, ref, status, string);
        }
    }
}

/*! \brief Registers a callback facility to the OpenVX implementation to receive error logs.
* \param [in] context The overall context to OpenVX.
* \param [in] callback The callback function. If NULL, the previous callback is removed.
* \param [in] reentrant If reentrancy flag is <tt>\ref vx_true_e</tt>, then the callback may be entered from multiple
* simultaneous tasks or threads (if the host OS supports this).
* \ingroup group_log
*/
VX_API_ENTRY void VX_API_CALL vxRegisterLogCallback(vx_context context, vx_log_callback_f callback, vx_bool reentrant)
{
    agoRegisterLogCallback(context, callback, reentrant);
}

/*==============================================================================
LUT
=============================================================================*/

/*! \brief Creates LUT object of a given type. The value of <tt>\ref VX_LUT_OFFSET</tt> is equal to 0
* for data_type = <tt>\ref VX_TYPE_UINT8</tt>, and (vx_uint32)(count/2) for <tt>\ref VX_TYPE_INT16</tt>.
* \param [in] context The reference to the context.
* \param [in] data_type The type of data stored in the LUT.
* \param [in] count The number of entries desired.
* \if OPENVX_STRICT_1_0
* \note For OpenVX 1.0, data_type can only be \ref VX_TYPE_UINT8 or \ref VX_TYPE_INT16. If data_type
* is \ref VX_TYPE_UINT8, count should be not greater than 256. If data_type is \ref VX_TYPE_INT16,
* count should not be greater than 65536.
* \endif
* \returns An LUT reference <tt>\ref vx_lut</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
* \ingroup group_lut
*/
VX_API_ENTRY vx_lut VX_API_CALL vxCreateLUT(vx_context context, vx_enum data_type, vx_size count)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context)) {
        CAgoLock lock(context->cs);
        char desc[512]; snprintf(desc, sizeof(desc), "lut:%s," VX_FMT_SIZE "", agoEnum2Name(data_type), count);
        data = agoCreateDataFromDescription(context, NULL, desc, true);
        if (data) {
            agoGenerateDataName(context, "lut", data->name);
            agoAddData(&context->dataList, data);
        }
    }
    return (vx_lut)data;
}

/*! \brief Creates an opaque reference to a LUT object with no direct user access
* \param [in] graph The reference to the parent graph.
* \param [in] data_type The type of data stored in the LUT.
* \param [in] count The number of entries desired.
* \returns An LUT reference <tt>\ref vx_lut</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
* \ingroup group_lut
*/
VX_API_ENTRY vx_lut VX_API_CALL vxCreateVirtualLUT(vx_graph graph, vx_enum data_type, vx_size count)
{
    AgoData * data = NULL;
    if (agoIsValidGraph(graph)) {
        CAgoLock lock(graph->cs);
        char desc[512]; snprintf(desc, sizeof(desc), "lut-virtual:%s," VX_FMT_SIZE "", agoEnum2Name(data_type), count);
        data = agoCreateDataFromDescription(graph->ref.context, graph, desc, true);
        if (data) {
            agoGenerateVirtualDataName(graph, "lut", data->name);
            agoAddData(&graph->dataList, data);
        }
    }
    return (vx_lut)data;
}

/*! \brief Releases a reference to a LUT object.
* The object may not be garbage collected until its total reference count is zero.
* \param [in] lut The pointer to the LUT to release.
* \post After returning from this function the reference is zeroed.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
* \ingroup group_lut
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseLUT(vx_lut *lut)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (lut && agoIsValidData((AgoData*)*lut, VX_TYPE_LUT)) {
        if (!agoReleaseData((AgoData*)*lut, true)) {
            *lut = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Queries attributes from a LUT.
* \param [in] lut The LUT to query.
* \param [in] attribute The attribute to query. Use a <tt>\ref vx_lut_attribute_e</tt> enumeration.
* \param [out] ptr The location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_lut
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryLUT(vx_lut lut, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)lut;
    if (agoIsValidData(data, VX_TYPE_LUT)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_LUT_TYPE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = data->u.lut.type;
                    status = VX_SUCCESS;
                }
                break;
            case VX_LUT_COUNT:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->u.lut.count;
                    status = VX_SUCCESS;
                }
                break;
            case VX_LUT_SIZE:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->size;
                    status = VX_SUCCESS;
                }
                break;
            case VX_LUT_OFFSET:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = data->u.lut.offset;
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Gets direct access to the LUT table data.
* \details There are several variations of call methodology:
* \arg If \a ptr is NULL (which means the current data of the LUT is not desired),
* the LUT reference count is incremented.
* \arg If \a ptr is not NULL but (*ptr) is NULL, (*ptr) will contain the address of the LUT data when the function returns and
* the reference count will be incremented. Whether the (*ptr) address is mapped
* or allocated is undefined. (*ptr) must be returned to <tt>\ref vxCommitLUT</tt>.
* \arg If \a ptr is not NULL and (*ptr) is not NULL, the user is signalling the implementation to copy the LUT data into the location specified
* by (*ptr). Users must use <tt>\ref vxQueryLUT</tt> with <tt>\ref VX_LUT_ATTRIBUTE_SIZE</tt> to
* determine how much memory to allocate for the LUT data.
*
* In any case, <tt>\ref vxCommitLUT</tt> must be called after LUT access is complete.
* \param [in] lut The LUT from which to get the data.
* \param [in,out] ptr The address of the location to store the pointer to the LUT memory.
* \param [in] usage This declares the intended usage of the pointer using the * <tt>\ref vx_accessor_e</tt> enumeration.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \post <tt>\ref vxCommitLUT</tt>
* \ingroup group_lut
*/
VX_API_ENTRY vx_status VX_API_CALL vxAccessLUT(vx_lut lut, void **ptr, vx_enum usage)
{
    AgoData * data = (AgoData *)lut;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_LUT)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if (ptr) {
            if (!data->buffer) {
                CAgoLock lock(data->ref.context->cs);
                if (agoAllocData(data)) {
                    return VX_FAILURE;
                }
            }
            vx_uint8 * ptr_internal = data->buffer;
            vx_uint8 * ptr_returned = *ptr ? (vx_uint8 *)*ptr : ptr_internal;
            // save the pointer and usage for use in vxCommitXXX
            status = VX_SUCCESS;
            for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
                if (i->ptr == ptr_returned) {
                    // can't support vxAccessXXX() more than once with same pointer, the application
                    // needs to call vxCommitXXX() before calling vxAccessXXX()
                    status = VX_FAILURE;
                }
            }
            if (status == VX_SUCCESS) {
                MappedData item = { data->nextMapId++, ptr_returned, usage, (ptr_returned != ptr_internal) ? true : false };
                data->mapped.push_back(item);
                *ptr = ptr_returned;
                if (usage == VX_READ_ONLY || usage == VX_READ_AND_WRITE) {
#if ENABLE_OPENCL
                    if (data->opencl_buffer && !(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                        // make sure dirty OpenCL buffers are synched before giving access for read
                        if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                            size_t origin[3] = { 0, 0, 0 };
                            size_t region[3] = { 256, 1, 1 };
                            cl_int err = clEnqueueReadImage(data->ref.context->opencl_cmdq, data->opencl_buffer, CL_TRUE, origin, region, 256, 0, data->buffer, 0, NULL, NULL);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&data->ref, status, "ERROR: vxAccessLUT: clEnqueueWriteImage() => %d\n", err);
                                return status;
                            }
                            data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                        }
                    }
#elif ENABLE_HIP
                    if (data->hip_memory && !(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                        // make sure dirty OpenCL buffers are synched before giving access for read
                        if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                            // transfer only valid data
                            vx_size size = data->size;
                            if (size > 0 && data->hip_memory) {
                                hipError_t err = hipMemcpyDtoH((void *)data->buffer, (data->hip_memory + data->gpu_buffer_offset), size);
                                if (err) {
                                    status = VX_FAILURE;
                                    agoAddLogEntry(&data->ref, status, "ERROR: vxMapLUT: hipMemcpyDtoH() => %d\n", err);
                                    return status;
                                }
                            }
                            data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                        }
                    }
#endif
                    if (item.used_external_ptr) {
                        // copy if read is requested with explicit external buffer
                        HafCpu_BinaryCopy_U8_U8(data->size, ptr_returned, ptr_internal);
                    }
                }
            }
        }
    }
    return status;
}

/*! \brief Commits the Lookup Table.
* \details Commits the data back to the LUT object and decrements the reference count.
* There are several variations of call methodology:
* \arg If a user should allocated their own memory for the LUT data copy, the user is
* obligated to free this memory.
* \arg If \a ptr is not NULL and the (*ptr) for <tt>\ref vxAccessLUT</tt> was NULL,
* it is undefined whether the implementation will unmap or copy and free the memory.
* \param [in] lut The LUT to modify.
* \param [in] ptr The pointer used with <tt>\ref vxAccessLUT</tt>. This cannot be NULL.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \pre <tt>\ref vxAccessLUT</tt>.
* \ingroup group_lut
*/
VX_API_ENTRY vx_status VX_API_CALL vxCommitLUT(vx_lut lut, const void *ptr)
{
    AgoData * data = (AgoData *)lut;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_LUT)) {
        // check for valid arguments
        status = VX_ERROR_INVALID_PARAMETERS;
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if (ptr) {
            status = VX_SUCCESS;
            if (!data->buffer) {
                status = VX_FAILURE;
            }
            else if (!data->mapped.empty()) {
                vx_enum usage = VX_READ_ONLY;
                bool used_external_ptr = false;
                for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
                    if (i->ptr == ptr) {
                        usage = i->usage;
                        used_external_ptr = i->used_external_ptr;
                        data->mapped.erase(i);
                        break;
                    }
                }
                if (usage == VX_WRITE_ONLY || usage == VX_READ_AND_WRITE) {
                    if (used_external_ptr) {
                        // copy from external buffer
                        HafCpu_BinaryCopy_U8_U8(data->size, data->buffer, (vx_uint8 *)ptr);
                    }
                    // update sync flags
                    data->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                    data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
                }
            }
        }
    }
    return status;
}

/*! \brief Allows the application to copy from/into a LUT object.
* \param [in] lut The reference to the LUT object that is the source or the
* destination of the copy.
* \param [in] user_ptr The address of the memory location where to store the requested data
* if the copy was requested in read mode, or from where to get the data to store into the LUT
* object if the copy was requested in write mode. In the user memory, the LUT is
* represented as a array with elements of the type corresponding to
* <tt>\ref VX_LUT_TYPE</tt>, and with a number of elements equal to the value
* returned via tt>\ref VX_LUT_COUNT</tt>. The accessible memory must be large enough
* to contain this array:
* accessible memory in bytes >= sizeof(data_element) * count.
* \param [in] usage This declares the effect of the copy with regard to the LUT object
* using the <tt>\ref vx_accessor_e</tt> enumeration. Only VX_READ_ONLY and VX_WRITE_ONLY
* are supported:
* \arg VX_READ_ONLY means that data are copied from the LUT object into the user memory.
* \arg VX_WRITE_ONLY means that data are copied into the LUT object from the user memory.
* \param [in] user_mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that specifies
* the memory type of the memory referenced by the user_addr.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_INVALID_REFERENCE The LUT reference is not actually a LUT reference.
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \ingroup group_lut
*/
VX_API_ENTRY vx_status VX_API_CALL vxCopyLUT(vx_lut lut, void *user_ptr, vx_enum usage, vx_enum user_mem_type)
{
    AgoData * data = (AgoData *)lut;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_LUT))
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        if ((user_mem_type == VX_MEMORY_TYPE_HOST) && user_ptr && (usage == VX_READ_ONLY || usage == VX_WRITE_ONLY)) {
            status = vxAccessLUT(lut, &user_ptr, usage);
            if (status == VX_SUCCESS) {
                status = vxCommitLUT(lut, user_ptr);
            }
        }
    }
    return status;
}

/*! \brief Allows the application to get direct access to LUT object.
* \param [in] lut The reference to the LUT object to map.
* \param [out] map_id The address of a vx_map_id variable where the function
* returns a map identifier.
* \arg (*map_id) must eventually be provided as the map_id parameter of a call to
* <tt>\ref vxUnmapLUT</tt>.
* \param [out] ptr The address of a pointer that the function sets to the
* address where the requested data can be accessed. In the mapped memory area,
* the LUT data are structured as an array with elements of the type corresponding
* to <tt>\ref VX_LUT_TYPE</tt>, with a number of elements equal to
* the value returned via tt>\ref VX_LUT_COUNT</tt>. Accessing the
* memory out of the bound of this array is forbidden and has an undefined behavior.
* The returned (*ptr) address is only valid between the call to the function and
* the corresponding call to <tt>\ref vxUnmapLUT</tt>.
* \param [in] usage This declares the access mode for the LUT, using
* the <tt>\ref vx_accessor_e</tt> enumeration.
* \arg VX_READ_ONLY: after the function call, the content of the memory location
* pointed by (*ptr) contains the LUT data. Writing into this memory location
* is forbidden and its behavior is undefined.
* \arg VX_READ_AND_WRITE : after the function call, the content of the memory
* location pointed by (*ptr) contains the LUT data; writing into this memory
* is allowed only for the location of entries and will result in a modification
* of the affected entries in the LUT object once the LUT is unmapped.
* \arg VX_WRITE_ONLY: after the function call, the memory location pointed by(*ptr)
* contains undefined data; writing each entry of LUT is required prior to
* unmapping. Entries not written by the application before unmap will become
* undefined after unmap, even if they were well defined before map.
* \param [in] mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that
* specifies the type of the memory where the LUT is requested to be mapped.
* \param [in] flags An integer that allows passing options to the map operation.
* Use 0 for this option.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_INVALID_REFERENCE The LUT reference is not actually a LUT
* reference.
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \ingroup group_lut
* \post <tt>\ref vxUnmapLUTRange </tt> with same (*map_id) value.
*/
VX_API_ENTRY vx_status VX_API_CALL vxMapLUT(vx_lut lut, vx_map_id *map_id, void **ptr, vx_enum usage, vx_enum mem_type, vx_bitfield flags)
{
    AgoData * data = (AgoData *)lut;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_LUT)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if (ptr) {
            if (!data->buffer) {
                CAgoLock lock(data->ref.context->cs);
                if (agoAllocData(data)) {
                    return VX_FAILURE;
                }
            }
            vx_uint8 * ptr_returned = data->buffer;
            status = VX_SUCCESS;
            for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
                if (i->ptr == ptr_returned) {
                    // can't support vxAccessXXX() more than once with same pointer, the application
                    // needs to call vxCommitXXX() before calling vxAccessXXX()
                    status = VX_FAILURE;
                }
            }
            if (status == VX_SUCCESS) {
#if ENABLE_OPENCL
                if (usage == VX_READ_ONLY || usage == VX_READ_AND_WRITE) {
                    if (data->opencl_buffer && !(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                        // make sure dirty OpenCL buffers are synched before giving access for read
                        if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                            size_t origin[3] = { 0, 0, 0 };
                            size_t region[3] = { 256, 1, 1 };
                            cl_int err = clEnqueueReadImage(data->ref.context->opencl_cmdq, data->opencl_buffer, CL_TRUE, origin, region, 256, 0, data->buffer, 0, NULL, NULL);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&data->ref, status, "ERROR: vxMapLUT: clEnqueueWriteImage() => %d\n", err);
                                return status;
                            }
                            data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                        }
                    }
                }
#elif ENABLE_HIP
                if (usage == VX_READ_ONLY || usage == VX_READ_AND_WRITE) {
                    if (data->hip_memory && !(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                        // make sure dirty OpenCL buffers are synched before giving access for read
                        if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                        // transfer only valid data
                            vx_size size = data->size;
                            if (size > 0 && data->hip_memory) {
                                hipError_t err = hipMemcpyDtoH((void *)data->buffer, (data->hip_memory + data->gpu_buffer_offset), size);
                                if (err) {
                                    status = VX_FAILURE;
                                    agoAddLogEntry(&data->ref, status, "ERROR: vxMapLUT: hipMemcpyDtoH() => %d\n", err);
                                    return status;
                                }
                            }
                            data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                        }
                    }
                }
#endif
                MappedData item = { data->nextMapId++, ptr_returned, usage, false };
                data->mapped.push_back(item);
                *map_id = item.map_id;
                *ptr = ptr_returned;
            }
        }
    }
    return status;
}

/*! \brief Unmap and commit potential changes to LUT object that was previously mapped.
* Unmapping a LUT invalidates the memory location from which the LUT data could
* be accessed by the application. Accessing this memory location after the unmap function
* completes has an undefined behavior.
* \param [in] lut The reference to the LUT object to unmap.
* \param [out] map_id The unique map identifier that was returned when calling
* <tt>\ref vxMapLUT</tt> .
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_INVALID_REFERENCE The LUT reference is not actually a LUT reference.
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \ingroup group_lut
* \pre <tt>\ref vxMapLUT</tt> returning the same map_id value
*/
VX_API_ENTRY vx_status VX_API_CALL vxUnmapLUT(vx_lut lut, vx_map_id map_id)
{
    AgoData * data = (AgoData *)lut;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_LUT)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
            if (i->map_id == map_id) {
                vx_enum usage = i->usage;
                data->mapped.erase(i);
                if (usage == VX_WRITE_ONLY || usage == VX_READ_AND_WRITE) {
                    // update sync flags
                    data->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                    data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
                }
                status = VX_SUCCESS;
                break;
            }
        }
    }
    return status;
}

/*==============================================================================
DISTRIBUTION
=============================================================================*/

/*! \brief Creates a reference to a 1D Distribution with a start offset, valid range, and number of equally weighted bins.
* \param [in] context The reference to the overall context.
* \param [in] numBins The number of bins in the distribution.
* \param [in] offset The offset into the range value.
* \param [in] range The total range of the values.
* \return <tt>\ref vx_distribution</tt>
* \ingroup group_distribution
*/
VX_API_ENTRY vx_distribution VX_API_CALL vxCreateDistribution(vx_context context, vx_size numBins, vx_int32 offset, vx_uint32 range)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context) && numBins > 0 && range > 0) {
        CAgoLock lock(context->cs);
        char desc[512]; snprintf(desc, sizeof(desc), "distribution:" VX_FMT_SIZE ",%d,%u", numBins, offset, range);
        data = agoCreateDataFromDescription(context, NULL, desc, true);
        if (data) {
            agoGenerateDataName(context, "dist", data->name);
            agoAddData(&context->dataList, data);
        }
    }
    return (vx_distribution)data;
}

/*! \brief Creates an opaque reference to a 1D Distribution object without direct user access.
* \param [in] graph The reference to the parent graph.
* \param [in] numBins The number of bins in the distribution.
* \param [in] offset The start offset into the range value that marks the begining of the 1D Distribution.
* \param [in] range The total number of the consecutive values of the distribution interval.
* \return <tt>\ref vx_distribution</tt>
* \ingroup group_distribution
*/
VX_API_ENTRY vx_distribution VX_API_CALL vxCreateVirtualDistribution(vx_graph graph, vx_size numBins, vx_int32 offset, vx_uint32 range)
{
    AgoData * data = NULL;
    if (agoIsValidGraph(graph) && numBins > 0 && range > 0) {
        CAgoLock lock(graph->cs);
        char desc[512]; snprintf(desc, sizeof(desc), "distribution-virtual:" VX_FMT_SIZE ",%d,%u", numBins, offset, range);
        data = agoCreateDataFromDescription(graph->ref.context, graph, desc, true);
        if (data) {
            agoGenerateVirtualDataName(graph, "dist", data->name);
            agoAddData(&graph->dataList, data);
        }
    }
    return (vx_distribution)data;
}

/*! \brief Releases a reference to a distribution object.
* The object may not be garbage collected until its total reference count is zero.
* \param [in] distribution The reference to the distribution to release.
* \post After returning from this function the reference is zeroed.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
* \ingroup group_distribution
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseDistribution(vx_distribution *distribution)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (distribution && agoIsValidData((AgoData*)*distribution, VX_TYPE_DISTRIBUTION)) {
        if (!agoReleaseData((AgoData*)*distribution, true)) {
            *distribution = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Queries a Distribution object.
* \param [in] distribution The reference to the distribution to query.
* \param [in] attribute The attribute to query. Use a <tt>\ref vx_distribution_attribute_e</tt> enumeration.
* \param [out] ptr The location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_distribution
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryDistribution(vx_distribution distribution, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)distribution;
    if (agoIsValidData(data, VX_TYPE_DISTRIBUTION)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_DISTRIBUTION_ATTRIBUTE_DIMENSIONS:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = 1;
                    status = VX_SUCCESS;
                }
                break;
            case VX_DISTRIBUTION_ATTRIBUTE_OFFSET:
                if (size == sizeof(vx_int32)) {
                    *(vx_int32 *)ptr = data->u.dist.offset;
                    status = VX_SUCCESS;
                }
                break;
            case VX_DISTRIBUTION_ATTRIBUTE_RANGE:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = data->u.dist.range;
                    status = VX_SUCCESS;
                }
                break;
            case VX_DISTRIBUTION_ATTRIBUTE_BINS:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->u.dist.numbins;
                    status = VX_SUCCESS;
                }
                break;
            case VX_DISTRIBUTION_ATTRIBUTE_WINDOW:
                if (size == sizeof(vx_uint32)) {
                    vx_uint32 window = (data->u.dist.window * data->u.dist.numbins == data->u.dist.range) ? data->u.dist.window : 0;
                    *(vx_uint32 *)ptr = window;
                    status = VX_SUCCESS;
                }
                break;
            case VX_DISTRIBUTION_ATTRIBUTE_SIZE:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->size;
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Gets direct access to a Distribution in memory.
* \param [in] distribution The reference to the distribution to access.
* \param [out] ptr The address of the location to store the pointer to the
* Distribution memory.
* \arg If (*ptr) is not NULL, the Distribution will be copied to that address.
* \arg If (*ptr) is NULL, the pointer will be allocated, mapped, or use internal memory.
*
* In any case, <tt>\ref vxCommitDistribution</tt> must be called with (*ptr).
* \param [in] usage The <tt>\ref vx_accessor_e</tt> value to describe the access of the object.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \post <tt>\ref vxCommitDistribution</tt>
* \ingroup group_distribution
*/
VX_API_ENTRY vx_status VX_API_CALL vxAccessDistribution(vx_distribution distribution, void **ptr, vx_enum usage)
{
    AgoData * data = (AgoData *)distribution;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_DISTRIBUTION)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if (ptr) {
            if (!data->buffer) {
                CAgoLock lock(data->ref.context->cs);
                if (agoAllocData(data)) {
                    return VX_FAILURE;
                }
            }
            vx_uint8 * ptr_internal = data->buffer;
            vx_uint8 * ptr_returned = *ptr ? (vx_uint8 *)*ptr : ptr_internal;
            // save the pointer and usage for use in vxCommitXXX
            status = VX_SUCCESS;
            for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
                if (i->ptr == ptr_returned) {
                    // can't support vxAccessXXX() more than once with same pointer, the application
                    // needs to call vxCommitXXX() before calling vxAccessXXX()
                    status = VX_FAILURE;
                }
            }
            if (status == VX_SUCCESS) {
                MappedData item = { data->nextMapId++, ptr_returned, usage, (ptr_returned != ptr_internal) ? true : false };
                data->mapped.push_back(item);
                *ptr = ptr_returned;
                if (item.used_external_ptr && (usage == VX_READ_ONLY || usage == VX_READ_AND_WRITE)) {
                    // copy if read is requested with explicit external buffer
                    HafCpu_BinaryCopy_U8_U8(data->size, ptr_returned, ptr_internal);
                }
            }
        }
    }
    return status;
}

/*! \brief Sets the Distribution back to the memory. The memory must be
* a vx_uint32 array of a value at least as big as the value returned via <tt>\ref VX_DISTRIBUTION_ATTRIBUTE_RANGE</tt>.
* \param [in] distribution The Distribution to modify.
* \param [in] ptr The pointer returned from (or not modified by) <tt>\ref vxAccessDistribution</tt>.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \pre <tt>\ref vxAccessDistribution</tt>.
* \ingroup group_distribution
*/
VX_API_ENTRY vx_status VX_API_CALL vxCommitDistribution(vx_distribution distribution, const void * ptr)
{
    AgoData * data = (AgoData *)distribution;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_DISTRIBUTION)) {
        // check for valid arguments
        status = VX_ERROR_INVALID_PARAMETERS;
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if (ptr) {
            status = VX_SUCCESS;
            if (!data->buffer) {
                status = VX_FAILURE;
            }
            else if (!data->mapped.empty()) {
                vx_enum usage = VX_READ_ONLY;
                bool used_external_ptr = false;
                for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
                    if (i->ptr == ptr) {
                        usage = i->usage;
                        used_external_ptr = i->used_external_ptr;
                        data->mapped.erase(i);
                        break;
                    }
                }
                if (usage == VX_WRITE_ONLY || usage == VX_READ_AND_WRITE) {
                    if (used_external_ptr) {
                        // copy from external buffer
                        HafCpu_BinaryCopy_U8_U8(data->size, data->buffer, (vx_uint8 *)ptr);
                    }
                }
            }
        }
    }
    return status;
}

/*! \brief Allows the application to copy from/into a distribution object.
* \param [in] distribution The reference to the distribution object that is the source or the
* destination of the copy.
* \param [in] user_ptr The address of the memory location where to store the requested data
* if the copy was requested in read mode, or from where to get the data to store into the distribution
* object if the copy was requested in write mode. In the user memory, the distribution is
* represented as a <tt>\ref vx_uint32</tt> array with a number of elements equal to the value returned via
* <tt>\ref VX_DISTRIBUTION_BINS</tt>. The accessible memory must be large enough
* to contain this vx_uint32 array:
* accessible memory in bytes >= sizeof(vx_uint32) * num_bins.
* \param [in] usage This declares the effect of the copy with regard to the distribution object
* using the <tt>\ref vx_accessor_e</tt> enumeration. Only VX_READ_ONLY and VX_WRITE_ONLY
* are supported:
* \arg VX_READ_ONLY means that data are copied from the distribution object into the user memory.
* \arg VX_WRITE_ONLY means that data are copied into the distribution object from the user memory.
* \param [in] user_mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that specifies
* the memory type of the memory referenced by the user_addr.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_INVALID_REFERENCE The distribution reference is not actually a distribution reference.
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \ingroup group_distribution
*/
VX_API_ENTRY vx_status VX_API_CALL vxCopyDistribution(vx_distribution distribution, void *user_ptr, vx_enum usage, vx_enum user_mem_type)
{
    AgoData * data = (AgoData *)distribution;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_DISTRIBUTION))
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        if ((user_mem_type == VX_MEMORY_TYPE_HOST) && user_ptr && (usage == VX_READ_ONLY || usage == VX_WRITE_ONLY)) {
            status = vxAccessDistribution(distribution, &user_ptr, usage);
            if (status == VX_SUCCESS) {
                status = vxCommitDistribution(distribution, user_ptr);
            }
        }
    }
    return status;
}

/*! \brief Allows the application to get direct access to distribution object.
* \param [in] distribution The reference to the distribution object to map.
* \param [out] map_id The address of a vx_map_id variable where the function
* returns a map identifier.
* \arg (*map_id) must eventually be provided as the map_id parameter of a call to
* <tt>\ref vxUnmapDistribution</tt>.
* \param [out] ptr The address of a pointer that the function sets to the
* address where the requested data can be accessed. In the mapped memory area,
* data are structured as a vx_uint32 array with a number of elements equal to
* the value returned via <tt>\ref VX_DISTRIBUTION_BINS</tt>. Each
* element of this array corresponds to a bin of the distribution, with a range-major
* ordering. Accessing the memory out of the bound of this array
* is forbidden and has an undefined behavior. The returned (*ptr) address
* is only valid between the call to the function and the corresponding call to
* <tt>\ref vxUnmapDistribution</tt>.
* \param [in] usage This declares the access mode for the distribution, using
* the <tt>\ref vx_accessor_e</tt> enumeration.
* \arg VX_READ_ONLY: after the function call, the content of the memory location
* pointed by (*ptr) contains the distribution data. Writing into this memory location
* is forbidden and its behavior is undefined.
* \arg VX_READ_AND_WRITE : after the function call, the content of the memory
* location pointed by (*ptr) contains the distribution data; writing into this memory
* is allowed only for the location of bins and will result in a modification of the
* affected bins in the distribution object once the distribution is unmapped.
* \arg VX_WRITE_ONLY: after the function call, the memory location pointed by (*ptr)
* contains undefined data; writing each bin of distribution is required prior to
* unmapping. Bins not written by the application before unmap will become
* undefined after unmap, even if they were well defined before map.
* \param [in] mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that
* specifies the type of the memory where the distribution is requested to be mapped.
* \param [in] flags An integer that allows passing options to the map operation.
* Use 0 for this option.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_INVALID_REFERENCE The distribution reference is not actually a distribution
* reference.
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \ingroup group_distribution
* \post <tt>\ref vxUnmapDistributionRange </tt> with same (*map_id) value.
*/
VX_API_ENTRY vx_status VX_API_CALL vxMapDistribution(vx_distribution distribution, vx_map_id *map_id, void **ptr, vx_enum usage, vx_enum mem_type, vx_bitfield flags)
{
    AgoData * data = (AgoData *)distribution;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_DISTRIBUTION)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if (ptr) {
            if (!data->buffer) {
                CAgoLock lock(data->ref.context->cs);
                if (agoAllocData(data)) {
                    return VX_FAILURE;
                }
            }
            vx_uint8 * ptr_returned = data->buffer;
            // save the pointer and usage for use in vxCommitXXX
            status = VX_SUCCESS;
            for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
                if (i->ptr == ptr_returned) {
                    // can't support vxAccessXXX() more than once with same pointer, the application
                    // needs to call vxCommitXXX() before calling vxAccessXXX()
                    status = VX_FAILURE;
                }
            }
            if (status == VX_SUCCESS) {
                MappedData item = { data->nextMapId++, ptr_returned, usage, false };
                data->mapped.push_back(item);
                *map_id = item.map_id;
                *ptr = ptr_returned;
            }
        }
    }
    return status;
}

/*! \brief Unmap and commit potential changes to distribution object that was previously mapped.
* Unmapping a distribution invalidates the memory location from which the distribution data
* could be accessed by the application. Accessing this memory location after the unmap
* function completes has an undefined behavior.
* \param [in] distribution The reference to the distribution object to unmap.
* \param [out] map_id The unique map identifier that was returned when calling
* <tt>\ref vxMapDistribution</tt> .
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_INVALID_REFERENCE The distribution reference is not actually a distribution reference.
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \ingroup group_distribution
* \pre <tt>\ref vxMapDistribution</tt> returning the same map_id value
*/
VX_API_ENTRY vx_status VX_API_CALL vxUnmapDistribution(vx_distribution distribution, vx_map_id map_id)
{
    AgoData * data = (AgoData *)distribution;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_DISTRIBUTION)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
            if (i->map_id == map_id) {
                vx_enum usage = i->usage;
                data->mapped.erase(i);
                if (usage == VX_WRITE_ONLY || usage == VX_READ_AND_WRITE) {
                    // update sync flags
                    data->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                    data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
                }
                status = VX_SUCCESS;
                break;
            }
        }
    }
    return status;
}

/*==============================================================================
THRESHOLD
=============================================================================*/

/*! \brief Creates a reference to a threshold object of a given type.
* \param [in] c The reference to the overall context.
* \param [in] thresh_type The type of threshold to create.
* \param [in] data_type The data type of the threshold's value(s).
* \if OPENVX_STRICT_1_0
* \note For OpenVX 1.0, data_type can only be <tt>\ref VX_TYPE_UINT8</tt>.
* \endif
* \return <tt>\ref vx_threshold</tt>
* \ingroup group_threshold
*/
VX_API_ENTRY vx_threshold VX_API_CALL vxCreateThreshold(vx_context context, vx_enum thresh_type, vx_enum data_type)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context) && (thresh_type == VX_THRESHOLD_TYPE_BINARY || thresh_type == VX_THRESHOLD_TYPE_RANGE) &&
        (((data_type >= VX_TYPE_INT8) && (data_type <= VX_TYPE_INT32)) || data_type == VX_TYPE_BOOL))
    {
        CAgoLock lock(context->cs);
        char desc[512]; snprintf(desc, sizeof(desc), "threshold:%s,%s", agoEnum2Name(thresh_type), agoEnum2Name(data_type));
        data = agoCreateDataFromDescription(context, NULL, desc, true);

        if (data) {
            agoGenerateDataName(context, "thr", data->name);
            agoAddData(&context->dataList, data);
            switch(data_type)
            {
            case VX_TYPE_BOOL:
                data->u.thr.true_value.U1 = AGO_U1_THRESHOLD_TRUE_VALUE;
                data->u.thr.false_value.U1 = AGO_U1_THRESHOLD_FALSE_VALUE;
                break;
            case VX_TYPE_INT8:
                data->u.thr.true_value.U8 = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                data->u.thr.false_value.U8 = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                break;
            case VX_TYPE_UINT8:
                data->u.thr.true_value.U8 = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                data->u.thr.false_value.U8 = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                break;
            case VX_TYPE_UINT16:
                data->u.thr.true_value.U16 = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                data->u.thr.false_value.U16 = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                break;
            case VX_TYPE_INT16:
                data->u.thr.true_value.S16 = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                data->u.thr.false_value.S16 = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                break;
            case VX_TYPE_INT32:
                data->u.thr.true_value.S32 = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                data->u.thr.false_value.S32 = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                break;
            case VX_TYPE_UINT32:
                data->u.thr.true_value.U32 = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                data->u.thr.false_value.U32 = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                break;
            default:
                break;
            }
        }
    }
    return (vx_threshold)data;
}

/*! \brief Creates a threshold object and returns a reference to it.
* \param [in] context       The reference to the context in which the object is created.
* \param [in] thresh_type   The type of thresholding operation.
* \param [in] input_format  The format of images that will be used as input of the thresholding operation.
*                           The input image formats <tt>\ref VX_DF_IMAGE_U8</tt> and <tt>\ref VX_DF_IMAGE_S16</tt> shall be supported.
* \param [in] output_format The format of images that will be generated by the thresholding operation.
*                           The output image format <tt>\ref VX_DF_IMAGE_U8</tt> shall be supported.
*							If other output formats are supported, the true/false values listed above must be used.
* \return                   A threshold reference <tt>\ref vx_threshold</tt>.
*
* \note The threshold object defines the parameters of a thresholding operation to an input image, that generates an output image that can have a different format.
* The thresholding 'false' or 'true' output values are specified per pixel channels of the output format and can be modified with <tt>\ref vxCopyThresholdOutput<tt>.
* The default 'false' output value of pixels channels should be 0, and the default 'true' value should be non-zero.
*
* \ingroup group_threshold
*/
VX_API_ENTRY vx_threshold VX_API_CALL vxCreateThresholdForImage(vx_context context, vx_enum thresh_type, vx_df_image input_format, vx_df_image output_format)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context) && (thresh_type == VX_THRESHOLD_TYPE_BINARY || thresh_type == VX_THRESHOLD_TYPE_RANGE))
    {
        /*if((input_format == VX_DF_IMAGE_RGB  || input_format == VX_DF_IMAGE_RGBX || input_format == VX_DF_IMAGE_NV12 ||
            input_format == VX_DF_IMAGE_NV21 || input_format == VX_DF_IMAGE_UYVY || input_format == VX_DF_IMAGE_YUYV ||
            input_format == VX_DF_IMAGE_IYUV || input_format == VX_DF_IMAGE_YUV4) && (output_format == VX_DF_IMAGE_RGB  ||
            output_format == VX_DF_IMAGE_RGBX || output_format == VX_DF_IMAGE_NV12 || output_format == VX_DF_IMAGE_NV21 ||
            output_format == VX_DF_IMAGE_UYVY || output_format == VX_DF_IMAGE_YUYV || output_format == VX_DF_IMAGE_IYUV ||
            output_format == VX_DF_IMAGE_YUV4)) {*/

            CAgoLock lock(context->cs);

            /*vx_enum data_type;
            if(output_format == VX_DF_IMAGE_U8)
                data_type = VX_TYPE_UINT8;
            else if(output_format == VX_DF_IMAGE_U1)
                data_type = VX_TYPE_BOOL;
            */
            const char * desc_type = agoEnum2Name(thresh_type);
            if(desc_type){
                char desc[512]; snprintf(desc, sizeof(desc), "threshold:%s,%u,%u", desc_type, input_format, output_format);
                data = agoCreateDataFromDescription(context, NULL, desc, true);
                if (data) {
                    agoGenerateDataName(context, "thr", data->name);
                    agoAddData(&context->dataList, data);

                    switch (input_format)
                    {
                        case VX_DF_IMAGE_RGB:
                        {
                            data->u.thr.data_type = VX_TYPE_DF_IMAGE;
                            data->u.thr.true_value.RGB[0] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                            data->u.thr.true_value.RGB[1] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                            data->u.thr.true_value.RGB[2] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                            data->u.thr.false_value.RGB[0] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                            data->u.thr.false_value.RGB[1] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                            data->u.thr.false_value.RGB[2] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                            break;
                        }
                        case VX_DF_IMAGE_RGBX:
                        {
                            data->u.thr.data_type = VX_TYPE_DF_IMAGE;
                            data->u.thr.true_value.RGBX[0] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                            data->u.thr.true_value.RGBX[1] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                            data->u.thr.true_value.RGBX[2] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                            data->u.thr.true_value.RGBX[3] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                            data->u.thr.false_value.RGBX[0] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                            data->u.thr.false_value.RGBX[1] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                            data->u.thr.false_value.RGBX[2] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                            data->u.thr.false_value.RGBX[3] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                            break;
                        }
                        case VX_DF_IMAGE_NV12:
                        case VX_DF_IMAGE_NV21:
                        case VX_DF_IMAGE_UYVY:
                        case VX_DF_IMAGE_YUYV:
                        case VX_DF_IMAGE_IYUV:
                        case VX_DF_IMAGE_YUV4:
                        {
                            data->u.thr.data_type = VX_TYPE_DF_IMAGE;
                            data->u.thr.true_value.YUV[0] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                            data->u.thr.true_value.YUV[1] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                            data->u.thr.true_value.YUV[2] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                            data->u.thr.false_value.YUV[0] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                            data->u.thr.false_value.YUV[1] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                            data->u.thr.false_value.YUV[2] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                            break;
                        }
                        case VX_DF_IMAGE_U1:
                        {
                            data->u.thr.data_type = VX_TYPE_BOOL;
                            data->u.thr.true_value.U1  = AGO_U1_THRESHOLD_TRUE_VALUE;
                            data->u.thr.false_value.U1 = AGO_U1_THRESHOLD_FALSE_VALUE;
                            break;
                        }
                        case VX_DF_IMAGE_U8:
                        {
                            data->u.thr.data_type = VX_TYPE_UINT8;
                            data->u.thr.true_value.U8  = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                            data->u.thr.false_value.U8 = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                            break;
                        }
                        case VX_DF_IMAGE_S16:
                        {
                            data->u.thr.data_type = VX_TYPE_INT16;
                            data->u.thr.true_value.S16  = AGO_S16_THRESHOLD_TRUE_VALUE;
                            data->u.thr.false_value.S16 = AGO_S16_THRESHOLD_FALSE_VALUE;
                            break;
                        }
                        case VX_DF_IMAGE_U16:
                        {
                            data->u.thr.data_type = VX_TYPE_UINT16;
                            data->u.thr.true_value.U16  = AGO_U16_THRESHOLD_TRUE_VALUE;
                            data->u.thr.false_value.U16 = AGO_U16_THRESHOLD_FALSE_VALUE;
                            break;
                        }
                        case VX_DF_IMAGE_S32:
                        {
                            data->u.thr.data_type = VX_TYPE_INT32;
                            data->u.thr.true_value.S32  = AGO_S32_THRESHOLD_TRUE_VALUE;
                            data->u.thr.false_value.S32 = AGO_S32_THRESHOLD_FALSE_VALUE;
                            break;
                        }
                        case VX_DF_IMAGE_U32:
                        {
                            data->u.thr.data_type = VX_TYPE_UINT32;
                            data->u.thr.true_value.U32  = AGO_U32_THRESHOLD_TRUE_VALUE;
                            data->u.thr.false_value.U32 = AGO_U32_THRESHOLD_FALSE_VALUE;
                            break;
                        }
                        default:
                        {
                            break;
                        }
                    }
                //}
            }
        }
    }
    return (vx_threshold)data;
}

/*! \brief Creates an opaque reference to a threshold object without direct user access.
* \param [in] graph         The reference to the parent graph.
* \param [in] thresh_type   The type of thresholding operation.
* \param [in] input_format  The format of images that will be used as input of the thresholding operation.
* \param [in] output_format The format of images that will be generated by the thresholding operation.
*
* \return                   A threshold reference <tt>\ref vx_threshold</tt>.
*
* \ingroup group_threshold
*/
VX_API_ENTRY vx_threshold VX_API_CALL vxCreateVirtualThresholdForImage(vx_graph graph, vx_enum thresh_type, vx_df_image input_format, vx_df_image output_format)
{
    AgoData * data = NULL;
    if (agoIsValidGraph(graph) && (thresh_type == VX_THRESHOLD_TYPE_BINARY || thresh_type == VX_THRESHOLD_TYPE_RANGE) &&
        (input_format == VX_DF_IMAGE_U8 || input_format == VX_DF_IMAGE_S16) && (output_format == VX_DF_IMAGE_U8 || output_format == VX_DF_IMAGE_U1)) {
        CAgoLock lock(graph->cs);

        const char * desc_type = agoEnum2Name(thresh_type);

        /*vx_enum data_type;
        if(output_format == VX_DF_IMAGE_U8)
            data_type = VX_TYPE_UINT8;
        else if(output_format == VX_DF_IMAGE_U1)
            data_type = VX_TYPE_BOOL;
        */
        if (desc_type) {
            char desc[512];
            snprintf(desc, sizeof(desc), "threshold-virtual:%s,%u,%u", desc_type, input_format, output_format);
            data = agoCreateDataFromDescription(graph->ref.context, graph, desc, true);
            if (data) {
                agoGenerateVirtualDataName(graph, "thr", data->name);
                agoAddData(&graph->dataList, data);
                switch (output_format)
                {
                    case VX_DF_IMAGE_RGB:
                    {
                        data->u.thr.data_type = VX_TYPE_DF_IMAGE;
                        data->u.thr.true_value.RGB[0] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                        data->u.thr.true_value.RGB[1] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                        data->u.thr.true_value.RGB[2] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                        data->u.thr.false_value.RGB[0] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                        data->u.thr.false_value.RGB[1] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                        data->u.thr.false_value.RGB[2] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                        break;
                    }
                    case VX_DF_IMAGE_RGBX:
                    {
                        data->u.thr.data_type = VX_TYPE_DF_IMAGE;
                        data->u.thr.true_value.RGBX[0] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                        data->u.thr.true_value.RGBX[1] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                        data->u.thr.true_value.RGBX[2] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                        data->u.thr.true_value.RGBX[3] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                        data->u.thr.false_value.RGBX[0] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                        data->u.thr.false_value.RGBX[1] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                        data->u.thr.false_value.RGBX[2] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                        data->u.thr.false_value.RGBX[3] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                        break;
                    }
                    case VX_DF_IMAGE_NV12:
                    case VX_DF_IMAGE_NV21:
                    case VX_DF_IMAGE_UYVY:
                    case VX_DF_IMAGE_YUYV:
                    case VX_DF_IMAGE_IYUV:
                    case VX_DF_IMAGE_YUV4:
                    {
                        data->u.thr.data_type = VX_TYPE_DF_IMAGE;
                        data->u.thr.true_value.YUV[0] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                        data->u.thr.true_value.YUV[1] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                        data->u.thr.true_value.YUV[2] = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                        data->u.thr.false_value.YUV[0] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                        data->u.thr.false_value.YUV[1] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                        data->u.thr.false_value.YUV[2] = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                        break;
                    }
                    case VX_DF_IMAGE_U1:
                    {
                        data->u.thr.data_type = VX_TYPE_BOOL;
                        data->u.thr.true_value.U1  = AGO_U1_THRESHOLD_TRUE_VALUE;
                        data->u.thr.false_value.U1 = AGO_U1_THRESHOLD_FALSE_VALUE;
                        break;
                    }
                    case VX_DF_IMAGE_U8:
                    {
                        data->u.thr.data_type = VX_TYPE_UINT8;
                        data->u.thr.true_value.U8  = AGO_DEFAULT_THRESHOLD_TRUE_VALUE;
                        data->u.thr.false_value.U8 = AGO_DEFAULT_THRESHOLD_FALSE_VALUE;
                        break;
                    }
                    case VX_DF_IMAGE_S16:
                    {
                        data->u.thr.data_type = VX_TYPE_INT16;
                        data->u.thr.true_value.S16  = AGO_S16_THRESHOLD_TRUE_VALUE;
                        data->u.thr.false_value.S16 = AGO_S16_THRESHOLD_FALSE_VALUE;
                        break;
                    }
                    case VX_DF_IMAGE_U16:
                    {
                        data->u.thr.data_type = VX_TYPE_UINT16;
                        data->u.thr.true_value.U16  = AGO_U16_THRESHOLD_TRUE_VALUE;
                        data->u.thr.false_value.U16 = AGO_U16_THRESHOLD_FALSE_VALUE;
                        break;
                    }
                    case VX_DF_IMAGE_S32:
                    {
                        data->u.thr.data_type = VX_TYPE_INT32;
                        data->u.thr.true_value.S32  = AGO_S32_THRESHOLD_TRUE_VALUE;
                        data->u.thr.false_value.S32 = AGO_S32_THRESHOLD_FALSE_VALUE;
                        break;
                    }
                    case VX_DF_IMAGE_U32:
                    {
                        data->u.thr.data_type = VX_TYPE_UINT32;
                        data->u.thr.true_value.U32  = AGO_U32_THRESHOLD_TRUE_VALUE;
                        data->u.thr.false_value.U32 = AGO_U32_THRESHOLD_FALSE_VALUE;
                        break;
                    }
                    default:
                    {
                        break;
                    }
                }
            }
        }
    }
    return (vx_threshold)data;
}

/*! \brief Allows the application to copy the thresholding value from/into a threshold object with type <tt>\ref VX_THRESHOLD_TYPE_BINARY<\tt>.
* \param [in] thresh    	The reference to the threshold object that is the source or the destination of the copy.
* \param [in,out] value_ptr 	The address of the memory location where to store the thresholding value if the copy was requested in read mode,
*							or from where to get the thresholding value to store into the threshold object if the copy was requested in write mode.
* \param [in] usage     	This declares the effect of the copy with regard to the threshold object using the <tt>\ref vx_accessor_e</tt> enumeration.
*							Only VX_READ_ONLY and VX_WRITE_ONLY are supported:
* \arg VX_READ_ONLY 		Means that the thresholding value is copied from the threshold object into the user memory.
*							After the copy, only the field of the (*value_ptr) union that corresponds to the input image format of the threshold object is meaningful.
* \arg VX_WRITE_ONLY 		Means the field of the (value_ptr) union corresponding to the input format of the threshold object is copied into the threshold object.
* \param [in] user_mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that specifies the type of the memory referenced by value_ptr.
*
* \return 					A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS 		No errors; any other value indicates failure
* \retval VX_ERROR_INVALID_REFERENCE  The threshold reference is not actually a threshold reference.
* \retval VX_ERROR_NOT_COMPATIBLE     The threshold object doesn’t have type VX_THRESHOLD_TYPE_BINARY
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \retval VX_ERROR_NO_MEMORY          Internal memory allocation failed.
*
* \ingroup group_scalar
*/
VX_API_ENTRY vx_status VX_API_CALL vxCopyThresholdValue(vx_threshold thresh, vx_pixel_value_t *value_ptr, vx_enum usage, vx_enum user_mem_type)
{
    AgoData * data = (AgoData *)thresh;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_THRESHOLD)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if ((user_mem_type == VX_MEMORY_TYPE_HOST) && value_ptr) {
            status = VX_ERROR_NOT_COMPATIBLE;
            if(data->u.thr.thresh_type == VX_THRESHOLD_TYPE_BINARY) {
                status = VX_ERROR_NO_MEMORY;
                AgoData * dataToSync = data->u.tensor.roiMaster ? data->u.tensor.roiMaster : data;
#if ENABLE_OPENCL
            if (dataToSync->opencl_buffer && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                // make sure dirty OpenCL buffers are synched before giving access for read
                if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                    // transfer only valid data
                    if (dataToSync->size > 0) {
                        cl_int err = clEnqueueReadBuffer(dataToSync->ref.context->opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->gpu_buffer_offset, dataToSync->size, dataToSync->buffer, 0, NULL, NULL);
                        if (err) {
                            status = VX_FAILURE;
                            agoAddLogEntry(&dataToSync->ref, status, "ERROR: vxCopyThresholdValue: clEnqueueReadBuffer() => %d\n", err);
                            return status;
                        }
                    }
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                }
            }
#elif ENABLE_HIP
            if (dataToSync->hip_memory && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                // make sure dirty HIP buffers are synched before giving access for read
                if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                    // transfer only valid data
                    if (dataToSync->size > 0) {
                        hipError_t err = hipMemcpyDtoH((void *)dataToSync->buffer, dataToSync->hip_memory + dataToSync->gpu_buffer_offset, dataToSync->size);
                        if (err) {
                            status = VX_FAILURE;
                            agoAddLogEntry(&dataToSync->ref, status, "ERROR: vxCopyThresholdValue: hipMemcpyDtoH() => %d\n", err);
                            return status;
                        }
                    }
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                }
            }
#endif
                if (usage == VX_READ_ONLY){
                    //*(vx_int32 *)value_ptr = data->u.thr.threshold_lower;
                    memcpy(value_ptr, &data->u.thr.threshold_value, sizeof(vx_pixel_value_t));
                    status = VX_SUCCESS;
                }
                else if (usage == VX_WRITE_ONLY){
                    //data->u.thr.threshold_lower = *(vx_int32 *)value_ptr;
                    memcpy(&data->u.thr.threshold_value, value_ptr, sizeof(vx_pixel_value_t));
                    status = VX_SUCCESS;
                }
            }
        }
    }
    return status;
}

/*! \brief Allows the application to copy thresholding values from/into a threshold object with type <tt>\ref VX_THRESHOLD_TYPE_RANGE</tt>.
 * \param [in] thresh 			   	The reference to the threshold object that is the source or the destination of the copy.
 * \param [in,out] lower_value_ptr 	The address of the memory location where to store the lower thresholding value if the copy was requested in read mode,
 * 									or from where to get the lower thresholding value to store into the threshold object if the copy was requested in write mode.
 * \param [in,out] upper_value_ptr 	The address of the memory location where to store the upper thresholding value if the copy was requested in read mode,
 *									or from where to get the upper thresholding value to store into the threshold object if the copy was requested in write mode.
 * \param [in] usage 				This declares the effect of the copy with regard to the threshold object using the <tt>\ref vx_accessor_e</tt> enumeration.
 * 									Only <tt>\ref VX_READ_ONLY</tt> and <tt>\ref VX_WRITE_ONLY</tt> are supported:
 * \arg <tt>\ref VX_READ_ONLY</tt> 	Means that thresholding values are copied from the threshold object into the user memory.
 * 									After the copy, only the field of (*lower_value_ptr) and (*upper_value_ptr) unions that corresponds
 * 									to the input image format of the threshold object is meaningful.
 * \arg <tt>\ref VX_WRITE_ONLY</tt> Means the field of the (*lower_value_ptr) and (*upper_value_ptr) unions corresponding to the input format of the
 * 									threshold object is copied into the threshold object.
 * \param [in] user_mem_type 		A <tt>\ref vx_memory_type_e</tt> enumeration that specifies the type of the memory referenced by \p lower_value_ptr and
 * 									\p upper_value_ptr.
 *
 * \return 								A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS 					No errors; any other value indicates failure
 * \retval VX_ERROR_INVALID_REFERENCE  	The threshold reference is not actually a threshold reference.
 * \retval VX_ERROR_NOT_COMPATIBLE     	The threshold object doesn’t have type VX_THRESHOLD_TYPE_RANGE
 * \retval VX_ERROR_INVALID_PARAMETERS 	An other parameter is incorrect.
 * \retval VX_ERROR_NO_MEMORY          	Internal memory allocation failed.
 *
 * \ingroup group_threshold
 */
VX_API_ENTRY vx_status VX_API_CALL vxCopyThresholdRange(vx_threshold thresh, vx_pixel_value_t * lower_value_ptr, vx_pixel_value_t * upper_value_ptr, vx_enum usage, vx_enum user_mem_type)
{
    AgoData * data = (AgoData *)thresh;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_THRESHOLD)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if ((user_mem_type == VX_MEMORY_TYPE_HOST) && lower_value_ptr && upper_value_ptr) {
            status = VX_ERROR_NOT_COMPATIBLE;
            if(data->u.thr.thresh_type == VX_THRESHOLD_TYPE_RANGE) {
                status = VX_ERROR_NO_MEMORY;
                AgoData * dataToSync = data->u.tensor.roiMaster ? data->u.tensor.roiMaster : data;
#if ENABLE_OPENCL
            if (dataToSync->opencl_buffer && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                // make sure dirty OpenCL buffers are synched before giving access for read
                if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                    // transfer only valid data
                    if (dataToSync->size > 0) {
                        cl_int err = clEnqueueReadBuffer(dataToSync->ref.context->opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->gpu_buffer_offset, dataToSync->size, dataToSync->buffer, 0, NULL, NULL);
                        if (err) {
                            status = VX_FAILURE;
                            agoAddLogEntry(&dataToSync->ref, status, "ERROR: vxCopyThresholdValue: clEnqueueReadBuffer() => %d\n", err);
                            return status;
                        }
                    }
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                }
            }
#elif ENABLE_HIP
            if (dataToSync->hip_memory && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                // make sure dirty HIP buffers are synched before giving access for read
                if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                    // transfer only valid data
                    if (dataToSync->size > 0) {
                        hipError_t err = hipMemcpyDtoH((void *)dataToSync->buffer, dataToSync->hip_memory + dataToSync->gpu_buffer_offset, dataToSync->size);
                        if (err) {
                            status = VX_FAILURE;
                            agoAddLogEntry(&dataToSync->ref, status, "ERROR: vxCopyThresholdValue: hipMemcpyDtoH() => %d\n", err);
                            return status;
                        }
                    }
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                }
            }
#endif
                if (usage == VX_READ_ONLY){
                    memcpy(lower_value_ptr, &data->u.thr.threshold_lower, sizeof(vx_pixel_value_t));
                    memcpy(upper_value_ptr, &data->u.thr.threshold_upper, sizeof(vx_pixel_value_t));
                    status = VX_SUCCESS;
                }
                else if (usage == VX_WRITE_ONLY){
                    memcpy(&data->u.thr.threshold_lower, lower_value_ptr, sizeof(vx_pixel_value_t));
                    memcpy(&data->u.thr.threshold_upper, upper_value_ptr, sizeof(vx_pixel_value_t));
                    status = VX_SUCCESS;
                }
            }
        }
    }
    return status;
}

/*! \brief Allows the application to copy the true and false output values from/into a threshold object.
 * \param [in] thresh 				The reference to the threshold object that is the source or the destination of the copy.
 * \param [in,out] true_value_ptr 	The address of the memory location where to store the true output value if the copy was requested in read mode,
 * 									or from where to get the true output value to store into the threshold object if the copy was requested in write mode.
 * \param [in,out] false_value_ptr 	The address of the memory location where to store the false output value if the copy was requested in read mode, or
 * 									from where to get the false output value to store into the threshold object if the copy was requested in write mode.
 * \param [in] usage 				This declares the effect of the copy with regard to the threshold object using the <tt>\ref vx_accessor_e</tt> enumeration.
 * 									Only <tt>\ref VX_READ_ONLY</tt> and <tt>\ref VX_WRITE_ONLY</tt> are supported:
 * \arg <tt>\ref VX_READ_ONLY</tt> 	Means that true and false output values are copied from the threshold object into the user memory. After the copy,
 * 									only the field of (*true_value_ptr) and (*false_value_ptr) unions that corresponds to the output image format of the threshold object is meaningful.
 * \arg <tt>\ref VX_WRITE_ONLY</tt> Means the field of the (*true_value_ptr) and (*false_value_ptr) unions corresponding to the output format of the
 * 									threshold object is copied into the threshold object.
 * \param [in] user_mem_type 		A <tt>\ref vx_memory_type_e</tt> enumeration that specifies the type of the memory referenced by \p true_value_ptr and \p false_value_ptr.
 *
 * \return 								A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS					No errors; any other value indicates failure
 * \retval VX_ERROR_INVALID_REFERENCE 	The threshold reference is not actually a threshold reference.
 * \retval VX_ERROR_INVALID_PARAMETERS 	An other parameter is incorrect.
 * \retval VX_ERROR_NO_MEMORY 			Internal memory allocation failed
 *
 * \ingroup group_threshold
 */
VX_API_ENTRY vx_status VX_API_CALL vxCopyThresholdOutput(vx_threshold thresh, vx_pixel_value_t * true_value_ptr, vx_pixel_value_t * false_value_ptr, vx_enum usage, vx_enum user_mem_type)
{
    AgoData * data = (AgoData *)thresh;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_THRESHOLD)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if ((user_mem_type == VX_MEMORY_TYPE_HOST) && true_value_ptr && false_value_ptr) {
            status = VX_ERROR_NO_MEMORY;
                AgoData * dataToSync = data->u.tensor.roiMaster ? data->u.tensor.roiMaster : data;
#if ENABLE_OPENCL
            if (dataToSync->opencl_buffer && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                // make sure dirty OpenCL buffers are synched before giving access for read
                if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                    // transfer only valid data
                    if (dataToSync->size > 0) {
                        cl_int err = clEnqueueReadBuffer(dataToSync->ref.context->opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->gpu_buffer_offset, dataToSync->size, dataToSync->buffer, 0, NULL, NULL);
                        if (err) {
                            status = VX_FAILURE;
                            agoAddLogEntry(&dataToSync->ref, status, "ERROR: vxCopyThresholdValue: clEnqueueReadBuffer() => %d\n", err);
                            return status;
                        }
                    }
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                }
            }
#elif ENABLE_HIP
            if (dataToSync->hip_memory && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                // make sure dirty HIP buffers are synched before giving access for read
                if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                    // transfer only valid data
                    if (dataToSync->size > 0) {
                        hipError_t err = hipMemcpyDtoH((void *)dataToSync->buffer, dataToSync->hip_memory + dataToSync->gpu_buffer_offset, dataToSync->size);
                        if (err) {
                            status = VX_FAILURE;
                            agoAddLogEntry(&dataToSync->ref, status, "ERROR: vxCopyThresholdValue: hipMemcpyDtoH() => %d\n", err);
                            return status;
                        }
                    }
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                }
            }
#endif
            if (usage == VX_READ_ONLY){
                memcpy(true_value_ptr, &data->u.thr.true_value, sizeof(vx_pixel_value_t));
                memcpy(false_value_ptr, &data->u.thr.false_value, sizeof(vx_pixel_value_t));
                status = VX_SUCCESS;
            }
            else if (usage == VX_WRITE_ONLY){
                memcpy(&data->u.thr.true_value, true_value_ptr, sizeof(vx_pixel_value_t));
                memcpy(&data->u.thr.false_value, false_value_ptr, sizeof(vx_pixel_value_t));
                status = VX_SUCCESS;
            }
        }
    }
    return status;
}

/*! \brief Releases a reference to a threshold object.
* The object may not be garbage collected until its total reference count is zero.
* \param [in] thresh The pointer to the threshold to release.
* \post After returning from this function the reference is zeroed.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
* \ingroup group_threshold
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseThreshold(vx_threshold *thresh)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (thresh && agoIsValidData((AgoData*)*thresh, VX_TYPE_THRESHOLD)) {
        if (!agoReleaseData((AgoData*)*thresh, true)) {
            *thresh = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Sets attributes on the threshold object.
* \param [in] thresh The threshold object to set.
* \param [in] attribute The attribute to modify. Use a <tt>\ref vx_threshold_attribute_e</tt> enumeration.
* \param [in] ptr The pointer to the value to which to set the attribute.
* \param [in] size The size of the data pointed to by \a ptr.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_threshold
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetThresholdAttribute(vx_threshold thresh, vx_enum attribute, const void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)thresh;
    if (agoIsValidData(data, VX_TYPE_THRESHOLD)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE:
                if (size == sizeof(vx_int32) && data->u.thr.thresh_type == VX_THRESHOLD_TYPE_BINARY) {
                    data->u.thr.threshold_value = *(vx_pixel_value_t *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_THRESHOLD_TYPE:
                if (size == sizeof(vx_enum)) {
                    data->u.thr.thresh_type = *(vx_enum *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_THRESHOLD_INPUT_FORMAT:
                if (size == sizeof(vx_df_image)) {
                    data->u.thr.input_format = *(vx_df_image *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_THRESHOLD_OUTPUT_FORMAT:
                if (size == sizeof(vx_df_image)) {
                    data->u.thr.output_format = *(vx_df_image *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER:
                if (size == sizeof(vx_int32) && data->u.thr.thresh_type == VX_THRESHOLD_TYPE_RANGE) {
                    data->u.thr.threshold_lower = *(vx_pixel_value_t *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER:
                if (size == sizeof(vx_int32) && data->u.thr.thresh_type == VX_THRESHOLD_TYPE_RANGE) {
                    data->u.thr.threshold_upper = *(vx_pixel_value_t *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Queries an attribute on the threshold object.
* \param [in] thresh The threshold object to set.
* \param [in] attribute The attribute to query. Use a <tt>\ref vx_threshold_attribute_e</tt> enumeration.
* \param [out] ptr The location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_threshold
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryThreshold(vx_threshold thresh, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)thresh;
    if (agoIsValidData(data, VX_TYPE_THRESHOLD)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_THRESHOLD_ATTRIBUTE_TYPE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = data->u.thr.thresh_type;
                    status = VX_SUCCESS;
                }
                break;
            case VX_THRESHOLD_INPUT_FORMAT:
                if (size == sizeof(vx_df_image)) {
                    *(vx_df_image *)ptr = data->u.thr.input_format;
                    status = VX_SUCCESS;
                }
                break;
            case VX_THRESHOLD_OUTPUT_FORMAT:
                if (size == sizeof(vx_df_image)) {
                    *(vx_df_image *)ptr = data->u.thr.output_format;
                    status = VX_SUCCESS;
                }
                break;
            case VX_THRESHOLD_ATTRIBUTE_DATA_TYPE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = data->u.thr.data_type;
                    status = VX_SUCCESS;
                }
                break;
            case VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE:
                if (size == sizeof(vx_int32) && data->u.thr.thresh_type == VX_THRESHOLD_TYPE_BINARY) {
                    *(vx_pixel_value_t *)ptr = data->u.thr.threshold_value;
                    status = VX_SUCCESS;
                }
                break;
            case VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER:
                if (size == sizeof(vx_int32) && data->u.thr.thresh_type == VX_THRESHOLD_TYPE_RANGE) {
                    *(vx_pixel_value_t *)ptr = data->u.thr.threshold_lower;
                    status = VX_SUCCESS;
                }
                break;
            case VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER:
                if (size == sizeof(vx_int32) && data->u.thr.thresh_type == VX_THRESHOLD_TYPE_RANGE) {
                    *(vx_pixel_value_t *)ptr = data->u.thr.threshold_upper;
                    status = VX_SUCCESS;
                }
                break;
            case VX_THRESHOLD_ATTRIBUTE_TRUE_VALUE:
                if (size == sizeof(vx_int32)) {
                    *(vx_pixel_value_t *)ptr = data->u.thr.true_value;
                    status = VX_SUCCESS;
                }
                break;
            case VX_THRESHOLD_ATTRIBUTE_FALSE_VALUE:
                if (size == sizeof(vx_int32)) {
                    *(vx_pixel_value_t *)ptr = data->u.thr.false_value;
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*==============================================================================
MATRIX
=============================================================================*/

/*! \brief Creates a reference to a matrix object.
* \param [in] c The reference to the overall context.
* \param [in] data_type The unit format of the matrix. <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_INT32</tt> or <tt>\ref VX_TYPE_FLOAT32</tt>.
* \param [in] columns The first dimensionality.
* \param [in] rows The second dimensionality.
* \returns An matrix reference <tt>\ref vx_matrix</tt>. Any possible errors preventing a
* successful creation should be checked using <tt>\ref vxGetStatus</tt>.
* \ingroup group_matrix
*/
VX_API_ENTRY vx_matrix VX_API_CALL vxCreateMatrix(vx_context context, vx_enum data_type, vx_size columns, vx_size rows)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context) && (data_type == VX_TYPE_INT32 || data_type == VX_TYPE_FLOAT32 || data_type == VX_TYPE_UINT8) && columns > 0 && rows > 0) {
        CAgoLock lock(context->cs);
        char desc[512]; snprintf(desc, sizeof(desc), "matrix:%s," VX_FMT_SIZE "," VX_FMT_SIZE "", agoEnum2Name(data_type), columns, rows);
        data = agoCreateDataFromDescription(context, NULL, desc, true);
        if (data) {
            agoGenerateDataName(context, "matrix", data->name);
            agoAddData(&context->dataList, data);
        }
    }
    return (vx_matrix)data;
}

/*! \brief Creates an opaque reference to a matrix object without direct user access.
* \param [in] graph The reference to the parent graph.
* \param [in] data_type The unit format of the matrix.
* \param [in] columns The first dimensionality.
* \param [in] rows The second dimensionality.
* \returns An matrix reference <tt>\ref vx_matrix</tt>. Any possible errors preventing a
* successful creation should be checked using <tt>\ref vxGetStatus</tt>.
* \ingroup group_matrix
*/
VX_API_ENTRY vx_matrix VX_API_CALL vxCreateVirtualMatrix(vx_graph graph, vx_enum data_type, vx_size columns, vx_size rows)
{
    AgoData * data = NULL;
    if (agoIsValidGraph(graph) && (data_type == VX_TYPE_INT32 || data_type == VX_TYPE_FLOAT32 || data_type == VX_TYPE_UINT8) && columns > 0 && rows > 0) {
        CAgoLock lock(graph->cs);
        char desc[512]; snprintf(desc, sizeof(desc), "matrix-virtual:%s," VX_FMT_SIZE "," VX_FMT_SIZE "", agoEnum2Name(data_type), columns, rows);
        data = agoCreateDataFromDescription(graph->ref.context, graph, desc, true);
        if (data) {
            agoGenerateVirtualDataName(graph, "matrix", data->name);
            agoAddData(&graph->dataList, data);
        }
    }
    return (vx_matrix)data;
}

/*! \brief Releases a reference to a matrix object.
* The object may not be garbage collected until its total reference count is zero.
* \param [in] mat The matrix reference to release.
* \post After returning from this function the reference is zeroed.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
* \ingroup group_matrix
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseMatrix(vx_matrix *mat)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (mat && agoIsValidData((AgoData*)*mat, VX_TYPE_MATRIX)) {
        if (!agoReleaseData((AgoData*)*mat, true)) {
            *mat = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Queries an attribute on the matrix object.
* \param [in] mat The matrix object to set.
* \param [in] attribute The attribute to query. Use a <tt>\ref vx_matrix_attribute_e</tt> enumeration.
* \param [out] ptr The location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_matrix
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryMatrix(vx_matrix mat, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)mat;
    if (agoIsValidData(data, VX_TYPE_MATRIX)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_MATRIX_TYPE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = data->u.mat.type;
                    status = VX_SUCCESS;
                }
                break;
            case VX_MATRIX_ROWS:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->u.mat.rows;
                    status = VX_SUCCESS;
                }
                break;
            case VX_MATRIX_COLUMNS:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->u.mat.columns;
                    status = VX_SUCCESS;
                }
                break;
            case VX_MATRIX_SIZE:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->size;
                    status = VX_SUCCESS;
                }
                break;
            case VX_MATRIX_PATTERN:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = data->u.mat.pattern ? data->u.mat.pattern : VX_PATTERN_OTHER;
                    status = VX_SUCCESS;
                }
                break;
            case VX_MATRIX_ORIGIN:
                if (size == sizeof(vx_coordinates2d_t)) {
                    *(vx_coordinates2d_t *)ptr = data->u.mat.origin;
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Gets the matrix data (copy).
* \param [in] mat The reference to the matrix.
* \param [out] array The array in which to place the matrix.
* \see vxQueryMatrix and <tt>\ref VX_MATRIX_ATTRIBUTE_COLUMNS</tt> and <tt>\ref VX_MATRIX_ATTRIBUTE_ROWS</tt>
* to get the needed number of elements of the array.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \post <tt>\ref vxCommitMatrix</tt>
* \ingroup group_matrix
*/
VX_API_ENTRY vx_status VX_API_CALL vxReadMatrix(vx_matrix mat, void *array)
{
    AgoData * data = (AgoData *)mat;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_MATRIX)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else {
            if (array) {
                if (!data->buffer) {
                    CAgoLock lock(data->ref.context->cs);
                    if (agoAllocData(data)) {
                        return VX_FAILURE;
                    }
                }
#if ENABLE_OPENCL
                if (data->opencl_buffer && !(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                    // make sure dirty OpenCL buffers are synched before giving access for read
                    if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                        // transfer only valid data
                        vx_size size = data->size;
                        if (size > 0) {
                            cl_int err = clEnqueueReadBuffer(data->ref.context->opencl_cmdq, data->opencl_buffer, CL_TRUE, 0, size, data->buffer, 0, NULL, NULL);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&data->ref, status, "ERROR: vxReadMatrix: clEnqueueReadBuffer() => %d\n", err);
                                return status;
                            }
                        }
                        data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                    }
                }
#elif ENABLE_HIP
                if (data->hip_memory && !(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                    // make sure dirty OpenCL buffers are synched before giving access for read
                    if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                        // transfer only valid data
                        vx_size size = data->size;
                        if (size > 0 && data->hip_memory) {
                            hipError_t err = hipMemcpyDtoH((void *)data->buffer, (data->hip_memory + data->gpu_buffer_offset), size);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&data->ref, status, "ERROR: vxReadMatrix: hipMemcpyDtoH() => %d\n", err);
                                return status;
                            }
                        }
                        data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                    }
                }
#endif
                // copy to external buffer
                HafCpu_BinaryCopy_U8_U8(data->size, (vx_uint8 *)array, data->buffer);
            }
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Sets the matrix data (copy)
* \param [in] mat The reference to the matrix.
* \param [out] array The array to read the matrix.
* \see vxQueryMatrix and <tt>\ref VX_MATRIX_ATTRIBUTE_COLUMNS</tt> and <tt>\ref VX_MATRIX_ATTRIBUTE_ROWS</tt>
* to get the needed number of elements of the array.'
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \pre <tt>\ref vxAccessMatrix</tt>
* \ingroup group_matrix
*/
VX_API_ENTRY vx_status VX_API_CALL vxWriteMatrix(vx_matrix mat, const void *array)
{
    AgoData * data = (AgoData *)mat;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_MATRIX)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if (data->ref.read_only) {
            status = VX_ERROR_NOT_SUPPORTED;
        }
        else {
            if (array) {
                if (!data->buffer) {
                    CAgoLock lock(data->ref.context->cs);
                    if (agoAllocData(data)) {
                        return VX_FAILURE;
                    }
                }
                // copy from external buffer
                HafCpu_BinaryCopy_U8_U8(data->size, data->buffer, (vx_uint8 *)array);
                // update sync flags
                data->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
            }
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Allows the application to copy from/into a matrix object.
* \param [in] matrix The reference to the matrix object that is the source or the
* destination of the copy.
* \param [in] user_ptr The address of the memory location where to store the requested data
* if the copy was requested in read mode, or from where to get the data to store into the matrix
* object if the copy was requested in write mode. In the user memory, the matrix is
* structured as a row-major 2D array with elements of the type corresponding to
* <tt>\ref VX_MATRIX_TYPE</tt>, with a number of rows corresponding to
* <tt>\ref VX_MATRIX_ROWS</tt> and a number of columns corresponding to
* <tt>\ref VX_MATRIX_COLUMNS</tt>. The accessible memory must be large
* enough to contain this 2D array:
* accessible memory in bytes >= sizeof(data_element) * rows * columns.
* \param [in] usage This declares the effect of the copy with regard to the matrix object
* using the <tt>\ref vx_accessor_e</tt> enumeration. Only VX_READ_ONLY and VX_WRITE_ONLY
* are supported:
* \arg VX_READ_ONLY means that data are copied from the matrix object into the user memory.
* \arg VX_WRITE_ONLY means that data are copied into the matrix object from the user memory.
* \param [in] user_mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that specifies
* the memory type of the memory referenced by the user_addr.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_INVALID_REFERENCE The matrix reference is not actually a matrix reference.
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \ingroup group_matrix
*/
VX_API_ENTRY vx_status VX_API_CALL vxCopyMatrix(vx_matrix matrix, void *user_ptr, vx_enum usage, vx_enum user_mem_type)
{
    AgoData * data = (AgoData *)matrix;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_MATRIX))
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        if ((user_mem_type == VX_MEMORY_TYPE_HOST) && user_ptr) {
            if (usage == VX_READ_ONLY)
                status = vxReadMatrix(matrix, user_ptr);
            else if (usage == VX_WRITE_ONLY)
                status = vxWriteMatrix(matrix, user_ptr);
        }
    }
    return status;
}

/*! \brief Creates a reference to a matrix object from a boolean pattern.
*
* The matrix created by this function is of type <tt>\ref vx_uint8</tt>, with the value 0 representing False,
* and the value 255 representing True. It supports patterns described below. See <tt>\ref vx_pattern_e</tt>.
* - VX_PATTERN_BOX is a matrix with dimensions equal to the given number of rows and columns, and all cells equal to 255.
*   Dimensions of 3x3 and 5x5 must be supported.
* - VX_PATTERN_CROSS is a matrix with dimensions equal to the given number of rows and columns, which both must be odd numbers.
*   All cells in the center row and center column are equal to 255, and the rest are equal to zero.
*   Dimensions of 3x3 and 5x5 must be supported.
* - VX_PATTERN_DISK is an RxC matrix, where R and C are odd and cell (c, r) is 255 if: \n
*   (r-R/2 + 0.5)^2 / (R/2)^2 + (c-C/2 + 0.5)^2/(C/2)^2 is less than or equal to 1,\n and 0 otherwise.
* - VX_PATTERN_OTHER is any other pattern than the above (matrix created is still binary, with a value of 0 or 255).
*
* If the matrix was created via <tt>\ref vxCreateMatrixFromPattern</tt>, this attribute must be set to the
* appropriate pattern enum. Otherwise the attribute must be set to VX_PATTERN_OTHER.
* The vx_matrix objects returned by this function are read-only. The behavior when attempting to modify such a matrix is undefined.
*
* \param [in] context The reference to the overall context.
* \param [in] pattern The pattern of the matrix. See <tt>\ref VX_MATRIX_PATTERN</tt>.
* \param [in] columns The first dimensionality.
* \param [in] rows The second dimensionality.
* \returns An matrix reference <tt>\ref vx_matrix</tt> of type <tt>\ref vx_uint8</tt>. Any possible errors preventing a
* successful creation should be checked using <tt>\ref vxGetStatus</tt>.
* \ingroup group_matrix
*/
VX_API_ENTRY vx_matrix VX_API_CALL vxCreateMatrixFromPattern(vx_context context, vx_enum pattern, vx_size columns, vx_size rows)
{
    vx_matrix mat = nullptr;
    // check for supported patterns
    if (pattern == VX_PATTERN_BOX || pattern == VX_PATTERN_CROSS || pattern == VX_PATTERN_DISK || pattern == VX_PATTERN_OTHER)
    {
        // create a matrix object
        mat = vxCreateMatrix(context, VX_TYPE_UINT8, columns, rows);
        vx_status status = vxGetStatus((vx_reference)mat);
        if (status == VX_SUCCESS) {
            // initialize with the pattern and mark it
            AgoData * data = (AgoData *)mat;
            data->u.mat.pattern = pattern;
            if (pattern != VX_PATTERN_OTHER) {
                vx_uint8 * buf = new vx_uint8[columns * rows];
                if (!buf) {
                    vxReleaseMatrix(&mat);
                    return mat;
                }
                if (pattern == VX_PATTERN_CROSS) {
                    // cross pattern
                    memset(buf, 0, columns * rows);
                    for (vx_size x = 0; x < columns; x++)
                        buf[columns * (rows / 2) + x] = 255;
                    for (vx_size y = 0; y < rows; y++)
                        buf[(columns / 2) + y * columns] = 255;
                }
                else if (pattern == VX_PATTERN_DISK) {
                    // disk pattern
                    for (vx_size r = 0; r < rows; r++) {
                        float y = ((r - rows*0.5f + 0.5f) / (rows*0.5f));
                        float y2 = y * y;
                        for (vx_size c = 0; c < columns; c++) {
                            float x = ((c - columns*0.5f + 0.5f) / (columns*0.5f));
                            float x2 = x * x;
                            buf[c + r * columns] = ((x2 + y2) <= 1.0f) ? 255 : 0;
                        }
                    }
                }
                else {
                    // box pattern
                    memset(buf, 255, columns * rows);
                }
                status = vxCopyMatrix(mat, buf, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
                if (status) {
                    vxReleaseMatrix(&mat);
                    return mat;
                }
                delete[] buf;
            }
            // mark the matrix read-only
            data->ref.read_only = true;
        }
    }
    return mat;
}

/*! \brief Creates a reference to a matrix object from a boolean pattern, with a user-specified origin.
 *
 * The matrix created by this function is of type <tt>\ref VX_TYPE_UINT8</tt>, with the value 0 representing False,
 * and the value 255 representing True. It supports the patterns as described below:
 * - VX_PATTERN_BOX is a matrix with dimensions equal to the given number of rows and columns, and all cells equal to 255.
 *   Dimensions of 3x3 and 5x5 must be supported.
 * - VX_PATTERN_CROSS is a matrix with dimensions equal to the given number of rows and columns, which both must be odd numbers.
 *   All cells in the center row and center column are equal to 255, and the rest are equal to zero.
 *   Dimensions of 3x3 and 5x5 must be supported.
 * - VX_PATTERN_DISK is a matrix with dimensions equal to the given number of rows (R) and columns (C),
 *   where R and C are odd and cell (c, r) is 255 if: \n
 *   (r-R/2 + 0.5)^2 / (R/2)^2 + (c-C/2 + 0.5)^2/(C/2)^2 is less than or equal to 1,\n and 0 otherwise.
 *
 * A matrix created from pattern is read-only. The behavior when attempting to modify such a matrix is undefined.
 *
 * \param [in] context The reference to the overall context.
 * \param [in] pattern The pattern of the matrix. See <tt>\ref VX_MATRIX_PATTERN</tt>.
 * \param [in] columns The first dimensionality.
 * \param [in] rows The second dimensionality.
 * \param [in] origin_col The origin (first dimensionality).
 * \param [in] origin_row The origin (second dimensionality).
 * \returns A matrix reference <tt>\ref vx_matrix</tt> of type <tt>\ref VX_TYPE_UINT8</tt>. Any possible errors
 * preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_matrix
 */
VX_API_ENTRY vx_matrix VX_API_CALL vxCreateMatrixFromPatternAndOrigin(vx_context context, vx_enum pattern, vx_size columns, vx_size rows, vx_size origin_col, vx_size origin_row)
{
    vx_matrix mat = nullptr;
    // check for supported patterns
    if (pattern == VX_PATTERN_BOX || pattern == VX_PATTERN_CROSS || pattern == VX_PATTERN_DISK || pattern == VX_PATTERN_OTHER)
    {
        // create a matrix object
        mat = vxCreateMatrix(context, VX_TYPE_UINT8, columns, rows);
        vx_status status = vxGetStatus((vx_reference)mat);
        if (status == VX_SUCCESS) {
            // initialize with the pattern and mark it
            AgoData * data = (AgoData *)mat;
            data->u.mat.pattern = pattern;
            data->u.mat.origin.x = (vx_uint32)origin_col;
            data->u.mat.origin.y = (vx_uint32)origin_row;
            if (pattern != VX_PATTERN_OTHER) {
                vx_uint8 * buf = new vx_uint8[columns * rows];
                if (!buf) {
                    vxReleaseMatrix(&mat);
                    return mat;
                }
                if (pattern == VX_PATTERN_CROSS) {
                    // cross pattern
                    memset(buf, 0, columns * rows);
                    for (vx_size x = 0; x < columns; x++)
                        buf[columns * (rows / 2) + x] = 255;
                    for (vx_size y = 0; y < rows; y++)
                        buf[(columns / 2) + y * columns] = 255;
                }
                else if (pattern == VX_PATTERN_DISK) {
                    // disk pattern
                    for (vx_size r = 0; r < rows; r++) {
                        float y = ((r - rows*0.5f + 0.5f) / (rows*0.5f));
                        float y2 = y * y;
                        for (vx_size c = 0; c < columns; c++) {
                            float x = ((c - columns*0.5f + 0.5f) / (columns*0.5f));
                            float x2 = x * x;
                            buf[c + r * columns] = ((x2 + y2) <= 1.0f) ? 255 : 0;
                        }
                    }
                }
                else {
                    // box pattern
                    memset(buf, 255, columns * rows);
                }
                status = vxCopyMatrix(mat, buf, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
                if (status) {
                    vxReleaseMatrix(&mat);
                    return mat;
                }
                delete[] buf;
            }
            // mark the matrix read-only
            data->ref.read_only = true;
        }
    }
    return mat;
}

/*==============================================================================
CONVOLUTION
=============================================================================*/

/*! \brief Creates a reference to a convolution matrix object.
* \param [in] context The reference to the overall context.
* \param [in] columns The columns dimension of the convolution.
* Must be odd and greater than or equal to 3 and less than the value returned
* from <tt>\ref VX_CONTEXT_ATTRIBUTE_CONVOLUTION_MAXIMUM_DIMENSION</tt>.
* \param [in] rows The rows dimension of the convolution.
* Must be odd and greater than or equal to 3 and less than the value returned
* from <tt>\ref VX_CONTEXT_ATTRIBUTE_CONVOLUTION_MAXIMUM_DIMENSION</tt>.
* \return <tt>\ref vx_convolution</tt>
* \ingroup group_convolution
*/
VX_API_ENTRY vx_convolution VX_API_CALL vxCreateConvolution(vx_context context, vx_size columns, vx_size rows)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context) && columns > 0 && rows > 0) {
        CAgoLock lock(context->cs);
        char desc[512]; snprintf(desc, sizeof(desc), "convolution:" VX_FMT_SIZE "," VX_FMT_SIZE "", columns, rows);
        data = agoCreateDataFromDescription(context, NULL, desc, true);
        if (data) {
            agoGenerateDataName(context, "conv", data->name);
            agoAddData(&context->dataList, data);
        }
    }
    return (vx_convolution)data;
}

/*! \brief Creates an opaque reference to a convolution matrix object without direct user access.
* \param [in] graph - The reference to the parent graph.
* \param [in] columns - The columns dimension of the convolution. Must be odd and greater than or equal to 3 and less than the value returned from VX_CONTEXT_CONVOLUTION_MAX_DIMENSION
* \param [in] rows - The rows dimension of the convolution. Must be odd and greater than or equal to 3 and less than the value returned from VX_CONTEXT_CONVOLUTION_MAX_DIMENSION
* \return <tt>\ref vx_convolution</tt>
* Any possible errors preventing a successful creation should be checked using vxGetStatus.
*/
VX_API_ENTRY vx_convolution VX_API_CALL vxCreateVirtualConvolution(vx_graph graph, vx_size columns, vx_size rows)
{
    AgoData * data = NULL;
    if (agoIsValidGraph(graph)) {
        CAgoLock lock(graph->cs);
        if(columns >= 3 && rows >= 3 && (columns % 2 == 1) && (rows % 2 == 1)) {
            char desc[512]; snprintf(desc, sizeof(desc), "convolution-virtual:" VX_FMT_SIZE "," VX_FMT_SIZE "", columns, rows);
            data = agoCreateDataFromDescription(graph->ref.context, graph, desc, true);
            if (data) {
                agoGenerateVirtualDataName(graph, "conv", data->name);
                agoAddData(&graph->dataList, data);
            }
        }
    }
    return (vx_convolution)data;
}

/*! \brief Releases the reference to a convolution matrix.
* The object may not be garbage collected until its total reference count is zero.
* \param [in] conv The pointer to the convolution matrix to release.
* \post After returning from this function the reference is zeroed.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
* \ingroup group_convolution
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseConvolution(vx_convolution *conv)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (conv && agoIsValidData((AgoData*)*conv, VX_TYPE_CONVOLUTION)) {
        if (!agoReleaseData((AgoData*)*conv, true)) {
            *conv = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Queries an attribute on the convolution matrix object.
* \param [in] conv The convolution matrix object to set.
* \param [in] attribute The attribute to query. Use a <tt>\ref vx_convolution_attribute_e</tt> enumeration.
* \param [out] ptr The location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_convolution
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryConvolution(vx_convolution conv, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)conv;
    if (agoIsValidData(data, VX_TYPE_CONVOLUTION)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_CONVOLUTION_ATTRIBUTE_ROWS:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->u.conv.rows;
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONVOLUTION_ATTRIBUTE_COLUMNS:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->u.conv.columns;
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONVOLUTION_ATTRIBUTE_SCALE:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = 1u << data->u.conv.shift;
                    status = VX_SUCCESS;
                }
                break;
            case VX_CONVOLUTION_ATTRIBUTE_SIZE:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->size;
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Sets attributes on the convolution object.
* \param [in] conv The coordinates object to set.
* \param [in] attribute The attribute to modify. Use a <tt>\ref vx_convolution_attribute_e</tt> enumeration.
* \param [in] ptr The pointer to the value to which to set the attribute.
* \param [in] size The size of the data pointed to by \a ptr.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_convolution
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetConvolutionAttribute(vx_convolution conv, vx_enum attribute, const void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)conv;
    if (agoIsValidData(data, VX_TYPE_CONVOLUTION)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_CONVOLUTION_ATTRIBUTE_SCALE:
                if (size == sizeof(vx_uint32)) {
                    status = VX_ERROR_INVALID_VALUE;
                    vx_uint32 scale = *(vx_uint32 *)ptr;
                    for (vx_uint32 shift = 0; shift < 32; shift++) {
                        if (scale == (1u << shift)) {
                            data->u.conv.shift = shift;
                            status = VX_SUCCESS;
                            if (data->buffer && data->reserved) {
                                // update float values
                                vx_uint32 N = (vx_uint32)data->u.conv.columns * (vx_uint32)data->u.conv.rows;
                                float scale = 1.0f / (float)(1 << data->u.conv.shift);
                                short * ps = (short *)data->buffer;
                                float * pf = (float *)data->reserved;
                                for (vx_uint32 i = 0; i < N; i++)
                                    pf[N - 1 - i] = scale * ps[i]; // NOTE: the reversing of coefficients order required to be able to re-use linear filter
                            }
                            break;
                        }
                    }
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Gets the convolution data (copy).
* \param [in] conv The reference to the convolution.
* \param [out] array The array to place the convolution.
* \see vxQueryConvolution and <tt>\ref VX_CONVOLUTION_ATTRIBUTE_SIZE</tt> to get the
* needed number of bytes of the array.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \post <tt>\ref vxWriteConvolutionCoefficients</tt>
* \ingroup group_convolution
*/
VX_API_ENTRY vx_status VX_API_CALL vxReadConvolutionCoefficients(vx_convolution conv, vx_int16 *array)
{
    AgoData * data = (AgoData *)conv;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_CONVOLUTION)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else {
            if (array) {
                if (!data->buffer) {
                    CAgoLock lock(data->ref.context->cs);
                    if (agoAllocData(data)) {
                        return VX_FAILURE;
                    }
                }
                // copy to external buffer
                HafCpu_BinaryCopy_U8_U8(data->size, (vx_uint8 *)array, data->buffer);
            }
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Sets the convolution data (copy),
* \param [in] conv The reference to the convolution.
* \param [out] array The array to read the convolution.
* \see <tt>\ref vxQueryConvolution</tt> and <tt>\ref VX_CONVOLUTION_ATTRIBUTE_SIZE</tt> to get the
* needed number of bytes of the array.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \pre <tt>\ref vxReadConvolutionCoefficients</tt>
* \ingroup group_convolution
*/
VX_API_ENTRY vx_status VX_API_CALL vxWriteConvolutionCoefficients(vx_convolution conv, const vx_int16 *array)
{
    AgoData * data = (AgoData *)conv;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_CONVOLUTION)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if (data->ref.read_only) {
            status = VX_ERROR_NOT_SUPPORTED;
        }
        else {
            if (array) {
                if (!data->buffer) {
                    CAgoLock lock(data->ref.context->cs);
                    if (agoAllocData(data)) {
                        return VX_FAILURE;
                    }
                }
                // copy from external buffer
                HafCpu_BinaryCopy_U8_U8(data->size, data->buffer, (vx_uint8 *)array);
                // update float values
                vx_uint32 N = (vx_uint32)data->u.conv.columns * (vx_uint32)data->u.conv.rows;
                float scale = 1.0f / (float)(1 << data->u.conv.shift);
                if (data->buffer && data->reserved) {
                    short * ps = (short *)data->buffer;
                    float * pf = (float *)data->reserved;
                    for (vx_uint32 i = 0; i < N; i++)
                        pf[N - 1 - i] = scale * ps[i]; // NOTE: the reversing of coefficients order required to be able to re-use linear filter
                }
                // update sync flags
                data->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
            }
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Allows the application to copy coefficients from/into a convolution object.
* \param [in] convolution The reference to the convolution object that is the source or the destination of the copy.
* \param [in] user_ptr The address of the memory location where to store the requested
* coefficient data if the copy was requested in read mode, or from where to get the
* coefficient data to store into the convolution object if the copy was requested in
* write mode. In the user memory, the convolution coefficient data is structured as a
* row-major 2D array with elements of the type corresponding
* to <tt>\ref VX_TYPE_CONVOLUTION</tt>, with a number of rows corresponding to
* <tt>\ref VX_CONVOLUTION_ROWS</tt> and a number of columns corresponding to
* <tt>\ref VX_CONVOLUTION_COLUMNS</tt>. The accessible memory must be large
* enough to contain this 2D array:
* accessible memory in bytes >= sizeof(data_element) * rows * columns.
* \param [in] usage This declares the effect of the copy with regard to the convolution object
* using the <tt>\ref vx_accessor_e</tt> enumeration. Only VX_READ_ONLY and VX_WRITE_ONLY
* are supported:
* \arg VX_READ_ONLY means that data are copied from the convolution object into the user memory.
* \arg VX_WRITE_ONLY means that data are copied into the convolution object from the user memory.
* \param [in] user_mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that specifies
* the memory type of the memory referenced by the user_addr.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_INVALID_REFERENCE The convolution reference is not actually a convolution reference.
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \ingroup group_convolution
*/
VX_API_ENTRY vx_status VX_API_CALL vxCopyConvolutionCoefficients(vx_convolution conv, void *user_ptr, vx_enum usage, vx_enum user_mem_type)
{
    AgoData * data = (AgoData *)conv;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_CONVOLUTION))
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        if ((user_mem_type == VX_MEMORY_TYPE_HOST) && user_ptr) {
            if (usage == VX_READ_ONLY)
                status = vxReadConvolutionCoefficients(conv, (vx_int16 *)user_ptr);
            else if (usage == VX_WRITE_ONLY)
                status = vxWriteConvolutionCoefficients(conv, (const vx_int16 *)user_ptr);
        }
    }
    return status;
}

/*==============================================================================
PYRAMID
=============================================================================*/

/*! \brief Creates a reference to a pyramid object of the supplied number of levels.
* \param [in] context The reference to the overall context.
* \param [in] levels The number of levels desired. This is required to be a non-zero value.
* \param [in] scale Used to indicate the scale between pyramid levels. This is required to be a non-zero positive value.
* \if OPENVX_STRICT_1_0
* In OpenVX 1.0, the only permissible values are <tt>\ref VX_SCALE_PYRAMID_HALF</tt> or <tt>\ref VX_SCALE_PYRAMID_ORB</tt>.
* \endif
* \param [in] width The width of the 0th level image in pixels.
* \param [in] height The height of the 0th level image in pixels.
* \param [in] format The format of all images in the pyramid.
* \return <tt>\ref vx_pyramid</tt>
* \retval 0 No pyramid was created.
* \retval * A pyramid reference.
* \ingroup group_pyramid
*/
VX_API_ENTRY vx_pyramid VX_API_CALL vxCreatePyramid(vx_context context, vx_size levels, vx_float32 scale, vx_uint32 width, vx_uint32 height, vx_df_image format)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context)) {
        CAgoLock lock(context->cs);
        char desc_scale[64];
        if (scale == VX_SCALE_PYRAMID_HALF) snprintf(desc_scale, sizeof(desc_scale), "HALF");
        else if (scale == VX_SCALE_PYRAMID_ORB) snprintf(desc_scale, sizeof(desc_scale), "ORB");
        else snprintf(desc_scale, sizeof(desc_scale), "%.12g", scale);
        char desc[512]; snprintf(desc, sizeof(desc), "pyramid:%4.4s,%d,%d," VX_FMT_SIZE ",%s", FORMAT_STR(format), width, height, levels, desc_scale);
        data = agoCreateDataFromDescription(context, NULL, desc, true);
        if (data) {
            agoGenerateDataName(context, "pyramid", data->name);
            agoAddData(&context->dataList, data);
            // add the children too
            for (vx_uint32 i = 0; i < data->numChildren; i++) {
                agoAddData(&context->dataList, data->children[i]);
                for (vx_uint32 j = 0; j < data->children[i]->numChildren; j++) {
                    if (data->children[i]->children[j]) {
                        agoAddData(&context->dataList, data->children[i]->children[j]);
                    }
                }
            }
        }
    }
    return (vx_pyramid)data;
}

/*! \brief Creates a reference to a virtual pyramid object of the supplied number of levels.
* \details Virtual Pyramids can be used to connect Nodes together when the contents of the pyramids will
* not be accessed by the user of the API.
* All of the following constructions are valid:
* \code
* vx_context context = vxCreateContext();
* vx_graph graph = vxCreateGraph(context);
* vx_pyramid virt[] = {
*     vxCreateVirtualPyramid(graph, 4, VX_SCALE_PYRAMID_HALF, 0, 0, VX_DF_IMAGE_VIRT), // no dimension and format specified for level 0
*     vxCreateVirtualPyramid(graph, 4, VX_SCALE_PYRAMID_HALF, 640, 480, VX_DF_IMAGE_VIRT), // no format specified.
*     vxCreateVirtualPyramid(graph, 4, VX_SCALE_PYRAMID_HALF, 640, 480, VX_DF_IMAGE_U8), // no access
* };
* \endcode
* \param [in] graph The reference to the parent graph.
* \param [in] levels The number of levels desired. This is required to be a non-zero value.
* \param [in] scale Used to indicate the scale between pyramid levels. This is required to be a non-zero positive value.
* \if OPENVX_STRICT_1_0
* In OpenVX 1.0, the only permissible values are <tt>\ref VX_SCALE_PYRAMID_HALF</tt> or <tt>\ref VX_SCALE_PYRAMID_ORB</tt>.
* \endif
* \param [in] width The width of the 0th level image in pixels. This may be set to zero to indicate to the interface that the value is unspecified.
* \param [in] height The height of the 0th level image in pixels. This may be set to zero to indicate to the interface that the value is unspecified.
* \param [in] format The format of all images in the pyramid. This may be set to <tt>\ref VX_DF_IMAGE_VIRT</tt> to indicate that the format is unspecified.
* \return A <tt>\ref vx_pyramid</tt> reference.
* \note Images extracted with <tt>\ref vxGetPyramidLevel</tt> behave as Virtual Images and
* cause <tt>\ref vxAccessImagePatch</tt> to return errors.
* \retval 0 No pyramid was created.
* \retval * A pyramid reference.
* \ingroup group_pyramid
*/
VX_API_ENTRY vx_pyramid VX_API_CALL vxCreateVirtualPyramid(vx_graph graph, vx_size levels, vx_float32 scale, vx_uint32 width, vx_uint32 height, vx_df_image format)
{
    AgoData * data = NULL;
    if (agoIsValidGraph(graph)) {
        CAgoLock lock(graph->cs);
        char desc_scale[64];
        if (scale == VX_SCALE_PYRAMID_HALF) snprintf(desc_scale, sizeof(desc_scale), "HALF");
        else if (scale == VX_SCALE_PYRAMID_ORB) snprintf(desc_scale, sizeof(desc_scale), "ORB");
        else snprintf(desc_scale, sizeof(desc_scale), "%.12g", scale);
        char desc[512]; snprintf(desc, sizeof(desc), "pyramid-virtual:%4.4s,%d,%d," VX_FMT_SIZE ",%s", FORMAT_STR(format), width, height, levels, desc_scale);
        data = agoCreateDataFromDescription(graph->ref.context, graph, desc, true);
        if (data) {
            agoGenerateVirtualDataName(graph, "pyramid", data->name);
            agoAddData(&graph->dataList, data);
            // add the children too
            for (vx_uint32 i = 0; i < data->numChildren; i++) {
                agoAddData(&graph->dataList, data->children[i]);
                for (vx_uint32 j = 0; j < data->children[i]->numChildren; j++) {
                    if (data->children[i]->children[j]) {
                        agoAddData(&graph->dataList, data->children[i]->children[j]);
                    }
                }
            }
        }
    }
    return (vx_pyramid)data;
}


/*! \brief Releases a reference to a pyramid object.
* The object may not be garbage collected until its total reference count is zero.
* \param [in] pyr The pointer to the pyramid to release.
* \ingroup group_pyramid
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
* \post After returning from this function the reference is zeroed.
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleasePyramid(vx_pyramid *pyr)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (pyr && agoIsValidData((AgoData*)*pyr, VX_TYPE_PYRAMID)) {
        if (!agoReleaseData((AgoData*)*pyr, true)) {
            *pyr = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Queries an attribute from an image pyramid.
* \param [in] pyr The pyramid to query.
* \param [in] attribute The attribute for which to query. Use a <tt>\ref vx_pyramid_attribute_e</tt> enumeration.
* \param [out] ptr The location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_pyramid
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryPyramid(vx_pyramid pyr, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)pyr;
    if (agoIsValidData(data, VX_TYPE_PYRAMID)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_PYRAMID_ATTRIBUTE_LEVELS:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->u.pyr.levels;
                    status = VX_SUCCESS;
                }
                break;
            case VX_PYRAMID_ATTRIBUTE_SCALE:
                if (size == sizeof(vx_float32)) {
                    *(vx_float32 *)ptr = data->u.pyr.scale;
                    status = VX_SUCCESS;
                }
                break;
            case VX_PYRAMID_ATTRIBUTE_WIDTH:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = data->u.pyr.width;
                    status = VX_SUCCESS;
                }
                break;
            case VX_PYRAMID_ATTRIBUTE_HEIGHT:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32  *)ptr = data->u.pyr.height;
                    status = VX_SUCCESS;
                }
                break;
            case VX_PYRAMID_ATTRIBUTE_FORMAT:
                if (size == sizeof(vx_df_image)) {
                    *(vx_df_image *)ptr = data->u.pyr.format;
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Retrieves a level of the pyramid as a <tt>\ref vx_image</tt>, which can be used
* elsewhere in OpenVX.
* \param [in] pyr The pyramid object.
* \param [in] index The index of the level, such that index is less than levels.
* \return A <tt>\ref vx_image</tt> reference.
* \retval 0 Indicates that the index or the object is invalid.
* \ingroup group_pyramid
*/
VX_API_ENTRY vx_image VX_API_CALL vxGetPyramidLevel(vx_pyramid pyr, vx_uint32 index)
{
    AgoData * data = (AgoData *)pyr;
    AgoData * img = NULL;
    if (agoIsValidData(data, VX_TYPE_PYRAMID) && (index < data->u.pyr.levels) && !data->isNotFullyConfigured) {
        img = data->children[index];
        agoRetainData((AgoGraph *)data->ref.scope, img, true);
        data->ref.context->num_active_references++;
    }
    return (vx_image)img;
}

/*==============================================================================
REMAP
=============================================================================*/

/*! \brief Creates a remap table object.
* \param [in] context The reference to the overall context.
* \param [in] src_width Width of the source image in pixel.
* \param [in] src_height Height of the source image in pixels.
* \param [in] dst_width Width of the destination image in pixels.
* \param [in] dst_height Height of the destination image in pixels.
* \ingroup group_remap
* \return <tt>\ref vx_remap</tt>
* \retval 0 Object could not be created.
* \retval * Object was created.
*/
VX_API_ENTRY vx_remap VX_API_CALL vxCreateRemap(vx_context context,
    vx_uint32 src_width,
    vx_uint32 src_height,
    vx_uint32 dst_width,
    vx_uint32 dst_height)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context) && src_width > 0 && src_height > 0 && dst_width > 0 && dst_height > 0) {
        CAgoLock lock(context->cs);
        char desc[512]; snprintf(desc, sizeof(desc), "remap:%u,%u,%u,%u", src_width, src_height, dst_width, dst_height);
        data = agoCreateDataFromDescription(context, NULL, desc, true);
        if (data) {
            agoGenerateDataName(context, "remap", data->name);
            agoAddData(&context->dataList, data);
        }
    }
    return (vx_remap)data;
}

/*! \brief Creates an opaque reference to a remap table object without direct user access.
 * \param [in] graph 		The reference to the parent graph.
 * \param [in] src_width 	Width of the source image in pixel.
 * \param [in] src_height 	Height of the source image in pixels.
 * \param [in] dst_width 	Width of the destination image in pixels.
 * \param [in] dst_height 	Height of the destination image in pixels.
 * \see <tt>\ref vxCreateRemap</tt>
 * \ingroup group_remap
 * \returns A remap reference <tt>\ref vx_remap</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_remap VX_API_CALL vxCreateVirtualRemap(vx_graph graph,
                              vx_uint32 src_width,
                              vx_uint32 src_height,
                              vx_uint32 dst_width,
                              vx_uint32 dst_height)
{
    AgoData * data = NULL;
    if (agoIsValidGraph(graph) && src_width > 0 && src_height > 0 && dst_width > 0 && dst_height > 0) {
        CAgoLock lock(graph->cs);
        char desc[512];
        snprintf(desc, sizeof(desc), "remap-virtual:%u,%u,%u,%u", src_width, src_height, dst_width, dst_height);
        data = agoCreateDataFromDescription(graph->ref.context, graph, desc, true);
        if (data) {
            agoGenerateVirtualDataName(graph, "remap", data->name);
            agoAddData(&graph->dataList, data);
        }
    }
    return (vx_remap)data;
}

/*! \brief Allows the application to copy a rectangular patch from/into a remap object.
 *
 * The patch is specified within the destination dimensions and its
 * data provide the corresponding coordinate within the source dimensions.
 * The patch in user memory is a 2D array of elements of the type associated with the
 * \p coordinate_type parameter (i.e., <tt>\ref vx_coordinates2df_t</tt> for
 * <tt>\ref VX_TYPE_COORDINATES2DF</tt>).
 * The memory layout of this array follows a row-major order where rows are
 * compact (without any gap between elements), and where the potential padding
 * after each line is determined by the \p user_stride_y parameter.

 * \param [in] remap The reference to the remap object that is the source or the
 * destination of the patch copy.
 *
 * \param [in] rect The coordinates of remap patch. The patch must be specified
 * within the bounds of the remap destination dimensions
 * (<tt>\ref VX_REMAP_DESTINATION_WIDTH</tt> x <tt>\ref VX_REMAP_DESTINATION_HEIGHT</tt>).
 * (start_x, start_y) gives the coordinate of the topleft element inside the patch,
 * while (end_x, end_y) gives the coordinate of the bottomright element out of the patch.
 *
 * \param [in] user_stride_y The difference between the address of the first element
 * of two successive lines of the remap patch in user memory (pointed by
 * \p user_ptr). The layout of the user memory must follow a row major order and user_stride_y
 * must follow the following rule :
 *  user_stride_y >= sizeof(<ELEMENT_TYPE>) * (rect->end_x - rect->start_x).
 *
 * \param [in] user_ptr The address of the user memory location where to store the requested
 * remap data if the copy was requested in read mode, or from where to get the remap data to
 * store into the remap object if the copy was requested in write mode. \p user_ptr is the
 * address of the the top-left element of the remap patch.
 * The accessible user memory must be large enough to contain the specified patch with
 * the specified layout:
 * accessible memory in bytes >= (rect->end_y - rect->start_y) * user_stride_y.
 *
 * \param [in] user_coordinate_type This declares the type of the source coordinate remap
 * data in the user memory. It must be <tt>\ref VX_TYPE_COORDINATES2DF</tt>.
 *
 * \param [in] usage This declares the effect of the copy with regard to the remap object
 * using the <tt>\ref vx_accessor_e</tt> enumeration. Only VX_READ_ONLY and VX_WRITE_ONLY are
 * supported:
 * \arg <tt>\ref VX_READ_ONLY</tt> means that data is copied from the remap object into the user
 * memory pointer by \p user_ptr. The potential padding after each line in user
 * memory will stay unchanged.
 * \arg <tt>\ref VX_WRITE_ONLY</tt> means that data is copied into the remap object from
 * the user memory.
 *
 * \param [in] user_mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that specifies
 * the type of the memory pointer by \p user_ptr.
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 * \retval VX_ERROR_INVALID_REFERENCE remap is not a valid <tt>\ref vx_remap</tt> reference.
 * \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
 *
 * \ingroup group_remap
*/
VX_API_ENTRY vx_status VX_API_CALL vxCopyRemapPatch(vx_remap remap,
                                                    const vx_rectangle_t *rect,
                                                    vx_size user_stride_y,
                                                    void * user_ptr,
                                                    vx_enum user_coordinate_type,
                                                    vx_enum usage,
                                                    vx_enum user_mem_type)
{
    AgoData * data = (AgoData *)remap;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_REMAP))
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        vx_uint32 start_x = rect ? rect->start_x : 0u;
        vx_uint32 start_y = rect ? rect->start_y : 0u;
        vx_uint32 end_x = rect ? rect->end_x : 0u;
        vx_uint32 end_y = rect ? rect->end_y : 0u;
        vx_bool zero_area = ((((end_x - start_x) == 0) || ((end_y - start_y) == 0)) ? vx_true_e : vx_false_e);
        bool paramsValid = true;

        if(user_coordinate_type != VX_TYPE_COORDINATES2DF)
            paramsValid = false;
        if(user_stride_y < sizeof(vx_coordinates2df_t)*(rect->end_x - rect->start_x))
            paramsValid = false;
        if(user_mem_type != VX_MEMORY_TYPE_HOST && user_mem_type != VX_MEMORY_TYPE_NONE)
            paramsValid = false;
        if((usage != VX_READ_ONLY && usage != VX_WRITE_ONLY) || (rect == NULL) || (user_ptr == NULL))
            paramsValid = false;
        if (zero_area == vx_false_e && ((start_x >= end_x) || (start_y >= end_y)))
            paramsValid = false;

        if(paramsValid){
            status = VX_ERROR_NO_MEMORY;
            if (!data->buffer) {
                CAgoLock lock(data->ref.context->cs);
                if (agoAllocData(data)) {
                    return VX_FAILURE;
                }
            }
            vx_size stride = user_stride_y / sizeof(vx_coordinates2df_t);
            AgoData * dataToSync = data;
#if ENABLE_OPENCL
            if (dataToSync->opencl_buffer && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                // make sure dirty OpenCL buffers are synched before giving access for read
                if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                    // transfer only valid data
                    if (dataToSync->size > 0) {
                        cl_int err = clEnqueueReadBuffer(dataToSync->ref.context->opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->gpu_buffer_offset, dataToSync->size, dataToSync->buffer, 0, NULL, NULL);
                        if (err) {
                            status = VX_FAILURE;
                            agoAddLogEntry(&dataToSync->ref, status, "ERROR: vxCopyRemapPatch: clEnqueueReadBuffer() => %d\n", err);
                            return status;
                        }
                    }
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                }
            }
#elif ENABLE_HIP
            if (dataToSync->hip_memory  && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                // make sure dirty HIP buffers are synched before giving access for read
                if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                    // transfer only valid data
                    if (dataToSync->size > 0) {
                        hipError_t err = hipMemcpyDtoH((void *)dataToSync->buffer, dataToSync->hip_memory + dataToSync->gpu_buffer_offset, dataToSync->size);
                        if (err) {
                            status = VX_FAILURE;
                            agoAddLogEntry(&dataToSync->ref, status, "ERROR: vxCopyRemapPatch: hipMemcpyDtoH() => %d\n", err);
                            return status;
                        }
                    }
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                }
            }
#endif
            if (usage == VX_READ_ONLY) {
                vx_coordinates2df_t *ptr = (vx_coordinates2df_t*)user_ptr;
                vx_uint32 i;
                vx_uint32 j;
                for (i = start_y; i < end_y; i++)
                {
                    for (j = start_x; j < end_x; j++)
                    {
                        vx_coordinates2df_t *coord_ptr = &(ptr[i * stride + j]);
                        status = vxGetRemapPoint(remap, j, i, &coord_ptr->x, &coord_ptr->y);
                        if(status != VX_SUCCESS)
                        {
                            break;
                        }

                    }
                }
            }
            else {
                vx_coordinates2df_t *ptr = (vx_coordinates2df_t*)user_ptr;
                vx_uint32 i;
                vx_uint32 j;
                for (i = start_y; i < end_y; i++)
                {
                    for (j = start_x; j < end_x; j++)
                    {
                        vx_coordinates2df_t *coord_ptr = &(ptr[i * stride + j]);
                        status = vxSetRemapPoint(remap, j, i, coord_ptr->x, coord_ptr->y);
                        if(status != VX_SUCCESS)
                        {
                            break;
                        }

                    }
                }
            }
            // update sync flags
            dataToSync->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
            dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Allows the application to get direct access to a rectangular patch of a remap object.
 *
 * The patch is specified within the destination dimensions and its
 * data provide the corresponding coordinate within the source dimensions.
 * The patch is mapped as a 2D array of elements of the type associated
 * with the \p coordinate_type parameter (i.e., <tt>\ref vx_coordinates2df_t</tt>
 * for <tt>\ref VX_TYPE_COORDINATES2DF</tt>).
 * The memory layout of the mapped 2D array follows a row-major order where rows are
 * compact (without any gap between elements), and where the potential
 * padding after each lines is determined by (* \p stride_y).
 *
 * \param [in] remap The reference to the remap object that contains the
 * patch to map.
 *
 * \param [in] rect The coordinates of remap patch. The patch must be specified
 * within the bounds of the remap destination dimensions
 * (<tt>\ref VX_REMAP_DESTINATION_WIDTH</tt> x <tt>\ref VX_REMAP_DESTINATION_HEIGHT</tt>).
 * (start_x, start_y) gives the coordinate of the topleft element inside the patch,
 * while (end_x, end_y) gives the coordinate of the bottomright element out of the patch.
 *
 * \param [out] map_id The address of a <tt>\ref vx_map_id</tt> variable
 * where the function returns a map identifier.
 * \arg (*map_id) must eventually be provided as the map_id parameter of a call
 * to <tt>\ref vxUnmapRemapPatch</tt>.
 *
 * \param [out] stride_y The address of a vx_size variable where the function
 * returns the difference between the address of the first element of two
 * successive lines in the mapped remap patch. The stride value follows the
 * following rule :
 * (*stride_y) >= sizeof(<ELEMENT_TYPE>) * (rect->end_x - rect->start_x)
 *
 * \param [out] ptr The address of a pointer where the function returns where
 * remap patch data can be accessed. (*ptr) is the address of the the top-left
 * element of the remap patch.
 * The returned (*ptr) address is only valid between the call to this function
 * and the corresponding call to <tt>\ref vxUnmapRemapPatch</tt>.
 *
 * \param [in] coordinate_type This declares the type of the source coordinate
 * data that the application wants to access in the remap patch.
 * It must be <tt>\ref VX_TYPE_COORDINATES2DF</tt>.
 *
 * \param [in] usage This declares the access mode for the remap patch, using
 * the <tt>\ref vx_accessor_e</tt> enumeration.
 * \arg <tt>\ref VX_READ_ONLY</tt>: after the function call, the content of the
 * memory location pointed by (*ptr) contains the remap patch data. Writing into
 * this memory location is forbidden and its behavior is undefined.
 * \arg <tt>\ref VX_READ_AND_WRITE</tt>: after the function call, the content of
 * the memory location pointed by (*ptr) contains the remap patch data; writing
 * into this memory is allowed for the location of elements only and will
 * result in a modification of the written elements in the remap object once the
 * patch is unmapped. Writing into a gap between element lines
 * (when (*stride_y) > sizeof(<ELEMENT_TYPE>) * (rect->end_x - rect->start_x))
 * is forbidden and its behavior is undefined.
 * \arg <tt>\ref VX_WRITE_ONLY</tt>: after the function call, the memory location
 * pointed by (*ptr) contains undefined data; writing each element of the patch is
 * required prior to unmapping. Elements not written by the application before
 * unmap will become undefined after unmap, even if they were well defined before
 * map. Like for <tt>\ref VX_READ_AND_WRITE</tt>, writing into a gap between
 * element lines is forbidden and its behavior is undefined.
 *
 * \param [in] mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that
 * specifies the type of the memory where the remap patch is requested to be mapped.
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 * \retval VX_ERROR_INVALID_REFERENCE remap is not a valid <tt>\ref vx_remap</tt> reference.
 * \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
 *
 * \ingroup group_remap
 * \post <tt>\ref vxUnmapRemapPatch </tt> with same (*map_id) value.
 */
VX_API_ENTRY vx_status VX_API_CALL vxMapRemapPatch(vx_remap remap,
                                                   const vx_rectangle_t *rect,
                                                   vx_map_id *map_id,
                                                   vx_size *stride_y,
                                                   void **ptr,
                                                   vx_enum coordinate_type,
                                                   vx_enum usage,
                                                   vx_enum mem_type)
{
    AgoData * data = (AgoData *)remap;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_REMAP))
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        vx_uint32 start_x = rect ? rect->start_x : 0u;
        vx_uint32 start_y = rect ? rect->start_y : 0u;
        vx_uint32 end_x = rect ? rect->end_x : 0u;
        vx_uint32 end_y = rect ? rect->end_y : 0u;
        vx_bool zero_area = ((((end_x - start_x) == 0) || ((end_y - start_y) == 0)) ? vx_true_e : vx_false_e);
        bool paramsValid = true;

        if(coordinate_type != VX_TYPE_COORDINATES2DF)
            paramsValid = false;
        if(mem_type != VX_MEMORY_TYPE_HOST && mem_type != VX_MEMORY_TYPE_NONE)
            paramsValid = false;
        if((usage != VX_READ_ONLY && usage != VX_WRITE_ONLY) || (rect == NULL) || (ptr == NULL) || (map_id == NULL))
            paramsValid = false;
        if (zero_area == vx_false_e && ((start_x >= end_x) || (start_y >= end_y)))
            paramsValid = false;


        if (paramsValid) {
            if (!data->buffer) {
                CAgoLock lock(data->ref.context->cs);
                if (agoAllocData(data)) {
                    return VX_FAILURE;
                }
            }
#if ENABLE_OPENCL
            vx_enum mem_type_requested = mem_type;
            if (mem_type == VX_MEMORY_TYPE_OPENCL)
            {
                mem_type = VX_MEMORY_TYPE_HOST;
            }
#endif

            vx_size stride = (end_x - start_x);
            vx_size size = (stride * (end_y - start_y)) * sizeof(vx_coordinates2df_t);
            ago_coord2d_float_t* ptr_returned = (ago_coord2d_float_t *)data->reserved;

            // save the pointer and usage for use in vxUnmapRemapPatch
            status = VX_SUCCESS;
            for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
                if (i->ptr == ptr_returned) {
                    // can't support vxMapRemapPatch() more than once with same pointer
                    // the application needs to call vxUnmapTensorPatch() before calling vxMapRemapPatch()
                    status = VX_FAILURE;
                }
            }
            if (status == VX_SUCCESS) {
                AgoData * dataToSync = data;
#if ENABLE_OPENCL
                if (dataToSync->opencl_buffer && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                    // make sure dirty OpenCL buffers are synched before giving access for read
                    if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                        // transfer only valid data
                        if (dataToSync->size > 0) {
                            cl_int err = clEnqueueReadBuffer(dataToSync->ref.context->opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->gpu_buffer_offset, dataToSync->size, dataToSync->buffer, 0, NULL, NULL);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&dataToSync->ref, status, "ERROR: vxMapRemapPatch: clEnqueueReadBuffer() => %d\n", err);
                                return status;
                            }
                        }
                        dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                    }
                }
#elif ENABLE_HIP
                if (dataToSync->hip_memory  && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                    // make sure dirty HIP buffers are synched before giving access for read
                    if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                        // transfer only valid data
                        if (dataToSync->size > 0) {
                            hipError_t err = hipMemcpyDtoH((void *)dataToSync->buffer, dataToSync->hip_memory + dataToSync->gpu_buffer_offset, dataToSync->size);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&dataToSync->ref, status, "ERROR: vxMapRemapPatch: hipMemcpyDtoH() => %d\n", err);
                                return status;
                            }
                        }
                        dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                    }
                }
#endif
                MappedData item = { data->nextMapId++, ptr_returned, usage, false };
                data->mapped.push_back(item);
                *map_id = item.map_id;
                *ptr = ptr_returned;
                *stride_y = stride * sizeof(vx_coordinates2df_t);
            }
        }
    }
    return status;
}

/*! \brief Unmap and commit potential changes to a remap object patch that was previously mapped.
 *
 * Unmapping a remap patch invalidates the memory location from which the patch could
 * be accessed by the application. Accessing this memory location after the unmap function
 * completes has an undefined behavior.
 * \param [in] remap The reference to the remap object to unmap.
 * \param [out] map_id The unique map identifier that was returned by <tt>\ref vxMapRemapPatch</tt> .
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 * \retval VX_ERROR_INVALID_REFERENCE remap is not a valid <tt>\ref vx_remap</tt> reference.
 * \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
 * \ingroup group_remap
 * \pre <tt>\ref vxMapRemapPatch</tt> with same map_id value
*/
VX_API_ENTRY vx_status VX_API_CALL vxUnmapRemapPatch(vx_remap remap, vx_map_id map_id)
{
    AgoData * data = (AgoData *)remap;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_REMAP)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
            if (i->map_id == map_id) {
                vx_enum usage = i->usage;
                data->mapped.erase(i);
                if (usage == VX_WRITE_ONLY || usage == VX_READ_AND_WRITE) {
                    // update sync flags
                    AgoData * dataToSync = data;
                    dataToSync->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
                }
                status = VX_SUCCESS;
                break;
            }
        }
    }
    return status;
}

/*! \brief Releases a reference to a remap table object. The object may not be
* garbage collected until its total reference count is zero.
* \param [in] table The pointer to the remap table to release.
* \post After returning from this function the reference is zeroed.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
* \ingroup group_remap
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseRemap(vx_remap *table)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (table && agoIsValidData((AgoData*)*table, VX_TYPE_REMAP)) {
        if (!agoReleaseData((AgoData*)*table, true)) {
            *table = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Assigns a destination pixel mapping to the source pixel.
* \param [in] table The remap table reference.
* \param [in] dst_x The destination x coordinate.
* \param [in] dst_y The destination y coordinate.
* \param [in] src_x The source x coordinate in float representation to allow interpolation.
* \param [in] src_y The source y coordinate in float representation to allow interpolation.
* \ingroup group_remap
* \return A <tt>\ref vx_status_e</tt> enumeration.
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetRemapPoint(vx_remap table,
    vx_uint32 dst_x, vx_uint32 dst_y,
    vx_float32 src_x, vx_float32 src_y)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)table;
    if (agoIsValidData(data, VX_TYPE_REMAP)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (!data->buffer) {
            CAgoLock lock(data->ref.context->cs);
            if (agoAllocData(data)) {
                return VX_FAILURE;
            }
        }
        if (dst_x < data->u.remap.dst_width && dst_y < data->u.remap.dst_height && data->buffer && data->reserved) {
            ago_coord2d_ushort_t * item_fixed = ((ago_coord2d_ushort_t *)data->buffer) + (dst_y * data->u.remap.dst_width) + dst_x;
            ago_coord2d_float_t * item_float = ((ago_coord2d_float_t *)data->reserved) + (dst_y * data->u.remap.dst_width) + dst_x;
            item_float->x = src_x;
            item_float->y = src_y;
            item_fixed->x = (vx_uint16)(src_x * (vx_float32)(1 << data->u.remap.remap_fractional_bits) + 0.5f); // convert to fixed-point with rounding
            item_fixed->y = (vx_uint16)(src_y * (vx_float32)(1 << data->u.remap.remap_fractional_bits) + 0.5f); // convert to fixed-point with rounding
            // special handing for border cases
            if (src_x < 0.0f || src_y < 0.0f || src_x >= (vx_float32)(data->u.remap.src_width-1) || src_y >= (vx_float32)(data->u.remap.src_height-1)) {
                item_fixed->x = 0xffff;
                item_fixed->y = 0xffff;
            }
            status = VX_SUCCESS;
            // update sync flags
            data->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
            data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
        }
    }
    return status;
}

/*! \brief Retrieves the source pixel point from a destination pixel.
* \param [in] table The remap table reference.
* \param [in] dst_x The destination x coordinate.
* \param [in] dst_y The destination y coordinate.
* \param [out] src_x The pointer to the location to store the source x coordinate in float representation to allow interpolation.
* \param [out] src_y The pointer to the location to store the source y coordinate in float representation to allow interpolation.
* \ingroup group_remap
* \return A <tt>\ref vx_status_e</tt> enumeration.
*/
VX_API_ENTRY vx_status VX_API_CALL vxGetRemapPoint(vx_remap table,
    vx_uint32 dst_x, vx_uint32 dst_y,
    vx_float32 *src_x, vx_float32 *src_y)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)table;
    if (agoIsValidData(data, VX_TYPE_REMAP) && data->buffer && data->reserved) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (src_x && src_y && dst_x < data->u.remap.dst_width && dst_y < data->u.remap.dst_height) {
            ago_coord2d_float_t * item = ((ago_coord2d_float_t *)data->reserved) + (dst_y * data->u.remap.dst_width) + dst_x;
            *src_x = item->x;
            *src_y = item->y;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Queries attributes from a Remap table.
* \param [in] r The remap to query.
* \param [in] attribute The attribute to query. Use a <tt>\ref vx_remap_attribute_e</tt> enumeration.
* \param [out] ptr The location at which to store the resulting value.
* \param [in] size The size of the container to which \a ptr points.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_remap
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryRemap(vx_remap r, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)r;
    if (agoIsValidData(data, VX_TYPE_REMAP)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_REMAP_ATTRIBUTE_SOURCE_WIDTH:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = data->u.remap.src_width;
                    status = VX_SUCCESS;
                }
                break;
            case VX_REMAP_ATTRIBUTE_SOURCE_HEIGHT:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = data->u.remap.src_height;
                    status = VX_SUCCESS;
                }
                break;
            case VX_REMAP_ATTRIBUTE_DESTINATION_WIDTH:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32 *)ptr = data->u.remap.dst_width;
                    status = VX_SUCCESS;
                }
                break;
            case VX_REMAP_ATTRIBUTE_DESTINATION_HEIGHT:
                if (size == sizeof(vx_uint32)) {
                    *(vx_uint32  *)ptr = data->u.remap.dst_height;
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*==============================================================================
ARRAY
=============================================================================*/

/*!
* \brief Creates a reference to an Array object.
*
* User must specify the Array capacity (i.e., the maximal number of items that the array can hold).
*
* \param [in] context      The reference to the overall Context.
* \param [in] item_type    The type of objects to hold. Use:
*                          \arg <tt>\ref VX_TYPE_RECTANGLE</tt> for <tt>\ref vx_rectangle_t</tt>.
*                          \arg <tt>\ref VX_TYPE_KEYPOINT</tt> for <tt>\ref vx_keypoint_t</tt>.
*                          \arg <tt>\ref VX_TYPE_COORDINATES2D</tt> for <tt>\ref vx_coordinates2d_t</tt>.
*                          \arg <tt>\ref VX_TYPE_COORDINATES3D</tt> for <tt>\ref vx_coordinates3d_t</tt>.
*                          \arg <tt>\ref vx_enum</tt> Returned from <tt>\ref vxRegisterUserStruct</tt>.
* \param [in] capacity     The maximal number of items that the array can hold.
*
* \return <tt>\ref vx_array</tt>.
* \retval 0 No Array was created.
* \retval * An Array was created.
*
* \ingroup group_array
*/
VX_API_ENTRY vx_array VX_API_CALL vxCreateArray(vx_context context, vx_enum item_type, vx_size capacity)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context) && capacity > 0) {
        CAgoLock lock(context->cs);
        const char * desc_type = agoEnum2Name(item_type);
        if (!desc_type) {
            desc_type = agoGetUserStructName(context, item_type);
        }
        if (desc_type) {
            char desc[512]; snprintf(desc, sizeof(desc), "array:%s," VX_FMT_SIZE "", desc_type, capacity);
            data = agoCreateDataFromDescription(context, NULL, desc, true);
            if (data) {
                agoGenerateDataName(context, "array", data->name);
                agoAddData(&context->dataList, data);
            }
        }
    }
    return (vx_array)data;
}

/*!
* \brief Creates an opaque reference to a virtual Array with no direct user access.
*
* Virtual Arrays are useful when item type or capacity are unknown ahead of time
* and the Array is used as internal graph edge. Virtual arrays are scoped within the parent graph only.
*
* All of the following constructions are allowed.
* \code
* vx_context context = vxCreateContext();
* vx_graph graph = vxCreateGraph(context);
* vx_array virt[] = {
*     vxCreateVirtualArray(graph, 0, 0), // totally unspecified
*     vxCreateVirtualArray(graph, VX_TYPE_KEYPOINT, 0), // unspecified capacity
*     vxCreateVirtualArray(graph, VX_TYPE_KEYPOINT, 1000), // no access
* };
* \endcode
*
* \param [in] graph        The reference to the parent graph.
* \param [in] item_type    The type of objects to hold.
*                          This may to set to zero to indicate an unspecified item type.
* \param [in] capacity     The maximal number of items that the array can hold.
*                          This may be to set to zero to indicate an unspecified capacity.
* \see vxCreateArray for a type list.
* \return <tt>\ref vx_array</tt>.
* \retval 0 No Array was created.
* \retval * An Array was created or an error occurred. Use <tt>\ref vxGetStatus</tt> to determine.
*
* \ingroup group_array
*/
VX_API_ENTRY vx_array VX_API_CALL vxCreateVirtualArray(vx_graph graph, vx_enum item_type, vx_size capacity)
{
    AgoData * data = NULL;
    if (agoIsValidGraph(graph)) {
        CAgoLock lock(graph->cs);
        const char * desc_type = agoEnum2Name(item_type);
        if (item_type && !desc_type) {
            desc_type = agoGetUserStructName(graph->ref.context, item_type);
        }
        if (!item_type || desc_type) {
            char desc[512];
            if (desc_type) snprintf(desc, sizeof(desc), "array-virtual:%s," VX_FMT_SIZE "", desc_type, capacity);
            else snprintf(desc, sizeof(desc), "array-virtual:0," VX_FMT_SIZE "", capacity);
            data = agoCreateDataFromDescription(graph->ref.context, graph, desc, true);
            if (data) {
                agoGenerateVirtualDataName(graph, "array", data->name);
                agoAddData(&graph->dataList, data);
            }
        }
    }
    return (vx_array)data;
}

/*!
* \brief Releases a reference of an Array object.
* The object may not be garbage collected until its total reference count is zero.
* After returning from this function the reference is zeroed.
* \param [in] arr          The pointer to the Array to release.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_ERROR_INVALID_REFERENCE If graph is not a <tt>\ref vx_graph</tt>.
* \ingroup group_array
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseArray(vx_array *arr)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (arr && agoIsValidData((AgoData*)*arr, VX_TYPE_ARRAY)) {
        if (!agoReleaseData((AgoData*)*arr, true)) {
            *arr = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*!
* \brief Queries the Array for some specific information.
*
* \param [in] arr          The reference to the Array.
* \param [in] attribute    The attribute to query. Use a <tt>\ref vx_array_attribute_e</tt>.
* \param [out] ptr         The location at which to store the resulting value.
* \param [in] size         The size of the container to which \a ptr points.
*
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS                   No errors.
* \retval VX_ERROR_INVALID_REFERENCE   If the \a arr is not a <tt>\ref vx_array</tt>.
* \retval VX_ERROR_NOT_SUPPORTED       If the \a attribute is not a value supported on this implementation.
* \retval VX_ERROR_INVALID_PARAMETERS  If any of the other parameters are incorrect.
*
* \ingroup group_array
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryArray(vx_array arr, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)arr;
    if (agoIsValidData(data, VX_TYPE_ARRAY)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_ARRAY_ATTRIBUTE_ITEMTYPE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = data->u.arr.itemtype;
                    status = VX_SUCCESS;
                }
                break;
            case VX_ARRAY_ATTRIBUTE_NUMITEMS:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->u.arr.numitems;
                    status = VX_SUCCESS;
                }
                break;
            case VX_ARRAY_ATTRIBUTE_CAPACITY:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->u.arr.capacity;
                    status = VX_SUCCESS;
                }
                break;
            case VX_ARRAY_ATTRIBUTE_ITEMSIZE:
                if (size == sizeof(vx_size)) {
                    *(vx_size  *)ptr = data->u.arr.itemsize;
                    status = VX_SUCCESS;
                }
                break;
#if (ENABLE_OPENCL||ENABLE_HIP)
            case VX_ARRAY_OFFSET_GPU:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->gpu_buffer_offset;
                    status = VX_SUCCESS;
                }
                break;
#if ENABLE_OPENCL
            case VX_ARRAY_BUFFER_OPENCL:
                if (size == sizeof(cl_mem)) {
                    if (data->opencl_buffer) {
                        *(cl_mem *)ptr = data->opencl_buffer;
                    }
                    else {
#if defined(CL_VERSION_2_0)
                        *(vx_uint8 **)ptr = data->opencl_svm_buffer;
#else
                        *(vx_uint8 **)ptr = NULL;
#endif
                    }
                    status = VX_SUCCESS;
                }
                break;
#else
            case VX_ARRAY_BUFFER_HIP:
                if (size == sizeof(vx_uint8 *)) {
                    if (data->hip_memory) {
                        *(vx_uint8 **)ptr = data->hip_memory;
                    }
                    else {
                        *(vx_uint8 **)ptr = NULL;
                    }
                    status = VX_SUCCESS;
                }
                break;

#endif
#endif
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*!
* \brief Adds items to the Array.
*
* This function increases the container size.
*
* By default, the function does not reallocate memory,
* so if the container is already full (number of elements is equal to capacity)
* or it doesn't have enough space,
* the function returns <tt>\ref VX_FAILURE</tt> error code.
*
* \param [in] arr          The reference to the Array.
* \param [in] count        The total number of elements to insert.
* \param [in] ptr          The location at which to store the input values.
* \param [in] stride       The stride in bytes between elements. User can pass 0, which means that stride is equal to item size.
*
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS                   No errors.
* \retval VX_ERROR_INVALID_REFERENCE   If the \a arr is not a <tt>\ref vx_array</tt>.
* \retval VX_FAILURE                   If the Array is full.
* \retval VX_ERROR_INVALID_PARAMETERS  If any of the other parameters are incorrect.
*
* \ingroup group_array
*/
VX_API_ENTRY vx_status VX_API_CALL vxAddArrayItems(vx_array arr, vx_size count, const void *ptr, vx_size stride)
{
    AgoData * data = (AgoData *)arr;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_ARRAY)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if (ptr && (data->u.arr.numitems + count <= data->u.arr.capacity)) {
            if (!data->buffer) {
                CAgoLock lock(data->ref.context->cs);
                if (agoAllocData(data)) {
                    return VX_FAILURE;
                }
            }
            if (count > 0) {
#if ENABLE_OPENCL
                if (data->opencl_buffer && !(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                    // make sure dirty OpenCL buffers are synched before giving access for read
                    if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                        // transfer only valid data
                        vx_size size = data->u.arr.itemsize * data->u.arr.numitems;
                        if (size > 0) {
                            cl_int err = clEnqueueReadBuffer(data->ref.context->opencl_cmdq, data->opencl_buffer, CL_TRUE, data->gpu_buffer_offset, size, data->buffer, 0, NULL, NULL);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&data->ref, status, "ERROR: vxAccessArrayRange: clEnqueueReadBuffer() => %d\n", err);
                                return status;
                            }
                        }
                        data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                    }
                }
#elif ENABLE_HIP
                if (data->hip_memory && !(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                    // make sure dirty HIP buffers are synched before giving access for read
                    if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                        // transfer only valid data
                        vx_size size = data->u.arr.itemsize * data->u.arr.numitems;
                        if (size > 0) {
                            hipError_t err = hipMemcpyDtoH((void *)data->buffer, data->hip_memory + data->gpu_buffer_offset, data->size);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&data->ref, status, "ERROR: vxAccessArrayRange: hipMemcpyDtoH() => %d\n", err);
                                return status;
                            }
                        }
                        data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                    }
                }
#endif
                // add items at the end of the array
                vx_uint8 * pSrc = (vx_uint8 *)ptr;
                vx_uint8 * pDst = data->buffer + data->u.arr.itemsize * data->u.arr.numitems;
                if (stride == data->u.arr.itemsize) {
                    HafCpu_BinaryCopy_U8_U8(data->u.arr.itemsize * count, pDst, pSrc);
                }
                else {
                    for (vx_size i = 0; i < count; i++, pSrc += stride, pDst += data->u.arr.itemsize) {
                        HafCpu_BinaryCopy_U8_U8(data->u.arr.itemsize, pDst, pSrc);
                    }
                }
                data->u.arr.numitems += count;
                // update sync flags
                data->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
            }
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*!
* \brief Truncates an Array (remove items from the end).
*
* \param [in,out] arr          The reference to the Array.
* \param [in] new_num_items    The new number of items for the Array.
*
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS                   No errors.
* \retval VX_ERROR_INVALID_REFERENCE   If the \a arr is not a <tt>\ref vx_array</tt>.
* \retval VX_ERROR_INVALID_PARAMETERS  The \a new_size is greater than the current size.
*
* \ingroup group_array
*/
VX_API_ENTRY vx_status VX_API_CALL vxTruncateArray(vx_array arr, vx_size new_num_items)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)arr;
    if (agoIsValidData(data, VX_TYPE_ARRAY)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (new_num_items <= data->u.arr.numitems) {
            data->u.arr.numitems = new_num_items;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*!
* \brief Grants access to a sub-range of an Array.
*
* \param [in] arr          The reference to the Array.
* \param [in] start        The start index.
* \param [in] end          The end index.
* \param [out] stride      The stride in bytes between elements.
* \param [out] ptr         The user-supplied pointer to a pointer, via which the requested contents are returned.
*                          If (*ptr) is non-NULL, data is copied to it, else (*ptr) is set to the address of existing internal memory, allocated, or mapped memory.
*                          (*ptr) must be given to <tt>\ref vxCommitArrayRange</tt>.
*                          Use a <tt>\ref vx_rectangle_t</tt> for <tt>\ref VX_TYPE_RECTANGLE</tt>
*                          and a <tt>\ref vx_keypoint_t</tt> for <tt>\ref VX_TYPE_KEYPOINT</tt>.
* \param [in] usage        This declares the intended usage of the pointer using the <tt>\ref vx_accessor_e</tt> enumeration.
*
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS                   No errors.
* \retval VX_ERROR_OPTIMIZED_AWAY      If the reference is a virtual array and cannot be accessed or committed.
* \retval VX_ERROR_INVALID_REFERENCE   If the \a arr is not a <tt>\ref vx_array</tt>.
* \retval VX_ERROR_INVALID_PARAMETERS  If any of the other parameters are incorrect.
* \post <tt>\ref vxCommitArrayRange</tt>
* \ingroup group_array
*/
VX_API_ENTRY vx_status VX_API_CALL vxAccessArrayRange(vx_array arr, vx_size start, vx_size end, vx_size *stride, void **ptr, vx_enum usage)
{
    AgoData * data = (AgoData *)arr;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_ARRAY)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if (ptr && stride && start < end && end <= data->u.arr.numitems) {
            if (!data->buffer) {
                CAgoLock lock(data->ref.context->cs);
                if (agoAllocData(data)) {
                    return VX_FAILURE;
                }
            }
            vx_uint8 * ptr_internal = data->buffer + data->u.arr.itemsize * start;
            vx_uint8 * ptr_returned = *ptr ? (vx_uint8 *)*ptr : ptr_internal;
            // save the pointer and usage for use in vxCommitXXX
            status = VX_SUCCESS;
            for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
                if (i->ptr == ptr_returned) {
                    // can't support vxAccessXXX() more than once with same pointer
                    // the application needs to call vxCommitXXX() before calling vxAccessXXX()
                    status = VX_FAILURE;
                }
            }
            if (status == VX_SUCCESS) {
                MappedData item = { data->nextMapId++, ptr_returned, usage, (ptr_returned != ptr_internal) ? true : false, (ptr_returned != ptr_internal) ? *stride : data->u.arr.itemsize };
                data->mapped.push_back(item);
                *ptr = ptr_returned;
                *stride = item.stride;
#if ENABLE_OPENCL
                if (data->opencl_buffer && !(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                    // make sure dirty OpenCL buffers are synched before giving access for read
                    if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                        // transfer only valid data
                        vx_size size = data->u.arr.itemsize * data->u.arr.numitems;
                        if (size > 0) {
                            cl_int err = clEnqueueReadBuffer(data->ref.context->opencl_cmdq, data->opencl_buffer, CL_TRUE, data->gpu_buffer_offset, size, data->buffer, 0, NULL, NULL);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&data->ref, status, "ERROR: vxAccessArrayRange: clEnqueueReadBuffer() => %d\n", err);
                                return status;
                            }
                        }
                        data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                    }
                }
#elif ENABLE_HIP
                if (data->hip_memory && !(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                    // make sure dirty GPU buffers are synched before giving access for read
                    if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                        // transfer only valid data
                        vx_size size = data->u.arr.itemsize * data->u.arr.numitems;
                        if (size > 0) {
                            hipError_t err = hipMemcpyDtoH((void *)data->buffer, (data->hip_memory + data->gpu_buffer_offset), size);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&data->ref, status, "ERROR: vxAccessArrayRange: hipMemcpyDtoH() => %d\n", err);
                                return status;
                            }
                        }
                    data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                   }
                }
#endif
                if (item.used_external_ptr && (usage == VX_READ_ONLY || usage == VX_READ_AND_WRITE)) {
                    // copy if read is requested with explicit external buffer
                    vx_uint8 * pSrc = ptr_internal;
                    vx_uint8 * pDst = ptr_returned;
                    if (item.stride == data->u.arr.itemsize) {
                        HafCpu_BinaryCopy_U8_U8(data->u.arr.itemsize * (end - start), ptr_returned, ptr_internal);
                    }
                    else {
                        for (vx_size i = start; i < end; i++, pSrc += data->u.arr.itemsize, pDst += item.stride) {
                            HafCpu_BinaryCopy_U8_U8(data->u.arr.itemsize, pDst, pSrc);
                        }
                    }
                }
            }
        }
    }
    return status;
}

/*!
* \brief Commits data back to the Array object.
*
* \details This allows a user to commit data to a sub-range of an Array.
*
* \param [in] arr          The reference to the Array.
* \param [in] start        The start index.
* \param [in] end          The end index.
* \param [in] ptr          The user supplied pointer.
*
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS                   No errors.
* \retval VX_ERROR_OPTIMIZED_AWAY      If the reference is a virtual array and cannot be accessed or committed.
* \retval VX_ERROR_INVALID_REFERENCE   If the \a arr is not a <tt>\ref vx_array</tt>.
* \retval VX_ERROR_INVALID_PARAMETERS  If any of the other parameters are incorrect.
*
* \ingroup group_array
*/
VX_API_ENTRY vx_status VX_API_CALL vxCommitArrayRange(vx_array arr, vx_size start, vx_size end, const void *ptr)
{
    AgoData * data = (AgoData *)arr;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_ARRAY)) {
        // check for valid arguments
        status = VX_ERROR_INVALID_PARAMETERS;
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if (ptr && start <= end && end <= data->u.arr.numitems) {
            status = VX_SUCCESS;
            if (!data->buffer) {
                status = VX_FAILURE;
            }
            else if (!data->mapped.empty()) {
                vx_enum usage = VX_READ_ONLY;
                bool used_external_ptr = false;
                vx_size stride = data->u.arr.itemsize;
                for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
                    if (i->ptr == ptr) {
                        if (start < end) {
                            usage = i->usage;
                            used_external_ptr = i->used_external_ptr;
                            stride = i->stride;
                        }
                        data->mapped.erase(i);
                        break;
                    }
                }
                if ((start < end) && (usage == VX_WRITE_ONLY || usage == VX_READ_AND_WRITE)) {
                    if (used_external_ptr) {
                        // copy from external buffer
                        vx_uint8 * pSrc = (vx_uint8 *)ptr;
                        vx_uint8 * pDst = data->buffer + start * data->u.arr.itemsize;
                        if (stride == data->u.arr.itemsize) {
                            HafCpu_BinaryCopy_U8_U8(data->u.arr.itemsize * (end - start), pDst, pSrc);
                        }
                        else {
                            for (vx_size i = start; i < end; i++, pSrc += stride, pDst += data->u.arr.itemsize) {
                                HafCpu_BinaryCopy_U8_U8(data->u.arr.itemsize, pDst, pSrc);
                            }
                        }
                    }
                    // update sync flags
                    data->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                    data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
                }
            }
        }
    }
    return status;
}

/*! \brief Allows the application to copy a range from/into an array object.
* \param [in] array The reference to the array object that is the source or the
* destination of the copy.
* \param [in] range_start The index of the first item of the array object to copy.
* \param [in] range_end The index of the item following the last item of the
* array object to copy. (range_end range_start) items are copied from index
* range_start included. The range must be within the bounds of the array:
* 0 <= range_start < range_end <= number of items in the array.
* \param [in] user_stride The number of bytes between the beginning of two consecutive
* items in the user memory pointed by user_ptr. The layout of the user memory must
* follow an item major order:
* user_stride >= element size in bytes.
* \param [in] user_ptr The address of the memory location where to store the requested data
* if the copy was requested in read mode, or from where to get the data to store into the array
* object if the copy was requested in write mode. The accessible memory must be large enough
* to contain the specified range with the specified stride:
* accessible memory in bytes >= (range_end range_start) * user_stride.
* \param [in] usage This declares the effect of the copy with regard to the array object
* using the <tt>\ref vx_accessor_e</tt> enumeration. Only VX_READ_ONLY and VX_WRITE_ONLY
* are supported:
* \arg VX_READ_ONLY means that data are copied from the array object into the user memory.
* \arg VX_WRITE_ONLY means that data are copied into the array object from the user memory.
* \param [in] user_mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that specifies
* the memory type of the memory referenced by the user_addr.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_OPTIMIZED_AWAY This is a reference to a virtual array that cannot be
* accessed by the application.
* \retval VX_ERROR_INVALID_REFERENCE The array reference is not actually an array reference.
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \ingroup group_array
*/
VX_API_ENTRY vx_status VX_API_CALL vxCopyArrayRange(vx_array array, vx_size range_start, vx_size range_end, vx_size user_stride, void *user_ptr, vx_enum usage, vx_enum user_mem_type)
{
    AgoData * data = (AgoData *)array;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_ARRAY))
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        if ((user_mem_type == VX_MEMORY_TYPE_HOST) && user_ptr && (usage == VX_READ_ONLY || usage == VX_WRITE_ONLY)) {
            status = vxAccessArrayRange(array, range_start, range_end, &user_stride, &user_ptr, usage);
            if (status == VX_SUCCESS) {
                status = vxCommitArrayRange(array, range_start, range_end, user_ptr);
            }
        }
    }
    return status;
}

/*! \brief Allows the application to get direct access to a range of an array object.
* \param [in] array The reference to the array object that contains the range to map.
* \param [in] range_start The index of the first item of the array object to map.
* \param [in] range_end The index of the item following the last item of the
* array object to map. (range_end range_start) items are mapped, starting from index
* range_start included. The range must be within the bounds of the array:
* Must be 0 <= range_start < range_end <= number of items.
* \param [out] map_id The address of a vx_map_id variable where the function
* returns a map identifier.
* \arg (*map_id) must eventually be provided as the map_id parameter of a call to
* <tt>\ref vxUnmapArrayRange</tt>.
* \param [out] stride The address of a vx_size variable where the function
* returns the memory layout of the mapped array range. The function sets (*stride)
* to the number of bytes between the beginning of two consecutive items.
* The application must consult (*stride) to access the array items starting from
* address (*ptr). The layout of the mapped array follows an item major order:
* (*stride) >= item size in bytes.
* \param [out] ptr The address of a pointer that the function sets to the
* address where the requested data can be accessed. The returned (*ptr) address
* is only valid between the call to the function and the corresponding call to
* <tt>\ref vxUnmapArrayRange</tt>.
* \param [in] usage This declares the access mode for the array range, using
* the <tt>\ref vx_accessor_e</tt> enumeration.
* \arg VX_READ_ONLY: after the function call, the content of the memory location
* pointed by (*ptr) contains the array range data. Writing into this memory location
* is forbidden and its behavior is undefined.
* \arg VX_READ_AND_WRITE : after the function call, the content of the memory
* location pointed by (*ptr) contains the array range data; writing into this memory
* is allowed only for the location of items and will result in a modification of the
* affected items in the array object once the range is unmapped. Writing into
* a gap between items (when (*stride) > item size in bytes) is forbidden and its
* behavior is undefined.
* \arg VX_WRITE_ONLY: after the function call, the memory location pointed by (*ptr)
* contains undefined data; writing each item of the range is required prior to
* unmapping. Items not written by the application before unmap will become
* undefined after unmap, even if they were well defined before map. Like for
* VX_READ_AND_WRITE, writing into a gap between items is forbidden and its behavior
* is undefined.
* \param [in] mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that
* specifies the type of the memory where the array range is requested to be mapped.
* \param [in] flags An integer that allows passing options to the map operation.
* Use the <tt>\ref vx_map_flag_e</tt> enumeration.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_OPTIMIZED_AWAY This is a reference to a virtual array that cannot be
* accessed by the application.
* \retval VX_ERROR_INVALID_REFERENCE The array reference is not actually an array
* reference.
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \ingroup group_array
* \post <tt>\ref vxUnmapArrayRange </tt> with same (*map_id) value.
*/
VX_API_ENTRY vx_status VX_API_CALL vxMapArrayRange(vx_array array, vx_size range_start, vx_size range_end, vx_map_id *map_id, vx_size *stride, void **ptr, vx_enum usage, vx_enum mem_type, vx_uint32 flags)
{
    AgoData * data = (AgoData *)array;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_ARRAY)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if (ptr && stride && range_start < range_end && range_end <= data->u.arr.numitems) {
            if (!data->buffer) {
                CAgoLock lock(data->ref.context->cs);
                if (agoAllocData(data)) {
                    return VX_FAILURE;
                }
            }
            vx_uint8 * ptr_returned = data->buffer + data->u.arr.itemsize * range_start;
            // save the pointer and usage for use in vxCommitXXX
            status = VX_SUCCESS;
            for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
                if (i->ptr == ptr_returned) {
                    // can't support vxAccessXXX() more than once with same pointer
                    // the application needs to call vxCommitXXX() before calling vxAccessXXX()
                    status = VX_FAILURE;
                }
            }
            if (status == VX_SUCCESS) {
#if ENABLE_OPENCL
                if (data->opencl_buffer && !(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                    // make sure dirty OpenCL buffers are synched before giving access for read
                    if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                        // transfer only valid data
                        vx_size size = data->u.arr.itemsize * data->u.arr.numitems;
                        if (size > 0) {
                            cl_int err = clEnqueueReadBuffer(data->ref.context->opencl_cmdq, data->opencl_buffer, CL_TRUE, data->gpu_buffer_offset, size, data->buffer, 0, NULL, NULL);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&data->ref, status, "ERROR: vxMapArrayRange: clEnqueueReadBuffer() => %d\n", err);
                                return status;
                            }
                        }
                        data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                    }
                }
#elif ENABLE_HIP
                if (data->hip_memory && !(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                    // make sure dirty OpenCL buffers are synched before giving access for read
                    if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                        // transfer only valid data
                        vx_size size = data->u.arr.itemsize * data->u.arr.numitems;
                        if (size > 0) {
                            hipError_t err = hipMemcpyDtoH((void *)data->buffer, (data->hip_memory + data->gpu_buffer_offset), size);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&data->ref, status, "ERROR: vxMapArrayRange: hipMemcpyDtoH() => %d\n", err);
                                return status;
                            }
                        }
                    data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                   }
                }
#endif
                MappedData item = { data->nextMapId++, ptr_returned, usage, false, data->u.arr.itemsize };
                data->mapped.push_back(item);
                *map_id = item.map_id;
                *ptr = ptr_returned;
                *stride = item.stride;
            }
        }
    }
    return status;
}

/*! \brief Unmap and commit potential changes to an array object range that was previously mapped.
* Unmapping an array range invalidates the memory location from which the range could
* be accessed by the application. Accessing this memory location after the unmap function
* completes has an undefined behavior.
* \param [in] array The reference to the array object to unmap.
* \param [out] map_id The unique map identifier that was returned when calling
* <tt>\ref vxMapArrayRange</tt> .
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_ERROR_INVALID_REFERENCE The array reference is not actually an array reference.
* \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
* \ingroup group_array
* \pre <tt>\ref vxMapArrayRange</tt> returning the same map_id value
*/
VX_API_ENTRY vx_status VX_API_CALL vxUnmapArrayRange(vx_array array, vx_map_id map_id)
{
    AgoData * data = (AgoData *)array;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_ARRAY)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
            if (i->map_id == map_id) {
                vx_enum usage = i->usage;
                data->mapped.erase(i);
                if (usage == VX_WRITE_ONLY || usage == VX_READ_AND_WRITE) {
                    // update sync flags
                    data->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                    data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
                }
                status = VX_SUCCESS;
                break;
            }
        }
    }
    return status;
}

/*==============================================================================
OBJECTARRAY
=============================================================================*/

/*!
* \brief Creates a reference to an ObjectArray object.
*
* User must specify the ObjectArray count (i.e., the number of objects to create in the ObjectArray).
*
* \param [in] context      The reference to the overall Context.
* \param [in] exemplar     The exemplar object that defines the metadata of the created objects in the ObjectArray.
* \param [in] count        The number of Objects to create in the ObjectArray. This value must be greater than zero
*
* \return <tt>\ref vx_object_array</tt>.
* \retval 0 No Array was created.
* \retval * An Array was created.
*
* \ingroup group_object_array
*
*/
VX_API_ENTRY vx_object_array VX_API_CALL vxCreateObjectArray(vx_context context, vx_reference exemplar, vx_size count)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context) && agoIsValidReference(exemplar) && count > 0) {
        CAgoLock lock(context->cs);
        char desc_exemplar[MAX_DESCRIPTION_DATA_SIZE]; agoGetDescriptionFromData(context, desc_exemplar, (AgoData *)exemplar);
        char desc[2048]; snprintf(desc, sizeof(desc), "objectarray:" VX_FMT_SIZE ",[%s]", count, desc_exemplar);
        data = agoCreateDataFromDescription(context, NULL, desc, true);
        if (data) {
            agoGenerateDataName(context, "objectarray", data->name);
            agoAddData(&context->dataList, data);
            // add the children too
            for (vx_uint32 i = 0; i < data->numChildren; i++) {
                agoAddData(&context->dataList, data->children[i]);
                for (vx_uint32 j = 0; j < data->children[i]->numChildren; j++) {
                    if (data->children[i]->children[j]) {
                        agoAddData(&context->dataList, data->children[i]->children[j]);
                    }
                }
            }
        }
    }
    /*AgoData * data = NULL;
    if (agoIsValidContext(context) && agoIsValidReference(exemplar) && exemplar->type != VX_TYPE_DELAY && exemplar->type != VX_TYPE_OBJECT_ARRAY && count > 0) {
        CAgoLock lock(context->cs);

        vx_enum item_type = exemplar->type;
        vx_status status = VX_SUCCESS;

        const char * desc_type = agoEnum2Name(item_type);
        if (!desc_type) {
            desc_type = agoGetUserStructName(context, item_type);
        }
        if(desc_type){
            char desc[512]; snprintf(desc, sizeof(desc), "objectarray:%s," VX_FMT_SIZE "", desc_type, count);
            data = agoCreateDataFromDescription(context, NULL, desc, true);
            if(data) {
                agoGenerateDataName(context, "objectarray", data->name);
                agoAddData(&context->dataList, data);
            }
        }

        vx_uint32 image_width, image_height;
        vx_df_image image_format;
        vx_size array_capacity;
        vx_enum array_itemtype;
        vx_uint32 pyramid_width, pyramid_height;
        vx_df_image pyramid_format;
        vx_size pyramid_levels;
        vx_float32 pyramid_scale;
        vx_enum scalar_type;
        vx_enum matrix_type;
        vx_size matrix_rows, matrix_cols;
        vx_size distribution_bins;
        vx_int32 distribution_offset;
        vx_uint32 distribution_range;
        vx_uint32 remap_srcwidth, remap_srcheight;
        vx_uint32 remap_dstwidth, remap_dstheight;
        vx_enum lut_type;
        vx_size lut_count;
        vx_enum threshold_type, threshold_data_type;
        vx_df_image threshold_input_format, threshold_output_format;
        vx_size tensor_num_dims, tensor_dims[VX_MAX_TENSOR_DIMENSIONS];
        vx_enum tensor_type;
        vx_int8 tensor_fpp;

        switch(item_type)
        {
            case VX_TYPE_IMAGE:
                if (vxQueryImage((vx_image)exemplar, VX_IMAGE_WIDTH, &image_width, sizeof(image_width)) != VX_SUCCESS ||
                    vxQueryImage((vx_image)exemplar, VX_IMAGE_HEIGHT, &image_height, sizeof(image_height)) != VX_SUCCESS ||
                    vxQueryImage((vx_image)exemplar, VX_IMAGE_FORMAT, &image_format, sizeof(image_format)) != VX_SUCCESS)
                    status = VX_ERROR_INVALID_REFERENCE;
                break;
            case VX_TYPE_ARRAY:
                if (vxQueryArray((vx_array)exemplar, VX_ARRAY_CAPACITY, &array_capacity, sizeof(array_capacity)) != VX_SUCCESS ||
                    vxQueryArray((vx_array)exemplar, VX_ARRAY_ITEMTYPE, &array_itemtype, sizeof(array_itemtype)) != VX_SUCCESS)
                    status = VX_ERROR_INVALID_REFERENCE;
                break;
            case VX_TYPE_PYRAMID:
                if (vxQueryPyramid((vx_pyramid)exemplar, VX_PYRAMID_WIDTH, &pyramid_width, sizeof(pyramid_width)) != VX_SUCCESS ||
                    vxQueryPyramid((vx_pyramid)exemplar, VX_PYRAMID_HEIGHT, &pyramid_height, sizeof(pyramid_height)) != VX_SUCCESS ||
                    vxQueryPyramid((vx_pyramid)exemplar, VX_PYRAMID_FORMAT, &pyramid_format, sizeof(pyramid_format)) != VX_SUCCESS ||
                    vxQueryPyramid((vx_pyramid)exemplar, VX_PYRAMID_LEVELS, &pyramid_levels, sizeof(pyramid_levels)) != VX_SUCCESS ||
                    vxQueryPyramid((vx_pyramid)exemplar, VX_PYRAMID_SCALE, &pyramid_scale, sizeof(pyramid_scale)) != VX_SUCCESS)
                    status = VX_ERROR_INVALID_REFERENCE;
                break;
            case VX_TYPE_SCALAR:
                if (vxQueryScalar((vx_scalar)exemplar, VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)) != VX_SUCCESS)
                    status = VX_ERROR_INVALID_REFERENCE;
                break;
            case VX_TYPE_MATRIX:
                if (vxQueryMatrix((vx_matrix)exemplar, VX_MATRIX_TYPE, &matrix_type, sizeof(matrix_type)) != VX_SUCCESS ||
                    vxQueryMatrix((vx_matrix)exemplar, VX_MATRIX_ROWS, &matrix_rows, sizeof(matrix_rows)) != VX_SUCCESS ||
                    vxQueryMatrix((vx_matrix)exemplar, VX_MATRIX_COLUMNS, &matrix_cols, sizeof(matrix_cols)) != VX_SUCCESS)
                    status = VX_ERROR_INVALID_REFERENCE;
                break;
            case VX_TYPE_DISTRIBUTION:
                if (vxQueryDistribution((vx_distribution)exemplar, VX_DISTRIBUTION_BINS, &distribution_bins, sizeof(distribution_bins)) != VX_SUCCESS ||
                    vxQueryDistribution((vx_distribution)exemplar, VX_DISTRIBUTION_OFFSET, &distribution_offset, sizeof(distribution_offset)) != VX_SUCCESS ||
                    vxQueryDistribution((vx_distribution)exemplar, VX_DISTRIBUTION_RANGE, &distribution_range, sizeof(distribution_range)) != VX_SUCCESS)
                    status = VX_ERROR_INVALID_REFERENCE;
                break;
            case VX_TYPE_REMAP:
                if (vxQueryRemap((vx_remap)exemplar, VX_REMAP_SOURCE_WIDTH, &remap_srcwidth, sizeof(remap_srcwidth)) != VX_SUCCESS ||
                    vxQueryRemap((vx_remap)exemplar, VX_REMAP_SOURCE_HEIGHT, &remap_srcheight, sizeof(remap_srcheight)) != VX_SUCCESS ||
                    vxQueryRemap((vx_remap)exemplar, VX_REMAP_DESTINATION_WIDTH, &remap_dstwidth, sizeof(remap_dstwidth)) != VX_SUCCESS ||
                    vxQueryRemap((vx_remap)exemplar, VX_REMAP_DESTINATION_HEIGHT, &remap_dstheight, sizeof(remap_dstheight)) != VX_SUCCESS)
                    status = VX_ERROR_INVALID_REFERENCE;
                break;
            case VX_TYPE_LUT:
                if (vxQueryLUT((vx_lut)exemplar, VX_LUT_TYPE, &lut_type, sizeof(lut_type)) != VX_SUCCESS ||
                    vxQueryLUT((vx_lut)exemplar, VX_LUT_COUNT, &lut_count, sizeof(lut_count)) != VX_SUCCESS)
                    status = VX_ERROR_INVALID_REFERENCE;
                break;
            case VX_TYPE_THRESHOLD:
                if (vxQueryThreshold((vx_threshold)exemplar, VX_THRESHOLD_TYPE, &threshold_type, sizeof(threshold_type)) != VX_SUCCESS ||
                    vxQueryThreshold((vx_threshold)exemplar, VX_THRESHOLD_DATA_TYPE, &threshold_data_type, sizeof(threshold_data_type)) != VX_SUCCESS ||
                    vxQueryThreshold((vx_threshold)exemplar, VX_THRESHOLD_INPUT_FORMAT, &threshold_input_format, sizeof(threshold_input_format)) != VX_SUCCESS ||
                    vxQueryThreshold((vx_threshold)exemplar, VX_THRESHOLD_OUTPUT_FORMAT, &threshold_output_format, sizeof(threshold_output_format)) != VX_SUCCESS)
                    status = VX_ERROR_INVALID_REFERENCE;
                break;
            case VX_TYPE_TENSOR:
                if (vxQueryTensor((vx_tensor)exemplar, VX_TENSOR_NUMBER_OF_DIMS, &tensor_num_dims, sizeof(tensor_num_dims)) != VX_SUCCESS ||
                    tensor_num_dims > VX_MAX_TENSOR_DIMENSIONS ||
                    vxQueryTensor((vx_tensor)exemplar, VX_TENSOR_DIMS, &tensor_dims[0], sizeof(vx_size) * tensor_num_dims) != VX_SUCCESS ||
                    vxQueryTensor((vx_tensor)exemplar, VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)) != VX_SUCCESS ||
                    vxQueryTensor((vx_tensor)exemplar, VX_TENSOR_FIXED_POINT_POSITION, &tensor_fpp, sizeof(tensor_fpp)) != VX_SUCCESS)
                    status = VX_ERROR_INVALID_REFERENCE;
                break;
            default:
                status =  VX_ERROR_INVALID_TYPE;
                break;
        }
        if(status == VX_SUCCESS)
        {
            for (vx_uint32 i = 0u; i < count; i++)
            {
                vx_reference ref = NULL;

                switch (item_type)
                {
                    case VX_TYPE_IMAGE:
                        ref = (vx_reference)vxCreateImage(context, image_width, image_height, image_format);
                        break;
                    case VX_TYPE_ARRAY:
                        ref = (vx_reference)vxCreateArray(context, array_itemtype, array_capacity);
                        break;
                    case VX_TYPE_PYRAMID:
                        ref = (vx_reference)vxCreatePyramid(context, pyramid_levels, pyramid_scale, pyramid_width, pyramid_height, pyramid_format);
                        break;
                    case VX_TYPE_SCALAR:
                        ref = (vx_reference)vxCreateScalar(context, scalar_type, NULL);
                        break;
                    case VX_TYPE_MATRIX:
                        ref = (vx_reference)vxCreateMatrix(context, matrix_type, matrix_cols, matrix_rows);
                        break;
                    case VX_TYPE_DISTRIBUTION:
                        ref = (vx_reference)vxCreateDistribution(context, distribution_bins, distribution_offset, distribution_range);
                        break;
                    case VX_TYPE_REMAP:
                        ref = (vx_reference)vxCreateRemap(context, remap_srcwidth, remap_srcheight, remap_dstwidth, remap_dstheight);
                        break;
                    case VX_TYPE_LUT:
                        ref = (vx_reference)vxCreateLUT(context, lut_type, lut_count);
                        break;
                    case VX_TYPE_THRESHOLD:
                        ref = (vx_reference)vxCreateThresholdForImage(context, threshold_type, threshold_input_format, threshold_output_format);
                        break;
                    case VX_TYPE_TENSOR:
                        ref = (vx_reference)vxCreateTensor(context, tensor_num_dims, tensor_dims, tensor_type, tensor_fpp);
                        break;
                    default:
                        ref = NULL;
                        break;
                }
                if(agoIsValidReference(ref)){
                    printf("ref type = %u\n", ref->type);
                    //data->u.objarr.items[i] = (vx_reference)malloc(sizeof(ref));
                    //data->u.objarr.items[i] = ref;
                    //ref->scope = (vx_reference)data;
                    memcpy(&data->u.objarr.items[i], &ref, sizeof(ref));
                    printf("i ,data item type %u : %u\n", i, data->u.objarr.items[i]->type);
                }
                else
                {
                    for(vx_uint32 j = 0u; j < i; j++)
                        vxReleaseReference(&(data->u.objarr.items[j]));
                }
            }
        }
    }*/
    return (vx_object_array)data;
}

/*!
* \brief Creates an opaque reference to a virtual ObjectArray with no direct user access.
*
* This function creates an ObjectArray of count objects with similar behavior as vxCreateObjectArray.
* The only difference is that the objects that are created are virtual in the given graph.
*
* \param [in] graph        Reference to the graph where to create the virtual ObjectArray.
* \param [in] exemplar     The exemplar object that defines the type of object in the ObjectArray.
*                          Only exemplar type of vx_image, vx_array and vx_pyramid are allowed.
* \param [in] count        Number of Objects to create in the ObjectArray.
*
* \return <tt>\ref vx_object_array</tt>.
* \retval 0 No ObjectArray was created.
* \retval * An ObjectArray was created or an error occurred. Use <tt>\ref vxGetStatus</tt> to determine.
*
* \ingroup group_object_array
*
*/
VX_API_ENTRY vx_object_array VX_API_CALL vxCreateVirtualObjectArray(vx_graph graph, vx_reference exemplar, vx_size count)
{
    AgoData * data = NULL;
    if (agoIsValidGraph(graph) && agoIsValidReference(exemplar) && count > 0) {
        CAgoLock lock(graph->cs);
        char desc_exemplar[MAX_DESCRIPTION_DATA_SIZE]; agoGetDescriptionFromData(graph->ref.context, desc_exemplar, (AgoData *)exemplar);
        char desc[2048]; snprintf(desc, sizeof(desc), "objectarray:" VX_FMT_SIZE ",[%s]", count, desc_exemplar);
        data = agoCreateDataFromDescription(graph->ref.context, NULL, desc, true);
        if (data) {
            agoGenerateVirtualDataName(graph, "objectarray", data->name);
            agoAddData(&graph->ref.context->dataList, data);
            // add the children too
            for (vx_uint32 i = 0; i < data->numChildren; i++) {
                agoAddData(&graph->ref.context->dataList, data->children[i]);
                for (vx_uint32 j = 0; j < data->children[i]->numChildren; j++) {
                    if (data->children[i]->children[j]) {
                        agoAddData(&graph->ref.context->dataList, data->children[i]->children[j]);
                    }
                }
            }
        }
    }
    /*AgoData * data = NULL;

    if (agoIsValidGraph(graph) && agoIsValidReference(exemplar) && exemplar->type != VX_TYPE_DELAY && exemplar->type != VX_TYPE_OBJECT_ARRAY  && count > 0) {
        CAgoLock lock(graph->cs);

        vx_enum item_type = exemplar->type;
        vx_status status = VX_SUCCESS;

        const char * desc_type = agoEnum2Name(item_type);
        if (!desc_type) {
            desc_type = agoGetUserStructName(graph->ref.context, item_type);
        }
        char desc[512];
        if (desc_type) snprintf(desc, sizeof(desc), "objectarray-virtual:%s," VX_FMT_SIZE "", desc_type, count);
        else snprintf(desc, sizeof(desc), "objectarray-virtual:," VX_FMT_SIZE "", count);
        data = agoCreateDataFromDescription(graph->ref.context, graph, desc, true);
        if(data) {
            agoGenerateVirtualDataName(graph, "objectarray", data->name);
            agoAddData(&graph->dataList, data);
        }

        vx_uint32 image_width, image_height;
        vx_df_image image_format;
        vx_size array_capacity;
        vx_enum array_itemtype;
        vx_uint32 pyramid_width, pyramid_height;
        vx_df_image pyramid_format;
        vx_size pyramid_levels;
        vx_float32 pyramid_scale;

        switch (item_type)
        {
            case VX_TYPE_IMAGE:
                if (vxQueryImage((vx_image)exemplar, VX_IMAGE_WIDTH, &image_width, sizeof(image_width)) != VX_SUCCESS ||
                    vxQueryImage((vx_image)exemplar, VX_IMAGE_HEIGHT, &image_height, sizeof(image_height)) != VX_SUCCESS ||
                    vxQueryImage((vx_image)exemplar, VX_IMAGE_FORMAT, &image_format, sizeof(image_format)) != VX_SUCCESS)
                    status = VX_ERROR_INVALID_REFERENCE;
                break;
            case VX_TYPE_ARRAY:
                if (vxQueryArray((vx_array)exemplar, VX_ARRAY_CAPACITY, &array_capacity, sizeof(array_capacity)) != VX_SUCCESS ||
                    vxQueryArray((vx_array)exemplar, VX_ARRAY_ITEMTYPE, &array_itemtype, sizeof(array_itemtype)) != VX_SUCCESS)
                    status = VX_ERROR_INVALID_REFERENCE;
                break;
            case VX_TYPE_PYRAMID:
                if (vxQueryPyramid((vx_pyramid)exemplar, VX_PYRAMID_WIDTH, &pyramid_width, sizeof(pyramid_width)) != VX_SUCCESS ||
                    vxQueryPyramid((vx_pyramid)exemplar, VX_PYRAMID_HEIGHT, &pyramid_height, sizeof(pyramid_height)) != VX_SUCCESS ||
                    vxQueryPyramid((vx_pyramid)exemplar, VX_PYRAMID_FORMAT, &pyramid_format, sizeof(pyramid_format)) != VX_SUCCESS ||
                    vxQueryPyramid((vx_pyramid)exemplar, VX_PYRAMID_LEVELS, &pyramid_levels, sizeof(pyramid_levels)) != VX_SUCCESS ||
                    vxQueryPyramid((vx_pyramid)exemplar, VX_PYRAMID_SCALE, &pyramid_scale, sizeof(pyramid_scale)) != VX_SUCCESS)
                    status = VX_ERROR_INVALID_REFERENCE;
                break;
            default:
                status =  VX_ERROR_INVALID_TYPE;
                break;
        }

        if(status == VX_SUCCESS)
        {
            for (vx_uint32 i = 0u; i < count; i++)
            {
                vx_reference ref = NULL;

                switch (item_type)
                {
                    case VX_TYPE_IMAGE:
                        ref = (vx_reference)vxCreateVirtualImage(graph, image_width, image_height, image_format);
                        break;
                    case VX_TYPE_ARRAY:
                        ref = (vx_reference)vxCreateVirtualArray(graph, array_itemtype, array_capacity);
                        break;
                    case VX_TYPE_PYRAMID:
                        ref = (vx_reference)vxCreateVirtualPyramid(graph, pyramid_levels, pyramid_scale, pyramid_width, pyramid_height, pyramid_format);
                        break;
                    default:
                        ref = NULL;
                        break;
                }
                data->u.objarr.items[i] = ref;
            }
        }
    }*/
    return (vx_object_array)data;
}

/*!
* \brief Queries an attribute from the ObjectArray.
*
* \param [in] arr          The reference to the ObjectArray.
* \param [in] attribute    The attribute to query. Use a <tt>\ref  vx_object_array_attribute_e</tt>.
* \param [out] ptr         The location at which to store the resulting value.
* \param [in] size         The size of the container to which \a ptr points.
*
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS                   No errors; any other value indicates failure.
* \retval VX_ERROR_INVALID_REFERENCE   \a arr is not a valid <tt>\ref vx_object_array</tt> reference.
* \retval VX_ERROR_NOT_SUPPORTED       If the \a attribute is not a value supported on this implementation.
* \retval VX_ERROR_INVALID_PARAMETERS  If any of the other parameters are incorrect.
*
* \ingroup group_object_array
*
*/
VX_API_ENTRY vx_status VX_API_CALL vxQueryObjectArray(vx_object_array arr, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)arr;
    if (agoIsValidData(data, VX_TYPE_OBJECT_ARRAY)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_OBJECT_ARRAY_ATTRIBUTE_ITEMTYPE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = data->u.arr.itemtype;
                    status = VX_SUCCESS;
                }
                break;
            case VX_OBJECT_ARRAY_ATTRIBUTE_NUMITEMS:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->u.arr.numitems;
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Retrieves the reference to the OpenVX Object in location index of the ObjectArray.
*
* \param [in] arr    The ObjectArray.
* \param [in] index  The index of the object in the ObjectArray.
* \return            A <tt>\ref vx_reference</tt> to an OpenVX data object.
*
* \note This is a <tt>\refvx_reference</tt>, which can be used elsewhere in OpenVX.
* A call to vxRelease<Object> or vxReleaseReference is necessary to release the Object for each call to this function.
*
* \ingroup group_object_array
*
*/
VX_API_ENTRY vx_reference VX_API_CALL vxGetObjectArrayItem(vx_object_array arr, vx_uint32 index)
{
    /*AgoData * data = (AgoData *)arr;
    vx_reference item = NULL;
    if (agoIsValidData(data, VX_TYPE_OBJECT_ARRAY)) {
        if (index < data->u.objarr.numitems) {
            item = data->u.objarr.items[index];
            //printf("in Get!!!!data type =  %u\n", data->u.objarr.items);
        }
    }
    return item;*/
    AgoData * data = (AgoData *)arr;
    AgoData * item = NULL;
    if (agoIsValidData(data, VX_TYPE_OBJECT_ARRAY)) {
        // convert the index from 0..-(N-1) to 0..N-1
        if (index < data->u.objarr.numitems) {
            item = data->children[index];
            agoRetainData((AgoGraph *)data->ref.scope, item, true);
            data->ref.context->num_active_references++;
        }
    }
    return (vx_reference)item;
}

/*!
* \brief Releases a reference of an ObjectArray object.
* The object may not be garbage collected until its total reference and its contained objects count is zero.
* After returning from this function the reference is zeroed/cleared.
*
* \param [in] arr          The pointer to the ObjectArray to release.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors; any other value indicates failure
* \retval VX_ERROR_INVALID_REFERENCE If arr is not a <tt>\ref vx_object_array</tt>.
*
* \ingroup group_object_array
*
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseObjectArray(vx_object_array *arr)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (arr && agoIsValidData((AgoData*)*arr, VX_TYPE_OBJECT_ARRAY)) {
        if (!agoReleaseData((AgoData*)*arr, true)) {
            *arr = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*==============================================================================
META FORMAT
=============================================================================*/

/*! \brief This function allows a user to set the attributes of a <tt>\ref vx_meta_format</tt> object in a kernel output validator.
 *
 * The \ref vx_meta_format object contains two types of information : data object meta data and
 * some specific information that defines how the valid region of an image changes
 *
 * The meta data attributes that can be set are identified by this list:
 * - \ref vx_image : \ref VX_IMAGE_FORMAT, \ref VX_IMAGE_HEIGHT, \ref VX_IMAGE_WIDTH
 * - \ref vx_array : \ref VX_ARRAY_CAPACITY, \ref VX_ARRAY_ITEMTYPE
 * - \ref vx_pyramid : \ref VX_PYRAMID_FORMAT, \ref VX_PYRAMID_HEIGHT, \ref VX_PYRAMID_WIDTH, \ref VX_PYRAMID_LEVELS, \ref VX_PYRAMID_SCALE
 * - \ref vx_scalar : \ref VX_SCALAR_TYPE
 * - \ref vx_matrix : \ref VX_MATRIX_TYPE, \ref VX_MATRIX_ROWS, \ref VX_MATRIX_COLUMNS
 * - \ref vx_distribution : \ref VX_DISTRIBUTION_BINS, \ref VX_DISTRIBUTION_OFFSET, \ref VX_DISTRIBUTION_RANGE
 * - \ref vx_remap : \ref VX_REMAP_SOURCE_WIDTH, \ref VX_REMAP_SOURCE_HEIGHT, \ref VX_REMAP_DESTINATION_WIDTH, \ref VX_REMAP_DESTINATION_HEIGHT
 * - \ref vx_lut : \ref VX_LUT_TYPE, \ref VX_LUT_COUNT
 * - \ref vx_threshold : \ref VX_THRESHOLD_TYPE
 * - \ref VX_VALID_RECT_CALLBACK
 * \note For vx_image, a specific attribute can be used to specify the valid region evolution. This information is not a meta data.
 *
 * \param [in] meta The reference to the \ref vx_meta_format struct to set
 * \param [in] attribute Use the subset of data object attributes that define the meta data of this object or attributes from <tt>\ref vx_meta_format</tt>.
 * \param [in] ptr The input pointer of the value to set on the meta format object.
 * \param [in] size The size in bytes of the object to which \a ptr points.
 * \ingroup group_user_kernels
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS The attribute was set.
 * \retval VX_ERROR_INVALID_REFERENCE meta was not a <tt>\ref vx_meta_format</tt>.
 * \retval VX_ERROR_INVALID_PARAMETER size was not correct for the type needed.
 * \retval VX_ERROR_NOT_SUPPORTED the object attribute was not supported on the meta format object.
 * \retval VX_ERROR_INVALID_TYPE attribute type did not match known meta format type.
 */
VX_API_ENTRY vx_status VX_API_CALL vxSetMetaFormatAttribute(vx_meta_format meta, vx_enum attribute, const void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (meta && agoIsValidReference(&meta->data.ref)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            /**********************************************************************/
            case VX_VALID_RECT_CALLBACK:
                if (size == sizeof(vx_kernel_image_valid_rectangle_f)) {
                    meta->set_valid_rectangle_callback = *(vx_kernel_image_valid_rectangle_f*)ptr;
                    status = VX_SUCCESS;
                }
                break;
            /**********************************************************************/
            case VX_IMAGE_FORMAT:
                if (size == sizeof(vx_df_image) && meta->data.ref.type == VX_TYPE_IMAGE) {
                    meta->data.u.img.format = *(vx_df_image *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_IMAGE_HEIGHT:
                if (size == sizeof(vx_uint32) && meta->data.ref.type == VX_TYPE_IMAGE) {
                    meta->data.u.img.height = *(vx_uint32 *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_IMAGE_WIDTH:
                if (size == sizeof(vx_uint32) && meta->data.ref.type == VX_TYPE_IMAGE) {
                    meta->data.u.img.width = *(vx_uint32 *)ptr;
                    status = VX_SUCCESS;
                }
                break;
#if ENABLE_OPENCL || ENABLE_HIP
            case VX_IMAGE_ATTRIBUTE_AMD_ENABLE_USER_BUFFER_GPU:
                if (size == sizeof(vx_bool) && meta->data.ref.type == VX_TYPE_IMAGE) {
                    meta->data.u.img.enableUserBufferGPU = *(vx_bool *)ptr;
                    status = VX_SUCCESS;
                }
                break;
#endif
            /**********************************************************************/
            case VX_ARRAY_CAPACITY:
                if (size == sizeof(vx_size) && meta->data.ref.type == VX_TYPE_ARRAY) {
                    meta->data.u.arr.capacity = *(vx_size *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_ARRAY_ITEMTYPE:
                if (size == sizeof(vx_enum) && meta->data.ref.type == VX_TYPE_ARRAY) {
                    meta->data.u.arr.itemtype = *(vx_enum *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            /**********************************************************************/
            case VX_OBJECT_ARRAY_NUMITEMS:
                if (size == sizeof(vx_size) && meta->data.ref.type == VX_TYPE_OBJECT_ARRAY) {
                    meta->data.u.objarr.numitems = *(vx_size *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_OBJECT_ARRAY_ITEMTYPE:
                if (size == sizeof(vx_enum) && meta->data.ref.type == VX_TYPE_OBJECT_ARRAY) {
                    meta->data.u.objarr.itemtype = *(vx_enum *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            /**********************************************************************/
            case VX_REMAP_SOURCE_WIDTH:
                if (size == sizeof(vx_uint32) && meta->data.ref.type == VX_TYPE_REMAP) {
                    meta->data.u.remap.src_width = *(vx_uint32 *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_REMAP_SOURCE_HEIGHT:
                if (size == sizeof(vx_uint32) && meta->data.ref.type == VX_TYPE_REMAP) {
                    meta->data.u.remap.src_height = *(vx_uint32 *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_REMAP_DESTINATION_WIDTH:
                if (size == sizeof(vx_uint32) && meta->data.ref.type == VX_TYPE_REMAP) {
                    meta->data.u.remap.dst_width = *(vx_uint32 *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_REMAP_DESTINATION_HEIGHT:
                if (size == sizeof(vx_uint32) && meta->data.ref.type == VX_TYPE_REMAP) {
                    meta->data.u.remap.dst_height = *(vx_uint32 *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            /**********************************************************************/
            case VX_LUT_TYPE:
                if (size == sizeof(vx_enum) && meta->data.ref.type == VX_TYPE_LUT) {
                    meta->data.u.lut.type = *(vx_enum *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_LUT_COUNT:
                if (size == sizeof(vx_size) && meta->data.ref.type == VX_TYPE_LUT) {
                    meta->data.u.lut.count = *(vx_size *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            /**********************************************************************/
            case VX_THRESHOLD_TYPE:
                if (size == sizeof(vx_enum) && meta->data.ref.type == VX_TYPE_THRESHOLD) {
                    meta->data.u.thr.thresh_type = *(vx_enum *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            /**********************************************************************/
            case VX_DISTRIBUTION_BINS:
                if (size == sizeof(vx_size) && meta->data.ref.type == VX_TYPE_DISTRIBUTION) {
                    meta->data.u.dist.numbins = *(vx_size *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_DISTRIBUTION_OFFSET:
                if (size == sizeof(vx_int32) && meta->data.ref.type == VX_TYPE_DISTRIBUTION) {
                    meta->data.u.dist.offset = *(vx_int32 *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_DISTRIBUTION_RANGE:
                if (size == sizeof(vx_uint32) && meta->data.ref.type == VX_TYPE_DISTRIBUTION) {
                    meta->data.u.dist.range = *(vx_uint32 *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_DISTRIBUTION_WINDOW:
                if (size == sizeof(vx_uint32) && meta->data.ref.type == VX_TYPE_DISTRIBUTION) {
                    meta->data.u.dist.window = *(vx_uint32 *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            /**********************************************************************/
            case VX_PYRAMID_FORMAT:
                if (size == sizeof(vx_df_image) && meta->data.ref.type == VX_TYPE_PYRAMID) {
                    meta->data.u.pyr.format = *(vx_df_image *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_PYRAMID_HEIGHT:
                if (size == sizeof(vx_uint32) && meta->data.ref.type == VX_TYPE_PYRAMID) {
                    meta->data.u.pyr.height = *(vx_uint32 *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_PYRAMID_WIDTH:
                if (size == sizeof(vx_uint32) && meta->data.ref.type == VX_TYPE_PYRAMID) {
                    meta->data.u.pyr.width = *(vx_uint32 *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_PYRAMID_LEVELS:
                if (size == sizeof(vx_size) && meta->data.ref.type == VX_TYPE_PYRAMID) {
                    meta->data.u.pyr.levels = *(vx_size *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_PYRAMID_SCALE:
                if (size == sizeof(vx_float32) && meta->data.ref.type == VX_TYPE_PYRAMID) {
                    meta->data.u.pyr.scale = *(vx_float32 *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            /**********************************************************************/
            case VX_SCALAR_TYPE:
                if (size == sizeof(vx_enum) && meta->data.ref.type == VX_TYPE_SCALAR) {
                    meta->data.u.scalar.type = *(vx_enum *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            /**********************************************************************/
            case VX_MATRIX_TYPE:
                if (size == sizeof(vx_enum) && meta->data.ref.type == VX_TYPE_MATRIX) {
                    meta->data.u.mat.type = *(vx_enum *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_MATRIX_COLUMNS:
                if (size == sizeof(vx_size) && meta->data.ref.type == VX_TYPE_MATRIX) {
                    meta->data.u.mat.columns = *(vx_size *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_MATRIX_ROWS:
                if (size == sizeof(vx_size) && meta->data.ref.type == VX_TYPE_MATRIX) {
                    meta->data.u.mat.rows = *(vx_size *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            /**********************************************************************/
            case VX_TENSOR_NUMBER_OF_DIMS:
                if (size == sizeof(vx_size) && meta->data.ref.type == VX_TYPE_TENSOR) {
                    meta->data.u.tensor.num_dims = *(vx_size *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_TENSOR_DIMS:
                if (size <= AGO_MAX_TENSOR_DIMENSIONS * sizeof(vx_size) && meta->data.ref.type == VX_TYPE_TENSOR) {
                    memcpy(&meta->data.u.tensor.dims, ptr, size);
                    status = VX_SUCCESS;
                }
                break;
            case VX_TENSOR_DATA_TYPE:
                if (size == sizeof(vx_enum) && meta->data.ref.type == VX_TYPE_TENSOR) {
                    meta->data.u.tensor.data_type = *(vx_enum *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            case VX_TENSOR_FIXED_POINT_POSITION:
                if (size == sizeof(vx_uint8) && meta->data.ref.type == VX_TYPE_TENSOR) {
                    meta->data.u.tensor.fixed_point_pos = *(vx_int8 *)ptr;
                    status = VX_SUCCESS;
                }
                break;
            /**********************************************************************/
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Set a meta format object from an exemplar data object reference
*
* This function sets a \ref vx_meta_format object from the meta data of the exemplar
*
* \param [in] meta The meta format object to set
* \param [in] exemplar The exemplar data object.
* \ingroup group_user_kernels
* \return A \ref vx_status_e enumeration.
* \retval VX_SUCCESS The meta format was correctly set.
* \retval VX_ERROR_INVALID_REFERENCE the reference was not a reference to a data object
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetMetaFormatFromReference(vx_meta_format meta, vx_reference exemplar)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (meta && agoIsValidReference(&meta->data.ref) && agoIsValidReference(exemplar)) {
        AgoData * ref = (AgoData *)exemplar;
        status = VX_SUCCESS;
        switch (exemplar->type)
        {
        case VX_TYPE_IMAGE:
            meta->data.u.img.width = ref->u.img.width;
            meta->data.u.img.height = ref->u.img.height;
            meta->data.u.img.format = ref->u.img.format;
            break;
        case VX_TYPE_ARRAY:
            meta->data.u.arr.capacity = ref->u.arr.capacity;
            meta->data.u.arr.itemtype = ref->u.arr.itemtype;
            break;
        case VX_TYPE_PYRAMID:
            meta->data.u.pyr.levels = ref->u.pyr.levels;
            meta->data.u.pyr.scale = ref->u.pyr.scale;
            meta->data.u.pyr.width = ref->u.pyr.width;
            meta->data.u.pyr.height = ref->u.pyr.height;
            meta->data.u.pyr.format = ref->u.pyr.format;
            break;
        case VX_TYPE_SCALAR:
            meta->data.u.scalar.type = ref->u.scalar.type;
            break;
        case VX_TYPE_OBJECT_ARRAY:
            meta->data.u.objarr.numitems = ref->u.objarr.numitems;
            meta->data.u.objarr.itemtype = ref->u.objarr.itemtype;
            break;
        case VX_TYPE_REMAP:
            meta->data.u.remap.src_width = ref->u.remap.src_width;
            meta->data.u.remap.src_height = ref->u.remap.src_height;
            meta->data.u.remap.dst_width = ref->u.remap.dst_width;
            meta->data.u.remap.dst_height = ref->u.remap.dst_height;
            break;
        case VX_TYPE_LUT:
            meta->data.u.lut.type = ref->u.lut.type;
            meta->data.u.lut.count = ref->u.lut.count;
            meta->data.u.lut.offset = ref->u.lut.offset;
            break;
        case VX_TYPE_DISTRIBUTION:
            meta->data.u.dist.numbins = ref->u.dist.numbins;
            meta->data.u.dist.offset = ref->u.dist.offset;
            meta->data.u.dist.range = ref->u.dist.range;
            meta->data.u.dist.window = ref->u.dist.window;
            break;
        case VX_TYPE_THRESHOLD:
            meta->data.u.thr.thresh_type = ref->u.thr.thresh_type;
            break;
        case VX_TYPE_MATRIX:
            meta->data.u.mat.type = ref->u.mat.type;
            meta->data.u.mat.columns = ref->u.mat.columns;
            meta->data.u.mat.rows = ref->u.mat.rows;
            meta->data.u.mat.itemsize = ref->u.mat.itemsize;
            meta->data.u.mat.pattern = ref->u.mat.pattern;
            meta->data.u.mat.origin = ref->u.mat.origin;
            break;
        default:
            status = VX_ERROR_INVALID_REFERENCE;
            break;
        }
    }
    return status;
}

/*==============================================================================
MISCELLANEOUS
=============================================================================*/

VX_API_ENTRY vx_status VX_API_CALL vxGetReferenceName(vx_reference ref, vx_char name[], vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidReference(ref)) {
        if ((ref->type >= VX_TYPE_DELAY && ref->type <= VX_TYPE_REMAP) || ref->type == VX_TYPE_TENSOR || (ref->type >= VX_TYPE_VENDOR_OBJECT_START && ref->type <= VX_TYPE_VENDOR_OBJECT_END)) {
            strncpy(name, ((AgoData *)ref)->name.c_str(), size);
            status = VX_SUCCESS;
        }
        else if (ref->type == VX_TYPE_KERNEL) {
            strncpy(name, ((AgoKernel *)ref)->name, size);
            status = VX_SUCCESS;
        }
        else if (ref->type == VX_TYPE_NODE) {
            strncpy(name, ((AgoNode *)ref)->akernel->name, size);
            status = VX_SUCCESS;
        }
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxGetModuleInternalData(vx_context context, const vx_char * module, void ** ptr, vx_size * size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidContext(context)) {
        for (auto it = context->modules.begin(); it != context->modules.end(); it++) {
            if (it->hmodule && !strcmp(it->module_name, module)) {
                *ptr = it->module_internal_data_ptr;
                *size = it->module_internal_data_size;
                status = VX_SUCCESS;
            }
        }
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxSetModuleInternalData(vx_context context, const vx_char * module, void * ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidContext(context)) {
        for (auto it = context->modules.begin(); it != context->modules.end(); it++) {
            if (it->hmodule && !strcmp(it->module_name, module)) {
                it->module_internal_data_ptr = (vx_uint8 *)ptr;
                it->module_internal_data_size = size;
                status = VX_SUCCESS;
            }
        }
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxGetModuleHandle(vx_node node, const vx_char * module, void ** ptr)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidNode(node) && ptr) {
        vx_graph graph = (AgoGraph *)node->ref.scope;
        if(graph->moduleHandle.find(module) == graph->moduleHandle.end()) {
            *ptr = NULL;
        }
        else {
            *ptr = graph->moduleHandle[module];
        }
        status = VX_SUCCESS;
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxSetModuleHandle(vx_node node, const vx_char * module, void * ptr)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidNode(node)) {
        vx_graph graph = (AgoGraph *)node->ref.scope;
        graph->moduleHandle[module] = ptr;
        status = VX_SUCCESS;
    }
    return status;
}

//! \brief Create context from specified platform -- needed for ICD support
extern "C" VX_API_ENTRY vx_context VX_API_CALL vxCreateContextFromPlatform(struct _vx_platform * platform);
VX_API_ENTRY vx_context VX_API_CALL vxCreateContextFromPlatform(struct _vx_platform * platform)
{
    vx_context context = agoCreateContextFromPlatform(platform);
    return context;
}

/*==============================================================================
TENSOR DATA FUNCTIONS
=============================================================================*/

/*! \brief Creates an opaque reference to a tensor data buffer.
 * \details Not guaranteed to exist until the <tt>vx_graph</tt> containing it has been verified.
 * \param [in] context The reference to the implementation context.
 * \param [in] num_of_dims The number of dimensions.
 * \param [in] dims Dimensions sizes in elements.
 * \param [in] data_format The <tt>vx_type_t</tt> that represents the data type of the tensor data elements.
 * \param [in] fixed_point_pos Specifies the fixed point position when the input element type is vx_int16, if 0 calculations are performed in integer math
 * \return A tensor data reference or zero when an error is encountered.
 * \ingroup group_tensor
 */
VX_API_ENTRY vx_tensor VX_API_CALL vxCreateTensor(vx_context context, vx_size num_of_dims, const vx_size * dims, vx_enum data_format, vx_int8 fixed_point_pos)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context) && num_of_dims > 0 && num_of_dims <= AGO_MAX_TENSOR_DIMENSIONS) {
        CAgoLock lock(context->cs);
        char dimStr[256] = "";
        for (vx_size i = 0; i < num_of_dims; i++)
            snprintf(dimStr + strlen(dimStr), sizeof(dimStr), "%s%u", i ? "," : "", (vx_uint32)dims[i]);
        char desc[512];
        snprintf(desc, sizeof(desc), "tensor:%u,{%s},%s,%d", (vx_uint32)num_of_dims, dimStr, agoEnum2Name(data_format), fixed_point_pos);
        data = agoCreateDataFromDescription(context, NULL, desc, true);
        if (data) {
            agoGenerateDataName(context, "tensor", data->name);
            agoAddData(&context->dataList, data);
        }
    }
    return (vx_tensor)data;
}

/*! \brief Creates an opaque reference to a tensor data buffer with no direct
 * user access. This function allows setting the tensor data dimensions or data format.
 * \details Virtual data objects allow users to connect various nodes within a
 * graph via data references without access to that data, but they also permit the
 * implementation to take maximum advantage of possible optimizations. Use this
 * API to create a data reference to link two or more nodes together when the
 * intermediate data are not required to be accessed by outside entities. This API
 * in particular allows the user to define the tensor data format of the data without
 * requiring the exact dimensions. Virtual objects are scoped within the graph
 * they are declared a part of, and can't be shared outside of this scope.
 * \param [in] graph The reference to the parent graph.
 * \param [in] num_of_dims The number of dimensions.
 * \param [in] dims Dimensions sizes in elements.
 * \param [in] data_format The <tt>vx_type_t</tt> that represents the data type of the tensor data elements.
 * \param [in] fixed_point_pos Specifies the fixed point position when the input element type is vx_int16, if 0 calculations are performed in integer math
 * \return A tensor data reference or zero when an error is encountered.
 * \note Passing this reference to <tt>\ref vxCopyTensorPatch</tt> will return an error.
 * \ingroup group_tensor
 */
VX_API_ENTRY vx_tensor VX_API_CALL vxCreateVirtualTensor(vx_graph graph, vx_size num_of_dims, const vx_size * dims, vx_enum data_format, vx_int8 fixed_point_pos)
{
    AgoData * data = NULL;
    if (agoIsValidGraph(graph) && num_of_dims > 0 && num_of_dims <= AGO_MAX_TENSOR_DIMENSIONS) {
        vx_context context = graph->ref.context;
        CAgoLock lock(context->cs);
        char dimStr[256] = "";
        for (vx_size i = 0; i < num_of_dims; i++)
            snprintf(dimStr + strlen(dimStr), sizeof(dimStr), "%s%u", i ? "," : "", (vx_uint32)dims[i]);
        char desc[512];
        snprintf(desc, sizeof(desc), "tensor-virtual:%u,{%s},%s,%i", (vx_uint32)num_of_dims, dimStr, agoEnum2Name(data_format), fixed_point_pos);
        data = agoCreateDataFromDescription(context, graph, desc, true);
        if (data) {
            agoGenerateVirtualDataName(graph, "tensor", data->name);
            agoAddData(&graph->dataList, data);
        }
    }
    return (vx_tensor)data;
}

/*! \brief Creates a tensor data from another tensor data given a view. This second
 * reference refers to the data in the original tensor data. Updates to this tensor data
 * updates the parent tensor data. The view must be defined within the dimensions
 * of the parent tensor data.
 * \param [in] tensor The reference to the parent tensor data.
 * \param [in] num_of_dims The number of dimensions. Must be same as tensor num_of_dims.
 * \param [in] roi_start An array of start values of the roi within the bounds of tensor.
 * \param [in] roi_end An array of end values of the roi within the bounds of tensor.
 * within the parent tensor data dimensions. <tt>\ref vx_tensor_view</tt>
 * \return The reference to the sub-tensor or zero if the view is invalid.
 * \ingroup group_tensor
 */
VX_API_ENTRY vx_tensor VX_API_CALL vxCreateTensorFromView(vx_tensor tensor, vx_size num_of_dims, const vx_size * roi_start, const vx_size * roi_end)
{
    AgoData * master_tensor = (AgoData *)tensor;
    AgoData * data = NULL;
    if (agoIsValidData(master_tensor, VX_TYPE_TENSOR)) {
        if (master_tensor->u.tensor.num_dims != num_of_dims) {
            agoAddLogEntry(&master_tensor->ref, VX_ERROR_INVALID_PARAMETERS, "ERROR: vxCreateTensorFromROI: num_of_dims (%u) doesn't match tensor\n", (vx_uint32)num_of_dims);
            return NULL;
        }
        vx_context context = master_tensor->ref.context;
        CAgoLock lock(context->cs);
        char startStr[256] = "", endStr[256] = "";
        for (vx_size i = 0; i < num_of_dims; i++) {
            snprintf(startStr + strlen(startStr), sizeof(startStr), "%s%u", i ? "," : "", (vx_uint32)roi_start[i]);
            snprintf(endStr + strlen(endStr), sizeof(startStr), "%s%u", i ? "," : "", (vx_uint32)roi_end[i]);
        }
        char desc[1024];
        snprintf(desc, sizeof(desc), "tensor-from-roi:%s,%u,{%s},{%s}", master_tensor->name.c_str(), (vx_uint32)num_of_dims, startStr, endStr);
        if (master_tensor->isVirtual) {
            vx_graph graph = (vx_graph)master_tensor->ref.scope;
            data = agoCreateDataFromDescription(context, graph, desc, true);
            if (data) {
                agoGenerateVirtualDataName(graph, "tensor-from-roi", data->name);
                agoAddData(&graph->dataList, data);
            }
        }
        else {
            data = agoCreateDataFromDescription(context, NULL, desc, true);
            if (data) {
                agoGenerateDataName(context, "tensor-from-roi", data->name);
                agoAddData(&context->dataList, data);
            }
        }
    }
    return (vx_tensor)data;
}

/*! \brief Releases a reference to a tensor data object.
 * The object may not be garbage collected until its total reference count is zero.
 * \param [in] tensor The pointer to the tensor data to release.
 * \post After returning from this function the reference is zeroed.
 * \return A <tt>vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>vx_status_e</tt>.
 * \ingroup group_tensor
 */
VX_API_ENTRY vx_status VX_API_CALL vxReleaseTensor(vx_tensor *tensor)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (tensor && agoIsValidData((AgoData*)*tensor, VX_TYPE_TENSOR)) {
        if (!agoReleaseData((AgoData*)*tensor, true)) {
            *tensor = NULL;
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Retrieves various attributes of a tensor data.
 * \param [in] tensor The reference to the tensor data to query.
 * \param [in] attribute The attribute to query. Use a <tt>\ref vx_tensor_attribute_e</tt>.
 * \param [out] ptr The location at which to store the resulting value.
 * \param [in] size The size of the container to which \a ptr points.
 * \return A <tt>vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE If data is not a <tt>\ref vx_tensor</tt>.
 * \retval VX_ERROR_INVALID_PARAMETERS If any of the other parameters are incorrect.
 * \ingroup group_tensor
 */
VX_API_ENTRY vx_status VX_API_CALL vxQueryTensor(vx_tensor tensor, vx_enum attribute, void *ptr, vx_size size)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * data = (AgoData *)tensor;
    if (agoIsValidData(data, VX_TYPE_TENSOR)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (ptr) {
            switch (attribute)
            {
            case VX_TENSOR_NUMBER_OF_DIMS:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->u.tensor.num_dims;
                    status = VX_SUCCESS;
                }
                break;
            case VX_TENSOR_DIMS:
                if (size >= sizeof(vx_size)*data->u.tensor.num_dims) {
                    for (vx_size i = 0; i < data->u.tensor.num_dims; i++) {
                        ((vx_size *)ptr)[i] = data->u.tensor.dims[i];
                    }
                    status = VX_SUCCESS;
                }
                break;
            case VX_TENSOR_DATA_TYPE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = data->u.tensor.data_type;
                    status = VX_SUCCESS;
                }
                break;
            case VX_TENSOR_FIXED_POINT_POSITION:
                if (size == sizeof(vx_uint8)) {
                    *(vx_int8 *)ptr = data->u.tensor.fixed_point_pos;
                    status = VX_SUCCESS;
                }
                break;
            case VX_TENSOR_MEMORY_TYPE:
                if (size == sizeof(vx_enum)) {
                    *(vx_enum *)ptr = data->import_type;
                    status = VX_SUCCESS;
                }
                break;
#if (ENABLE_OPENCL||ENABLE_HIP)
            case VX_TENSOR_OFFSET_GPU:
                if (size == sizeof(vx_size)) {
                    *(vx_size *)ptr = data->u.tensor.offset;
                    status = VX_SUCCESS;
                }
                break;
            case VX_TENSOR_STRIDE_GPU:
                if (size >= sizeof(vx_size)*data->u.tensor.num_dims) {
                    for (vx_size i = 0; i < data->u.tensor.num_dims; i++) {
                        ((vx_size *)ptr)[i] = data->u.tensor.stride[i];
                    }
                    status = VX_SUCCESS;
                }
                break;
#if ENABLE_OPENCL
            case VX_TENSOR_BUFFER_OPENCL:
                if (size == sizeof(cl_mem)) {
                    if (data->opencl_buffer) {
                        *(cl_mem *)ptr = data->opencl_buffer;
                    }
                    else {
#if defined(CL_VERSION_2_0)
                        *(vx_uint8 **)ptr = data->opencl_svm_buffer;
#else
                        *(vx_uint8 **)ptr = NULL;
#endif
                    }
                    status = VX_SUCCESS;
                }
                break;
#else
            case VX_TENSOR_BUFFER_HIP:
                if (size == sizeof(vx_uint8 *)) {
                    if (data->hip_memory) {
                        *(vx_uint8 **)ptr = data->hip_memory;
                    }
                    else {
                        *(vx_uint8 **)ptr = NULL;
                    }
                    status = VX_SUCCESS;
                }
                break;

#endif
#endif
            case VX_TENSOR_BUFFER_HOST:
                if (size == sizeof(vx_uint8 *)) {
                    if (data->buffer) {
                        *(vx_uint8 **)ptr = data->buffer;
                    }
                    else {
                        *(vx_uint8 **)ptr = NULL;
                    }
                    status = VX_SUCCESS;
                }
                break;
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

/*! \brief Allows the application to copy a view patch from/into an tensor object .
 * \param [in] tensor The reference to the tensor object that is the source or the
 * destination of the copy.
 * \param [in] num_of_dims The number of dimensions. Must be same as tensor num_of_dims.
 * \param [in] roi_start An array of start values of the roi within the bounds of tensor. This is optional parameter and will be zero when NULL.
 * \param [in] roi_end An array of end values of the roi within the bounds of tensor. This is optional parameter and will be dims[] of tensor when NULL.
 * \param [in] user_stride An array of stride in all dimensions in bytes.
 * \param [in] user_ptr The address of the memory location where to store the requested data
 * if the copy was requested in read mode, or from where to get the data to store into the tensor
 * object if the copy was requested in write mode. The accessible memory must be large enough
 * to contain the specified patch with the specified layout:\n
 * accessible memory in bytes >= (end[last_dimension] - start[last_dimension]) * stride[last_dimension].
 * \param [in] usage This declares the effect of the copy with regard to the tensor object
 * using the <tt>vx_accessor_e</tt> enumeration. Only VX_READ_ONLY and VX_WRITE_ONLY are supported:
 * \arg VX_READ_ONLY means that data is copied from the tensor object into the application memory
 * \arg VX_WRITE_ONLY means that data is copied into the tensor object from the application memory
 * \param [in] user_mem_type A <tt>vx_memory_type_e</tt> enumeration that specifies
 * the memory type of the memory referenced by the user_addr.
 * \return A <tt>vx_status_e</tt> enumeration.
 * \retval VX_ERROR_OPTIMIZED_AWAY This is a reference to a virtual tensor that cannot be
 * accessed by the application.
 * \retval VX_ERROR_INVALID_REFERENCE The tensor reference is not actually an tensor reference.
 * \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
 * \ingroup group_tensor
 */
VX_API_ENTRY vx_status VX_API_CALL vxCopyTensorPatch(vx_tensor tensor, vx_size num_of_dims, const vx_size * roi_start, const vx_size * roi_end, const vx_size * user_stride, void * user_ptr, vx_enum usage, vx_enum user_mem_type)
{
    AgoData * data = (AgoData *)tensor;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_TENSOR))
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        bool paramsValid = false;
        bool singleCopy = true;
        vx_size size = 0, start[AGO_MAX_TENSOR_DIMENSIONS], end[AGO_MAX_TENSOR_DIMENSIONS];
        memset(start, 0, sizeof(start));
        memcpy(end, data->u.tensor.dims, sizeof(end));
        if (num_of_dims == data->u.tensor.num_dims) {
            paramsValid = true;
            size = data->u.tensor.stride[0];
            for (vx_size i = 0; i < num_of_dims; i++) {
                if (roi_start)
                    start[i] = roi_start[i];
                if (roi_end)
                    end[i] = roi_end[i];
                if (start[i] >= end[i] || end[i] > data->u.tensor.dims[i])
                    paramsValid = false;
                if (((i == 0) && (user_stride[i] != size)) || ((i > 0) && (user_stride[i] < size)))
                    paramsValid = false; // stride[0] must match and other strides shouldn't be smaller than actual dimensions
                if (user_stride[i] != data->u.tensor.stride[i] || start[i] != 0 || end[i] != data->u.tensor.dims[i] || data->u.tensor.start[i] != 0 || data->u.tensor.end[i] != data->u.tensor.dims[i])
                    singleCopy = false;
                size *= (end[i] - start[i]);
            }
        }
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if (paramsValid && (user_mem_type == VX_MEMORY_TYPE_HOST) && user_ptr && (usage == VX_READ_ONLY || usage == VX_WRITE_ONLY)) {
            if (!data->buffer) {
                CAgoLock lock(data->ref.context->cs);
                if (agoAllocData(data)) {
                    return VX_FAILURE;
                }
            }
            AgoData * dataToSync = data->u.tensor.roiMaster ? data->u.tensor.roiMaster : data;
#if ENABLE_OPENCL
            if (dataToSync->opencl_buffer && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                // make sure dirty OpenCL buffers are synched before giving access for read
                if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                    // transfer only valid data
                    if (dataToSync->size > 0) {
                        cl_int err = clEnqueueReadBuffer(dataToSync->ref.context->opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->gpu_buffer_offset, dataToSync->size, dataToSync->buffer, 0, NULL, NULL);
                        if (err) {
                            status = VX_FAILURE;
                            agoAddLogEntry(&dataToSync->ref, status, "ERROR: vxCopyTensorPatch: clEnqueueReadBuffer() => %d\n", err);
                            return status;
                        }
                    }
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                }
            }
#elif ENABLE_HIP
            if (dataToSync->hip_memory && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                // make sure dirty GPU buffers are synched before giving access for read
                // copy from Device to Host
                if (dataToSync->size > 0 && dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                    hipError_t err = hipMemcpyDtoH((void *)dataToSync->buffer, (dataToSync->hip_memory + dataToSync->gpu_buffer_offset), dataToSync->size);
                    if (err) {
                        status = VX_FAILURE;
                        agoAddLogEntry(&dataToSync->ref, status, "ERROR: vxCopyTensorPatch: hipMemcpyDtoH() => %d\n", err);
                        return status;
                    }
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                }
            }
#endif
            if (usage == VX_READ_ONLY) {
                if (singleCopy) {
                    memcpy(user_ptr, data->buffer, size);
                }
                else {
                    vx_size size0 = data->u.tensor.stride[0] * data->u.tensor.dims[0];
                    for (vx_size d3 = start[3]; d3 < end[3]; d3++) {
                        for (vx_size d2 = start[2]; d2 < end[2]; d2++) {
                            for (vx_size d1 = start[1]; d1 < end[1]; d1++) {
                                vx_size offset =
                                    data->u.tensor.stride[3] * d3 +
                                    data->u.tensor.stride[2] * d2 +
                                    data->u.tensor.stride[1] * d1 +
                                    data->u.tensor.stride[0] * start[0];
                                vx_size uoffset =
                                    user_stride[3] * d3 +
                                    user_stride[2] * d2 +
                                    user_stride[1] * d1 +
                                    user_stride[0] * start[0];
                                memcpy(((vx_uint8 *)user_ptr) + uoffset, data->buffer + offset, size0);
                            }
                        }
                    }
                }
            }
            else {
                if (singleCopy) {
                    memcpy(data->buffer, user_ptr, size);
                }
                else {
                    vx_size size0 = data->u.tensor.stride[0] * data->u.tensor.dims[0];
                    for (vx_size d3 = start[3]; d3 < end[3]; d3++) {
                        for (vx_size d2 = start[2]; d2 < end[2]; d2++) {
                            for (vx_size d1 = start[1]; d1 < end[1]; d1++) {
                                vx_size offset =
                                    data->u.tensor.stride[3] * d3 +
                                    data->u.tensor.stride[2] * d2 +
                                    data->u.tensor.stride[1] * d1 +
                                    data->u.tensor.stride[0] * start[0];
                                vx_size uoffset =
                                    user_stride[3] * d3 +
                                    user_stride[2] * d2 +
                                    user_stride[1] * d1 +
                                    user_stride[0] * start[0];
                                memcpy(data->buffer + offset, ((vx_uint8 *)user_ptr) + uoffset, size0);
                            }
                        }
                    }
                }
                // update sync flags
                dataToSync->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
            }
            status = VX_SUCCESS;
        }
    }
    return status;
}

/*! \brief Allows the application to get direct access to a patch of tensor object.
 * \param [in] tensor The reference to the tensor object that is the source or the
 * destination of the copy.
 * \param [in] num_of_dims The number of dimensions. Must be same as tensor num_of_dims.
 * \param [in] roi_start An array of start values of the roi within the bounds of tensor. This is optional parameter and will be zero when NULL.
 * \param [in] roi_end An array of end values of the roi within the bounds of tensor. This is optional parameter and will be dims[] of tensor when NULL.
 * \param [out] map_id The address of a vx_map_id variable where the function returns a map identifier.
 * \arg (*map_id) must eventually be provided as the map_id parameter of a call to <tt>\ref vxUnmapTensorPatch</tt>.
 * \param [out] stride An array of stride in all dimensions in bytes.
 * \param [out] ptr The address of a pointer that the function sets to the
 * address where the requested data can be accessed. The returned (*ptr) address
 * is only valid between the call to the function and the corresponding call to
 * <tt>\ref vxUnmapTensorPatch</tt>.
 * \param [in] usage This declares the access mode for the tensor patch, using
 * the <tt>\ref vx_accessor_e</tt> enumeration.
 * \arg VX_READ_ONLY: after the function call, the content of the memory location
 * pointed by (*ptr) contains the tensor patch data. Writing into this memory location
 * is forbidden and its behavior is undefined.
 * \arg VX_READ_AND_WRITE : after the function call, the content of the memory
 * location pointed by (*ptr) contains the tensor patch data; writing into this memory
 * is allowed only for the location of items and will result in a modification of the
 * affected items in the tensor object once the range is unmapped. Writing into
 * a gap between items (when (*stride) > item size in bytes) is forbidden and its
 * behavior is undefined.
 * \arg VX_WRITE_ONLY: after the function call, the memory location pointed by (*ptr)
 * contains undefined data; writing each item of the range is required prior to
 * unmapping. Items not written by the application before unmap will become
 * undefined after unmap, even if they were well defined before map. Like for
 * VX_READ_AND_WRITE, writing into a gap between items is forbidden and its behavior
 * is undefined.
 * \param [in] mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that
 * specifies the type of the memory where the tensor patch is requested to be mapped.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_ERROR_OPTIMIZED_AWAY This is a reference to a virtual tensor that cannot be accessed by the application.
 * \retval VX_ERROR_INVALID_REFERENCE The tensor reference is not actually an tensor reference.
 * \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
 * \ingroup group_tensor
 * \post <tt>\ref vxUnmapTensorPatch </tt> with same (*map_id) value.
 */
VX_API_ENTRY vx_status VX_API_CALL vxMapTensorPatch(vx_tensor tensor, vx_size num_of_dims, const vx_size * roi_start, const vx_size * roi_end, vx_map_id * map_id, vx_size * stride, void ** ptr, vx_enum usage, vx_enum mem_type)
{
    AgoData * data = (AgoData *)tensor;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_TENSOR)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        bool paramsValid = false;
        vx_size start[AGO_MAX_TENSOR_DIMENSIONS], end[AGO_MAX_TENSOR_DIMENSIONS];
        memset(start, 0, sizeof(start));
        memcpy(end, data->u.tensor.dims, sizeof(end));
        if (num_of_dims == data->u.tensor.num_dims) {
            paramsValid = true;
            for (vx_size i = 0; i < num_of_dims; i++) {
                if (roi_start)
                    start[i] = roi_start[i];
                if (roi_end)
                    end[i] = roi_end[i];
                if (start[i] >= end[i] || end[i] > data->u.tensor.dims[i])
                    paramsValid = false;
            }
        }
        if (data->isVirtual && !data->buffer) {
            status = VX_ERROR_OPTIMIZED_AWAY;
        }
        else if (paramsValid && ptr && stride && map_id) {
            if (!data->buffer) {
                CAgoLock lock(data->ref.context->cs);
                if (agoAllocData(data)) {
                    return VX_FAILURE;
                }
            }
            vx_size offset = 0;
            for (vx_size i = 0; i < num_of_dims; i++) {
                stride[i] = data->u.tensor.stride[i];
                offset += start[i] * stride[i];
            }
            vx_uint8 * ptr_returned = data->buffer + offset;
            // save the pointer and usage for use in vxUnmapTensorPatch
            status = VX_SUCCESS;
            for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
                if (i->ptr == ptr_returned) {
                    // can't support vxMapTensorPatch() more than once with same pointer
                    // the application needs to call vxUnmapTensorPatch() before calling vxMapTensorPatch()
                    status = VX_FAILURE;
                }
            }
            if (status == VX_SUCCESS) {
                AgoData * dataToSync = data->u.tensor.roiMaster ? data->u.tensor.roiMaster : data;
#if ENABLE_OPENCL
                if (dataToSync->opencl_buffer && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                    // make sure dirty OpenCL buffers are synched before giving access for read
                    if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                        // transfer only valid data
                        if (dataToSync->size > 0) {
                            cl_int err = clEnqueueReadBuffer(dataToSync->ref.context->opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->gpu_buffer_offset, dataToSync->size, dataToSync->buffer, 0, NULL, NULL);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&dataToSync->ref, status, "ERROR: vxMapTensorPatch: clEnqueueReadBuffer() => %d\n", err);
                                return status;
                            }
                        }
                        dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                    }
                }
#elif ENABLE_HIP
                    if (dataToSync->hip_memory && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
                        // make sure dirty OpenCL buffers are synched before giving access for read
                        // copy from Device to Host
                        if (dataToSync->size > 0 && dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                            hipError_t err = hipMemcpyDtoH((void *)dataToSync->buffer, (dataToSync->hip_memory + dataToSync->gpu_buffer_offset), dataToSync->size);
                            if (err) {
                                status = VX_FAILURE;
                                agoAddLogEntry(&dataToSync->ref, status, "ERROR: vxMapTensorPatch: hipMemcpyDtoH() => %d\n", err);
                                return status;
                            }
                            dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                        }
                    }
#endif
                MappedData item = { data->nextMapId++, ptr_returned, usage, false };
                data->mapped.push_back(item);
                *map_id = item.map_id;
                *ptr = ptr_returned;
            }
        }
    }
    return status;
}

/*! \brief Unmap and commit potential changes to a tensor object patch that was previously mapped.
 * Unmapping a tensor patch invalidates the memory location from which the patch could
 * be accessed by the application. Accessing this memory location after the unmap function
 * completes has an undefined behavior.
 * \param [in] tensor The reference to the tensor object to unmap.
 * \param [out] map_id The unique map identifier that was returned when calling
 * <tt>\ref vxMapTensorPatch</tt> .
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_ERROR_INVALID_REFERENCE The tensor reference is not actually an tensor reference.
 * \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
 * \ingroup group_tensor
 * \pre <tt>\ref vxMapTensorPatch</tt> returning the same map_id value
 */
VX_API_ENTRY vx_status VX_API_CALL vxUnmapTensorPatch(vx_tensor tensor, vx_map_id map_id)
{
    AgoData * data = (AgoData *)tensor;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_TENSOR)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        for (auto i = data->mapped.begin(); i != data->mapped.end(); i++) {
            if (i->map_id == map_id) {
                vx_enum usage = i->usage;
                data->mapped.erase(i);
                if (usage == VX_WRITE_ONLY || usage == VX_READ_AND_WRITE) {
                    // update sync flags
                    AgoData * dataToSync = data->u.tensor.roiMaster ? data->u.tensor.roiMaster : data;
                    dataToSync->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
                }
                status = VX_SUCCESS;
                break;
            }
        }
    }
    return status;
}

VX_API_ENTRY vx_tensor VX_API_CALL vxCreateTensorFromHandle(vx_context context, vx_size number_of_dims, const vx_size * dims, vx_enum data_type, vx_int8 fixed_point_position, const vx_size * stride, void * ptr, vx_enum memory_type)
{
    AgoData * data = NULL;
    if (agoIsValidContext(context) && number_of_dims > 0 && number_of_dims <= AGO_MAX_TENSOR_DIMENSIONS) {
        CAgoLock lock(context->cs);
        if (memory_type == VX_MEMORY_TYPE_HOST) {
            char dimStr[256] = "";
            for (vx_size i = 0; i < number_of_dims; i++)
                snprintf(dimStr + strlen(dimStr), sizeof(dimStr), "%s%u", i ? "," : "", (vx_uint32)dims[i]);
            char desc[512];
            snprintf(desc, sizeof(desc), "tensor:%u,{%s},%s,%d", (vx_uint32)number_of_dims, dimStr, agoEnum2Name(data_type), fixed_point_position);
            data = agoCreateDataFromDescription(context, NULL, desc, true);
            if (data) {
                agoGenerateDataName(context, "tensor", data->name);
                agoAddData(&context->dataList, data);
            }
            data->import_type = VX_MEMORY_TYPE_HOST;
            data->buffer = (vx_uint8 *)ptr;
            data->gpu_buffer_offset = 0;
            for (vx_size i = 0; i < number_of_dims; i++) {
                if(data->u.tensor.stride[i] != stride[i]) {
                    agoAddLogEntry(&context->ref, VX_ERROR_INVALID_VALUE, "ERROR: vxCreateTensorFromHandle: invalid stride[%ld]=%ld (must be %ld)\n", i, stride[i], data->u.tensor.stride[i]);
                    vxReleaseTensor((vx_tensor *)&data);
                    break;
                }
            }
        }
#if ENABLE_OPENCL
        else if (memory_type == VX_MEMORY_TYPE_OPENCL) {
            char dimStr[256] = "";
            for (vx_size i = 0; i < number_of_dims; i++)
                snprintf(dimStr + strlen(dimStr), sizeof(dimStr), "%s%u", i ? "," : "", (vx_uint32)dims[i]);
            char desc[512];
            snprintf(desc, sizeof(desc), "tensor:%u,{%s},%s,%d", (vx_uint32)number_of_dims, dimStr, agoEnum2Name(data_type), fixed_point_position);
            data = agoCreateDataFromDescription(context, NULL, desc, true);
            if (data) {
                agoGenerateDataName(context, "tensor", data->name);
                agoAddData(&context->dataList, data);
            }
            data->import_type = VX_MEMORY_TYPE_OPENCL;
            data->opencl_buffer = (cl_mem)ptr;
            data->gpu_buffer_offset = 0;
            for (vx_size i = 0; i < number_of_dims; i++) {
                if(data->u.tensor.stride[i] != stride[i]) {
                    agoAddLogEntry(&context->ref, VX_ERROR_INVALID_VALUE, "ERROR: vxCreateTensorFromHandle: invalid stride[%ld]=%ld (must be %ld)\n", i, stride[i], data->u.tensor.stride[i]);
                    vxReleaseTensor((vx_tensor *)&data);
                    break;
                }
            }
        }
#elif ENABLE_HIP
        else if (memory_type == VX_MEMORY_TYPE_HIP) {
            char dimStr[256] = "";
            for (vx_size i = 0; i < number_of_dims; i++)
                snprintf(dimStr + strlen(dimStr), sizeof(dimStr), "%s%u", i ? "," : "", (vx_uint32)dims[i]);
            char desc[512];
            snprintf(desc, sizeof(desc), "tensor:%u,{%s},%s,%d", (vx_uint32)number_of_dims, dimStr, agoEnum2Name(data_type), fixed_point_position);
            data = agoCreateDataFromDescription(context, NULL, desc, true);
            if (data) {
                agoGenerateDataName(context, "tensor", data->name);
                agoAddData(&context->dataList, data);
            }
            data->import_type = VX_MEMORY_TYPE_HIP;
            data->hip_memory = (vx_uint8 *)ptr;
            data->gpu_buffer_offset = 0;
            for (vx_size i = 0; i < number_of_dims; i++) {
                if(data->u.tensor.stride[i] != stride[i]) {
                    agoAddLogEntry(&context->ref, VX_ERROR_INVALID_VALUE, "ERROR: vxCreateTensorFromHandle: invalid stride[%ld]=%ld (must be %ld)\n", i, stride[i], data->u.tensor.stride[i]);
                    vxReleaseTensor((vx_tensor *)&data);
                    break;
                }
            }
        }
#endif
    }
    return (vx_tensor)data;
}

VX_API_ENTRY vx_status VX_API_CALL vxSwapTensorHandle(vx_tensor tensor, void * new_ptr, void** prev_ptr)
{
    AgoData * data = (AgoData *)tensor;
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidData(data, VX_TYPE_TENSOR) && !data->u.tensor.roiMaster) {
        CAgoLock lock(data->ref.context->cs);
        status = VX_ERROR_INVALID_PARAMETERS;
        if (data->import_type == VX_MEMORY_TYPE_HOST) {
            status = VX_SUCCESS;
            if (prev_ptr) *prev_ptr = data->buffer;
            data->buffer = (vx_uint8 *)new_ptr;
            if (data->buffer) {
                data->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT;
            }
            // propagate to ROIs
            for (auto roi = data->roiDepList.begin(); roi != data->roiDepList.end(); roi++) {
                (*roi)->buffer = data->buffer + (*roi)->u.tensor.offset;
            }
        }
#if ENABLE_OPENCL
        else if (data->import_type == VX_MEMORY_TYPE_OPENCL) {
            status = VX_SUCCESS;
            if (prev_ptr) *prev_ptr = data->opencl_buffer;
            data->opencl_buffer = (cl_mem)new_ptr;
            if (data->opencl_buffer) {
                data->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL;
            }
            // propagate to ROIs
            for (auto roi = data->roiDepList.begin(); roi != data->roiDepList.end(); roi++) {
                (*roi)->opencl_buffer = data->opencl_buffer;
            }
        }
#elif ENABLE_HIP
        else if (data->import_type == VX_MEMORY_TYPE_HIP) {
            status = VX_SUCCESS;
            if (prev_ptr) *prev_ptr = data->hip_memory;
            data->hip_memory = (vx_uint8 *)new_ptr;
            if (data->hip_memory) {
                data->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL;
            }
            // propagate to ROIs
            for (auto roi = data->roiDepList.begin(); roi != data->roiDepList.end(); roi++) {
                (*roi)->hip_memory = data->hip_memory;
            }
        }
#endif
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxAliasTensor(vx_tensor tensorMaster, vx_size offset, vx_tensor tensor)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    AgoData * dataMaster = (AgoData *)tensorMaster;
    AgoData * data = (AgoData *)tensor;
    if (agoIsValidData(dataMaster, VX_TYPE_TENSOR) && agoIsValidData(data, VX_TYPE_TENSOR) &&
        !dataMaster->u.tensor.roiMaster && !data->u.tensor.roiMaster &&
        dataMaster->isVirtual && data->isVirtual)
    {
        data->alias_data = dataMaster;
        data->alias_offset = offset;
        status = VX_SUCCESS;
    }
    return status;
}

VX_API_ENTRY vx_bool VX_API_CALL vxIsTensorAliased(vx_tensor tensorMaster, vx_size offset, vx_tensor tensor)
{
    bool status = vx_false_e;
    AgoData * dataMaster = (AgoData *)tensorMaster;
    AgoData * data = (AgoData *)tensor;
    if (agoIsValidData(dataMaster, VX_TYPE_TENSOR) && agoIsValidData(data, VX_TYPE_TENSOR) &&
        !dataMaster->u.tensor.roiMaster && !data->u.tensor.roiMaster &&
        dataMaster->isVirtual && data->isVirtual &&
        dataMaster == data->alias_data && offset == data->alias_offset)
    {
        status = vx_true_e;
    }
    return status;
}
