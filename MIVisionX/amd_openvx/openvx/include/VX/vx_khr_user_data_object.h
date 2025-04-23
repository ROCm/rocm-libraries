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

#ifndef _OPENVX_USER_DATA_OBJECT_H_
#define _OPENVX_USER_DATA_OBJECT_H_

/*!
 * \file
 * \brief The OpenVX User Data Object extension API.
 */

#define OPENVX_KHR_USER_DATA_OBJECT  "vx_khr_user_data_object"

#include <VX/vx.h>

#ifdef  __cplusplus
extern "C" {
#endif


/*! \brief The User Data Object. User Data Object is a strongly-typed container for other data structures.
 * \ingroup group_user_data_object
 */
typedef struct _vx_user_data_object * vx_user_data_object;


/*! \brief The object type enumeration for user data object.
 * \ingroup group_user_data_object
 */
#define VX_TYPE_USER_DATA_OBJECT          0x816 /*!< \brief A <tt>\ref vx_user_data_object</tt>. */

/*! \brief The user data object attributes.
 * \ingroup group_user_data_object
 */
enum vx_user_data_object_attribute_e {
    /*! \brief The type name of the user data object. Read-only. Use a <tt>\ref vx_char</tt>[<tt>\ref VX_MAX_REFERENCE_NAME</tt>] array. */
    VX_USER_DATA_OBJECT_NAME = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_USER_DATA_OBJECT) + 0x0,
    /*! \brief The number of bytes in the user data object. Read-only. Use a <tt>\ref vx_size</tt> parameter. */
    VX_USER_DATA_OBJECT_SIZE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_USER_DATA_OBJECT) + 0x1,
};


/*!
 * \brief Creates a reference to a User Data Object.
 *
 * User data objects can be used to pass a user kernel defined data structure or blob of memory as a parameter
 * to a user kernel.
 *
 * \param [in] context      The reference to the overall Context.
 * \param [in] type_name    Pointer to the '\0' terminated string that identifies the type of object.
 *                          The string is copied by the function so that it stays the property of the caller.
 *                          The length of the string shall be lower than VX_MAX_REFERENCE_NAME bytes.
 *                          The string passed here is what shall be returned when passing the
 *                          <tt>\ref VX_USER_DATA_OBJECT_NAME</tt> attribute enum to the <tt>\ref vxQueryUserDataObject</tt> function.
 *                          In the case where NULL is passed to type_name, then the query of the <tt>\ref VX_USER_DATA_OBJECT_NAME</tt>
 *                          attribute enum will return a single character '\0' string.
 * \param [in] size         The number of bytes required to store this instance of the user data object.
 * \param [in] ptr          The pointer to the initial value of the user data object. If NULL, then entire size bytes of the user data object
 *                          is initialized to all 0s, otherwise, <tt>size</tt> bytes is copied into the object
 *                          from ptr to initialize the object
 *
 * \returns A user data object reference <tt>\ref vx_user_data_object</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 *
 * \ingroup group_user_data_object
 */
VX_API_ENTRY vx_user_data_object VX_API_CALL vxCreateUserDataObject(vx_context context, const vx_char *type_name, vx_size size, const void *ptr);

/*!
 * \brief Creates an opaque reference to a virtual User Data Object with no direct user access.
 *
 * Virtual User Data Objects are useful when the User Data Object is used as internal graph edge.
 * Virtual User Data Objects are scoped within the parent graph only.
 *
 * \param [in] graph        The reference to the parent graph.
 * \param [in] type_name    Pointer to the '\0' terminated string that identifies the type of object.
 *                          The string is copied by the function so that it stays the property of the caller.
 *                          The length of the string shall be lower than VX_MAX_REFERENCE_NAME bytes.
 *                          The string passed here is what shall be returned when passing the
 *                          <tt>\ref VX_USER_DATA_OBJECT_NAME</tt> attribute enum to the <tt>\ref vxQueryUserDataObject</tt> function.
 *                          In the case where NULL is passed to type_name, then the query of the <tt>\ref VX_USER_DATA_OBJECT_NAME</tt>
 *                          attribute enum will return a single character '\0' string.
 * \param [in] size         The number of bytes required to store this instance of the user data object.
 *
 * \returns A user data object reference <tt>\ref vx_user_data_object</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 *
 * \ingroup group_user_data_object
 */
VX_API_ENTRY vx_user_data_object VX_API_CALL vxCreateVirtualUserDataObject(vx_graph graph, const vx_char *type_name, vx_size size);

/*!
 * \brief Releases a reference of a User data object.
 * The object may not be garbage collected until its total reference count is zero.
 * After returning from this function the reference is zeroed.
 * \param [in] user_data_object  The pointer to the User Data Object to release.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE If user_data_object is not a <tt>\ref vx_user_data_object</tt>.
 * \ingroup group_user_data_object
 */
VX_API_ENTRY vx_status VX_API_CALL vxReleaseUserDataObject(vx_user_data_object *user_data_object);

/*!
 * \brief Queries the User data object for some specific information.
 *
 * \param [in] user_data_object  The reference to the User data object.
 * \param [in] attribute         The attribute to query. Use a <tt>\ref vx_user_data_object_attribute_e</tt>.
 * \param [out] ptr              The location at which to store the resulting value.
 * \param [in] size              The size in bytes of the container to which \a ptr points.
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS                   No errors.
 * \retval VX_ERROR_INVALID_REFERENCE   If the \a user_data_object is not a <tt>\ref vx_user_data_object</tt>.
 * \retval VX_ERROR_NOT_SUPPORTED       If the \a attribute is not a value supported on this implementation.
 * \retval VX_ERROR_INVALID_PARAMETERS  If any of the other parameters are incorrect.
 *
 * \ingroup group_user_data_object
 */
VX_API_ENTRY vx_status VX_API_CALL vxQueryUserDataObject(vx_user_data_object user_data_object, vx_enum attribute, void *ptr, vx_size size);

/*! \brief Allows the application to copy a subset from/into a user data object.
 * \param [in] user_data_object   The reference to the user data object that is the source or the
 *                                destination of the copy.
 * \param [in] offset             The byte offset into the user data object to copy.
 * \param [in] size               The number of bytes to copy.  The size must be within the bounds of the user data object:
 *                                0 <= (offset + size) <= size of the user data object. If zero, then copy until the end of the object.
 * \param [in] user_ptr           The address of the memory location where to store the requested data
 *                                if the copy was requested in read mode, or from where to get the data to store into the user data object
 *                                if the copy was requested in write mode. The accessible memory must be large enough
 *                                to contain the specified size.
 * \param [in] usage               This declares the effect of the copy with regard to the user data object
 *                                using the <tt>\ref vx_accessor_e</tt> enumeration. Only VX_READ_ONLY and VX_WRITE_ONLY
 *                                are supported:
 *                                \arg VX_READ_ONLY means that data are copied from the user data object into the user memory.
 *                                \arg VX_WRITE_ONLY means that data are copied into the user data object from the user memory.
 * \param [in] user_mem_type      A <tt>\ref vx_memory_type_e</tt> enumeration that specifies
 *                                the memory type of the memory referenced by the user_addr.
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_ERROR_OPTIMIZED_AWAY This is a reference to a virtual user data object that cannot be
 * accessed by the application.
 * \retval VX_ERROR_INVALID_REFERENCE The user_data_object reference is not actually a user data object reference.
 * \retval VX_ERROR_INVALID_PARAMETERS Another parameter is incorrect.
 * \ingroup group_user_data_object
 */
VX_API_ENTRY vx_status VX_API_CALL vxCopyUserDataObject(vx_user_data_object user_data_object, vx_size offset, vx_size size, void *user_ptr, vx_enum usage, vx_enum user_mem_type);

/*! \brief Allows the application to get direct access to a subset of the user data object.
 * \param [in] user_data_object   The reference to the user data object that contains the subset to map.
 * \param [in] offset             The byte offset into the user data object to map.
 * \param [in] size               The number of bytes to map.  The size must be within the bounds of the user data object:
 *                                0 <= (offset + size) <= size of the user data object. If zero, then map until the end of the object.
 * \param [out] map_id            The address of a vx_map_id variable where the function returns a map identifier.
 *                                \arg (*map_id) must eventually be provided as the map_id parameter of a call to
 *                                <tt>\ref vxUnmapUserDataObject</tt>.
 * \param [out] ptr               The address of a pointer that the function sets to the
 *                                address where the requested data can be accessed. The returned (*ptr) address
 *                                is only valid between the call to the function and the corresponding call to
 *                                <tt>\ref vxUnmapUserDataObject</tt>.
 * \param [in] usage              This declares the access mode for the user data object subset, using
 *                                the <tt>\ref vx_accessor_e</tt> enumeration.
 *                                \arg VX_READ_ONLY: after the function call, the content of the memory location
 *                                pointed by (*ptr) contains the user data object subset data. Writing into this memory location
 *                                is forbidden and its behavior is implementation specific.
 *                                \arg VX_READ_AND_WRITE : after the function call, the content of the memory
 *                                location pointed by (*ptr) contains the user data object subset data; writing into this memory
 *                                is allowed only for the location of data and will result in a modification of the
 *                                affected data in the user data object once the subset is unmapped.
 *                                \arg VX_WRITE_ONLY: after the function call, the memory location pointed by (*ptr)
 *                                contains undefined data; writing to all data in the subset is required prior to
 *                                unmapping. Data values not written by the application before unmap may be defined differently in
 *                                different implementations after unmap, even if they were well defined before map.
 * \param [in] mem_type           A <tt>\ref vx_memory_type_e</tt> enumeration that
 *                                specifies the type of the memory where the user data object subset is requested to be mapped.
 * \param [in] flags              An integer that allows passing options to the map operation.
 *                                Use the <tt>\ref vx_map_flag_e</tt> enumeration.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_ERROR_OPTIMIZED_AWAY This is a reference to a virtual user data object that cannot be accessed by the application.
 * \retval VX_ERROR_INVALID_REFERENCE The user_data_object reference is not actually a user data object reference.
 * \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
 * \ingroup group_user_data_object
 * \post <tt>\ref vxUnmapUserDataObject </tt> with same (*map_id) value.
 */
VX_API_ENTRY vx_status VX_API_CALL vxMapUserDataObject(vx_user_data_object user_data_object, vx_size offset, vx_size size, vx_map_id *map_id, void **ptr, vx_enum usage, vx_enum mem_type, vx_uint32 flags);

/*! \brief Unmap and commit potential changes to a user data object subset that was previously mapped.
 * Unmapping a user data object subset invalidates the memory location from which the subset could
 * be accessed by the application. Accessing this memory location after the unmap function
 * completes is implementation specific.
 * \param [in] user_data_object   The reference to the user data object to unmap.
 * \param [in] map_id             The unique map identifier that was returned when calling
 *                                <tt>\ref vxMapUserDataObject</tt> .
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_ERROR_INVALID_REFERENCE The user_data_object reference is not actually a user data object reference.
 * \retval VX_ERROR_INVALID_PARAMETERS Another parameter is incorrect.
 * \ingroup group_user_data_object
 * \pre <tt>\ref vxMapUserDataObject</tt> returning the same map_id value
 */
VX_API_ENTRY vx_status VX_API_CALL vxUnmapUserDataObject(vx_user_data_object user_data_object, vx_map_id map_id);

#ifdef  __cplusplus
}
#endif

#endif
