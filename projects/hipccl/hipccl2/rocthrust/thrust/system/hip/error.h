/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


/*! \file thrust/system/hip/error.h
 *  \brief HIP-specific error reporting
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/system/error_code.h>
#include <thrust/system/hip/detail/guarded_driver_types.h>

namespace thrust
{

namespace system
{

namespace hip_rocprim
{

/*! \addtogroup system
 *  \{
 */

// To construct an error_code after a HIP Runtime error:
//
//   error_code(::hipGetLastError(), hip_category())

// XXX N3000 prefers enum class errc { ... }
namespace errc
{

/*! \p errc_t enumerates the kinds of HIP Runtime errors.
 */
enum errc_t
{
    // from hip/include/driver_types.h
    // mirror their order
    success                            = hipSuccess,
    missing_configuration              = hipErrorMissingConfiguration,
    memory_allocation                  = hipErrorMemoryAllocation,
    initialization_error               = hipErrorInitializationError,
    launch_failure                     = hipErrorLaunchFailure,
    prior_launch_failure               = hipErrorPriorLaunchFailure,
    launch_timeout                     = hipErrorLaunchTimeOut,
    launch_out_of_resources            = hipErrorLaunchOutOfResources,
    invalid_device_function            = hipErrorInvalidDeviceFunction,
    invalid_configuration              = hipErrorInvalidConfiguration,
    invalid_device                     = hipErrorInvalidDevice,
    invalid_value                      = hipErrorInvalidValue,
    invalid_pitch_value                = hipErrorTbd,
    invalid_symbol                     = hipErrorInvalidSymbol,
    map_buffer_object_failed           = hipErrorMapBufferObjectFailed,
    unmap_buffer_object_failed         = hipErrorTbd,
    invalid_host_pointer               = hipErrorTbd,
    invalid_device_pointer             = hipErrorInvalidDevicePointer,
    invalid_texture                    = hipErrorTbd,
    invalid_texture_binding            = hipErrorTbd,
    invalid_channel_descriptor         = hipErrorTbd,
    invalid_memcpy_direction           = hipErrorInvalidMemcpyDirection,
    address_of_constant_error          = hipErrorTbd,
    texture_fetch_failed               = hipErrorTbd,
    texture_not_bound                  = hipErrorTbd,
    synchronization_error              = hipErrorTbd,
    invalid_filter_setting             = hipErrorTbd,
    invalid_norm_setting               = hipErrorTbd,
    mixed_device_execution             = hipErrorTbd,
    cuda_runtime_unloading             = hipErrorTbd,
    unknown                            = hipErrorUnknown,
    not_yet_implemented                = hipErrorTbd,
    memory_value_too_large             = hipErrorTbd,
    invalid_resource_handle            = hipErrorInvalidResourceHandle,
    not_ready                          = hipErrorNotReady,
    insufficient_driver                = hipErrorTbd,
    set_on_active_process_error        = hipErrorSetOnActiveProcess,
    no_device                          = hipErrorNoDevice,
    ecc_uncorrectable                  = hipErrorTbd,
    shared_object_symbol_not_found     = hipErrorSharedObjectSymbolNotFound,
    shared_object_init_failed          = hipErrorSharedObjectInitFailed,
    unsupported_limit                  = hipErrorUnsupportedLimit,
    duplicate_variable_name            = hipErrorTbd,
    duplicate_texture_name             = hipErrorTbd,
    duplicate_surface_name             = hipErrorTbd,
    devices_unavailable                = hipErrorTbd,
    invalid_kernel_image               = hipErrorTbd,
    no_kernel_image_for_device         = hipErrorTbd,
    incompatible_driver_context        = hipErrorTbd,
    peer_access_already_enabled        = hipErrorPeerAccessAlreadyEnabled,
    peer_access_not_enabled            = hipErrorPeerAccessNotEnabled,
    device_already_in_use              = hipErrorTbd,
    profiler_disabled                  = hipErrorTbd,
    assert_triggered                   = hipErrorTbd,
    too_many_peers                     = hipErrorTbd,
    host_memory_already_registered     = hipErrorHostMemoryAlreadyRegistered,
    host_memory_not_registered         = hipErrorHostMemoryNotRegistered,
    operating_system_error             = hipErrorTbd,
    peer_access_unsupported            = hipErrorTbd,
    launch_max_depth_exceeded          = hipErrorTbd,
    launch_file_scoped_texture_used    = hipErrorTbd,
    launch_file_scoped_surface_used    = hipErrorTbd,
    sync_depth_exceeded                = hipErrorTbd,
    attempted_operation_not_permitted  = hipErrorTbd,
    attempted_operation_not_supported  = hipErrorTbd,
    startup_failure                    = hipErrorTbd
}; // end errc_t


} // end namespace errc

} // end namespace hip_rocprim

/*! \return A reference to an object of a type derived from class \p thrust::error_category.
 *  \note The object's \p equivalent virtual functions shall behave as specified
 *        for the class \p thrust::error_category. The object's \p name virtual function shall
 *        return a pointer to the string <tt>"hip"</tt>. The object's
 *        \p default_error_condition virtual function shall behave as follows:
 *
 *        If the argument <tt>ev</tt> corresponds to a HIP error value, the function
 *        shall return <tt>error_condition(ev,hip_category())</tt>.
 *        Otherwise, the function shall return <tt>system_category.default_error_condition(ev)</tt>.
 */
inline const error_category &hip_category(void);


// XXX N3000 prefers is_error_code_enum<hip::errc>

/*! Specialization of \p is_error_code_enum for \p hip::errc::errc_t
 */
template<> struct is_error_code_enum<hip_rocprim::errc::errc_t> : thrust::detail::true_type {};


/*! \return <tt>error_code(static_cast<int>(e), hip::error_category())</tt>
 */
inline error_code make_error_code(hip_rocprim::errc::errc_t e);


/*! \return <tt>error_condition(static_cast<int>(e), hip::error_category())</tt>.
 */
inline error_condition make_error_condition(hip_rocprim::errc::errc_t e);

/*! \} // end system
 */


} // end system

namespace system {
namespace hip {
namespace errc {
using system::hip_rocprim::errc::errc_t;
} // namespace errc
} // namespace hip
} // namespace system

namespace hip
{
// XXX replace with using system::hip_errc upon c++0x
namespace errc = system::hip::errc;
} // end hip

using system::hip_category;

} // end namespace thrust

#include <thrust/system/hip/detail/error.inl>
