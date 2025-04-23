!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! ==============================================================================
! hipfort: FORTRAN Interfaces for GPU kernels
! ==============================================================================
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
! [MITx11 License]
! 
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
! 
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
! 
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          
           
module hipfort_rocfft
  use hipfort_rocfft_enums
  implicit none

 
  !>  @brief Library setup function, called once in program before start of
  !>  library use
  interface rocfft_setup
    function rocfft_setup_() bind(c, name="rocfft_setup")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_setup_
    end function

  end interface
  !>  @brief Library cleanup function, called once in program after end of library
  !>  use
  interface rocfft_cleanup
    function rocfft_cleanup_() bind(c, name="rocfft_cleanup")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_cleanup_
    end function

  end interface
  !>  @brief Create an FFT plan
  !> 
  !>   @details This API creates a plan, which the user can execute
  !>   subsequently.  This function takes many of the fundamental
  !>   parameters needed to specify a transform.
  !> 
  !>   The dimensions parameter can take a value of 1, 2, or 3. The
  !>   'lengths' array specifies the size of data in each dimension. Note
  !>   that lengths[0] is the size of the innermost dimension, lengths[1]
  !>   is the next higher dimension and so on (column-major ordering).
  !> 
  !>   The 'number_of_transforms' parameter specifies how many
  !>   transforms (of the same kind) needs to be computed. By specifying
  !>   a value greater than 1, a batch of transforms can be computed
  !>   with a single API call.
  !> 
  !>   Additionally, a handle to a plan description can be passed for
  !>   more detailed transforms. For simple transforms, this parameter
  !>   can be set to NULL.
  !> 
  !>   The plan must be destroyed with a call to rocfft_plan_destroy.
  !> 
  !>   @param[out] plan plan handle
  !>   @param[in] placement placement of result
  !>   @param[in] transform_type type of transform
  !>   @param[in] precision precision
  !>   @param[in] dimensions dimensions
  !>   @param[in] lengths dimensions-sized array of transform lengths
  !>   @param[in] number_of_transforms number of transforms
  !>   @param[in] description description handle created by
  !>  rocfft_plan_description_create; can be
  !>   NULL for simple transforms
  interface rocfft_plan_create
    function rocfft_plan_create_(plan,placement,transform_type,myPrecision,dimensions,lengths,number_of_transforms,description) bind(c, name="rocfft_plan_create")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_create_
      type(c_ptr) :: plan
      integer(kind(rocfft_placement_inplace)),value :: placement
      integer(kind(rocfft_transform_type_complex_forward)),value :: transform_type
      integer(kind(rocfft_precision_single)),value :: myPrecision
      integer(c_size_t),value :: dimensions
      type(c_ptr),value :: lengths
      integer(c_size_t),value :: number_of_transforms
      type(c_ptr),value :: description
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      rocfft_plan_create_rank_0,&
      rocfft_plan_create_rank_1
#endif
  end interface
  !>  @brief Execute an FFT plan
  !> 
  !>   @details This API executes an FFT plan on buffers given by the user.
  !> 
  !>   If the transform is in-place, only the input buffer is needed and
  !>   the output buffer parameter can be set to NULL. For not in-place
  !>   transforms, output buffers have to be specified.
  !> 
  !>   Input and output buffer are arrays of pointers.  Interleaved
  !>   array formats are the default, and require just one pointer per
  !>   input or output buffer.  Planar array formats require two
  !>   pointers per input or output buffer - real and imaginary
  !>   pointers, in that order.
  !> 
  !>   Note that input buffers may still be overwritten during execution
  !>   of a transform, even if the transform is not in-place.
  !> 
  !>   The final parameter in this function is a rocfft_execution_info
  !>   handle. This optional parameter serves as a way for the user to control
  !>   execution streams and work buffers.
  !> 
  !>   @param[in] plan plan handle
  !>   @param[in,out] in_buffer array (of size 1 for interleaved data, of size 2
  !>  for planar data) of input buffers
  !>   @param[in,out] out_buffer array (of size 1 for interleaved data, of size 2
  !>  for planar data) of output buffers, ignored for in-place transforms
  !>   @param[in] info execution info handle created by
  !>  rocfft_execution_info_create
  interface rocfft_execute
    function rocfft_execute_(plan,in_buffer,out_buffer,myInfo) bind(c, name="rocfft_execute")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_execute_
      type(c_ptr),value :: plan
      type(c_ptr) :: in_buffer
      type(c_ptr) :: out_buffer
      type(c_ptr),value :: myInfo
    end function

  end interface
  !>  @brief Destroy an FFT plan
  !>   @details This API frees the plan after it is no longer needed.
  !>   @param[in] plan plan handle
  interface rocfft_plan_destroy
    function rocfft_plan_destroy_(plan) bind(c, name="rocfft_plan_destroy")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_destroy_
      type(c_ptr),value :: plan
    end function

  end interface
  !>  @brief Set scaling factor in single precision
  !>   @details This is one of plan description functions to specify optional additional plan properties using the description handle. This API specifies scaling factor.
  !>   @param[in] description description handle
  !>   @param[in] scale scaling factor
  interface rocfft_plan_description_set_scale_float
    function rocfft_plan_description_set_scale_float_(description,scale) bind(c, name="rocfft_plan_description_set_scale_float")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_description_set_scale_float_
      type(c_ptr),value :: description
      real(c_float),value :: scale
    end function

  end interface
  !>  @brief Set scaling factor in double precision
  !>   @details This is one of plan description functions to specify optional additional plan properties using the description handle. This API specifies scaling factor.
  !>   @param[in] description description handle
  !>   @param[in] scale scaling factor
  interface rocfft_plan_description_set_scale_double
    function rocfft_plan_description_set_scale_double_(description,scale) bind(c, name="rocfft_plan_description_set_scale_double")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_description_set_scale_double_
      type(c_ptr),value :: description
      real(c_double),value :: scale
    end function

  end interface
  !>   @brief Set advanced data layout parameters on a plan description
  !>  
  !>   @details This API specifies advanced layout of input/output
  !>   buffers for a plan description.
  !>  
  !>   The following parameters are supported for inputs and outputs:
  !> 
  !>   * Array type (real, hermitian, or complex data, in either
  !>     interleaved or planar format).
  !>       * Real forward transforms require real input and hermitian output.
  !>       * Real inverse transforms require hermitian input and real output.
  !>       * Complex transforms require complex input and output.
  !>       * Hermitian and complex data defaults to interleaved if a specific
  !>           format is not specified.
  !>   * Offset of first data element in the data buffer.  Defaults to 0 if unspecified.
  !>   * Stride between consecutive elements in each dimension.  Defaults
  !>       to contiguous data in all dimensions if unspecified.
  !>   * Distance between consecutive batches.  Defaults to contiguous
  !>       batches if unspecified.
  !> 
  !>   Not all combinations of array types are supported and error codes
  !>   will be returned for unsupported cases.
  !> 
  !>   @param[in, out] description description handle
  !>   @param[in] in_array_type array type of input buffer
  !>   @param[in] out_array_type array type of output buffer
  !>   @param[in] in_offsets offsets, in element units, to start of data in input buffer
  !>   @param[in] out_offsets offsets, in element units, to start of data in output buffer
  !>   @param[in] in_strides_size size of in_strides array (must be equal to transform dimensions)
  !>   @param[in] in_strides array of strides, in each dimension, of
  !>    input buffer; if set to null ptr library chooses defaults
  !>   @param[in] in_distance distance between start of each data instance in input buffer
  !>   @param[in] out_strides_size size of out_strides array (must be
  !>   equal to transform dimensions)
  !>   @param[in] out_strides array of strides, in each dimension, of
  !>    output buffer; if set to null ptr library chooses defaults
  !>   @param[in] out_distance distance between start of each data instance in output buffer
  interface rocfft_plan_description_set_data_layout
    function rocfft_plan_description_set_data_layout_(description,in_array_type,out_array_type,in_offsets,out_offsets,in_strides_size,in_strides,in_distance,out_strides_size,out_strides,out_distance) bind(c, name="rocfft_plan_description_set_data_layout")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_description_set_data_layout_
      type(c_ptr),value :: description
      integer(kind(rocfft_array_type_complex_interleaved)),value :: in_array_type
      integer(kind(rocfft_array_type_complex_interleaved)),value :: out_array_type
      type(c_ptr),value :: in_offsets
      type(c_ptr),value :: out_offsets
      integer(c_size_t),value :: in_strides_size
      type(c_ptr),value :: in_strides
      integer(c_size_t),value :: in_distance
      integer(c_size_t),value :: out_strides_size
      type(c_ptr),value :: out_strides
      integer(c_size_t),value :: out_distance
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      rocfft_plan_description_set_data_layout_rank_0,&
      rocfft_plan_description_set_data_layout_rank_1
#endif
  end interface
  !>  @brief Get library version string
  !> 
  !>  @param[in, out] buf buffer that receives the version string
  !>  @param[in] len length of buf, minimum 30 characters
  interface rocfft_get_version_string
    function rocfft_get_version_string_(buf,len) bind(c, name="rocfft_get_version_string")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_get_version_string_
      type(c_ptr),value :: buf
      integer(c_size_t),value :: len
    end function

  end interface
  !>  @brief Set devices in plan description
  !>   @details This is one of plan description functions to specify optional additional plan properties using the description handle. This API specifies what compute devices to target.
  !>   @param[in] description description handle
  !>   @param[in] devices array of device identifiers
  !>   @param[in] number_of_devices number of devices (size of devices array)
  interface rocfft_plan_description_set_devices
    function rocfft_plan_description_set_devices_(description,devices,number_of_devices) bind(c, name="rocfft_plan_description_set_devices")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_description_set_devices_
      type(c_ptr),value :: description
      type(c_ptr),value :: devices
      integer(c_size_t),value :: number_of_devices
    end function

  end interface
  !>  @brief Get work buffer size
  !>   @details Get the work buffer size required for a plan.
  !>   @param[in] plan plan handle
  !>   @param[out] size_in_bytes size of needed work buffer in bytes
  interface rocfft_plan_get_work_buffer_size
    function rocfft_plan_get_work_buffer_size_(plan,size_in_bytes) bind(c, name="rocfft_plan_get_work_buffer_size")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_get_work_buffer_size_
      type(c_ptr),value :: plan
      integer(c_size_t) :: size_in_bytes
    end function

  end interface
  !>  @brief Print all plan information
  !>   @details Prints plan details to stdout, to aid debugging
  !>   @param[in] plan plan handle
  interface rocfft_plan_get_print
    function rocfft_plan_get_print_(plan) bind(c, name="rocfft_plan_get_print")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_get_print_
      type(c_ptr),value :: plan
    end function

  end interface
  !>  @brief Create plan description
  !>   @details This API creates a plan description with which the user
  !>  can set extra plan properties.  The plan description must be freed
  !>  with a call to rocfft_plan_description_destroy.
  !>   @param[out] description plan description handle
  interface rocfft_plan_description_create
    function rocfft_plan_description_create_(description) bind(c, name="rocfft_plan_description_create")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_description_create_
      type(c_ptr) :: description
    end function

  end interface
  !>  @brief Destroy a plan description
  !>   @details This API frees the plan description.  A plan description
  !>   can be freed any time after it is passed to rocfft_plan_create.
  !>   @param[in] description plan description handle
  interface rocfft_plan_description_destroy
    function rocfft_plan_description_destroy_(description) bind(c, name="rocfft_plan_description_destroy")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_description_destroy_
      type(c_ptr),value :: description
    end function

  end interface
  !>  @brief Create execution info
  !>   @details This API creates an execution info with which the user
  !>  can control plan execution and work buffers.  The execution info must be freed
  !>  with a call to rocfft_execution_info_destroy.
  !>   @param[out] info execution info handle
  interface rocfft_execution_info_create
    function rocfft_execution_info_create_(myInfo) bind(c, name="rocfft_execution_info_create")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_execution_info_create_
      type(c_ptr) :: myInfo
    end function

  end interface
  !>  @brief Destroy an execution info
  !>   @details This API frees the execution info.  An execution info
  !>   object can be freed any time after it is passed to
  !>   rocfft_execute.
  !>   @param[in] info execution info handle
  interface rocfft_execution_info_destroy
    function rocfft_execution_info_destroy_(myInfo) bind(c, name="rocfft_execution_info_destroy")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_execution_info_destroy_
      type(c_ptr),value :: myInfo
    end function

  end interface
  !>  @brief Set work buffer in execution info
  !> 
  !>   @details This is one of the execution info functions to specify
  !>   optional additional information to control execution.  This API
  !>   provides a work buffer for the transform. It must be called
  !>   before rocfft_execute.
  !> 
  !>   When a non-zero value is obtained from
  !>   rocfft_plan_get_work_buffer_size, that means the library needs a
  !>   work buffer to compute the transform. In this case, the user
  !>   should allocate the work buffer and pass it to the library via
  !>   this API.
  !> 
  !>   If a work buffer is required for the transform but is not
  !>   specified using this function, rocfft_execute will automatically
  !>   allocate the required buffer and free it when execution is
  !>   finished.
  !> 
  !>   Users should allocate their own work buffers if they need precise
  !>   control over the lifetimes of those buffers, or if multiple plans
  !>   need to share the same buffer.
  !> 
  !>   @param[in] info execution info handle
  !>   @param[in] work_buffer work buffer
  !>   @param[in] size_in_bytes size of work buffer in bytes
  interface rocfft_execution_info_set_work_buffer
    function rocfft_execution_info_set_work_buffer_(myInfo,work_buffer,size_in_bytes) bind(c, name="rocfft_execution_info_set_work_buffer")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_execution_info_set_work_buffer_
      type(c_ptr),value :: myInfo
      type(c_ptr),value :: work_buffer
      integer(c_size_t),value :: size_in_bytes
    end function

  end interface
  !>  @brief Set execution mode in execution info
  !>   @details This is one of the execution info functions to specify optional additional information to control execution.
  !>   This API specifies execution mode. It has to be called before the call to rocfft_execute.
  !>   Appropriate enumeration value can be specified to control blocking/non-blocking behavior of the rocfft_execute call.
  !>   @param[in] info execution info handle
  !>   @param[in] mode execution mode
  interface rocfft_execution_info_set_mode
    function rocfft_execution_info_set_mode_(myInfo,mode) bind(c, name="rocfft_execution_info_set_mode")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_execution_info_set_mode_
      type(c_ptr),value :: myInfo
      integer(kind(rocfft_exec_mode_nonblocking)),value :: mode
    end function

  end interface
  !>  @brief Set stream in execution info
  !>   @details Associates an existing compute stream to a plan.  This
  !>  must be called before the call to rocfft_execute.
  !> 
  !>   Once the association is made, execution of the FFT will run the
  !>   computation through the specified stream.
  !> 
  !>   The stream must be of type hipStream_t. It is an error to pass
  !>   the address of a hipStream_t object.
  !> 
  !>   @param[in] info execution info handle
  !>   @param[in] stream underlying compute stream
  interface rocfft_execution_info_set_stream
    function rocfft_execution_info_set_stream_(myInfo,stream) bind(c, name="rocfft_execution_info_set_stream")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_execution_info_set_stream_
      type(c_ptr),value :: myInfo
      type(c_ptr),value :: stream
    end function

  end interface
  !>  @brief Set a load callback for a plan execution (experimental)
  !>   @details This function specifies a user-defined callback function
  !>   that is run to load input from global memory at the start of the
  !>   transform.  Callbacks are an experimental feature in rocFFT.
  !> 
  !>   Callback function pointers/data are given as arrays, with one
  !>   function/data pointer per device executing this plan.  Currently,
  !>   plans can only use one device.
  !> 
  !>   The provided function pointers replace any previously-specified
  !>   load callback for this execution info handle.
  !> 
  !>   Load callbacks have the following signature:
  !> 
  !>   @code
  !>   T load_cb(T* data, size_t offset, void* cbdata, void* sharedMem);
  !>   @endcode
  !> 
  !>   'T' is the type of a single element of the input buffer.  It is
  !>   the caller's responsibility to ensure that the function type is
  !>   appropriate for the plan (for example, a single-precision
  !>   real-to-complex transform would load single-precision real
  !>   elements).
  !> 
  !>   A null value for 'cb' may be specified to clear any previously
  !>   registered load callback.
  !> 
  !>   Currently, 'shared_mem_bytes' must be 0.  Callbacks are not
  !>   supported on transforms that use planar formats for either input
  !>   or output.
  !> 
  !>   @param[in] info execution info handle
  !>   @param[in] cb callback function pointers
  !>   @param[in] cbdata callback function data, passed to the function pointer when it is called
  !>   @param[in] shared_mem_bytes amount of shared memory to allocate for the callback function to use
  interface rocfft_execution_info_set_load_callback
    function rocfft_execution_info_set_load_callback_(myInfo,cb_functions,cb_data,shared_mem_bytes) bind(c, name="rocfft_execution_info_set_load_callback")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_execution_info_set_load_callback_
      type(c_ptr),value :: myInfo
      type(c_ptr) :: cb_functions
      type(c_ptr) :: cb_data
      integer(c_size_t),value :: shared_mem_bytes
    end function

  end interface
  !>  @brief Set a store callback for a plan execution (experimental)
  !>   @details This function specifies a user-defined callback function
  !>   that is run to store output to global memory at the end of the
  !>   transform.  Callbacks are an experimental feature in rocFFT.
  !> 
  !>   Callback function pointers/data are given as arrays, with one
  !>   function/data pointer per device executing this plan.  Currently,
  !>   plans can only use one device.
  !> 
  !>   The provided function pointers replace any previously-specified
  !>   store callback for this execution info handle.
  !> 
  !>   Store callbacks have the following signature:
  !> 
  !>   @code
  !>   void store_cb(T* data, size_t offset, T element, void* cbdata, void* sharedMem);
  !>   @endcode
  !> 
  !>   'T' is the type of a single element of the output buffer.  It is
  !>   the caller's responsibility to ensure that the function type is
  !>   appropriate for the plan (for example, a single-precision
  !>   real-to-complex transform would store single-precision complex
  !>   elements).
  !> 
  !>   A null value for 'cb' may be specified to clear any previously
  !>   registered store callback.
  !> 
  !>   Currently, 'shared_mem_bytes' must be 0.  Callbacks are not
  !>   supported on transforms that use planar formats for either input
  !>   or output.
  !> 
  !>   @param[in] info execution info handle
  !>   @param[in] cb callbacks function pointers
  !>   @param[in] cbdata callback function data, passed to the function pointer when it is called
  !>   @param[in] shared_mem_bytes amount of shared memory to allocate for the callback function to use
  interface rocfft_execution_info_set_store_callback
    function rocfft_execution_info_set_store_callback_(myInfo,cb_functions,cb_data,shared_mem_bytes) bind(c, name="rocfft_execution_info_set_store_callback")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_execution_info_set_store_callback_
      type(c_ptr),value :: myInfo
      type(c_ptr) :: cb_functions
      type(c_ptr) :: cb_data
      integer(c_size_t),value :: shared_mem_bytes
    end function

  end interface
  !>  @brief Get events from execution info
  !>   @details This is one of the execution info functions to retrieve information from execution.
  !>   This API obtains event information. It has to be called after the call to rocfft_execute.
  !>   This gets handles to events that the library created around one or more kernel launches during execution.
  !>   @param[in] info execution info handle
  !>   @param[out] events array of events
  !>   @param[out] number_of_events number of events (size of events array)
  interface rocfft_execution_info_get_events
    function rocfft_execution_info_get_events_(myInfo,events,number_of_events) bind(c, name="rocfft_execution_info_get_events")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_execution_info_get_events_
      type(c_ptr),value :: myInfo
      type(c_ptr) :: events
      integer(c_size_t) :: number_of_events
    end function

  end interface
  !>  @brief Serialize compiled kernel cache
  !> 
  !>   @details Serialize rocFFT's cache of compiled kernels into a
  !>   buffer.  This buffer is allocated by rocFFT and must be freed
  !>   with a call to rocfft_cache_buffer_free.  The length of the
  !>   buffer in bytes is written to 'buffer_len_bytes'.
  interface rocfft_cache_serialize
    function rocfft_cache_serialize_(buffer,buffer_len_bytes) bind(c, name="rocfft_cache_serialize")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_cache_serialize_
      type(c_ptr) :: buffer
      type(c_ptr),value :: buffer_len_bytes
    end function

  end interface
  !>  @brief Free cache serialization buffer
  !> 
  !>   @details Deallocate a buffer allocated by rocfft_cache_serialize.
  interface rocfft_cache_buffer_free
    function rocfft_cache_buffer_free_(buffer) bind(c, name="rocfft_cache_buffer_free")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_cache_buffer_free_
      type(c_ptr),value :: buffer
    end function

  end interface
  !>  @brief Deserialize a buffer into the compiled kernel cache.
  !> 
  !>   @details Kernels in the buffer that match already-cached kernels
  !>   will replace those kernels that are in the cache.  Already-cached
  !>   kernels that do not match those in the buffer are unmodified by
  !>   this operation.  The cache is unmodified if either a null buffer
  !>   pointer or a zero length is passed.
  interface rocfft_cache_deserialize
    function rocfft_cache_deserialize_(buffer,buffer_len_bytes) bind(c, name="rocfft_cache_deserialize")
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_cache_deserialize_
      type(c_ptr),value :: buffer
      integer(c_size_t),value :: buffer_len_bytes
    end function

  end interface

#ifdef USE_FPOINTER_INTERFACES
  contains
    function rocfft_plan_create_rank_0(plan,placement,transform_type,myPrecision,dimensions,lengths,number_of_transforms,description)
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_create_rank_0
      type(c_ptr) :: plan
      integer(kind(rocfft_placement_inplace)) :: placement
      integer(kind(rocfft_transform_type_complex_forward)) :: transform_type
      integer(kind(rocfft_precision_single)) :: myPrecision
      integer(c_size_t) :: dimensions
      integer(c_size_t),target :: lengths
      integer(c_size_t) :: number_of_transforms
      type(c_ptr) :: description
      !
      rocfft_plan_create_rank_0 = rocfft_plan_create_(plan,placement,transform_type,myPrecision,dimensions,c_loc(lengths),number_of_transforms,description)
    end function

    function rocfft_plan_create_rank_1(plan,placement,transform_type,myPrecision,dimensions,lengths,number_of_transforms,description)
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_create_rank_1
      type(c_ptr) :: plan
      integer(kind(rocfft_placement_inplace)) :: placement
      integer(kind(rocfft_transform_type_complex_forward)) :: transform_type
      integer(kind(rocfft_precision_single)) :: myPrecision
      integer(c_size_t) :: dimensions
      integer(c_size_t),target,dimension(:) :: lengths
      integer(c_size_t) :: number_of_transforms
      type(c_ptr) :: description
      !
      rocfft_plan_create_rank_1 = rocfft_plan_create_(plan,placement,transform_type,myPrecision,dimensions,c_loc(lengths),number_of_transforms,description)
    end function

    function rocfft_plan_description_set_data_layout_rank_0(description,in_array_type,out_array_type,in_offsets,out_offsets,in_strides_size,in_strides,in_distance,out_strides_size,out_strides,out_distance)
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_description_set_data_layout_rank_0
      type(c_ptr) :: description
      integer(kind(rocfft_array_type_complex_interleaved)) :: in_array_type
      integer(kind(rocfft_array_type_complex_interleaved)) :: out_array_type
      integer(c_size_t),target :: in_offsets
      integer(c_size_t),target :: out_offsets
      integer(c_size_t) :: in_strides_size
      integer(c_size_t),target :: in_strides
      integer(c_size_t) :: in_distance
      integer(c_size_t) :: out_strides_size
      integer(c_size_t),target :: out_strides
      integer(c_size_t) :: out_distance
      !
      rocfft_plan_description_set_data_layout_rank_0 = rocfft_plan_description_set_data_layout_(description,in_array_type,out_array_type,c_loc(in_offsets),c_loc(out_offsets),in_strides_size,c_loc(in_strides),in_distance,out_strides_size,c_loc(out_strides),out_distance)
    end function

    function rocfft_plan_description_set_data_layout_rank_1(description,in_array_type,out_array_type,in_offsets,out_offsets,in_strides_size,in_strides,in_distance,out_strides_size,out_strides,out_distance)
      use iso_c_binding
      use hipfort_rocfft_enums
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_description_set_data_layout_rank_1
      type(c_ptr) :: description
      integer(kind(rocfft_array_type_complex_interleaved)) :: in_array_type
      integer(kind(rocfft_array_type_complex_interleaved)) :: out_array_type
      integer(c_size_t),target,dimension(:) :: in_offsets
      integer(c_size_t),target,dimension(:) :: out_offsets
      integer(c_size_t) :: in_strides_size
      integer(c_size_t),target,dimension(:) :: in_strides
      integer(c_size_t) :: in_distance
      integer(c_size_t) :: out_strides_size
      integer(c_size_t),target,dimension(:) :: out_strides
      integer(c_size_t) :: out_distance
      !
      rocfft_plan_description_set_data_layout_rank_1 = rocfft_plan_description_set_data_layout_(description,in_array_type,out_array_type,c_loc(in_offsets),c_loc(out_offsets),in_strides_size,c_loc(in_strides),in_distance,out_strides_size,c_loc(out_strides),out_distance)
    end function

  
#endif
end module hipfort_rocfft
