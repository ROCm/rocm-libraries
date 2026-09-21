!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! ==============================================================================
! hipfort: FORTRAN Interfaces for GPU kernels
! ==============================================================================
! Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

module rocfft
  use, intrinsic :: iso_c_binding
  implicit none

  ! rocfft_status_e
  enum, bind(c)
    enumerator :: rocfft_status_success = 0
    enumerator :: rocfft_status_failure = 1
    enumerator :: rocfft_status_invalid_arg_value = 2
    enumerator :: rocfft_status_invalid_dimensions = 3
    enumerator :: rocfft_status_invalid_array_type = 4
    enumerator :: rocfft_status_invalid_strides = 5
    enumerator :: rocfft_status_invalid_distance = 6
    enumerator :: rocfft_status_invalid_offset = 7
    enumerator :: rocfft_status_invalid_work_buffer = 8
  end enum

  ! rocfft_transform_type_e
  enum, bind(c)
    enumerator :: rocfft_transform_type_complex_forward = 0
    enumerator :: rocfft_transform_type_complex_inverse = 1
    enumerator :: rocfft_transform_type_real_forward = 2
    enumerator :: rocfft_transform_type_real_inverse = 3
  end enum

  ! rocfft_precision_e
  enum, bind(c)
    enumerator :: rocfft_precision_single = 0
    enumerator :: rocfft_precision_double = 1
    enumerator :: rocfft_precision_half = 2
  end enum

  ! rocfft_result_placement_e
  enum, bind(c)
    enumerator :: rocfft_placement_inplace = 0
    enumerator :: rocfft_placement_notinplace = 1
  end enum

  ! rocfft_array_type_e
  enum, bind(c)
    enumerator :: rocfft_array_type_complex_interleaved = 0
    enumerator :: rocfft_array_type_complex_planar = 1
    enumerator :: rocfft_array_type_real = 2
    enumerator :: rocfft_array_type_hermitian_interleaved = 3
    enumerator :: rocfft_array_type_hermitian_planar = 4
    enumerator :: rocfft_array_type_unset = 5
  end enum

  ! rocfft_comm_type_e
  enum, bind(c)
    enumerator :: rocfft_comm_none = 0
    enumerator :: rocfft_comm_mpi = 1
  end enum


  interface

    !---------------------------------------------
    ! rocfft_setup
    !---------------------------------------------
    !>  @brief Library setup function, called at least once in program before
    !>  start of library use
    function rocfft_setup() &
       result(setup) &
       bind(C, name="rocfft_setup")
       import :: rocfft_status_success
       integer(kind(rocfft_status_success)) :: setup
    end function rocfft_setup

    !---------------------------------------------
    ! rocfft_cleanup
    !---------------------------------------------
    !>  @brief Library cleanup function, called exactly as many times as rocfft_setup
    !>  in program after end of library use
    function rocfft_cleanup() &
       result(cleanup) &
       bind(C, name="rocfft_cleanup")
       import :: rocfft_status_success
       integer(kind(rocfft_status_success)) :: cleanup
    end function rocfft_cleanup

    !---------------------------------------------
    ! rocfft_execute
    !---------------------------------------------
    !>  @brief Execute an FFT plan
    !>
    !>   @details This API executes an FFT plan on buffers given by the user.
    !>
    !>   If the transform is in-place, only the input buffer is needed and
    !>   the output buffer parameter can be set to NULL. For not in-place
    !>   transforms, output buffers have to be specified.
    !>
    !>   Input and output buffers are arrays of pointers.  Interleaved
    !>   array formats are the default, and require just one pointer per
    !>   input or output buffer.  Planar array formats require two
    !>   pointers per input or output buffer - real and imaginary
    !>   pointers, in that order.
    !>
    !>   If fields have been set for transform input or output, these
    !>   arrays have one pointer per brick in the input or output field,
    !>   provided in the order that the bricks were added to the field.
    !>
    !>   Note that input buffers may still be overwritten during execution
    !>   of a transform, even if the transform is not in-place.
    !>
    !>   The final parameter in this function is a rocfft_execution_info
    !>   handle. This optional parameter serves as a way for the user to control
    !>   execution streams and work buffers.
    !>
    !>   @param[in] plan - plan handle
    !>   @param[in,out] in_buffer - array (of size 1 for interleaved data, of size 2
    !>  for planar data, or one per brick if an input field is set) of input buffers
    !>   @param[in,out] out_buffer - array (of size 1 for interleaved data, of size 2
    !>  for planar data, or one per brick if an output field is set) of output buffers,
    !>  ignored for in-place transforms
    !>   @param[in] myInfo - execution info handle created by
    !>  rocfft_execution_info_create
    function rocfft_execute(plan, in_buffer, out_buffer, myInfo) &
       result(execute) &
       bind(C, name="rocfft_execute")
       import :: c_ptr, rocfft_status_success
       type(c_ptr), value :: plan
       type(c_ptr) :: in_buffer
       type(c_ptr) :: out_buffer
       type(c_ptr), value :: myInfo
       integer(kind(rocfft_status_success)) :: execute
    end function rocfft_execute

    !---------------------------------------------
    ! rocfft_plan_destroy
    !---------------------------------------------
    !>  @brief Destroy an FFT plan
    !>   @details This API frees the plan after it is no longer needed.
    !>   @param[in] plan - plan handle
    function rocfft_plan_destroy(plan) &
       result(plan_destroy) &
       bind(C, name="rocfft_plan_destroy")
       import :: c_ptr, rocfft_status_success
       type(c_ptr), value :: plan
       integer(kind(rocfft_status_success)) :: plan_destroy
    end function rocfft_plan_destroy

    !---------------------------------------------
    ! rocfft_plan_description_set_scale_factor
    !---------------------------------------------
    !>  @brief Set scaling factor.
    !>   @details rocFFT multiplies each element of the result by the given factor at the end of the
    !>   transform.
    !>
    !>   The supplied factor must be a finite number.  That is, it must neither be infinity nor NaN.
    !>
    !>   @param[in] description - description handle
    !>   @param[in] scale_factor - scaling factor
    function rocfft_plan_description_set_scale_factor(description, scale_factor) &
       result(plan_description_set_scale_factor) &
       bind(C, name="rocfft_plan_description_set_scale_factor")
       import :: c_ptr, c_double, rocfft_status_success
       type(c_ptr), value :: description
       real(c_double), value :: scale_factor
       integer(kind(rocfft_status_success)) :: plan_description_set_scale_factor
    end function rocfft_plan_description_set_scale_factor

    !---------------------------------------------
    ! rocfft_field_create
    !---------------------------------------------
    !>  @brief Create a rocfft field struct.
    !>
    !>   @warning Experimental!  This feature is part of an experimental API preview.
    function rocfft_field_create(field) &
       result(field_create) &
       bind(C, name="rocfft_field_create")
       import :: c_ptr, rocfft_status_success
       type(c_ptr) :: field
       integer(kind(rocfft_status_success)) :: field_create
    end function rocfft_field_create

    !---------------------------------------------
    ! rocfft_field_destroy
    !---------------------------------------------
    !>  @brief Destroy a rocfft field struct
    !>
    !>  The field struct can be destroyed after being added to the plan description; it is not used
    !>  for
    !>  plan execution.
    !>
    !>   @warning Experimental!  This feature is part of an experimental API preview.
    function rocfft_field_destroy(field) &
       result(field_destroy) &
       bind(C, name="rocfft_field_destroy")
       import :: c_ptr, rocfft_status_success
       type(c_ptr), value :: field
       integer(kind(rocfft_status_success)) :: field_destroy
    end function rocfft_field_destroy

    !---------------------------------------------
    ! rocfft_get_version_string
    !---------------------------------------------
    !>  @brief Get library version string
    !>
    !>  @param[in, out] buf - buffer that receives the version string
    !>  @param[in] len - length of buf, minimum 30 characters
    function rocfft_get_version_string(buf, len) &
       result(get_version_string) &
       bind(C, name="rocfft_get_version_string")
       import :: c_ptr, c_size_t, rocfft_status_success
       type(c_ptr), value :: buf
       integer(c_size_t), value :: len
       integer(kind(rocfft_status_success)) :: get_version_string
    end function rocfft_get_version_string

    !---------------------------------------------
    ! rocfft_plan_description_set_comm
    !---------------------------------------------
    !>  @brief Set the communication library for distributed transforms.
    !>
    !>   @details Set the multi-processing communication library for a plan.
    !>
    !>   Multi-processing communication libraries require library-specific
    !>   handle to also be specified.  For MPI libraries, this is a
    !>   pointer to an MPI communicator.
    !>
    !>   @param[in] description - description handle
    !>   @param[in] comm_type - communicator type
    !>   @param[in] comm_handle - handle to communication-library-specific state
    function rocfft_plan_description_set_comm(description, comm_type, comm_handle) &
       result(plan_description_set_comm) &
       bind(C, name="rocfft_plan_description_set_comm")
       import :: c_ptr, rocfft_comm_none, rocfft_status_success
       type(c_ptr), value :: description
       integer(kind(rocfft_comm_none)), value :: comm_type
       type(c_ptr), value :: comm_handle
       integer(kind(rocfft_status_success)) :: plan_description_set_comm
    end function rocfft_plan_description_set_comm

    !---------------------------------------------
    ! rocfft_brick_create
    !---------------------------------------------
    !>  @brief Define a brick as part of a decomposition of a field.
    !>
    !>  Fields can contain a full-dimensional data distribution.  The
    !>  decomposition is specified by providing a lower coordinate and an
    !>  upper coordinate in the field's index space.  The lower coordinate
    !>  is inclusive (contained within the brick) and the upper coordinate
    !>  is exclusive (first index past the end of the brick).
    !>
    !>  One must also provide a stride for the brick data which specifies
    !>  how the brick's data is arranged in memory.
    !>
    !>  All coordinates and strides must include batch dimensions, and are in
    !>  column-major order (fastest-moving dimension first).
    !>
    !>  A HIP device ID is also provided - each brick may reside on a
    !>  different device.
    !>
    !>  All arrays may be re-used or freed immediately after the function returns.
    !>
    !>  @param[out] brick - : brick structure
    !>  @param[in] field_lower - : array of length `dim_with_batch` specifying the lower index
    !>  (inclusive) for the brick in the field's index space.
    !>  @param[in] field_upper - : array of length `dim_with_batch` specifying the upper index
    !>  (exclusive) for the brick in the field's index space.
    !>  @param[in] brick_stride - : array of length `dim_with_batch` specifying the brick's stride
    !>  in memory
    !>  @param[in] dim_with_batch - : length of the arrays; this must match the dimension of
    !>  the FFT plus one for the batch dimension.
    !>  @param[in] deviceID - : HIP device ID for the device on which the brick's data is resident.
    !>
    !>   @warning Experimental!  This feature is part of an experimental API preview.
    function rocfft_brick_create(brick, field_lower, field_upper, brick_stride, dim_with_batch, &
                                 deviceID) &
       result(brick_create) &
       bind(C, name="rocfft_brick_create")
       import :: c_ptr, c_size_t, c_int, rocfft_status_success
       type(c_ptr) :: brick
       type(c_ptr), value :: field_lower
       type(c_ptr), value :: field_upper
       type(c_ptr), value :: brick_stride
       integer(c_size_t), value :: dim_with_batch
       integer(c_int), value :: deviceID
       integer(kind(rocfft_status_success)) :: brick_create
    end function rocfft_brick_create

    !---------------------------------------------
    ! rocfft_brick_destroy
    !---------------------------------------------
    !>  @brief Deallocate a brick created with rocfft_brick_create.
    !>
    !>   @warning Experimental!  This feature is part of an experimental API preview.
    function rocfft_brick_destroy(brick) &
       result(brick_destroy) &
       bind(C, name="rocfft_brick_destroy")
       import :: c_ptr, rocfft_status_success
       type(c_ptr), value :: brick
       integer(kind(rocfft_status_success)) :: brick_destroy
    end function rocfft_brick_destroy

    !---------------------------------------------
    ! rocfft_field_add_brick
    !---------------------------------------------
    !>  @brief Add a brick to a field.
    !>
    !>  Note that the order in which the bricks are added is significant;
    !>  the pointers provided for each brick to `rocfft_execute` are in
    !>  the same order that the bricks were added to the field.
    !>
    !>  The brick may be added to another field or destroyed any time
    !>  after this function returns.
    !>
    !>  @param[in, out] field - : `rocfft_field` struct which holds the brick decomposition.
    !>  @param[in] brick - : `rocfft_brick` struct to add to the field.
    !>
    !>   @warning Experimental!  This feature is part of an experimental API preview.
    function rocfft_field_add_brick(field, brick) &
       result(field_add_brick) &
       bind(C, name="rocfft_field_add_brick")
       import :: c_ptr, rocfft_status_success
       type(c_ptr), value :: field
       type(c_ptr), value :: brick
       integer(kind(rocfft_status_success)) :: field_add_brick
    end function rocfft_field_add_brick

    !---------------------------------------------
    ! rocfft_plan_description_add_infield
    !---------------------------------------------
    !>  @brief Add a `rocfft_field` to a `rocfft_plan_description` as an input.
    !>
    !>  The field may be reused or freed immediately after the function returns.
    !>
    !>  @param[in, out] description - : `rocfft_plan_description` that will pass the field
    !>  information to plan creation
    !>  @param[in] field - : `rocfft_field` struct added as an input field
    !>
    !>   @warning Experimental!  This feature is part of an experimental API preview.
    function rocfft_plan_description_add_infield(description, field) &
       result(plan_description_add_infield) &
       bind(C, name="rocfft_plan_description_add_infield")
       import :: c_ptr, rocfft_status_success
       type(c_ptr), value :: description
       type(c_ptr), value :: field
       integer(kind(rocfft_status_success)) :: plan_description_add_infield
    end function rocfft_plan_description_add_infield

    !---------------------------------------------
    ! rocfft_plan_description_add_outfield
    !---------------------------------------------
    !>  @brief Add a `rocfft_field` to a `rocfft_plan_description` as an output.
    !>
    !>  The field may be reused or freed immediately after the function returns.
    !>
    !>  @param[in, out] description - : `rocfft_plan_description` that will pass the field
    !>  information to plan creation
    !>  @param[in] field - : `rocfft_field` struct added as an output field
    !>
    !>   @warning Experimental!  This feature is part of an experimental API preview.
    function rocfft_plan_description_add_outfield(description, field) &
       result(plan_description_add_outfield) &
       bind(C, name="rocfft_plan_description_add_outfield")
       import :: c_ptr, rocfft_status_success
       type(c_ptr), value :: description
       type(c_ptr), value :: field
       integer(kind(rocfft_status_success)) :: plan_description_add_outfield
    end function rocfft_plan_description_add_outfield

    !---------------------------------------------
    ! rocfft_plan_get_work_buffer_size
    !---------------------------------------------
    !>  @brief Get work buffer size on current HIP device
    !>   @details Get the work buffer size required for a plan on the current HIP device.
    !>
    !>   Work memory may be required on any device(s) with input or output
    !>   data for the transform, and also the current device when the plan
    !>   was created.  If the FFT plan uses multiple devices then this
    !>   function can be called repeatedly with each of those devices as
    !>   the current HIP device, to know the complete work memory
    !>   requirements for all devices.
    !>
    !>   @param[in] plan - plan handle
    !>   @param[out] size_in_bytes - size of needed work buffer in bytes
    function rocfft_plan_get_work_buffer_size(plan, size_in_bytes) &
       result(plan_get_work_buffer_size) &
       bind(C, name="rocfft_plan_get_work_buffer_size")
       import :: c_ptr, c_size_t, rocfft_status_success
       type(c_ptr), value :: plan
       integer(c_size_t) :: size_in_bytes
       integer(kind(rocfft_status_success)) :: plan_get_work_buffer_size
    end function rocfft_plan_get_work_buffer_size

    !---------------------------------------------
    ! rocfft_plan_get_print
    !---------------------------------------------
    !>  @brief Print all plan information
    !>   @details Prints plan details to stdout, to aid debugging
    !>   @param[in] plan - plan handle
    function rocfft_plan_get_print(plan) &
       result(plan_get_print) &
       bind(C, name="rocfft_plan_get_print")
       import :: c_ptr, rocfft_status_success
       type(c_ptr), value :: plan
       integer(kind(rocfft_status_success)) :: plan_get_print
    end function rocfft_plan_get_print

    !---------------------------------------------
    ! rocfft_plan_description_create
    !---------------------------------------------
    !>  @brief Create plan description
    !>   @details This API creates a plan description with which the user
    !>  can set extra plan properties.  The plan description must be freed
    !>  with a call to `rocfft_plan_description_destroy`.
    !>   @param[out] description - plan description handle
    function rocfft_plan_description_create(description) &
       result(plan_description_create) &
       bind(C, name="rocfft_plan_description_create")
       import :: c_ptr, rocfft_status_success
       type(c_ptr) :: description
       integer(kind(rocfft_status_success)) :: plan_description_create
    end function rocfft_plan_description_create

    !---------------------------------------------
    ! rocfft_plan_description_destroy
    !---------------------------------------------
    !>  @brief Destroy a plan description
    !>   @details This API frees the plan description.  A plan description
    !>   can be freed any time after it is passed to `rocfft_plan_create`.
    !>   @param[in] description - plan description handle
    function rocfft_plan_description_destroy(description) &
       result(plan_description_destroy) &
       bind(C, name="rocfft_plan_description_destroy")
       import :: c_ptr, rocfft_status_success
       type(c_ptr), value :: description
       integer(kind(rocfft_status_success)) :: plan_description_destroy
    end function rocfft_plan_description_destroy

    !---------------------------------------------
    ! rocfft_execution_info_create
    !---------------------------------------------
    !>  @brief Create execution info
    !>   @details This API creates an execution info with which the user
    !>  can control plan execution and work buffers.  The execution info must be freed
    !>  with a call to `rocfft_execution_info_destroy`.
    !>   @param[out] myInfo - execution info handle
    function rocfft_execution_info_create(myInfo) &
       result(execution_info_create) &
       bind(C, name="rocfft_execution_info_create")
       import :: c_ptr, rocfft_status_success
       type(c_ptr) :: myInfo
       integer(kind(rocfft_status_success)) :: execution_info_create
    end function rocfft_execution_info_create

    !---------------------------------------------
    ! rocfft_execution_info_destroy
    !---------------------------------------------
    !>  @brief Destroy an execution info
    !>   @details This API frees the execution info.  An execution info
    !>   object can be freed any time after it is passed to
    !>   `rocfft_execute`.
    !>   @param[in] myInfo - execution info handle
    function rocfft_execution_info_destroy(myInfo) &
       result(execution_info_destroy) &
       bind(C, name="rocfft_execution_info_destroy")
       import :: c_ptr, rocfft_status_success
       type(c_ptr), value :: myInfo
       integer(kind(rocfft_status_success)) :: execution_info_destroy
    end function rocfft_execution_info_destroy

    !---------------------------------------------
    ! rocfft_execution_info_set_work_buffer
    !---------------------------------------------
    !>  @brief Set work buffer in execution info for the current HIP device
    !>
    !>   @details This is one of the execution info functions to specify
    !>   optional additional information to control execution.  This API
    !>   provides a work buffer for the transform. It must be called
    !>   before `rocfft_execute`.
    !>
    !>   Work memory may be required on any device(s) with input or output
    !>   data for the transform, and also the current device when the plan
    !>   was created.  If the FFT plan uses multiple devices then this
    !>   function can be called repeatedly with each of those devices as
    !>   the current HIP device, to set work memory for all devices.
    !>
    !>   When a non-zero value is obtained from
    !>   `rocfft_plan_get_work_buffer_size`, that means the library needs a
    !>   work buffer to compute the transform. In this case, the user
    !>   should allocate the work buffer and pass it to the library via
    !>   this API.
    !>
    !>   If a work buffer is required for the transform but is not
    !>   specified using this function, `rocfft_execute` will automatically
    !>   allocate the required buffer and free it when execution is
    !>   finished.
    !>
    !>   Users should allocate their own work buffers if they need precise
    !>   control over the lifetimes of those buffers, or if multiple plans
    !>   need to share the same buffer.
    !>
    !>   @param[in] myInfo - execution info handle
    !>   @param[in] work_buffer - work buffer
    !>   @param[in] size_in_bytes - size of work buffer in bytes
    function rocfft_execution_info_set_work_buffer(myInfo, work_buffer, size_in_bytes) &
       result(execution_info_set_work_buffer) &
       bind(C, name="rocfft_execution_info_set_work_buffer")
       import :: c_ptr, c_size_t, rocfft_status_success
       type(c_ptr), value :: myInfo
       type(c_ptr), value :: work_buffer
       integer(c_size_t), value :: size_in_bytes
       integer(kind(rocfft_status_success)) :: execution_info_set_work_buffer
    end function rocfft_execution_info_set_work_buffer

    !---------------------------------------------
    ! rocfft_execution_info_set_stream
    !---------------------------------------------
    !>  @brief Set stream in execution info
    !>   @details Associates an existing compute stream to a plan.  This
    !>  must be called before the call to `rocfft_execute`.
    !>
    !>   Once the association is made, execution of the FFT will run the
    !>   computation through the specified stream.
    !>
    !>   The stream must be of type hipStream_t. It is an error to pass
    !>   the address of a hipStream_t object.
    !>
    !>   @param[in] myInfo - execution info handle
    !>   @param[in] stream - underlying compute stream
    function rocfft_execution_info_set_stream(myInfo, stream) &
       result(execution_info_set_stream) &
       bind(C, name="rocfft_execution_info_set_stream")
       import :: c_ptr, rocfft_status_success
       type(c_ptr), value :: myInfo
       type(c_ptr), value :: stream
       integer(kind(rocfft_status_success)) :: execution_info_set_stream
    end function rocfft_execution_info_set_stream

    !---------------------------------------------
    ! rocfft_execution_info_set_load_callback
    !---------------------------------------------
    !>  @brief Set a load callback for a plan execution (experimental)
    !>   @details This function specifies a user-defined callback function
    !>   that is run to load input from global memory at the start of the
    !>   transform.  Callbacks are an experimental feature in rocFFT.
    !>
    !>   Callback function pointers/data are given as arrays, with one
    !>   function/data pointer per brick in the input field of the plan.
    !>   Load callbacks require at least one brick in the input field to
    !>   be assigned to the current device used at plan creation. A plan
    !>   with no input field specified is considered to have one brick on
    !>   the current device used at plan creation.
    !>
    !>   All functions in the array must perform the same logical
    !>   operation.  That is, any function in the array must be
    !>   substitutable for any other function in the array if the data
    !>   being loaded were moved to another brick.
    !>
    !>   The provided function pointers replace any previously-specified
    !>   load callback for this execution info handle.
    !>
    !>   Load callbacks have the following signature:
    !>
    !>   @code
    !>   Tdata load_cb(Tdata* data, size_t offset, void* cbdata, void* sharedMem);
    !>   @endcode
    !>
    !>   'Tdata' is the type of a single element of the input buffer.  It is
    !>   the caller's responsibility to ensure that the function type is
    !>   appropriate for the plan (for example, a single-precision
    !>   real-to-complex transform would load single-precision real
    !>   elements).
    !>
    !>   A null value for 'cb_functions' may be specified to clear any
    !>   previously registered load callback.  'cb_data' may be null if
    !>   the functions require no additional pointer to be passed to them.
    !>
    !>   Currently, 'shared_mem_bytes' must be 0.  Callbacks are not
    !>   supported on transforms that use planar formats for either input
    !>   or output.
    !>
    !>   @param[in] myInfo - execution info handle
    !>   @param[in] cb_functions - callback function pointers
    !>   @param[in] cb_data - callback function data, passed to the function pointer when it is
    !>   called
    !>   @param[in] shared_mem_bytes - amount of shared memory to allocate for the callback function
    !>   to use
    function rocfft_execution_info_set_load_callback(myInfo, cb_functions, cb_data, &
                                                     shared_mem_bytes) &
       result(execution_info_set_load_callback) &
       bind(C, name="rocfft_execution_info_set_load_callback")
       import :: c_ptr, c_size_t, rocfft_status_success
       type(c_ptr), value :: myInfo
       type(c_ptr), value :: cb_functions
       type(c_ptr), value :: cb_data
       integer(c_size_t), value :: shared_mem_bytes
       integer(kind(rocfft_status_success)) :: execution_info_set_load_callback
    end function rocfft_execution_info_set_load_callback

    !---------------------------------------------
    ! rocfft_execution_info_set_store_callback
    !---------------------------------------------
    !>  @brief Set a store callback for a plan execution (experimental)
    !>   @details This function specifies a user-defined callback function
    !>   that is run to store output to global memory at the end of the
    !>   transform.  Callbacks are an experimental feature in rocFFT.
    !>
    !>   Callback function pointers/data are given as arrays, with one
    !>   function/data pointer per brick in the output field of the plan.
    !>   Store callbacks require at least one brick in the output field to
    !>   be assigned to the current device used at plan creation. A plan
    !>   with no output field specified is considered to have one brick on
    !>   the current device used at plan creation.
    !>
    !>   All functions in the array must perform the same logical
    !>   operation.  That is, any function in the array must be
    !>   substitutable for any other function in the array if the data
    !>   being stored were moved to another brick.
    !>
    !>   The provided function pointers replace any previously-specified
    !>   store callback for this execution info handle.
    !>
    !>   Store callbacks have the following signature:
    !>
    !>   @code
    !>   void store_cb(Tdata* data, size_t offset, Tdata element, void* cbdata, void* sharedMem);
    !>   @endcode
    !>
    !>   'Tdata' is the type of a single element of the output buffer.  It is
    !>   the caller's responsibility to ensure that the function type is
    !>   appropriate for the plan (for example, a single-precision
    !>   real-to-complex transform would store single-precision complex
    !>   elements).
    !>
    !>   A null value for 'cb_functions' may be specified to clear any
    !>   previously registered load callback.  'cb_data' may be null if
    !>   the functions require no additional pointer to be passed to them.
    !>
    !>   Currently, 'shared_mem_bytes' must be 0.  Callbacks are not
    !>   supported on transforms that use planar formats for either input
    !>   or output.
    !>
    !>   @param[in] myInfo - execution info handle
    !>   @param[in] cb_functions - callback function pointers
    !>   @param[in] cb_data - callback function data, passed to the function pointer when it is
    !>   called
    !>   @param[in] shared_mem_bytes - amount of shared memory to allocate for the callback function
    !>   to use
    function rocfft_execution_info_set_store_callback(myInfo, cb_functions, cb_data, &
                                                      shared_mem_bytes) &
       result(execution_info_set_store_callback) &
       bind(C, name="rocfft_execution_info_set_store_callback")
       import :: c_ptr, c_size_t, rocfft_status_success
       type(c_ptr), value :: myInfo
       type(c_ptr), value :: cb_functions
       type(c_ptr), value :: cb_data
       integer(c_size_t), value :: shared_mem_bytes
       integer(kind(rocfft_status_success)) :: execution_info_set_store_callback
    end function rocfft_execution_info_set_store_callback

    !---------------------------------------------
    ! rocfft_cache_serialize
    !---------------------------------------------
    !>  @brief Serialize compiled kernel cache
    !>
    !>   @details Serialize rocFFT's cache of compiled kernels into a
    !>   buffer.  This buffer is allocated by rocFFT and must be freed
    !>   with a call to `rocfft_cache_buffer_free`.  The length of the
    !>   buffer in bytes is written to 'buffer_len_bytes'.
    function rocfft_cache_serialize(buffer, buffer_len_bytes) &
       result(cache_serialize) &
       bind(C, name="rocfft_cache_serialize")
       import :: c_ptr, rocfft_status_success
       type(c_ptr) :: buffer
       type(c_ptr), value :: buffer_len_bytes
       integer(kind(rocfft_status_success)) :: cache_serialize
    end function rocfft_cache_serialize

    !---------------------------------------------
    ! rocfft_cache_buffer_free
    !---------------------------------------------
    !>  @brief Free cache serialization buffer
    !>
    !>   @details Deallocate a buffer allocated by `rocfft_cache_serialize`.
    function rocfft_cache_buffer_free(buffer) &
       result(cache_buffer_free) &
       bind(C, name="rocfft_cache_buffer_free")
       import :: c_ptr, rocfft_status_success
       type(c_ptr), value :: buffer
       integer(kind(rocfft_status_success)) :: cache_buffer_free
    end function rocfft_cache_buffer_free

    !---------------------------------------------
    ! rocfft_cache_deserialize
    !---------------------------------------------
    !>  @brief Deserialize a buffer into the compiled kernel cache.
    !>
    !>   @details Kernels in the buffer that match already-cached kernels
    !>   will replace those kernels that are in the cache.  Already-cached
    !>   kernels that do not match those in the buffer are unmodified by
    !>   this operation.  The cache is unmodified if either a null buffer
    !>   pointer or a zero length is passed.
    function rocfft_cache_deserialize(buffer, buffer_len_bytes) &
       result(cache_deserialize) &
       bind(C, name="rocfft_cache_deserialize")
       import :: c_ptr, c_size_t, rocfft_status_success
       type(c_ptr), value :: buffer
       integer(c_size_t), value :: buffer_len_bytes
       integer(kind(rocfft_status_success)) :: cache_deserialize
    end function rocfft_cache_deserialize

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
  !>   The plan must be destroyed with a call to `rocfft_plan_destroy`.
  !>
  !>   @param[out] plan - plan handle
  !>   @param[in] placement - placement of result
  !>   @param[in] transform_type - type of transform
  !>   @param[in] myPrecision - precision
  !>   @param[in] dimensions - dimensions
  !>   @param[in] lengths - dimensions-sized array of transform lengths
  !>   @param[in] number_of_transforms - number of transforms
  !>   @param[in] description - description handle created by
  !>  rocfft_plan_description_create; can be
  !>   NULL for simple transforms
  interface rocfft_plan_create
    function rocfft_plan_create_(plan,placement,transform_type,myPrecision,dimensions,lengths, &
        number_of_transforms,description) &
        bind(c, name="rocfft_plan_create")
      use iso_c_binding
      import
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
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure rocfft_plan_create_assumed_rank
#else
    module procedure &
      rocfft_plan_create_rank_0,&
      rocfft_plan_create_rank_1
#endif
#endif
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
  !>   * Stride between consecutive elements in each dimension. Defaults to packed data
  !>       layout consistent with the type of transform and its placement (requested at
  !>       plan creation), if unspecified.
  !>   * Distance between consecutive batches. Zero values are interpreted as defaults
  !>       to be deduced from the corresponding length and stride along the last transform
  !>       dimension.
  !>
  !>   Not all combinations of array types are supported and error codes
  !>   will be returned for unsupported cases.
  !>
  !>   Offset, stride, and distance for either input or output provided
  !>   here is ignored if a field is set for the corresponding input or
  !>   output.
  !>   @note Non-zero offsets are not supported yet.
  !>
  !>   @param[in, out] description - description handle
  !>   @param[in] in_array_type - array type of input buffer
  !>   @param[in] out_array_type - array type of output buffer
  !>   @param[in] in_offsets - offsets, in element units, to start of data in input buffer
  !>   @param[in] out_offsets - offsets, in element units, to start of data in output buffer
  !>   @param[in] in_strides_size - size of in_strides array (must be equal to transform dimensions)
  !>   @param[in] in_strides - array of strides, in each dimension, of
  !>    input buffer; if set to null ptr library chooses defaults
  !>   @param[in] in_distance - distance between start of each data instance in input buffer
  !>   @param[in] out_strides_size - size of out_strides array (must be
  !>   equal to transform dimensions)
  !>   @param[in] out_strides - array of strides, in each dimension, of
  !>    output buffer; if set to null ptr library chooses defaults
  !>   @param[in] out_distance - distance between start of each data instance in output buffer
  interface rocfft_plan_description_set_data_layout
    function rocfft_plan_description_set_data_layout_(description,in_array_type,out_array_type, &
        in_offsets,out_offsets,in_strides_size,in_strides,in_distance,out_strides_size, &
        out_strides,out_distance) &
        bind(c, name="rocfft_plan_description_set_data_layout")
      use iso_c_binding
      import
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
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure rocfft_plan_description_set_data_layout_assumed_rank
#else
    module procedure &
      rocfft_plan_description_set_data_layout_rank_0,&
      rocfft_plan_description_set_data_layout_rank_1
#endif
#endif
  end interface


  contains

    subroutine rocfftCheck(status)
      implicit none
      integer(kind(rocfft_status_success)) :: status
      if (status /= rocfft_status_success) then
        write (*, *) "ROCFFT ERROR: code = ", status
        stop 1
      end if
    end subroutine rocfftCheck

#ifdef USE_FPOINTER_INTERFACES

#ifdef USE_ASSUMED_RANK_INTERFACES
    function rocfft_plan_create_assumed_rank(plan,placement,transform_type,myPrecision,dimensions, &
        lengths,number_of_transforms,description)
      use iso_c_binding
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_create_assumed_rank
      type(c_ptr) :: plan
      integer(kind(rocfft_placement_inplace)) :: placement
      integer(kind(rocfft_transform_type_complex_forward)) :: transform_type
      integer(kind(rocfft_precision_single)) :: myPrecision
      integer(c_size_t) :: dimensions
      integer(c_size_t),target,contiguous,dimension(..) :: lengths
      integer(c_size_t) :: number_of_transforms
      type(c_ptr) :: description
      !
      rocfft_plan_create_assumed_rank = rocfft_plan_create_(plan,placement,transform_type, &
        myPrecision,dimensions,c_loc(lengths),number_of_transforms,description)
    end function

#else
    function rocfft_plan_create_rank_0(plan,placement,transform_type,myPrecision,dimensions, &
        lengths,number_of_transforms,description)
      use iso_c_binding
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
      rocfft_plan_create_rank_0 = rocfft_plan_create_(plan,placement,transform_type,myPrecision, &
        dimensions,c_loc(lengths),number_of_transforms,description)
    end function

    function rocfft_plan_create_rank_1(plan,placement,transform_type,myPrecision,dimensions, &
        lengths,number_of_transforms,description)
      use iso_c_binding
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
      rocfft_plan_create_rank_1 = rocfft_plan_create_(plan,placement,transform_type,myPrecision, &
        dimensions,c_loc(lengths),number_of_transforms,description)
    end function

#endif
#ifdef USE_ASSUMED_RANK_INTERFACES
    function rocfft_plan_description_set_data_layout_assumed_rank(description,in_array_type, &
        out_array_type,in_offsets,out_offsets,in_strides_size,in_strides,in_distance, &
        out_strides_size,out_strides,out_distance)
      use iso_c_binding
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_description_set_data_layout_assumed_rank
      type(c_ptr) :: description
      integer(kind(rocfft_array_type_complex_interleaved)) :: in_array_type
      integer(kind(rocfft_array_type_complex_interleaved)) :: out_array_type
      integer(c_size_t),target,contiguous,dimension(..) :: in_offsets
      integer(c_size_t),target,contiguous,dimension(..) :: out_offsets
      integer(c_size_t) :: in_strides_size
      integer(c_size_t),target,contiguous,dimension(..) :: in_strides
      integer(c_size_t) :: in_distance
      integer(c_size_t) :: out_strides_size
      integer(c_size_t),target,contiguous,dimension(..) :: out_strides
      integer(c_size_t) :: out_distance
      !
      rocfft_plan_description_set_data_layout_assumed_rank = &
        rocfft_plan_description_set_data_layout_(description,in_array_type,out_array_type, &
        c_loc(in_offsets),c_loc(out_offsets),in_strides_size,c_loc(in_strides),in_distance, &
        out_strides_size,c_loc(out_strides),out_distance)
    end function

#else
    function rocfft_plan_description_set_data_layout_rank_0(description,in_array_type, &
        out_array_type,in_offsets,out_offsets,in_strides_size,in_strides,in_distance, &
        out_strides_size,out_strides,out_distance)
      use iso_c_binding
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
      rocfft_plan_description_set_data_layout_rank_0 = rocfft_plan_description_set_data_layout_( &
        description,in_array_type,out_array_type,c_loc(in_offsets),c_loc(out_offsets), &
        in_strides_size,c_loc(in_strides),in_distance,out_strides_size,c_loc(out_strides), &
        out_distance)
    end function

    function rocfft_plan_description_set_data_layout_rank_1(description,in_array_type, &
        out_array_type,in_offsets,out_offsets,in_strides_size,in_strides,in_distance, &
        out_strides_size,out_strides,out_distance)
      use iso_c_binding
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
      rocfft_plan_description_set_data_layout_rank_1 = rocfft_plan_description_set_data_layout_( &
        description,in_array_type,out_array_type,c_loc(in_offsets),c_loc(out_offsets), &
        in_strides_size,c_loc(in_strides),in_distance,out_strides_size,c_loc(out_strides), &
        out_distance)
    end function

#endif
#endif
end module rocfft
