!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
!
! SPDX-License-Identifier: MIT
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
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
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
    function rocfft_setup() &
       result(setup) &
       bind(C, name="rocfft_setup")
       import :: rocfft_status_success
       integer(kind(rocfft_status_success)) :: setup
    end function rocfft_setup

    !---------------------------------------------
    ! rocfft_cleanup
    !---------------------------------------------
    function rocfft_cleanup() &
       result(cleanup) &
       bind(C, name="rocfft_cleanup")
       import :: rocfft_status_success
       integer(kind(rocfft_status_success)) :: cleanup
    end function rocfft_cleanup

    !---------------------------------------------
    ! rocfft_execute
    !---------------------------------------------
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
    ! rocfft_plan_description_set_load_callback
    !---------------------------------------------
    function rocfft_plan_description_set_load_callback(description, symbol_name, bitcode_data, &
                                                       bitcode_len_bytes, shared_mem_bytes) &
       result(plan_description_set_load_callback) &
       bind(C, name="rocfft_plan_description_set_load_callback")
       import :: c_ptr, c_size_t, rocfft_status_success
       type(c_ptr), value :: description
       type(c_ptr), value :: symbol_name
       type(c_ptr), value :: bitcode_data
       integer(c_size_t), value :: bitcode_len_bytes
       integer(c_size_t), value :: shared_mem_bytes
       integer(kind(rocfft_status_success)) :: plan_description_set_load_callback
    end function rocfft_plan_description_set_load_callback

    !---------------------------------------------
    ! rocfft_plan_description_set_store_callback
    !---------------------------------------------
    function rocfft_plan_description_set_store_callback(description, symbol_name, bitcode_data, &
                                                        bitcode_len_bytes, shared_mem_bytes) &
       result(plan_description_set_store_callback) &
       bind(C, name="rocfft_plan_description_set_store_callback")
       import :: c_ptr, c_size_t, rocfft_status_success
       type(c_ptr), value :: description
       type(c_ptr), value :: symbol_name
       type(c_ptr), value :: bitcode_data
       integer(c_size_t), value :: bitcode_len_bytes
       integer(c_size_t), value :: shared_mem_bytes
       integer(kind(rocfft_status_success)) :: plan_description_set_store_callback
    end function rocfft_plan_description_set_store_callback

    !---------------------------------------------
    ! rocfft_brick_create
    !---------------------------------------------
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
    ! rocfft_execution_info_set_load_callback_data
    !---------------------------------------------
    function rocfft_execution_info_set_load_callback_data(myInfo, cb_data, count) &
       result(execution_info_set_load_callback_data) &
       bind(C, name="rocfft_execution_info_set_load_callback_data")
       import :: c_ptr, c_size_t, rocfft_status_success
       type(c_ptr), value :: myInfo
       type(c_ptr), value :: cb_data
       integer(c_size_t), value :: count
       integer(kind(rocfft_status_success)) :: execution_info_set_load_callback_data
    end function rocfft_execution_info_set_load_callback_data

    !---------------------------------------------
    ! rocfft_execution_info_set_store_callback
    !---------------------------------------------
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
    ! rocfft_execution_info_set_store_callback_data
    !---------------------------------------------
    function rocfft_execution_info_set_store_callback_data(myInfo, cb_data, count) &
       result(execution_info_set_store_callback_data) &
       bind(C, name="rocfft_execution_info_set_store_callback_data")
       import :: c_ptr, c_size_t, rocfft_status_success
       type(c_ptr), value :: myInfo
       type(c_ptr), value :: cb_data
       integer(c_size_t), value :: count
       integer(kind(rocfft_status_success)) :: execution_info_set_store_callback_data
    end function rocfft_execution_info_set_store_callback_data

    !---------------------------------------------
    ! rocfft_cache_serialize
    !---------------------------------------------
    function rocfft_cache_serialize(buffer, buffer_len_bytes) &
       result(cache_serialize) &
       bind(C, name="rocfft_cache_serialize")
       import :: c_ptr, c_size_t, rocfft_status_success
       type(c_ptr) :: buffer
       integer(c_size_t) :: buffer_len_bytes
       integer(kind(rocfft_status_success)) :: cache_serialize
    end function rocfft_cache_serialize

    !---------------------------------------------
    ! rocfft_cache_buffer_free
    !---------------------------------------------
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
    function rocfft_cache_deserialize(buffer, buffer_len_bytes) &
       result(cache_deserialize) &
       bind(C, name="rocfft_cache_deserialize")
       import :: c_ptr, c_size_t, rocfft_status_success
       type(c_ptr), value :: buffer
       integer(c_size_t), value :: buffer_len_bytes
       integer(kind(rocfft_status_success)) :: cache_deserialize
    end function rocfft_cache_deserialize

  end interface

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

#ifdef USE_ASSUMED_RANK
    module procedure rocfft_plan_create_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      rocfft_plan_create_rank_0,&
      rocfft_plan_create_rank_1
#endif
#endif
  end interface

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

#ifdef USE_ASSUMED_RANK
    module procedure rocfft_plan_description_set_data_layout_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
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

#if defined(USE_ASSUMED_SHAPE) || defined(USE_ASSUMED_RANK)

#ifdef USE_ASSUMED_RANK
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
#ifdef USE_ASSUMED_RANK
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
