!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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


  interface rocfft_setup
    function rocfft_setup_() bind(c, name="rocfft_setup")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_setup_
    end function
  end interface

  interface rocfft_cleanup
    function rocfft_cleanup_() bind(c, name="rocfft_cleanup")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_cleanup_
    end function
  end interface

  interface rocfft_plan_create
    function rocfft_plan_create_(plan,placement,transform_type,myPrecision,dimensions,lengths, &
        number_of_transforms,description) &
        bind(c, name="rocfft_plan_create")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_plan_create_
      type(c_ptr) :: plan
      integer(c_int),value :: placement
      integer(c_int),value :: transform_type
      integer(c_int),value :: myPrecision
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

  interface rocfft_execute
    function rocfft_execute_(plan,in_buffer,out_buffer,myInfo) bind(c, name="rocfft_execute")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_execute_
      type(c_ptr),value :: plan
      type(c_ptr) :: in_buffer
      type(c_ptr) :: out_buffer
      type(c_ptr),value :: myInfo
    end function
  end interface

  interface rocfft_plan_destroy
    function rocfft_plan_destroy_(plan) bind(c, name="rocfft_plan_destroy")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_plan_destroy_
      type(c_ptr),value :: plan
    end function
  end interface

  interface rocfft_plan_description_set_scale_factor
    function rocfft_plan_description_set_scale_factor_(description,scale_factor) &
        bind(c, name="rocfft_plan_description_set_scale_factor")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_plan_description_set_scale_factor_
      type(c_ptr),value :: description
      real(c_double),value :: scale_factor
    end function
  end interface

  interface rocfft_plan_description_set_data_layout
    function rocfft_plan_description_set_data_layout_(description,in_array_type,out_array_type, &
        in_offsets,out_offsets,in_strides_size,in_strides,in_distance,out_strides_size, &
        out_strides,out_distance) &
        bind(c, name="rocfft_plan_description_set_data_layout")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_plan_description_set_data_layout_
      type(c_ptr),value :: description
      integer(c_int),value :: in_array_type
      integer(c_int),value :: out_array_type
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

  interface rocfft_field_create
    function rocfft_field_create_(field) bind(c, name="rocfft_field_create")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_field_create_
      type(c_ptr) :: field
    end function
  end interface

  interface rocfft_field_destroy
    function rocfft_field_destroy_(field) bind(c, name="rocfft_field_destroy")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_field_destroy_
      type(c_ptr),value :: field
    end function
  end interface

  interface rocfft_get_version_string
    function rocfft_get_version_string_(buf,len) bind(c, name="rocfft_get_version_string")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_get_version_string_
      type(c_ptr),value :: buf
      integer(c_size_t),value :: len
    end function
  end interface

  interface rocfft_plan_description_set_comm
    function rocfft_plan_description_set_comm_(description,comm_type,comm_handle) &
        bind(c, name="rocfft_plan_description_set_comm")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_plan_description_set_comm_
      type(c_ptr),value :: description
      integer(c_int),value :: comm_type
      type(c_ptr),value :: comm_handle
    end function
  end interface

  interface rocfft_plan_description_set_load_callback
    function rocfft_plan_description_set_load_callback_(description,symbol_name,bitcode_data, &
        bitcode_len_bytes,shared_mem_bytes) &
        bind(c, name="rocfft_plan_description_set_load_callback")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_plan_description_set_load_callback_
      type(c_ptr),value :: description
      type(c_ptr),value :: symbol_name
      type(c_ptr),value :: bitcode_data
      integer(c_size_t),value :: bitcode_len_bytes
      integer(c_size_t),value :: shared_mem_bytes
    end function
  end interface

  interface rocfft_plan_description_set_store_callback
    function rocfft_plan_description_set_store_callback_(description,symbol_name,bitcode_data, &
        bitcode_len_bytes,shared_mem_bytes) &
        bind(c, name="rocfft_plan_description_set_store_callback")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_plan_description_set_store_callback_
      type(c_ptr),value :: description
      type(c_ptr),value :: symbol_name
      type(c_ptr),value :: bitcode_data
      integer(c_size_t),value :: bitcode_len_bytes
      integer(c_size_t),value :: shared_mem_bytes
    end function
  end interface

  interface rocfft_brick_create
    function rocfft_brick_create_(brick,field_lower,field_upper,brick_stride,dim_with_batch, &
        deviceID) &
        bind(c, name="rocfft_brick_create")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_brick_create_
      type(c_ptr) :: brick
      type(c_ptr),value :: field_lower
      type(c_ptr),value :: field_upper
      type(c_ptr),value :: brick_stride
      integer(c_size_t),value :: dim_with_batch
      integer(c_int),value :: deviceID
    end function
  end interface

  interface rocfft_brick_destroy
    function rocfft_brick_destroy_(brick) bind(c, name="rocfft_brick_destroy")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_brick_destroy_
      type(c_ptr),value :: brick
    end function
  end interface

  interface rocfft_field_add_brick
    function rocfft_field_add_brick_(field,brick) bind(c, name="rocfft_field_add_brick")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_field_add_brick_
      type(c_ptr),value :: field
      type(c_ptr),value :: brick
    end function
  end interface

  interface rocfft_plan_description_add_infield
    function rocfft_plan_description_add_infield_(description,field) &
        bind(c, name="rocfft_plan_description_add_infield")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_plan_description_add_infield_
      type(c_ptr),value :: description
      type(c_ptr),value :: field
    end function
  end interface

  interface rocfft_plan_description_add_outfield
    function rocfft_plan_description_add_outfield_(description,field) &
        bind(c, name="rocfft_plan_description_add_outfield")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_plan_description_add_outfield_
      type(c_ptr),value :: description
      type(c_ptr),value :: field
    end function
  end interface

  interface rocfft_plan_get_work_buffer_size
    function rocfft_plan_get_work_buffer_size_(plan,size_in_bytes) &
        bind(c, name="rocfft_plan_get_work_buffer_size")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_plan_get_work_buffer_size_
      type(c_ptr),value :: plan
      type(c_ptr),value :: size_in_bytes
    end function

    module procedure rocfft_plan_get_work_buffer_size_typed
  end interface

  interface rocfft_plan_get_print
    function rocfft_plan_get_print_(plan) bind(c, name="rocfft_plan_get_print")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_plan_get_print_
      type(c_ptr),value :: plan
    end function
  end interface

  interface rocfft_plan_description_create
    function rocfft_plan_description_create_(description) &
        bind(c, name="rocfft_plan_description_create")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_plan_description_create_
      type(c_ptr) :: description
    end function
  end interface

  interface rocfft_plan_description_destroy
    function rocfft_plan_description_destroy_(description) &
        bind(c, name="rocfft_plan_description_destroy")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_plan_description_destroy_
      type(c_ptr),value :: description
    end function
  end interface

  interface rocfft_execution_info_create
    function rocfft_execution_info_create_(myInfo) bind(c, name="rocfft_execution_info_create")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_execution_info_create_
      type(c_ptr) :: myInfo
    end function
  end interface

  interface rocfft_execution_info_destroy
    function rocfft_execution_info_destroy_(myInfo) bind(c, name="rocfft_execution_info_destroy")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_execution_info_destroy_
      type(c_ptr),value :: myInfo
    end function
  end interface

  interface rocfft_execution_info_set_work_buffer
    function rocfft_execution_info_set_work_buffer_(myInfo,work_buffer,size_in_bytes) &
        bind(c, name="rocfft_execution_info_set_work_buffer")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_execution_info_set_work_buffer_
      type(c_ptr),value :: myInfo
      type(c_ptr),value :: work_buffer
      integer(c_size_t),value :: size_in_bytes
    end function
  end interface

  interface rocfft_execution_info_set_stream
    function rocfft_execution_info_set_stream_(myInfo,stream) &
        bind(c, name="rocfft_execution_info_set_stream")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_execution_info_set_stream_
      type(c_ptr),value :: myInfo
      type(c_ptr),value :: stream
    end function
  end interface

  interface rocfft_execution_info_set_load_callback
    function rocfft_execution_info_set_load_callback_(myInfo,cb_functions,cb_data, &
        shared_mem_bytes) &
        bind(c, name="rocfft_execution_info_set_load_callback")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_execution_info_set_load_callback_
      type(c_ptr),value :: myInfo
      type(c_ptr),value :: cb_functions
      type(c_ptr),value :: cb_data
      integer(c_size_t),value :: shared_mem_bytes
    end function
  end interface

  interface rocfft_execution_info_set_load_callback_data
    function rocfft_execution_info_set_load_callback_data_(myInfo,cb_data,count) &
        bind(c, name="rocfft_execution_info_set_load_callback_data")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_execution_info_set_load_callback_data_
      type(c_ptr),value :: myInfo
      type(c_ptr),value :: cb_data
      integer(c_size_t),value :: count
    end function
  end interface

  interface rocfft_execution_info_set_store_callback
    function rocfft_execution_info_set_store_callback_(myInfo,cb_functions,cb_data, &
        shared_mem_bytes) &
        bind(c, name="rocfft_execution_info_set_store_callback")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_execution_info_set_store_callback_
      type(c_ptr),value :: myInfo
      type(c_ptr),value :: cb_functions
      type(c_ptr),value :: cb_data
      integer(c_size_t),value :: shared_mem_bytes
    end function
  end interface

  interface rocfft_execution_info_set_store_callback_data
    function rocfft_execution_info_set_store_callback_data_(myInfo,cb_data,count) &
        bind(c, name="rocfft_execution_info_set_store_callback_data")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_execution_info_set_store_callback_data_
      type(c_ptr),value :: myInfo
      type(c_ptr),value :: cb_data
      integer(c_size_t),value :: count
    end function
  end interface

  interface rocfft_cache_serialize
    function rocfft_cache_serialize_(buffer,buffer_len_bytes) bind(c, name="rocfft_cache_serialize")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_cache_serialize_
      type(c_ptr) :: buffer
      type(c_ptr),value :: buffer_len_bytes
    end function

    module procedure rocfft_cache_serialize_typed
  end interface

  interface rocfft_cache_buffer_free
    function rocfft_cache_buffer_free_(buffer) bind(c, name="rocfft_cache_buffer_free")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_cache_buffer_free_
      type(c_ptr),value :: buffer
    end function
  end interface

  interface rocfft_cache_deserialize
    function rocfft_cache_deserialize_(buffer,buffer_len_bytes) &
        bind(c, name="rocfft_cache_deserialize")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocfft_cache_deserialize_
      type(c_ptr),value :: buffer
      integer(c_size_t),value :: buffer_len_bytes
    end function
  end interface


  contains

    subroutine rocfftCheck(status)
      implicit none
      integer(kind(rocfft_status_success)) :: status
      character(len=:), allocatable :: name
      if (status /= rocfft_status_success) then
        select case (status)
        case (rocfft_status_failure)
          name = "rocfft_status_failure"
        case (rocfft_status_invalid_arg_value)
          name = "rocfft_status_invalid_arg_value"
        case (rocfft_status_invalid_dimensions)
          name = "rocfft_status_invalid_dimensions"
        case (rocfft_status_invalid_array_type)
          name = "rocfft_status_invalid_array_type"
        case (rocfft_status_invalid_strides)
          name = "rocfft_status_invalid_strides"
        case (rocfft_status_invalid_distance)
          name = "rocfft_status_invalid_distance"
        case (rocfft_status_invalid_offset)
          name = "rocfft_status_invalid_offset"
        case (rocfft_status_invalid_work_buffer)
          name = "rocfft_status_invalid_work_buffer"
        case default
          name = "unknown status"
        end select
        write (*, "(a,i0,a)") "ROCFFT ERROR: " // name // " (code ", status, ")"
        stop 1
      end if
    end subroutine rocfftCheck
    function rocfft_plan_get_work_buffer_size_typed(plan,size_in_bytes)
      use iso_c_binding
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_plan_get_work_buffer_size_typed
      type(c_ptr), value :: plan
      integer(c_size_t),target :: size_in_bytes
      !
      rocfft_plan_get_work_buffer_size_typed = rocfft_plan_get_work_buffer_size_(plan, &
        c_loc(size_in_bytes))
    end function

    function rocfft_cache_serialize_typed(buffer,buffer_len_bytes)
      use iso_c_binding
      implicit none
      integer(kind(rocfft_status_success)) :: rocfft_cache_serialize_typed
      type(c_ptr) :: buffer
      integer(c_size_t),target :: buffer_len_bytes
      !
      rocfft_cache_serialize_typed = rocfft_cache_serialize_(buffer,c_loc(buffer_len_bytes))
    end function


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
      integer(c_size_t),target,dimension(..) :: lengths
      integer(c_size_t) :: number_of_transforms
      type(c_ptr), value :: description
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
      type(c_ptr), value :: description
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
      type(c_ptr), value :: description
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
      type(c_ptr), value :: description
      integer(kind(rocfft_array_type_complex_interleaved)) :: in_array_type
      integer(kind(rocfft_array_type_complex_interleaved)) :: out_array_type
      integer(c_size_t),target,dimension(..) :: in_offsets
      integer(c_size_t),target,dimension(..) :: out_offsets
      integer(c_size_t) :: in_strides_size
      integer(c_size_t),target,dimension(..) :: in_strides
      integer(c_size_t) :: in_distance
      integer(c_size_t) :: out_strides_size
      integer(c_size_t),target,dimension(..) :: out_strides
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
      type(c_ptr), value :: description
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
      type(c_ptr), value :: description
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
