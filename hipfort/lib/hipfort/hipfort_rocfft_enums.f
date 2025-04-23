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
          
           
module hipfort_rocfft_enums
  implicit none

  enum, bind(c)
    enumerator :: rocfft_status_success
    enumerator :: rocfft_status_failure
    enumerator :: rocfft_status_invalid_arg_value
    enumerator :: rocfft_status_invalid_dimensions
    enumerator :: rocfft_status_invalid_array_type
    enumerator :: rocfft_status_invalid_strides
    enumerator :: rocfft_status_invalid_distance
    enumerator :: rocfft_status_invalid_offset
    enumerator :: rocfft_status_invalid_work_buffer
  end enum

  enum, bind(c)
    enumerator :: rocfft_transform_type_complex_forward
    enumerator :: rocfft_transform_type_complex_inverse
    enumerator :: rocfft_transform_type_real_forward
    enumerator :: rocfft_transform_type_real_inverse
  end enum

  enum, bind(c)
    enumerator :: rocfft_precision_single
    enumerator :: rocfft_precision_double
  end enum

  enum, bind(c)
    enumerator :: rocfft_placement_inplace
    enumerator :: rocfft_placement_notinplace
  end enum

  enum, bind(c)
    enumerator :: rocfft_array_type_complex_interleaved
    enumerator :: rocfft_array_type_complex_planar
    enumerator :: rocfft_array_type_real
    enumerator :: rocfft_array_type_hermitian_interleaved
    enumerator :: rocfft_array_type_hermitian_planar
    enumerator :: rocfft_array_type_unset
  end enum

  enum, bind(c)
    enumerator :: rocfft_exec_mode_nonblocking
    enumerator :: rocfft_exec_mode_nonblocking_with_flush
    enumerator :: rocfft_exec_mode_blocking
  end enum

 

#ifdef USE_FPOINTER_INTERFACES

  
#endif
end module hipfort_rocfft_enums