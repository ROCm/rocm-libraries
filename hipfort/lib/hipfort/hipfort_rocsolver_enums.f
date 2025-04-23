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
          
           
module hipfort_rocsolver_enums
  implicit none

  enum, bind(c)
    enumerator :: rocblas_layer_mode_ex_log_kernel = 16
  end enum

  enum, bind(c)
    enumerator :: rocblas_forward_direction = 171
    enumerator :: rocblas_backward_direction = 172
  end enum

  enum, bind(c)
    enumerator :: rocblas_column_wise = 181
    enumerator :: rocblas_row_wise = 182
  end enum

  enum, bind(c)
    enumerator :: rocblas_svect_all = 191
    enumerator :: rocblas_svect_singular = 192
    enumerator :: rocblas_svect_overwrite = 193
    enumerator :: rocblas_svect_none = 194
  end enum

  enum, bind(c)
    enumerator :: rocblas_outofplace = 201
    enumerator :: rocblas_inplace = 202
  end enum

  enum, bind(c)
    enumerator :: rocblas_evect_original = 211
    enumerator :: rocblas_evect_tridiagonal = 212
    enumerator :: rocblas_evect_none = 213
  end enum

  enum, bind(c)
    enumerator :: rocblas_eform_ax = 221
    enumerator :: rocblas_eform_abx = 222
    enumerator :: rocblas_eform_bax = 223
  end enum

 

#ifdef USE_FPOINTER_INTERFACES

  
#endif
end module hipfort_rocsolver_enums