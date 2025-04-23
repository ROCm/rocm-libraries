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
          
           
module hipfort_rocrand_enums
  implicit none

  enum, bind(c)
    enumerator :: ROCRAND_STATUS_SUCCESS = 0
    enumerator :: ROCRAND_STATUS_VERSION_MISMATCH = 100
    enumerator :: ROCRAND_STATUS_NOT_CREATED = 101
    enumerator :: ROCRAND_STATUS_ALLOCATION_FAILED = 102
    enumerator :: ROCRAND_STATUS_TYPE_ERROR = 103
    enumerator :: ROCRAND_STATUS_OUT_OF_RANGE = 104
    enumerator :: ROCRAND_STATUS_LENGTH_NOT_MULTIPLE = 105
    enumerator :: ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106
    enumerator :: ROCRAND_STATUS_LAUNCH_FAILURE = 107
    enumerator :: ROCRAND_STATUS_INTERNAL_ERROR = 108
  end enum

  enum, bind(c)
    enumerator :: ROCRAND_RNG_PSEUDO_DEFAULT = 400
    enumerator :: ROCRAND_RNG_PSEUDO_XORWOW = 401
    enumerator :: ROCRAND_RNG_PSEUDO_MRG32K3A = 402
    enumerator :: ROCRAND_RNG_PSEUDO_MTGP32 = 403
    enumerator :: ROCRAND_RNG_PSEUDO_PHILOX4_32_10 = 404
    enumerator :: ROCRAND_RNG_QUASI_DEFAULT = 500
    enumerator :: ROCRAND_RNG_QUASI_SOBOL32 = 501
    enumerator :: ROCRAND_RNG_QUASI_SOBOL64 = 504
  end enum

 

#ifdef USE_FPOINTER_INTERFACES

  
#endif
end module hipfort_rocrand_enums