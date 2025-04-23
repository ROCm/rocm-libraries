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
          
           
module hipfort_hipsolver_enums
  implicit none

  enum, bind(c)
    enumerator :: HIPSOLVER_STATUS_SUCCESS = 0
    enumerator :: HIPSOLVER_STATUS_NOT_INITIALIZED = 1
    enumerator :: HIPSOLVER_STATUS_ALLOC_FAILED = 2
    enumerator :: HIPSOLVER_STATUS_INVALID_VALUE = 3
    enumerator :: HIPSOLVER_STATUS_MAPPING_ERROR = 4
    enumerator :: HIPSOLVER_STATUS_EXECUTION_FAILED = 5
    enumerator :: HIPSOLVER_STATUS_INTERNAL_ERROR = 6
    enumerator :: HIPSOLVER_STATUS_NOT_SUPPORTED = 7
    enumerator :: HIPSOLVER_STATUS_ARCH_MISMATCH = 8
    enumerator :: HIPSOLVER_STATUS_HANDLE_IS_NULLPTR = 9
    enumerator :: HIPSOLVER_STATUS_INVALID_ENUM = 10
    enumerator :: HIPSOLVER_STATUS_UNKNOWN = 11
  end enum

  enum, bind(c)
    enumerator :: HIPSOLVER_OP_N = 111
    enumerator :: HIPSOLVER_OP_T = 112
    enumerator :: HIPSOLVER_OP_C = 113
  end enum

  enum, bind(c)
    enumerator :: HIPSOLVER_FILL_MODE_UPPER = 121
    enumerator :: HIPSOLVER_FILL_MODE_LOWER = 122
    enumerator :: HIPSOLVER_FILL_MODE_FULL = 123
  end enum

  enum, bind(c)
    enumerator :: HIPSOLVER_SIDE_LEFT = 141
    enumerator :: HIPSOLVER_SIDE_RIGHT = 142
  end enum

  enum, bind(c)
    enumerator :: HIPSOLVER_EIG_MODE_NOVECTOR = 201
    enumerator :: HIPSOLVER_EIG_MODE_VECTOR = 202
  end enum

  enum, bind(c)
    enumerator :: HIPSOLVER_EIG_TYPE_1 = 211
    enumerator :: HIPSOLVER_EIG_TYPE_2 = 212
    enumerator :: HIPSOLVER_EIG_TYPE_3 = 213
  end enum

  enum, bind(c)
    enumerator :: HIPSOLVER_EIG_RANGE_ALL = 221
    enumerator :: HIPSOLVER_EIG_RANGE_V = 222
    enumerator :: HIPSOLVER_EIG_RANGE_I = 223
  end enum

  enum, bind(c)
    enumerator :: HIPSOLVER_DETERMINISTIC_RESULTS = 241
    enumerator :: HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS = 242
  end enum

  enum, bind(c)
    enumerator :: HIPSOLVER_ALG_0 = 231
    enumerator :: HIPSOLVER_ALG_1 = 232
  end enum

  enum, bind(c)
    enumerator :: HIPSOLVERDN_GETRF = 0
  end enum
 
  enum, bind(c)
    enumerator :: HIPSOLVERRF_FACTORIZATION_ALG0 = 0
    enumerator :: HIPSOLVERRF_FACTORIZATION_ALG1 = 1
    enumerator :: HIPSOLVERRF_FACTORIZATION_ALG2 = 2
  end enum

  enum, bind(c)
    enumerator :: HIPSOLVERRF_MATRIX_FORMAT_CSR = 0
    enumerator :: HIPSOLVERRF_MATRIX_FORMAT_CSC = 1
  end enum

  enum, bind(c)
    enumerator :: HIPSOLVERRF_NUMERIC_BOOST_NOT_USED = 0
    enumerator :: HIPSOLVERRF_NUMERIC_BOOST_USED = 1
  end enum

  enum, bind(c)
    enumerator :: HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF = 0
    enumerator :: HIPSOLVERRF_RESET_VALUES_FAST_MODE_ON = 1
  end enum

  enum, bind(c)
    enumerator :: HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1 = 1
    enumerator :: HIPSOLVERRF_TRIANGULAR_SOLVE_ALG2 = 2
    enumerator :: HIPSOLVERRF_TRIANGULAR_SOLVE_ALG3 = 3
  end enum

  enum, bind(c)
    enumerator :: HIPSOLVERRF_UNIT_DIAGONAL_STORED_L = 0
    enumerator :: HIPSOLVERRF_UNIT_DIAGONAL_STORED_U = 1
    enumerator :: HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2
    enumerator :: HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_U = 3
  end enum

#ifdef USE_FPOINTER_INTERFACES

  
#endif
end module hipfort_hipsolver_enums
