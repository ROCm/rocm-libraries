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

module hipfort_hipblas_auxiliary

  interface hipblasSetVector
#ifdef USE_CUDA_NAMES
    function hipblasSetVector_(n,elem_size,x,incx,y,incy) bind(c, name="cublasSetVector") result(ret)
#else
    function hipblasSetVector_(n,elem_size,x,incx,y,incy) bind(c, name="hipblasSetVector") result(ret)
#endif
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_int),value :: elem_size
      type(c_ptr),value :: x
      integer(c_int),value :: incx
      type(c_ptr),value :: y
      integer(c_int),value :: incy
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipblasSetVector_l_rank_0,hipblasSetVector_l_full_rank,&
      hipblasSetVector_i4_rank_0,hipblasSetVector_i4_full_rank,&
      hipblasSetVector_i8_rank_0,hipblasSetVector_i8_full_rank,&
      hipblasSetVector_r4_rank_0,hipblasSetVector_r4_full_rank,&
      hipblasSetVector_r8_rank_0,hipblasSetVector_r8_full_rank,&
      hipblasSetVector_c4_rank_0,hipblasSetVector_c4_full_rank,&
      hipblasSetVector_c8_rank_0,hipblasSetVector_c8_full_rank 
#endif
  end interface

  interface hipblasGetVector
#ifdef USE_CUDA_NAMES
    function hipblasGetVector_(n,elem_size,x,incx,y,incy) bind(c, name="cublasGetVector") result(ret)
#else
    function hipblasGetVector_(n,elem_size,x,incx,y,incy) bind(c, name="hipblasGetVector") result(ret)
#endif
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_int),value :: elem_size
      type(c_ptr),value :: x
      integer(c_int),value :: incx
      type(c_ptr),value :: y
      integer(c_int),value :: incy
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipblasGetVector_l_rank_0,hipblasGetVector_l_full_rank,&
      hipblasGetVector_i4_rank_0,hipblasGetVector_i4_full_rank,&
      hipblasGetVector_i8_rank_0,hipblasGetVector_i8_full_rank,&
      hipblasGetVector_r4_rank_0,hipblasGetVector_r4_full_rank,&
      hipblasGetVector_r8_rank_0,hipblasGetVector_r8_full_rank,&
      hipblasGetVector_c4_rank_0,hipblasGetVector_c4_full_rank,&
      hipblasGetVector_c8_rank_0,hipblasGetVector_c8_full_rank 
#endif
  end interface

  interface hipblasSetMatrix
#ifdef USE_CUDA_NAMES
    function hipblasSetMatrix_(rows,cols,elem_size,A,lda,B,ldb) bind(c, name="cublasSetMatrix") result(ret)
#else
    function hipblasSetMatrix_(rows,cols,elem_size,A,lda,B,ldb) bind(c, name="hipblasSetMatrix") result(ret)
#endif
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),value :: elem_size
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipblasSetMatrix_l_full_rank,hipblasSetMatrix_l_rank_0,hipblasSetMatrix_l_rank_1,&
      hipblasSetMatrix_i4_full_rank,hipblasSetMatrix_i4_rank_0,hipblasSetMatrix_i4_rank_1,&
      hipblasSetMatrix_i8_full_rank,hipblasSetMatrix_i8_rank_0,hipblasSetMatrix_i8_rank_1,&
      hipblasSetMatrix_r4_full_rank,hipblasSetMatrix_r4_rank_0,hipblasSetMatrix_r4_rank_1,&
      hipblasSetMatrix_r8_full_rank,hipblasSetMatrix_r8_rank_0,hipblasSetMatrix_r8_rank_1,&
      hipblasSetMatrix_c4_full_rank,hipblasSetMatrix_c4_rank_0,hipblasSetMatrix_c4_rank_1,&
      hipblasSetMatrix_c8_full_rank,hipblasSetMatrix_c8_rank_0,hipblasSetMatrix_c8_rank_1 
#endif
  end interface

  interface hipblasGetMatrix
#ifdef USE_CUDA_NAMES
    function hipblasGetMatrix_(rows,cols,elem_size,A,lda,B,ldb) bind(c, name="cublasGetMatrix") result(ret)
#else
    function hipblasGetMatrix_(rows,cols,elem_size,A,lda,B,ldb) bind(c, name="hipblasGetMatrix") result(ret)
#endif
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),value :: elem_size
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipblasGetMatrix_l_full_rank,hipblasGetMatrix_l_rank_0,hipblasGetMatrix_l_rank_1,&
      hipblasGetMatrix_i4_full_rank,hipblasGetMatrix_i4_rank_0,hipblasGetMatrix_i4_rank_1,&
      hipblasGetMatrix_i8_full_rank,hipblasGetMatrix_i8_rank_0,hipblasGetMatrix_i8_rank_1,&
      hipblasGetMatrix_r4_full_rank,hipblasGetMatrix_r4_rank_0,hipblasGetMatrix_r4_rank_1,&
      hipblasGetMatrix_r8_full_rank,hipblasGetMatrix_r8_rank_0,hipblasGetMatrix_r8_rank_1,&
      hipblasGetMatrix_c4_full_rank,hipblasGetMatrix_c4_rank_0,hipblasGetMatrix_c4_rank_1,&
      hipblasGetMatrix_c8_full_rank,hipblasGetMatrix_c8_rank_0,hipblasGetMatrix_c8_rank_1 
#endif
  end interface

  interface hipblasSetVectorAsync
#ifdef USE_CUDA_NAMES
    function hipblasSetVectorAsync_(n,elem_size,x,incx,y,incy,stream) bind(c, name="cublasSetVectorAsync") result(ret)
#else
    function hipblasSetVectorAsync_(n,elem_size,x,incx,y,incy,stream) bind(c, name="hipblasSetVectorAsync") result(ret)
#endif
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_int),value :: elem_size
      type(c_ptr),value :: x
      integer(c_int),value :: incx
      type(c_ptr),value :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipblasSetVectorAsync_l_rank_0,hipblasSetVectorAsync_l_full_rank,&
      hipblasSetVectorAsync_i4_rank_0,hipblasSetVectorAsync_i4_full_rank,&
      hipblasSetVectorAsync_i8_rank_0,hipblasSetVectorAsync_i8_full_rank,&
      hipblasSetVectorAsync_r4_rank_0,hipblasSetVectorAsync_r4_full_rank,&
      hipblasSetVectorAsync_r8_rank_0,hipblasSetVectorAsync_r8_full_rank,&
      hipblasSetVectorAsync_c4_rank_0,hipblasSetVectorAsync_c4_full_rank,&
      hipblasSetVectorAsync_c8_rank_0,hipblasSetVectorAsync_c8_full_rank 
#endif
  end interface

  interface hipblasGetVectorAsync
#ifdef USE_CUDA_NAMES
    function hipblasGetVectorAsync_(n,elem_size,x,incx,y,incy,stream) bind(c, name="cublasGetVectorAsync") result(ret)
#else
    function hipblasGetVectorAsync_(n,elem_size,x,incx,y,incy,stream) bind(c, name="hipblasGetVectorAsync") result(ret)
#endif
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_int),value :: elem_size
      type(c_ptr),value :: x
      integer(c_int),value :: incx
      type(c_ptr),value :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipblasGetVectorAsync_l_rank_0,hipblasGetVectorAsync_l_full_rank,&
      hipblasGetVectorAsync_i4_rank_0,hipblasGetVectorAsync_i4_full_rank,&
      hipblasGetVectorAsync_i8_rank_0,hipblasGetVectorAsync_i8_full_rank,&
      hipblasGetVectorAsync_r4_rank_0,hipblasGetVectorAsync_r4_full_rank,&
      hipblasGetVectorAsync_r8_rank_0,hipblasGetVectorAsync_r8_full_rank,&
      hipblasGetVectorAsync_c4_rank_0,hipblasGetVectorAsync_c4_full_rank,&
      hipblasGetVectorAsync_c8_rank_0,hipblasGetVectorAsync_c8_full_rank 
#endif
  end interface

  interface hipblasSetMatrixAsync
#ifdef USE_CUDA_NAMES
    function hipblasSetMatrixAsync_(rows,cols,elem_size,A,lda,B,ldb,stream) bind(c, name="cublasSetMatrixAsync") result(ret)
#else
    function hipblasSetMatrixAsync_(rows,cols,elem_size,A,lda,B,ldb,stream) bind(c, name="hipblasSetMatrixAsync") result(ret)
#endif
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),value :: elem_size
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipblasSetMatrixAsync_l_full_rank,hipblasSetMatrixAsync_l_rank_0,hipblasSetMatrixAsync_l_rank_1,&
      hipblasSetMatrixAsync_i4_full_rank,hipblasSetMatrixAsync_i4_rank_0,hipblasSetMatrixAsync_i4_rank_1,&
      hipblasSetMatrixAsync_i8_full_rank,hipblasSetMatrixAsync_i8_rank_0,hipblasSetMatrixAsync_i8_rank_1,&
      hipblasSetMatrixAsync_r4_full_rank,hipblasSetMatrixAsync_r4_rank_0,hipblasSetMatrixAsync_r4_rank_1,&
      hipblasSetMatrixAsync_r8_full_rank,hipblasSetMatrixAsync_r8_rank_0,hipblasSetMatrixAsync_r8_rank_1,&
      hipblasSetMatrixAsync_c4_full_rank,hipblasSetMatrixAsync_c4_rank_0,hipblasSetMatrixAsync_c4_rank_1,&
      hipblasSetMatrixAsync_c8_full_rank,hipblasSetMatrixAsync_c8_rank_0,hipblasSetMatrixAsync_c8_rank_1 
#endif
  end interface

  interface hipblasGetMatrixAsync
#ifdef USE_CUDA_NAMES
    function hipblasGetMatrixAsync_(rows,cols,elem_size,A,lda,B,ldb,stream) bind(c, name="cublasGetMatrixAsync") result(ret)
#else
    function hipblasGetMatrixAsync_(rows,cols,elem_size,A,lda,B,ldb,stream) bind(c, name="hipblasGetMatrixAsync") result(ret)
#endif
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),value :: elem_size
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipblasGetMatrixAsync_l_full_rank,hipblasGetMatrixAsync_l_rank_0,hipblasGetMatrixAsync_l_rank_1,&
      hipblasGetMatrixAsync_i4_full_rank,hipblasGetMatrixAsync_i4_rank_0,hipblasGetMatrixAsync_i4_rank_1,&
      hipblasGetMatrixAsync_i8_full_rank,hipblasGetMatrixAsync_i8_rank_0,hipblasGetMatrixAsync_i8_rank_1,&
      hipblasGetMatrixAsync_r4_full_rank,hipblasGetMatrixAsync_r4_rank_0,hipblasGetMatrixAsync_r4_rank_1,&
      hipblasGetMatrixAsync_r8_full_rank,hipblasGetMatrixAsync_r8_rank_0,hipblasGetMatrixAsync_r8_rank_1,&
      hipblasGetMatrixAsync_c4_full_rank,hipblasGetMatrixAsync_c4_rank_0,hipblasGetMatrixAsync_c4_rank_1,&
      hipblasGetMatrixAsync_c8_full_rank,hipblasGetMatrixAsync_c8_rank_0,hipblasGetMatrixAsync_c8_rank_1 
#endif
  end interface

#ifdef USE_FPOINTER_INTERFACES
contains

    function hipblasSetVector_l_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target :: x
      integer(c_int),value :: incx
      logical(c_bool),target :: y
      integer(c_int),value :: incy
      !
      ret = hipblasSetVector_(n,1,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasSetVector_l_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target,dimension(:) :: x
      integer(c_int),value :: incx
      logical(c_bool),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = hipblasSetVector_(n,1,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasGetVector_l_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target :: x
      integer(c_int),value :: incx
      logical(c_bool),target :: y
      integer(c_int),value :: incy
      !
      ret = hipblasGetVector_(n,1,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasGetVector_l_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target,dimension(:) :: x
      integer(c_int),value :: incx
      logical(c_bool),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = hipblasGetVector_(n,1,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasSetMatrix_l_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:,:) :: A
      logical(c_bool),target,dimension(:,:) :: B
      !
      ret = hipblasSetMatrix_(rows,cols,1,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function hipblasSetMatrix_l_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target :: A
      integer(c_int),value :: lda
      logical(c_bool),target :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasSetMatrix_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasSetMatrix_l_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:) :: A
      integer(c_int),value :: lda
      logical(c_bool),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasSetMatrix_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasGetMatrix_l_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:,:) :: A
      logical(c_bool),target,dimension(:,:) :: B
      !
      ret = hipblasGetMatrix_(rows,cols,1,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function hipblasGetMatrix_l_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target :: A
      integer(c_int),value :: lda
      logical(c_bool),target :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasGetMatrix_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasGetMatrix_l_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:) :: A
      integer(c_int),value :: lda
      logical(c_bool),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasGetMatrix_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasSetVectorAsync_l_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target :: x
      integer(c_int),value :: incx
      logical(c_bool),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasSetVectorAsync_(n,1,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasSetVectorAsync_l_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target,dimension(:) :: x
      integer(c_int),value :: incx
      logical(c_bool),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasSetVectorAsync_(n,1,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasGetVectorAsync_l_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target :: x
      integer(c_int),value :: incx
      logical(c_bool),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasGetVectorAsync_(n,1,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasGetVectorAsync_l_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target,dimension(:) :: x
      integer(c_int),value :: incx
      logical(c_bool),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasGetVectorAsync_(n,1,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasSetMatrixAsync_l_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:,:) :: A
      logical(c_bool),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,1,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function hipblasSetMatrixAsync_l_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target :: A
      integer(c_int),value :: lda
      logical(c_bool),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasSetMatrixAsync_l_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:) :: A
      integer(c_int),value :: lda
      logical(c_bool),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasGetMatrixAsync_l_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:,:) :: A
      logical(c_bool),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,1,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function hipblasGetMatrixAsync_l_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target :: A
      integer(c_int),value :: lda
      logical(c_bool),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasGetMatrixAsync_l_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:) :: A
      integer(c_int),value :: lda
      logical(c_bool),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb,stream)
    end function
    function hipblasSetVector_i4_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_int),target :: x
      integer(c_int),value :: incx
      integer(c_int),target :: y
      integer(c_int),value :: incy
      !
      ret = hipblasSetVector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasSetVector_i4_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_int),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_int),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = hipblasSetVector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasGetVector_i4_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_int),target :: x
      integer(c_int),value :: incx
      integer(c_int),target :: y
      integer(c_int),value :: incy
      !
      ret = hipblasGetVector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasGetVector_i4_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_int),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_int),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = hipblasGetVector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasSetMatrix_i4_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:,:) :: A
      integer(c_int),target,dimension(:,:) :: B
      !
      ret = hipblasSetMatrix_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function hipblasSetMatrix_i4_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target :: A
      integer(c_int),value :: lda
      integer(c_int),target :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasSetMatrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasSetMatrix_i4_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_int),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasSetMatrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasGetMatrix_i4_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:,:) :: A
      integer(c_int),target,dimension(:,:) :: B
      !
      ret = hipblasGetMatrix_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function hipblasGetMatrix_i4_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target :: A
      integer(c_int),value :: lda
      integer(c_int),target :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasGetMatrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasGetMatrix_i4_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_int),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasGetMatrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasSetVectorAsync_i4_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_int),target :: x
      integer(c_int),value :: incx
      integer(c_int),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasSetVectorAsync_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasSetVectorAsync_i4_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_int),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_int),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasSetVectorAsync_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasGetVectorAsync_i4_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_int),target :: x
      integer(c_int),value :: incx
      integer(c_int),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasGetVectorAsync_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasGetVectorAsync_i4_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_int),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_int),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasGetVectorAsync_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasSetMatrixAsync_i4_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:,:) :: A
      integer(c_int),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function hipblasSetMatrixAsync_i4_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target :: A
      integer(c_int),value :: lda
      integer(c_int),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasSetMatrixAsync_i4_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_int),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasGetMatrixAsync_i4_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:,:) :: A
      integer(c_int),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function hipblasGetMatrixAsync_i4_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target :: A
      integer(c_int),value :: lda
      integer(c_int),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasGetMatrixAsync_i4_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_int),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function
    function hipblasSetVector_i8_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_long),target :: x
      integer(c_int),value :: incx
      integer(c_long),target :: y
      integer(c_int),value :: incy
      !
      ret = hipblasSetVector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasSetVector_i8_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_long),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_long),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = hipblasSetVector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasGetVector_i8_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_long),target :: x
      integer(c_int),value :: incx
      integer(c_long),target :: y
      integer(c_int),value :: incy
      !
      ret = hipblasGetVector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasGetVector_i8_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_long),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_long),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = hipblasGetVector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasSetMatrix_i8_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:,:) :: A
      integer(c_long),target,dimension(:,:) :: B
      !
      ret = hipblasSetMatrix_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function hipblasSetMatrix_i8_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target :: A
      integer(c_int),value :: lda
      integer(c_long),target :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasSetMatrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasSetMatrix_i8_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_long),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasSetMatrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasGetMatrix_i8_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:,:) :: A
      integer(c_long),target,dimension(:,:) :: B
      !
      ret = hipblasGetMatrix_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function hipblasGetMatrix_i8_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target :: A
      integer(c_int),value :: lda
      integer(c_long),target :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasGetMatrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasGetMatrix_i8_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_long),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasGetMatrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasSetVectorAsync_i8_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_long),target :: x
      integer(c_int),value :: incx
      integer(c_long),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasSetVectorAsync_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasSetVectorAsync_i8_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_long),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_long),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasSetVectorAsync_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasGetVectorAsync_i8_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_long),target :: x
      integer(c_int),value :: incx
      integer(c_long),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasGetVectorAsync_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasGetVectorAsync_i8_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      integer(c_long),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_long),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasGetVectorAsync_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasSetMatrixAsync_i8_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:,:) :: A
      integer(c_long),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function hipblasSetMatrixAsync_i8_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target :: A
      integer(c_int),value :: lda
      integer(c_long),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasSetMatrixAsync_i8_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_long),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasGetMatrixAsync_i8_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:,:) :: A
      integer(c_long),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function hipblasGetMatrixAsync_i8_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target :: A
      integer(c_int),value :: lda
      integer(c_long),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasGetMatrixAsync_i8_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_long),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function
    function hipblasSetVector_r4_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_float),target :: x
      integer(c_int),value :: incx
      real(c_float),target :: y
      integer(c_int),value :: incy
      !
      ret = hipblasSetVector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasSetVector_r4_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_float),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_float),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = hipblasSetVector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasGetVector_r4_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_float),target :: x
      integer(c_int),value :: incx
      real(c_float),target :: y
      integer(c_int),value :: incy
      !
      ret = hipblasGetVector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasGetVector_r4_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_float),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_float),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = hipblasGetVector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasSetMatrix_r4_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:,:) :: A
      real(c_float),target,dimension(:,:) :: B
      !
      ret = hipblasSetMatrix_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function hipblasSetMatrix_r4_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target :: A
      integer(c_int),value :: lda
      real(c_float),target :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasSetMatrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasSetMatrix_r4_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_float),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasSetMatrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasGetMatrix_r4_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:,:) :: A
      real(c_float),target,dimension(:,:) :: B
      !
      ret = hipblasGetMatrix_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function hipblasGetMatrix_r4_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target :: A
      integer(c_int),value :: lda
      real(c_float),target :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasGetMatrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasGetMatrix_r4_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_float),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasGetMatrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasSetVectorAsync_r4_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_float),target :: x
      integer(c_int),value :: incx
      real(c_float),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasSetVectorAsync_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasSetVectorAsync_r4_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_float),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_float),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasSetVectorAsync_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasGetVectorAsync_r4_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_float),target :: x
      integer(c_int),value :: incx
      real(c_float),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasGetVectorAsync_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasGetVectorAsync_r4_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_float),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_float),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasGetVectorAsync_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasSetMatrixAsync_r4_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:,:) :: A
      real(c_float),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function hipblasSetMatrixAsync_r4_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target :: A
      integer(c_int),value :: lda
      real(c_float),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasSetMatrixAsync_r4_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_float),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasGetMatrixAsync_r4_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:,:) :: A
      real(c_float),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function hipblasGetMatrixAsync_r4_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target :: A
      integer(c_int),value :: lda
      real(c_float),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasGetMatrixAsync_r4_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_float),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function
    function hipblasSetVector_r8_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_double),target :: x
      integer(c_int),value :: incx
      real(c_double),target :: y
      integer(c_int),value :: incy
      !
      ret = hipblasSetVector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasSetVector_r8_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_double),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_double),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = hipblasSetVector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasGetVector_r8_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_double),target :: x
      integer(c_int),value :: incx
      real(c_double),target :: y
      integer(c_int),value :: incy
      !
      ret = hipblasGetVector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasGetVector_r8_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_double),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_double),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = hipblasGetVector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasSetMatrix_r8_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:,:) :: A
      real(c_double),target,dimension(:,:) :: B
      !
      ret = hipblasSetMatrix_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function hipblasSetMatrix_r8_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target :: A
      integer(c_int),value :: lda
      real(c_double),target :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasSetMatrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasSetMatrix_r8_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_double),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasSetMatrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasGetMatrix_r8_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:,:) :: A
      real(c_double),target,dimension(:,:) :: B
      !
      ret = hipblasGetMatrix_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function hipblasGetMatrix_r8_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target :: A
      integer(c_int),value :: lda
      real(c_double),target :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasGetMatrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasGetMatrix_r8_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_double),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasGetMatrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasSetVectorAsync_r8_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_double),target :: x
      integer(c_int),value :: incx
      real(c_double),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasSetVectorAsync_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasSetVectorAsync_r8_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_double),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_double),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasSetVectorAsync_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasGetVectorAsync_r8_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_double),target :: x
      integer(c_int),value :: incx
      real(c_double),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasGetVectorAsync_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasGetVectorAsync_r8_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      real(c_double),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_double),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasGetVectorAsync_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasSetMatrixAsync_r8_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:,:) :: A
      real(c_double),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function hipblasSetMatrixAsync_r8_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target :: A
      integer(c_int),value :: lda
      real(c_double),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasSetMatrixAsync_r8_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_double),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasGetMatrixAsync_r8_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:,:) :: A
      real(c_double),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function hipblasGetMatrixAsync_r8_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target :: A
      integer(c_int),value :: lda
      real(c_double),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasGetMatrixAsync_r8_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_double),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function
    function hipblasSetVector_c4_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target :: y
      integer(c_int),value :: incy
      !
      ret = hipblasSetVector_(n,2*4,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasSetVector_c4_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = hipblasSetVector_(n,2*4,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasGetVector_c4_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target :: y
      integer(c_int),value :: incy
      !
      ret = hipblasGetVector_(n,2*4,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasGetVector_c4_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = hipblasGetVector_(n,2*4,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasSetMatrix_c4_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:,:) :: A
      complex(c_float_complex),target,dimension(:,:) :: B
      !
      ret = hipblasSetMatrix_(rows,cols,2*4,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function hipblasSetMatrix_c4_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasSetMatrix_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasSetMatrix_c4_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasSetMatrix_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasGetMatrix_c4_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:,:) :: A
      complex(c_float_complex),target,dimension(:,:) :: B
      !
      ret = hipblasGetMatrix_(rows,cols,2*4,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function hipblasGetMatrix_c4_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasGetMatrix_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasGetMatrix_c4_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasGetMatrix_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasSetVectorAsync_c4_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasSetVectorAsync_(n,2*4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasSetVectorAsync_c4_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasSetVectorAsync_(n,2*4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasGetVectorAsync_c4_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasGetVectorAsync_(n,2*4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasGetVectorAsync_c4_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasGetVectorAsync_(n,2*4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasSetMatrixAsync_c4_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:,:) :: A
      complex(c_float_complex),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,2*4,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function hipblasSetMatrixAsync_c4_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasSetMatrixAsync_c4_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasGetMatrixAsync_c4_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:,:) :: A
      complex(c_float_complex),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,2*4,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function hipblasGetMatrixAsync_c4_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasGetMatrixAsync_c4_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function
    function hipblasSetVector_c8_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target :: y
      integer(c_int),value :: incy
      !
      ret = hipblasSetVector_(n,2*8,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasSetVector_c8_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = hipblasSetVector_(n,2*8,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasGetVector_c8_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target :: y
      integer(c_int),value :: incy
      !
      ret = hipblasGetVector_(n,2*8,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasGetVector_c8_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = hipblasGetVector_(n,2*8,c_loc(x),incx,c_loc(y),incy)
    end function

    function hipblasSetMatrix_c8_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:,:) :: A
      complex(c_double_complex),target,dimension(:,:) :: B
      !
      ret = hipblasSetMatrix_(rows,cols,2*8,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function hipblasSetMatrix_c8_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasSetMatrix_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasSetMatrix_c8_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasSetMatrix_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasGetMatrix_c8_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:,:) :: A
      complex(c_double_complex),target,dimension(:,:) :: B
      !
      ret = hipblasGetMatrix_(rows,cols,2*8,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function hipblasGetMatrix_c8_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasGetMatrix_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasGetMatrix_c8_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = hipblasGetMatrix_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function hipblasSetVectorAsync_c8_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasSetVectorAsync_(n,2*8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasSetVectorAsync_c8_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasSetVectorAsync_(n,2*8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasGetVectorAsync_c8_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasGetVectorAsync_(n,2*8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasGetVectorAsync_c8_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = hipblasGetVectorAsync_(n,2*8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function hipblasSetMatrixAsync_c8_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:,:) :: A
      complex(c_double_complex),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,2*8,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function hipblasSetMatrixAsync_c8_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasSetMatrixAsync_c8_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasSetMatrixAsync_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasGetMatrixAsync_c8_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:,:) :: A
      complex(c_double_complex),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,2*8,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function hipblasGetMatrixAsync_c8_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function hipblasGetMatrixAsync_c8_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_hipblas_enums
      implicit none
      integer(kind(HIPBLAS_STATUS_SUCCESS)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = hipblasGetMatrixAsync_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function
 

#endif
end module