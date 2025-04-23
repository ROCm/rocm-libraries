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

module hipfort_rocblas_auxiliary

  interface rocblas_set_vector
    function rocblas_set_vector_(n,elem_size,x,incx,y,incy) bind(c, name="rocblas_set_vector") result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_int),value :: elem_size
      type(c_ptr),value :: x
      integer(c_int),value :: incx
      type(c_ptr),value :: y
      integer(c_int),value :: incy
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocblas_set_vector_l_rank_0,rocblas_set_vector_l_full_rank,&
      rocblas_set_vector_i4_rank_0,rocblas_set_vector_i4_full_rank,&
      rocblas_set_vector_i8_rank_0,rocblas_set_vector_i8_full_rank,&
      rocblas_set_vector_r4_rank_0,rocblas_set_vector_r4_full_rank,&
      rocblas_set_vector_r8_rank_0,rocblas_set_vector_r8_full_rank,&
      rocblas_set_vector_c4_rank_0,rocblas_set_vector_c4_full_rank,&
      rocblas_set_vector_c8_rank_0,rocblas_set_vector_c8_full_rank 
#endif
  end interface

  interface rocblas_get_vector
    function rocblas_get_vector_(n,elem_size,x,incx,y,incy) bind(c, name="rocblas_get_vector") result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_int),value :: elem_size
      type(c_ptr),value :: x
      integer(c_int),value :: incx
      type(c_ptr),value :: y
      integer(c_int),value :: incy
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocblas_get_vector_l_rank_0,rocblas_get_vector_l_full_rank,&
      rocblas_get_vector_i4_rank_0,rocblas_get_vector_i4_full_rank,&
      rocblas_get_vector_i8_rank_0,rocblas_get_vector_i8_full_rank,&
      rocblas_get_vector_r4_rank_0,rocblas_get_vector_r4_full_rank,&
      rocblas_get_vector_r8_rank_0,rocblas_get_vector_r8_full_rank,&
      rocblas_get_vector_c4_rank_0,rocblas_get_vector_c4_full_rank,&
      rocblas_get_vector_c8_rank_0,rocblas_get_vector_c8_full_rank 
#endif
  end interface

  interface rocblas_set_matrix
    function rocblas_set_matrix_(rows,cols,elem_size,A,lda,B,ldb) bind(c, name="rocblas_set_matrix") result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),value :: elem_size
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocblas_set_matrix_l_full_rank,rocblas_set_matrix_l_rank_0,rocblas_set_matrix_l_rank_1,&
      rocblas_set_matrix_i4_full_rank,rocblas_set_matrix_i4_rank_0,rocblas_set_matrix_i4_rank_1,&
      rocblas_set_matrix_i8_full_rank,rocblas_set_matrix_i8_rank_0,rocblas_set_matrix_i8_rank_1,&
      rocblas_set_matrix_r4_full_rank,rocblas_set_matrix_r4_rank_0,rocblas_set_matrix_r4_rank_1,&
      rocblas_set_matrix_r8_full_rank,rocblas_set_matrix_r8_rank_0,rocblas_set_matrix_r8_rank_1,&
      rocblas_set_matrix_c4_full_rank,rocblas_set_matrix_c4_rank_0,rocblas_set_matrix_c4_rank_1,&
      rocblas_set_matrix_c8_full_rank,rocblas_set_matrix_c8_rank_0,rocblas_set_matrix_c8_rank_1 
#endif
  end interface

  interface rocblas_get_matrix
    function rocblas_get_matrix_(rows,cols,elem_size,A,lda,B,ldb) bind(c, name="rocblas_get_matrix") result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),value :: elem_size
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocblas_get_matrix_l_full_rank,rocblas_get_matrix_l_rank_0,rocblas_get_matrix_l_rank_1,&
      rocblas_get_matrix_i4_full_rank,rocblas_get_matrix_i4_rank_0,rocblas_get_matrix_i4_rank_1,&
      rocblas_get_matrix_i8_full_rank,rocblas_get_matrix_i8_rank_0,rocblas_get_matrix_i8_rank_1,&
      rocblas_get_matrix_r4_full_rank,rocblas_get_matrix_r4_rank_0,rocblas_get_matrix_r4_rank_1,&
      rocblas_get_matrix_r8_full_rank,rocblas_get_matrix_r8_rank_0,rocblas_get_matrix_r8_rank_1,&
      rocblas_get_matrix_c4_full_rank,rocblas_get_matrix_c4_rank_0,rocblas_get_matrix_c4_rank_1,&
      rocblas_get_matrix_c8_full_rank,rocblas_get_matrix_c8_rank_0,rocblas_get_matrix_c8_rank_1 
#endif
  end interface

  interface rocblas_set_vector_async
    function rocblas_set_vector_async_(n,elem_size,x,incx,y,incy,stream) bind(c, name="rocblas_set_vector_async") result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_int),value :: elem_size
      type(c_ptr),value :: x
      integer(c_int),value :: incx
      type(c_ptr),value :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocblas_set_vector_async_l_rank_0,rocblas_set_vector_async_l_full_rank,&
      rocblas_set_vector_async_i4_rank_0,rocblas_set_vector_async_i4_full_rank,&
      rocblas_set_vector_async_i8_rank_0,rocblas_set_vector_async_i8_full_rank,&
      rocblas_set_vector_async_r4_rank_0,rocblas_set_vector_async_r4_full_rank,&
      rocblas_set_vector_async_r8_rank_0,rocblas_set_vector_async_r8_full_rank,&
      rocblas_set_vector_async_c4_rank_0,rocblas_set_vector_async_c4_full_rank,&
      rocblas_set_vector_async_c8_rank_0,rocblas_set_vector_async_c8_full_rank 
#endif
  end interface

  interface rocblas_get_vector_async
    function rocblas_get_vector_async_(n,elem_size,x,incx,y,incy,stream) bind(c, name="rocblas_get_vector_async") result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_int),value :: elem_size
      type(c_ptr),value :: x
      integer(c_int),value :: incx
      type(c_ptr),value :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocblas_get_vector_async_l_rank_0,rocblas_get_vector_async_l_full_rank,&
      rocblas_get_vector_async_i4_rank_0,rocblas_get_vector_async_i4_full_rank,&
      rocblas_get_vector_async_i8_rank_0,rocblas_get_vector_async_i8_full_rank,&
      rocblas_get_vector_async_r4_rank_0,rocblas_get_vector_async_r4_full_rank,&
      rocblas_get_vector_async_r8_rank_0,rocblas_get_vector_async_r8_full_rank,&
      rocblas_get_vector_async_c4_rank_0,rocblas_get_vector_async_c4_full_rank,&
      rocblas_get_vector_async_c8_rank_0,rocblas_get_vector_async_c8_full_rank 
#endif
  end interface

  interface rocblas_set_matrix_async
    function rocblas_set_matrix_async_(rows,cols,elem_size,A,lda,B,ldb,stream) bind(c, name="rocblas_set_matrix_async") result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
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
    module procedure rocblas_set_matrix_async_l_full_rank,rocblas_set_matrix_async_l_rank_0,rocblas_set_matrix_async_l_rank_1,&
      rocblas_set_matrix_async_i4_full_rank,rocblas_set_matrix_async_i4_rank_0,rocblas_set_matrix_async_i4_rank_1,&
      rocblas_set_matrix_async_i8_full_rank,rocblas_set_matrix_async_i8_rank_0,rocblas_set_matrix_async_i8_rank_1,&
      rocblas_set_matrix_async_r4_full_rank,rocblas_set_matrix_async_r4_rank_0,rocblas_set_matrix_async_r4_rank_1,&
      rocblas_set_matrix_async_r8_full_rank,rocblas_set_matrix_async_r8_rank_0,rocblas_set_matrix_async_r8_rank_1,&
      rocblas_set_matrix_async_c4_full_rank,rocblas_set_matrix_async_c4_rank_0,rocblas_set_matrix_async_c4_rank_1,&
      rocblas_set_matrix_async_c8_full_rank,rocblas_set_matrix_async_c8_rank_0,rocblas_set_matrix_async_c8_rank_1 
#endif
  end interface

  interface rocblas_get_matrix_async
    function rocblas_get_matrix_async_(rows,cols,elem_size,A,lda,B,ldb,stream) bind(c, name="rocblas_get_matrix_async") result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
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
    module procedure rocblas_get_matrix_async_l_full_rank,rocblas_get_matrix_async_l_rank_0,rocblas_get_matrix_async_l_rank_1,&
      rocblas_get_matrix_async_i4_full_rank,rocblas_get_matrix_async_i4_rank_0,rocblas_get_matrix_async_i4_rank_1,&
      rocblas_get_matrix_async_i8_full_rank,rocblas_get_matrix_async_i8_rank_0,rocblas_get_matrix_async_i8_rank_1,&
      rocblas_get_matrix_async_r4_full_rank,rocblas_get_matrix_async_r4_rank_0,rocblas_get_matrix_async_r4_rank_1,&
      rocblas_get_matrix_async_r8_full_rank,rocblas_get_matrix_async_r8_rank_0,rocblas_get_matrix_async_r8_rank_1,&
      rocblas_get_matrix_async_c4_full_rank,rocblas_get_matrix_async_c4_rank_0,rocblas_get_matrix_async_c4_rank_1,&
      rocblas_get_matrix_async_c8_full_rank,rocblas_get_matrix_async_c8_rank_0,rocblas_get_matrix_async_c8_rank_1 
#endif
  end interface

#ifdef USE_FPOINTER_INTERFACES
contains

    function rocblas_set_vector_l_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target :: x
      integer(c_int),value :: incx
      logical(c_bool),target :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_set_vector_(n,1,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_set_vector_l_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target,dimension(:) :: x
      integer(c_int),value :: incx
      logical(c_bool),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_set_vector_(n,1,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_get_vector_l_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target :: x
      integer(c_int),value :: incx
      logical(c_bool),target :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_get_vector_(n,1,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_get_vector_l_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target,dimension(:) :: x
      integer(c_int),value :: incx
      logical(c_bool),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_get_vector_(n,1,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_set_matrix_l_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:,:) :: A
      logical(c_bool),target,dimension(:,:) :: B
      !
      ret = rocblas_set_matrix_(rows,cols,1,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function rocblas_set_matrix_l_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target :: A
      integer(c_int),value :: lda
      logical(c_bool),target :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_set_matrix_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_set_matrix_l_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:) :: A
      integer(c_int),value :: lda
      logical(c_bool),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_set_matrix_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_get_matrix_l_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:,:) :: A
      logical(c_bool),target,dimension(:,:) :: B
      !
      ret = rocblas_get_matrix_(rows,cols,1,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function rocblas_get_matrix_l_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target :: A
      integer(c_int),value :: lda
      logical(c_bool),target :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_get_matrix_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_get_matrix_l_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:) :: A
      integer(c_int),value :: lda
      logical(c_bool),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_get_matrix_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_set_vector_async_l_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target :: x
      integer(c_int),value :: incx
      logical(c_bool),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_vector_async_(n,1,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_set_vector_async_l_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target,dimension(:) :: x
      integer(c_int),value :: incx
      logical(c_bool),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_vector_async_(n,1,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_get_vector_async_l_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target :: x
      integer(c_int),value :: incx
      logical(c_bool),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_vector_async_(n,1,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_get_vector_async_l_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      logical(c_bool),target,dimension(:) :: x
      integer(c_int),value :: incx
      logical(c_bool),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_vector_async_(n,1,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_set_matrix_async_l_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:,:) :: A
      logical(c_bool),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,1,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function rocblas_set_matrix_async_l_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target :: A
      integer(c_int),value :: lda
      logical(c_bool),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_set_matrix_async_l_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:) :: A
      integer(c_int),value :: lda
      logical(c_bool),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_get_matrix_async_l_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:,:) :: A
      logical(c_bool),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,1,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function rocblas_get_matrix_async_l_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target :: A
      integer(c_int),value :: lda
      logical(c_bool),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_get_matrix_async_l_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      logical(c_bool),target,dimension(:) :: A
      integer(c_int),value :: lda
      logical(c_bool),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,1,c_loc(A),lda,c_loc(B),ldb,stream)
    end function
    function rocblas_set_vector_i4_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_int),target :: x
      integer(c_int),value :: incx
      integer(c_int),target :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_set_vector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_set_vector_i4_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_int),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_int),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_set_vector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_get_vector_i4_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_int),target :: x
      integer(c_int),value :: incx
      integer(c_int),target :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_get_vector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_get_vector_i4_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_int),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_int),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_get_vector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_set_matrix_i4_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:,:) :: A
      integer(c_int),target,dimension(:,:) :: B
      !
      ret = rocblas_set_matrix_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function rocblas_set_matrix_i4_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target :: A
      integer(c_int),value :: lda
      integer(c_int),target :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_set_matrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_set_matrix_i4_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_int),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_set_matrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_get_matrix_i4_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:,:) :: A
      integer(c_int),target,dimension(:,:) :: B
      !
      ret = rocblas_get_matrix_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function rocblas_get_matrix_i4_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target :: A
      integer(c_int),value :: lda
      integer(c_int),target :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_get_matrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_get_matrix_i4_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_int),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_get_matrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_set_vector_async_i4_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_int),target :: x
      integer(c_int),value :: incx
      integer(c_int),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_vector_async_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_set_vector_async_i4_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_int),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_int),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_vector_async_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_get_vector_async_i4_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_int),target :: x
      integer(c_int),value :: incx
      integer(c_int),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_vector_async_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_get_vector_async_i4_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_int),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_int),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_vector_async_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_set_matrix_async_i4_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:,:) :: A
      integer(c_int),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function rocblas_set_matrix_async_i4_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target :: A
      integer(c_int),value :: lda
      integer(c_int),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_set_matrix_async_i4_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_int),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_get_matrix_async_i4_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:,:) :: A
      integer(c_int),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function rocblas_get_matrix_async_i4_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target :: A
      integer(c_int),value :: lda
      integer(c_int),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_get_matrix_async_i4_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_int),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_int),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function
    function rocblas_set_vector_i8_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_long),target :: x
      integer(c_int),value :: incx
      integer(c_long),target :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_set_vector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_set_vector_i8_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_long),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_long),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_set_vector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_get_vector_i8_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_long),target :: x
      integer(c_int),value :: incx
      integer(c_long),target :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_get_vector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_get_vector_i8_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_long),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_long),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_get_vector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_set_matrix_i8_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:,:) :: A
      integer(c_long),target,dimension(:,:) :: B
      !
      ret = rocblas_set_matrix_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function rocblas_set_matrix_i8_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target :: A
      integer(c_int),value :: lda
      integer(c_long),target :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_set_matrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_set_matrix_i8_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_long),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_set_matrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_get_matrix_i8_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:,:) :: A
      integer(c_long),target,dimension(:,:) :: B
      !
      ret = rocblas_get_matrix_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function rocblas_get_matrix_i8_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target :: A
      integer(c_int),value :: lda
      integer(c_long),target :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_get_matrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_get_matrix_i8_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_long),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_get_matrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_set_vector_async_i8_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_long),target :: x
      integer(c_int),value :: incx
      integer(c_long),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_vector_async_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_set_vector_async_i8_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_long),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_long),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_vector_async_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_get_vector_async_i8_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_long),target :: x
      integer(c_int),value :: incx
      integer(c_long),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_vector_async_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_get_vector_async_i8_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      integer(c_long),target,dimension(:) :: x
      integer(c_int),value :: incx
      integer(c_long),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_vector_async_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_set_matrix_async_i8_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:,:) :: A
      integer(c_long),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function rocblas_set_matrix_async_i8_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target :: A
      integer(c_int),value :: lda
      integer(c_long),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_set_matrix_async_i8_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_long),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_get_matrix_async_i8_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:,:) :: A
      integer(c_long),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function rocblas_get_matrix_async_i8_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target :: A
      integer(c_int),value :: lda
      integer(c_long),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_get_matrix_async_i8_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      integer(c_long),target,dimension(:) :: A
      integer(c_int),value :: lda
      integer(c_long),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function
    function rocblas_set_vector_r4_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_float),target :: x
      integer(c_int),value :: incx
      real(c_float),target :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_set_vector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_set_vector_r4_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_float),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_float),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_set_vector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_get_vector_r4_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_float),target :: x
      integer(c_int),value :: incx
      real(c_float),target :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_get_vector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_get_vector_r4_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_float),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_float),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_get_vector_(n,4,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_set_matrix_r4_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:,:) :: A
      real(c_float),target,dimension(:,:) :: B
      !
      ret = rocblas_set_matrix_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function rocblas_set_matrix_r4_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target :: A
      integer(c_int),value :: lda
      real(c_float),target :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_set_matrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_set_matrix_r4_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_float),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_set_matrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_get_matrix_r4_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:,:) :: A
      real(c_float),target,dimension(:,:) :: B
      !
      ret = rocblas_get_matrix_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function rocblas_get_matrix_r4_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target :: A
      integer(c_int),value :: lda
      real(c_float),target :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_get_matrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_get_matrix_r4_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_float),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_get_matrix_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_set_vector_async_r4_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_float),target :: x
      integer(c_int),value :: incx
      real(c_float),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_vector_async_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_set_vector_async_r4_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_float),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_float),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_vector_async_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_get_vector_async_r4_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_float),target :: x
      integer(c_int),value :: incx
      real(c_float),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_vector_async_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_get_vector_async_r4_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_float),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_float),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_vector_async_(n,4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_set_matrix_async_r4_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:,:) :: A
      real(c_float),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function rocblas_set_matrix_async_r4_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target :: A
      integer(c_int),value :: lda
      real(c_float),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_set_matrix_async_r4_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_float),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_get_matrix_async_r4_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:,:) :: A
      real(c_float),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,4,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function rocblas_get_matrix_async_r4_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target :: A
      integer(c_int),value :: lda
      real(c_float),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_get_matrix_async_r4_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_float),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_float),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function
    function rocblas_set_vector_r8_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_double),target :: x
      integer(c_int),value :: incx
      real(c_double),target :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_set_vector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_set_vector_r8_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_double),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_double),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_set_vector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_get_vector_r8_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_double),target :: x
      integer(c_int),value :: incx
      real(c_double),target :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_get_vector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_get_vector_r8_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_double),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_double),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_get_vector_(n,8,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_set_matrix_r8_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:,:) :: A
      real(c_double),target,dimension(:,:) :: B
      !
      ret = rocblas_set_matrix_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function rocblas_set_matrix_r8_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target :: A
      integer(c_int),value :: lda
      real(c_double),target :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_set_matrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_set_matrix_r8_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_double),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_set_matrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_get_matrix_r8_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:,:) :: A
      real(c_double),target,dimension(:,:) :: B
      !
      ret = rocblas_get_matrix_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function rocblas_get_matrix_r8_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target :: A
      integer(c_int),value :: lda
      real(c_double),target :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_get_matrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_get_matrix_r8_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_double),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_get_matrix_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_set_vector_async_r8_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_double),target :: x
      integer(c_int),value :: incx
      real(c_double),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_vector_async_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_set_vector_async_r8_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_double),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_double),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_vector_async_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_get_vector_async_r8_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_double),target :: x
      integer(c_int),value :: incx
      real(c_double),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_vector_async_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_get_vector_async_r8_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      real(c_double),target,dimension(:) :: x
      integer(c_int),value :: incx
      real(c_double),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_vector_async_(n,8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_set_matrix_async_r8_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:,:) :: A
      real(c_double),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function rocblas_set_matrix_async_r8_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target :: A
      integer(c_int),value :: lda
      real(c_double),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_set_matrix_async_r8_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_double),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_get_matrix_async_r8_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:,:) :: A
      real(c_double),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,8,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function rocblas_get_matrix_async_r8_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target :: A
      integer(c_int),value :: lda
      real(c_double),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_get_matrix_async_r8_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      real(c_double),target,dimension(:) :: A
      integer(c_int),value :: lda
      real(c_double),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function
    function rocblas_set_vector_c4_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_set_vector_(n,2*4,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_set_vector_c4_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_set_vector_(n,2*4,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_get_vector_c4_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_get_vector_(n,2*4,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_get_vector_c4_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_get_vector_(n,2*4,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_set_matrix_c4_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:,:) :: A
      complex(c_float_complex),target,dimension(:,:) :: B
      !
      ret = rocblas_set_matrix_(rows,cols,2*4,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function rocblas_set_matrix_c4_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_set_matrix_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_set_matrix_c4_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_set_matrix_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_get_matrix_c4_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:,:) :: A
      complex(c_float_complex),target,dimension(:,:) :: B
      !
      ret = rocblas_get_matrix_(rows,cols,2*4,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function rocblas_get_matrix_c4_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_get_matrix_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_get_matrix_c4_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_get_matrix_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_set_vector_async_c4_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_vector_async_(n,2*4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_set_vector_async_c4_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_vector_async_(n,2*4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_get_vector_async_c4_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_vector_async_(n,2*4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_get_vector_async_c4_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_float_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_float_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_vector_async_(n,2*4,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_set_matrix_async_c4_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:,:) :: A
      complex(c_float_complex),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,2*4,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function rocblas_set_matrix_async_c4_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_set_matrix_async_c4_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_get_matrix_async_c4_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:,:) :: A
      complex(c_float_complex),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,2*4,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function rocblas_get_matrix_async_c4_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_get_matrix_async_c4_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,2*4,c_loc(A),lda,c_loc(B),ldb,stream)
    end function
    function rocblas_set_vector_c8_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_set_vector_(n,2*8,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_set_vector_c8_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_set_vector_(n,2*8,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_get_vector_c8_rank_0(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_get_vector_(n,2*8,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_get_vector_c8_full_rank(n,x,incx,y,incy) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      !
      ret = rocblas_get_vector_(n,2*8,c_loc(x),incx,c_loc(y),incy)
    end function

    function rocblas_set_matrix_c8_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:,:) :: A
      complex(c_double_complex),target,dimension(:,:) :: B
      !
      ret = rocblas_set_matrix_(rows,cols,2*8,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function rocblas_set_matrix_c8_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_set_matrix_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_set_matrix_c8_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_set_matrix_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_get_matrix_c8_full_rank(rows,cols,A,B) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:,:) :: A
      complex(c_double_complex),target,dimension(:,:) :: B
      !
      ret = rocblas_get_matrix_(rows,cols,2*8,c_loc(A),size(A,1),c_loc(B),size(B,1))
    end function

    function rocblas_get_matrix_c8_rank_0(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_get_matrix_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_get_matrix_c8_rank_1(rows,cols,A,lda,B,ldb) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      !
      ret = rocblas_get_matrix_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb)
    end function

    function rocblas_set_vector_async_c8_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_vector_async_(n,2*8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_set_vector_async_c8_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_vector_async_(n,2*8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_get_vector_async_c8_rank_0(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_vector_async_(n,2*8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_get_vector_async_c8_full_rank(n,x,incx,y,incy,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: n
      complex(c_double_complex),target,dimension(:) :: x
      integer(c_int),value :: incx
      complex(c_double_complex),target,dimension(:) :: y
      integer(c_int),value :: incy
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_vector_async_(n,2*8,c_loc(x),incx,c_loc(y),incy,stream)
    end function

    function rocblas_set_matrix_async_c8_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:,:) :: A
      complex(c_double_complex),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,2*8,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function rocblas_set_matrix_async_c8_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_set_matrix_async_c8_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_set_matrix_async_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_get_matrix_async_c8_full_rank(rows,cols,A,B,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:,:) :: A
      complex(c_double_complex),target,dimension(:,:) :: B
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,2*8,c_loc(A),size(A,1),c_loc(B),size(B,1),stream)
    end function

    function rocblas_get_matrix_async_c8_rank_0(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function

    function rocblas_get_matrix_async_c8_rank_1(rows,cols,A,lda,B,ldb,stream) result(ret)
      use iso_c_binding
      use hipfort_rocblas_enums
      implicit none
      integer(kind(rocblas_status_success)) :: ret
      integer(c_int),value :: rows
      integer(c_int),value :: cols
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int),value :: lda
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: stream
      !
      ret = rocblas_get_matrix_async_(rows,cols,2*8,c_loc(A),lda,c_loc(B),ldb,stream)
    end function
 

#endif
end module