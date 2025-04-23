module hip_blas
  use hip_blas_enums
  implicit none
  interface

     function hipblasCreate(handle) bind(c, name="hipblasCreate")
       use iso_c_binding
       use hip_blas_enums
       implicit none
       integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCreate
       type(c_ptr) :: handle
     end function hipblasCreate

     function hipblasDestroy(handle) bind(c, name="hipblasDestroy")
       use iso_c_binding
       use hip_blas_enums
       implicit none
       integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDestroy
       type(c_ptr), value :: handle
     end function hipblasDestroy

     function hipblasDscal(handle,n,alpha,x,incx) bind(c,name="hipblasDscal")
       use iso_c_binding
       use hip_blas_enums
       implicit none
       integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDscal
       type(c_ptr), value :: handle
       integer, value :: n
       double precision, intent(in) :: alpha
       type(c_ptr),value :: x
       integer, value :: incx
     end function hipblasDscal

     function hipblasDger(handle,m,n,alpha,x,incx,y,incy,A,lda) bind(c,name="hipblasDger")
       use iso_c_binding
       use hip_blas_enums
       implicit none
       integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDger
       type(c_ptr), value :: handle
       integer, value :: m, n
       double precision, intent(in) :: alpha
       type(c_ptr),value :: x, y
       integer, value :: incx, incy
       type(c_ptr),value :: A
       integer, value :: lda
     end function hipblasDger

     function hipblasDgemm(handle, &
                           transa,transb,&
                           m,n,k,&
                           alpha,A,lda,&
                           B,ldb,beta,&
                           C,ldc ) bind(c,name="hipblasDgemm")
       use iso_c_binding
       use hip_blas_enums
       implicit none
       integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgemm
       type(c_ptr), value :: handle
       integer(kind(HIPBLAS_OP_N)),value :: transa,transb
       integer, value :: m, n, k, lda, ldb, ldc
       double precision, intent(in) :: alpha, beta
       type(c_ptr),value :: A,B
       type(c_ptr),value :: C
     end function hipblasDgemm

  end interface

  contains
     subroutine hipblasCheck(hipblasError_t)
       use hip_blas_enums
       implicit none
       integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasError_t
       if (hipblasError_t /= HIPBLAS_STATUS_SUCCESS) then
          write(*,*) "HIPBLAS ERROR: Error code = ", hipblasError_t
          call exit(hipblasError_t)
       end if
     end subroutine hipblasCheck

end module hip_blas
