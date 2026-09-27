program hip_cgemm_batched

  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer(kind(HIPBLAS_OP_N)), parameter :: transa = HIPBLAS_OP_N, transb = HIPBLAS_OP_N
  complex(kind=4), parameter :: alpha = (1.1,0.0), beta = (0.9,0.0)

  integer, parameter :: m = 256, n = 256, k = 256, batch_count = 4
  integer, parameter :: bytes_per_element = 8 ! 2x float
  integer, parameter :: size_a = m*k, size_b = k*n, size_c = m*n
  integer(c_size_t), parameter :: Nabytes = size_a*bytes_per_element, &
                                  Nbbytes = size_b*bytes_per_element, &
                                  Ncbytes = size_c*bytes_per_element
  integer :: lda, ldb, ldc, i, b

  complex(kind=4), allocatable, target, dimension(:,:) :: ha, hb, hc ! (elems, batch)
  complex(kind=4), allocatable, dimension(:) :: hc_exact ! one value per batch

  ! Host arrays of device pointers (one device matrix per batch entry)
  type(c_ptr), target :: da(batch_count), db(batch_count), dc(batch_count)
  ! Device-resident pointer arrays passed to the batched routine
  type(c_ptr) :: da_p = c_null_ptr, db_p = c_null_ptr, dc_p = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr

  double precision :: error
  ! epsilon of the data kind, not the accumulator: these results are complex(kind=4).
  double precision, parameter :: error_max = 10*epsilon(real(1.0,kind=4))

  write(*,"(a)",advance="no") "-- Running test 'CGEMM_BATCHED' (Fortran 2003 interfaces) - "

  ! hipBLAS defaults to host pointer mode: no set-pointer-mode call needed
  call hipblasCheck(hipblasCreate(handle))

  lda = m; ldb = k; ldc = m

  allocate(ha(size_a,0:batch_count-1))
  allocate(hb(size_b,0:batch_count-1))
  allocate(hc(size_c,0:batch_count-1))
  allocate(hc_exact(0:batch_count-1))

  ! Constant matrices with a distinct per-batch value so the exact answer is
  ! a distinct constant per batch. A is held constant; B varies per batch.
  ha(:,:) = (1.0,0.0)
  do b = 0, batch_count-1
     hb(:,b) = cmplx(b+1,0.0)
     hc(:,b) = (3.0,0.0)
     hc_exact(b) = alpha*k*cmplx(b+1,0.0) + beta*(3.0,0.0)
  end do

  ! Allocate one device matrix per batch entry and copy its host data across
  do b = 1, batch_count
     call hipCheck(hipMalloc(da(b),Nabytes))
     call hipCheck(hipMalloc(db(b),Nbbytes))
     call hipCheck(hipMalloc(dc(b),Ncbytes))
     call hipCheck(hipMemcpy(da(b), c_loc(ha(1,b-1)), Nabytes, hipMemcpyHostToDevice))
     call hipCheck(hipMemcpy(db(b), c_loc(hb(1,b-1)), Nbbytes, hipMemcpyHostToDevice))
     call hipCheck(hipMemcpy(dc(b), c_loc(hc(1,b-1)), Ncbytes, hipMemcpyHostToDevice))
  end do

  ! Allocate the device-resident pointer arrays and copy the host pointer
  ! arrays into them (hipBLAS requires the pointer array in device memory)
  call hipCheck(hipMalloc(da_p, int(batch_count,c_size_t)*c_sizeof(da(1))))
  call hipCheck(hipMalloc(db_p, int(batch_count,c_size_t)*c_sizeof(db(1))))
  call hipCheck(hipMalloc(dc_p, int(batch_count,c_size_t)*c_sizeof(dc(1))))
  call hipCheck(hipMemcpy(da_p, c_loc(da(1)), int(batch_count,c_size_t)*c_sizeof(da(1)), hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(db_p, c_loc(db(1)), int(batch_count,c_size_t)*c_sizeof(db(1)), hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dc_p, c_loc(dc(1)), int(batch_count,c_size_t)*c_sizeof(dc(1)), hipMemcpyHostToDevice))

  call hipblasCheck(hipblasCgemmBatched(handle,transa,transb,m,n,k, &
       alpha,da_p,lda,db_p,ldb,beta,dc_p,ldc,batch_count))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer each batch result back to host memory
  do b = 1, batch_count
     call hipCheck(hipMemcpy(c_loc(hc(1,b-1)), dc(b), Ncbytes, hipMemcpyDeviceToHost))
  end do

  do b = 0, batch_count-1
     do i = 1, size_c
        error = abs((hc_exact(b) - hc(i,b))/hc_exact(b))
        if( error > error_max )then
           write(*,*) "FAILED! Error bigger than max! batch = ", b, " error = ", error
           call exit(1)
        end if
     end do
  end do

  do b = 1, batch_count
     call hipCheck(hipFree(da(b)))
     call hipCheck(hipFree(db(b)))
     call hipCheck(hipFree(dc(b)))
  end do
  call hipCheck(hipFree(da_p))
  call hipCheck(hipFree(db_p))
  call hipCheck(hipFree(dc_p))

  call hipblasCheck(hipblasDestroy(handle))

  deallocate(ha)
  deallocate(hb)
  deallocate(hc)
  deallocate(hc_exact)

  write(*,*) "PASSED!"

end program hip_cgemm_batched
