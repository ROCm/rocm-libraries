program fortran_hip
  use iso_c_binding
  use hipfort
  use hipfort_check
  use hipfort_roctx

  implicit none

  interface
     subroutine launch() bind(c)
       use iso_c_binding
       implicit none
     end subroutine
  end interface

  integer :: i
  integer :: ret
  type(hipDeviceProp_t),target :: props

  call hipCheck(hipGetDeviceProperties(props,0))
  write(*,"(a)",advance="no") "-- Running test 'roctx' (Fortran 2003 interfaces)"
  write(*,"(a)",advance="no") "- device: "
  i=1
  do while ( iachar(props%name(i)) .ne. 0 ) ! print till end char
    write(*,"(a)",advance="no") props%name(i)
    i = i+1
  end do
  write(*,"(a)",advance="no") " - "

  ret = roctxRangePush(c_char_"hello_world"//c_null_char)
  if (ret /= 0) then
    write (*, *) "ROCTX ERROR: roctxRangePush: Invalid nested range level ", ret
  end if

  call launch()
  call hipCheck(hipDeviceSynchronize())

  ret = roctxRangePop()
  if (ret /= 0) then
    write (*, *) "ROCTX ERROR: roctxRangePop: Invalid nested range level ", ret
  end if

  write(*,*) "PASSED!"

end program fortran_hip
