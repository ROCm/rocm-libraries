! Demonstrates rocfft_get_version_string, the rocFFT library version query.
program rocfft_version
  use iso_c_binding
  use hip
  use rocfft

  implicit none

  ! rocfft_get_version_string requires a buffer of at least 30 characters.
  integer(c_size_t), parameter :: buflen = 64

  character(kind=c_char), target :: buf(buflen)
  character(kind=c_char, len=buflen) :: version
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'rocFFT version string' &
                              &(Fortran 2003 interfaces) - "

  call rocfftCheck(rocfft_setup())

  ! The C API writes a NUL-terminated string into the buffer, so pass the
  ! address of the first element and the buffer capacity.
  buf(:) = c_null_char
  call rocfftCheck(rocfft_get_version_string(c_loc(buf(1)), buflen))

  call rocfftCheck(rocfft_cleanup())

  ! Copy the C string into a Fortran character variable, stopping at the NUL.
  version = c_char_''
  do i = 1, buflen
     if (buf(i) == c_null_char) exit
     version(i:i) = buf(i)
  end do

  if (len_trim(version) == 0) then
     write(*,*) "FAILED! rocfft_get_version_string returned an empty string"
     STOP 1
  end if

  write(*,*) "PASSED! rocFFT version: ", trim(version)

end program rocfft_version
