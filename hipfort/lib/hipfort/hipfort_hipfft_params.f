module hipfort_hipfft_params
  use iso_c_binding
  implicit none

  integer(c_int), parameter, public :: HIPFFT_FORWARD = -1, HIPFFT_BACKWARD = 1
  integer(c_int), parameter, public :: HIPFFT_INVERSE = 1 

end module hipfort_hipfft_params
