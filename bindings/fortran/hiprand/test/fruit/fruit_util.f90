
! Copyright (c) 2005-2010, 2012-2013, Andrew Hang Chen and contributors,
! All rights reserved.
! Licensed under the 3-clause BSD license.

module fruit_util
  private

  public :: equals, to_s, strip

  interface equals
     module procedure equalEpsilon
     module procedure floatEqual
     module procedure integerEqual
     module procedure doublePrecisionEqual
     module procedure stringEqual
     module procedure logicalEqual
  end interface

  interface to_s
     module procedure to_s_int_
     module procedure to_s_real_
     module procedure to_s_logical_
     module procedure to_s_double_
     module procedure to_s_complex_
     module procedure to_s_double_complex_
     module procedure to_s_string_
  end interface

  interface strip
    module procedure strip_
    module procedure strip_length_
  end interface
contains

  function to_s_int_ (value)
    implicit none
    character(len=500):: to_s_int_
    integer, intent(in) :: value
    character(len=500) :: result
    write (result, *) value
    to_s_int_ = adjustl(trim(result))
  end function to_s_int_

  function to_s_real_ (value)
    implicit none
    character(len=500):: to_s_real_
    real, intent(in) :: value
    character(len=500) :: result
    write (result, *) value
    to_s_real_ = adjustl(trim(result))
  end function to_s_real_

  function to_s_double_ (value)
    implicit none
    character(len=500):: to_s_double_
    double precision, intent(in) :: value
    character(len=500) :: result
    write (result, *) value
    to_s_double_ = adjustl(trim(result))
  end function to_s_double_

  function to_s_complex_ (value)
    implicit none
    character(len=500):: to_s_complex_
    complex, intent(in) :: value
    character(len=500) :: result
    write (result, *) value
    to_s_complex_ = adjustl(trim(result))
  end function to_s_complex_

  function to_s_double_complex_ (value)
    implicit none
    character(len=500):: to_s_double_complex_
    complex(kind=kind(1.0D0)), intent(in) :: value
    character(len=500) :: result
    write (result, *) value
    to_s_double_complex_ = adjustl(trim(result))
  end function to_s_double_complex_

  function to_s_logical_ (value)
    implicit none
    character(len=500):: to_s_logical_
    logical, intent(in) :: value
    character(len=500) :: result
    write (result, *) value
    to_s_logical_ = adjustl(trim(result))
  end function to_s_logical_

  function to_s_string_ (value)
    implicit none
    character(len=500):: to_s_string_
    character(len=*), intent(in) :: value
    to_s_string_ = value
  end function to_s_string_

  function strip_(value)
    implicit none
    character(len=500):: strip_
    character(len=*), intent(in) :: value
    strip_ = trim(adjustl(value))
  end function strip_

  function strip_length_(value, length)
    implicit none
    character(len=*), intent(in) :: value
    integer, intent(in) :: length
    character(len= length):: strip_length_
    strip_length_ = trim(adjustl(value))
  end function strip_length_

  !------------------------
  ! test if 2 values are close
  !------------------------
  !logical function equals (number1, number2)
  !  real,  intent (in) :: number1, number2
  !
  !  return equalEpsilon (number1, number2, epsilon(number1))
  !
  !end function equals


  function equalEpsilon (number1, number2, epsilon ) result (resultValue)
    real , intent (in) :: number1, number2, epsilon
    logical :: resultValue

    resultValue = .false.

    ! test very small number1
    if ( abs(number1) < epsilon .and.  abs(number1 - number2) < epsilon ) then
       resultValue = .true.
    else
       if ((abs(( number1 - number2)) / number1) < epsilon ) then
          resultValue = .true.
       else
          resultValue = .false.
       end if
    end if

  end function equalEpsilon

  function floatEqual (number1, number2 ) result (resultValue)
    real , intent (in) :: number1, number2
    real :: epsilon
    logical :: resultValue

    resultValue = .false.
    epsilon = 1E-6

    ! test very small number1
    if ( abs(number1) < epsilon .and.  abs(number1 - number2) < epsilon ) then
       resultValue = .true.
    else
       if ((abs(( number1 - number2)) / number1) < epsilon ) then
          resultValue = .true.
       else
          resultValue = .false.
       end if
    end if
  end function floatEqual

  function doublePrecisionEqual (number1, number2 ) result (resultValue)
    double precision , intent (in) :: number1, number2
    real :: epsilon
    logical :: resultValue

    resultValue = .false.
    epsilon = 1E-6
    !epsilon = epsilon (number1)

    ! test very small number1
    if ( abs(number1) < epsilon .and.  abs(number1 - number2) < epsilon ) then
       resultValue = .true.
    else
       if ((abs(( number1 - number2)) / number1) < epsilon ) then
          resultValue = .true.
       else
          resultValue = .false.
       end if
    end if
  end function doublePrecisionEqual

  function integerEqual (number1, number2 ) result (resultValue)
    integer , intent (in) :: number1, number2
    logical :: resultValue

    resultValue = .false.

    if ( number1 .eq. number2 ) then
       resultValue = .true.
    else
       resultValue = .false.
    end if
  end function integerEqual

  function stringEqual (str1, str2 ) result (resultValue)
    character(*) , intent (in) :: str1, str2
    logical :: resultValue

    resultValue = .false.

    if ( str1 .eq. str2 ) then
       resultValue = .true.
    end if
  end function stringEqual

  function logicalEqual (l1, l2 ) result (resultValue)
    logical, intent (in) :: l1, l2
    logical              :: resultValue

    resultValue = .false.

    if ( l1 .eqv. l2 ) then
       resultValue = .true.
    end if
  end function logicalEqual

end module fruit_util
