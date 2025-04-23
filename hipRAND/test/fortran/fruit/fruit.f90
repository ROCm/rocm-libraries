
! Copyright (c) 2005-2010, 2012-2013, Andrew Hang Chen and contributors,
! All rights reserved.
! Licensed under the 3-clause BSD license.

!------------------------
! FORTRAN unit test utility
!
! Author: Andrew H. Chen meihome @at@ gmail.com
!------------------------
!
! Unit test framework for FORTRAN.  (FoRtran UnIT)
!
! This package is to perform unit test for FORTRAN subroutines
!
! The method used most are: assert_true, assert_equals
!
! Coding convention:
!   1) All methods must be exposed by interface.  i.e. interface init_fruit
!   2) Variable and methods are lower case connected with underscores.  i.e. init_fruit, and
!      failed_assert_count
!
module fruit
  use fruit_util
  implicit none
  private

  integer, parameter :: STDOUT_DEFAULT = 6
  integer :: stdout   = STDOUT_DEFAULT

  integer, parameter :: XML_OPEN = 20
  integer, parameter :: XML_WORK_DEFAULT = 21
  integer :: xml_work = XML_WORK_DEFAULT
  character (len = *), parameter :: xml_filename     = "result.xml"
  character (len = *), parameter :: XML_FN_WORK_DEF = "result_tmp.xml"
  character (len = 50) :: xml_filename_work = XML_FN_WORK_DEF

  integer, parameter :: STRLEN_T = 12

  integer, parameter :: NUMBER_LENGTH = 10

  integer, parameter :: MSG_LENGTH = 256
  integer, parameter :: MAX_MSG_STACK_SIZE = 2000
  integer, parameter :: MSG_ARRAY_INCREMENT = 50
  integer, parameter :: MAX_MARKS_PER_LINE = 78

  character(*), parameter :: DEFAULT_CASE_NAME = '_not_set_'

  !---------- save ----------
  integer, private, save :: successful_assert_count = 0
  integer, private, save :: failed_assert_count = 0

  integer, private, save :: message_index = 1
  integer, private, save :: message_index_from = 1
  integer, private, save :: current_max = 50

  character (len = MSG_LENGTH), private, allocatable :: message_array(:)
  character (len = MSG_LENGTH), private, save :: msg = '[unit name not set from set_name]: '
  character (len = MSG_LENGTH), private, save :: case_name  = DEFAULT_CASE_NAME

  integer, private, save :: successful_case_count = 0
  integer, private, save :: failed_case_count = 0
  integer, private, save :: testCaseIndex = 1
  logical, private, save :: last_passed = .false.
  logical, private, save :: case_passed = .false.
  integer, private, save :: case_time_from = 0
  integer, private, save :: linechar_count = 0

  integer, parameter :: FRUIT_PREFIX_LEN_MAX = 50
  character(len = FRUIT_PREFIX_LEN_MAX) :: prefix = ""
  !---------- save ----------

  type ty_stack
    integer :: successful_assert_count
    integer :: failed_assert_count

    integer :: message_index
    integer :: message_index_from
    integer :: current_max

    character (len = MSG_LENGTH), pointer :: message_array(:)
    character (len = MSG_LENGTH) :: case_name !  = DEFAULT_CASE_NAME

    integer :: successful_case_count
    integer :: failed_case_count
    integer :: testCaseIndex
    logical :: last_passed
    logical :: case_passed
    integer :: case_time_from
    integer :: linechar_count
  end type ty_stack

  type(ty_stack), save :: stashed_suite

  public :: &
    init_fruit
  public :: &
    get_last_message, &
    is_last_passed, &
    is_case_passed, &
    add_success, addSuccess, &
    set_unit_name, get_unit_name, &
    set_case_name, get_case_name, &
    failed_assert_action, get_total_count, getTotalCount, &
    get_failed_count, getFailedCount, is_all_successful, isAllSuccessful, &
    run_test_case, runTestCase
  public :: assert_equals,     assertEquals
  public :: assert_not_equals, assertNotEquals
  public :: assert_true,       assertTrue
  public :: stash_test_suite, restore_test_suite
  public :: FRUIT_PREFIX_LEN_MAX
  public :: override_xml_work, end_override_xml_work
  public :: get_assert_and_case_count
  public :: fruit_summary_table

  public :: initializeFruit
  interface initializeFruit
     module procedure obsolete_initializeFruit_
  end interface

  public :: getTestSummary
  interface getTestSummary
     module procedure obsolete_getTestSummary_
  end interface

  interface assertTrue
     module procedure obsolete_assert_true_logical_
  end interface

  public ::          assert_false
  interface          assert_false
    module procedure assert_false_
  end interface

  interface assert_equals
  !====== begin of generated interface ======
    module procedure assert_eq_logical_
    module procedure assert_eq_1d_logical_
    module procedure assert_eq_2d_logical_
    module procedure assert_eq_string_
    module procedure assert_eq_1d_string_
    module procedure assert_eq_2d_string_
    module procedure assert_eq_int_
    module procedure assert_eq_1d_int_
    module procedure assert_eq_2d_int_
    module procedure assert_eq_real_
    module procedure assert_eq_real_in_range_
    module procedure assert_eq_1d_real_
    module procedure assert_eq_1d_real_in_range_
    module procedure assert_eq_2d_real_
    module procedure assert_eq_2d_real_in_range_
    module procedure assert_eq_double_
    module procedure assert_eq_double_in_range_
    module procedure assert_eq_1d_double_
    module procedure assert_eq_1d_double_in_range_
    module procedure assert_eq_2d_double_
    module procedure assert_eq_2d_double_in_range_
    module procedure assert_eq_complex_
    module procedure assert_eq_complex_in_range_
    module procedure assert_eq_1d_complex_
    module procedure assert_eq_1d_complex_in_range_
    module procedure assert_eq_2d_complex_
    module procedure assert_eq_2d_complex_in_range_
  !====== end of generated inteface ======
  end interface

  interface assertEquals
  !====== begin of generated interface ======
    module procedure assert_eq_logical_
    module procedure assert_eq_1d_logical_
    module procedure assert_eq_2d_logical_
    module procedure assert_eq_string_
    module procedure assert_eq_1d_string_
    module procedure assert_eq_2d_string_
    module procedure assert_eq_int_
    module procedure assert_eq_1d_int_
    module procedure assert_eq_2d_int_
    module procedure assert_eq_real_
    module procedure assert_eq_real_in_range_
    module procedure assert_eq_1d_real_
    module procedure assert_eq_1d_real_in_range_
    module procedure assert_eq_2d_real_
    module procedure assert_eq_2d_real_in_range_
    module procedure assert_eq_double_
    module procedure assert_eq_double_in_range_
    module procedure assert_eq_1d_double_
    module procedure assert_eq_1d_double_in_range_
    module procedure assert_eq_2d_double_
    module procedure assert_eq_2d_double_in_range_
    module procedure assert_eq_complex_
    module procedure assert_eq_complex_in_range_
    module procedure assert_eq_1d_complex_
    module procedure assert_eq_1d_complex_in_range_
    module procedure assert_eq_2d_complex_
    module procedure assert_eq_2d_complex_in_range_
  !====== end of generated inteface ======
  end interface

  interface assert_not_equals
  !====== begin of generated interface ======
    module procedure assert_not_equals_logical_
    module procedure assert_not_equals_1d_logical_
    module procedure assert_not_equals_2d_logical_
    module procedure assert_not_equals_string_
    module procedure assert_not_equals_1d_string_
    module procedure assert_not_equals_2d_string_
    module procedure assert_not_equals_int_
    module procedure assert_not_equals_1d_int_
    module procedure assert_not_equals_2d_int_
    module procedure assert_not_equals_real_
    module procedure assert_not_equals_real_in_range_
    module procedure assert_not_equals_1d_real_
    module procedure assert_not_equals_1d_real_in_range_
    module procedure assert_not_equals_2d_real_
    module procedure assert_not_equals_2d_real_in_range_
    module procedure assert_not_equals_double_
    module procedure assert_not_equals_double_in_range_
    module procedure assert_not_equals_1d_double_
    module procedure assert_not_equals_1d_double_in_range_
    module procedure assert_not_equals_2d_double_
    module procedure assert_not_equals_2d_double_in_range_
    module procedure assert_not_equals_complex_
    module procedure assert_not_equals_complex_in_range_
    module procedure assert_not_equals_1d_complex_
    module procedure assert_not_equals_1d_complex_in_range_
    module procedure assert_not_equals_2d_complex_
    module procedure assert_not_equals_2d_complex_in_range_
  !====== end of generated inteface ======

  end interface

  interface assertNotEquals
  !====== begin of generated interface ======
    module procedure assert_not_equals_logical_
    module procedure assert_not_equals_1d_logical_
    module procedure assert_not_equals_2d_logical_
    module procedure assert_not_equals_string_
    module procedure assert_not_equals_1d_string_
    module procedure assert_not_equals_2d_string_
    module procedure assert_not_equals_int_
    module procedure assert_not_equals_1d_int_
    module procedure assert_not_equals_2d_int_
    module procedure assert_not_equals_real_
    module procedure assert_not_equals_real_in_range_
    module procedure assert_not_equals_1d_real_
    module procedure assert_not_equals_1d_real_in_range_
    module procedure assert_not_equals_2d_real_
    module procedure assert_not_equals_2d_real_in_range_
    module procedure assert_not_equals_double_
    module procedure assert_not_equals_double_in_range_
    module procedure assert_not_equals_1d_double_
    module procedure assert_not_equals_1d_double_in_range_
    module procedure assert_not_equals_2d_double_
    module procedure assert_not_equals_2d_double_in_range_
    module procedure assert_not_equals_complex_
    module procedure assert_not_equals_complex_in_range_
    module procedure assert_not_equals_1d_complex_
    module procedure assert_not_equals_1d_complex_in_range_
    module procedure assert_not_equals_2d_complex_
    module procedure assert_not_equals_2d_complex_in_range_
  !====== end of generated inteface ======

  end interface

  interface addSuccess
     module procedure obsolete_addSuccess_
  end interface

  public ::           add_fail
  interface           add_fail
     module procedure add_fail_
     module procedure add_fail_unit_
  end interface

  public ::           addFail
  interface           addFail
     module procedure add_fail_
     module procedure add_fail_unit_
  end interface

  interface getTotalCount
     module procedure obsolete_getTotalCount_
  end interface

  interface getFailedCount
     module procedure obsolete_getFailedCount_
  end interface

  interface isAllSuccessful
     module procedure obsolete_isAllSuccessful_
  end interface

  interface run_test_case
     module procedure run_test_case_
     module procedure run_test_case_named_
  end interface

  interface runTestCase
     module procedure run_test_case_
     module procedure run_test_case_named_
  end interface

  public ::          init_fruit_xml
  interface          init_fruit_xml
    module procedure init_fruit_xml_
  end interface

  public ::          fruit_summary
  interface          fruit_summary
    module procedure fruit_summary_
  end interface

  public ::          fruit_summary_xml
  interface          fruit_summary_xml
    module procedure fruit_summary_xml_
  end interface

  public ::          case_passed_xml
  interface          case_passed_xml
    module procedure case_passed_xml_
  end interface

  public ::          case_failed_xml
  interface          case_failed_xml
    module procedure case_failed_xml_
  end interface

  public ::          override_stdout
  interface          override_stdout
    module procedure override_stdout_
  end interface

  public ::          end_override_stdout
  interface          end_override_stdout
    module procedure end_override_stdout_
  end interface

  interface          override_xml_work
    module procedure override_xml_work_
  end interface

  interface          end_override_xml_work
    module procedure end_override_xml_work_
  end interface

  public ::          get_xml_filename_work
  interface          get_xml_filename_work
    module procedure get_xml_filename_work_
  end interface

  public ::          set_xml_filename_work
  interface          set_xml_filename_work
    module procedure set_xml_filename_work_
  end interface

  public ::          get_message_index
  interface          get_message_index
    module procedure get_message_index_
  end interface

  public ::          get_messages
  interface          get_messages
    module procedure get_messages_
  end interface

  public ::          get_message_array
  interface          get_message_array
    module procedure get_message_array_
  end interface

  interface          set_unit_name
    module procedure set_case_name_
  end interface
  interface          set_case_name
    module procedure set_case_name_
  end interface

  interface          get_unit_name
    module procedure get_case_name_
  end interface
  interface          get_case_name
    module procedure get_case_name_
  end interface

  public ::          fruit_finalize
  interface          fruit_finalize
    module procedure fruit_finalize_
  end interface

  public ::          set_prefix
  interface          set_prefix
    module procedure set_prefix_
  end interface

  public ::          get_prefix
  interface          get_prefix
    module procedure get_prefix_
  end interface

  interface          get_assert_and_case_count
    module procedure get_assert_and_case_count_
  end interface

  interface          fruit_summary_table
    module procedure fruit_summary_table_
  end interface
contains

  subroutine init_fruit(rank)
    integer, intent(in), optional :: rank
    logical :: if_write

    successful_assert_count = 0
    failed_assert_count = 0
    message_index = 1
    message_index_from = 1

    if_write = .true.
    if (present(rank)) then
      if (rank /= 0) if_write = .false.
    endif

    if (if_write) then
      write (stdout,*)
      write (stdout,*) "Test module initialized"
      write (stdout,*)
      write (stdout,*) "   . : successful assert,   F : failed assert "
      write (stdout,*)
    endif
    if ( .not. allocated(message_array) ) then
      allocate(message_array(MSG_ARRAY_INCREMENT))
    end if
  end subroutine init_fruit

  subroutine fruit_finalize_
    if (allocated(message_array)) then
      deallocate(message_array)
    endif
    if(failed_case_count > 0) then
      STOP 1
    endif
  end subroutine fruit_finalize_

  subroutine init_fruit_xml_(rank)
    integer, optional, intent(in) :: rank
    logical :: rank_zero_or_single

    rank_zero_or_single = .true.
    if (present(rank)) then
      if (rank /= 0) then
        rank_zero_or_single = .false.
      endif
    endif


    if (rank_zero_or_single) then
      open (XML_OPEN, file = xml_filename, action ="write", status = "replace")
        write(XML_OPEN, '("<?xml version=""1.0"" encoding=""UTF-8""?>")')
        write(XML_OPEN, '("<testsuites>")')
        write(XML_OPEN, '("  <testsuite ")', advance = "no")
        write(XML_OPEN, '(      "errors=""0"" "   )', advance = "no")
        write(XML_OPEN, '(      "tests=""1"" "    )', advance = "no")
        write(XML_OPEN, '(      "failures=""1"" " )', advance = "no")
        write(XML_OPEN, '(      "name=""", a, """ ")', advance = "no") "name of test suite"
        write(XML_OPEN, '(      "id=""1"">")')

        write(XML_OPEN, &
        &  '("    <testcase name=""", a, """ classname=""", a, """ time=""", a, """>")') &
        &  "dummy_testcase", "dummy_classname", "0"

        write(XML_OPEN, '(a)', advance = "no") "      <failure type=""failure"" message="""
        write(XML_OPEN, '(a)', advance = "no") "FRUIT did not generate regular content of result.xml."
        write(XML_OPEN, '(a)')                 """/>"
        write(XML_OPEN, '("    </testcase>")')

        write(XML_OPEN, '("  </testsuite>")')
        write(XML_OPEN, '("</testsuites>")')
      close(XML_OPEN)
    endif

    open (xml_work, FILE = xml_filename_work, action ="write", status='replace')
    close(xml_work)
  end subroutine init_fruit_xml_

  function  case_delta_t()
    character(len = STRLEN_T) :: case_delta_t
    real :: delta_t
    integer :: case_time_to, time_rate, time_max

    call system_clock(case_time_to, time_rate, time_max)
    if (time_rate > 0) then
      delta_t = real(case_time_to - case_time_from) / real(time_rate)
      if (delta_t < 0) then
        delta_t = delta_t + real(time_max) / real(time_rate)
      endif
    else
      delta_t = 0
    endif

    write(case_delta_t, '(g12.4)') delta_t
    case_delta_t = adjustl(case_delta_t)
  end function case_delta_t

  subroutine case_passed_xml_(tc_name, classname)
    character(*), intent(in) :: tc_name
    character(*), intent(in) :: classname
    character(len = STRLEN_T) :: case_time

    case_time = case_delta_t()

    open (xml_work, FILE = xml_filename_work, position='append')
    write(xml_work, &
   &  '("    <testcase name=""", a, """ classname=""", a, a, """ time=""", a, """/>")') &
   &  trim(tc_name), trim(prefix), trim(classname), trim(case_time)
    close(xml_work)
  end subroutine case_passed_xml_

  subroutine case_failed_xml_(tc_name, classname)
    character(*), intent(in) :: tc_name
    character(*), intent(in) :: classname
    integer :: i
    character(len = STRLEN_T) :: case_time

    case_time = case_delta_t()

    open (xml_work, FILE = xml_filename_work, position='append')
    write(xml_work, &
   &  '("    <testcase name=""", a, """ classname=""", a, a, """ time=""", a, """>")') &
   &  trim(tc_name), trim(prefix), trim(classname), trim(case_time)

    write(xml_work, '("      <failure type=""failure"" message=""")', advance = "no")
    do i = message_index_from, message_index - 1
      write(xml_work, '(a)', advance = "no") trim(strip(message_array(i)))
      if (i == message_index - 1) then
        continue
      else
        write(xml_work, '("&#xA;")', advance="no")
      endif
    enddo
    write(xml_work, '("""/>")')

    write(xml_work, &
   &  '("    </testcase>")')
    close(xml_work)
  end subroutine case_failed_xml_

  subroutine fruit_summary_xml_
    character(len = 1000) :: whole_line
    character(len = 100) :: full_count
    character(len = 100) :: fail_count

    full_count = int_to_str(successful_case_count + failed_case_count)
    fail_count = int_to_str(failed_case_count)

    open (XML_OPEN, file = xml_filename, action ="write", status = "replace")
    write(XML_OPEN, '("<?xml version=""1.0"" encoding=""UTF-8""?>")')
    write(XML_OPEN, '("<testsuites>")')
    write(XML_OPEN, '("  <testsuite errors=""0"" ")', advance = "no")
    write(XML_OPEN, '("tests=""", a, """ ")', advance = "no") &
    &  trim(full_count)
    write(XML_OPEN, '("failures=""", a, """ ")', advance = "no") &
    &  trim(fail_count)
    write(XML_OPEN, '("name=""", a, """ ")', advance = "no") &
    &  "name of test suite"
    write(XML_OPEN, '("id=""1"">")')

    open (xml_work, FILE = xml_filename_work)
    do
      read(xml_work, '(a)', end = 999) whole_line
      write(XML_OPEN, '(a)') trim(whole_line)
    enddo
999 continue
    close(xml_work)

    write(XML_OPEN, '("  </testsuite>")')
    write(XML_OPEN, '("</testsuites>")')
    close(XML_OPEN)
  end subroutine fruit_summary_xml_

  function int_to_str(i)
    integer, intent(in) :: i
    character(LEN = NUMBER_LENGTH) :: int_to_str

    write(int_to_str, '(i10)') i
    int_to_str = adjustl(int_to_str)
  end function int_to_str

  subroutine obsolete_initializeFruit_
    call obsolete_ ("initializeFruit is OBSOLETE.  replaced by init_fruit")
    call init_fruit
  end subroutine obsolete_initializeFruit_

  subroutine obsolete_getTestSummary_
    call obsolete_ ( "getTestSummary is OBSOLETE.  replaced by fruit_summary")
    call fruit_summary_
  end subroutine obsolete_getTestSummary_

  ! Run a named test case
  subroutine run_test_case_named_( tc, tc_name )
    interface
       subroutine tc()
       end subroutine
    end interface
    character(*), intent(in) :: tc_name

    integer :: initial_failed_assert_count

    initial_failed_assert_count = failed_assert_count

    ! Set the name of the unit test
    call set_case_name( tc_name )

    last_passed = .true.
    case_passed = .true.
    linechar_count = 0  !! reset linechar_count for each test case.
    message_index_from = message_index
    call system_clock(case_time_from)

    call tc()

    if ( initial_failed_assert_count .eq. failed_assert_count ) then
       ! If no additional assertions failed during the run of this test case
       ! then the test case was successful
       successful_case_count = successful_case_count+1
    else
       failed_case_count = failed_case_count+1
       case_passed = .false.
    end if

    testCaseIndex = testCaseIndex+1

    ! Reset the name of the unit test back to the default
    call set_case_name( DEFAULT_CASE_NAME )

  end subroutine run_test_case_named_

  ! Run an 'unnamed' test case
  subroutine run_test_case_( tc )
    interface
       subroutine tc()
       end subroutine
    end interface

    call run_test_case_named_( tc, '_unnamed_' )

  end subroutine run_test_case_


  subroutine fruit_summary_
    integer :: i

    write (stdout,*)
    write (stdout,*)
    write (stdout,*) '    Start of FRUIT summary: '
    write (stdout,*)

    if (failed_assert_count > 0) then
       write (stdout,*) 'Some tests failed!'
    else
       write (stdout,*) 'SUCCESSFUL!'
    end if

    write (stdout,*)
    if ( message_index > 1) then
       write (stdout,*) '  -- Failed assertion messages:'

       do i = 1, message_index - 1
          write (stdout,"(A)") '   '//trim(strip(message_array(i)))
       end do

       write (stdout,*) '  -- end of failed assertion messages.'
       write (stdout,*)
    else
       write (stdout,*) '  No messages '
    end if

    if (successful_assert_count + failed_assert_count /= 0) then
       call fruit_summary_table_(&
       & successful_assert_count, failed_assert_count, &
       & successful_case_count,   failed_case_count &
       &)
    end if
    write (stdout, *) '  -- end of FRUIT summary'
  end subroutine fruit_summary_

  subroutine fruit_summary_table_(&
  & succ_assert, fail_assert, &
  & succ_case  , fail_case    &
  &)
    integer, intent(in) :: succ_assert, fail_assert
    integer, intent(in) :: succ_case  , fail_case

    write (stdout,*) 'Total asserts :   ', succ_assert + fail_assert
    write (stdout,*) 'Successful    :   ', succ_assert
    write (stdout,*) 'Failed        :   ', fail_assert
    write (stdout,'("Successful rate:   ",f6.2,"%")')  real(succ_assert) * 100.0 / &
         real (succ_assert + fail_assert)
    write (stdout, *)
    write (stdout,*) 'Successful asserts / total asserts : [ ',&
         succ_assert, '/', succ_assert + fail_assert, ' ]'
    write (stdout,*) 'Successful cases   / total cases   : [ ', succ_case, '/', &
         succ_case + fail_case, ' ]'
  end subroutine fruit_summary_table_

  subroutine obsolete_addSuccess_
    call obsolete_ ("addSuccess is OBSOLETE.  replaced by add_success")
    call add_success
  end subroutine obsolete_addSuccess_

  subroutine add_fail_ (message)
    character (*), intent (in), optional :: message
    call failed_assert_action('none', 'none', message, if_is = .true.)
  end subroutine add_fail_

  subroutine add_fail_unit_ (unitName, message)
    character (*), intent (in) :: unitName
    character (*), intent (in) :: message

    call add_fail_ ("[in " //  unitName // "(fail)]: " //  message)
  end subroutine add_fail_unit_

  subroutine obsolete_isAllSuccessful_(result)
    logical, intent(out) :: result
    call obsolete_ ('subroutine isAllSuccessful is changed to function is_all_successful.')
    result = (failed_assert_count .eq. 0 )
  end subroutine obsolete_isAllSuccessful_

  subroutine is_all_successful(result)
    logical, intent(out) :: result
    result= (failed_assert_count .eq. 0 )
  end subroutine is_all_successful

  ! Private, helper routine to wrap lines of success/failed marks
  subroutine output_mark_( chr )
    character(1), intent(in) :: chr
  !!  integer, save :: linechar_count = 0
  !!  Definition of linechar_count is moved to module,
  !!  so that it can be stashed and restored.

    linechar_count = linechar_count + 1
    if ( linechar_count .lt. MAX_MARKS_PER_LINE ) then
       write(stdout,"(A1)",ADVANCE='NO') chr
    else
       write(stdout,"(A1)",ADVANCE='YES') chr
       linechar_count = 0
    endif

  end subroutine output_mark_

  subroutine success_mark_
    call output_mark_( '.' )
  end subroutine success_mark_

  subroutine failed_mark_
    call output_mark_( 'F' )
  end subroutine failed_mark_

  subroutine increase_message_stack_
    character(len=MSG_LENGTH) :: msg_swap_holder(current_max)

    if (message_index > MAX_MSG_STACK_SIZE) then
       write(stdout,*) "Stop because there are too many error messages to put into stack."
       write(stdout,*) "Try to increase MAX_MSG_STACK_SIZE if you really need so."
       call getTestSummary ()
       stop 1
    end if

    if (message_index > current_max) then
      msg_swap_holder(1:current_max) = message_array(1:current_max)
      deallocate(message_array)
      current_max = current_max + MSG_ARRAY_INCREMENT
      allocate(message_array(current_max))
      message_array(1:current_max - MSG_ARRAY_INCREMENT) &
                   = msg_swap_holder(1: current_max - MSG_ARRAY_INCREMENT)
    end if

    message_array (message_index) = msg
    message_index = message_index + 1
  end subroutine increase_message_stack_


  subroutine get_xml_filename_work_(string)
    character(len = *), intent(out) :: string
    string = trim(xml_filename_work)
  end subroutine get_xml_filename_work_

  subroutine set_xml_filename_work_(string)
    character(len = *), intent(in) :: string
    xml_filename_work = trim(string)
  end subroutine set_xml_filename_work_


  function get_last_message()
    character(len=MSG_LENGTH) :: get_last_message
    if (message_index > 1) then
       get_last_message = strip(message_array(message_index-1), MSG_LENGTH)
    else
       get_last_message = ''
    end if
  end function get_last_message

  subroutine get_message_index_(index)
    integer, intent(out) :: index

    index = message_index
  end subroutine get_message_index_


  subroutine get_message_array_(msgs)
    character(len = *), intent(out) :: msgs(:)
    integer :: i
    msgs(:) = ""

    do i = 1, message_index - 1
      msgs(i) = trim(strip(message_array(i)))
    enddo
  end subroutine get_message_array_


  subroutine get_messages_(msgs)
    character(len = *), intent(out) :: msgs(:)
    integer :: i, j

    msgs(:) = ""
    do i = message_index_from, message_index - 1
      j = i - message_index_from + 1
      if (j > ubound(msgs, 1)) exit
      msgs(j) = trim(strip(message_array(i)))
    enddo
  end subroutine get_messages_

  subroutine obsolete_getTotalCount_ (count)
    integer, intent (out) :: count
    call obsolete_ (' getTotalCount subroutine is replaced by function get_total_count')
    call get_total_count(count)
  end subroutine obsolete_getTotalCount_

  subroutine get_total_count(count)
    integer, intent(out) :: count

    count = successful_assert_count + failed_assert_count
  end subroutine get_total_count

  subroutine obsolete_getFailedCount_ (count)
    integer, intent (out) :: count

    call obsolete_ (' getFailedCount subroutine is replaced by function get_failed_count')
    call get_failed_count (count)

  end subroutine obsolete_getFailedCount_

  subroutine get_failed_count (count)
    integer, intent(out) :: count
    count = failed_assert_count
  end subroutine get_failed_count

  subroutine obsolete_ (message)
    character (*), intent (in), optional :: message
    write (stdout,*)
    write (stdout,*) "<<<<<<<<<<<<<<<<<<<<<<<<<< WARNING from FRUIT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    write (stdout,*) message
    write (stdout,*)
    write (stdout,*) " old calls will be replaced in the next release in Jan 2009"
    write (stdout,*) " Naming convention for all the method calls are changed to: first_name from"
    write (stdout,*) " firstName.  Subroutines that will be deleted: assertEquals, assertNotEquals,"
    write (stdout,*) " assertTrue, addSuccessful, addFail, etc."
    write (stdout,*) "<<<<<<<<<<<<<<<<<<<<<<<<<< WARNING from FRUIT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    write (stdout,*)
  end subroutine obsolete_

  subroutine add_success
    successful_assert_count = successful_assert_count + 1
    last_passed = .true.
    case_passed = .true.
    call success_mark_
  end subroutine add_success

  subroutine failed_assert_action (expected, got, message, if_is)
    character(*), intent(in) :: expected, got
    character(*), intent(in), optional :: message
    logical, intent(in), optional :: if_is

    if (present(if_is)) then
      call make_error_msg_ (expected, got, if_is,  message)
    else
      call make_error_msg_ (expected, got, .true., message)
    endif
    call increase_message_stack_
    failed_assert_count = failed_assert_count + 1
    last_passed = .false.
    case_passed = .false.
    call failed_mark_
  end subroutine failed_assert_action

  subroutine set_case_name_(value)
    character(*), intent(in) :: value
    case_name = strip(value, MSG_LENGTH)
  end subroutine set_case_name_

  subroutine get_case_name_(value)
    character(*), intent(out) :: value
    value = strip(case_name)
  end subroutine get_case_name_

  subroutine make_error_msg_ (var1, var2, if_is, message)
    character(*), intent(in) :: var1, var2
    logical,      intent(in)           :: if_is
    character(*), intent(in), optional :: message

    msg = '[' // trim(strip(case_name)) // ']: '
    if (if_is) then
      msg = trim(msg) //     'Expected'
    else
      msg = trim(msg) // 'Expected Not'
    endif
    msg = trim(msg) // " " // '[' // trim(strip(var1)) // '], '
    msg = trim(msg) // " " // 'Got'
    msg = trim(msg) // " " // '[' // trim(strip(var2)) // ']'

    if (present(message)) then
       msg = trim(msg) // '; User message: [' // trim(message) // ']'
    endif
  end subroutine make_error_msg_

  function is_last_passed()
    logical:: is_last_passed
    is_last_passed = last_passed
  end function is_last_passed

  function is_case_passed()
    logical:: is_case_passed
    is_case_passed = case_passed
  end function is_case_passed

  subroutine override_stdout_(write_unit, filename)
    integer, intent(in) ::    write_unit
    character(len = *), intent(in) :: filename

    stdout = write_unit
    open(stdout, file = filename, action = "write", status = "replace")
  end subroutine override_stdout_

  subroutine override_xml_work_(new_unit, filename)
    integer, intent(in) ::    new_unit
    character(len = *), intent(in) :: filename

    xml_work = new_unit
    xml_filename_work = filename
    open(xml_work, file = filename, action = "write", status = "replace")
  end subroutine override_xml_work_

  subroutine stash_test_suite
    stashed_suite%successful_assert_count = successful_assert_count
                  successful_assert_count = 0

    stashed_suite%failed_assert_count     = failed_assert_count
                  failed_assert_count = 0

    allocate(stashed_suite%message_array(current_max))
             stashed_suite%message_array(1:message_index) = &
                         & message_array(1:message_index)
    deallocate(message_array)
      allocate(message_array(MSG_ARRAY_INCREMENT))

    stashed_suite%message_index = message_index
                  message_index = 1
    stashed_suite%message_index_from = message_index_from
                  message_index_from = 1

    stashed_suite%current_max = current_max
                  current_max = 50
    stashed_suite%successful_case_count = successful_case_count
                  successful_case_count = 0
    stashed_suite%failed_case_count     = failed_case_count
                  failed_case_count = 0
    stashed_suite%testCaseIndex         = testCaseIndex
                  testCaseIndex = 1
    stashed_suite%case_name = case_name
                  case_name = DEFAULT_CASE_NAME

    stashed_suite%last_passed = last_passed
                  last_passed = .false.
    stashed_suite%case_passed = case_passed
                  case_passed = .false.
    stashed_suite%case_time_from = case_time_from
                  case_time_from = 0
    stashed_suite%linechar_count = linechar_count
                  linechar_count = 0
  end subroutine stash_test_suite

  subroutine restore_test_suite
    successful_assert_count = stashed_suite%successful_assert_count
    failed_assert_count     = stashed_suite%failed_assert_count

    message_index = stashed_suite%message_index
    message_index_from = stashed_suite%message_index_from
    current_max  = stashed_suite%current_max

    deallocate(message_array)
    allocate(  message_array(current_max))
    message_array(1:message_index) = &
             & stashed_suite%message_array(1:message_index)
    deallocate(stashed_suite%message_array)

    successful_case_count = stashed_suite%successful_case_count
    failed_case_count     = stashed_suite%failed_case_count
    testCaseIndex         = stashed_suite%testCaseIndex

    case_name          = stashed_suite%case_name
    last_passed        = stashed_suite%last_passed
    case_passed        = stashed_suite%case_passed
    case_time_from     = stashed_suite%case_time_from
    linechar_count     = stashed_suite%linechar_count
  end subroutine restore_test_suite

  subroutine end_override_stdout_
    close(stdout)
    stdout = STDOUT_DEFAULT
  end subroutine end_override_stdout_

  subroutine end_override_xml_work_
    close(xml_work)
    xml_work = XML_WORK_DEFAULT
    xml_filename_work = XML_FN_WORK_DEF
  end subroutine end_override_xml_work_

  subroutine set_prefix_(str)
    character (len = *), intent(in) :: str
    character (len = len_trim(str)) :: str2

    str2 = trim(adjustl(str))
    if (len_trim(str2) <= FRUIT_PREFIX_LEN_MAX) then
      prefix = str2
    else
      prefix = str2(1:FRUIT_PREFIX_LEN_MAX)
    endif
  end subroutine set_prefix_

  subroutine get_prefix_(str)
    character (len = *), intent(out) :: str

    if (len(str) <= len(prefix)) then
      str = trim(prefix)
    else
      str = prefix
    endif
  end subroutine get_prefix_

  subroutine get_assert_and_case_count_(fail_assert, suc_assert, fail_case, suc_case)
    integer, intent(out) :: fail_assert, suc_assert, fail_case, suc_case

    fail_assert =     failed_assert_count
     suc_assert = successful_assert_count
    fail_case   =     failed_case_count
     suc_case   = successful_case_count
  end subroutine get_assert_and_case_count_

  !--------------------------------------------------------------------------------
  ! all assertions
  !--------------------------------------------------------------------------------
  subroutine obsolete_assert_true_logical_(var1, message)
    logical, intent (in) :: var1
    character (*), intent (in), optional :: message

    call obsolete_ ('assertTrue subroutine is replaced by function assert_true')
    call assert_true(var1, message)
  end subroutine obsolete_assert_true_logical_

  subroutine assert_true (var1, message)
    logical, intent (in) :: var1
    character (*), intent (in), optional :: message

    if ( var1 .eqv. .true.) then
       call add_success
    else
       call failed_assert_action(to_s(.true.), to_s(var1), message, if_is = .true.)
    end if
  end subroutine assert_true


  subroutine assert_false_(var1, message)
    logical, intent(in) :: var1
    character(len = *), intent(in), optional :: message

    if (var1 .eqv. .false.) then
      call add_success
    else
      call failed_assert_action(to_s(.true.), to_s(var1), message, if_is = .false.)
    endif
  end subroutine assert_false_

  !====== begin of generated code ======
  !------ 0d_logical ------
  subroutine assert_eq_logical_(var1, var2, message)

    logical, intent (in) :: var1, var2

    character(len = *), intent (in), optional :: message

        if (var1 .neqv. var2) then
          call failed_assert_action(&
          & to_s(var1), &
          & to_s(var2), message, if_is = .true.)
          return
        endif

    call add_success
  end subroutine assert_eq_logical_

  !------ 1d_logical ------
  subroutine assert_eq_1d_logical_(var1, var2, n, message)
    integer, intent (in) :: n
    integer              :: i
    logical, intent (in) :: var1(n), var2(n)

    character(len = *), intent (in), optional :: message
    do i = 1, n
        if (var1(i) .neqv. var2(i)) then
          call failed_assert_action(&
          & to_s(var1(i)), &
          & to_s(var2(i)), '1d array has difference, ' // message, if_is = .true.)
          return
        endif
    enddo
    call add_success
  end subroutine assert_eq_1d_logical_

  !------ 2d_logical ------
  subroutine assert_eq_2d_logical_(var1, var2, n, m, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    logical, intent (in) :: var1(n, m), var2(n, m)

    character(len = *), intent (in), optional :: message
    do j = 1, m
      do i = 1, n
        if (var1(i, j) .neqv. var2(i, j)) then
          call failed_assert_action(&
          & to_s(var1(i, j)), &
          & to_s(var2(i, j)), '2d array has difference, ' // message, if_is = .true.)
          return
        endif
      enddo
    enddo
    call add_success
  end subroutine assert_eq_2d_logical_

  !------ 0d_string ------
  subroutine assert_eq_string_(var1, var2, message)

    character (len = *), intent (in) :: var1, var2

    character(len = *), intent (in), optional :: message

        if (trim(strip(var1)) /= trim(strip(var2))) then
          call failed_assert_action(&
          & to_s(var1), &
          & to_s(var2), message, if_is = .true.)
          return
        endif

    call add_success
  end subroutine assert_eq_string_

  !------ 1d_string ------
  subroutine assert_eq_1d_string_(var1, var2, n, message)
    integer, intent (in) :: n
    integer              :: i
    character (len = *), intent (in) :: var1(n), var2(n)

    character(len = *), intent (in), optional :: message
    do i = 1, n
        if (trim(strip(var1(i))) /= trim(strip(var2(i)))) then
          call failed_assert_action(&
          & to_s(var1(i)), &
          & to_s(var2(i)), '1d array has difference, ' // message, if_is = .true.)
          return
        endif
    enddo
    call add_success
  end subroutine assert_eq_1d_string_

  !------ 2d_string ------
  subroutine assert_eq_2d_string_(var1, var2, n, m, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    character (len = *), intent (in) :: var1(n, m), var2(n, m)

    character(len = *), intent (in), optional :: message
    do j = 1, m
      do i = 1, n
        if (trim(strip(var1(i, j))) /= trim(strip(var2(i, j)))) then
          call failed_assert_action(&
          & to_s(var1(i, j)), &
          & to_s(var2(i, j)), '2d array has difference, ' // message, if_is = .true.)
          return
        endif
      enddo
    enddo
    call add_success
  end subroutine assert_eq_2d_string_

  !------ 0d_int ------
  subroutine assert_eq_int_(var1, var2, message)

    integer, intent (in) :: var1, var2

    character(len = *), intent (in), optional :: message

        if (var1 /= var2) then
          call failed_assert_action(&
          & to_s(var1), &
          & to_s(var2), message, if_is = .true.)
          return
        endif

    call add_success
  end subroutine assert_eq_int_

  !------ 1d_int ------
  subroutine assert_eq_1d_int_(var1, var2, n, message)
    integer, intent (in) :: n
    integer              :: i
    integer, intent (in) :: var1(n), var2(n)

    character(len = *), intent (in), optional :: message
    do i = 1, n
        if (var1(i) /= var2(i)) then
          call failed_assert_action(&
          & to_s(var1(i)), &
          & to_s(var2(i)), '1d array has difference, ' // message, if_is = .true.)
          return
        endif
    enddo
    call add_success
  end subroutine assert_eq_1d_int_

  !------ 2d_int ------
  subroutine assert_eq_2d_int_(var1, var2, n, m, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    integer, intent (in) :: var1(n, m), var2(n, m)

    character(len = *), intent (in), optional :: message
    do j = 1, m
      do i = 1, n
        if (var1(i, j) /= var2(i, j)) then
          call failed_assert_action(&
          & to_s(var1(i, j)), &
          & to_s(var2(i, j)), '2d array has difference, ' // message, if_is = .true.)
          return
        endif
      enddo
    enddo
    call add_success
  end subroutine assert_eq_2d_int_

  !------ 0d_real ------
  subroutine assert_eq_real_(var1, var2, message)

    real, intent (in) :: var1, var2

    character(len = *), intent (in), optional :: message

        if ((var1 < var2) .or. (var1 > var2)) then
          call failed_assert_action(&
          & to_s(var1), &
          & to_s(var2), message, if_is = .true.)
          return
        endif

    call add_success
  end subroutine assert_eq_real_

  !------ 0d_real ------
  subroutine assert_eq_real_in_range_(var1, var2, delta, message)

    real, intent (in) :: var1, var2
    real, intent (in) :: delta
    character(len = *), intent (in), optional :: message

        if (abs(var1 - var2) > delta) then
          call failed_assert_action(&
          & to_s(var1), &
          & to_s(var2), message, if_is = .true.)
          return
        endif

    call add_success
  end subroutine assert_eq_real_in_range_

  !------ 1d_real ------
  subroutine assert_eq_1d_real_(var1, var2, n, message)
    integer, intent (in) :: n
    integer              :: i
    real, intent (in) :: var1(n), var2(n)

    character(len = *), intent (in), optional :: message
    do i = 1, n
        if ((var1(i) < var2(i)) .or. (var1(i) > var2(i))) then
          call failed_assert_action(&
          & to_s(var1(i)), &
          & to_s(var2(i)), '1d array has difference, ' // message, if_is = .true.)
          return
        endif
    enddo
    call add_success
  end subroutine assert_eq_1d_real_

  !------ 1d_real ------
  subroutine assert_eq_1d_real_in_range_(var1, var2, n, delta, message)
    integer, intent (in) :: n
    integer              :: i
    real, intent (in) :: var1(n), var2(n)
    real, intent (in) :: delta
    character(len = *), intent (in), optional :: message
    do i = 1, n
        if (abs(var1(i) - var2(i)) > delta) then
          call failed_assert_action(&
          & to_s(var1(i)), &
          & to_s(var2(i)), '1d array has difference, ' // message, if_is = .true.)
          return
        endif
    enddo
    call add_success
  end subroutine assert_eq_1d_real_in_range_

  !------ 2d_real ------
  subroutine assert_eq_2d_real_(var1, var2, n, m, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    real, intent (in) :: var1(n, m), var2(n, m)

    character(len = *), intent (in), optional :: message
    do j = 1, m
      do i = 1, n
        if ((var1(i, j) < var2(i, j)) .or. (var1(i, j) > var2(i, j))) then
          call failed_assert_action(&
          & to_s(var1(i, j)), &
          & to_s(var2(i, j)), '2d array has difference, ' // message, if_is = .true.)
          return
        endif
      enddo
    enddo
    call add_success
  end subroutine assert_eq_2d_real_

  !------ 2d_real ------
  subroutine assert_eq_2d_real_in_range_(var1, var2, n, m, delta, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    real, intent (in) :: var1(n, m), var2(n, m)
    real, intent (in) :: delta
    character(len = *), intent (in), optional :: message
    do j = 1, m
      do i = 1, n
        if (abs(var1(i, j) - var2(i, j)) > delta) then
          call failed_assert_action(&
          & to_s(var1(i, j)), &
          & to_s(var2(i, j)), '2d array has difference, ' // message, if_is = .true.)
          return
        endif
      enddo
    enddo
    call add_success
  end subroutine assert_eq_2d_real_in_range_

  !------ 0d_double ------
  subroutine assert_eq_double_(var1, var2, message)

    double precision, intent (in) :: var1, var2

    character(len = *), intent (in), optional :: message

        if ((var1 < var2) .or. (var1 > var2)) then
          call failed_assert_action(&
          & to_s(var1), &
          & to_s(var2), message, if_is = .true.)
          return
        endif

    call add_success
  end subroutine assert_eq_double_

  !------ 0d_double ------
  subroutine assert_eq_double_in_range_(var1, var2, delta, message)

    double precision, intent (in) :: var1, var2
    double precision, intent (in) :: delta
    character(len = *), intent (in), optional :: message

        if (abs(var1 - var2) > delta) then
          call failed_assert_action(&
          & to_s(var1), &
          & to_s(var2), message, if_is = .true.)
          return
        endif

    call add_success
  end subroutine assert_eq_double_in_range_

  !------ 1d_double ------
  subroutine assert_eq_1d_double_(var1, var2, n, message)
    integer, intent (in) :: n
    integer              :: i
    double precision, intent (in) :: var1(n), var2(n)

    character(len = *), intent (in), optional :: message
    do i = 1, n
        if ((var1(i) < var2(i)) .or. (var1(i) > var2(i))) then
          call failed_assert_action(&
          & to_s(var1(i)), &
          & to_s(var2(i)), '1d array has difference, ' // message, if_is = .true.)
          return
        endif
    enddo
    call add_success
  end subroutine assert_eq_1d_double_

  !------ 1d_double ------
  subroutine assert_eq_1d_double_in_range_(var1, var2, n, delta, message)
    integer, intent (in) :: n
    integer              :: i
    double precision, intent (in) :: var1(n), var2(n)
    double precision, intent (in) :: delta
    character(len = *), intent (in), optional :: message
    do i = 1, n
        if (abs(var1(i) - var2(i)) > delta) then
          call failed_assert_action(&
          & to_s(var1(i)), &
          & to_s(var2(i)), '1d array has difference, ' // message, if_is = .true.)
          return
        endif
    enddo
    call add_success
  end subroutine assert_eq_1d_double_in_range_

  !------ 2d_double ------
  subroutine assert_eq_2d_double_(var1, var2, n, m, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    double precision, intent (in) :: var1(n, m), var2(n, m)

    character(len = *), intent (in), optional :: message
    do j = 1, m
      do i = 1, n
        if ((var1(i, j) < var2(i, j)) .or. (var1(i, j) > var2(i, j))) then
          call failed_assert_action(&
          & to_s(var1(i, j)), &
          & to_s(var2(i, j)), '2d array has difference, ' // message, if_is = .true.)
          return
        endif
      enddo
    enddo
    call add_success
  end subroutine assert_eq_2d_double_

  !------ 2d_double ------
  subroutine assert_eq_2d_double_in_range_(var1, var2, n, m, delta, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    double precision, intent (in) :: var1(n, m), var2(n, m)
    double precision, intent (in) :: delta
    character(len = *), intent (in), optional :: message
    do j = 1, m
      do i = 1, n
        if (abs(var1(i, j) - var2(i, j)) > delta) then
          call failed_assert_action(&
          & to_s(var1(i, j)), &
          & to_s(var2(i, j)), '2d array has difference, ' // message, if_is = .true.)
          return
        endif
      enddo
    enddo
    call add_success
  end subroutine assert_eq_2d_double_in_range_

  !------ 0d_complex ------
  subroutine assert_eq_complex_(var1, var2, message)

    complex(kind=kind(1.0D0)), intent (in) :: var1, var2

    character(len = *), intent (in), optional :: message

        if ((real (var1) < real (var2)) .or. &
&(real (var1) > real (var2)) .or. &
&(aimag(var1) < aimag(var2)) .or. &
&(aimag(var1) > aimag(var2))) then
          call failed_assert_action(&
          & to_s(var1), &
          & to_s(var2), message, if_is = .true.)
          return
        endif

    call add_success
  end subroutine assert_eq_complex_

  !------ 0d_complex ------
  subroutine assert_eq_complex_in_range_(var1, var2, delta, message)

    complex(kind=kind(1.0D0)), intent (in) :: var1, var2
    double precision, intent (in) :: delta
    character(len = *), intent (in), optional :: message

        if (abs(var1 - var2) > delta) then
          call failed_assert_action(&
          & to_s(var1), &
          & to_s(var2), message, if_is = .true.)
          return
        endif

    call add_success
  end subroutine assert_eq_complex_in_range_

  !------ 1d_complex ------
  subroutine assert_eq_1d_complex_(var1, var2, n, message)
    integer, intent (in) :: n
    integer              :: i
    complex(kind=kind(1.0D0)), intent (in) :: var1(n), var2(n)

    character(len = *), intent (in), optional :: message
    do i = 1, n
        if ((real (var1(i)) < real (var2(i))) .or. &
&(real (var1(i)) > real (var2(i))) .or. &
&(aimag(var1(i)) < aimag(var2(i))) .or. &
&(aimag(var1(i)) > aimag(var2(i)))) then
          call failed_assert_action(&
          & to_s(var1(i)), &
          & to_s(var2(i)), '1d array has difference, ' // message, if_is = .true.)
          return
        endif
    enddo
    call add_success
  end subroutine assert_eq_1d_complex_

  !------ 1d_complex ------
  subroutine assert_eq_1d_complex_in_range_(var1, var2, n, delta, message)
    integer, intent (in) :: n
    integer              :: i
    complex(kind=kind(1.0D0)), intent (in) :: var1(n), var2(n)
    double precision, intent (in) :: delta
    character(len = *), intent (in), optional :: message
    do i = 1, n
        if (abs(var1(i) - var2(i)) > delta) then
          call failed_assert_action(&
          & to_s(var1(i)), &
          & to_s(var2(i)), '1d array has difference, ' // message, if_is = .true.)
          return
        endif
    enddo
    call add_success
  end subroutine assert_eq_1d_complex_in_range_

  !------ 2d_complex ------
  subroutine assert_eq_2d_complex_(var1, var2, n, m, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    complex(kind=kind(1.0D0)), intent (in) :: var1(n, m), var2(n, m)

    character(len = *), intent (in), optional :: message
    do j = 1, m
      do i = 1, n
        if ((real (var1(i, j)) < real (var2(i, j))) .or. &
&(real (var1(i, j)) > real (var2(i, j))) .or. &
&(aimag(var1(i, j)) < aimag(var2(i, j))) .or. &
&(aimag(var1(i, j)) > aimag(var2(i, j)))) then
          call failed_assert_action(&
          & to_s(var1(i, j)), &
          & to_s(var2(i, j)), '2d array has difference, ' // message, if_is = .true.)
          return
        endif
      enddo
    enddo
    call add_success
  end subroutine assert_eq_2d_complex_

  !------ 2d_complex ------
  subroutine assert_eq_2d_complex_in_range_(var1, var2, n, m, delta, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    complex(kind=kind(1.0D0)), intent (in) :: var1(n, m), var2(n, m)
    double precision, intent (in) :: delta
    character(len = *), intent (in), optional :: message
    do j = 1, m
      do i = 1, n
        if (abs(var1(i, j) - var2(i, j)) > delta) then
          call failed_assert_action(&
          & to_s(var1(i, j)), &
          & to_s(var2(i, j)), '2d array has difference, ' // message, if_is = .true.)
          return
        endif
      enddo
    enddo
    call add_success
  end subroutine assert_eq_2d_complex_in_range_

  !------ 0d_logical ------
  subroutine assert_not_equals_logical_(var1, var2, message)

    logical, intent (in) :: var1, var2

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.

        if (var1 .neqv. var2) then
          same_so_far = .false.
        endif

    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1), &
      & to_s(var2), message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_logical_

  !------ 1d_logical ------
  subroutine assert_not_equals_1d_logical_(var1, var2, n, message)
    integer, intent (in) :: n
    integer              :: i
    logical, intent (in) :: var1(n), var2(n)

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do i = 1, n
        if (var1(i) .neqv. var2(i)) then
          same_so_far = .false.
        endif
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1)), &
      & to_s(var2(1)), '1d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_1d_logical_

  !------ 2d_logical ------
  subroutine assert_not_equals_2d_logical_(var1, var2, n, m, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    logical, intent (in) :: var1(n, m), var2(n, m)

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do j = 1, m
      do i = 1, n
        if (var1(i, j) .neqv. var2(i, j)) then
          same_so_far = .false.
        endif
      enddo
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1, 1)), &
      & to_s(var2(1, 1)), '2d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_2d_logical_

  !------ 0d_string ------
  subroutine assert_not_equals_string_(var1, var2, message)

    character (len = *), intent (in) :: var1, var2

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.

        if (trim(strip(var1)) /= trim(strip(var2))) then
          same_so_far = .false.
        endif

    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1), &
      & to_s(var2), message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_string_

  !------ 1d_string ------
  subroutine assert_not_equals_1d_string_(var1, var2, n, message)
    integer, intent (in) :: n
    integer              :: i
    character (len = *), intent (in) :: var1(n), var2(n)

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do i = 1, n
        if (trim(strip(var1(i))) /= trim(strip(var2(i)))) then
          same_so_far = .false.
        endif
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1)), &
      & to_s(var2(1)), '1d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_1d_string_

  !------ 2d_string ------
  subroutine assert_not_equals_2d_string_(var1, var2, n, m, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    character (len = *), intent (in) :: var1(n, m), var2(n, m)

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do j = 1, m
      do i = 1, n
        if (trim(strip(var1(i, j))) /= trim(strip(var2(i, j)))) then
          same_so_far = .false.
        endif
      enddo
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1, 1)), &
      & to_s(var2(1, 1)), '2d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_2d_string_

  !------ 0d_int ------
  subroutine assert_not_equals_int_(var1, var2, message)

    integer, intent (in) :: var1, var2

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.

        if (var1 /= var2) then
          same_so_far = .false.
        endif

    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1), &
      & to_s(var2), message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_int_

  !------ 1d_int ------
  subroutine assert_not_equals_1d_int_(var1, var2, n, message)
    integer, intent (in) :: n
    integer              :: i
    integer, intent (in) :: var1(n), var2(n)

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do i = 1, n
        if (var1(i) /= var2(i)) then
          same_so_far = .false.
        endif
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1)), &
      & to_s(var2(1)), '1d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_1d_int_

  !------ 2d_int ------
  subroutine assert_not_equals_2d_int_(var1, var2, n, m, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    integer, intent (in) :: var1(n, m), var2(n, m)

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do j = 1, m
      do i = 1, n
        if (var1(i, j) /= var2(i, j)) then
          same_so_far = .false.
        endif
      enddo
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1, 1)), &
      & to_s(var2(1, 1)), '2d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_2d_int_

  !------ 0d_real ------
  subroutine assert_not_equals_real_(var1, var2, message)

    real, intent (in) :: var1, var2

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.

        if ((var1 < var2) .or. (var1 > var2)) then
          same_so_far = .false.
        endif

    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1), &
      & to_s(var2), message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_real_

  !------ 0d_real ------
  subroutine assert_not_equals_real_in_range_(var1, var2, delta, message)

    real, intent (in) :: var1, var2
    real, intent (in) :: delta
    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.

        if (abs(var1 - var2) > delta) then
          same_so_far = .false.
        endif

    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1), &
      & to_s(var2), message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_real_in_range_

  !------ 1d_real ------
  subroutine assert_not_equals_1d_real_(var1, var2, n, message)
    integer, intent (in) :: n
    integer              :: i
    real, intent (in) :: var1(n), var2(n)

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do i = 1, n
        if ((var1(i) < var2(i)) .or. (var1(i) > var2(i))) then
          same_so_far = .false.
        endif
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1)), &
      & to_s(var2(1)), '1d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_1d_real_

  !------ 1d_real ------
  subroutine assert_not_equals_1d_real_in_range_(var1, var2, n, delta, message)
    integer, intent (in) :: n
    integer              :: i
    real, intent (in) :: var1(n), var2(n)
    real, intent (in) :: delta
    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do i = 1, n
        if (abs(var1(i) - var2(i)) > delta) then
          same_so_far = .false.
        endif
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1)), &
      & to_s(var2(1)), '1d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_1d_real_in_range_

  !------ 2d_real ------
  subroutine assert_not_equals_2d_real_(var1, var2, n, m, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    real, intent (in) :: var1(n, m), var2(n, m)

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do j = 1, m
      do i = 1, n
        if ((var1(i, j) < var2(i, j)) .or. (var1(i, j) > var2(i, j))) then
          same_so_far = .false.
        endif
      enddo
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1, 1)), &
      & to_s(var2(1, 1)), '2d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_2d_real_

  !------ 2d_real ------
  subroutine assert_not_equals_2d_real_in_range_(var1, var2, n, m, delta, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    real, intent (in) :: var1(n, m), var2(n, m)
    real, intent (in) :: delta
    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do j = 1, m
      do i = 1, n
        if (abs(var1(i, j) - var2(i, j)) > delta) then
          same_so_far = .false.
        endif
      enddo
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1, 1)), &
      & to_s(var2(1, 1)), '2d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_2d_real_in_range_

  !------ 0d_double ------
  subroutine assert_not_equals_double_(var1, var2, message)

    double precision, intent (in) :: var1, var2

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.

        if ((var1 < var2) .or. (var1 > var2)) then
          same_so_far = .false.
        endif

    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1), &
      & to_s(var2), message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_double_

  !------ 0d_double ------
  subroutine assert_not_equals_double_in_range_(var1, var2, delta, message)

    double precision, intent (in) :: var1, var2
    double precision, intent (in) :: delta
    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.

        if (abs(var1 - var2) > delta) then
          same_so_far = .false.
        endif

    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1), &
      & to_s(var2), message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_double_in_range_

  !------ 1d_double ------
  subroutine assert_not_equals_1d_double_(var1, var2, n, message)
    integer, intent (in) :: n
    integer              :: i
    double precision, intent (in) :: var1(n), var2(n)

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do i = 1, n
        if ((var1(i) < var2(i)) .or. (var1(i) > var2(i))) then
          same_so_far = .false.
        endif
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1)), &
      & to_s(var2(1)), '1d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_1d_double_

  !------ 1d_double ------
  subroutine assert_not_equals_1d_double_in_range_(var1, var2, n, delta, message)
    integer, intent (in) :: n
    integer              :: i
    double precision, intent (in) :: var1(n), var2(n)
    double precision, intent (in) :: delta
    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do i = 1, n
        if (abs(var1(i) - var2(i)) > delta) then
          same_so_far = .false.
        endif
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1)), &
      & to_s(var2(1)), '1d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_1d_double_in_range_

  !------ 2d_double ------
  subroutine assert_not_equals_2d_double_(var1, var2, n, m, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    double precision, intent (in) :: var1(n, m), var2(n, m)

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do j = 1, m
      do i = 1, n
        if ((var1(i, j) < var2(i, j)) .or. (var1(i, j) > var2(i, j))) then
          same_so_far = .false.
        endif
      enddo
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1, 1)), &
      & to_s(var2(1, 1)), '2d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_2d_double_

  !------ 2d_double ------
  subroutine assert_not_equals_2d_double_in_range_(var1, var2, n, m, delta, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    double precision, intent (in) :: var1(n, m), var2(n, m)
    double precision, intent (in) :: delta
    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do j = 1, m
      do i = 1, n
        if (abs(var1(i, j) - var2(i, j)) > delta) then
          same_so_far = .false.
        endif
      enddo
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1, 1)), &
      & to_s(var2(1, 1)), '2d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_2d_double_in_range_

  !------ 0d_complex ------
  subroutine assert_not_equals_complex_(var1, var2, message)

    complex(kind=kind(1.0D0)), intent (in) :: var1, var2

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.

        if ((real (var1) < real (var2)) .or. &
&(real (var1) > real (var2)) .or. &
&(aimag(var1) < aimag(var2)) .or. &
&(aimag(var1) > aimag(var2))) then
          same_so_far = .false.
        endif

    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1), &
      & to_s(var2), message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_complex_

  !------ 0d_complex ------
  subroutine assert_not_equals_complex_in_range_(var1, var2, delta, message)

    complex(kind=kind(1.0D0)), intent (in) :: var1, var2
    double precision, intent (in) :: delta
    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.

        if (abs(var1 - var2) > delta) then
          same_so_far = .false.
        endif

    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1), &
      & to_s(var2), message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_complex_in_range_

  !------ 1d_complex ------
  subroutine assert_not_equals_1d_complex_(var1, var2, n, message)
    integer, intent (in) :: n
    integer              :: i
    complex(kind=kind(1.0D0)), intent (in) :: var1(n), var2(n)

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do i = 1, n
        if ((real (var1(i)) < real (var2(i))) .or. &
&(real (var1(i)) > real (var2(i))) .or. &
&(aimag(var1(i)) < aimag(var2(i))) .or. &
&(aimag(var1(i)) > aimag(var2(i)))) then
          same_so_far = .false.
        endif
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1)), &
      & to_s(var2(1)), '1d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_1d_complex_

  !------ 1d_complex ------
  subroutine assert_not_equals_1d_complex_in_range_(var1, var2, n, delta, message)
    integer, intent (in) :: n
    integer              :: i
    complex(kind=kind(1.0D0)), intent (in) :: var1(n), var2(n)
    double precision, intent (in) :: delta
    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do i = 1, n
        if (abs(var1(i) - var2(i)) > delta) then
          same_so_far = .false.
        endif
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1)), &
      & to_s(var2(1)), '1d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_1d_complex_in_range_

  !------ 2d_complex ------
  subroutine assert_not_equals_2d_complex_(var1, var2, n, m, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    complex(kind=kind(1.0D0)), intent (in) :: var1(n, m), var2(n, m)

    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do j = 1, m
      do i = 1, n
        if ((real (var1(i, j)) < real (var2(i, j))) .or. &
&(real (var1(i, j)) > real (var2(i, j))) .or. &
&(aimag(var1(i, j)) < aimag(var2(i, j))) .or. &
&(aimag(var1(i, j)) > aimag(var2(i, j)))) then
          same_so_far = .false.
        endif
      enddo
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1, 1)), &
      & to_s(var2(1, 1)), '2d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_2d_complex_

  !------ 2d_complex ------
  subroutine assert_not_equals_2d_complex_in_range_(var1, var2, n, m, delta, message)
    integer, intent (in) :: n, m
    integer              :: i, j
    complex(kind=kind(1.0D0)), intent (in) :: var1(n, m), var2(n, m)
    double precision, intent (in) :: delta
    character(len = *), intent (in), optional :: message
    logical :: same_so_far

    same_so_far = .true.
    do j = 1, m
      do i = 1, n
        if (abs(var1(i, j) - var2(i, j)) > delta) then
          same_so_far = .false.
        endif
      enddo
    enddo
    if (same_so_far) then
      call failed_assert_action(&
      & to_s(var1(1, 1)), &
      & to_s(var2(1, 1)), '2d array has no difference, ' // message, if_is = .false.)
      return
    endif
    call add_success
  end subroutine assert_not_equals_2d_complex_in_range_

  !====== end of generated code ======

end module fruit
