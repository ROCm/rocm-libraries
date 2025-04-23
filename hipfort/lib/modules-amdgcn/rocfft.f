module rocfft
  use rocfft_enums
  implicit none
  interface

     function rocfft_setup() bind(c,name="rocfft_setup")
       use rocfft_enums
       implicit none
       integer(kind(rocfft_status_success)) :: rocfft_setup
     end function rocfft_setup

     function rocfft_cleanup() bind(c,name="rocfft_cleanup")
       use rocfft_enums
       implicit none
       integer(kind(rocfft_status_success)) :: rocfft_cleanup
     end function rocfft_cleanup

     function rocfft_plan_create(plan,&
                                 placement,&
                                 transform_type,&
                                 precision,&
                                 dimensions,&
                                 lengths,&
                                 number_of_transforms,&
                                 description) bind(c,name="rocfft_plan_create")
       use rocfft_enums
       use iso_c_binding
       implicit none
       integer(kind(rocfft_status_success)) :: rocfft_plan_create
       type(c_ptr) :: plan
       integer(kind(rocfft_placement_inplace)), value :: placement
       integer(kind(rocfft_transform_type_complex_forward)), value :: transform_type
       integer(kind(rocfft_precision_single)), value :: precision
       integer(c_size_t), value :: dimensions
       type(c_ptr), value, intent(in) :: lengths
       integer(c_size_t), value :: number_of_transforms
       type(c_ptr), value, intent(in) :: description
     end function rocfft_plan_create

     function rocfft_plan_destroy(plan) bind(c,name="rocfft_plan_destroy")
       use rocfft_enums
       use iso_c_binding
       implicit none
       integer(kind(rocfft_status_success)) :: rocfft_plan_destroy
       type(c_ptr), value :: plan
     end function rocfft_plan_destroy

     function rocfft_execute(plan,in_buffer,out_buffer,execution_info) bind(c,name="rocfft_execute")
       use rocfft_enums
       use iso_c_binding
       implicit none
       integer(kind(rocfft_status_success)) :: rocfft_execute
       type(c_ptr), value, intent(in) :: plan
       type(c_ptr) :: in_buffer, out_buffer
       type(c_ptr), value :: execution_info
     end function rocfft_execute

  end interface

  contains
     subroutine rocfftCheck(rocfft_status)
       use rocfft_enums
       implicit none
       integer(kind(rocfft_status_success)) :: rocfft_status
       if(rocfft_status /= rocfft_status_success)then
         write(*,*) "rocFFT ERROR: Error code = ", rocfft_status
         call exit(rocfft_status)
       end if
     end subroutine rocfftCheck

end module rocfft
