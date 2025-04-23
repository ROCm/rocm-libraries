module rocfft_enums
  implicit none

  ! rocfft_status
  enum, bind(c)
     enumerator :: rocfft_status_success
     enumerator :: rocfft_status_failure
     enumerator :: rocfft_status_invalid_arg_value
     enumerator :: rocfft_status_invalid_dimensions
     enumerator :: rocfft_status_invalid_array_type
     enumerator :: rocfft_status_invalid_strides
     enumerator :: rocfft_status_invalid_distance
     enumerator :: rocfft_status_invalid_offset
  end enum

  !rocfft_transform_type
  enum, bind(c)
     enumerator :: rocfft_transform_type_complex_forward
     enumerator :: rocfft_transform_type_complex_inverse
     enumerator :: rocfft_transform_type_real_forward
     enumerator :: rocfft_transform_type_real_inverse
  end enum

  !rocfft_precision
  enum, bind(c)
     enumerator :: rocfft_precision_single
     enumerator :: rocfft_precision_double
  end enum

  !rocfft_result_placement
  enum, bind(c)
     enumerator :: rocfft_placement_inplace
     enumerator :: rocfft_placement_notinplace
  end enum

end module rocfft_enums
