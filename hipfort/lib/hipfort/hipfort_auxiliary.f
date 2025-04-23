module hipfort_auxiliary
  !>
  !>   @brief Returns device properties.
  !>
  !>   @param [out] prop written with device properties
  !>   @param [in]  deviceId which device to query for information
  !>
  !>   @return hipSuccess, hipErrorInvalidDevice
  !>   @bug HCC always returns 0 for maxThreadsPerMultiProcessor
  !>   @bug HCC always returns 0 for regsPerBlock
  !>   @bug HCC always returns 0 for l2CacheSize
  !>
  !>   Populates hipGetDeviceProperties with information for the specified device.
  !>
  interface hipGetDeviceProperties
#ifdef USE_CUDA_NAMES
    function hipGetDeviceProperties_(prop,deviceId) bind(c, name="cudaGetDeviceProperties")
#else
    function hipGetDeviceProperties_(prop,deviceId) bind(c, name="hipGetDeviceProperties")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGetDeviceProperties_
#else
      integer(kind(hipSuccess)) :: hipGetDeviceProperties_
#endif
      type(hipDeviceProp_t),intent(out) :: prop
      integer(c_int),value :: deviceId
    end function
  end interface

end module hipfort_auxiliary
