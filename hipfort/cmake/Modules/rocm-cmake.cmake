# Find or download/install rocm-cmake project

find_package(ROCmCMakeBuildTools QUIET PATHS ${ROCM_PATH} /opt/rocm )
if(NOT ROCmCMakeBuildTools_FOUND)
  find_package(ROCM QUIET CONFIG PATHS ${ROCM_PATH} /opt/rocm)
  if(NOT ROCM_FOUND)
    set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
    file(
      DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
      ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
      STATUS rocm_cmake_download_status LOG rocm_cmake_download_log
    )
    list(GET rocm_cmake_download_status 0 rocm_cmake_download_error_code)
    if(rocm_cmake_download_error_code)
      message(FATAL_ERROR "Error: downloading "
        "https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip failed "
        "error_code: ${rocm_cmake_download_error_code} "
        "log: ${rocm_cmake_download_log} "
      )
    endif()
  
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      RESULT_VARIABLE rocm_cmake_unpack_error_code
    )
    if(rocm_cmake_unpack_error_code)
      message(FATAL_ERROR "Error: unpacking  ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip failed")
    endif()
    find_package(ROCM REQUIRED CONFIG PATHS ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag})
  endif()
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
