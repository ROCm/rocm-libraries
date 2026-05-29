# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TEST_MATRICES
  SNAP/amazon0312
  Muite/Chebyshev4
  FEMLAB/sme3Dc
  Williams/webbase-1M
  Bova/rma10
  JGD_BIBD/bibd_22_8
  Williams/mac_econ_fwd500
  Williams/mc2depi
  Hamm/scircuit
  Sandia/ASIC_320k
  GHS_psdef/bmwcra_1
  HB/nos1
  HB/nos2
  HB/nos3
  HB/nos4
  HB/nos5
  HB/nos6
  HB/nos7
  DNVS/shipsec1
)

set(TEST_MD5HASH
  f567e5f5029d052e3004bc69bb3f13f5
  e39879103dafab21f4cf942e0fe42a85
  a95eee14d980a9cfbbaf5df4a3c64713
  2d4c239daad6f12d66a1e6a2af44cbdb
  a899a0c48b9a58d081c52ffd88a84955
  455d5b699ea10232bbab5bc002219ae6
  f1b0e56fbb75d1d6862874e3d7d33060
  8c8633eada6455c1784269b213c85ea6
  3e62f7ea83914f7e20019aefb2a5176f
  fcfaf8a25c8f49b8d29f138f3c65c08f
  8a3cf5448a4fe73dcbdb5a16b326715f
  b203f7605cb1f20f83280061068f7ec7
  b0f812ffcc9469f0bf9be701205522c4
  f185514062a0eeabe86d2909275fe1dc
  04b781415202db404733ca0c159acbef
  c98e35f1cfd1ee8177f37bdae155a6e7
  c39375226aa5c495293003a5f637598f
  9a6481268847e6cf0d70671f2ff1ddcd
  73372e7d6a0848f8b19d64a924fab73e
)

if(NOT HIPSPARSE_TEST_DATA_CACHE_DIR AND DEFINED ENV{HIPSPARSE_TEST_DATA_CACHE_DIR})
  set(HIPSPARSE_TEST_DATA_CACHE_DIR "$ENV{HIPSPARSE_TEST_DATA_CACHE_DIR}")
endif()
if(HIPSPARSE_TEST_DATA_CACHE_DIR)
  get_filename_component(HIPSPARSE_TEST_DATA_CACHE_DIR "${HIPSPARSE_TEST_DATA_CACHE_DIR}"
                         ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
  file(MAKE_DIRECTORY "${HIPSPARSE_TEST_DATA_CACHE_DIR}")
  message("Using HIPSPARSE_TEST_DATA_CACHE_DIR: ${HIPSPARSE_TEST_DATA_CACHE_DIR}")
endif()

file(MAKE_DIRECTORY "${CMAKE_MATRICES_DIR}")

function(hipsparse_download_test_matrix_archive matrix_path matrix_name expected_md5 output_path)
  set(cache_path "")
  if(HIPSPARSE_TEST_DATA_CACHE_DIR)
    set(cache_path "${HIPSPARSE_TEST_DATA_CACHE_DIR}/${matrix_path}.tar.gz")
    if(EXISTS "${cache_path}")
      file(MD5 "${cache_path}" cache_hash)
      if(cache_hash STREQUAL expected_md5)
        message("-- Using cached test matrix ${matrix_path}.tar.gz from ${HIPSPARSE_TEST_DATA_CACHE_DIR}")
        configure_file("${cache_path}" "${output_path}" COPYONLY)
        return()
      endif()
      message(WARNING "-- Ignoring cached test matrix ${cache_path}: checksum mismatch")
    endif()
  endif()

  # First try user specified mirror, if available
  if(DEFINED ENV{HIPSPARSE_TEST_MIRROR} AND NOT $ENV{HIPSPARSE_TEST_MIRROR} STREQUAL "")
    message("-- Downloading test matrix ${matrix_path}.tar.gz from user specified test mirror: $ENV{HIPSPARSE_TEST_MIRROR}")
    file(DOWNLOAD "$ENV{HIPSPARSE_TEST_MIRROR}/${matrix_name}.tar.gz" "${output_path}"
         INACTIVITY_TIMEOUT 3
         STATUS DL)

    list(GET DL 0 stat)
    list(GET DL 1 msg)

    if(NOT stat EQUAL 0)
      message(FATAL_ERROR "-- Timeout has been reached, specified test mirror is not reachable: ${msg}")
    endif()
  else()
    message("-- Downloading test matrix ${matrix_path}.tar.gz")
    file(DOWNLOAD "https://sparse.tamu.edu/MM/${matrix_path}.tar.gz" "${output_path}"
         INACTIVITY_TIMEOUT 3
         STATUS DL)

    list(GET DL 0 stat)
    list(GET DL 1 msg)

    if(NOT stat EQUAL 0)
      message("-- Timeout has been reached, trying mirror ...")
      # Try again using ufl links
      file(DOWNLOAD "https://www.cise.ufl.edu/research/sparse/MM/${matrix_path}.tar.gz" "${output_path}"
           INACTIVITY_TIMEOUT 3
           STATUS DL)

      list(GET DL 0 stat)
      list(GET DL 1 msg)

      if(NOT stat EQUAL 0)
        message(FATAL_ERROR "${msg}")
      endif()
    endif()
  endif()

  file(MD5 "${output_path}" hash)
  if(NOT hash STREQUAL expected_md5)
    message(FATAL_ERROR "${matrix_name}.tar.gz is corrupted")
  endif()

  if(cache_path)
    get_filename_component(cache_dir "${cache_path}" DIRECTORY)
    file(MAKE_DIRECTORY "${cache_dir}")
    configure_file("${output_path}" "${cache_path}" COPYONLY)
  endif()
endfunction()

function(hipsparse_prepare_test_matrices converter_executable converter_name)
  list(LENGTH TEST_MATRICES len)
  math(EXPR len1 "${len} - 1")

  foreach(i RANGE 0 ${len1})
    list(GET TEST_MATRICES ${i} m)
    list(GET TEST_MD5HASH ${i} md5)

    string(REPLACE "/" ";" sep_m ${m})
    list(GET sep_m 1 mat)

    # Download test matrices if not already downloaded
    if(NOT EXISTS "${CMAKE_MATRICES_DIR}/${mat}.bin")
      if(NOT HIPSPARSE_MTX_DIR)
        hipsparse_download_test_matrix_archive("${m}" "${mat}" "${md5}" "${CMAKE_MATRICES_DIR}/${mat}.tar.gz")

        execute_process(COMMAND tar xf "${mat}.tar.gz"
          RESULT_VARIABLE STATUS
          WORKING_DIRECTORY "${CMAKE_MATRICES_DIR}")
        if(STATUS AND NOT STATUS EQUAL 0)
          message(FATAL_ERROR "uncompressing failed, aborting.")
        endif()

        file(RENAME "${CMAKE_MATRICES_DIR}/${mat}/${mat}.mtx" "${CMAKE_MATRICES_DIR}/${mat}.mtx")
      else()
        file(RENAME "${HIPSPARSE_MTX_DIR}/${mat}/${mat}.mtx" "${CMAKE_MATRICES_DIR}/${mat}.mtx")
      endif()
      execute_process(COMMAND "${converter_executable}" "${mat}.mtx" "${mat}.bin"
        RESULT_VARIABLE STATUS
        WORKING_DIRECTORY "${CMAKE_MATRICES_DIR}")
      if(STATUS AND NOT STATUS EQUAL 0)
        message(FATAL_ERROR "${converter_name} failed, aborting.")
      else()
        message(STATUS "${mat} success.")
      endif()
      file(REMOVE_RECURSE "${CMAKE_MATRICES_DIR}/${mat}.tar.gz" "${CMAKE_MATRICES_DIR}/${mat}" "${CMAKE_MATRICES_DIR}/${mat}.mtx")
    endif()
  endforeach()
endfunction()
