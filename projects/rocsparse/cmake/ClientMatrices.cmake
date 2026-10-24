# ########################################################################
# Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

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
  Cote/mplate
  Bai/qc2534
  Chevron/Chevron2
  Chevron/Chevron3
  Chevron/Chevron4
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
  ad5963d0a39a943fcb0dc2b119d5b22a
  fda33f178963fbb39dfc8c051fd0279e
  c093666487879a4e44409eb7be1c0348
  5e784e1f8c6341287a2842bd188b347a
  01e49e63fa0ac2204baef0f5f33974ad
)

if(NOT CMAKE_MATRICES_DIR)
  message(FATAL_ERROR "Unspecified CMAKE_MATRICES_DIR")
endif()

if(NOT CONVERT_SOURCE)
  set(CONVERT_SOURCE ${CMAKE_SOURCE_DIR}/deps/convert.cpp)
endif()

# convert relative path to absolute
get_filename_component(PROJECT_BINARY_DIR "${PROJECT_BINARY_DIR}"
                       ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
get_filename_component(CMAKE_MATRICES_DIR "${CMAKE_MATRICES_DIR}"
                       ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")

# SuiteSparse archives tracked by DVC, present after dvc pull
get_filename_component(ROCSPARSE_DVC_MATRICES_DIR "${CMAKE_CURRENT_LIST_DIR}/../clients/matrices" ABSOLUTE)

set(unconverted_matrices)
set(missing_matrices)
foreach(m IN LISTS TEST_MATRICES)
  get_filename_component(mat "${m}" NAME)
  if(NOT EXISTS "${CMAKE_MATRICES_DIR}/${mat}.csr")
    list(APPEND unconverted_matrices "${mat}")
    if(NOT (ROCSPARSE_MTX_DIR AND EXISTS "${ROCSPARSE_MTX_DIR}/${mat}/${mat}.mtx")
       AND NOT EXISTS "${ROCSPARSE_DVC_MATRICES_DIR}/${mat}.tar.gz")
      list(APPEND missing_matrices "${mat}")
    endif()
  endif()
endforeach()

if(NOT unconverted_matrices)
  return()
endif()

if(missing_matrices AND NOT ALLOW_FETCH_DEPS)
  list(JOIN missing_matrices ", " missing_matrices)
  message(FATAL_ERROR
    "Test matrices not found: ${missing_matrices}.  Pull them with "
    "'dvc pull -R projects/rocsparse/clients/matrices' from the rocm-libraries root (in a "
    "sparse checkout, run 'git sparse-checkout add .dvc' first), point "
    "CMAKE_MATRICES_DIR at a directory with the converted .csr files, or configure with "
    "-DALLOW_FETCH_DEPS=ON to download them from SuiteSparse.")
endif()

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR})
file(MAKE_DIRECTORY ${CMAKE_MATRICES_DIR})

# Set relative RUNPATH to the executable
## determine separator for rpath argument
if("${CMAKE_CXX_COMPILER}" MATCHES ".*amdclang")
  set(rpath_sep "=")
else()
  # default to old behaviour
  set(rpath_sep ",")
endif()

if(BUILD_ADDRESS_SANITIZER)
  # Static ASan runtime: the shared one is not on the library path when mtx2csr.exe runs
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${CONVERT_SOURCE} -O3 -fsanitize=address -static-libasan -Wl,--build-id=sha1 -o ${PROJECT_BINARY_DIR}/mtx2csr.exe
    COMMAND_ERROR_IS_FATAL ANY)
else()
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${CONVERT_SOURCE} -O3 -Wl,--build-id=sha1 -o ${PROJECT_BINARY_DIR}/mtx2csr.exe
    COMMAND_ERROR_IS_FATAL ANY)
endif()

list(LENGTH TEST_MATRICES len)
math(EXPR len1 "${len} - 1")

foreach(i RANGE 0 ${len1})
  list(GET TEST_MATRICES ${i} m)
  list(GET TEST_MD5HASH ${i} md5)

  string(REPLACE "/" ";" sep_m ${m})
  list(GET sep_m 0 dir)
  list(GET sep_m 1 mat)

  # Extract or download test matrices if not already converted
  if(NOT EXISTS "${CMAKE_MATRICES_DIR}/${mat}.csr")
    # Only a downloaded archive is removed after conversion, never the DVC-tracked one
    set(downloaded_archive)
    set(extracted_dir)
    set(mtx_file "${ROCSPARSE_MTX_DIR}/${mat}/${mat}.mtx")
    if(ROCSPARSE_MTX_DIR AND EXISTS "${mtx_file}")
      # Copy, so that the user's file survives the cleanup after conversion
      file(COPY "${mtx_file}" DESTINATION "${CMAKE_MATRICES_DIR}")
    else()
      set(archive "${ROCSPARSE_DVC_MATRICES_DIR}/${mat}.tar.gz")
      if(EXISTS "${archive}")
        message("-- Extracting test matrix ${m}.tar.gz from ${ROCSPARSE_DVC_MATRICES_DIR}")
      else()
        set(archive "${CMAKE_MATRICES_DIR}/${mat}.tar.gz")
        set(downloaded_archive "${archive}")
        # First try user specified mirror, if available
        if(DEFINED ENV{ROCSPARSE_TEST_MIRROR} AND NOT $ENV{ROCSPARSE_TEST_MIRROR} STREQUAL "")
          message("-- Downloading and extracting test matrix ${m}.tar.gz from user specified test mirror: $ENV{ROCSPARSE_TEST_MIRROR}")
          file(DOWNLOAD $ENV{ROCSPARSE_TEST_MIRROR}/${mat}.tar.gz ${archive}
               INACTIVITY_TIMEOUT 10
               STATUS DL)

          list(GET DL 0 stat)
          list(GET DL 1 msg)

          if(NOT stat EQUAL 0)
            message(FATAL_ERROR "-- Timeout has been reached, specified test mirror is not reachable: ${msg}")
          endif()
        else()
          message("-- Downloading and extracting test matrix ${m}.tar.gz")
          file(DOWNLOAD https://sparse.tamu.edu/MM/${m}.tar.gz ${archive}
               INACTIVITY_TIMEOUT 10
               STATUS DL)

          list(GET DL 0 stat)
          list(GET DL 1 msg)

          if(NOT stat EQUAL 0)
            message("-- Timeout has been reached, trying mirror ...")
            # Try again using ufl links
            file(DOWNLOAD https://www.cise.ufl.edu/research/sparse/MM/${m}.tar.gz ${archive}
                 INACTIVITY_TIMEOUT 10
                 STATUS DL)

            list(GET DL 0 stat)
            list(GET DL 1 msg)

            if(NOT stat EQUAL 0)
              message(FATAL_ERROR "${msg}")
            endif()
          endif()
        endif()
      endif()

      # Check MD5 hash before continuing
      file(MD5 ${archive} hash)

      # Compare hash
      if(NOT hash STREQUAL md5)
        message(FATAL_ERROR "${archive} is corrupted")
      endif()

      execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf ${archive}
        WORKING_DIRECTORY ${CMAKE_MATRICES_DIR}
        COMMAND_ERROR_IS_FATAL ANY)
      set(extracted_dir "${CMAKE_MATRICES_DIR}/${mat}")

      file(RENAME ${CMAKE_MATRICES_DIR}/${mat}/${mat}.mtx ${CMAKE_MATRICES_DIR}/${mat}.mtx)
    endif()
    # Write to a temporary name so that an interrupted conversion leaves no .csr that a rerun would skip
    execute_process(COMMAND ${PROJECT_BINARY_DIR}/mtx2csr.exe ${mat}.mtx ${mat}.csr.part
      RESULT_VARIABLE STATUS
      WORKING_DIRECTORY ${CMAKE_MATRICES_DIR})
    if(NOT STATUS EQUAL 0)
      file(REMOVE ${CMAKE_MATRICES_DIR}/${mat}.csr.part)
      message(FATAL_ERROR "mtx2csr.exe failed for ${mat}, aborting.")
    endif()
    file(RENAME ${CMAKE_MATRICES_DIR}/${mat}.csr.part ${CMAKE_MATRICES_DIR}/${mat}.csr)
    message(STATUS "${mat} success.")
    file(REMOVE_RECURSE ${extracted_dir} ${CMAKE_MATRICES_DIR}/${mat}.mtx ${downloaded_archive})

  endif()
endforeach()
