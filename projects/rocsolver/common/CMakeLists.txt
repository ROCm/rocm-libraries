# ##########################################################################
# Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# ##########################################################################

add_library(rocsolver-common INTERFACE)

target_include_directories(rocsolver-common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(rocsolver-common INTERFACE __HIP_HCC_COMPAT_MODE__=1)

if(WIN32)
  target_compile_definitions(rocsolver-common INTERFACE
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
    NOMINMAX
    __HIP_ROCclr__=1
    __HIP_PLATFORM_AMD__=1
  )
  target_compile_options(rocsolver-common INTERFACE
    -fms-extensions
    -fms-compatibility
    -Wno-ignored-attributes
    -Wno-unused-command-line-argument
  )
endif()

if(WERROR)
  target_compile_options(rocsolver-common INTERFACE
    -Werror=vla
    -Werror=reorder
    -Werror=return-type
  )
endif()

if(BUILD_ADDRESS_SANITIZER)
  target_compile_options(rocsolver-common INTERFACE
    -fsanitize=address
    -shared-libasan
  )
  target_link_options(rocsolver-common INTERFACE
    -fsanitize=address
    -shared-libasan
    -fuse-ld=lld
  )
endif()

if(BUILD_CODE_COVERAGE)
  target_compile_options(rocsolver-common INTERFACE
    -fprofile-arcs
    -ftest-coverage
  )
  target_link_options(rocsolver-common INTERFACE --coverage)
endif()

if(BUILD_COMPRESSED_DBG)
  target_compile_options(rocsolver-common INTERFACE
    "$<$<CONFIG:Debug>:-gz>"
    "$<$<CONFIG:RelWithDebInfo>:-gz>"
  )
endif()
