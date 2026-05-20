# Q6 smoke for RFC 0001: external-project consumer that uses
# find_package(miopen) to link the installed library.
#
# Driven by tools/wrapper/check_find_package_smoke.sh, which configures and
# builds this project against an installed MIOpen prefix and asserts the
# binary links cleanly. With MIOPEN_ENABLE_HIPDNN_WRAPPER=ON, the install
# tree exports both MIOpen and MIOpen_private; both must be discoverable.

cmake_minimum_required(VERSION 3.20)
project(miopen_find_package_smoke LANGUAGES C CXX)

find_package(miopen CONFIG REQUIRED)

# 1) Public path: link against the wrapper-or-only target. Same as a real
#    consumer (PyTorch, TensorFlow, hipDNN's miopen-provider when built
#    against a legacy install).
add_executable(consumer_public ${CMAKE_CURRENT_SOURCE_DIR}/../consumer_smoke.c)
target_link_libraries(consumer_public PRIVATE MIOpen)

# 2) Bypass path: link against MIOpen_private when present. This is the
#    Phase 4 short-circuit miopen-provider uses to avoid re-entering hipDNN.
#    With the legacy install (no wrapper), the target is absent; skip.
if(TARGET MIOpen_private)
    add_executable(consumer_private ${CMAKE_CURRENT_SOURCE_DIR}/../consumer_smoke.c)
    target_link_libraries(consumer_private PRIVATE MIOpen_private)
    message(STATUS "Q6 find_package_smoke: MIOpen_private target present; building bypass consumer")
else()
    message(STATUS "Q6 find_package_smoke: MIOpen_private target absent (legacy install); skipping bypass consumer")
endif()
