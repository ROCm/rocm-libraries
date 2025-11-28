# Define a macro to check for the struct
macro(check_f8 RESULT_VAR)
  # Create a temporary source file
  file(WRITE ${CMAKE_BINARY_DIR}/CheckF8.cxx
    "
    #include <hip/hip_fp8.h>
    struct __hip_fp8_e5m2 e5m2;
    struct __hip_fp8_e4m3 e4m3;

    int main() { return 0; }
    "
  )

  # Try to compile the test program
  try_compile(HAS_F8
    ${CMAKE_BINARY_DIR}
    SOURCES ${CMAKE_BINARY_DIR}/CheckF8.cxx
    COMPILE_DEFINITIONS -xhip
  )

  # Set the result variable
  if(HAS_F8)
    set(${RESULT_VAR} TRUE)
  else()
    set(${RESULT_VAR} FALSE)
  endif()

  # Clean up the temporary file (optional but recommended)
  file(REMOVE ${CMAKE_BINARY_DIR}/CheckF8.cxx)
endmacro()
