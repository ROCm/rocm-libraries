# Command-line argument processing
if(NOT BENCHMARK_BINARY_DIR)
  message(STATUS "BENCHMARK_BINARY_DIR not provided, defaulting to working directory")
  set(BENCHMARK_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()
if(NOT BENCHMARK_OUTPUT_DIR)
  message(STATUS "BENCHMARK_OUTPUT_DIR not provided, defaulting to BENCHMARK_BINARY_DIR")
  set(BENCHMARK_OUTPUT_DIR ${BENCHMARK_BINARY_DIR})
endif()
if(NOT BENCHMARK_QUIET)
  set(OUTPUT_QUIET OUTPUT_QUIET)
else()
  set(OUTPUT_QUIET)
endif()

# Search for command-line tools
find_program(CURL_EXECUTABLE
  NAMES curl
)
if(NOT CURL_EXECUTABLE)
  message(FATAL_ERROR "curl executable not found. Please provide a path to it via CMAKE_PREFIX_PATH")
endif()

if(DEFINED ENV{CI_COMMIT_SHA})
  set(GIT_HASH $ENV{CI_COMMIT_SHA})
  message(STATUS "Environment has CI_COMMIT_SHA: $ENV{CI_COMMIT_SHA}")
else()
  find_package(Git
    REQUIRED
  )
  if(NOT GIT_FOUND)
    message(FATAL_ERROR "git executable not found. Please provide a path to it via CMAKE_PREFIX_PATH")
  endif()

  execute_process(
      COMMAND ${GIT_EXECUTABLE}
        rev-parse HEAD
      RESULT_VARIABLE GIT_EXIT_CODE
      OUTPUT_VARIABLE GIT_HASH
      ERROR_VARIABLE GIT_STDERR
    )
  if(NOT GIT_EXIT_CODE EQUAL 0)
    message(FATAL_ERROR "git rev-parse HEAD returned exit code ${GIT_EXIT_CODE}")
  else()
    message(STATUS "git rev-parse HEAD reported hash: ${GIT_HASH}")
  endif()
endif()

string(STRIP ${GIT_HASH} GIT_HASH)

# Benchmark processing
file(GLOB BENCHMARKS "${BENCHMARK_BINARY_DIR}/benchmark_*")
foreach(BENCHMARK IN LISTS BENCHMARKS)
  get_filename_component(BENCHMARK_NAME "${BENCHMARK}" NAME_WE)

  if(BENCHMARK_REST_ENDPOINT) # else() not needed, as default is console dump.
    set(BENCHMARK_ARGS "--benchmark_format=json --benchmark_out=${BENCHMARK_OUTPUT_DIR}/${BENCHMARK_NAME}.json")
  endif()

  message(STATUS "Running ${BENCHMARK}")
  execute_process(
    COMMAND ${BENCHMARK}
      ${BENCHMARK_ARGS}
    RESULT_VARIABLE BENCHMARK_EXIT_CODE
    OUTPUT_VARIABLE BENCHMARK_STDOUT
    ERROR_VARIABLE BENCHMARK_STDERR
  )
  if(NOT BENCHMARK_EXIT_CODE EQUAL 0)
    message(FATAL_ERROR "${BENCHMARK_NAME} returned exit code ${BENCHMARK_EXIT_CODE}" "Stdout:\n${BENCHMARK_STDOUT}" "Stderr:\n${BENCHMARK_STDERR}")
  endif()
  if(NOT BENCHMARK_QUIET)
    message(STATUS "${BENCHMARK_STDOUT}")
  endif()

  if(NOT BENCHMARK_REST_ENDPOINT)
    continue() # If we're not submitting, no need to enrich and send over the wire.
  endif()

  # Enrich data
  #
  # NOTE: regex matching first line instead of entire BENCHMARK_STDOUT
  #       because searching up until the first newline character in any
  #       way is borked.
  string(FIND "${BENCHMARK_STDOUT}" "\n" FIRST_NEWLINE)
  string(SUBSTRING "${BENCHMARK_STDOUT}" 0 ${FIRST_NEWLINE} FIRST_LINE)
  string(REGEX MATCH
    [[^\[(HIP|CUDA)\] Device name: (.*)$]]
    DEVICE_MATCH
    "${FIRST_LINE}"
  )
  if(CMAKE_MATCH_0)
    set(DEVICE_NAME "${CMAKE_MATCH_2}")
  else()
    message(FATAL_ERROR "Device name not found on console output of ${BENCHMARK_NAME}. Output was:" "${BENCHMARK_STDOUT}")
  endif()

  file(READ
    ${BENCHMARK_OUTPUT_DIR}/${BENCHMARK_NAME}.json
    BENCHMARK_FILEOUT
  )
  string(REGEX REPLACE
    [[("context": {)]]
    "\"context\": {\n    \"device\": \"${DEVICE_NAME}\",\n    \"hash\": \"${GIT_HASH}\"," JSON_PAYLOAD
    "${BENCHMARK_FILEOUT}"
  )
  file(WRITE
    ${BENCHMARK_OUTPUT_DIR}/${BENCHMARK_NAME}.json
    "${JSON_PAYLOAD}"
  )

  # Submit data
  execute_process(
    COMMAND ${CURL_EXECUTABLE}
      --header "Content-Type: application/json"
      --request POST
      --data-binary "@${BENCHMARK_OUTPUT_DIR}/${BENCHMARK_NAME}.json"
      ${BENCHMARK_REST_ENDPOINT}
    RESULT_VARIABLE CURL_EXIT_CODE
    OUTPUT_VARIABLE CURL_STDOUT
    ERROR_VARIABLE CURL_STDERR
  )
  if(NOT CURL_EXIT_CODE EQUAL 0)
    message(FATAL_ERROR "curl returned exit code ${CURL_EXIT_CODE}")
  endif()
endforeach()