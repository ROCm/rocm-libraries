# Writes version.cpp for hipconv::version() and config_version(), run at build time by the
# hipconv_version target.
#
# Inputs: SOURCE_DIR (the library root), TEMPLATE, OUTPUT.
#
# A published snapshot carries a VERSION file and is read first: it sits inside the consumer's
# repository, where git describe would name the consumer's tags. A checkout asks git the way
# publish/publish_to_miopen.sh does, so the two agree.

set(HIPCONV_VERSION "unknown")
if(EXISTS "${SOURCE_DIR}/VERSION")
    file(STRINGS "${SOURCE_DIR}/VERSION" HIPCONV_VERSION LIMIT_COUNT 1)
else()
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" describe --tags --match "v[0-9]*"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            OUTPUT_VARIABLE described
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE described_status
            ERROR_QUIET)
        if(described_status EQUAL 0)
            set(HIPCONV_VERSION "${described}")
        else()
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" rev-parse --short=7 HEAD
                WORKING_DIRECTORY "${SOURCE_DIR}"
                OUTPUT_VARIABLE sha
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE sha_status
                ERROR_QUIET)
            if(sha_status EQUAL 0)
                set(HIPCONV_VERSION "v0.0.0-0-g${sha}")
            endif()
        endif()
    endif()
endif()

# v0.0.0 is the pseudo-version of a build no release tag precedes, which promises nothing
# about its configs.
set(HIPCONV_CONFIG_VERSION "unknown")
if(NOT HIPCONV_VERSION MATCHES "^v0\\.0\\.0" AND HIPCONV_VERSION MATCHES "^(v[0-9]+\\.[0-9]+)\\.")
    # The last MATCHES evaluated sets CMAKE_MATCH_1.
    set(HIPCONV_CONFIG_VERSION "${CMAKE_MATCH_1}")
endif()

# configure_file rewrites OUTPUT only when its content changes, so an unchanged version
# rebuilds nothing.
configure_file("${TEMPLATE}" "${OUTPUT}" @ONLY)
