if(NOT CLIENT OR NOT READELF)
    message(FATAL_ERROR "CLIENT and READELF are required")
endif()

execute_process(
    COMMAND "${READELF}" -d "${CLIENT}"
    RESULT_VARIABLE readelf_status
    OUTPUT_VARIABLE readelf_output
    ERROR_VARIABLE readelf_error
)
if(NOT readelf_status EQUAL 0)
    message(FATAL_ERROR "readelf failed for ${CLIENT}: ${readelf_error}")
endif()

if(readelf_output MATCHES "NEEDED[^\n]*libtensilelite-host")
    message(FATAL_ERROR
        "${CLIENT} still has a DT_NEEDED entry for libtensilelite-host. "
        "The development client must link that library statically so an installed copy on LD_LIBRARY_PATH cannot be substituted.")
endif()
