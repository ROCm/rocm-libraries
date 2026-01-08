set(SUPPORTED_ARCHITECTURES
    gfx908
    gfx90a
    gfx942
    gfx950
    gfx1100
    gfx1101
    gfx1102
    gfx1103
    gfx1150
    gfx1151
    gfx1152
    gfx1153
    gfx11-generic
    gfx1200
    gfx1201
    gfx12-generic
)

function(hiptensor_is_supported_architecture ARCHITECTURE_NAME RESULT_VAR)
    list(FIND SUPPORTED_ARCHITECTURES "${ARCHITECTURE_NAME}" INDEX)
    if (INDEX EQUAL -1)
        set(${RESULT_VAR} FALSE PARENT_SCOPE)
    else()
        set(${RESULT_VAR} TRUE PARENT_SCOPE)
    endif()
endfunction()

function(hiptensor_get_supported_architectures RESULT_VAR)
    string(JOIN ";" SUPPORTED_ARCHITECTURES_STRING ${SUPPORTED_ARCHITECTURES})
    set(${RESULT_VAR} "${SUPPORTED_ARCHITECTURES_STRING}" PARENT_SCOPE)
endfunction()
