# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Resolve an imported runtime library, including configured exports. Never use
# the Windows import library, which cannot be loaded by a runtime consumer.
function(rocke_imported_library_location target out_path)
    set(_derived "")
    # A multi-config generator does not know the configuration at configure
    # time, so CMAKE_BUILD_TYPE may be empty and the rest of the order decides.
    # NOCONFIG is what an export() with no build type emits.
    set(_preferred "")
    if(CMAKE_BUILD_TYPE)
        string(TOUPPER "${CMAKE_BUILD_TYPE}" _preferred)
    endif()
    list(APPEND _preferred RELEASE RELWITHDEBINFO MINSIZEREL DEBUG NOCONFIG)

    # IN_LIST against a NOTFOUND property is a hard error, which is what the
    # guard prevents.
    get_target_property(_cfgs ${target} IMPORTED_CONFIGURATIONS)
    if(_cfgs)
        foreach(_cfg IN LISTS _preferred)
            if(NOT _cfg IN_LIST _cfgs)
                continue()
            endif()
            get_target_property(_loc ${target} IMPORTED_LOCATION_${_cfg})
            if(_loc AND EXISTS "${_loc}")
                set(_derived "${_loc}")
                break()
            endif()
        endforeach()
    endif()
    # Nothing preferred resolved: any real library beats none.
    if(NOT _derived AND _cfgs)
        foreach(_cfg IN LISTS _cfgs)
            get_target_property(_loc ${target} IMPORTED_LOCATION_${_cfg})
            if(_loc AND EXISTS "${_loc}")
                set(_derived "${_loc}")
                break()
            endif()
        endforeach()
    endif()
    # A package that exports no configurations at all.
    if(NOT _derived)
        get_target_property(_loc ${target} IMPORTED_LOCATION)
        if(_loc AND EXISTS "${_loc}")
            set(_derived "${_loc}")
        endif()
    endif()
    set(${out_path} "${_derived}" PARENT_SCOPE)
endfunction()
