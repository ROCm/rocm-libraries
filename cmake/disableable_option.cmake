# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Helper that honours DISABLE_ALL_LIBS and preserves explicit user choices
# Usage:
#   disableable_option(<option_name> <doc> <default>)
#
# Arguments:
#   <option_name> - Name of the option to add.
#   <doc>         - Documentation string for the option.
#   <default>     - Default value for the option.
function(disableable_option _option_name _doc _default)
    if(DEFINED ${_option_name})
        return()
    endif()

    if(DISABLE_ALL_LIBS)
        option(${_option_name} "${_doc}" OFF)
    else()
        option(${_option_name} "${_doc}" ${_default})
    endif()
endfunction()