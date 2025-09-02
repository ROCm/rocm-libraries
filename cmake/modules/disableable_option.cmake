# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Helper that honours ROCM_LIBRARIES_DISABLE_ALL and preserves explicit user choices
# Usage:
#   disableable_option(<option_name> <doc> <default> <disable>)
#
# Arguments:
#   <option_name> - Name of the option to add.
#   <doc>         - Documentation string for the option.
#   <default>     - Default value for the option.
#   <disable>     - Whether to disable the option. If the option is already set
#                   via the CLI, this will be ignored.
function(disableable_option _option_name _doc _default _disable)
    if(${_disable})
        option(${_option_name} "${_doc}" OFF)
    else()
        option(${_option_name} "${_doc}" ${_default})
    endif()
endfunction()
