# #############################################################################
# Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# #############################################################################


# Attempt (best effort) to return a list of user specified parameters cmake was invoked with
# NOTE: Even if the user specifies CMAKE_INSTALL_PREFIX on the command line, the parameter is
# not returned because it does not have the matching helpstring

function( append_cmake_cli_arguments initial_cli_args return_cli_args )

  # Retrieves the contents of CMakeCache.txt
  get_cmake_property( cmake_properties CACHE_VARIABLES )

  foreach( property ${cmake_properties} )
    get_property(help_string CACHE ${property} PROPERTY HELPSTRING )

    # Properties specified on the command line have boilerplate text
    if( help_string MATCHES "variable specified on the command line" )
      # message( STATUS "property: ${property}")
      # message( STATUS "value: ${${property}}")

      list( APPEND cli_args "-D${property}=${${property}}")
    endif( )
  endforeach( )

  # message( STATUS "get_command_line_arguments: ${cli_args}")
  set( ${return_cli_args} ${${initial_cli_args}} ${cli_args} PARENT_SCOPE )

endfunction( )