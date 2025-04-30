# #############################################################################
# Copyright (C) 2020 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

# ########################################################################
# A helper function to generate packaging scripts to register libraries with system
# ########################################################################
function( write_rocm_package_script_files scripts_write_dir library_name library_link_name )

set( ld_conf_file "/etc/ld.so.conf.d/${library_name}-dev.conf" )

file( WRITE ${scripts_write_dir}/postinst
"#!/bin/bash

set -e

do_ldconfig() {
  echo ${CPACK_PACKAGING_INSTALL_PREFIX}/${LIB_INSTALL_DIR} > ${ld_conf_file} && ldconfig
}

case \"\$1\" in
   configure)
        do_ldconfig
   ;;
   abort-upgrade|abort-remove|abort-deconfigure)
        echo \"\$1\"
   ;;
   *)
        exit 0
   ;;
esac
" )

file( WRITE ${scripts_write_dir}/prerm
"#!/bin/bash

set -e

rm_ldconfig() {
    rm -f ${ld_conf_file} && ldconfig
}


case \"\$1\" in
   remove|purge)
       rm_ldconfig
   ;;
   *)
        exit 0
   ;;
esac
" )

endfunction( )
