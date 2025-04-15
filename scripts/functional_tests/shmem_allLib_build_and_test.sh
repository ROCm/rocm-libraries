###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

#!/bin/bash
###############################################
# Script : shmem_allLib_build_and_test.sh
# Description : the script is to build and execute test for 
#               given libraries as parameter to the script.
#               the script used in math_ci jenkins pipeline
# version 1.0 (Version 1)
# Assumption : the script is placed under clients/functional_tests
###############################################
echo ""
echo ""
echo "Starting script .."$0
echo ""
#check parameters. parameter of libs that need to be built and tested will be passed
if [ $# -ne 1 ]
then
  echo "Invalid Options for $0 ";
  echo "Usage : $0 <liblist> "
  echo "example : $0 RC_SINGLE,RC_MULTI "
  exit 1
fi

liblist=$1
tliblist=`echo $liblist | sed 's/,/ /g'` # replace , with space, so that can be passed for loop
echo "Parameters passed ==> $tliblist"
echo

for libi in $tliblist
do
  echo
  echo "==> Processing Lib...$libi"
  case $libi in
   RC_SINGLE)
      libnm=rc_single
      libBuildDir=$libnm"_build"
      threadType="single_thread"
      ;;
   RC_MULTI)
      libnm=rc_multi
      libBuildDir=$libnm"_build"
      threadType="multi_thread"
      ;;
   RC_MULTI_WF_COAL)
      libnm=rc_multi_wf_coal
      libBuildDir=$libnm"_build"
      threadType="multi_thread"
      ;;
   DC_SINGLE)
      libnm=dc_single
      libBuildDir=$libnm"_build"
      threadType="single_thread"
      ;;
   DC_MULTI)
      libnm=dc_multi
      libBuildDir=$libnm"_build"
      threadType="multi_thread"
      ;;
   RO_NET)
      libnm=ro_net
      libBuildDir=$libnm"_build"
      threadType="single_thread"
      ;;
   *)
     echo "Unable to find the option for $libi. Plesae check shmem admin"
     echo "** warning ** Skipping the process for $libi "
     ;;
  esac

  if [ "$libnm" != "" ] #process only if libname found
  then 
      echo "+-------------------------------------------------------------------------------------------------+"
      echo `date +%Y%m%d%H%M%S`" ==> Start | $threadType - build_configs/$libnm <=="
      echo "starting with params==> $libnm ; $libBuildDir ; $threadType"
      echo
    
      echo "Library build at ==> "$libBuildDir

      mkdir $libBuildDir
      cd $libBuildDir
      ../scripts/build_configs/$libnm # from build directory generating the build

      # test exeuction based on lib
      if [ "$libnm" == "ro_net" ]
      then
        ROCSHMEM_RO=1
        ROC_NET_CPU_QUEUE=1
        UCX_TLS=rc
        #echo $ROCSHMEM_RO"--"$ROC_NET_CPU_QUEUE "--"$UCX_TLS
        ../scripts/functional_tests/driver.sh tests/functional_tests/rocshmem_example_driver $threadType .
      else
        ../scripts/functional_tests/driver.sh tests/functional_tests/rocshmem_example_driver $threadType .
      fi

      if [ $? -ne 0 ]
      then
        echo "Lib $libnm functional Test exited with ** error $? **"
      else
        echo "Lib $libnm sucessfull "
      fi

      cd ../   # move to base directory

      echo
      echo `date +%Y%m%d%H%M%S`" ==> End | $threadType - build_configs/$libnm <=="
      echo "+-------------------------------------------------------------------------------------------------+"
  fi

  libnm="" #reset the parameter after completion of build
  libBuildDir=""
  threadType=""
done
echo 
echo "Script execution ==> $0 <== is done"
echo 

