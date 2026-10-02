#!/bin/bash

set -ex
set +e
ERR1=0
/victorwu/rocm-libraries/projects/hipblaslt/tensilelite/build_tmp/tensilelite/client/tensilelite-client --config-file /victorwu/rocm-libraries/projects/hipblaslt/tensilelite/f8_tn_maf_b/1_BenchmarkProblems/Cijk_Alik_Bljk_F8BS_BH_UserArgs_00/00_Final/caches/27f18e29975b/source/ClientParameters.ini
ERR2=$?


ERR=0
if [[ $ERR1 -ne 0 ]]
then
    echo one
    ERR=$ERR1
fi
if [[ $ERR2 -ne 0 ]]
then
    echo two
    ERR=$ERR2
fi
exit $ERR
