#!/bin/bash

set -ex
set +e
ERR1=0
#/workspace/rocm-libraries/projects/hipblaslt/tensilelite/build_tmp/tensilelite/client/tensilelite-client --config-file /workspace/rocm-libraries/projects/hipblaslt/tensilelite/early_terminate/1_BenchmarkProblems/Cijk_Alik_Bljk_BBS_BH_UserArgs_00/00_Final/caches/738adac7f635/source/ClientParameters.ini
./early_terminate/tensilelite-client --config-file ./early_terminate/1_BenchmarkProblems/Cijk_Alik_Bljk_BBS_BH_UserArgs_00/00_Final/caches/738adac7f635/source/ClientParameters.ini
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
