#!/bin/bash

set -ex
set +e
ERR1=0
/TENSILE_CLIENT_EXE --config-file /SRC/ClientParameters.ini
/TENSILE_CLIENT_EXE --config-file /SRC/ClientParameters_Granularity.ini
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
