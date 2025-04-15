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

if [ $# -eq 0 ] ; then
    echo "This script must be run with at least 2 arguments."
    echo 'Usage: ${0} argument1 argument2 [argument3]'
    echo "  argument1 : path to the directory with the tests"
    echo "  argument2 : test name to run, e.g hello"
    echo "  argument3 : directory to put the output logs"
    exit 1
fi

echo "Test Name ${2}"

check() {
    if [ $? -ne 0 ]; then
        if [ "$1" != "global_exit" ]; then
            echo "Failed $1" >&2
            exit 1
        fi
    fi
}

case $2 in
    *"short")
        mpirun -np 2 -bind-to core $1/micro_unit_shmem
        check micro_unit_shmem
        mpirun -np 2 -bind-to core $1/broadcast_active_set
        check broadcast_active_set
        mpirun -np 2 -bind-to core $1/to_all
        check to_all
        mpirun -np 2 -bind-to core $1/put_nbi
        check put_nbi
        mpirun -np 2 -bind-to core $1/get_nbi
        check get_nbi
        mpirun -np 2 -bind-to core $1/mt_a2a
        check mt_a2a
        mpirun -np 2 -bind-to core $1/shmem_team_translate
        check shmem_team_translate
        mpirun -np 4 -bind-to core $1/shmem_team_b2b_collectives
        check shmem_team_b2b_collectives
        mpirun -np 2 -env MPIR_CVAR_CH4_NUM_VCIS 32 -bind-to core $1/many-ctx
        check many-ctx
        ;;
    *"all")
        mpirun -np 2 -bind-to core $1/hello
        check hello
        mpirun -np 2 -bind-to core $1/barrier
        check barrier
        mpirun -np 2 -bind-to core $1/global_exit
        check global_exit
        mpirun -np 2 -bind-to core $1/asym_alloc
        check asym_alloc
        mpirun -np 2 -bind-to core $1/shmalloc
        check shmalloc
        mpirun -np 2 -bind-to core $1/bcast
        check bcast
        mpirun -np 2 -bind-to core $1/broadcast_active_set
        check broadcast_active_set
        mpirun -np 2 -bind-to core $1/bcast_flood
        check bcast_flood
        mpirun -np 2 -bind-to core $1/to_all
        check to_all
        mpirun -np 2 -bind-to core $1/reduce_in_place
        check reduce_in_place
        mpirun -np 2 -bind-to core $1/reduce_active_set
        check reduce_active_set
        mpirun -np 2 -bind-to core $1/max_reduction
        check max_reduction
        mpirun -np 2 -bind-to core $1/big_reduction
        check big_reduction
        mpirun -np 2 -bind-to core $1/cxx_test_shmem_p
        check cxx_test_shmem_p
        mpirun -np 2 -bind-to core $1/cxx_test_shmem_g
        check cxx_test_shmem_g
        mpirun -np 2 -bind-to core $1/put1
        check put1
        mpirun -np 2 -bind-to core $1/get1
        check get1
        mpirun -np 2 -bind-to core $1/put_nbi
        check put_nbi
        mpirun -np 2 -bind-to core $1/get_nbi
        check get_nbi
        mpirun -np 2 -bind-to core $1/bigput -l 10
        check bigput
        mpirun -np 2 -bind-to core $1/bigget -l 10
        check bigget
        mpirun -np 2 -bind-to core $1/waituntil
        check waituntil
        mpirun -np 2 -bind-to core $1/cxx_test_shmem_wait_until
        check cxx_test_shmem_wait_until
        mpirun -np 2 -bind-to core $1/shmem_test
        check shmem_test
        mpirun -np 2 -bind-to core $1/cxx_test_shmem_test
        check cxx_test_shmem_test
        mpirun -np 2 -bind-to core $1/atomic_inc
        check atomic_inc
        mpirun -np 2 -bind-to core $1/cxx_test_shmem_atomic_add
        check cxx_test_shmem_atomic_add
        mpirun -np 2 -bind-to core $1/cxx_test_shmem_atomic_fetch
        check cxx_test_shmem_atomic_fetch
        mpirun -np 2 -bind-to core $1/cxx_test_shmem_atomic_cswap
        check cxx_test_shmem_atomic_cswap
        mpirun -np 2 -bind-to core $1/cxx_test_shmem_atomic_inc
        check cxx_test_shmem_atomic_inc
        mpirun -np 2 -bind-to core $1/lfinc
        check lfinc
        mpirun -np 2 -bind-to core $1/query_thread
        check query_thread
        mpirun -np 2 -bind-to core $1/threading
        check threading
        mpirun -np 2 -bind-to core $1/thread_wait
        check thread_wait
        mpirun -np 2 -bind-to core $1/mt_contention
        check mt_contention
        mpirun -np 2 -bind-to core $1/mt_a2a
        check mt_a2a
        mpirun -np 2 -bind-to core $1/micro_unit_shmem
        check micro_unit_shmem
        mpirun -np 2 -bind-to core $1/circular_shift
        check circular_shift
        mpirun -np 2 -bind-to core $1/pi
        check pi
        mpirun -np 2 -bind-to core $1/ping
        check ping
        mpirun -np 2 -bind-to core $1/sping
        check sping
        mpirun -np 4 -bind-to core $1/shmem_team_b2b_collectives
        check shmem_team_b2b_collectives
        mpirun -np 4 -bind-to core $1/shmem_team_reduce
        check shmem_team_reduce
        mpirun -np 2 -bind-to core $1/shmem_team_reuse_teams
        check shmem_team_reuse_teams
        mpirun -np 2 -bind-to core $1/shmem_team_translate
        check shmem_team_translate
        ;;
    *"hello")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"barrier")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"global_exit")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"asym_alloc")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"shmalloc")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"bcast")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"broadcast_active_set")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"bcast_flood")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"to_all")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"reduce_in_place")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"reduce_active_set")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"max_reduction")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"big_reduction")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"cxx_test_shmem_p")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"cxx_test_shmem_g")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"put1")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"get1")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"put_nbi")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"get_nbi")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"bigput")
        mpirun -np 2 -bind-to core $1/$2 -l 10
        ;;
    *"bigget")
        mpirun -np 2 -bind-to core $1/$2 -l 10
        ;;
    *"waituntil")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"cxx_test_shmem_wait_until")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"shmem_test")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"cxx_test_shmem_test")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"atomic_inc")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"cxx_test_shmem_atomic_add")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"cxx_test_shmem_atomic_fetch")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"cxx_test_shmem_atomic_cswap")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"cxx_test_shmem_atomic_inc")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"lfinc")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"query_thread")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"threading")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"thread_wait")
        mpirun -np 2 -bind-to core $1/$2 # this could just be -np 1
        ;;
    *"mt_contention")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"mt_a2a")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"micro_unit_shmem")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"circular_shift")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"pi")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"ping")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"sping")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"shmem_team_b2b_collectives")
        mpirun -np 4 -bind-to core $1/$2
        ;;
    *"shmem_team_reduce")
        mpirun -np 4 -bind-to core $1/$2
        ;;
    *"shmem_team_reuse_teams")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *"shmem_team_translate")
        mpirun -np 2 -bind-to core $1/$2
        ;;
    *)
        echo "UNKNOWN TEST TYPE: $2"
        exit -1
        ;;
esac

exit $?
