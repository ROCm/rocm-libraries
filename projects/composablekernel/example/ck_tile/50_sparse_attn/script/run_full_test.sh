#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

#
# Build tile_example_sparse_attn_fwd first, then:
#   ./run_full_test.sh <env_tag> <branch> <host> <gpu_arch>   (e.g. gfx90a / gfx942)

set -euo pipefail

#get the command line arguments:
export env_type=$1
echo 'Environment type: ' $env_type
export branch=$2
echo 'Branch name: ' $branch
export host_name=$3
echo 'Host name: ' $host_name
export GPU_arch=$4
echo 'GPU_arch: ' $GPU_arch

function print_log_header(){
	rm -f $1;
	echo 'On branch ' $3 &> $1;
	echo 'Node name: ' $4 >> $1;
	#get GPU_arch and number of compute units from rocminfo
	echo -n "GPU_arch: " >> $1; rocminfo | grep "Name:" | grep "gfx" >> $1;
	rocminfo | grep "Compute Unit:" >> $1;
	hipcc --version | grep -e 'HIP version'  >> $1;
	echo 'Environment type: ' $2 >> $1;
	/opt/rocm/bin/amdclang++ --version | grep -e 'InstalledDir' >> $1;
}

#run verification tests (full matrix: both perms, long seqlen, all sage qscales)
time PERMS="0 1" SL=4096 QSCALES="perwarp perblock perthread pertensor" \
    example/ck_tile/50_sparse_attn/script/smoke_test_sparse_attn.sh

#run performance benchmarks
export sparse_attn_fwd_log="perf_sparse_attn_fwd_$GPU_arch.log"
print_log_header $sparse_attn_fwd_log $env_type $branch $host_name
time example/ck_tile/50_sparse_attn/script/benchmark_sparse_attn.sh 2>&1 | tee -a $sparse_attn_fwd_log
