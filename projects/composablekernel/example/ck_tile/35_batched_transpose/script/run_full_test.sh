#!/bin/bash 
#
# in order to run this script you'd first need to build the tile_example_batched_transpose executables in ../build/bin/
#
# run the script as "./run_full_test.sh <tag for your test environment> <branch name> <host name> <gpu_arch>
# input arguments: 
# environment tag  : a string describing the specifics of your test environment
# branch name      : name of the branch in git repo (git status | grep -e 'On branch')
# host name        : $hostname
# gpu architecture: e.g., gfx90a, or gfx942, etc.

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

#run verification tests
example/ck_tile/35_batched_transpose/script/smoke_test.sh

#run performance benchmarks

