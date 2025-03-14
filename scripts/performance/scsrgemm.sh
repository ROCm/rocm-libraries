#!/usr/bin/env bash
# Author: Nico Trost

# Helper function
function display_help()
{
    echo "rocSPARSE benchmark helper script"
    echo "    [-h|--help] prints this help message"
    echo "    [-d|--device] select device"
    echo "    [-p|--path] path to rocsparse-bench"
    echo "    [-m|--matrices-dir] directory of matrix files, this option discards the environment variable MATRICES_DIR. "
    echo "    [-n|--sizen] number of dense columns"
}

# Check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
    echo "This script uses getopt to parse arguments; try installing the util-linux package";
    exit 1;
fi

dev=0
path=../../build/release/clients/staging
sizen=500000

# Parse command line parameters
getopt -T
if [[ $? -eq 4 ]]; then
    GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,device:,path:,matrices-dir:,samples-dir:,sizen: --options hd:p:m:s:n: -- "$@")
else
    echo "Need a new version of getopt"
    exit 1
fi

if [[ $? -ne 0 ]]; then
    echo "getopt invocation failed; could not parse the command line";
    exit 1
fi

if [[ ( ${MATRICES_DIR} == "" ) ]];then
    matrices_dir=./matrices
else
    matrices_dir=${MATRICES_DIR}
fi

samples_dir=../samples

eval set -- "${GETOPT_PARSE}"

while true; do
    case "${1}" in
        -m|--matrices-dir)
            matrices_dir=${2}
            shift 2 ;;
        -s|--samples-dir)
            samples_dir=${2}
            shift 2 ;;
        -h|--help)
            display_help
            exit 0
            ;;
        -d|--device)
            dev=${2}
            shift 2 ;;
        -p|--path)
            path=${2}
            shift 2 ;;
        -n|--sizen)
            sizen=${2}
            shift 2 ;;
        --) shift ; break ;;
        *)  echo "Unexpected command line parameter received; aborting";
            exit 1
            ;;
    esac
done

bench=$path/rocsparse-bench

# Check if binary is available
if [ ! -f $bench ]; then
    echo $bench not found, exit...
    exit 1
else
    echo ">>" $(realpath $(ldd $bench | grep rocsparse | awk '{print $3;}'))
fi

# Generate logfile name
logname=scsrgemm_$(date +'%Y%m%d%H%M%S').log
truncate -s 0 $logname

which=`ls $matrices_dir/*.csr`
filenames=`for i in $which;do basename $i;done`

arguments=`python3 ${samples_dir}/readConfig.py ${samples_dir}/qa/csrgemm/float_500000.json`

# Run csrgemm for all matrices available
for filename in $filenames; do
    $bench --matrices-dir $matrices_dir $(eval echo ${arguments}) 2>&1 | tee -a $logname
done
