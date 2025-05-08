
#Tuning with tensile

> [!NOTE]
> The published documentation is available at [Tensile](https://rocm.docs.amd.com/projects/Tensile/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `Tensile/docs/src` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

## Using Tensile and rocBLAS

#### Tensile tuning step

See the [Tensile tuning](https://github.com/ROCmSoftwarePlatform/Tensile/wiki) wiki for more details of Tensile and the specifics of the tuning process. The staging area we use for the our process is within build directories in the root level of the Tensile clone. This step will produce LibraryLogic files within each for the build subdirectories. `../build-hgemm_asm_full/3_LibraryLogic`.

#### Building rocBLAS with new configurations

The updated configurations get placed in the rocBLAS path ` rocBLAS/library/src/blas3/Tensile/Logic/asm_full`. They can either go in raw if a complete tuning was targeted or the configuration can be merged using the Tensile merging tool, `Tensile/Utilities/merge_rocblas_yaml_files.py`.

#### Build rocBLAS

run `./install -c` in the rocBLAS source directory. See [rocBLAS build](https://github.com/ROCmSoftwarePlatform/rocBLAS/wiki/1.Build) for more information.

#### Performance measurements

The performance can be measured using rocblas-bench.

Run benchmarks with the following:

`cd ${BUILD_DIR}/release/clients/staging`
`./rocblas-bench -h`

The following are examples for running particular gemm and gemv benchmark:

`./rocblas-bench -f gemm -r s -m 1024 -n 1024 -k 1024 --transposeB T -v 1`
`./rocblas-bench -f gemv -m 9216 -n 9216 --lda 9216 --transposeA T`



# Automation

## Installing

The automation utilities are intended to be a set of tools independent of any formal Tensile components and as such, there is no need to have a Tensile clone to work with them. They can be copied to any working area in which the tuning is being performed. Any provisioning operation of Tensile or rocBLAS which the tuning work-flow depends upon are provided by the utilities.

### installing the required components

```bash
$ sudo apt install python-pip
$ sudo apt install python3-pip
$ sudo pip install pandas
$ sudo pip3 install pandas
$ sudo pip3 install matplotlib
$ sudo pip install openpyxl
$ sudo apt-get install python3-tk
```

The automation scripts is a set of utilities which enables the work-flow of the process for performance optimization using Tensile and rocBLAS.

## Provisioning the Tuning.

### Extracting rocBLAS call information
This step is driven by the scripts/provision_tuning.sh script. This script will take a log file traces the rocblas-bench calls extracts the sizes and generates the log file which can be used in tuning. It will also provision Tensile and stage it to run the actual tuning.


The log file can be generated using ROCBLAS_LAYER=2 in rocblas bench. e.g.

`$ ROCBLAS_LAYER=2 ./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1600 -n 512 -k 1024 --alpha -1.0 --lda 1600 --ldb 1024 --beta 1.0 --ldc 1600 -i 10`

This will produce the calls to rocblas. Example output.

```
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1600 -n 512 -k 1024 --alpha -1 --lda 1600 --ldb 1024 --beta 1 --ldc 1600 -i 10
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1600 -n 512 -k 1024 --alpha -1 --lda 1600 --ldb 1024 --beta 1 --ldc 1600 -i 10
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1600 -n 512 -k 1024 --alpha -1 --lda 1600 --ldb 1024 --beta 1 --ldc 1600 -i 10
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1600 -n 512 -k 1024 --alpha -1 --lda 1600 --ldb 1024 --beta 1 --ldc 1600 -i 10
transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,us
N,N,1600,512,1024,-1,1600,1024,1,1600,6641.81,252.6
```

### Tuning scripts
#### `provision_tuning.sh`

```bash
usage:

./provision_tuning.sh [-w|--working-path <path>] [-z | --size-log <logfile path>] [-b|--branch <branch>] [-c <github commit id>] [-t|--tag <github tag>] [-o|--output <configuration filename>] [-y | --type <data type>] [-l | --library <library/schedule>] [-n] [[-h|--help]

args:
-w|--working-path       the working directory
-z | --size-log         the path to the log file used to generate the sizes
-b|--branch             the github branch id
-c                      github commit id
-t|--tag                github tag
-o|--output             output file name
-y | --type             the datatype to tune for (hgemm|sgemm|dgemm)
-l | --library          the library to use (arcturus|vega10|vega20)
-n                      if this is enabled the utility will generate the config files without provisioning tensile
-h|--help               this help file

```

Example use case

```bash
$ ./tuning/scripts/provision_tuning.sh WORKING_PATH LOG_PATH OUTPUT_SUFFIX.yaml LIBRARY [options]
```

When this is run the tuning will be provisioned in the directory ./tensile_tuing. The following directories will be generated.

```
configs:        this is where the tuning configurations that were created will be placed
make:           this constructs a set of scripts which will execute the tuning process get placed in the Tensile path
scripts:        generated scripts with rocblas calls which will be used for testing
sizes:          the extracted sizes csv file
tensile:        the directory which tensile gets cloned to
```


After this is executed everything will be provisioned. The tuning can be launched with the following steps:

```bash
$ cd tensile_tuning/tensile/Tensile
$ ./doit-all.sh
```


### provision and run verification
#### provision_verification.sh

```bash
usage: ./provision_verification.sh [-w|--working-path <path>] [-r <Tensile reference>] [-b|--branch <branch>] [-c | --commit <github commit id>] [-t|--tag <github tag>]  [-h|--help]

args:
-h | --help             help
-w | --working-path     the working path name
-t | --tag              a github tag
-b | --branch           github branch name
-c | --commit           github commit id
-r                      the Tensile path where the tuning which contains the tuning results


When this utility is ran, it will create the following directories in the working path:

```
library\exact               contains the new logic files
library\merge               contains the new logic merged into the reference logic file, this gets copied to rocBLAS-verify for testing
rocblas\rocBLAS-reference   contains the reference build of rocblas
rocblas\rocBLAS-verify      contains a build of rocblas with the new logic merged in
```

This will provision and build the reference and the verification version of rocBLAS. After it is provisioned, we can copy over the test scripts and execute the scripts in both clones.

Example:

```bash
$ ./tuning/scripts/provision_verification.sh -w validate -r tensile_tuning/tensile/Tensile
$ cp tensile_tuning/scripts/* validate/rocblas/rocBLAS-reference/build/release/clients/staging
$ cp tensile_tuning/scripts/* validate/rocblas/rocBLAS-validate/build/release/clients/staging
$ pushd validate/rocblas/rocBLAS-reference/build/release/clients/staging
$ ./doit_all.sh
$ popd
$ pushd validate/rocblas/rocBLAS-validate/build/release/clients/staging
$ ./doit_all.sh
$ popd

```


### Analyze benchmark results
#### analyze-results.sh


```bash
usage: ./analyze-results.sh [-b|--benchmark-path <benchmark results path>] [-r| --reference-path <reference results path>] [-o|--output <output path>] [-f] [-s] [-z] [-g|--gpu] [-m|--mfma] [-h|--help]

args:
-b|--benchmark-path     the benchmark path (rocBLAS with new logic)
-r|--reference-path     the reference path (rocBLAS for reference)
-o|--output             the output of the analysis
-f                      frequency which the gpu was set at during validation
-s                      the size of the datatype
-z			            the log file used, in order to collect the call_count (-i)
-g|--gpu                the gpu used when tuning
-m|--mfma               whether mfma instructions were used during validation
-h|--help               the help

```

When this utility is ran, it will generate analysis in the following directories:

```
final           results with the old and new logic compared
new             results for the version of rocBLAS with the new logic
reference       results for the version of rocBLAS without the new logic
```

Example:
```
$ ./tuning/scripts/analyze-results.sh -o analysis -s 2 -f 1301 -r validate/rocblas/rocBLAS-reference/build/release/clients/staging/results -b validate/rocblas/rocBLAS-verify/build/release/clients/staging/results -z sizes.log -g mi50 -m disabled
```
