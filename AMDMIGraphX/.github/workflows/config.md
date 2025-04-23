#=====ROCM INFO=====
ROCM_VERSION : '6.3.1'
#default ROCm version to be used
ROCM_BASE_IMAGE : 'rocm/dev-ubuntu-22.04'
#base image from dockerhub to be used
ROCM_BUILT_IMAGE : 'rocm-migraphx'
#name of the docker image built upon ROCm base
USE_NAVI : '0'
#disable NAVI in image build
OVERWRITE_EXISTING : 'true'
#building new ROCm image overwrites old with same version

#=====REPOS INFO=====
ORGANIZATION_REPO : 'AMD'
BENCHMARK_UTILS_REPO : 'ROCm/migraphx-benchmark-utils'
PERFORMANCE_REPORTS_REPO : 'ROCm/migraphx-reports'
PERFORMANCE_BACKUP_REPO : 'migraphx-benchmark/performance-backup'

#=====PERFORMANCE SCRIPT PARAMETERS=====
RESULTS_TO_COMPARE : '10'
#number of previous performance results to be used in calculations
CALCULATION_METHOD_FLAG : '-r'
#calculation method used in reporting, -m for Max value; -s for Std dev; -r for Threshold file
PERFORMANCE_TEST_TIMEOUT : '30m'
#timeout for each model after which test is aborted

#===== W A R N I N G =====
#VARIABLE NAMES NOT TO BE CHANGED, VALUES ONLY!
#VALUES MUST BE ENGLOSED IN SINGLE QUOTES!
