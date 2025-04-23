# RPP UNIT TESTS
The ROCm Performance Primitives library provides a test suite, to run unit tests on its functionalities, executed under 3 backend scenarios - (HOST/OCL/HIP):
- HOST backend - (On a CPU with HOST backend)
- OCL backend - (On a GPU with OpenCL backend)
- HIP backend - (On a GPU with HIP backend)
## Test Suite for HOST backend
### Testing all functionalities with default images (3 224x224 images - batch size = 3)
Executing the following should run and dump output images for all variants of functionalities running with HOST backend.
```
cd utilities/rpp-unittests/HOST_NEW
./testAllScript.sh
```
### Testing all functionalities with a differnet image folder
Navigate to utilities/rpp-unittests/HOST_NEW/testAllScript.sh and set the correct SRC1, SRC2 and DST paths under "MANUAL OVERRIDE".
```
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="<your complete folder path that containes images for  srcPtr1>"
SRC_FOLDER_2="<your complete folder path that containes images for  srcPtr2 (can be same as SRC_FOLDER_1 for trials)>"
DST_FOLDER="<your complete folder path where the dstPtr images can be dumped>"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
```
Run the script
```
./testAllScript.sh
```
### Testing individual functionalities and specific sub-cases
First build the HOST backend test suite.
```
cd utilities/rpp-unittests/HOST_NEW
mkdir build
cd build
cmake ..
make
```
Run either of the following executibles for packed-3channel / planar-3channel / planar-1channel images. This will show a list of arguments to be passed to the executible.
```
./BatchPD_host_pkd3
./BatchPD_host_pln3
./BatchPD_host_pln1
```

## Test Suite for OCL backend
### Testing all functionalities with default images (3 224x224 images - batch size = 3)
Executing the following should run and dump output images for all variants of functionalities running with OCL backend.
```
cd utilities/rpp-unittests/OCL_NEW
./testAllScript.sh
```
### Testing all functionalities with a differnet image folder
Navigate to utilities/rpp-unittests/OCL_NEW/testAllScript.sh and set the correct SRC1, SRC2 and DST paths under "MANUAL OVERRIDE".
```
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="<your complete folder path that containes images for  srcPtr1>"
SRC_FOLDER_2="<your complete folder path that containes images for  srcPtr2 (can be same as SRC_FOLDER_1 for trials)>"
DST_FOLDER="<your complete folder path where the dstPtr images can be dumped>"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
```
Run the script
```
./testAllScript.sh
```
### Testing individual functionalities and specific sub-cases
First build the OCL backend test suite.
```
cd utilities/rpp-unittests/OCL_NEW
mkdir build
cd build
cmake ..
make
```
Run either of the following executibles for packed-3channel / planar-3channel / planar-1channel images. This will show a list of arguments to be passed to the executible.
```
./BatchPD_ocl_pkd3
./BatchPD_ocl_pln3
./BatchPD_ocl_pln1
```

## Test Suite for HIP backend
### Testing all functionalities with default images (3 224x224 images - batch size = 3)
Executing the following should run and dump output images for all variants of functionalities running with HIP backend.
```
cd utilities/rpp-unittests/HIP_NEW
./testAllScript.sh
```
### Testing all functionalities with a differnet image folder
Navigate to utilities/rpp-unittests/HIP_NEW/testAllScript.sh and set the correct SRC1, SRC2 and DST paths under "MANUAL OVERRIDE".
```
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="<your complete folder path that containes images for  srcPtr1>"
SRC_FOLDER_2="<your complete folder path that containes images for  srcPtr2 (can be same as SRC_FOLDER_1 for trials)>"
DST_FOLDER="<your complete folder path where the dstPtr images can be dumped>"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
```
Run the script
```
./testAllScript.sh
```
### Testing individual functionalities and specific sub-cases
First build the HIP backend test suite.
```
cd utilities/rpp-unittests/HIP_NEW
mkdir build
cd build
cmake ..
make
```
Run either of the following executibles for packed-3channel / planar-3channel / planar-1channel images. This will show a list of arguments to be passed to the executible.
```
./BatchPD_hip_pkd3
./BatchPD_hip_pln3
./BatchPD_hip_pln1
```