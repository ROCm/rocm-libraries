# RPP PERFORMANCE TESTS
The ROCm Performance Primitives library provides a test suite, to run performance tests on its functionalities, executed under 3 backend scenarios - (HOST/OCL/HIP):
- HOST backend - (On a CPU with HOST backend)
- OCL backend - (On a GPU with OpenCL backend)
- HIP backend - (On a GPU with HIP backend)
## Test Suite for HOST backend
### Testing all functionalities with default images (3 224x224 images - batch size = 3)
Executing the following should run and report max/min/avg times in seconds over 100 runs for all variants of functionalities running with HOST backend.
```
cd utilities/rpp-performancetests/HOST_NEW
python3 generatePerformanceLogs.py
```
### Testing all functionalities with a differnet image folder
Navigate to utilities/rpp-performancetests/HOST_NEW/rawLogsGenScript.sh and set the correct SRC1, SRC2 and DST paths under "MANUAL OVERRIDE".
```
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="<your complete folder path that containes images for  srcPtr1>"
SRC_FOLDER_2="<your complete folder path that containes images for  srcPtr2 (can be same as SRC_FOLDER_1 for trials)>"
DST_FOLDER="<your complete folder path where the dstPtr images can be dumped>"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
```
Run the python script
```
python3 generatePerformanceLogs.py
```

## Test Suite for OCL backend
### Testing all functionalities with default images (3 224x224 images - batch size = 3)
Executing the following should run and report max/min/avg times in seconds over 100 runs for all variants of functionalities running with OCL backend.
```
cd utilities/rpp-performancetests/OCL_NEW
python3 generatePerformanceLogs.py
```
### Testing all functionalities with a differnet image folder
Navigate to utilities/rpp-performancetests/OCL_NEW/rawLogsGenScript.sh and set the correct SRC1, SRC2 and DST paths under "MANUAL OVERRIDE".
```
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="<your complete folder path that containes images for  srcPtr1>"
SRC_FOLDER_2="<your complete folder path that containes images for  srcPtr2 (can be same as SRC_FOLDER_1 for trials)>"
DST_FOLDER="<your complete folder path where the dstPtr images can be dumped>"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
```
Run the python script
```
python3 generatePerformanceLogs.py
```

## Test Suite for HIP backend
### Testing all functionalities with default images (3 224x224 images - batch size = 3)
Executing the following should run and report max/min/avg times in seconds over 100 runs for all variants of functionalities running with HIP backend.
```
cd utilities/rpp-performancetests/HIP_NEW
python3 generatePerformanceLogs.py
```
### Testing all functionalities with a differnet image folder
Navigate to utilities/rpp-performancetests/HIP_NEW/rawLogsGenScript.sh and set the correct SRC1, SRC2 and DST paths under "MANUAL OVERRIDE".
```
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="<your complete folder path that containes images for  srcPtr1>"
SRC_FOLDER_2="<your complete folder path that containes images for  srcPtr2 (can be same as SRC_FOLDER_1 for trials)>"
DST_FOLDER="<your complete folder path where the dstPtr images can be dumped>"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
```
Run the python script
```
python3 generatePerformanceLogs.py
```
