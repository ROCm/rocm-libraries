@echo off
REM Build hipDNN python bindings inside the MSVC + ROCm wheel environment
REM Ensure vswhere is reachable so vcvars can locate the Windows SDK
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "C:\develop\dist\vs-buildtools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo VCVARS FAILED & exit /b 1)

REM vcvars cannot locate the Windows SDK (unregistered BuildTools instance),
REM so add the SDK lib/include paths explicitly.
set "WINSDK=C:\Program Files (x86)\Windows Kits\10"
set "WINSDKVER=10.0.22621.0"
set "LIB=%LIB%;%WINSDK%\Lib\%WINSDKVER%\um\x64;%WINSDK%\Lib\%WINSDKVER%\ucrt\x64"
set "INCLUDE=%INCLUDE%;%WINSDK%\Include\%WINSDKVER%\um;%WINSDK%\Include\%WINSDKVER%\ucrt;%WINSDK%\Include\%WINSDKVER%\shared"

set SRC=C:/Develop/rocm-libraries/projects/hipdnn
set BLD=C:/Develop/rocm-libraries/projects/hipdnn/build
set WHEEL=C:/develop/latest_wheels/Lib/site-packages/_rocm_sdk_devel

if "%1"=="configure" (
  cmake -S %SRC% -B %BLD% -GNinja ^
    -DCMAKE_CXX_COMPILER=%WHEEL%/lib/llvm/bin/clang++.exe ^
    -DCMAKE_MAKE_PROGRAM=C:/Users/tvy/AppData/Local/Microsoft/WinGet/Packages/Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe/ninja.exe ^
    -DCMAKE_PREFIX_PATH=%WHEEL% ^
    -DROCM_CMAKE_PATH=%WHEEL% ^
    -DPython_EXECUTABLE=C:/develop/latest_wheels/Scripts/python.exe ^
    -DGPU_TARGETS=gfx1150 ^
    -DENABLE_CLANG_FORMAT=OFF ^
    -DHIPDNN_BUILD_PYTHON_BINDINGS=ON
  exit /b %errorlevel%
)

if "%1"=="build" (
  cmake --build %BLD% --target hipdnn_frontend_python
  exit /b %errorlevel%
)

echo Usage: build_py_bindings.bat [configure^|build]
exit /b 1
