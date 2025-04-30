#!/usr/bin/env bash

# #################################################
# helper functions
# #################################################
function display_help()
{
cat <<EOF

rocSOLVER library build & installation helper script.

Usage:
  $0 (build rocsolver and put library files at ./build/rocsolver-install)
  $0 <options> (modify default behavior according to the following flags)

Options:
  -h | --help                  Print this help message.

  --build-path <builddir>      Specify path to the configure & build process output directory.
                               Relative paths are relative to the current directory.
                               (Default is ./build)

  --lib-path <libdir>          Specify path to the directory where the library generated files
                               will be located. Relative paths are relative to builddir/release
                               or builddir/debug, depending on the build type.
                               (Default is builddir/release/rocsolver-install)

  --install-path <installdir>  Specify path to the directory where the library package
                               (when generated) will be installed. Use only absolute paths.
                               (Default is /opt/rocm)

  --rocblas-path <blasdir>     Specify path to an existing rocBLAS install directory.
                               (e.g. /src/rocBLAS/build/release/rocblas-install)

  --rocsolver-path <solverdir> Specify path to an existing rocSOLVER install directory.
                               (e.g. /src/rocSOLVER/build/release/rocsolver-install)

  --rocsparse-path <sparsedir> Specify path to an existing rocSPARSE install directory.
                               (e.g. /src/rocSPARSE/build/release/rocsparse-install)

  --cleanup                    Pass this flag to remove intermediary build files after build and reduce disk usage

  -g | --debug                 Pass this flag to build in Debug mode (equivalent to set CMAKE_BUILD_TYPE=Debug).
                               (Default build type is Release)

  -p | --package               Pass this flag to generate library and client packages after build.

  -i | --install               Pass this flag to generate and install library and client packages after build.

  -d | --dependencies          Pass this flag to also build and install external dependencies.
                               Dependencies are to be installed in /usr/local. This should be done only once.
                               (this does not install rocBLAS nor ROCm software stack)

  -c | --clients               Pass this flag to also build the library clients benchmark and gtest.
                               (Generated binaries will be located at builddir/clients/staging)

  --clients-only               Pass this flag to skip building the library and only build the clients.

  --hip-clang                  Pass this flag to build using the hip-clang compiler.
                               hip-clang is currently the only supported compiler, so this flag has no effect.

  -s | --static                Pass this flag to build rocsolver as a static library.
                               (rocsolver must be built statically when the used companion rocblas is also static).

  -r | --relocatable           Pass this to add RUNPATH(based on ROCM_RPATH) and remove ldconf entry.

  -n | --no-optimizations      Pass this flag to disable optimizations for small sizes.

  --[no-]sparse                Pass this flag to add [or remove] rocSPARSE as build-time dependency.

  -a | --architecture          Set GPU architecture target, e.g. "gfx803;gfx900;gfx906;gfx908".
                               If you don't know the architecture of the GPU in your local machine, it can be
                               queried by running "mygpu".

  --address-sanitizer          Pass this flag to build with address sanitizer enabled

  --docs                       (experimental) Pass this flag to build the documentation from source.
                               Official documentation is available online at https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/index.html
                               Building locally with this flag will require docker on your machine. If you are
                               familiar with doxygen, sphinx and documentation tools, you can alternatively
                               use the scripts provided in the docs directory.

  --codecoverage               Build with code coverage profiling enabled, excluding release mode.

  -k | --relwithdebinfo        Pass this flag to build in release debug mode (equivalent to set CMAKE_BUILD_TYPE=RelWithDebInfo).
                               (Default build type is Release)

  --cmake-arg <argument>       Forward the given argument to CMake when configuring the build.
EOF
}

# Find project root directory
main=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
# true is a system command that completes successfully, function returns success
# prereq: ${ID} must be defined before calling
supported_distro( )
{
  if [ -z ${ID+foo} ]; then
    printf "supported_distro(): \$ID must be set\n"
    exit 2
  fi

  case "${ID}" in
    ubuntu|centos|rhel|fedora|sles|opensuse-leap)
        true
        ;;
    *)  printf "This script is currently supported on Ubuntu, CentOS, RHEL, SLES, OpenSUSE-Leap, and Fedora\n"
        exit 2
        ;;
  esac
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
elevate_if_not_root( )
{
  if (( $EUID )); then
    sudo "$@" || exit
  else
    "$@" || exit
  fi
}

# Take an array of packages as input, and install those packages with 'apt' if they are not already installed
install_apt_packages( )
{
  packages=("$@")
  printf "\033[32mInstalling \033[33m${packages[*]}\033[32m from distro package manager\033[0m\n"
  elevate_if_not_root apt install -y --no-install-recommends "${packages[@]}"
}

# Take an array of packages as input, and install those packages with 'yum' if they are not already installed
install_yum_packages( )
{
  packages=("$@")
  printf "\033[32mInstalling \033[33m${packages[*]}\033[32m from distro package manager\033[0m\n"
  elevate_if_not_root yum -y --nogpgcheck install "${packages[@]}"
}

# Take an array of packages as input, and install those packages with 'dnf' if they are not already installed
install_dnf_packages( )
{
  packages=("$@")
  printf "\033[32mInstalling \033[33m${packages[*]}\033[32m from distro package manager\033[0m\n"
  elevate_if_not_root dnf install -y "${packages[@]}"
}

install_zypper_packages( )
{
  packages=("$@")
  printf "\033[32mInstalling \033[33m${packages[*]}\033[32m from distro package manager\033[0m\n"
  elevate_if_not_root zypper install -y "${packages[@]}"
}

install_fmt_from_source( )
{
  fmt_version=7.1.3
  fmt_srcdir=fmt-$fmt_version-src
  fmt_blddir=fmt-$fmt_version-bld
  wget -nv -O fmt-$fmt_version.tar.gz \
      https://github.com/fmtlib/fmt/archive/refs/tags/$fmt_version.tar.gz
  rm -rf "$fmt_srcdir" "$fmt_blddir"
  tar xzf fmt-$fmt_version.tar.gz --one-top-level="$fmt_srcdir" --strip-components 1
  ${cmake_executable} \
    -H$fmt_srcdir -B$fmt_blddir \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DFMT_DOC=OFF \
    -DFMT_TEST=OFF
  make -j$(nproc) -C "$fmt_blddir"
  elevate_if_not_root make -C "$fmt_blddir" install
}

install_lapack_from_source( )
{
  lapack_version=3.9.1
  lapack_srcdir=lapack-$lapack_version-src
  lapack_blddir=lapack-$lapack_version-bld
  wget -nv -O lapack-$lapack_version.tar.gz \
      https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v$lapack_version.tar.gz
  rm -rf "$lapack_srcdir" "$lapack_blddir"
  tar xzf lapack-$lapack_version.tar.gz --one-top-level="$lapack_srcdir" --strip-components 1
  ${cmake_executable} \
    -H$lapack_srcdir -B$lapack_blddir \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_Fortran_FLAGS=-fno-optimize-sibling-calls \
    -DBUILD_TESTING=OFF \
    -DCBLAS=ON \
    -DLAPACKE=OFF
  make -j$(nproc) -C "$lapack_blddir"
  elevate_if_not_root make -C "$lapack_blddir" install
}

install_gtest_from_source( )
{
  gtest_version=1.11.0
  gtest_srcdir=gtest-$gtest_version-src
  gtest_blddir=gtest-$gtest_version-bld
  wget -nv -O gtest-$gtest_version.tar.gz \
      https://github.com/google/googletest/archive/refs/tags/release-$gtest_version.tar.gz
  rm -rf "$gtest_srcdir" "$gtest_blddir"
  tar xzf gtest-$gtest_version.tar.gz --one-top-level="$gtest_srcdir" --strip-components 1
  ${cmake_executable} \
    -H$gtest_srcdir -B$gtest_blddir \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF
  make -j$(nproc) -C "$gtest_blddir"
  elevate_if_not_root make -C "$gtest_blddir" install
}

# Take an array of packages as input, and delegate the work to the appropriate distro installer
# prereq: ${ID} must be defined before calling
# prereq: ${build_clients} must be defined before calling
install_packages( )
{
  if [ -z ${ID+foo} ]; then
    printf "install_packages(): \$ID must be set\n"
    exit 2
  fi

  if [ -z ${build_clients+foo} ]; then
    printf "install_packages(): \$build_clients must be set\n"
    exit 2
  fi

  # dependencies needed to build the rocsolver library
  local library_dependencies_ubuntu=( "make" "cmake" "wget" )
  local library_dependencies_centos_7=( "epel-release" "make" "cmake3" "rpm-build" "wget" )
  local library_dependencies_centos=( "epel-release" "make" "cmake3" "rpm-build" "wget" )
  local library_dependencies_fedora=( "make" "cmake" "rpm-build" "wget" )
  local library_dependencies_sles=( "make" "cmake" "rpm-build" "wget" )

  if [[ "${build_clients}" == true ]]; then
    # dependencies to build the clients
    library_dependencies_ubuntu+=( "gfortran" )
    library_dependencies_centos_7+=( "devtoolset-7-gcc-gfortran" )
    library_dependencies_centos+=( "gcc-gfortran" )
    library_dependencies_fedora+=( "gcc-gfortran" )
    library_dependencies_sles+=( "gcc-fortran" )
  fi

  case "${ID}" in
    ubuntu)
      elevate_if_not_root apt update
      install_apt_packages "${library_dependencies_ubuntu[@]}"
      ;;

    centos|rhel)
      if (( "${VERSION_ID%%.*}" < "8" )); then
        install_yum_packages "${library_dependencies_centos_7[@]}"
      else
        install_yum_packages "${library_dependencies_centos[@]}"
      fi
      ;;

    fedora)
      install_dnf_packages "${library_dependencies_fedora[@]}"
      ;;

    sles|opensuse-leap)
      install_zypper_packages "${library_dependencies_sles[@]}"
      ;;
    *)
      echo "This script is currently supported on Ubuntu, CentOS, RHEL, SLES, OpenSUSE-Leap, and Fedora"
      exit 2
      ;;
  esac
}

# given a relative path, returns the absolute path
make_absolute_path( ) {
  (cd "$1" && pwd -P)
}

# #################################################
# Pre-requisites check
# #################################################
# Exit code 0: all is well
# Exit code 1: problems with getopt
# Exit code 2: problems with supported platforms

# check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
  echo "This script uses getopt to parse arguments; try installing the util-linux package";
  exit 1
fi

# os-release file describes the system
if [[ -e "/etc/os-release" ]]; then
  source /etc/os-release
else
  echo "This script depends on the /etc/os-release file"
  exit 2
fi

# The following function exits script if an unsupported distro is detected
supported_distro

# #################################################
# global variables
# #################################################
rocm_path=/opt/rocm
if ! [ -z ${ROCM_PATH+x} ]; then
  rocm_path=${ROCM_PATH}
fi

install_package=false
build_package=false
install_dependencies=false
static_lib=false
build_library=true
build_clients=false
lib_dir=rocsolver-install
install_dir=${rocm_path}
build_dir=./build
build_type=Release
build_relocatable=false
build_docs=false
optimal=true
cleanup=false
build_sanitizer=false
build_codecoverage=false
unset build_with_sparse
unset architecture
unset rocblas_path
unset rocsolver_path
unset rocsparse_path
declare -a cmake_common_options
declare -a cmake_client_options

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,package,clients,clients-only,dependencies,cleanup,debug,hip-clang,codecoverage,relwithdebinfo,build_dir:,build-path:,lib_dir:,lib-path:,install_dir:,install-path:,rocblas_dir:,rocblas-path:,rocsolver_dir:,rocsolver-path:,rocsparse_dir:,rocsparse-path:,architecture:,static,relocatable,no-optimizations,sparse,no-sparse,docs,address-sanitizer,cmake-arg: --options hipcdgsrnka: -- "$@")
else
  echo "Need a new version of getopt"
  exit 1
fi

if [[ $? -ne 0 ]]; then
  echo "getopt invocation failed; could not parse the command line";
  exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
  case "${1}" in
    -h|--help)
        display_help
        exit 0
        ;;
    -i|--install)
        build_package=true
        install_package=true
        shift ;;
    -p|--package)
        build_package=true
        shift ;;
    -d|--dependencies)
        install_dependencies=true
        shift ;;
    -c|--clients)
        build_clients=true
        shift ;;
    --clients-only)
        build_library=false
        build_clients=true
        shift ;;
    -g|--debug)
        build_type=Debug
        shift ;;
    -s|--static)
        static_lib=true
        shift ;;
    --hip-clang)
        # flag has no effect; hip-clang is the default
        shift ;;
    -n | --no-optimizations)
        optimal=false
        shift ;;
    --sparse)
        build_with_sparse=true
        shift ;;
    --no-sparse)
        build_with_sparse=false
        shift ;;
    --build_dir|--build-path)
        build_dir=${2}
        shift 2;;
    --lib_dir|--lib-path)
        lib_dir=${2}
        shift 2 ;;
    --install_dir|--install-path)
        install_dir=${2}
        shift 2 ;;
    --rocblas_dir|--rocblas-path)
        rocblas_path=${2}
        shift 2 ;;
    --rocsolver_dir|--rocsolver-path)
        rocsolver_path=${2}
        shift 2 ;;
    --rocsparse_dir|--rocsparse-path)
        rocsparse_path=${2}
        shift 2 ;;
    --cleanup)
        cleanup=true
        shift ;;
    -a|--architecture)
        architecture=${2}
        shift 2 ;;
    --docs)
        build_docs=true
        shift ;;
    --address-sanitizer)
        build_sanitizer=true
        shift ;;
    -r|--relocatable)
        build_relocatable=true
        shift ;;
    --codecoverage)
        build_codecoverage=true
        shift ;;
    -k|--relwithdebinfo)
        build_type=RelWithDebInfo
        shift ;;
    --cmake-arg)
        cmake_common_options+=("${2}")
        shift 2 ;;
    --) shift ; break ;;
    *)
        echo "Unexpected command line parameter received (${1}); aborting";
        exit 1
        ;;
  esac
done

set -x
printf "\033[32mCreating project build directory in: \033[33m${build_dir}\033[0m\n"

# #################################################
# prep
# #################################################
# ensure a clean build environment
if [[ "${build_docs}" == true ]]; then
  rm -rf -- "${build_dir}/html"
elif [[ "${build_type}" == Release ]]; then
  rm -rf -- "${build_dir}/release"
elif [[ "${build_type}" == RelWithDebInfo ]]; then
  rm -rf -- "${build_dir}/release-debug"
else
  rm -rf -- "${build_dir}/debug"
fi

# resolve relative paths
if [[ -n "${rocblas_path+x}" ]]; then
  rocblas_path="$(make_absolute_path "${rocblas_path}")"
fi
if [[ -n "${rocsolver_path+x}" ]]; then
  rocsolver_path="$(make_absolute_path "${rocsolver_path}")"
fi
if [[ -n "${rocsparse_path+x}" ]]; then
  rocsparse_path="$(make_absolute_path "${rocsparse_path}")"
fi

# Default cmake executable is called cmake
cmake_executable=cmake

case "${ID}" in
  centos|rhel)
    if (( "${VERSION_ID%%.*}" < "8" )); then
      cmake_executable=cmake3
    fi
  ;;
esac

export CXX="${rocm_path}/bin/amdclang++"
export CC="${rocm_path}/bin/amdclang"
export FC="gfortran"
export PATH="${rocm_path}/bin:${rocm_path}/hip/bin:${rocm_path}/llvm/bin:${PATH}"

# #################################################
# dependencies
# #################################################
if [[ "${install_dependencies}" == true ]]; then
  install_packages

  cmake_version=$(${cmake_executable} --version | grep -oP '(?<=version )[^ ]*')
  printf "\033[32mUsing \033[33m$(command -v ${cmake_executable})\033[32m (version ${cmake_version})\033[0m\n"

  pushd .
  mkdir -p "${build_dir}/deps"
  cd "${build_dir}/deps"
  printf "\033[32mBuilding \033[33mfmt\033[32m and installing into \033[33m/usr/local\033[0m\n"
  install_fmt_from_source

  if [[ "${build_clients}" == true ]]; then
    printf "\033[32mBuilding \033[33mlapack\033[32m and installing into \033[33m/usr/local\033[0m\n"
    install_lapack_from_source
    printf "\033[32mBuilding \033[33mgoogletest\033[32m and installing into \033[33m/usr/local\033[0m\n"
    install_gtest_from_source
  fi
  popd
fi

# #################################################
# configure & build
# #################################################
pushd .

mkdir -p "$build_dir"

# build documentation
if [[ "${build_docs}" == true ]]; then
  set -eu
  container_name="build_$(head -c 10 /dev/urandom | base32)"
  docs_build_command='cp -r /mnt/rocsolver /home/docs/ && /home/docs/rocsolver/docs/run_doc.sh'
  docker build -t rocsolver:docs -f "$main/docs/Dockerfile" "$main/docs"
  docker run -v "$main:/mnt/rocsolver:ro" --name "$container_name" rocsolver:docs /bin/sh -c "$docs_build_command"
  docker cp "$container_name:/home/docs/rocsolver/build/html" "$main/build/html"
  absolute_build_dir=$(make_absolute_path "$build_dir")
  set +x
  echo 'Documentation Built:'
  echo "HTML: file://$absolute_build_dir/html/index.html"
  exit
fi

cd "$build_dir"

if [[ "${build_type}" == Debug ]]; then
  mkdir -p debug && cd debug
elif [[ "${build_type}" == RelWithDebInfo ]]; then
  mkdir -p release-debug && cd release-debug
else
  mkdir -p release && cd release
fi

cmake_common_options+=(
  "--toolchain=toolchain-linux.cmake"
  "-DROCM_PATH=${rocm_path}"
  '-DCPACK_SET_DESTDIR=OFF'
  "-DCMAKE_INSTALL_PREFIX=${lib_dir}"
  "-DCPACK_PACKAGING_INSTALL_PREFIX=${install_dir}"
  "-DCMAKE_BUILD_TYPE=${build_type}"
)

if [[ -n "${rocblas_path+x}" ]]; then
  cmake_common_options+=("-Drocblas_DIR=${rocblas_path}/lib/cmake/rocblas")
fi

if [[ -n "${rocsolver_path+x}" ]]; then
  cmake_common_options+=("-Drocsolver_DIR=${rocsolver_path}/lib/cmake/rocsolver")
fi

if [[ -n "${rocsparse_path+x}" ]]; then
  cmake_common_options+=("-Drocsparse_DIR=${rocsparse_path}/lib/cmake/rocsparse")
fi

if [[ "${static_lib}" == true ]]; then
  cmake_common_options+=('-DBUILD_SHARED_LIBS=OFF')
fi

if [[ "${optimal}" == false ]]; then
  cmake_common_options+=('-DOPTIMAL=OFF')
fi

if [[ -n "${build_with_sparse+x}" ]]; then
  if [[ "${build_with_sparse}" == true ]]; then
    cmake_common_options+=('-DBUILD_WITH_SPARSE=ON')
  else
    cmake_common_options+=('-DBUILD_WITH_SPARSE=OFF')
  fi
fi

if [[ -n "${architecture+x}" ]]; then
  cmake_common_options+=("-DAMDGPU_TARGETS=${architecture}")
fi

if [[ "${build_sanitizer}" == true ]]; then
  cmake_common_options+=('-DBUILD_ADDRESS_SANITIZER=ON')
fi

if [[ "${build_clients}" == true ]]; then
  cmake_client_options+=('-DBUILD_CLIENTS_TESTS=ON' '-DBUILD_CLIENTS_BENCHMARKS=ON' '-DBUILD_CLIENTS_SAMPLES=ON')
  cmake_client_options+=('-DBUILD_TESTING=ON' '-DBUILD_CLIENTS_EXTRA_TESTS=ON')
fi

if [[ "${build_library}" == false ]]; then
  cmake_client_options+=('-DBUILD_LIBRARY=OFF')
fi

rocm_rpath=""
if [[ "${build_relocatable}" == true ]]; then
    rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,/opt/rocm/lib:/opt/rocm/lib64"
    if ! [ -z ${ROCM_RPATH+x} ]; then
        rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,${ROCM_RPATH}"
    fi
    cmake_common_options+=('-DROCM_DISABLE_LDCONFIG=ON')
fi

if [[ "${build_codecoverage}" == true ]]; then
    if [[ "${build_type}" == Release ]]; then
        echo "Code coverage is chosen to be disabled in Release mode, to enable code coverage select either Debug mode (-g | --debug) or RelWithDebInfo mode (-k | --relwithdebinfo); aborting";
        exit 1
    fi
    cmake_common_options+=('-DBUILD_CODE_COVERAGE=ON')
fi

# check exit codes for everything from here onwards
set -eu

${cmake_executable} "${cmake_common_options[@]}" "${cmake_client_options[@]}" -DCMAKE_SHARED_LINKER_FLAGS="${rocm_rpath}" "${main}"

if [[ "${build_library}" == true ]]; then
  ${cmake_executable} --build . -j$(nproc) --target install
else
  ${cmake_executable} --build . -j$(nproc)
fi

# #################################################
# package build & install
# #################################################
# installing through package manager, which makes uninstalling easy
if [[ "${build_package}" == true ]]; then
  make package

  if [[ "${install_package}" == true ]]; then
    case "${ID}" in
      ubuntu)
        elevate_if_not_root dpkg -i rocsolver[-\_]*.deb
        ;;
      centos|rhel)
        elevate_if_not_root yum -y localinstall rocsolver-*.rpm
        ;;
      fedora)
        elevate_if_not_root dnf install rocsolver-*.rpm
        ;;
      sles|opensuse-leap)
        elevate_if_not_root zypper -n --no-gpg-checks install rocsolver-*.rpm
        ;;
    esac
  fi
fi

if [[ "${cleanup}" == true ]]; then
  rm -rf  _CPack_Packages/
  find -name '*.o' -delete
fi

popd
