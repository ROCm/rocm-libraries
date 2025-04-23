#!/usr/bin/env bash
# Author: Nico Trost

#set -x #echo on

# #################################################
# helper functions
# #################################################
function display_help()
{
  echo "rocALUTION build & installation helper script"
  echo "./install [-h|--help] "
  echo "    [-h|--help] prints this help message"
#  echo "    [--prefix] Specify an alternate CMAKE_INSTALL_PREFIX for cmake"
  echo "    [-i|--install] install after build"
  echo "    [-d|--dependencies] install build dependencies"
  echo "    [-r]--relocatable] create a package to support relocatable ROCm"
  echo "    [-c|--clients] build library clients too (combines with -i & -d)"
  echo "    [-g|--debug] -DCMAKE_BUILD_TYPE=Debug (default is =Release)"
  echo "    [-k|--relwithdebinfo] -DCMAKE_BUILD_TYPE=RelWithDebInfo"
  echo "    [--build-dir] build directory (default is ./build)"
  echo "    [--host] build library for host backend only"
  echo "    [--no-openmp] build library without OpenMP"
  echo "    [--mpi=<dir>] Path to external MPI install (Default: disabled)"
  echo "    [--static] build static library"
  echo "    [--compiler] compiler to build with"
  echo "    [--verbose] print additional cmake build information"
  echo "    [--address-sanitizer] Build with address sanitizer enabled. Uses amdclang++ as compiler"
  echo "    [--codecoverage] build with code coverage profiling enabled"
  echo "    [--rm-legacy-include-dir] Remove legacy include dir Packaging added for file/folder reorg backward compatibility"
}

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
    *)  printf "This script is currently supported on Ubuntu, CentOS, RHEL, Fedora and SLES\n"
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
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(dpkg-query --show --showformat='${db:Status-Abbrev}\n' ${package} 2> /dev/null | grep -q "ii"; echo $?) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root apt install -y --no-install-recommends ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'yum' if they are not already installed
install_yum_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(yum list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root yum -y --nogpgcheck install ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'dnf' if they are not already installed
install_dnf_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(dnf list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root dnf install -y ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'zypper' if they are not already installed
install_zypper_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(rpm -q ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root zypper -n --no-gpg-checks install ${package}
    fi
  done
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

  # dependencies needed for library and clients to build
  local library_dependencies_ubuntu=( "make" "cmake" "pkg-config" )
  local library_dependencies_centos=( "epel-release" "make" "cmake3" "gcc-c++" "rpm-build" )
  local library_dependencies_fedora=( "make" "cmake" "gcc-c++" "libcxx-devel" "rpm-build" )
  local library_dependencies_sles=( "make" "cmake" "gcc-c++" "rpm-build" )

  local client_dependencies_ubuntu=( "libboost-program-options-dev" )
  local client_dependencies_centos=( "boost-devel" )
  local client_dependencies_fedora=( "boost-devel" )
  local client_dependencies_sles=( "libboost_program_options1_66_0-devel" "pkg-config" "dpkg" )

  if [[ ( "${ID}" == "sles" ) ]]; then
    if [[ -f /etc/os-release ]]; then
      . /etc/os-release

      function version { echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }'; }
      if [[ $(version $VERSION_ID) -ge $(version 15.4) ]]; then
          library_dependencies_sles+=( "libcxxtools10" )
      else
          library_dependencies_sles+=( "libcxxtools9" )
      fi
    fi
  fi

  # Host build
  if [[ "${build_host}" == false ]]; then
    library_dependencies_ubuntu+=( "libnuma1" )
    library_dependencies_centos+=( "numactl-libs" )
    library_dependencies_fedora+=( "numactl-libs" )
    library_dependencies_sles+=( "libnuma1" )
  fi

  # OpenMP
  if [[ "${build_omp}" == true ]]; then
    library_dependencies_ubuntu+=( "libomp5" "libomp-dev" )
    library_dependencies_centos+=( "" ) # TODO
    library_dependencies_fedora+=( "" ) # TODO
    library_dependencies_sles+=( "libgomp1" )
  fi

  case "${ID}" in
    ubuntu)
      elevate_if_not_root apt update
      install_apt_packages "${library_dependencies_ubuntu[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_apt_packages "${client_dependencies_ubuntu[@]}"
      fi
      ;;

    centos|rhel)
#     yum -y update brings *all* installed packages up to date
#     without seeking user approval
#     elevate_if_not_root yum -y update
      install_yum_packages "${library_dependencies_centos[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_yum_packages "${client_dependencies_centos[@]}"
      fi
      ;;

    fedora)
#     elevate_if_not_root dnf -y update
      install_dnf_packages "${library_dependencies_fedora[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_dnf_packages "${client_dependencies_fedora[@]}"
      fi
      ;;

    sles|opensuse-leap)
#     elevate_if_not_root zypper -y update
      install_zypper_packages "${library_dependencies_sles[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_zypper_packages "${client_dependencies_sles[@]}"
      fi
      ;;
    *)
      echo "This script is currently supported on Ubuntu, CentOS, RHEL and Fedora"
      exit 2
      ;;
    *)
      echo "This script is currently supported on Ubuntu, CentOS, RHEL and Fedora"
      exit 2
      ;;
  esac
}

# Clone and build OpenMPI+UCX in rochpcg/openmpi
install_openmpi( )
{
  if [ ! -d "./deps/ucx" ]; then
    cd deps
    git clone --branch v1.10.0 https://github.com/openucx/ucx.git ucx
    cd ucx; ./autogen.sh; ./autogen.sh #why do we have to run this twice?
    mkdir build; cd build
    ../contrib/configure-opt --prefix=${PWD}/../ --with-rocm=${with_rocm} --without-knem --without-cuda --without-java
    make -j$(nproc); make install; cd ../../..
  fi

  if [ ! -d "./deps/openmpi" ]; then
    cd deps
    git clone --branch v4.1.0 https://github.com/open-mpi/ompi.git openmpi
    cd openmpi; ./autogen.pl; mkdir build; cd build
    ../configure --prefix=${PWD}/../ --with-ucx=${PWD}/../../ucx --without-verbs
    make -j$(nproc); make install; cd ../../..
  fi
}

# #################################################
# Pre-requisites check
# #################################################
# Exit code 0: alls well
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
install_package=false
install_dependencies=false
build_clients=false
build_host=false
build_mpi=false
mpi_dir=deps/openmpi
build_omp=true
build_release=true
build_release_debug=false
build_dir=./build
install_prefix=rocalution-install
rocm_path=/opt/rocm
build_relocatable=false
build_static=false
build_address_sanitizer=false
build_codecoverage=false
build_freorg_bkwdcomp=false
compiler=c++
verb=false

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,clients,dependencies,debug,build-dir:,host,no-openmp,mpi:,relocatable,codecoverage,relwithdebinfo,static,compiler:,verbose,address-sanitizer,rm-legacy-include-dir --options hicgdrk -- "$@")
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
        install_package=true
        shift ;;
    -d|--dependencies)
        install_dependencies=true
        shift ;;
    -c|--clients)
        build_clients=true
        shift ;;
    -r|--relocatable)
        build_relocatable=true
        shift ;;
    -g|--debug)
        build_release=false
        shift ;;
    --build-dir)
        build_dir=${2}
        shift 2 ;;
    --host)
        build_host=true
        shift ;;
    --no-openmp)
        build_omp=false
        shift ;;
    --address-sanitizer)
        build_address_sanitizer=true
        compiler=amdclang++
        shift ;;
    -k|--relwithdebinfo)
        build_release=false
        build_release_debug=true
        shift ;;
    --codecoverage)
        build_codecoverage=true
        shift ;;
    --rm-legacy-include-dir)
        build_freorg_bkwdcomp=false
        shift ;;
    --mpi)
        mpi_dir=${2}
        if [[ "${mpi_dir}" == off || "${mpi_dir}" == false || "${mpi_dir}" == 0 || "${mpi_dir}" == disabled ]]; then
          build_mpi=false
        else
          build_mpi=true
        fi
        shift 2 ;;
    --static)
        #Forcing amdclang++ for static builds temporarily
        compiler=${rocm_path}/bin/amdclang++
        build_static=true
        shift ;;
    --compiler)
        compiler=${2}
        shift 2 ;;
    --verbose)
        verb=true
        shift ;;
    --prefix)
        install_prefix=${2}
        shift 2 ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
  esac
done

printf "\033[32mCreating project build directory in: \033[33m${build_dir}\033[0m\n"

# #################################################
# prep
# #################################################
# ensure a clean build environment
if [[ "${build_release}" == true ]]; then
  rm -rf ${build_dir}/release
elif [[ "${build_release_debug}" == true ]]; then
  rm -rf ${build_dir}/release-debug
else
  rm -rf ${build_dir}/debug
fi

if [[ "${build_relocatable}" == true ]]; then
    if ! [ -z ${ROCM_PATH+x} ]; then
        rocm_path=${ROCM_PATH}
    fi

    rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,/opt/rocm/lib:/opt/rocm/lib64"
    if ! [ -z ${ROCM_RPATH+x} ]; then
        rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,${ROCM_RPATH}"
    fi
fi

# Default cmake executable is called cmake
cmake_executable=cmake

case "${ID}" in
  centos|rhel)
  cmake_executable=cmake3
  ;;
esac

# #################################################
# dependencies
# #################################################
if [[ "${install_dependencies}" == true ]]; then

  install_packages

  # The following builds googletest from source, installs into cmake default /usr/local
  pushd .
    printf "\033[32mBuilding \033[33mgoogletest\033[32m from source; installing into \033[33m/usr/local\033[0m\n"
    mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
    ${cmake_executable} ../../deps
    make -j$(nproc)
    elevate_if_not_root make install
  popd
fi

# We append customary rocm path; if user provides custom rocm path in ${path}, our
# hard-coded path has lesser priority
if [[ "${build_relocatable}" == true ]]; then
    export PATH=${rocm_path}/bin:${PATH}
else
    export PATH=${PATH}:/opt/rocm/bin
fi

pushd .

  # #################################################
  # MPI
  # #################################################
  if [[ "${build_mpi}" == true ]]; then

    shopt -s nocasematch

    echo "${mpi_dir}"

    # If asking for MPI build, but no valid installation dir is given, install it
    if [[ "${mpi_dir}" == on || "${mpi_dir}" == true || "${mpi_dir}" == 1 || "${mpi_dir}" == enabled ]]; then
      install_openmpi
      mpi_dir=deps/openmpi
    fi

    shopt -u nocasematch

  fi

  # #################################################
  # configure & build
  # #################################################
  cmake_common_options=""
  cmake_client_options=""

  # build type
  if [[ "${build_release}" == true ]]; then
    mkdir -p ${build_dir}/release/clients && cd ${build_dir}/release
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Release"
  elif [[ "${build_release_debug}" == true ]]; then
    mkdir -p ${build_dir}/release-debug/clients && cd ${build_dir}/release-debug
    cmake_common_options+=("-DCMAKE_BUILD_TYPE=RelWithDebInfo")
  else
    mkdir -p ${build_dir}/debug/clients && cd ${build_dir}/debug
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Debug"
  fi

  # library type
  if [[ "${build_static}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_SHARED_LIBS=OFF"
  fi

  # clients
  if [[ "${build_clients}" == true ]]; then
    cmake_client_options="${cmake_client_options} -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON"
  fi

  # OpenMP
  if [[ "${build_omp}" == false ]]; then
    cmake_common_options="${cmake_common_options} -DSUPPORT_OMP=OFF"
  fi

  # MPI
  if [[ "${build_mpi}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DSUPPORT_MPI=ON -DROCALUTION_MPI_DIR=${mpi_dir}"
  fi

  # HIP / Host only
  if [[ "${build_host}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DSUPPORT_HIP=OFF"
  else
    cmake_common_options="${cmake_common_options} -DSUPPORT_HIP=ON"
  fi

  # sanitizer
  if [[ "${build_address_sanitizer}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_ADDRESS_SANITIZER=ON"
  fi

  # code coverage
  if [[ "${build_codecoverage}" == true ]]; then
      if [[ "${build_release}" == true ]]; then
          echo "Code coverage is disabled in Release mode, to enable code coverage select either Debug mode (-g | --debug) or RelWithDebInfo mode (-k | --relwithdebinfo); aborting";
          exit 1
      fi
      cmake_common_options="${cmake_common_options} -DBUILD_CODE_COVERAGE=ON"
  fi

  #Enable backward compatibility wrappers
  if [[ "${build_freorg_bkwdcomp}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=ON"
  else
    cmake_common_options="${cmake_common_options} -DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=OFF"
  fi

  # Verbose cmake
  if [[ "${verb}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_VERBOSE=ON"
  fi

  # cpack
  if [[ "${build_relocatable}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DCPACK_SET_DESTDIR=OFF -DCPACK_PACKAGING_INSTALL_PREFIX=${rocm_path}"
  else
    cmake_common_options="${cmake_common_options} -DCPACK_SET_DESTDIR=OFF -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm"
  fi

  # Build library with AMD toolchain because of existense of device kernels
  if [[ "${build_relocatable}" == true ]]; then
    CXX=${compiler} ${cmake_executable} ${cmake_common_options} ${cmake_client_options} \
      -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
      -DCMAKE_SHARED_LINKER_FLAGS="${rocm_rpath}" \
      -DCMAKE_PREFIX_PATH="${rocm_path} ${rocm_path}/hcc ${rocm_path}/hip" \
      -DCMAKE_MODULE_PATH="${rocm_path}/lib/cmake/hip ${rocm_path}/hip/cmake" \
      -DCMAKE_EXE_LINKER_FLAGS=" -Wl,--enable-new-dtags -Wl,--rpath,${rocm_path}/lib:${rocm_path}/lib64" \
      -DROCM_DISABLE_LDCONFIG=ON \
      -DROCM_PATH="${rocm_path}" ../.. || exit
  else
    CXX=${compiler} ${cmake_executable} ${cmake_common_options} ${cmake_client_options} -DCMAKE_INSTALL_PREFIX=${install_prefix} -DROCM_PATH=${rocm_path} ../.. || exit
  fi

  make -j$(nproc) install || exit

  # #################################################
  # install
  # #################################################
  # installing through package manager, which makes uninstalling easy
  if [[ "${install_package}" == true ]]; then
    make package || exit

    case "${ID}" in
      ubuntu)
        elevate_if_not_root dpkg -i rocalution[-\_]*.deb
      ;;
      centos|rhel)
        elevate_if_not_root yum -y localinstall rocalution-*.rpm
      ;;
      fedora)
        elevate_if_not_root dnf install rocalution-*.rpm
      ;;
      sles|opensuse-leap)
        elevate_if_not_root zypper -n --no-gpg-checks install rocalution-*.rpm
      ;;
    esac

  fi
popd
