#!/usr/bin/env bash

declare -a input_args
input_args="$@"

#use readlink rather than realpath for CentOS 6.10 support
ROCBLAS_SRC_PATH=`dirname "$(readlink -m $0)"`

/bin/ln -fs ../../.githooks/pre-commit "$(dirname "$0")/.git/hooks/"

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
check_exit_code( )
{
  if (( $1 != 0 )); then
    exit $1
  fi
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
elevate_if_not_root( )
{
  local uid=$(id -u)

  if (( ${uid} )); then
    sudo $@
    check_exit_code "$?"
  else
    $@
    check_exit_code "$?"
  fi
}

# Take an array of packages as input, and install those packages with 'apt' if they are not already installed
install_apt_packages( )
{
  package_dependencies="$@"
  printf "\033[32mInstalling following packages from distro package manager: \033[33m${package_dependencies}\033[32m \033[0m\n"
  elevate_if_not_root apt-get -y --no-install-recommends install ${package_dependencies}
}

# Take an array of packages as input, and install those packages with 'yum' if they are not already installed
install_yum_packages( )
{
  package_dependencies="$@"
  printf "\033[32mInstalling following packages from distro package manager: \033[33m${package_dependencies}\033[32m \033[0m\n"
  elevate_if_not_root yum -y --nogpgcheck install ${package_dependencies}
}

# Take an array of packages as input, and install those packages with 'dnf' if they are not already installed
install_dnf_packages( )
{
  package_dependencies="$@"
  printf "\033[32mInstalling following packages from distro package manager: \033[33m${package_dependencies}\033[32m \033[0m\n"
  elevate_if_not_root dnf install -y ${package_dependencies}
}

install_zypper_packages( )
{
    package_dependencies="$@"
    printf "\033[32mInstalling following packages from distro package manager: \033[33m${package_dependencies}\033[32m \033[0m\n"
    elevate_if_not_root zypper install -y ${package_dependencies}
}

install_msgpack_from_source( )
{
    if [[ ! -d "${build_dir}/deps/msgpack-c" ]]; then
      pushd .
      mkdir -p ${build_dir}/deps
      cd ${build_dir}/deps
      git clone -b cpp-3.0.1 https://github.com/msgpack/msgpack-c.git
      cd msgpack-c
      CXX=${cxx} CC=${cc} ${cmake_executable} -DMSGPACK_BUILD_TESTS=OFF -DMSGPACK_BUILD_EXAMPLES=OFF .
      make
      elevate_if_not_root make install
      popd
    fi
}

# Take an array of packages as input, and delegate the work to the appropriate distro installer
# prereq: ${ID} must be defined before calling
# prereq: ${build_clients} must be defined before calling
# prereq: ${tensile_msgpack_backend} must be defined before calling
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

  # dependencies needed to build the rocblas library
  # Note python3-joblib is for Tensile and also installed by pip requirements.txt but there are known packaging errors so added here as workaround
  local library_dependencies_ubuntu=( "make"
                                      "python3" "python3-yaml" "python3-venv" "python3*-pip" )
  local library_dependencies_centos_rhel=( "epel-release"
                                      "make" "rpm-build"
                                      "python34" "python3*-PyYAML" "python3-virtualenv"
                                      "gcc-c++" )
  local library_dependencies_centos_8=( "epel-release"
                                      "make" "rpm-build"
                                      "python3" "python3*-PyYAML" "python3-virtualenv"
                                      "gcc-c++" )
  local library_dependencies_rhel_8=( "epel-release"
                                      "make" "rpm-build"
                                      "python36" "python3*-PyYAML" "python3-virtualenv"
                                      "gcc-c++" )
  local library_dependencies_rhel_9=( "epel-release" "openssl-devel"
                                      "make" "rpm-build"
                                      "python39" "python3*-PyYAML" "python3-virtualenv"
                                      "gcc-c++" )
  local library_dependencies_fedora=( "make" "rpm-build"
                                      "python34" "python3*-PyYAML" "python3-virtualenv"
                                      "gcc-c++" "libcxx-devel" )
  local library_dependencies_sles=(   "make" "python3-PyYAML" "python3-virtualenv"
                                      "gcc-c++" "rpm-build" )

  if [[ "${tensile_msgpack_backend}" == true ]]; then
    library_dependencies_ubuntu+=("libmsgpack-dev")
    library_dependencies_fedora+=("msgpack-devel")
  fi

  # wget is needed for msgpack in this case
  if [[ ("${ID}" == "ubuntu") && ("${VERSION_ID}" == "16.04") && "${tensile_msgpack_backend}" == true ]]; then
    if ! $(dpkg -s "libmsgpackc2" &> /dev/null) || $(dpkg --compare-versions $(dpkg-query -f='${Version}' --show libmsgpackc2) lt 2.1.5-1); then
      library_dependencies_ubuntu+=("wget")
    fi
  fi

  # wget and openssl are needed for cmake
  if [ -z "$CMAKE_VERSION" ] || $(dpkg --compare-versions $CMAKE_VERSION lt $CMAKE_MIN_VERSION); then
    if $update_cmake == true; then
      library_dependencies_ubuntu+=("wget" "libssl-dev")
      library_dependencies_centos_rhel+=("wget" "openssl-devel")
      library_dependencies_centos_8+=("wget" "openssl-devel")
      library_dependencies_rhel_8+=("wget" "openssl-devel")
      library_dependencies_rhel_9+=("wget" "openssl-devel")
      library_dependencies_fedora+=("wget")
      library_dependencies_sles+=("wget" "libopenssl-devel")
    fi
  fi

  if [[ "${build_clients}" == true ]]; then
    # dependencies to build the client
    library_dependencies_ubuntu+=( "gfortran" "libomp-dev" )
    library_dependencies_centos_rhel+=( "devtoolset-7-gcc-gfortran" "libgomp" )
    library_dependencies_centos_8+=( "gcc-gfortran" "libgomp" )
    library_dependencies_rhel_8+=( "gcc-gfortran" "libgomp" )
    library_dependencies_rhel_9+=( "gcc-gfortran" "libgomp" )
    library_dependencies_fedora+=( "gcc-gfortran" "libgomp" )
    library_dependencies_sles+=( "gcc-fortran" "libgomp1" )

    # wget is needed for blis
    if [[ ! -e "${build_dir}/deps/blis/lib/libblis.a" ]] && [[ ! -e "/usr/local/lib/libblis.a" ]]; then
      library_dependencies_ubuntu+=("wget")
      library_dependencies_centos_rhel+=("wget")
      library_dependencies_centos_8+=("wget")
      library_dependencies_rhel_8+=("wget")
      library_dependencies_rhel_9+=("wget")
      library_dependencies_fedora+=("wget")
      library_dependencies_sles+=("wget")
    fi
  fi

  case "${ID}" in
    ubuntu)
#     elevate_if_not_root apt-get update
      install_apt_packages "${library_dependencies_ubuntu[@]}"
      ;;

    centos)
      if (( "${VERSION_ID%%.*}" >= "8" )); then
        install_yum_packages "${library_dependencies_centos_8[@]}"
      else
  #     yum -y update brings *all* installed packages up to date
  #     without seeking user approval
  #     elevate_if_not_root yum -y update
        install_yum_packages "${library_dependencies_centos_rhel[@]}"
      fi
      ;;

    rhel)
      if (( "${VERSION_ID%%.*}" >= "9" )); then
        install_yum_packages "${library_dependencies_rhel_9[@]}"
      elif (( "${VERSION_ID%%.*}" >= "8" )); then
        install_yum_packages "${library_dependencies_rhel_8[@]}"
      else
  #     yum -y update brings *all* installed packages up to date
  #     without seeking user approval
  #     elevate_if_not_root yum -y update
        install_yum_packages "${library_dependencies_centos_rhel[@]}"
      fi
      ;;

    fedora)
#     elevate_if_not_root dnf -y update
      install_dnf_packages "${library_dependencies_fedora[@]}"
      ;;

    sles|opensuse-leap)
      if (( "${VERSION_ID%%.*}" >= "15" )); then
        library_dependencies_sles+=( "libcxxtools10" )
      else
        library_dependencies_sles+=( "libcxxtools9" )
      fi
      install_zypper_packages "${library_dependencies_sles[@]}"
      ;;

    *)
      echo "This script is currently supported on Ubuntu, CentOS, RHEL, SLES, OpenSUSE-Leap, and Fedora"
      exit 2
      ;;
  esac

  if [[ ("${ID}" == "ubuntu") && ("${VERSION_ID}" == "16.04") && "${tensile_msgpack_backend}" == true ]]; then
    # On Ubuntu 16.04, the version of msgpack provided in the repository is outdated, so a newer version
    # must be manually downloaded and installed.  Trying to match or exceed Ubuntu 18 default
    if ! $(dpkg -s "libmsgpackc2" &> /dev/null) || $(dpkg --compare-versions $(dpkg-query -f='${Version}' --show libmsgpackc2) lt 2.1.5-1); then
      wget -nv -P ./ "http://ftp.us.debian.org/debian/pool/main/m/msgpack-c/libmsgpackc2_3.0.1-3_amd64.deb"
      wget -nv -P ./ "http://ftp.us.debian.org/debian/pool/main/m/msgpack-c/libmsgpack-dev_3.0.1-3_amd64.deb"
      elevate_if_not_root dpkg -i ./libmsgpackc2_3.0.1-3_amd64.deb ./libmsgpack-dev_3.0.1-3_amd64.deb
      rm libmsgpack-dev_3.0.1-3_amd64.deb libmsgpackc2_3.0.1-3_amd64.deb
    fi
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

# /etc/*-release files describe the system
if [[ -e "/etc/os-release" ]]; then
  source /etc/os-release
elif [[ -e "/etc/centos-release" ]]; then
  ID=$(cat /etc/centos-release | awk '{print tolower($1)}')
  VERSION_ID=$(cat /etc/centos-release | grep -oP '(?<=release )[^ ]*' | cut -d "." -f1)
else
  echo "This script depends on the /etc/*-release files"
  exit 2
fi

# The following function exits script if an unsupported distro is detected
supported_distro

# #################################################
# helper functions
# #################################################
function display_help()
{
cat <<EOF
rocBLAS dependency & installation helper script. Invokes rmake.py for build steps, you may directly invoke rmake.py instead.

  Usage:
    $0 (build rocBLAS and put library files at e.g. <builddir>/release/rocblas-install)
    $0 <options> (modify default behavior according to the following flags)

  General Build Options:
    --build_dir <builddir>           Specify the directory path to build and save library files, dependencies and executables.
                                     Relative paths are relative to the current directory. (Default is ./build)

    --cleanup                        Remove intermediary build files after build and reduce disk usage.

    -c, --clients                    Build the library clients benchmark and gtest.
                                     (Generated binaries will be located at <builddir>/release/clients/staging)
    --clients-only                   Skip building the library and only build the clients with a pre-built library.

    --cmake_install                  Install minimum cmake version if required.

    -d, --dependencies               Build and install external dependencies.
                                     Dependencies are to be installed in /usr/local. This should be done only once.

    -g, --debug                      Build-in Debug mode, equivalent to set CMAKE_BUILD_TYPE=Debug.
                                     (Default build type is Release)

    -h, --help                       Print this help message

    -i, --install                    Generate and install library package after build.

  Experimental Build Options:

    -k, --relwithdebinfo             Build-in release debug mode, equivalent to set CMAKE_BUILD_TYPE=RelWithDebInfo.
                                     (Default build type is Release)

    --no-msgpack                     Build Tensile backend not to use MessagePack.

EOF
}

# #################################################
# option parsed variable defaults
# #################################################
build_cleanup=false
build_clients=false
#use readlink rather than realpath for CentOS 6.10 support
build_dir=$(readlink -m ./build)
build_release=true
build_release_debug=false
install_dependencies=false
install_package=false
rmake_invoked=false
tensile_msgpack_backend=true
update_cmake=false


# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions build_dir:,cleanup,clients,clients,clients-only,cmake_install,debug,dependencies,help,install,no-msgpack,relwithdebinfo,rmake_invoked --options :cdghik -- "$@")
else
  echo "Need a new version of getopt"
  exit 1
fi

# don't check args as rmake.py handles additional options
# if [[ $? -ne 0 ]]; then
#   echo "getopt invocation failed; could not parse the command line";
#   exit 1
# fi

eval set -- "${GETOPT_PARSE}"

while true; do
  case "${1}" in
    -h|--help)
        display_help
        python3 ./rmake.py --help
        exit 0
        ;;
    -i|--install)
        install_package=true
        shift ;;
    -d|--dependencies)
        install_dependencies=true
        shift ;;
    --cleanup)
        build_cleanup=true
        shift ;;
    -c|--clients)
        build_clients=true
        shift ;;
    --clients-only)
        build_clients=true
        shift ;;
    --build_dir)
        #use readlink rather than realpath for CentOS 6.10 support
        build_dir=$(readlink -m ${2})
        shift 2;;
    --cmake_install)
        update_cmake=true
        shift ;;
    -g|--debug)
        build_release=false
        shift ;;
    -k|--relwithdebinfo)
        build_release=false
        build_release_debug=true
        shift ;;
    --no-msgpack)
        tensile_msgpack_backend=false
        shift ;;
    --rmake_invoked)
        rmake_invoked=true
        shift ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
  esac
done

set -x

printf "\033[32mCreating project build directory in: \033[33m${build_dir}\033[0m\n"

install_blis()
{
    if [[ ! -e "/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libblis-mt.a" ]] &&
        [[ ! -e "/opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/lib_ILP64/libblis-mt.a" ]] &&
        [[ ! -e "/opt/AMD/aocl/aocl-linux-aocc-4.0/lib_ILP64/libblis-mt.a"  ]] &&
        [[ ! -e "/usr/local/lib/libblis.a" ]]; then
        pushd .
        #Download prebuilt AMD multithreaded blis
        if [[ ! -e "./blis/lib/libblis.a" ]]; then
          case "${ID}" in
              centos|rhel|sles|opensuse-leap)
                  wget -nv -O blis.tar.gz https://github.com/amd/blis/releases/download/2.0/aocl-blis-mt-centos-2.0.tar.gz
                  ;;
              ubuntu)
                  wget -nv -O blis.tar.gz https://github.com/amd/blis/releases/download/2.0/aocl-blis-mt-ubuntu-2.0.tar.gz
                  ;;
              *)
                  echo "Unsupported OS for this script"
                  wget -nv -O blis.tar.gz https://github.com/amd/blis/releases/download/2.0/aocl-blis-mt-ubuntu-2.0.tar.gz
                  ;;
          esac

          tar -xvf blis.tar.gz
          rm -rf blis/amd-blis-mt
          mv amd-blis-mt blis
          rm blis.tar.gz
          cd blis/lib
          ln -sf libblis-mt.a libblis.a
        fi
        popd
    fi
}

# #################################################
# default tools
# #################################################

# Default cmake executable is called cmake
cmake_executable=cmake
cxx="g++"
cc="gcc"
fc="gfortran"

# #################################################
# dependencies
# #################################################
if [[ "${install_dependencies}" == true ]]; then
  CMAKE_VERSION=$(${cmake_executable} --version | grep -oP '(?<=version )[^ ]*')
  CMAKE_MIN_VERSION="3.24.4"

  install_packages

  if [ -z "$CMAKE_VERSION" ] || $(dpkg --compare-versions $CMAKE_VERSION lt $CMAKE_MIN_VERSION); then
      if $update_cmake == true; then
        pushd .
        printf "\033[32mBuilding \033[33mcmake\033[32m from source; installing into \033[33m/usr/local\033[0m\n"
        CMAKE_REPO="https://github.com/Kitware/CMake/releases/download"
        CMAKE_TARGZ="cmake-${CMAKE_MIN_VERSION}.tar.gz"
        mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
        wget -nv ${CMAKE_REPO}/v${CMAKE_MIN_VERSION}/${CMAKE_TARGZ}
        tar -xvf ${CMAKE_TARGZ}
        rm ${CMAKE_TARGZ}
        cd cmake-${CMAKE_MIN_VERSION}
        ./bootstrap --no-system-curl --parallel=16
        make -j16
        elevate_if_not_root make install
        popd
      else
          echo "rocBLAS requires CMake version >= ${CMAKE_MIN_VERSION} and CMake version ${CMAKE_VERSION} is installed. Run install.sh again with --cmake_install flag and CMake version ${CMAKE_MIN_VERSION} will be installed to /usr/local"
          exit 2
      fi
  fi

  # cmake is needed to install msgpack
  case "${ID}" in
    centos|rhel|sles|opensuse-leap)
      if [[ "${tensile_msgpack_backend}" == true ]]; then
        install_msgpack_from_source
      fi
      ;;
  esac

  if [[ "${build_clients}" == true ]]; then
    # The following builds googletest from source, installs into cmake default /usr/local
    pushd .
    mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
    install_blis

    # build other deps
    printf "\033[32mBuilding \033[33mgoogletest; installing into \033[33m/usr/local\033[0m\n"
    CXX=${cxx} CC=${cc} FC=${fc} ${cmake_executable} ${ROCBLAS_SRC_PATH}/deps
    make build_deps
    elevate_if_not_root make install_deps
    popd
  fi
elif [[ "${build_clients}" == true ]]; then
  pushd .
  mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
  install_blis
  popd
fi

# #################################################
# configure & build
# #################################################

full_build_dir=""
if [[ "${build_release}" == true ]]; then
  full_build_dir=${build_dir}/release
elif [[ "${build_release_debug}" == true ]]; then
  full_build_dir=${build_dir}/release-debug
else
  full_build_dir=${build_dir}/debug
fi

if [[ "${rmake_invoked}" == false ]]; then
  pushd .

  # ensure a clean build environment
  rm -rf ${full_build_dir}

  #rmake.py at top level same as install.sh
  python3 ./rmake.py --install_invoked ${input_args} --build_dir=${build_dir} --src_path=${ROCBLAS_SRC_PATH}
  check_exit_code "$?"

  popd
else
  # only dependency install supported when called from rmake
  exit 0
fi

# #################################################
# install
# #################################################

pushd .

cd ${full_build_dir}

# installing through package manager, which makes uninstalling easy
if [[ "${install_package}" == true ]]; then
  make package
  check_exit_code "$?"

  case "${ID}" in
    ubuntu)
      elevate_if_not_root dpkg -i rocblas[_\-]*.deb
    ;;
    centos|rhel)
      elevate_if_not_root yum -y localinstall rocblas-*.rpm
    ;;
    fedora)
      elevate_if_not_root dnf install rocblas-*.rpm
    ;;
    sles|opensuse-leap)
      elevate_if_not_root zypper -n --no-gpg-checks install rocblas-*.rpm
    ;;
  esac

fi
check_exit_code "$?"

if [[ "${build_cleanup}" == true ]]; then
    find -name '*.o' -delete
    find -type d -name '*build_tmp*' -exec rm -rf {} +
    find -type d -name '*_CPack_Packages*' -exec rm -rf {} +
fi
check_exit_code "$?"

popd
