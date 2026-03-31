#!/usr/bin/env bash

declare -a input_args
input_args="$@"

# install.sh-only flags (not passed to rmake.py)
# Add new install.sh-specific flags to this list for automatic filtering
declare -a install_sh_only_flags
install_sh_only_flags=(
  "--clean-deps"
  "--skip-aocl"
)

#use readlink rather than realpath for CentOS 6.10 support
ROCBLAS_SRC_PATH=`dirname "$(readlink -m $0)"`

/bin/ln -fs ../../.githooks/pre-commit "$(dirname "$0")/.git/hooks/"

# Distribution-agnostic version comparison function
# Returns: 0 if $1 < $2, 1 otherwise
# Usage: version_lt "3.24.0" "3.26.0" && echo "older"
version_lt() {
    local ver1=$1
    local ver2=$2
    
    # Handle empty versions
    [ -z "$ver1" ] && return 0
    [ -z "$ver2" ] && return 1
    
    # Use sort -V (version sort) to compare
    # If ver1 appears first when sorted, it's less than ver2
    [ "$ver1" = "$ver2" ] && return 1
    [ "$(printf '%s\n%s\n' "$ver1" "$ver2" | sort -V | head -n1)" = "$ver1" ]
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

# Filter out install.sh-only flags from arguments before passing to rmake.py
# Uses the install_sh_only_flags array defined at the top of the script
filter_rmake_args( )
{
  local args="$1"
  for flag in "${install_sh_only_flags[@]}"; do
    args=$(echo "${args}" | sed -e "s/${flag}//g")
  done
  echo "${args}"
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

# Variant that captures exit code without exiting - useful for special error handling
elevate_if_not_root_capture_exit( )
{
  local uid=$(id -u)
  local exit_code=0

  if (( ${uid} )); then
    sudo $@ || exit_code=$?
  else
    $@ || exit_code=$?
  fi

  return ${exit_code}
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

    local exit_code=0
    elevate_if_not_root_capture_exit zypper -n install -y ${package_dependencies} || exit_code=$?

    # Exit code 106 means some repos failed to refresh, but packages may still be available
    # Only fail if it's a different error
    if (( ${exit_code} != 0 && ${exit_code} != 106 )); then
        exit ${exit_code}
    fi
}

install_msgpack_from_source( )
{
    # Clean msgpack if requested
    if [[ "${clean_deps}" == true ]] && [[ -d "${build_dir}/deps/msgpack-c" ]]; then
      printf "\033[33mRemoving existing msgpack build (--clean-deps specified)\033[0m\n"
      rm -rf "${build_dir}/deps/msgpack-c"
    fi

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

# Detect and validate available AOCL/BLAS libraries
# INPUTS:  AOCL_ROOT (env var), HOME (env var)
# OUTPUTS: AOCL_DETECTED ("aocl_root" | "system_5x" | "none")
# EXITS:   on error if AOCL_ROOT is set but invalid
detect_aocl( )
{
    AOCL_DETECTED="none"

    # Check 1: Validate AOCL_ROOT if set (highest priority)
    if [[ -n "${AOCL_ROOT}" ]]; then
        printf "\033[36m==== Validating AOCL_ROOT ====\033[0m\n"
        printf "\033[33mAOCL_ROOT is set to: ${AOCL_ROOT}\033[0m\n"

        # Check for AOCL 5.x (libaocl)
        if [[ -f "${AOCL_ROOT}/lib/libaocl.so" ]] || [[ -f "${AOCL_ROOT}/lib/libaocl.a" ]] || \
           [[ -f "${AOCL_ROOT}/lib64/libaocl.so" ]] || [[ -f "${AOCL_ROOT}/lib64/libaocl.a" ]]; then
            printf "\033[32m✓ Found valid AOCL 5.x at AOCL_ROOT\033[0m\n"
            printf "\033[36m==============================\033[0m\n"
            AOCL_DETECTED="aocl_root"
            return 0
        fi

        # Check for AOCL 4.x BLIS structure
        for compiler in gcc aocc; do
            if [[ -f "${AOCL_ROOT}/${compiler}/lib_ILP64/libblis-mt.a" ]] || \
               [[ -f "${AOCL_ROOT}/lib_ILP64/libblis-mt.a" ]]; then
                printf "\033[32m✓ Found valid AOCL 4.x BLIS at AOCL_ROOT\033[0m\n"
                printf "\033[36m==============================\033[0m\n"
                AOCL_DETECTED="aocl_root"
                return 0
            fi
        done

        # Check for generic BLIS in AOCL_ROOT
        if [[ -f "${AOCL_ROOT}/lib/libblis.so" ]] || [[ -f "${AOCL_ROOT}/lib/libblis.a" ]] || \
           [[ -f "${AOCL_ROOT}/lib64/libblis.so" ]] || [[ -f "${AOCL_ROOT}/lib64/libblis.a" ]]; then
            printf "\033[33m⚠ Found BLIS library at AOCL_ROOT (may lack ILP64 support)\033[0m\n"
            printf "\033[36m==============================\033[0m\n"
            AOCL_DETECTED="aocl_root"
            return 0
        fi

        # AOCL_ROOT is set but no valid AOCL found - this is an error
        printf "\033[31m✗ ERROR: AOCL_ROOT is set but no valid AOCL installation found!\033[0m\n"
        printf "\033[31m  Checked: ${AOCL_ROOT}/lib*/libaocl.* (5.x)\033[0m\n"
        printf "\033[31m  Checked: ${AOCL_ROOT}/*/lib_ILP64/libblis-mt.a (4.x)\033[0m\n"
        printf "\033[31m  Checked: ${AOCL_ROOT}/lib*/libblis.* (generic)\033[0m\n"
        printf "\033[33m  Either unset AOCL_ROOT or point it to a valid AOCL installation\033[0m\n"
        printf "\033[36m==============================\033[0m\n"
        exit 2
    fi

    # Check 2: Look for system AOCL 5.x installations
    printf "\033[36m==== Checking for system AOCL 5.x ====\033[0m\n"

    # Check $HOME/aocl/<version>/{gcc,aocc}/
    if [[ -n "${HOME}" ]]; then
        for compiler_dir in ${HOME}/aocl/*/*/lib/libaocl.* ${HOME}/aocl/*/*/lib64/libaocl.*; do
            if [[ -f "${compiler_dir}" ]]; then
                local found_dir=$(dirname "$(dirname "${compiler_dir}")")
                printf "\033[32m✓ Found AOCL 5.x in HOME: ${found_dir}\033[0m\n"
                printf "\033[36m======================================\033[0m\n"
                AOCL_DETECTED="system_5x"
                return 0
            fi
        done
    fi

    # Check /opt/AMD/aocl/ for 5.x installations
    for aocl_dir in /opt/AMD/aocl/*/lib*/libaocl.*; do
        if [[ -f "${aocl_dir}" ]]; then
            local found_dir=$(dirname "$(dirname "${aocl_dir}")")
            printf "\033[32m✓ Found AOCL 5.x in /opt/AMD/aocl: ${found_dir}\033[0m\n"
            printf "\033[36m======================================\033[0m\n"
            AOCL_DETECTED="system_5x"
            return 0
        fi
    done

    # Check standard system paths
    if [[ -f "/usr/local/lib/libaocl.so" ]] || [[ -f "/usr/local/lib/libaocl.a" ]] || \
       [[ -f "/usr/lib/libaocl.so" ]] || [[ -f "/usr/lib64/libaocl.so" ]]; then
        printf "\033[32m✓ Found AOCL 5.x in system libraries\033[0m\n"
        printf "\033[36m======================================\033[0m\n"
        AOCL_DETECTED="system_5x"
        return 0
    fi

    # Nothing found
    printf "\033[33mNo system AOCL 5.x installation found\033[0m\n"
    printf "\033[36m======================================\033[0m\n"
    AOCL_DETECTED="none"
    return 1
}

# Install CMake from pre-built binary
# INPUTS:  $1=version (e.g., "3.26.0"), build_dir
# OUTPUTS: cmake_executable (updated to installed version), PATH (updated)
install_cmake( )
{
    local cmake_version="$1"
    local CMAKE_REPO="https://github.com/Kitware/CMake/releases/download"
    local CMAKE_ARCH="linux-x86_64"
    local CMAKE_TARGZ="cmake-${cmake_version}-${CMAKE_ARCH}.tar.gz"
    local CMAKE_INSTALL_DIR="${build_dir}/deps/cmake-${cmake_version}"

    # Check if already installed
    if [[ -f "${CMAKE_INSTALL_DIR}/bin/cmake" ]]; then
      printf "\033[32mCMake ${cmake_version} already installed at \033[33m${CMAKE_INSTALL_DIR}\033[0m\n"
    else
      pushd .
      mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
      printf "\033[32mDownloading CMake ${cmake_version} from \033[33m${CMAKE_REPO}/v${cmake_version}/${CMAKE_TARGZ}\033[0m\n"
      if wget -nv ${CMAKE_REPO}/v${cmake_version}/${CMAKE_TARGZ}; then
        tar -xzf ${CMAKE_TARGZ}
        rm ${CMAKE_TARGZ}
        mv cmake-${cmake_version}-${CMAKE_ARCH} cmake-${cmake_version}
        printf "\033[32m✓ CMake ${cmake_version} successfully installed!\033[0m\n"
      else
        printf "\033[31mError: Failed to download CMake ${cmake_version} from \033[33m${CMAKE_REPO}/v${cmake_version}/${CMAKE_TARGZ}\033[0m\n"
        popd
        return 1
      fi
      popd
    fi

    # Update cmake_executable to use the newly installed version
    cmake_executable="${CMAKE_INSTALL_DIR}/bin/cmake"
    export PATH="${CMAKE_INSTALL_DIR}/bin:$PATH"
    printf "\033[32mUsing cmake from: \033[33m${cmake_executable}\033[0m\n"
    ${cmake_executable} --version
}

# Build AOCL 5.2 from source
# INPUTS:  cmake_executable, build_dir, cxx, cc
# OUTPUTS: AOCL 5.2 built in ${build_dir}/deps/aocl/install_package/
build_aocl_5_2( )
{
    printf "\033[32mBuilding \033[33mAOCL 5.2\033[32m from source (preferred for testing)\033[0m\n"

    # Build AOCL 5.2 (cmake_executable should already be set to 3.26+ by this point)
    pushd .
    mkdir -p ${build_dir}/deps
    cd ${build_dir}/deps
    git clone --quiet --depth 1 --branch AOCL-5.2 https://github.com/amd/aocl.git 2>&1 | grep -v "detached HEAD"
    if [[ ! -d aocl ]]; then
        printf "\033[31mFailed to clone AOCL 5.2 into %s/deps/aocl\033[0m\n" "${build_dir}"
        popd
        return 1
    fi
    cd aocl
    # Guard against CMAKE_CONFIGURATION_TYPES=Debug performance issue (see https://github.com/amd/aocl/issues/6)
    # Explicitly set CMAKE_CONFIGURATION_TYPES=Release to prevent debug builds with multi-config generators (Ninja Multi-Config, Visual Studio)
    CXX=${cxx} CC=${cc} ${cmake_executable} -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES=Release -DBUILD_SHARED_LIBS=OFF -DENABLE_ILP64=ON -DENABLE_AOCL_BLAS=ON -DENABLE_AOCL_UTILS=ON -DENABLE_AOCL_LAPACK=OFF -DENABLE_MULTITHREADING=ON -DOpenMP_libomp_LIBRARY="" -DCMAKE_INSTALL_PREFIX=$PWD/install_package
    elevate_if_not_root ${cmake_executable} --build build --config Release -j --target install
    printf "\033[32m✓ AOCL 5.2 successfully built with ILP64 support (static)\033[0m\n"
    printf "\033[32m  Location: \033[33m${build_dir}/deps/aocl/install_package/lib/libaocl.a\033[0m\n"
    popd
}

# Setup AOCL library for rocBLAS clients
# INPUTS:  clean_deps, build_dir, skip_aocl, AOCL_DETECTED (from detect_aocl)
# OUTPUTS: AOCL available (via detection or build), AOCL_DETECTED set
setup_aocl( )
{
    # Clean AOCL if requested (before any checks)
    if [[ "${clean_deps}" == true ]] && [[ -d "${build_dir}/deps/aocl" ]]; then
      printf "\033[33mRemoving existing AOCL build (--clean-deps specified)\033[0m\n"
      rm -rf "${build_dir}/deps/aocl"
    fi

    # Detect what AOCL/BLAS is available
    detect_aocl
    local detection_result=$?

    # Decision logic based on detection results
    case "${AOCL_DETECTED}" in
        aocl_root)
            printf "\033[32mUsing AOCL at AOCL_ROOT=${AOCL_ROOT}\033[0m\n"
            return  # CMake will find it via AOCL_ROOT environment variable
            ;;
        system_5x)
            printf "\033[32mUsing system-installed AOCL 5.x (no build needed)\033[0m\n"
            return  # CMake will find it in system paths
            ;;
        none)
            # No existing AOCL found, decide what to do
            if [[ "${skip_aocl}" == true ]]; then
                printf "\033[33mSkipping AOCL 5.2 build (--skip-aocl specified)\033[0m\n"
                printf "\033[33mCMake will search for: AOCL 4.x → system CBLAS\033[0m\n"
                return  # CMake will fall back to 4.x or system BLAS
            fi

            # Default: Build AOCL 5.2 locally
            if [[ -d "${build_dir}/deps/aocl" ]]; then
                printf "\033[32mAOCL 5.2 already built at \033[33m${build_dir}/deps/aocl\033[0m\n"
                printf "\033[32m(use --clean-deps to force rebuild)\033[0m\n"
                return
            fi

            # Build AOCL 5.2
            build_aocl_5_2
            ;;
    esac
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
                                      "python3" "python3*-PyYAML" "python3-virtualenv"
                                      "gcc-c++" )
  local library_dependencies_centos_8=( "epel-release"
                                      "make" "rpm-build"
                                      "python3" "python3*-PyYAML" "python3-virtualenv"
                                      "gcc-c++" )
  local library_dependencies_rhel_8=( "epel-release"
                                      "make" "rpm-build"
                                      "python3" "python3*-PyYAML" "python3-virtualenv"
                                      "gcc-c++" )
  local library_dependencies_rhel_9=( "epel-release" "openssl-devel"
                                      "make" "rpm-build"
                                      "python3" "python3*-PyYAML" "python3-virtualenv"
                                      "gcc-c++" )
  local library_dependencies_rhel_10=( "epel-release" "openssl-devel"
                                      "make" "rpm-build"
                                      "python3" "python3*-PyYAML" "python3-virtualenv"
                                      "gcc-c++" )
  local library_dependencies_fedora=( "make" "rpm-build"
                                      "python3" "python3*-PyYAML" "python3-virtualenv"
                                      "gcc-c++" "libcxx-devel" )
  local library_dependencies_sles=(   "make" "gcc-c++" "rpm-build"
                                      "python3" "python3-PyYAML" "python3-virtualenv" "python3-pip" )

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
  if version_lt "$CMAKE_VERSION" "$CMAKE_MIN_VERSION"; then
    if $update_cmake == true; then
      library_dependencies_ubuntu+=("wget" "libssl-dev")
      library_dependencies_centos_rhel+=("wget" "openssl-devel")
      library_dependencies_centos_8+=("wget" "openssl-devel")
      library_dependencies_rhel_8+=("wget" "openssl-devel")
      library_dependencies_rhel_9+=("wget" "openssl-devel")
      library_dependencies_rhel_10+=("wget" "openssl-devel")
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
    library_dependencies_rhel_10+=( "gcc-gfortran" "libgomp" )
    library_dependencies_fedora+=( "gcc-gfortran" "libgomp" )
    library_dependencies_sles+=( "gcc-fortran" "libgomp1" )

    # wget is needed for blis
    if [[ ! -e "${build_dir}/deps/blis/lib/libblis.a" ]] && [[ ! -e "/usr/local/lib/libblis.a" ]]; then
      library_dependencies_ubuntu+=("wget")
      library_dependencies_centos_rhel+=("wget")
      library_dependencies_centos_8+=("wget")
      library_dependencies_rhel_8+=("wget")
      library_dependencies_rhel_9+=("wget")
      library_dependencies_rhel_10+=("wget")
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
      if (( "${VERSION_ID%%.*}" >= "10" )); then
        install_yum_packages "${library_dependencies_rhel_10[@]}"
      elif (( "${VERSION_ID%%.*}" >= "9" )); then
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

    --cmake_install                  Force install of CMake 3.26.0 (even if system version is sufficient).
                                     Downloads pre-built binary to <builddir>/deps.
                                     Note: CMake is auto-installed when needed (no flag required):
                                       - If system CMake < 3.24.4 (rocBLAS minimum)
                                       - If system CMake < 3.26.0 and building AOCL 5.2

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

  Dependency Management Options:

    --clean-deps                     Remove existing dependency builds before building.
                                     Affects AOCL, googletest, and msgpack.
                                     Use with --skip-aocl to clean AOCL without rebuilding it.

    --skip-aocl                      Skip AOCL 5.2 automatic build.
                                     Falls back to AOCL 4.x (if installed) → system CBLAS.
                                     Use this if you want to use an existing AOCL 4.x installation.

  Environment Variables:

    AOCL_ROOT                        Override AOCL library detection. Point to AOCL installation root.
                                     Supports both AOCL 5.x and 4.x installations.
                                     install.sh validates AOCL_ROOT and errors if invalid.
                                     Example: export AOCL_ROOT=/usr/local
                                              export AOCL_ROOT=$HOME/aocl/5.2.0/gcc
                                              export AOCL_ROOT=/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc

  BLAS Library Selection Logic (simplified):

    1. AOCL_ROOT set?        → Validate and use it (errors if invalid)
    2. System AOCL 5.x?      → Use system installation
    3. --skip-aocl NOT set?  → Build AOCL 5.2 locally (default, preferred for testing)
    4. System AOCL 4.x?      → Use it (only if --skip-aocl was set)
    5. System CBLAS?         → Use via pkg-config (OpenBLAS, etc.)

  Notes:

    - AOCL 5.2 with ILP64 support (64-bit integers) is preferred for testing/compatibility.
    - CI/Docker: Pre-install AOCL 5.2+ for faster builds, or let install.sh build it automatically.
    - CMake is automatically installed when needed - no --cmake_install flag required!
    - AOCL build takes ~5-10 minutes on first run; subsequent builds reuse existing build.
    - Without ILP64 support, stress tests may fail: use --gtest_filter=-*stress* when testing.

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
clean_deps=false
install_dependencies=false
install_package=false
rmake_invoked=false
skip_aocl=false
tensile_msgpack_backend=true
update_cmake=false


# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions build_dir:,clean-deps,cleanup,clients,clients,clients-only,cmake_install,debug,dependencies,help,install,no-msgpack,relwithdebinfo,rmake_invoked,skip-aocl --options :cdghik -- "$@")
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
    --clean-deps)
        clean_deps=true
        shift ;;
    --no-msgpack)
        tensile_msgpack_backend=false
        shift ;;
    --rmake_invoked)
        rmake_invoked=true
        shift ;;
    --skip-aocl)
        skip_aocl=true
        shift ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
  esac
done

set -x

printf "\033[32mCreating project build directory in: \033[33m${build_dir}\033[0m\n"

# #################################################
# default tools
# #################################################

# Default cmake executable is called cmake
cmake_executable=cmake
cxx="g++"
cc="gcc"
fc="gfortran"

# Constants
CMAKE_MIN_VERSION="3.24.4"  # Minimum for rocBLAS itself
CMAKE_AOCL_MIN="3.26.0"     # Minimum for AOCL 5.2 build

# #################################################
# Dependency helper functions
# #################################################

# Detect if we'll need to build AOCL 5.2
# INPUTS:  build_clients, skip_aocl, AOCL_DETECTED (from detect_aocl), build_dir
# OUTPUTS: will_build_aocl (true/false)
determine_aocl_build_requirement( )
{
    will_build_aocl=false

    # AOCL only needed for clients
    if [[ "${build_clients}" != true ]]; then
        printf "\033[36mNot building clients - AOCL/BLAS not required\033[0m\n"
        return 1
    fi

    # User explicitly skipped AOCL
    if [[ "${skip_aocl}" == true ]]; then
        return 1
    fi

    # Check if AOCL already available
    detect_aocl > /dev/null 2>&1
    if [[ "${AOCL_DETECTED}" != "none" ]]; then
        return 1
    fi

    # Check if already built locally
    if [[ -d "${build_dir}/deps/aocl" ]]; then
        return 1
    fi

    # Will need to build AOCL
    will_build_aocl=true
    printf "\033[33mDetected: Will need to build AOCL 5.2 for clients\033[0m\n"
    return 0
}

# Determine which CMake version we need and if we need to install it
# INPUTS:  cmake_executable, will_build_aocl, update_cmake, CMAKE_MIN_VERSION, CMAKE_AOCL_MIN
# OUTPUTS: need_cmake_install (true/false), cmake_target_version (version string)
# LOGIC:   - Auto-install if CMake < 3.24.4 (rocBLAS minimum)
#          - Auto-install if CMake < 3.26.0 AND building AOCL
#          - Install if --cmake_install (force, even if version is higher)
determine_cmake_requirements( )
{
    local CMAKE_VERSION=$(${cmake_executable} --version | grep -oP '(?<=version )[^ ]*' || echo "")
    need_cmake_install=false
    cmake_target_version="${CMAKE_MIN_VERSION}"

    # Check if system CMake meets rocBLAS minimum (3.24.4)
    if version_lt "$CMAKE_VERSION" "$CMAKE_MIN_VERSION"; then
        printf "\033[33mSystem CMake ${CMAKE_VERSION:-none} < ${CMAKE_MIN_VERSION} - will auto-install\033[0m\n"
        need_cmake_install=true
        cmake_target_version="${CMAKE_MIN_VERSION}"
    fi

    # Check if we need higher version for AOCL (3.26.0)
    if [[ "${will_build_aocl}" == true ]]; then
        if version_lt "$CMAKE_VERSION" "$CMAKE_AOCL_MIN"; then
            printf "\033[33mAOCL 5.2 build requires CMake >= ${CMAKE_AOCL_MIN} - will auto-install\033[0m\n"
            need_cmake_install=true
            cmake_target_version="${CMAKE_AOCL_MIN}"
        fi
    fi

    # User explicitly requested CMake install (force, even if version is sufficient)
    if [[ "$update_cmake" == true ]]; then
        printf "\033[33m--cmake_install specified - will install CMake ${CMAKE_AOCL_MIN}\033[0m\n"
        need_cmake_install=true
        cmake_target_version="${CMAKE_AOCL_MIN}"  # Install higher version when forced
    fi
}

# Add wget to package dependencies for all distros
# INPUTS:  library_dependencies_* arrays (modified in place)
# OUTPUTS: library_dependencies_* arrays (wget added)
add_wget_dependency( )
{
    library_dependencies_ubuntu+=("wget")
    library_dependencies_centos_rhel+=("wget")
    library_dependencies_centos_8+=("wget")
    library_dependencies_rhel_8+=("wget")
    library_dependencies_rhel_9+=("wget")
    library_dependencies_rhel_10+=("wget")
    library_dependencies_fedora+=("wget")
    library_dependencies_sles+=("wget")
}

# Setup CMake - install if needed
# INPUTS:  need_cmake_install, cmake_target_version
# OUTPUTS: cmake_executable (updated if installed)
setup_cmake_for_build( )
{
    if [[ "${need_cmake_install}" == true ]]; then
        printf "\033[32mAuto-installing CMake ${cmake_target_version}\033[0m\n"
        install_cmake "${cmake_target_version}"
    fi
}

# Install msgpack on distros that need it
# INPUTS:  tensile_msgpack_backend, ID (distro name)
# OUTPUTS: msgpack installed (side effect), build_dir/deps/msgpack-c created
install_msgpack_if_needed( )
{
    if [[ "${tensile_msgpack_backend}" != true ]]; then
        return
    fi

    case "${ID}" in
        centos|rhel|sles|opensuse-leap)
            install_msgpack_from_source
            ;;
    esac
}

# Install client dependencies (AOCL and googletest)
# INPUTS:  build_clients, build_dir, cmake_executable, cxx, cc, fc, ROCBLAS_SRC_PATH
# OUTPUTS: AOCL setup (via setup_aocl), googletest installed to /usr/local
install_client_dependencies( )
{
    if [[ "${build_clients}" != true ]]; then
        return
    fi

    pushd .
    mkdir -p ${build_dir}/deps && cd ${build_dir}/deps

    # Setup AOCL first (may be needed by client tests)
    setup_aocl

    # Build googletest from source, installs into cmake default /usr/local
    printf "\033[32mBuilding \033[33mgoogletest; installing into \033[33m/usr/local\033[0m\n"
    CXX=${cxx} CC=${cc} FC=${fc} ${cmake_executable} ${ROCBLAS_SRC_PATH}/deps
    make build_deps
    elevate_if_not_root make install_deps
    popd
}

# #################################################
# dependencies - main flow
# #################################################
if [[ "${install_dependencies}" == true ]]; then
  # Phase 1: Detect requirements
  # Sets: will_build_aocl
  determine_aocl_build_requirement

  # Sets: need_cmake_install, cmake_target_version
  determine_cmake_requirements

  # Phase 2: Add wget if needed for CMake download
  # Uses: need_cmake_install
  if [[ "${need_cmake_install}" == true ]]; then
    add_wget_dependency
  fi

  # Phase 3: Install system packages
  # Uses: library_dependencies_* arrays (including wget if added)
  install_packages

  # Phase 4: Setup CMake
  # Uses: need_cmake_install, cmake_target_version
  # Updates: cmake_executable (if installed)
  setup_cmake_for_build

  # Phase 5: Install additional dependencies
  install_msgpack_if_needed
  install_client_dependencies
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
  # Filter out install.sh-only flags from args passed to rmake.py
  filtered_args=$(filter_rmake_args "${input_args}")

  python3 ./rmake.py --install_invoked ${filtered_args} --build_dir=${build_dir} --src_path=${ROCBLAS_SRC_PATH}
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
