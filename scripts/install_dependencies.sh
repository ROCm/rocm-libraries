#!/bin/bash

set -e
set -o pipefail

if [[ -z "${_ROCM_DIR}" ]]; then
  export _ROCM_DIR=/opt/rocm
fi

# Location of dependencies source code
export _INSTALL_DIR=$BUILD_DIR/install
export _DEPS_SRC_DIR=$_INSTALL_DIR/src

mkdir -p $_DEPS_SRC_DIR

#Adjust branches and installation location as necessary
export _UCX_INSTALL_DIR=$_INSTALL_DIR/ucx
export _UCX_REPO=https://github.com/ROCm/ucx.git
export _UCX_COMMIT_HASH=18770fdc1c3b5de202d14a088a14b734d2c4bbf3

export _OMPI_INSTALL_DIR=$_INSTALL_DIR/ompi
export _OMPI_REPO=https://github.com/ROCm/ompi.git
export _OMPI_COMMIT_HASH=720f556508ad3f2cbb17341eb184c2d8565a5133

# Step 1: Build UCX with ROCm support
cd $_DEPS_SRC_DIR
rm -rf ucx
git clone $_UCX_REPO
cd ucx
git checkout $_UCX_COMMIT_HASH
./autogen.sh
./contrib/configure-release --prefix=$_UCX_INSTALL_DIR \
                            --with-rocm=$_ROCM_DIR     \
                            --enable-mt                \
                            --without-go               \
                            --without-java             \
                            --without-cuda             \
                            --without-knem
make -j
make install

# Step 2: Install OpenMPI with UCX support
cd $_DEPS_SRC_DIR
rm -rf ompi
git clone --recursive $_OMPI_REPO
cd ompi
git checkout $_OMPI_COMMIT_HASH
git submodule update --init --recursive
./autogen.pl
./configure --prefix=$_OMPI_INSTALL_DIR  \
            --with-rocm=$_ROCM_DIR       \
            --with-ucx=$_UCX_INSTALL_DIR \
            --disable-oshmem             \
            --with-prrte=internal        \
            --with-hwloc=internal        \
            --with-libevent=internal     \
            --without-cuda               \
            --disable-mpi-fortran        \
            --without-ofi
make -j
make install

rm -rf $_DEPS_SRC_DIR

echo "Dependencies for rocSHMEM are now installed"
echo ""
echo "UCX ($_UCX_COMMIT_HASH) Installed to $_UCX_INSTALL_DIR"
echo "OpenMPI ($_OMPI_COMMIT_HASH) Installed to $_OMPI_INSTALL_DIR"
echo ""
echo "Please update your PATH and LD_LIBRARY_PATH"
