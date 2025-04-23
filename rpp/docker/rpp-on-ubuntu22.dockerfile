FROM ubuntu:22.04

ARG ROCM_INSTALLER_REPO=https://repo.radeon.com/amdgpu-install/6.1.1/ubuntu/jammy/amdgpu-install_6.1.60101-1_all.deb
ARG ROCM_INSTALLER_PACKAGE=amdgpu-install_6.1.60101-1_all.deb

ENV RPP_DEPS_ROOT=/rpp-deps
WORKDIR $RPP_DEPS_ROOT

RUN apt-get update -y
# install rpp base dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ clang cmake git wget unzip libbz2-dev python3-dev libssl-dev libomp-dev bzip2
# install ROCm for rpp OpenCL/HIP dependency
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install initramfs-tools libnuma-dev wget sudo keyboard-configuration libstdc++-12-dev &&  \
        sudo apt-get -y clean && \
        wget ${ROCM_INSTALLER_REPO} && \
        sudo apt-get install -y ./${ROCM_INSTALLER_PACKAGE} && \
        sudo apt-get update -y && \
        sudo amdgpu-install -y --usecase=rocm

RUN apt-get -y install sqlite3 libsqlite3-dev libtool build-essential half

ENV RPP_WORKSPACE=/workspace
WORKDIR $RPP_WORKSPACE

# install RPP
RUN git clone https://github.com/ROCm/rpp.git && mkdir build && cd build && \
        cmake ../rpp && make -j8 && make install