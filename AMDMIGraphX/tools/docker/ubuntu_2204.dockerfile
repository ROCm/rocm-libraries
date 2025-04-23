FROM ubuntu:22.04

ARG PREFIX=/usr/local

# Support multiarch
RUN dpkg --add-architecture i386

# Install rocm key
RUN apt-get update && apt-get install -y gnupg2 --no-install-recommends curl && \
    curl -fsSL http://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm-keyring.gpg

# Add rocm repository
RUN sh -c "echo 'deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] http://repo.radeon.com/rocm/apt/6.2 jammy main' > /etc/apt/sources.list.d/rocm.list"

# From docs.amd.com for installing rocm. Needed to install properly
RUN sh -c "echo 'Package: *\nPin: release o=repo.radeon.com\nPin-priority: 600' > /etc/apt/preferences.d/rocm-pin-600"


# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    build-essential \
    #clang-format-10 \
    cmake \
    curl \
    doxygen \
    #g++-7 \
    gdb \
    git \
    lcov \
    locales \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    software-properties-common \
    wget \
    rocm-device-libs \
    hip-dev \
    libnuma-dev \
    miopen-hip \
    rocblas \
    hipfft \
    rocthrust \
    rocrand \
    hipsparse \
    rccl \
    rccl-dev \
    rocm-smi-lib \
    rocm-dev \
    roctracer-dev \
    hipcub  \
    hipblas  \
    hipify-clang \
    half \
    libssl-dev \
    zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# add this for roctracer dependancies
RUN pip3 install CppHeaderParser

# Workaround broken rocm packages
RUN ln -s /opt/rocm-* /opt/rocm
RUN echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf
RUN echo "/opt/rocm/llvm/lib" > /etc/ld.so.conf.d/rocm-llvm.conf
RUN ldconfig

RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install dependencies
ADD dev-requirements.txt /dev-requirements.txt
ADD requirements.txt /requirements.txt
ADD rbuild.ini /rbuild.ini

COPY ./tools/install_prereqs.sh /
COPY ./tools/requirements-py.txt / 
RUN /install_prereqs.sh /usr/local / && rm /install_prereqs.sh && rm /requirements-py.txt
RUN test -f /usr/local/hash || exit 1

# Install yapf
RUN pip3 install yapf==0.28.0

# Install doc requirements
ADD docs/sphinx/requirements.txt /doc-requirements.txt
RUN pip3 install -r /doc-requirements.txt

# Install latest ccache version
RUN cget -p $PREFIX install facebook/zstd@v1.4.5 -X subdir -DCMAKE_DIR=build/cmake
RUN cget -p $PREFIX install ccache@v4.1 -DENABLE_TESTING=OFF
RUN cget -p /opt/cmake install kitware/cmake@v3.24.3

COPY ./test/onnx/.onnxrt-commit /

ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=main
ARG ONNXRUNTIME_COMMIT

RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
    cd onnxruntime && \
    if [ -z "$ONNXRUNTIME_COMMIT" ] ; then git checkout $(cat /.onnxrt-commit) ; else git checkout ${ONNXRUNTIME_COMMIT} ; fi && \
    /bin/sh /onnxruntime/dockerfiles/scripts/install_common_deps.sh


ADD tools/build_and_test_onnxrt.sh /onnxruntime/build_and_test_onnxrt.sh

RUN cget -p /usr/local install ROCmSoftwarePlatform/rocMLIR@a997d5f51314b45d7a4c04f1599966dcf53f9b4d -DBUILD_MIXR_TARGET=On -DLLVM_ENABLE_ZSTD=Off -DLLVM_ENABLE_THREADS=Off

ENV MIOPEN_FIND_DB_PATH=/tmp/miopen/find-db
ENV MIOPEN_USER_DB_PATH=/tmp/miopen/user-db
ENV LD_LIBRARY_PATH=$PREFIX/lib

# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1
ENV ASAN_OPTIONS=detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1
RUN ln -s /opt/rocm/llvm/bin/llvm-symbolizer /usr/bin/llvm-symbolizer

