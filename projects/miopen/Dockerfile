# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

ARG BUILD_TYPE=prebuilt
ARG THEROCK_BASE_IMAGE=rocm/miopen:therock

# ============================================================================
# Common base image with all dependencies
# ============================================================================
FROM ubuntu:24.04 AS base

ARG CCACHE_COMMIT=8c107aada09df2a40930d7064bbec8120ee76c91

ENV DEBIAN_FRONTEND=noninteractive
ENV HIP_PLATFORM=amd
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn

# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install dependencies
RUN apt-get update && apt-get install -y --allow-unauthenticated \
    apt-utils \
    automake \
    bison \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    doxygen \
    flex \
    g++ \
    gdb \
    gfortran \
    git \
    git-lfs \
    gnupg2 \
    lbzip2 \
    lcov \
    libegl1-mesa-dev \
    libncurses5-dev \
    libnuma-dev \
    libtool \
    ninja-build \
    patchelf \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-venv \
    redis \
    rpm \
    rsync \
    software-properties-common \
    stunnel \
    sudo \
    texinfo \
    unzip \
    vim \
    wget \
    xxd 
    
# Install DVC
RUN mkdir -p /etc/apt/keyrings && \
    wget -qO - https://dvc.org/deb/iterative.asc | sudo gpg --dearmor -o /etc/apt/keyrings/packages.iterative.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/packages.iterative.gpg] https://dvc.org/deb/ stable main" | sudo tee /etc/apt/sources.list.d/dvc.list && \
    chmod 644 /etc/apt/keyrings/packages.iterative.gpg /etc/apt/sources.list.d/dvc.list && \
    apt-get update && apt-get install -y dvc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

# Install an init system
RUN wget -O /tmp/ccache.tar.gz https://github.com/ccache/ccache/archive/${CCACHE_COMMIT}.tar.gz && \
    tar zxvf /tmp/ccache.tar.gz -C /tmp/ && mkdir /tmp/ccache-${CCACHE_COMMIT}/build && \
    cd /tmp/ccache-${CCACHE_COMMIT}/build && \
    cmake -DZSTD_FROM_INTERNET=ON -DHIREDIS_FROM_INTERNET=ON .. && make -j install && rm -rf /tmp/* && \
    ccache -s && \
    cd / 

# Purge packages conflicting with doc-requirements
RUN apt-get remove --purge -y python3-pyparsing python3-cryptography python3-jwt

# Add requirements files
ADD miopen/docs/sphinx/requirements.txt /doc-requirements.txt

# Install doc requirements
RUN pip3 install --break-system-packages -r /doc-requirements.txt && \
# Install an init system
    wget https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && rm dumb-init_*.deb && \
# Install cget && rbuild
    pip3 install --break-system-packages https://github.com/pfultz2/cget/archive/b5dd36cafa2804e6146c5021eb2469c4b2d3ad42.tar.gz && \
    pip3 install --break-system-packages https://github.com/RadeonOpenCompute/rbuild/archive/c33b1a7e74ae7f72379364b523f73d30fd4d765c.tar.gz && \
# groupadd render && video
    groupadd -f render && \
    groupadd -f video && \
    usermod -a -G render,video root

# ========================================================================================
# NIGHTLY: Build from source stage - clones and builds projects from TheRock that MIOpen
# depends on, with the exception of CK and external MIOpen deps which are built separately
# ========================================================================================
FROM base AS build_therock

ARG THEROCK_ASIC=gfx90a
ARG THEROCK_GIT_HASH=default
ARG THEROCK_BUILD_MODE=Release
ARG THEROCK_BUILD_PRESET=linux-release-package
ARG BUILD_JOBS=0
ARG ROCM_LIBRARIES_REF=default
ARG ROCM_SYSTEMS_REF=default

ARG USE_FIN="OFF"
ARG PREFIX=/usr/local

RUN git config --global user.email "docker@build.local" && \
    git config --global user.name "Docker Build"

RUN git clone https://github.com/ROCm/TheRock.git

WORKDIR /TheRock

RUN if [ "$THEROCK_GIT_HASH" != "default" ]; then \
        echo "Checking out specific hash: $THEROCK_GIT_HASH"; \
        git fetch; \
        git checkout $THEROCK_GIT_HASH; \
    else \
        echo "Using default branch"; \
    fi

# Update rocm-libraries submodule to specific ref if provided
# we need to commit the change so that fetch_sources.py does not overwrite the change
RUN if [ "$ROCM_LIBRARIES_REF" != "default" ]; then \
        echo "Updating rocm-libraries submodule to: $ROCM_LIBRARIES_REF"; \
        git submodule update --init rocm-libraries && \
        cd rocm-libraries && \
        git fetch origin && \
        git checkout $ROCM_LIBRARIES_REF && \
        cd .. && \
        git add rocm-libraries && \
        git commit -m "Update rocm-libraries submodule to $ROCM_LIBRARIES_REF"; \
    fi

# Update rocm-systems submodule to specific ref if provided
# we need to commit the change so that fetch_sources.py does not overwrite the change
RUN if [ "$ROCM_SYSTEMS_REF" != "default" ]; then \
        echo "Updating rocm-systems submodule to: $ROCM_SYSTEMS_REF"; \
        git submodule update --init rocm-systems && \
        cd rocm-systems && \
        git fetch origin && \
        git checkout $ROCM_SYSTEMS_REF && \
        cd .. && \
        git add rocm-systems && \
        git commit -m "Update rocm-systems submodule to $ROCM_SYSTEMS_REF"; \
    fi

RUN python3 -m venv /opt/venv

# Adding the venv into the path activates the python virtual env in a way.  We cant source it the
# normal way due to how dockerfiles run
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install -r requirements.txt

RUN git config --global url."https://github.com/".insteadOf git@github.com: && \
    git config --global credential.helper "!f() { echo username=anonymous; echo password=; }; f" && \
    python ./build_tools/fetch_sources.py

RUN mkdir build
WORKDIR /TheRock/build

RUN if [ "$THEROCK_BUILD_MODE" = "Preset" ]; then \
        echo "Building with TheRock preset: $THEROCK_BUILD_PRESET"; \
        cmake .. \
            --preset=$THEROCK_BUILD_PRESET \
            -DCMAKE_INSTALL_PREFIX=/opt/rocm \
            -DTHEROCK_ENABLE_ALL=OFF \
            -DTHEROCK_ENABLE_PROFILER=ON \
            -DTHEROCK_ENABLE_SYSDEPS=ON \
            -DTHEROCK_ENABLE_BASE=ON \
            -DTHEROCK_ENABLE_COMPILER=ON \
            -DTHEROCK_ENABLE_CORE_RUNTIME=ON \
            -DTHEROCK_ENABLE_HOST_BLAS=ON \
            -DTHEROCK_ENABLE_HOST_SUITE_SPARSE=ON \
            -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
            -DTHEROCK_ENABLE_OCL_RUNTIME=ON \
            -DTHEROCK_ENABLE_ROCPROFV3=ON \
            -DTHEROCK_ENABLE_ROCPROFILER_COMPUTE=ON \
            -DTHEROCK_ENABLE_BLAS=ON \
            -DTHEROCK_ENABLE_RAND=ON \
            -DTHEROCK_ENABLE_ROCPROFSYS=ON \
            -DTHEROCK_ENABLE_ROCPROF_TRACE_DECODER_BINARY=ON \
            -DTHEROCK_ENABLE_ROCGDB=ON \
            -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=multiarch \
            -DTHEROCK_AMDGPU_FAMILIES=$THEROCK_ASIC; \
    else \
        echo "Building with CMake build type: $THEROCK_BUILD_MODE"; \
        cmake .. \
            -GNinja \
            -DCMAKE_BUILD_TYPE=$THEROCK_BUILD_MODE \
            -DCMAKE_INSTALL_PREFIX=/opt/rocm \
            -DTHEROCK_ENABLE_ALL=OFF \
            -DTHEROCK_ENABLE_PROFILER=ON \
            -DTHEROCK_ENABLE_SYSDEPS=ON \
            -DTHEROCK_ENABLE_BASE=ON \
            -DTHEROCK_ENABLE_COMPILER=ON \
            -DTHEROCK_ENABLE_CORE_RUNTIME=ON \
            -DTHEROCK_ENABLE_HOST_BLAS=ON \
            -DTHEROCK_ENABLE_HOST_SUITE_SPARSE=ON \
            -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
            -DTHEROCK_ENABLE_OCL_RUNTIME=ON \
            -DTHEROCK_ENABLE_ROCPROFV3=ON \
            -DTHEROCK_ENABLE_ROCPROFILER_COMPUTE=ON \
            -DTHEROCK_ENABLE_BLAS=ON \
            -DTHEROCK_ENABLE_RAND=ON \
            -DTHEROCK_ENABLE_ROCPROFSYS=ON \
            -DTHEROCK_ENABLE_ROCPROF_TRACE_DECODER_BINARY=ON \
            -DTHEROCK_ENABLE_ROCGDB=ON \
            -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=multiarch \
            -DTHEROCK_BUILD_TESTING=OFF \
            -DTHEROCK_AMDGPU_FAMILIES=$THEROCK_ASIC; \
    fi

RUN if [ "$BUILD_JOBS" = "0" ]; then \
        cmake --build .; \
    else \
        echo "Building with $BUILD_JOBS parallel jobs"; \
        cmake --build . -j$BUILD_JOBS; \
    fi && \
    cmake --install .

#============================================================================
# REGULAR: Use the pre-built ROCm image from TheRock to build the rest of
# the MIOpen environment
#============================================================================
FROM ${THEROCK_BASE_IMAGE} AS prebuilt_therock

# ============================================================================
# Select which build stage to use
# ============================================================================
FROM ${BUILD_TYPE}_therock AS therock

# ================================================================================
# REGULAR: Build and install other 3rd party deps using pre-built ROCm image
# ================================================================================
FROM base AS miopen_deps
COPY --from=therock /opt/rocm /opt/rocm
ARG THEROCK_ASIC=gfx90a
ARG PREFIX="/usr/local"
ARG USE_FIN="OFF"

# Add requirements files
ADD miopen/rbuild.ini /rbuild.ini
ADD miopen/requirements.txt /requirements.txt
ADD miopen/dev-requirements.txt /dev-requirements.txt

# rbuild is used to trigger build of requirements.txt, dev-requirements.txt.
# Composable Kernel install separated from rbuild 
# Remove components that are already built by TheRock.
RUN cd / && \
    sed -i '/composablekernel/d; /half/d; /nlohmann/d' /requirements.txt && \
    if [ "$USE_FIN" = "ON" ]; then \
        rbuild prepare -s fin -d ${PREFIX} -DGPU_ARCHS="${THEROCK_ASIC}"; \
    else \
        rbuild prepare -s develop -d ${PREFIX} -DGPU_ARCHS="${THEROCK_ASIC}"; \
    fi
# ================================================================================
# NIGHTLY target: push this image as rocm/miopen:therock
# ================================================================================
FROM scratch AS update_therock
COPY --from=miopen_deps /opt/rocm /opt/rocm
COPY --from=miopen_deps /usr/local /usr/local

# ============================================================================
# Build and install CK
# ============================================================================
FROM miopen_deps AS build_ck
ARG THEROCK_ASIC=gfx90a
ADD miopen/script/redis-cli.conf /redis-cli.conf
ADD miopen/script/sccache_wrapper.sh /sccache_wrapper.sh

## Sccache binary built from source for ROCm, only install if MIOPEN_SCCACHE is defined
ENV SCCACHE_VERSION="0.14.0"
ENV SCCACHE_INSTALL_LOCATION=/usr/local/.cargo/bin
ENV PATH=$PATH:${SCCACHE_INSTALL_LOCATION}
ARG MIOPEN_SCCACHE=""
ENV MIOPEN_SCCACHE=$MIOPEN_SCCACHE
ARG COMPILER_LAUNCHER=""
ARG MIOPEN_SCCACHE_CUSTOM_CACHE_BUSTER="MiOpen-Docker-CK"
ARG THEROCK_ASIC=""
ENV THEROCK_ASIC=${THEROCK_ASIC}

# Copy Composable Kernel
COPY /composablekernel /composable_kernel

# Build and install Composable Kernel
RUN set -x && \
    echo Building for GPU Archs: ${THEROCK_ASIC} && \
    if [ "$MIOPEN_SCCACHE" != "" ]; then \
        mkdir -p ${SCCACHE_INSTALL_LOCATION} && \
        wget -qO sccache.tar.gz https://github.com/mozilla/sccache/releases/latest/download/sccache-v$SCCACHE_VERSION-x86_64-unknown-linux-musl.tar.gz && \
        tar -xzf sccache.tar.gz --strip-components=1 -C ${SCCACHE_INSTALL_LOCATION} && \
        chmod +x ${SCCACHE_INSTALL_LOCATION}/sccache && \
        export ROCM_PATH=/opt/rocm && \
        export SCCACHE_ENABLED=true && \
        export SCCACHE_LOG_LEVEL=debug && \
        export SCCACHE_IDLE_TIMEOUT=14400 && \
        export COMPILERS_HASH_DIR=/tmp/.sccache && \
        export SCCACHE_BIN=${SCCACHE_INSTALL_LOCATION}/sccache && \
        export SCCACHE_EXTRAFILES=/tmp/.sccache/rocm_compilers_hash_file && \
        export SCCACHE_REDIS="redis://$MIOPEN_SCCACHE" && \
        cd / && \
        echo "connect = $MIOPEN_SCCACHE" >> redis-cli.conf && \
        export SCCACHE_C_CUSTOM_CACHE_BUSTER="${MIOPEN_SCCACHE_CUSTOM_CACHE_BUSTER}" && \
        echo $SCCACHE_C_CUSTOM_CACHE_BUSTER && \
        stunnel redis-cli.conf && \
        export PATH=$PATH:${SCCACHE_INSTALL_LOCATION} && \
        ./sccache_wrapper.sh --enforce_redis; \
    fi &&\
    # Clear the ROCM_PATH TheRock causes HIP errors
    unset ROCM_PATH && \
    cd /composable_kernel && \
    mkdir build && cd build && \
    #Get the number of threads and RAM available
    num_threads=$(nproc) && \
    ram_gb=$(free -g | awk '/^Mem:/{print $7}') && \
    # CK's ram req per thread is ~4 GB,
    # so limit the number of threads accordingly
    if [ "$num_threads" -le 0 ]; then \
        num_threads=1; \
    fi && \
    if [ "$ram_gb" -lt $((num_threads * 4)) ]; then \
        if [ "$ram_gb" -ge 4 ]; then \
            num_threads=$((ram_gb / 4)); \
        else \
            num_threads=1; \
        fi; \
    fi && \
    echo Building CK with ${num_threads} threads && \
    CXX=/opt/rocm/llvm/bin/amdclang++ cmake \
    -G Ninja \
    -D CMAKE_PREFIX_PATH=/opt/rocm \
    -D CMAKE_CXX_COMPILER_LAUNCHER=${COMPILER_LAUNCHER} \
    -D CMAKE_HIP_COMPILER_LAUNCHER=${COMPILER_LAUNCHER} \
    -D CMAKE_C_COMPILER_LAUNCHER=${COMPILER_LAUNCHER} \
    -D CMAKE_BUILD_TYPE=Release \
    -D GPU_TARGETS="${THEROCK_ASIC}" \
    -D MIOPEN_REQ_LIBS_ONLY=ON \
    -D CMAKE_CXX_FLAGS=" -O3 " .. && \    
    ninja -j ${num_threads} install && \
    if [ "$MIOPEN_SCCACHE" != "" ]; then \
        sccache -s; \
    fi

# ============================================================================
# REGULAR target: final MIOpen development environment
# ============================================================================
FROM base AS miopen

COPY --from=build_ck /opt/rocm /opt/rocm
COPY --from=build_ck /usr/local /usr/local

# Install explicit version of tools that are supported for miopen development
# (clang-format, clang-tidy)
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add - && \
    echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-21 main" | sudo tee /etc/apt/sources.list.d/llvm21.list 

RUN apt-get update && apt-get install -y \
    clang-format-18 \
    clang-tidy-21 \
    && rm -rf /var/lib/apt/lists/*

# Workaround for broken clang-tidy in TheRock
RUN mv /opt/rocm/llvm/bin/clang-tidy /opt/rocm/llvm/bin/clang-tidy.disabled || true
RUN ln -sf /usr/bin/clang-tidy-21 /opt/rocm/llvm/bin/clang-tidy

# Use update-alternatives to set default versions
RUN update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-18 100
RUN update-alternatives --install /opt/rocm/llvm/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-21 100
RUN update-alternatives --set clang-tidy /usr/bin/clang-tidy-21

# Ensure the dynamic linker can find ROCm libraries by adding /opt/rocm/lib to the linker cache. 
RUN printf '%s\n' /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf && ldconfig

ENV HIP_PLATFORM=amd
ENV PATH="/opt/rocm/bin:/opt/rocm:${PATH}"
ENV ROCM_PATH=/opt/rocm

WORKDIR /workspace
