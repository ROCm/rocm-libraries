# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# ============================================================================
# Common base image with all dependencies
# ============================================================================
FROM ubuntu:24.04 AS base

ARG PROJ="miopen"
ARG CCACHE_SECONDARY_STORAGE=""
ARG CCACHE_DIR="/tmp"
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
    clang-format-18 \
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
RUN apt-get remove --purge -y python3-pyparsing python3-cryptography 

# Add requirements files
ADD ${PROJ}/docs/sphinx/requirements.txt /doc-requirements.txt

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

# ============================================================================
# Build from source stage - clones and builds TheRock
#
# Example build command for default source build (default branch, gfx90a):
# docker build -f Dockerfile.ubuntu24 -t miopen:fullbuild .
#
# Example build command for overriding ASIC and git hash:
# docker build -f Dockerfile.ubuntu24 --build-arg THEROCK_ASIC=gfx950 --build-arg THEROCK_GIT_HASH=abc123def456 -t miopen:custom_source_build .
#
# Build mode options:
# - THEROCK_BUILD_MODE can be: Preset, Debug, or Release (default)
# - When using "Preset", specify the preset with THEROCK_BUILD_PRESET
# - When using "Debug" or "Release", standard CMake build types are used
#
# Note: When THEROCK_BUILD_MODE=Preset, the THEROCK_BUILD_PRESET value is used.
#       When THEROCK_BUILD_MODE=Debug/Release, THEROCK_BUILD_PRESET is ignored.
#
# Build parallelism:
# - BUILD_JOBS controls the number of parallel build jobs
# - Default is 0, which means use all available CPU cores
#
# Example build command for Debug build with 4 cores:
# docker build -f Dockerfile.ubuntu24 --build-arg THEROCK_BUILD_MODE=Debug --build-arg BUILD_JOBS=4 -t miopen:debug .
#
# Example build command with custom TheRock preset:
# docker build -f Dockerfile.ubuntu24 --build-arg THEROCK_BUILD_MODE=Preset --build-arg THEROCK_BUILD_PRESET=linux-debug-package -t miopen:custom-preset .
# ============================================================================
FROM base AS build_therock_and_miopen_deps

ARG THEROCK_ASIC=gfx90a
ARG THEROCK_GIT_HASH=default
ARG THEROCK_BUILD_MODE=Release
ARG THEROCK_BUILD_PRESET=linux-release-package
ARG BUILD_JOBS=0
ARG ROCM_LIBRARIES_REF=default
ARG ROCM_SYSTEMS_REF=default

ARG PROJ="miopen"
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

RUN apt-get update && apt-get install -y --allow-unauthenticated ninja-build
RUN if [ "$THEROCK_BUILD_MODE" = "Preset" ]; then \
        echo "Building with TheRock preset: $THEROCK_BUILD_PRESET"; \
        cmake .. \
            --preset=$THEROCK_BUILD_PRESET \
            -DCMAKE_INSTALL_PREFIX=/opt/rocm \
            -DTHEROCK_ENABLE_ALL=OFF \
            -DTHEROCK_ENABLE_BLAS=ON \
            -DTHEROCK_ENABLE_COMPILER=ON \
            -DTHEROCK_ENABLE_OCL_RUNTIME=ON \
            -DTHEROCK_ENABLE_CORE_RUNTIME=ON \
            -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
            -DTHEROCK_ENABLE_RAND=ON \
            -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=multiarch \
            -DTHEROCK_AMDGPU_FAMILIES=$THEROCK_ASIC; \
    else \
        echo "Building with CMake build type: $THEROCK_BUILD_MODE"; \
        cmake .. \
            -GNinja \
            -DCMAKE_BUILD_TYPE=$THEROCK_BUILD_MODE \
            -DCMAKE_INSTALL_PREFIX=/opt/rocm \
            -DTHEROCK_ENABLE_ALL=OFF \
            -DTHEROCK_ENABLE_BLAS=ON \
            -DTHEROCK_ENABLE_COMPILER=ON \
            -DTHEROCK_ENABLE_OCL_RUNTIME=ON \
            -DTHEROCK_ENABLE_CORE_RUNTIME=ON \
            -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
            -DTHEROCK_ENABLE_RAND=ON \
            -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=multiarch \
            -DTHEROCK_AMDGPU_FAMILIES=$THEROCK_ASIC; \
    fi

RUN if [ "$BUILD_JOBS" = "0" ]; then \
        cmake --build .; \
    else \
        echo "Building with $BUILD_JOBS parallel jobs"; \
        cmake --build . -j$BUILD_JOBS; \
    fi && \
    cmake --install .

# ============================================================================
# Build and install miopen dependencies via rbuild
# ============================================================================
# Add requirements files
ADD ${PROJ}/rbuild.ini /rbuild.ini
ADD ${PROJ}/requirements.txt /requirements.txt
ADD ${PROJ}/dev-requirements.txt /dev-requirements.txt
# rbuild is used to trigger build of requirements.txt, dev-requirements.txt.
# Composable Kernel install separated from rbuild 
# Additionally remove components that are already installed by other compoenents as a part of the source build.
RUN cd / && \
    sed -i '/composablekernel/d; /half/d; /nlohmann/d' /requirements.txt && \
    if [ "$USE_FIN" = "ON" ]; then \
        rbuild prepare -s fin -d ${PREFIX} -DGPU_ARCHS=${THEROCK_ASIC}; \
    else \
        rbuild prepare -s develop -d ${PREFIX} -DGPU_ARCHS=${THEROCK_ASIC}; \
    fi && \
    ccache -s

# ============================================================================
# Build and install CK
# ============================================================================
# Make sure /opt/rocm is in the paths
ENV PATH="/opt/rocm:${PATH}"

ADD ${PROJ}/script/redis-cli.conf /redis-cli.conf
ADD ${PROJ}/script/sccache_wrapper.sh /sccache_wrapper.sh

## Sccache binary built from source for ROCm, only install if MIOPEN_SCCACHE is defined
ARG SCCACHE_REPO_URL=http://compute-artifactory.amd.com/artifactory/rocm-generic-experimental/rocm-sccache
ENV SCCACHE_INSTALL_LOCATION=/usr/local/.cargo/bin
ENV PATH=$PATH:${SCCACHE_INSTALL_LOCATION}
ENV MIOPEN_SCCACHE=$MIOPEN_SCCACHE

# Copy Composable Kernel
COPY composablekernel /composable_kernel

# Build and install Composable Kernel
RUN echo Building for GPU Archs: ${THEROCK_ASIC} && \
    if [ "$MIOPEN_SCCACHE" != "" ]; then \
    mkdir -p ${SCCACHE_INSTALL_LOCATION} && \
    curl ${SCCACHE_REPO_URL}/portable/0.2.16/sccache-0.2.16-alpha.1-rocm --output ${SCCACHE_INSTALL_LOCATION}/sccache && \
    chmod +x ${SCCACHE_INSTALL_LOCATION}/sccache; \
    export ROCM_PATH=/opt/rocm && \
    export SCCACHE_ENABLED=true && \
    export SCCACHE_LOG_LEVEL=debug && \
    export SCCACHE_IDLE_TIMEOUT=14400 && \
    export COMPILERS_HASH_DIR=/tmp/.sccache && \
    export SCCACHE_BIN=/usr/local/.cargo/bin/sccache && \
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
    cd /composable_kernel && \
    mkdir build && cd build && \
    #########################################################################
    ## Determine number of build threads based on CPU cores and RAM
    #########################################################################
    cpu_threads="$(nproc)" && \
    mem_limit_bytes="" && \
    if [ -r /sys/fs/cgroup/memory.max ]; then \
        mem_limit_bytes="$(cat /sys/fs/cgroup/memory.max)"; \
    fi && \
    if [ -n "$mem_limit_bytes" ] && [ "$mem_limit_bytes" != "max" ]; then \
        mem_bytes="$mem_limit_bytes"; \
    else \
        mem_kb="$(awk '/MemTotal/ {print $2}' /proc/meminfo)" && \
        mem_bytes="$((mem_kb * 1024))"; \
    fi && \
    mem_gb="$((mem_bytes / 1024 / 1024 / 1024))" && \
    if [ "$mem_gb" -lt 200 ]; then \
        ram_cap=64; \
    elif [ "$mem_gb" -ge 1024 ]; then \
        ram_cap=512; \
    else \
        ram_cap="$((64 + ((mem_gb - 200) * 192) / 824))"; \
        [ "$ram_cap" -lt 64 ] && ram_cap=64; \
        [ "$ram_cap" -gt 256 ] && ram_cap=256; \
    fi && \
    num_threads="$cpu_threads" && \
    [ "$num_threads" -gt "$ram_cap" ] && num_threads="$ram_cap" && \
    [ "$num_threads" -lt 1 ] && num_threads=1 && \
    echo "Detected mem=${mem_gb}GB, nproc=${cpu_threads}, Building CK with -j${num_threads}" && \
    #########################################################################
    CXX=/opt/rocm/llvm/bin/amdclang++ cmake \
    -D CMAKE_PREFIX_PATH=/opt/rocm \
    -D CMAKE_CXX_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}" \
    -D CMAKE_C_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D GPU_ARCHS=${THEROCK_ASIC} \
    -D MIOPEN_REQ_LIBS_ONLY=ON \
    -D CMAKE_CXX_FLAGS=" -O3 " .. && \
    make -j ${num_threads} install && \
    if [ "$MIOPEN_SCCACHE" != "" ]; then \
    sccache -s; \
    fi

# ============================================================================
# Common miopen development environment
# ============================================================================
FROM base AS miopen

COPY --from=build_therock_and_miopen_deps /opt/rocm /opt/rocm
COPY --from=build_therock_and_miopen_deps /usr/local /usr/local

ENV PATH="/opt/rocm/bin:/opt/rocm:${PATH}"
