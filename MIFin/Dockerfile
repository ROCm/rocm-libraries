#Use miopen ci base image
ARG BASEIMAGE=rocm/miopen:ci_ee5a5b

#FROM ubuntu:20.04
FROM $BASEIMAGE

ARG PREFIX=/opt/rocm

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -f -y --allow-unauthenticated \
    apt-utils \
    build-essential \
    clang-format-12 \
    cmake \ 
    curl \
    doxygen \
    gdb \
    git \
    lbzip2 \
    lcov \
    libncurses5-dev \
    libnuma-dev \
    libpthread-stubs0-dev \
    mysql-client \
    openssh-server \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    rocblas \
    software-properties-common \
    sqlite3 \
    vim \
    wget \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1

# Install an init system
RUN wget https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb
RUN dpkg -i dumb-init_*.deb && rm dumb-init_*.deb

# Install cget
RUN pip install cget

# Install rclone
RUN pip install https://github.com/pfultz2/rclone/archive/master.tar.gz

# Make sure /opt/rcom is in the paths
ENV PATH="/opt/rocm:${PATH}"
# Install MIOpen
ARG MIOPEN_DIR=/root/dMIOpen
#Clone MIOpen
RUN git clone https://github.com/ROCmSoftwarePlatform/MIOpen.git $MIOPEN_DIR
WORKDIR $MIOPEN_DIR
ARG CACHE_DATE=1
ARG MIOPEN_BRANCH=develop
RUN git pull && git checkout $MIOPEN_BRANCH

ARG TUNA_USER=miopenpdb
ARG BACKEND=HIP
# Build MIOpen
WORKDIR $MIOPEN_DIR/build
ARG MIOPEN_CACHE_DIR=/tmp/${TUNA_USER}/cache
ARG MIOPEN_USER_DB_PATH=/tmp/$TUNA_USER/config/miopen
ARG MIOPEN_USE_MLIR=On
ARG MIOPEN_CMAKE_ARGS="-DMIOPEN_USE_MLIR=${MIOPEN_USE_MLIR} -DMIOPEN_INSTALL_CXX_HEADERS=On -DMIOPEN_CACHE_DIR=${MIOPEN_CACHE_DIR} -DMIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH} -DMIOPEN_BACKEND=${BACKEND} -DCMAKE_PREFIX_PATH="${MIOPEN_DEPS};${PREFIX}""

RUN echo "MIOPEN: Selected $BACKEND backend."
RUN if [ $BACKEND = "OpenCL" ]; then \
        cmake -DMIOPEN_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ ${MIOPEN_CMAKE_ARGS} $MIOPEN_DIR ; \
    else \
        CXX=/opt/rocm/llvm/bin/clang++ cmake ${MIOPEN_CMAKE_ARGS} $MIOPEN_DIR ; \
    fi

RUN make -j $(nproc)
RUN make install

# Install dependencies
ADD requirements.txt /requirements.txt
RUN CXXFLAGS='-isystem $PREFIX/include' cget -p $PREFIX install -f /requirements.txt

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -f -y --allow-unauthenticated \
    cppcheck
