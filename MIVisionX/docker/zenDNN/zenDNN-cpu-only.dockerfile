# MIVisionX & ZenDNN supported OS
FROM ubuntu:20.04

# Deps Directory
ENV MIVISIONX_DEPS_ROOT=/opt/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

# set symbolic links to sh to use bash 
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Update Docker Image
RUN apt-get update -y

# install mivisionx base dependencies - CPU Only
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ cmake pkg-config git sudo wget

# install OpenCV for test app
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy \
        libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev unzip && \
        mkdir OpenCV && cd OpenCV && wget https://github.com/opencv/opencv/archive/4.6.0.zip && unzip 4.6.0.zip && \
        mkdir build && cd build && cmake -DWITH_GTK=ON -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_OPENCL=OFF ../opencv-4.6.0 && make -j8 && sudo make install && sudo ldconfig && cd

# install ZEN DNN Deps - AOCC & AOCL
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install wget unzip python3-dev dmidecode && \
	wget https://developer.amd.com/wordpress/media/files/aocl-linux-aocc-3.0-6.tar.gz && \
	tar -xvf aocl-linux-aocc-3.0-6.tar.gz && cd aocl-linux-aocc-3.0-6/ && \
	tar -xvf aocl-blis-linux-aocc-3.0-6.tar.gz && cd ../ && \
	wget  https://developer.amd.com/wordpress/media/files/aocc-compiler-3.2.0.tar && \
	tar -xvf aocc-compiler-3.2.0.tar && cd aocc-compiler-3.2.0 && bash install.sh

# set environment variable
ENV ZENDNN_AOCC_COMP_PATH=/opt/mivisionx-deps/aocc-compiler-3.2.0
ENV ZENDNN_BLIS_PATH=/opt/mivisionx-deps/aocl-linux-aocc-3.0-6/amd-blis
ENV ZENDNN_LIBM_PATH=/usr/lib/x86_64-linux-gnu

# Install Zen DNN required Packages
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y numactl libnuma-dev hwloc
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install hwloc-nox ccache libopenblas-dev

# Working Directory
ENV MIVISIONX_WORKING_ROOT=/workspace
WORKDIR $MIVISIONX_WORKING_ROOT

# set OMP variables
RUN echo "export OMP_NUM_THREADS=$(grep -c ^processor /proc/cpuinfo)" >> ~/.profile
RUN echo "export GOMP_CPU_AFFINITY=\"0-$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')\"" >> ~/.profile

# set environment variable
ENV ZENDNN_GIT_ROOT=/workspace/ZenDNN

# install Zen DNN
RUN DEBIAN_FRONTEND=noninteractive git clone https://github.com/amd/ZenDNN.git && cd ZenDNN && make clean && \
	source scripts/zendnn_aocc_build.sh

# install MIVisionX
RUN git clone https://github.com/ROCm/MIVisionX.git && mkdir build && cd build && \
    cmake ../MIVisionX && make -j8 && make install

ENTRYPOINT source ~/.profile && /bin/bash
