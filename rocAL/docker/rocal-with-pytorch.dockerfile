ARG PYTORCH_VERSION=latest
ARG ROCAL_PYTHON_VERSION_SUGGESTED=3.9
FROM rocm/pytorch:${PYTORCH_VERSION}

ENV ROCAL_DEPS_ROOT=/rocAL-deps
WORKDIR $ROCAL_DEPS_ROOT

RUN apt-get update -y

# install rocAL base dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ cmake pkg-config git

# install OpenCV
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev \
        libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev unzip && \
        mkdir OpenCV && cd OpenCV && wget https://github.com/opencv/opencv/archive/4.6.0.zip && unzip 4.6.0.zip && \
        mkdir build && cd build && cmake -DWITH_GTK=ON -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_OPENCL=OFF ../opencv-4.6.0 && make -j8 && sudo make install && sudo ldconfig && cd

# install FFMPEG
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev

# install rocAL neural net dependency
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install half rocblas-dev miopen-hip-dev migraphx-dev

# install rocAL dependency
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install rpp-dev wget libbz2-dev libssl-dev python3-dev libgflags-dev libgoogle-glog-dev liblmdb-dev nasm yasm libjsoncpp-dev clang && \
        git clone -b 3.0.2 https://github.com/libjpeg-turbo/libjpeg-turbo.git && cd libjpeg-turbo && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib -DWITH_JPEG8=TRUE ../ && \
        make -j4 && sudo make install && cd ../../ && \
        git clone -b v3.21.9 https://github.com/protocolbuffers/protobuf.git && cd protobuf && git submodule update --init --recursive && \
        ./autogen.sh && ./configure && make -j8 && make check -j8 && sudo make install && sudo ldconfig && cd ../

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install python3 python3-pip git g++ hipblas hipsparse rocrand hipfft rocfft rocthrust-dev hipcub-dev && \
        git clone https://github.com/Tencent/rapidjson.git && cd rapidjson && mkdir build && cd build && \
        cmake ../ && make -j4 && sudo make install && cd ../../ && \
        pip install pytest==7.3.1 && git clone -b v2.11.1 https://github.com/pybind/pybind11 && cd pybind11 && mkdir build && cd build && \
        cmake -DDOWNLOAD_CATCH=ON -DDOWNLOAD_EIGEN=ON ../ && make -j4 && sudo make install && cd ../../ && \
        git clone -b v1.2.20 https://repo.or.cz/libtar.git && cd libtar && \
        autoreconf --force --install && CFLAGS="-fPIC" ./configure && make -j$(nproc) && sudo make install && cd ../../ 

# install MIVisionX 
RUN git clone https://github.com/ROCm/MIVisionX.git && cd MIVisionX && \
        mkdir build && cd build && cmake -DBACKEND=HIP ../ && make -j8 && make install && cd

# install rocDecode
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install rocdecode-dev

ENV ROCAL_WORKSPACE=/workspace
WORKDIR $ROCAL_WORKSPACE

# Install rocAL
RUN pip install --upgrade pip
RUN git clone -b develop https://github.com/ROCm/rocAL && \
        mkdir build && cd build && cmake -D PYTHON_VERSION_SUGGESTED=${ROCAL_PYTHON_VERSION_SUGGESTED} ../rocAL && make -j8 && cmake --build . --target PyPackageInstall && make install
RUN export PYTHONPATH="/opt/rocm/lib:$PYTHONPATH"