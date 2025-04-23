FROM ubuntu:20.04

ENV MIVISIONX_DEPS_ROOT=/opt/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

RUN apt-get update -y
# install mivisionx base dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ cmake pkg-config git
# install ROCm for mivisionx OpenCL/HIP dependency
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install initramfs-tools libnuma-dev wget sudo keyboard-configuration &&  \
        wget https://repo.radeon.com/amdgpu-install/6.1.1/ubuntu/focal/amdgpu-install_6.1.60101-1_all.deb && \
        sudo apt-get install -y ./amdgpu-install_6.1.60101-1_all.deb && \
        sudo apt-get update -y && \
        sudo amdgpu-install -y --usecase=rocm

WORKDIR /workspace

ENV OPENVX_INC=/workspace/MIVisionX/amd_openvx/openvx
ENV OPENVX_DIR_OPENCL=/workspace/build-opencl
ENV OPENVX_DIR_HIP=/workspace/build-hip
ENV VX_TEST_DATA_PATH=/workspace/conformance_tests/OpenVX-cts/test_data/

# install MIVisionX OpenCL
RUN git clone https://github.com/ROCm/MIVisionX.git && \
        python MIVisionX/docker/conformance/system_info.py && \
        mkdir build-opencl && cd build-opencl && cmake ../MIVisionX && make -j8
RUN mkdir conformance_tests && cd conformance_tests && git clone -b openvx_1.3 https://github.com/KhronosGroup/OpenVX-cts.git && \
        mkdir build-cts-opencl && cd build-cts-opencl && \
        cmake -DOPENVX_INCLUDES=$OPENVX_INC/include -DOPENVX_LIBRARIES=$OPENVX_DIR_OPENCL/lib/libopenvx.so\;$OPENVX_DIR_OPENCL/lib/libvxu.so\;pthread\;dl\;m\;rt -DOPENVX_CONFORMANCE_VISION=ON ../OpenVX-cts && \
        cmake --build .
RUN cd conformance_tests/build-cts-opencl && AGO_DEFAULT_TARGET=CPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-CPU-CTS-OCL-centos7.md && \
        AGO_DEFAULT_TARGET=GPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-GPU-CTS-OCL-centos7.md && \
        mv *.md /workspace/
# install MIVisionX HIP
RUN mkdir build-hip && cd build-hip && cmake ../MIVisionX -DBACKEND=HIP && make -j8
RUN cd conformance_tests && mkdir build-cts-hip && cd build-cts-hip && \
        cmake -DOPENVX_INCLUDES=$OPENVX_INC/include -DOPENVX_LIBRARIES=$OPENVX_DIR_HIP/lib/libopenvx.so\;$OPENVX_DIR_HIP/lib/libvxu.so\;/opt/rocm/hip/lib/libamdhip64.so\;pthread\;dl\;m\;rt -DOPENVX_CONFORMANCE_VISION=ON ../OpenVX-cts && \
        cmake --build .
RUN cd conformance_tests/build-cts-hip && AGO_DEFAULT_TARGET=CPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-CPU-CTS-HIP-centos7.md && \
        AGO_DEFAULT_TARGET=GPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-GPU-CTS-HIP-centos7.md && \
        mv *.md /workspace/