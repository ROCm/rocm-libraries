# MIVisionX docker

Docker is a set of platform as a service (PaaS) products that use OS-level virtualization to deliver software in packages called containers. [Read More](https://github.com/ROCm/MIVisionX/wiki/Docker)

### Docker workflow on Ubuntu 20.04/22.04

#### Prerequisites
* Ubuntu `20.04`/`22.04`
* [ROCm supported hardware](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html)
* Install [ROCm](https://rocmdocs.amd.com/en/latest/deploy/linux/installer/install.html) with `--usecase=rocm`
* [Docker](https://docs.docker.com/engine/install/ubuntu/)

#### Workflow

* **Step 1** - Get latest docker image
  ```shell
  sudo docker pull mivisionx/ubuntu-20.04:latest
  ```
  * **NOTE:** Use the above command to bring in latest changes from upstream

* **Step 2** - Run docker image

### Run docker image: Local Machine
```shell
sudo docker run -it --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mem --cap-add=SYS_RAWIO  --group-add video --shm-size=4g --ipc="host" --network=host mivisionx/ubuntu-20.04:latest
```
* **Test** - Computer Vision Workflow
  ```shell
  python3 /workspace/MIVisionX/tests/vision_tests/runVisionTests.py --num_frames 1
  ```
* **Test** - Neural Network Workflow
  ```shell
  python3 /workspace/MIVisionX/tests/neural_network_tests/runNeuralNetworkTests.py --profiler_level 1
  ```
* **Test** - Khronos OpenVX 1.3.0 Conformance Test
  ```shell
  python3 /workspace/MIVisionX/tests/conformance_tests/runConformanceTests.py --backend_type HOST
  ```

#### Option 1: Map localhost directory on the docker image
* option to map the localhost directory with data to be accessed on the docker image
* **usage**: -v {LOCAL_HOST_DIRECTORY_PATH}:{DOCKER_DIRECTORY_PATH} 
  ```shell
  sudo docker run -it -v /home/:/root/hostDrive/ --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mem --cap-add=SYS_RAWIO  --group-add video --shm-size=4g --ipc="host" --network=host mivisionx/ubuntu-20.04:latest
  ```
#### Option 2:  Display with docker
* Using host display for docker

  ```shell
  xhost +local:root
  sudo docker run -it --privileged --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host --env DISPLAY=$DISPLAY --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --volume /tmp/.X11-unix/:/tmp/.X11-unix mivisionx/ubuntu-20.04:latest
  ```
* **Test** display with MIVisionX sample
  ```shell
  runvx -v /opt/rocm/share/mivisionx/samples/gdf/canny.gdf
  ```
### Run docker image with display: Remote Server Machine

  ```shell
  sudo docker run -it --privileged --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host --env DISPLAY=$DISPLAY --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --volume /tmp/.X11-unix/:/tmp/.X11-unix mivisionx/ubuntu-20.04:latest
  ```
* **Test** display with MIVisionX sample
  ```shell
  runvx -v /opt/rocm/share/mivisionx/samples/gdf/canny.gdf
  ```

## Build - dockerfiles

```
sudo docker build --build-arg {ARG_1_NAME}={ARG_1_VALUE} [--build-arg {ARG_2_NAME}={ARG_2_VALUE}] -f {DOCKER_FILE_NAME}.dockerfile -t {DOCKER_IMAGE_NAME} .
```

## Run - docker

```
sudo docker run -it --privileged --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host --env DISPLAY=$DISPLAY --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --volume /tmp/.X11-unix/:/tmp/.X11-unix {DOCKER_IMAGE_NAME}
```

## Ubuntu `20`/`22` DockerFiles

- ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `new component added to the level`
- ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `existing component from the previous level`

| Build Level | MIVisionX Dependencies                             | Modules                                                                  | Libraries and Executables                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Docker File                                                                                                                                                                                                     |
|-------------|----------------------------------------------------|--------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Level_1`   | cmake <br> gcc <br> g++                            | amd_openvx  <br> utilities                                                              | ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libopenvx.so` - OpenVX&trade; Lib - CPU <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libvxu.so` - OpenVX&trade; immediate node Lib - CPU <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `runvx` - OpenVX&trade; Graph Executor - CPU with Display OFF                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | level-1.dockerfile |
| `Level_2`   | ROCm OpenCL <br> +Level 1                          | amd_openvx <br> amd_openvx_extensions <br> utilities                     | ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libopenvx.so`  - OpenVX&trade; Lib - CPU/GPU <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libvxu.so` - OpenVX&trade; immediate node Lib - CPU/GPU <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libvx_loomsl.so` - Loom 360 Stitch Lib <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `loom_shell` - 360 Stitch App <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `runcl` - OpenCL&trade; program debug App <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `runvx` - OpenVX&trade; Graph Executor - Display OFF                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | level-2.dockerfile |
| `Level_3`   | OpenCV <br> FFMPEG <br> +Level 2                   | amd_openvx <br> amd_openvx_extensions <br> utilities                     | ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libopenvx.so`  - OpenVX&trade; Lib <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvxu.so` - OpenVX&trade; immediate node Lib <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvx_loomsl.so` - Loom 360 Stitch Lib <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `loom_shell` - 360 Stitch App <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `runcl` - OpenCL&trade; program debug App <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libvx_amd_media.so` - OpenVX&trade; Media Extension <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libvx_opencv.so` - OpenVX&trade; OpenCV InterOp Extension <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `mv_compile` - Neural Net Model Compile <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `runvx` - OpenVX&trade; Graph Executor - Display ON                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | level-3.dockerfile |
| `Level_4`   | MIOpenGEMM <br> MIOpen <br> ProtoBuf <br> +Level 3 | amd_openvx <br>  amd_openvx_extensions <br> apps <br> utilities          | ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libopenvx.so`  - OpenVX&trade; Lib <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvxu.so` - OpenVX&trade; immediate node Lib <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvx_loomsl.so` - Loom 360 Stitch Lib <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `loom_shell` - 360 Stitch App <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvx_amd_media.so` - OpenVX&trade; Media Extension <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvx_opencv.so` - OpenVX&trade; OpenCV InterOp Extension <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `mv_compile` - Neural Net Model Compile <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `runcl` - OpenCL&trade; program debug App <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `runvx` - OpenVX&trade; Graph Executor - Display ON <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libvx_nn.so` - OpenVX&trade; Neural Net Extension <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `inference_server_app` - Cloud Inference App                                                                                                                                                                                                                                                                                                                                       | level-4.dockerfile |
| `Level_5`   | AMD_RPP <br> AMD RPP deps <br> +Level 4               | amd_openvx <br> amd_openvx_extensions <br> apps <br> VX RPP <br> utilities | ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libopenvx.so`  - OpenVX&trade; Lib <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvxu.so` - OpenVX&trade; immediate node Lib <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvx_loomsl.so` - Loom 360 Stitch Lib <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `loom_shell` - 360 Stitch App <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvx_amd_media.so` - OpenVX&trade; Media Extension <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvx_opencv.so` - OpenVX&trade; OpenCV InterOp Extension <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `mv_compile` - Neural Net Model Compile <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `runcl` - OpenCL&trade; program debug App <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `runvx` - OpenVX&trade; Graph Executor - Display ON <br>  ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvx_nn.so` - OpenVX&trade; Neural Net Extension <br>  ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `inference_server_app` - Cloud Inference App <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libvx_rpp.so` - OpenVX&trade; RPP Extension | level-5.dockerfile |
