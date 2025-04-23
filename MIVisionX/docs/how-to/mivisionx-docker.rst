.. meta::
  :description: MIVisionX API
  :keywords: MIVisionX, ROCm, API, reference, data type, support

.. _mivisionx-docker:

******************************************
MIVisionX Docker documentation
******************************************

Docker is a set of platform as a service (PaaS) products that use OS-level virtualization to deliver software in packages called containers. Refer to `Rocm Docker Wiki <https://github.com/ROCm/MIVisionX/wiki/Docker>`_ for additional information.

Docker workflow on Ubuntu 20.04/22.04
=====================================

Prerequisites
-------------

* Ubuntu 20.04/22.04
* `ROCm supported hardware <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_
* `Install ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_ with ``--usecase=rocm``
* `Docker <https://docs.docker.com/engine/install/ubuntu/>`_


Workflow
--------

1. Get the latest docker image. Use the following command to bring in the latest changes from upstream.

.. code-block:: shell

    sudo docker pull mivisionx/ubuntu-20.04:latest

2.  Run docker image

Run docker image: Local Machine
-------------------------------

.. code-block:: shell

  sudo docker run -it --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mem --cap-add=SYS_RAWIO  --group-add video --shm-size=4g --ipc="host" --network=host mivisionx/ubuntu-20.04:latest

* Computer Vision Test

.. code-block:: shell

  python3 /workspace/MIVisionX/tests/vision_tests/runVisionTests.py --num_frames 1


* Neural Network Test

.. code-block:: shell

  python3 /workspace/MIVisionX/tests/neural_network_tests/runNeuralNetworkTests.py --profiler_level 1


* Khronos OpenVX 1.3.0 Conformance Test

.. code-block:: shell

  python3 /workspace/MIVisionX/tests/conformance_tests/runConformanceTests.py --backend_type HOST


Option 1: Map localhost directory on the docker image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Option to map the localhost directory with data to be accessed on the docker image: ``-v {LOCAL_HOST_DIRECTORY_PATH}:{DOCKER_DIRECTORY_PATH}``

.. code-block:: shell

  sudo docker run -it -v /home/:/root/hostDrive/ --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mem --cap-add=SYS_RAWIO  --group-add video --shm-size=4g --ipc="host" --network=host mivisionx/ubuntu-20.04:latest


Option 2: Display with docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Using host display for docker

.. code-block:: shell

  xhost +local:root
  sudo docker run -it --privileged --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host --env DISPLAY=$DISPLAY --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --volume /tmp/.X11-unix/:/tmp/.X11-unix mivisionx/ubuntu-20.04:latest


* Test display with MIVisionX sample

.. code-block:: shell

  runvx -v /opt/rocm/share/mivisionx/samples/gdf/canny.gdf


Run docker image with display: Remote Server Machine
----------------------------------------------------

.. code-block:: shell

  sudo docker run -it --privileged --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host --env DISPLAY=$DISPLAY --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --volume /tmp/.X11-unix/:/tmp/.X11-unix mivisionx/ubuntu-20.04:latest


* Display with MIVisionX sample

.. code-block:: shell

  runvx -v /opt/rocm/share/mivisionx/samples/gdf/canny.gdf


Build - dockerfiles
===================

.. code-block:: shell

  sudo docker build --build-arg {ARG_1_NAME}={ARG_1_VALUE} [--build-arg {ARG_2_NAME}={ARG_2_VALUE}] -f {DOCKER_FILE_NAME}.dockerfile -t {DOCKER_IMAGE_NAME} .


Run - docker
============

.. code-block:: shell

  sudo docker run -it --privileged --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host --env DISPLAY=$DISPLAY --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --volume /tmp/.X11-unix/:/tmp/.X11-unix {DOCKER_IMAGE_NAME}


Ubuntu `20`/`22` DockerFiles
============================

.. # COMMENT: The following lines define objects for use in the tabel below. 
.. |br| raw:: html 

    <br />

.. |green-sq| image:: https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png
    :alt: Green Square
.. |blue-sq| image:: https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png
    :alt: Blue Square
.. |ub-lvl1| image:: https://img.shields.io/docker/v/kiritigowda/ubuntu-18.04/mivisionx-level-1?style=flat-square
    :alt: Ubuntu 18.04 Level 1
.. |ub-lvl2| image:: https://img.shields.io/docker/v/kiritigowda/ubuntu-18.04/mivisionx-level-2?style=flat-square
    :alt: Ubuntu 18.04 Level 1
.. |ub-lvl3| image:: https://img.shields.io/docker/v/kiritigowda/ubuntu-18.04/mivisionx-level-3?style=flat-square
    :alt: Ubuntu 18.04 Level 1
.. |ub-lvl4| image:: https://img.shields.io/docker/v/kiritigowda/ubuntu-18.04/mivisionx-level-4?style=flat-square
    :alt: Ubuntu 18.04 Level 1
.. |ub-lvl5| image:: https://img.shields.io/docker/v/kiritigowda/ubuntu-18.04/mivisionx-level-5?style=flat-square
    :alt: Ubuntu 18.04 Level 1

* |green-sq| New component added to the level
* |blue-sq| Existing component from the previous level

.. csv-table::
  :widths: 5, 5, 8, 16, 5

    **Build Level**, **MIVisionX Dependencies**, **Modules**, **Libraries and Executables**, **Docker File**
    Level_1, cmake |br| gcc |br| g++, amd_openvx  |br| utilities, |green-sq| ``libopenvx.so`` - OpenVX Lib - CPU |br| |green-sq| ``libvxu.so`` - OpenVX immediate node Lib - CPU |br| |green-sq| ``runvx`` - OpenVX Graph Executor - CPU with Display OFF, ``level-1.dockerfile``
    Level_2, ROCm OpenCL |br| +Level 1, amd_openvx |br| amd_openvx_extensions |br| utilities, |blue-sq| ``libopenvx.so`` - OpenVX Lib - CPU/GPU |br| |blue-sq| ``libvxu.so`` - OpenVX immediate node Lib - CPU/GPU |br| |green-sq| ``libvx_loomsl.so`` - Loom 360 Stitch Lib |br| |green-sq| ``loom_shell`` - 360 Stitch App |br| |green-sq| ``runcl`` - OpenCL Debug App |br| |blue-sq| ``runvx`` - OpenVX Graph Executor - Display OFF, ``level-2.dockerfile``
    Level_3, OpenCV |br| FFMPEG |br| +Level 2, amd_openvx |br| amd_openvx_extensions |br| utilities, |blue-sq| ``libopenvx.so`` - OpenVX Lib |br| |blue-sq| ``libvxu.so`` - OpenVX immediate node Lib |br| |blue-sq| ``libvx_loomsl.so`` - Loom 360 Stitch Lib |br| |blue-sq| ``loom_shell`` - 360 Stitch App |br| |blue-sq| ``runcl`` - OpenCL Debug App |br| |green-sq| ``libvx_amd_media.so`` - OpenVX Media Extension |br| |green-sq| ``libvx_opencv.so`` - OpenVX OpenCV InterOp Extension |br| |green-sq| ``mv_compile`` - Neural Net Model Compile |br| |blue-sq| ``runvx`` - OpenVX Graph Executor - Display ON, ``level-3.dockerfile``
    Level_4, MIOpenGEMM |br| MIOpen |br| ProtoBuf |br| +Level 3, amd_openvx |br| amd_openvx_extensions |br| apps |br| utilities, |blue-sq| ``libopenvx.so`` - OpenVX Lib |br| |blue-sq| ``libvxu.so`` - OpenVX immediate node Lib |br| |blue-sq| ``libvx_loomsl.so`` - Loom 360 Stitch Lib |br| |blue-sq| ``loom_shell`` - 360 Stitch App |br| |blue-sq| ``libvx_amd_media.so`` - OpenVX Media Extension |br| |blue-sq| ``libvx_opencv.so`` - OpenVX OpenCV InterOp Extension |br| |blue-sq| ``mv_compile`` - Neural Net Model Compile |br| |blue-sq| ``runcl`` - OpenCL Debug App |br| |blue-sq| ``runvx`` - OpenVX Graph Executor - Display ON |br| |green-sq| ``libvx_nn.so`` - OpenVX Neural Net Extension |br| |green-sq| ``inference_server_app`` - Cloud Inference App, ``level-4.dockerfile``
    Level_5, AMD_RPP |br| RPP deps |br| +Level 4, amd_openvx |br| amd_openvx_extensions |br| apps |br| AMD VX RPP |br| utilities, |blue-sq| ``libopenvx.so`` - OpenVX Lib |br| |blue-sq| ``libvxu.so`` - OpenVX immediate node Lib |br| |blue-sq| ``libvx_loomsl.so`` - Loom 360 Stitch Lib |br| |blue-sq| ``loom_shell`` - 360 Stitch App |br| |blue-sq| ``libvx_amd_media.so`` - OpenVX Media Extension |br| |blue-sq| ``libvx_opencv.so`` - OpenVX OpenCV InterOp Extension |br| |blue-sq| ``mv_compile`` - Neural Net Model Compile |br| |blue-sq| ``runcl`` - OpenCL Debug App |br| |blue-sq| ``runvx`` - OpenVX Graph Executor - Display ON |br| |blue-sq| ``libvx_nn.so`` - OpenVX Neural Net Extension |br| |blue-sq| ``inference_server_app`` - Cloud Inference App |br| |green-sq| ``libvx_rpp.so`` - OpenVX RPP Extension, ``level-5.dockerfile``

