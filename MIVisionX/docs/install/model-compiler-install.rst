.. meta::
  :description: MIVisionX API
  :keywords: MIVisionX, ROCm, API, reference, data type, support

.. _model-compiler-install:

*********************************************
Model compiler installation and configuration
*********************************************

The following describes installing and setting up the model compiler & optimizer and neural net models for Linux. 

Prerequisites
==============

* Linux

  + Ubuntu 20.04 / 22.04
  + CentOS 7 / 8
  + RHEL 8 / 9

* MIVisionX installed as described in :ref:`installation`
* Install Linux Packages

  + Ubuntu

    .. code-block:: shell
      
      sudo apt-get -y install python3 python3-pip protobuf-compiler libprotoc-dev


  + CentOS/RHEL

    .. code-block:: shell
      
        sudo yum -y python3-devel python3-pip protobuf python3-protobuf


* Install PIP3 and Python Packages

  .. code-block:: shell
    
      sudo pip3 install future==1.0.0 pytz==2022.1 numpy==1.23.0


.. note::
  MIVisionX installs model compiler scripts at ``/opt/rocm/libexec/mivisionx/model_compiler/python/``


Setting up neural networks
==========================

The following lists the requirements and steps to configure supported neural nets for use with model compiler. 

Caffe
-----

* protobuf
* google


  .. code-block:: shell

    sudo pip3 install google==3.0.0 protobuf==3.12.4


ONNX
----

* protobuf
* onnx

  .. code-block:: shell

    sudo pip3 install protobuf==3.12.4 onnx==1.12.0


  .. note::
    ONNX Models are available at `ONNX Model Zoo <https://github.com/onnx/models>`_

NNEF
----

* `nnef-parser <https://github.com/KhronosGroup/NNEF-Tools>`_ - Build the nnef python module

  .. code-block:: shell

    git clone -b nnef-v1.0.0 https://github.com/KhronosGroup/NNEF-Tools.git
    cd NNEF-Tools/parser/cpp
    mkdir -p build && cd build
    cmake ../
    make
    sudo make install
    cd ../../../python
    sudo python3 setup.py install


  .. note::
    NNEF models are available at `NNEF Model Zoo <https://github.com/KhronosGroup/NNEF-Tools/tree/master/models#nnef-model-zoo>`_


