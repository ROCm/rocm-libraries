.. meta::
  :description: MIVisionX API
  :keywords: MIVisionX, ROCm, API, reference, data type, support

.. _mvdeploy-ref:

******************************************
MV_deploy reference
******************************************

.. note:: 
    This project has the source code for MIVIsionX model compiler in ``mv_compile.cpp``

The ``mv_deploy`` utility consists of a model-compiler and necessary header and ``.cpp`` files required to run inference for a specific Neural Net model. 

The ``mv_compile`` will be built as part of MIVisionX package installer. 
To build an application using ``mv_compile``, you can use the deployment API from ``mv_deploy.h``.
The use of ``mv_compile`` and deployment is shown in `mv_objdetect sample <https://github.com/ROCm/MIVisionX/tree/master/samples/mv_objdetect>`_.
The sample demonstrates the use of ``mv_compile`` utility to do video decoding and inference.

Prerequisites
=============

* Ubuntu 20.04 or 22.04, or CentOS 7 or 8
* `ROCm supported hardware <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_ 

	* AMD Radeon GPU or APU required

* `ROCm installation <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_
* MIVisionX :ref:`installation`

    .. note::
        MIVisionX installs model compiler at `/opt/rocm/libexec/mivisionx`
        ``mv_compile`` is installed at ``/opt/rocm/bin`` and ``mvdeploy_api.h`` is installed at ``/opt/rocm/include/mivisionx`` 


Usage
=====

The ``mv_compile`` utility generates deployment library, header files, and ``.cpp`` files required to run inference for the specified model.

.. code-block:: shell

    mv_compile
	     --model <model_name: name of the trained model with path> 		[required]
	     --install_folder <install_folder:  the location for compiled model> 		[required]
	     --input_dims <input_dims: n,c,h,w - batch size, channels, height, width> 	[required]
	     --backend <backend: name of the backend for compilation> 	  		[optional - default:OpenVX_Rocm_GPU]
	     --fuse_cba <fuse_cba: enable or disable Convolution_bias_activation fuse mode (0/1)> [optional - default: 0]
	     --quant_mode <quant_mode: fp32/fp16 - quantization_mode for the model: if enabled the model and weights would be converted [optional -default: fp32]


Examples
========

* Caffe

.. code-block:: shell

    ./mv_compile --model models/model.caffemodel --install_folder install_folder --input_dims 1,3,224,224


* ONNX

.. code-block:: shell

    ./mv_compile --model models/model.onnx --install_folder install_folder --input_dims 1,3,224,224


* NNEF

.. code-block:: shell

    ./mv_compile --model models/model.nnef --install_folder install_folder --input_dims 1,3,224,224

