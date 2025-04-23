.. meta::
  :description: MIVisionX API
  :keywords: MIVisionX, ROCm, API, reference, data type, support

.. _samples:

******************************************
MIVisionX samples documentation
******************************************

MIVisionX samples using OpenVX and OpenVX extensions. In the samples below you will learn how to run computer vision, inference, and a combination of computer vision & inference efficiently on target hardware.

* :ref:`gdf-samples`
* :ref:`cc-samples`
* :ref:`loom-samples`
* :ref:`mc-samples`
* :ref:`mv-samples`

.. _gdf-samples:

GDF - Graph Description Format
==============================

MIVisionX samples using `RunVX <https://github.com/ROCm/MIVisionX/tree/master/utilities/runvx/README.md>`_

To run the samples you need to put MIVisionX executables and libraries into the system path:

.. code-block:: shell

  export PATH=$PATH:/opt/rocm/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib

To get help on the ``runvx`` utility use the ``-h`` option:

.. code-block:: shell

  runvx -h


skintonedetect.gdf
------------------

.. image:: https://raw.githubusercontent.com/ROCm/MIVisionX/master/samples/images/skinToneDetect_image.PNG
  :alt: Image of face on left, and skin area highlighted on right

.. code-block:: shell

  runvx gdf/skintonedetect.gdf


Or, when using a live camera:

.. code-block:: shell

  runvx -frames:live gdf/skintonedetect-LIVE.gdf


canny.gdf
---------

.. image:: https://raw.githubusercontent.com/ROCm/MIVisionX/master/samples/images/canny_image.PNG
  :alt: Image of person on left and highlighted edges on right

.. code-block:: shell

  runvx gdf/canny.gdf


Or, when using a live camera:

.. code-block:: shell

  runvx -frames:live gdf/canny-LIVE.gdf


OpenCV_orb-LIVE.gdf
-------------------

Using a live camera:

.. code-block:: shell

  runvx -frames:live gdf/OpenCV_orb-LIVE.gdf


.. _cc-samples:

C/C++ Samples for OpenVX and OpenVX Extensions
==============================================

MIVisionX samples in C/C++

Canny
-----

.. code-block:: shell

  cd c_samples/canny/
  cmake .
  make
  ./cannyDetect --image <imageName> 
  ./cannyDetect --live


Orb Detect
----------

.. code-block:: shell

  cd c_samples/opencv_orb/
  cmake .
  make
  ./orbDetect


.. _loom-samples:

Radeon Loom 360 Stitch Samples
==============================

MIVisionX samples using `LoomShell <https://github.com/ROCm/MIVisionX/tree/master/utilities/loom_shell/README.md>`_

.. image:: https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/loom-4.png
  :alt: Image of video display
  :target: https://youtu.be/E8pPU04iZjw

To run the samples you need to put MIVisionX executables and libraries into the system path:

.. code-block:: shell

  export PATH=$PATH:/opt/rocm/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib

To get help on the ``loom_shell`` utility use the ``-help`` option:

.. code-block:: shell

  loom_shell -help


Sample #1
---------

Get data for the stitch:

.. code-block:: shell

  cd loom_360_stitch/sample-1/
  python loomStitch-sample1-get-data.py


Run ``loom_shell`` script to generate the 360 image:

.. code-block:: shell

  loom_shell loomStitch-sample1.txt


Expected Output:

.. code-block:: shell

  loom_shell loomStitch-sample1.txt 
  loom_shell 0.9.8 [loomsl 0.9.8]
  ... processing commands from loomStitch-sample1.txt
  ..ls_context context[1] created
  ..lsCreateContext: created context context[0]
  ..lsSetOutputConfig: successful for context[0]
  ..lsSetCameraConfig: successful for context[0]
  OK: OpenVX using GPU device#0 (gfx906+sram-ecc) [OpenCL 2.0 ] [SvmCaps 0 0]
  ..lsInitialize: successful for context[0] (1380.383 ms)
  ..cl_mem mem[2] created
  ..cl_context opencl_context[1] created
  ..lsGetOpenCLContext: get OpenCL context opencl_context[0] from context[0]
  OK: loaded cam00.bmp
  OK: loaded cam01.bmp
  OK: loaded cam02.bmp
  OK: loaded cam03.bmp
  ..lsSetCameraBuffer: set OpenCL buffer mem[0] for context[0]
  ..lsSetOutputBuffer: set OpenCL buffer mem[1] for context[0]
  OK: run: executed for 100 frames
  OK: run: Time: 0.919 ms (min); 1.004 ms (avg); 1.238 ms (max); 1.212 ms (1st-frame) of 100 frames
  OK: created LoomOutputStitch.bmp
  > stitch graph profile
  COUNT,tmp(ms),avg(ms),min(ms),max(ms),DEV,KERNEL
  100, 0.965, 1.005, 0.918, 1.237,CPU,GRAPH
  100, 0.959, 0.999, 0.915, 1.234,GPU,com.amd.loomsl.warp
  100, 0.955, 0.994, 0.908, 1.232,GPU,com.amd.loomsl.merge
  OK: OpenCL buffer usage: 324221600, 9/9
  ..lsReleaseContext: released context context[0]
  ... exit from loomStitch-sample1.txt


.. note::
  The stitched output image is saved as ``LoomOutputStitch.bmp``

Sample #2
---------

Get data for the stitch:

.. code-block:: shell

  cd loom_360_stitch/sample-2/
  python loomStitch-sample2-get-data.py


Run ``loom_shell`` script to generate the 360 image:

.. code-block:: shell

  loom_shell loomStitch-sample2.txt


Sample #3
---------

Get data for the stitch:

.. code-block:: shell

  cd loom_360_stitch/sample-3/
  python loomStitch-sample3-get-data.py


Run ``loom_shell`` script to generate the 360 image:

.. code-block:: shell

  loom_shell loomStitch-sample3.txt


.. _mc-samples:

Model Compiler Efficient Inference
==================================

.. image:: ../data/modelCompilerWorkflow.png
  :alt: Image of pretrained neural net models going into model compiler and moving into MIVisionX runtime

The sample applications available in `samples/model_compiler_samples <https://github.com/ROCm/MIVisionX/blob/master/samples/model_compiler_samples/README.md>`_, demonstrate how to run inference efficiently using AMD's open source implementation of OpenVX and OpenVX extensions. The samples review each step required to convert a pre-trained neural net model into an OpenVX graph and run this graph efficiently on the target hardware. 

* `Sample-1: Classification Using Pre-Trained ONNX Model <https://github.com/ROCm/MIVisionX/blob/master/samples/model_compiler_samples/README.md#sample-1---classification-using-pre-trained-onnx-model>`_
* `Sample-2: Detection Using Pre-Trained Caffe Model <https://github.com/ROCm/MIVisionX/blob/master/samples/model_compiler_samples/README.md#sample-2---detection-using-pre-trained-caffe-model>`_ 
* `Sample-3: Classification Using Pre-Trained NNEF Model <https://github.com/ROCm/MIVisionX/blob/master/samples/model_compiler_samples/README.md#sample-3---classification-using-pre-trained-nnef-model>`_
* `Sample-4: Classification Using Pre-Trained Caffe Model <https://github.com/ROCm/MIVisionX/blob/master/samples/model_compiler_samples/README.md#sample-4---classification-using-pre-trained-caffe-model>`_


.. _mv-samples:

MV Object Detect Sample
========================

The `mv_objdetect sample <https://github.com/ROCm/MIVisionX/blob/master/samples/mv_objdetect/README.md>`_ shows how to run video decoding and object detection using a pre-trained `YoloV2` Caffe model. 
The sample demonstrates the use of ``mv_compile`` utility to do video decoding and inference.

.. image:: https://raw.githubusercontent.com/ROCm/MIVisionX/master/samples/mv_objdetect/data/images/Video_4_screenshot.png
  :alt: Street scene with cars and trucks highlighted in boxes
