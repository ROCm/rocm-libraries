.. meta::
  :description: MIVisionX API
  :keywords: MIVisionX, ROCm, API, reference, data type, support

.. _toolkit:

******************************************
MIVisionX utilities documentation
******************************************

MIVisionX has utilities that can be used to develop, quickly prototype, and test sample applications.

* `Loom Shell <./utilities-loom_shell.html>`_ (``loom_shell``): is an interpreter that enables stitching 360 degree videos using a script. It provides direct access to Live Stitch API by encapsulating the calls to enable rapid prototyping.

* `mv_deploy <./utilities-mv_deploy.html>`_ (``mv_deploy``): consists of a model-compiler and necessary header/.cpp files which are required to run inference for a specific NeuralNet model. `mv_compile` will be built as part of MIVisionX package installer

* `RunCL <./utilities-runcl.html>`_ (``runcl``): RunCL is a command-line tool to build, execute, and debug OpenCL programs with a simple and easy-to-use interface.

* `RunVX <./utilities-runvx.html>`_ (``runvx``): RunVX is a command-line tool to execute OpenVX graphs, with a simple, easy-to-use interface. It encapsulates most of the routine OpenVX calls, thus speeding up development and enabling rapid prototyping. As input, RunVX takes a GDF (Graph Description Format) file, a simple and intuitive syntax to describe the various data, nodes, and their dependencies. The tool has other useful features, such as, file read/write, data compares, image and keypoint data visualization, etc.
