# Utilities

MIVisionX has utility applications which could be used by developers to test, quick prototype or develop sample applications.

## [Inference Generator](inference_generator/README.md#inference-generator)

caffe2openvx: Convert a pre-trained CAFFE model into a C library for use by applications.

## [Loom Shell](loom_shell/README.md#radeon-loomshell)

LoomShell is an interpreter that enables stitching 360 degree videos using a script. It provides direct access to Live Stitch API by encapsulating the calls to enable rapid prototyping.

## [mv_deploy](mv_deploy/README.md)

mv_deploy consists of a model-compiler and necessary header/.cpp files which are required to run inference for a specific NeuralNet model. `mv_compile` will be built as part of MIVisionX package installer

## [RunCL](runcl/README.md#amd-runcl)

RunCL is a command-line tool to build, execute, and debug OpenCL programs, with a simple, easy-to-use interface.

## [RunVX](runvx/README.md#amd-runvx)

RunVX is a command-line tool to execute OpenVX graphs, with a simple, easy-to-use interface. It encapsulates most of the routine OpenVX calls, thus speeding up development and enabling rapid prototyping. As input, RunVX takes a GDF (Graph Description Format) file, a simple and intuitive syntax to describe the various data, nodes, and their dependencies. The tool has other useful features, such as, file read/write, data compares, image and keypoint data visualization, etc.
