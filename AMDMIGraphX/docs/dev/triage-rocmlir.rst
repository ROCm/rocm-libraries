.. meta::
  :description: Issue Triaging Guide for suspected issues
  :keywords: MIGraphX, rocMLIR, issues, pipeline, compilation, bug, code base, kernel, contribution, developing

Issue Triaging Guide for suspected rocMLIR issue
================================================

This document serves as a guide to narrow down whether a bug is due to
rocMLIR backend of MIGraphX.

There are broadly 3 categories of bugs that can be due to rocMLIR.

1. [B1]rocMLIR compilation bug
2. [B2]rocMLIR runtime failure
3. [B3]accuracy issue from rocMLIR

Step 1 - Use MIGRAPHX_DISABLE_MLIR=1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First see if the bug persists after disabling MLIR. If so, it is highly likely
it is not a MLIR bug but rather a MIGraphX bug.

If you dont see a failure, please proceed.

Step 2 - See if its a B1 - rocMLIR compilation bug
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MLIR has two pipelines that are used to generate a kernel: highlevel
pipeline and backend pipeline

Step 2.1 If the highlevel pipeline fails, you should see error that starts with
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Invalid MLIR created:``

If the above is present, please disable threading first using
``MIGRAPHX_GPU_COMPILE_PARALLEL=1``. Then use ``MIGRAPHX_TRACE_MLIR=1``
and provide latest MLIR module that gets printed. An example would look
like as follows :

::

   module {
     func.func @mlir_convolution(%arg0: !migraphx.shaped<2x8x3x3xf32, 72x9x3x1>, %arg1: !migraphx.shaped<1x8x4x4xf32, 128x16x4x1>) -> !migraphx.shaped<1x2x2x2xf32, 8x4x2x1> attributes {arch = "gfx90a:sramecc+:xnack-", enable_splitk_for_tuning = true, kernel = "mixr", num_cu = 110 : i64} {
       %0 = migraphx.convolution %arg1, %arg0 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xf32, 128x16x4x1>, <2x8x3x3xf32, 72x9x3x1> -> <1x2x2x2xf32, 8x4x2x1>
       return %0 : !migraphx.shaped<1x2x2x2xf32, 8x4x2x1>
     }
   }

Please provide only the *SINGLE* module that fails and not a dump of
them for rocMLIR team figure out which is failing.

Step 2.2 If the backend pipeline fails for *all solutions*, you should see error that starts with
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``No solutions provided for mlir_*`` or
``No valid tuned compilation for mlir_*``

NOTE 1 : Currently MIGraphX does not run applicability pipeline of
rocMLIR. Therefore, you will see them for inapplicable tuning params as
well.

NOTE 2 : Ignore ``MLIR backend compilation failed: "`` if you don't see
``No solutions provided for mlir_*``

If the above is present, please disable threading first using
``MIGRAPHX_GPU_COMPILE_PARALLEL=1``. Then use ``MIGRAPHX_TRACE_MLIR=2``
and provide latest MLIR module that gets printed. An example would look
like as follows :

::

   module {
     func.func @mlir_convolution(%arg0: memref<1x1x32x32x8xf32>, %arg1: memref<1x16x3x3x8xf32>, %arg2: memref<16xf32>, %arg3: memref<1x1x30x30x16xf32>) attributes {kernel, arch = ""} {
       %0 = memref.alloc() : memref<1x1x30x30x16xf32>
       rock.conv(%arg1, %arg0, %0) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["gi", "ni", "0i", "1i", "ci"], output_layout = ["go", "no", "0o", "1o", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x16x3x3x8xf32>, memref<1x1x32x32x8xf32>, memref<1x1x30x30x16xf32>
       %4 = memref.expand_shape %arg2 [[0, 1, 2, 3, 4]] output_shape [1, 1, 1, 1, 16] : memref<16xf32> into memref<1x1x1x1x16xf32>
       linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%0, %4 : memref<1x1x30x30x16xf32>, memref<1x1x1x1x16xf32>) outs(%arg3 : memref<1x1x30x30x16xf32>) {
       ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
         %8 = arith.addf %arg4, %arg5 : f32
         linalg.yield %8 : f32
       }
       return
     }
   }

Please provide only the *SINGLE* module that fails and not a dump of
them for rocMLIR team figure out which is failing.

Step 3 See if its a B2 - runtime failure of MLIR-generated kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So kernels are first run in the benchmarking process of MIGraphX.
Therefore, set ``MIGRAPHX_TRACE_BENCHMARKING=3``

You should see

::

   Benchmarking mlir_* : xx configs
   Problem: <problem key>
   Benchmarking solution: <perf config>

So you might want to report that to rocMLIR team. However, if its a
fusion specific runtime issue, further triaging is needed.

You would need to manually go through the traces with
``MIGRAPHX_TRACE_MLIR=2`` and figure out which problem has the
``<problem key>`` that fails.

Then report the failing module as in Step 2.2 to rocMLIR team. This can
be very intensive unless some action is taken for
https://github.com/ROCm/AMDMIGraphX/issues/2332

Step 4 See if its a B3 - accuracy issue of MLIR-generated kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the hardest type of triages for rocMLIR.

Here you would need to get MIGraphX modules (not MLIR) that gets printed
in with ``MIGRAPHX_TRACE_MLIR=2``

For e.g. :

::

   # This is MIGRAPHX module

   arg2 = @param:arg2 -> float_type, {1, 5, 3}, {15, 3, 1}
   arg1 = @param:arg1 -> float_type, {1, 4, 3}, {12, 3, 1}
   arg0 = @param:arg0 -> float_type, {1, 5, 4}, {20, 4, 1}
   @3 = dot(arg0,arg1) -> float_type, {1, 5, 3}, {15, 3, 1}
   @4 = add(@3,arg2) -> float_type, {1, 5, 3}, {15, 3, 1}
   @5 = @return(@4)

   # This is the MLIR module

   module {
     func.func @mlir_dot_add(%arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>, %arg2: !migraphx.shaped<1x5x3xf32, 15x3x1>) -> !migraphx.shaped<1x5x3xf32, 15x3x1> attributes {arch = "gfx90a:sramecc+:xnack-", enable_splitk_for_tuning = true, kernel = "mixr", num_cu = 110 : i64} {
       %0 = migraphx.dot %arg0, %arg1 : <1x5x4xf32, 20x4x1>, <1x4x3xf32, 12x3x1> -> <1x5x3xf32, 15x3x1>
       %1 = migraphx.add %0, %arg2 : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1> -> <1x5x3xf32, 15x3x1>
       return %1 : !migraphx.shaped<1x5x3xf32, 15x3x1>
     }
   }

Then individually create MIGraphX program that only has the MIGRAPHX
module Then indiviually ``driver verify`` them to see which is the
failing module.
