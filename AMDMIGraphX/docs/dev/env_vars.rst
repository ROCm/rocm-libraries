.. meta::
  :description: MIGraphX internal environment variables
  :keywords: MIGraphX, code base, contribution, developing, env vars, environment variables

Environment Variables
=====================

For parsing
---------------

.. envvar:: MIGRAPHX_TRACE_ONNX_PARSER

Set to "1", "enable", "enabled", "yes", or "true" to use.
Prints debugging traces for the ONNX parser.
Prints: Initializers (if used), ONNX node operators, added MIGraphX instructions.

.. envvar:: MIGRAPHX_DISABLE_FP16_INSTANCENORM_CONVERT

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disables the conversion from fp16 to fp32 for the ``InstanceNormalization`` ONNX operator that MIGX does as a workaround for accuracy issues with `reduce_mean/variance`.
See ``parse_instancenorm.cpp`` for more details.


Matchers
------------

.. envvar:: MIGRAPHX_TRACE_MATCHES

Set to "1" to print the matcher that matches an instruction and the matched instruction.
Set to "2" and use the ``MIGRAPHX_TRACE_MATCHES_FOR`` flag to filter out results.

.. envvar:: MIGRAPHX_TRACE_MATCHES_FOR

Set to the name of any matcher to print the traces for that matcher only.

.. envvar:: MIGRAPHX_VALIDATE_MATCHES

Set to "1", "enable", "enabled", "yes", or "true" to use.
Validates the module after finding the matches (runs ``module.validate()``).

Program Execution 
---------------------

.. envvar:: MIGRAPHX_TRACE_EVAL

Set to "1", "2", or "3" to use.
"1" prints the instruction run and the time taken.
"2" prints everything in "1" and a snippet of the output argument and some output statistics (e.g. min, max, mean).
"3" prints everything in "1" and all output buffers.


Program Verification
------------------------

.. envvar:: MIGRAPHX_VERIFY_ENABLE_ALLCLOSE

Set to "1", "enable", "enabled", "yes", or "true" to use.
Uses ``allclose`` with the given ``atol`` and ``rtol`` for verifying ranges with ``driver verify`` or the tests that use ``migraphx/verify.hpp``.


Pass debugging or Pass controls
-----------------------------------

.. envvar:: MIGRAPHX_TRACE_ELIMINATE_CONTIGUOUS

Set to "1", "enable", "enabled", "yes", or "true" to use.
Debug print the instructions that have input ``contiguous`` instructions removed.

.. envvar:: MIGRAPHX_DISABLE_POINTWISE_FUSION

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disables the ``fuse_pointwise`` compile pass.

.. envvar:: MIGRAPHX_DEBUG_MEMORY_COLORING

Set to "1", "enable", "enabled", "yes", or "true" to use.
Prints debug statements for the ``memory_coloring`` pass.

.. envvar:: MIGRAPHX_TRACE_SCHEDULE

Set to "1", "enable", "enabled", "yes", or "true" to use.
Prints debug statements for the ``schedule`` pass.

.. envvar:: MIGRAPHX_TRACE_PROPAGATE_CONSTANT

Set to "1", "enable", "enabled", "yes", or "true" to use.
Traces instructions replaced with a constant.

.. envvar:: MIGRAPHX_TRACE_QUANTIZATION

Set to "1", "enable", "enabled", "yes", or "true" to use.
Prints traces for any passes run during quantization.

.. envvar:: MIGRAPHX_8BITS_QUANTIZATION_PARAMS

Set to "1", "enable", "enabled", "yes", or "true" to use.
Prints the quantization parameters in the main module only.

.. envvar:: MIGRAPHX_DISABLE_DNNL_POST_OPS_WORKAROUND

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disables the DNNL post ops workaround.

.. envvar:: MIGRAPHX_DISABLE_MIOPEN_FUSION

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disables MIOpen fusions.

.. envvar:: MIGRAPHX_DISABLE_SCHEDULE_PASS

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disables the ``schedule`` pass.

.. envvar:: MIGRAPHX_DISABLE_REDUCE_FUSION

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disables the ``fuse_reduce`` pass.

.. envvar:: MIGRAPHX_ENABLE_REWRITE_DOT

Set to "1", "enable", "enabled", "yes", or "true" to use.
Enable the ``rewrite_dot`` pass.

.. envvar:: MIGRAPHX_SPLIT_REDUCE_SIZE
Set to the minimum size of a reduction to do a split reduce. Overrides what
is set in the backend. Set to -1 to disable split reduce completely.

.. envvar:: MIGRAPHX_ENABLE_NHWC

Set to "1", "enable", "enabled", "yes", or "true" to use.
Enables the ``layout_nhwc`` pass.

.. envvar:: MIGRAPHX_ENABLE_CK

Set to "1", "enable", "enabled", "yes", or "true" to use.
Enables use of the Composable Kernels library.
Use it in conjunction with ``MIGRAPHX_DISABLE_MLIR=1``.

.. envvar:: MIGRAPHX_DISABLE_MLIR*
Set to "1", "enable", "enabled", "yes", or "true" to use.
Disables use of the rocMLIR library.

.. envvar:: MIGRAPHX_SET_GEMM_PROVIDER

Set to "hipblaslt" to use hipBLASLt.
Set to "rocblas" to use rocBLAS.

.. envvar:: MIGRAPHX_COPY_LITERALS

Set to "1", "enable", "enabled", "yes", or "true" to use.
Uses ``hip_copy_to_gpu`` with a new ``literal`` instruction rather than using ``hip_copy_literal{}``.

.. envvar:: MIGRAPHX_ENABLE_LAYERNORM_FUSION

Set to "1", "enable", "enabled", "yes", or "true" to use.
Enables former custom layernorm fusion kernel, which has been superseded by
the reduction fusion. This is only here to check for performance issues with
the reduction fusions, and will be removed in the future.


.. envvar:: MIGRAPHX_DISABLE_MIOPEN_POOLING

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disables use of MIOpen for pooling operations and uses JIT implementation instead.


Compilation traces
----------------------

.. envvar:: MIGRAPHX_TRACE_FINALIZE

Set to "1", "enable", "enabled", "yes", or "true" to use.
Debug print instructions during the ``module.finalize()`` step.

.. envvar:: MIGRAPHX_TRACE_COMPILE

Set to "1", "enable", "enabled", "yes", or "true" to use.
Prints trace information for the graph compilation process.

.. envvar:: MIGRAPHX_TRACE_PASSES

Set to "1", "enable", "enabled", "yes", or "true" to use.
Prints the compile pass and the program after the pass.

.. envvar:: MIGRAPHX_TIME_PASSES

Set to "1", "enable", "enabled", "yes", or "true" to use.
Times the compile passes.

.. envvar:: MIGRAPHX_DISABLE_PASSES

Set to the name of the pass you want to skip.  Comma separated list is supported
Example "rewrite_pooling,rewrite_gelu"


GPU kernels JIT compilation debugging 
----------------------------------------

These environment variables are applicable for both hiprtc and hipclang.

.. envvar:: MIGRAPHX_TRACE_CMD_EXECUTE

Set to "1", "enable", "enabled", "yes", or "true" to use.
Prints commands executed by the MIGraphX ``process``.

.. envvar:: MIGRAPHX_TRACE_HIPRTC

Set to "1", "enable", "enabled", "yes", or "true" to use.
Prints HIPRTC options and C++ file executed.

.. envvar:: MIGRAPHX_DEBUG_SAVE_TEMP_DIR

Set to "1", "enable", "enabled", "yes", or "true" to use.
Prevents deletion of the created temporary directories.

.. envvar:: MIGRAPHX_GPU_DEBUG

Set to "1", "enable", "enabled", "yes", or "true" to use.
Internally, this adds the option ``-DMIGRAPHX_DEBUG`` when compiling GPU kernels. It enables assertions and capture of source locations for the errors. 

.. envvar:: MIGRAPHX_GPU_DEBUG_SYM

Set to "1", "enable", "enabled", "yes", or "true" to use.
Adds the option ``-g`` when compiling HIPRTC.

.. envvar:: MIGRAPHX_GPU_DUMP_SRC

Set to "1", "enable", "enabled", "yes", or "true" to use.
Dumps the compiled HIPRTC source files.

.. envvar:: MIGRAPHX_GPU_DUMP_ASM

Set to "1", "enable", "enabled", "yes", or "true" to use.
Dumps the hip-clang assembly.

.. envvar:: MIGRAPHX_GPU_OPTIMIZE

Set the optimization mode for GPU compile (``-O`` option).
Defaults to ``-O3``.

.. envvar:: MIGRAPHX_GPU_COMPILE_PARALLEL

Set to the number of threads to use.
Compiles GPU code in parallel with the given number of threads.

.. envvar:: MIGRAPHX_TRACE_NARY

Set to "1", "enable", "enabled", "yes", or "true" to use.
Prints the ``nary`` device functions used.

.. envvar:: MIGRAPHX_ENABLE_HIPRTC_WORKAROUNDS

Set to "1", "enable", "enabled", "yes", or "true" to use.
Enables HIPRTC workarounds for bugs in HIPRTC.

.. envvar:: MIGRAPHX_USE_FAST_SOFTMAX

Set to "1", "enable", "enabled", "yes", or "true" to use.
Uses fast softmax optimization.

.. envvar:: MIGRAPHX_ENABLE_NULL_STREAM

Set to "1", "enable", "enabled", "yes", or "true" to use.
Allows using null stream for miopen and hipStream.

.. envvar:: MIGRAPHX_NSTREAMS

Set to the number of streams to use.
Defaults to 1.

.. envvar:: MIGRAPHX_TRACE_BENCHMARKING

Set to "1" to print benchmarking trace.
Set to "2" to print detailed benchmarking trace.
Set to "3" to print compiled traces.

.. envvar:: MIGRAPHX_PROBLEM_CACHE

Set to path to json file to load and save problem cache.
This will load the json file into the problem cache if it exists, and when
compilation finishes it will save the problem cache.

.. envvar:: MIGRAPHX_BENCHMARKING_BUNDLE

Set amount of configs to run in a bundle when timing.

.. envvar:: MIGRAPHX_BENCHMARKING_NRUNS

Set amount of timing runs for each config bundle being benchmarked.

MLIR vars
-------------

.. envvar:: MIGRAPHX_TRACE_MLIR

Set to "1" to trace MLIR and print any failures.
Set to "2" to additionally print all MLIR operations.

.. envvar:: MIGRAPHX_MLIR_USE_SPECIFIC_OPS

Set to the MLIR operations you want to always use regardless of the GPU architecture.
Accepts a list of operators separated by commas (e.g. "fused", "convolution", "dot").

.. envvar:: MIGRAPHX_MLIR_TUNING_DB

Set to the path of the MLIR tuning database to load.

.. envvar:: MIGRAPHX_MLIR_TUNING_CFG

Set to the path of the tuning configuration.
Appends to tuning cfg file that could be used with rocMLIR tuning scripts.

.. envvar:: MIGRAPHX_MLIR_TUNE_EXHAUSTIVE

Set to "1", "enable", "enabled", "yes", or "true" to use.
Performs exhaustive tuning for MLIR.

.. envvar:: MIGRAPHX_MLIR_TUNE_LIMIT

Set to an integer greater than 1.
Limits the number of solutions available to MLIR for tuning.

.. envvar:: MIGRAPHX_ENABLE_MLIR_INPUT_FUSION

Set to "1", "enable", "enabled", "yes", or "true" to use.
Enable input fusions in MLIR.

.. envvar:: MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION

Set to "1", "enable", "enabled", "yes", or "true" to use.
Enable reduction fusions in MLIR.

.. envvar:: MIGRAPHX_MLIR_ENABLE_SPLITK

Set to "1", "enable", "enabled", "yes", or "true" to use.
Enable Split-k perf configs when tuning with MLIR.

.. envvar:: MIGRAPHX_MLIR_DUMP_TO_MXR

Set to path where MXRs will be saved.
Dumps MLIRs module to mxr files.

.. envvar:: MIGRAPHX_MLIR_DUMP

Set to path where MLIRs will be saved.
Dumps MLIRs module to .mlir files.

CK vars
-----------

.. envvar:: MIGRAPHX_LOG_CK_GEMM

Set to "1", "enable", "enabled", "yes", or "true" to use.
Prints composable kernels GEMM traces.

.. envvar:: MIGRAPHX_CK_DEBUG

Set to "1", "enable", "enabled", "yes", or "true" to use.
Mandatorily adds ``-DMIGRAPHX_CK_CHECK=1`` for compiling composable kernel operators.

.. envvar:: MIGRAPHX_TUNE_CK

Set to "1", "enable", "enabled", "yes", or "true" to use.
Performs tuning for composable kernels.

hipBLASLt vars
--------------

.. envvar:: MIGRAPHX_ENABLE_HIP_GEMM_TUNING

Set to "1", "enable", "enabled", "yes", or "true" to use.
Performs exhaustive tuning for hipBLASLt.

Testing 
------------

.. envvar:: MIGRAPHX_TRACE_TEST_COMPILE

Set to the target whose compilation you want to trace (e.g. "gpu", "cpu").
Prints the compile trace for verify tests on the given target.
Don't use this flag in conjunction with ``MIGRAPHX_TRACE_COMPILE``.

.. envvar:: MIGRAPHX_TRACE_TEST

Set to "1", "enable", "enabled", "yes", or "true" to use.
Prints the reference and target programs even if the verify tests pass.

.. envvar:: MIGRAPHX_DUMP_TEST

Set to "1", "enable", "enabled", "yes", or "true" to use.
Dumps verify tests to ``.mxr`` files.

.. envvar:: MIGRAPHX_VERIFY_DUMP_DIFF

Set to "1", "enable", "enabled", "yes", or "true" to use.
Dumps the output of the test (and the reference) results when they differ.
