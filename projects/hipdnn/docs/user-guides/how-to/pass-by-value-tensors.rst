.. meta::
  :description: How to use compile-time constants and runtime pass-by-value tensors in hipDNN.
  :keywords: hipDNN, ROCm, pass-by-value, epsilon, momentum, runtime pass-by-value

.. _pass-by-value-tensors:

************************************
Use pass-by-value tensors in hipDNN
************************************

Several hipDNN operations accept scalar parameters — such as **epsilon** and **momentum** for
batch normalization — as pass-by-value tensors rather than device buffers. hipDNN supports two
ways to supply these tensors:

- **Compile-time constants**: The value is baked into the operation graph at ``build()`` time.
  No entry in the variant pack is needed at ``execute()`` time. This works with any plugin
  version and is the preferred choice when the value is fixed.

- **Runtime pass-by-value**: The value is supplied as a host pointer in the variant pack at
  ``execute()`` time. The graph can be built once and executed repeatedly with different scalar
  values without rebuilding. This requires plugin SDK ≥ 1.2.0.

When to use each
================

Use compile-time constants when the scalar value is fixed for the lifetime of the program
(for example, a standard epsilon of ``1e-5`` that never changes). The graph is slightly smaller
and no pointer bookkeeping is needed at execute time. Compile-time constants are also more
broadly supported: some kernels only accept compile-time constants for certain scalar
parameters, and reading a baked-in value avoids the host-pointer indirection that runtime
pass-by-value tensors require, giving a slight performance advantage.

Use runtime pass-by-value when the scalar value must change between executions — for example,
when the caller controls epsilon or momentum dynamically — and rebuilding the graph on each change
would be too expensive.

.. note::

   Compile-time constants are compatible with all implementations that support runtime
   pass-by-value, because the baked value can be treated as a constant runtime input internally.
   The reverse is not true: a runtime pass-by-value tensor requires plugin SDK ≥ 1.2.0 and will
   not work with older plugins.

Use compile-time constants
============================

Call ``set_compile_time_constant()`` on the tensor before passing it to the operation
attributes. The tensor must *not* appear in the variant pack at ``execute()``.

.. code-block:: cpp

   #include <hipdnn_data_sdk/utilities/Constants.hpp>
   #include <hipdnn_frontend.hpp>

   // Create epsilon as a compile-time constant.
   auto epsilon = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
   epsilon->set_compile_time_constant(hipdnn_data_sdk::utilities::BATCHNORM_DEFAULT_EPSILON);

   hipdnn_frontend::graph::BatchnormAttributes bnAttributes;
   bnAttributes.set_epsilon(epsilon);

   // build the graph ...
   graph->build(handle);

   // Execute: epsilon is NOT in the variant pack.
   std::unordered_map<int64_t, void*> variantPack;
   variantPack[x->get_uid()]     = xDevicePtr;
   variantPack[y->get_uid()]     = yDevicePtr;
   // ... other tensors ...
   graph->execute(handle, variantPack, workspace);

Use runtime pass-by-value
============================

Declare the tensor's shape and data type with ``set_dim()``, ``set_stride()``, and
``set_data_type()``, then call ``set_as_runtime_parameter()`` to switch the tensor into
runtime mode without baking in a value. Then at ``execute()``, include a host pointer to the
scalar value in the variant pack.

.. code-block:: cpp

   #include <hipdnn_data_sdk/utilities/Constants.hpp>
   #include <hipdnn_frontend.hpp>

   // Create epsilon as a runtime pass-by-value tensor.
   auto epsilon = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
   epsilon->set_dim({1}).set_stride({1}).set_data_type(hipdnn_frontend::DataType::FLOAT);
   // No baked value; the value is supplied as a host pointer in the variant pack at
   // execute() instead.
   epsilon->set_as_runtime_parameter();

   hipdnn_frontend::graph::BatchnormAttributes bnAttributes;
   bnAttributes.set_epsilon(epsilon);

   // build the graph once ...
   graph->build(handle);

   // Execute: supply epsilon as a host pointer in the variant pack.
   // The value type must match the data type set on the tensor (FLOAT -> float).
   // epsilonVal can differ on each call without rebuilding the graph.
   // execute() is not synchronous: the value is copied into the kernel dispatch, so
   // epsilonVal only needs to remain valid until execute() returns — it may be changed
   // immediately afterward (unlike a device pass-by-value tensor, whose buffer must stay
   // valid until the GPU kernel finishes, requiring a synchronize).
   float epsilonVal = 1e-5f;
   std::unordered_map<int64_t, void*> variantPack;
   variantPack[x->get_uid()]           = xDevicePtr;
   variantPack[y->get_uid()]           = yDevicePtr;
   variantPack[epsilon->get_uid()]     = &epsilonVal;  // host pointer; read by execute(), not stored
   // ... other tensors ...
   graph->execute(handle, variantPack, workspace);

   // Vary the value on the next execution without rebuilding the graph.
   epsilonVal = 1e-4f;
   graph->execute(handle, variantPack, workspace);

.. tip::

   The batchnorm samples (`hipdnn_sample_bn_training
   <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/samples/batchnorm/BnTraining.cpp>`_,
   `hipdnn_sample_bn_inference_with_variance
   <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/samples/batchnorm/BnInferenceWithVariance.cpp>`_,
   `hipdnn_sample_fused_bn_training_activ
   <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/samples/batchnorm/FusedBnTrainingActiv.cpp>`_,
   `hipdnn_sample_fused_bn_inference_variance_activ
   <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/samples/batchnorm/FusedBnInferenceVarianceActiv.cpp>`_,
   and `hipdnn_sample_sdpa_fprop
   <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/samples/sdpa/SdpaFprop.cpp>`_)
   demonstrate both modes. Pass ``--runtime-pass-by-value`` to activate the runtime pass-by-value
   path.
