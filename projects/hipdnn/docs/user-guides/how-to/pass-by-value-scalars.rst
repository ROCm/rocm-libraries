.. meta::
  :description: How to use compile-time constants and runtime pass-by-value scalars in hipDNN.
  :keywords: hipDNN, ROCm, pass-by-value, epsilon, momentum, runtime scalars

.. _pass-by-value-scalars:

*****************************
Use pass-by-value scalars
*****************************

Several hipDNN operations accept scalar parameters — such as **epsilon** and **momentum** for
batch normalization — as pass-by-value tensors rather than device buffers. hipDNN supports two
ways to supply these scalars:

- **Compile-time constants**: the value is baked into the operation graph at ``build()`` time.
  No entry in the variant pack is needed at ``execute()`` time. This works with any plugin
  version and is the preferred choice when the value is fixed.

- **Runtime pass-by-value**: the value is supplied as a host pointer in the variant pack at
  ``execute()`` time. The graph can be built once and executed repeatedly with different scalar
  values without rebuilding. This requires plugin SDK ≥ 1.2.0.

When to use each
================

Prefer **compile-time constants** when the scalar value is fixed for the lifetime of the program
(for example, a standard epsilon of ``1e-5`` that never changes). The graph is slightly smaller
and no pointer bookkeeping is needed at execute time.

Use **runtime pass-by-value** when the scalar value must change between executions — for example,
when the caller controls epsilon or momentum dynamically — and rebuilding the graph on each change
would be too expensive.

.. note::

   Compile-time constants are always compatible with implementations that support runtime
   pass-by-value, because the baked value can be treated as a constant runtime input internally.
   The reverse is not true: a runtime pass-by-value scalar requires plugin SDK ≥ 1.2.0 and will
   not work with older plugins.

Using compile-time constants
============================

Call ``set_value()`` on the scalar tensor before passing it to the operation attributes.
The tensor must **not** appear in the variant pack at ``execute()``.

.. code-block:: cpp

   #include <hipdnn_data_sdk/utilities/Constants.hpp>
   #include <hipdnn_frontend.hpp>

   // Create epsilon as a compile-time constant.
   auto epsilon = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
   epsilon->set_value(hipdnn_data_sdk::utilities::BATCHNORM_DEFAULT_EPSILON);

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

Using runtime pass-by-value
============================

Call ``set_as_runtime_parameter()`` instead of ``set_value()``. Then at ``execute()``,
include a host pointer to the scalar value in the variant pack.

.. code-block:: cpp

   #include <hipdnn_data_sdk/utilities/Constants.hpp>
   #include <hipdnn_frontend.hpp>

   // Create epsilon as a runtime pass-by-value scalar.
   // set_value() establishes the data type and shape; set_as_runtime_parameter() then
   // clears the baked value so the host pointer in the variant pack is used at execute().
   auto epsilon = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
   epsilon->set_value(1e-5f);
   epsilon->set_as_runtime_parameter();

   hipdnn_frontend::graph::BatchnormAttributes bnAttributes;
   bnAttributes.set_epsilon(epsilon);

   // build the graph once ...
   graph->build(handle);

   // Execute: supply epsilon as a host pointer in the variant pack.
   // The value type must match the data type set on the tensor (FLOAT -> float).
   // epsilonVal can differ on each call without rebuilding the graph.
   // It must remain alive until execute() returns (execute() is synchronous).
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

   The batchnorm samples (``hipdnn_sample_bn_training``,
   ``hipdnn_sample_bn_inference_with_variance``, ``hipdnn_sample_fused_bn_training_activ``,
   and ``hipdnn_sample_fused_bn_inference_variance_activ``) demonstrate both modes.
   Pass ``--runtime-scalars`` to activate the runtime pass-by-value path.
