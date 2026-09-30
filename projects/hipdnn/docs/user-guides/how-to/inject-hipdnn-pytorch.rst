.. meta::
  :description: Learn how to route PyTorch operations onto hipDNN engines with the hipdnn_torch injection layer.
  :keywords: hipDNN, ROCm, PyTorch, injection, monkeypatch, SDPA, attention, linear, RMSNorm

.. _inject-hipdnn-pytorch:

*********************************
Inject hipDNN into PyTorch models
*********************************

This topic demonstrates how to run an unmodified PyTorch model on hipDNN engines using
the ``hipdnn_torch`` injection layer, and how to read the report it produces of which
operations routed to hipDNN and which fell back to native PyTorch.

.. note::

  ``hipdnn_torch`` is an experimental bring-up and end-to-end testing aid, not a
  supported product. It monkeypatches PyTorch internals and depends on a hipDNN engine
  plugin whose kernel family and shape constraints are still narrow. Its value today is
  correctness parity plus an honest census of what still falls back to native PyTorch —
  the list that drives future kernel work.

What it does
============

``hipdnn_torch`` replaces a small set of ``torch.nn.functional`` entry points —
``F.linear``, ``F.rms_norm``, and ``F.scaled_dot_product_attention`` — with wrappers
that route to a hipDNN engine when the call meets the engine's constraints, and fall
back to stock PyTorch otherwise. The fallback is transparent (nothing breaks), counted,
and logged with a reason, so you get a ranked list of the operations and shapes hipDNN
still needs to serve.

Because the patch is applied to the functional entry points, no model code changes are
required: every ``nn.Linear``, ``nn.RMSNorm``, and ``F.scaled_dot_product_attention``
that resolves through ``torch.nn.functional`` at call time is intercepted.

Basic usage
===========

.. code:: python

    import hipdnn_torch

    hipdnn_torch.enable_logging()   # optional: print each native fallback
    hipdnn_torch.install()          # patch F.linear / F.rms_norm / F.scaled_dot_product_attention

    model(inputs)                   # your unmodified model

    print(hipdnn_torch.report())    # per-shape aot/native counts + why calls fell back
    hipdnn_torch.uninstall()

Importing the package does not import ``torch`` or touch the GPU; that happens lazily on
the first ``install()`` (or ``provider_ready()``), which is also where a misconfigured
environment raises a clear error naming the variable to set.

Environment setup
=================

The injection code is portable, but the runtime it plugs into is specific:

- A PyTorch build whose ROCm version matches the hipDNN backend it loads. The bootstrap
  ``dlopen``\ s PyTorch's own bundled ``libhipdnn_backend.so`` before importing the
  frontend to avoid the version-skew trap.
- The built engine plugin, pointed to by the required ``HIPDNN_TORCH_PROVIDER_SO``
  environment variable.
- On WSL2, the ``librocdxg`` GPU shim on ``LD_LIBRARY_PATH`` before ``torch``
  initializes CUDA.

Verify the environment is wired up without running a model:

.. code:: bash

    HIPDNN_TORCH_PROVIDER_SO=<build>/lib/hipdnn_plugins/engines/libhip_kernel_provider.so \
        python -c "import hipdnn_torch; print(hipdnn_torch.provider_ready())"

.. note::

  hipDNN is an early-release library, and default builds ship a deliberately limited set
  of providers and engines. The engine plugin this layer loads comes from
  ``hip-kernel-provider``, which is not part of the default or "supported" build presets,
  and the engine that serves ``F.linear`` / ``F.rms_norm`` /
  ``F.scaled_dot_product_attention`` is gated behind its own build option. As a result,
  ``provider_ready()`` can return ``True`` while nothing routes, simply because the loaded
  build doesn't yet cover these operations. Turning that coverage on takes a custom build;
  the ``hipdnn_torch`` README's "Getting a provider with the operations this layer needs"
  section has the recipe and links to a worked example.

Applicability and limitations
=============================

The injection only intercepts calls that go through ``torch.nn.functional``. Models
bypass it when they use ``torch.compile``/TorchInductor, custom fused attention
(FlashAttention, xformers, Triton, AITER), direct ``torch.ops.aten.*`` calls, quantized
paths, or non-PyTorch frameworks. Current engine coverage is limited to f16/bf16 on
constrained shapes (for example, SDPA requires ``B=1``, ``H=32``, ``D=64``).

Full reference
==============

The complete reference — the public API, the full environment-variable list, the exact
per-operation shape constraints, per-framework notes (ComfyUI, Hugging Face
Transformers, diffusers), the other ROCm attention backends, and the runnable samples —
lives in the ``hipdnn_torch`` README:

`projects/hipdnn/tools/hipdnn_torch/README.md <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/tools/hipdnn_torch/README.md>`_

Start with the self-contained ``samples/minimal_block.py``, which builds a tiny
transformer block, A/Bs it against native PyTorch, and prints the intercept report — no
model download or external checkout required.
