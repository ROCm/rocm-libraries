.. meta::
  :description: How to use hipDNN forwarding in MIOpen
  :keywords: MIOpen, ROCm, API, documentation, hipDNN, forwarding, convolution

********************************************************************
Use hipDNN forwarding
********************************************************************

hipDNN forwarding lets MIOpen pass some convolution calls to
`hipDNN <https://rocm.docs.amd.com/projects/hipdnn/en/latest/>`_ without any change to your code.
Your program still calls the MIOpen API and links against ``libMIOpen.so``. MIOpen decides whether
to run each call itself or pass it to hipDNN.

.. note::

  In this release, hipDNN forwarding is a preview feature and is off by default. Only the
  functions listed in :ref:`hipdnn-forwarded-functions` are forwarded. Forwarding is not supported
  for any other function.

How forwarding is controlled
============================

Forwarding has one build setting and two run-time settings:

.. list-table::
    :header-rows: 1
    :widths: 35,15,50

    * - **Setting**
      - **Default**
      - **What it does**

    * - | ``MIOPEN_ENABLE_HIPDNN_WRAPPER``
        | CMake option
      - ``Off``
      - Builds MIOpen with forwarding support.

    * - | ``MIOPEN_HIPDNN_FORWARDING``
        | Environment variable
      - Off
      - Turns forwarding on.

    * - | ``MIOPEN_DISABLE_HIPDNN_FOR``
        | Environment variable
      - Not set
      - Keeps the listed functions in MIOpen, even when forwarding is on.

Keep in mind:

* The environment variables do nothing unless MIOpen was built with
  ``MIOPEN_ENABLE_HIPDNN_WRAPPER=On``.
* Building with the option doesn't turn forwarding on. Until you set ``MIOPEN_HIPDNN_FORWARDING``,
  MIOpen runs every call itself.
* MIOpen reads the environment variables once per process. Changes take effect the next time your
  program starts, without rebuilding or reinstalling anything.

.. _hipdnn-forwarded-functions:

Forwarded functions
===================

When forwarding is on, MIOpen passes these functions to hipDNN:

* ``miopenConvolutionForward``
* ``miopenConvolutionBackwardData``
* ``miopenConvolutionBackwardWeights``
* ``miopenConvolutionBiasActivationForward``

All other MIOpen functions always run in MIOpen.

Turn forwarding on
==================

#. Build MIOpen with ``-DMIOPEN_ENABLE_HIPDNN_WRAPPER=On``. hipDNN must be installed where CMake can
   find it, for example through ``CMAKE_PREFIX_PATH``:

   .. code:: shell

      cmake -DMIOPEN_BACKEND=HIP -DMIOPEN_ENABLE_HIPDNN_WRAPPER=On ..

   Check that the CMake output includes a line starting with
   ``-- MIOPEN_ENABLE_HIPDNN_WRAPPER: ENABLED``. Then build and install MIOpen as usual. For what
   this build installs, see :ref:`build-hipdnn-forwarding`.

#. Make sure hipDNN is installed on the machine that runs your program. MIOpen finds it on its own
   when ``libhipdnn_backend.so`` is in the same directory as ``libMIOpen.so``, such as
   ``/opt/rocm/lib``. Otherwise, set ``HIPDNN_BACKEND_LIBRARY_PATH`` to the absolute path of the
   directory that holds it:

   .. code:: shell

      export HIPDNN_BACKEND_LIBRARY_PATH=/path/to/hipdnn/lib

#. Turn forwarding on, then run your program:

   .. code:: shell

      export MIOPEN_HIPDNN_FORWARDING=1

   ``1``, ``on``, and ``enabled`` all work. When forwarding is on, MIOpen prints a line starting
   with ``[MIOpen] MIOPEN_HIPDNN_FORWARDING=`` to ``stderr``. If the line is missing, MIOpen was
   built without forwarding support. If MIOpen doesn't recognize the value, it prints a warning
   and leaves forwarding off.

#. Optional: to see where each function runs, also set ``MIOPEN_LOG_LEVEL=6``. The first time each
   function is called, MIOpen prints a line like this:

   .. code:: text

      [MIOpen] hipDNN routing: miopenConvolutionForward mode=enabled route=hipdnn reason=in-forwarding-set

   For all the lines MIOpen prints, see :ref:`logging-hipdnn-forwarding`.

Keep some functions in MIOpen
=============================

To keep some functions in MIOpen while the rest are forwarded, list them in
``MIOPEN_DISABLE_HIPDNN_FOR``, separated by commas:

.. code:: shell

   export MIOPEN_DISABLE_HIPDNN_FOR=miopenConvolutionBackwardData,miopenConvolutionBackwardWeights

Names are case-sensitive. MIOpen warns about, and ignores, any name that isn't a forwarded function.

Turn forwarding off
===================

Either of these sends every call back to MIOpen the next time your program starts:

* Unset ``MIOPEN_HIPDNN_FORWARDING``, or set it to ``0``:

  .. code:: shell

     export MIOPEN_HIPDNN_FORWARDING=0

* Leave forwarding on, but keep every function in MIOpen:

  .. code:: shell

     export MIOPEN_DISABLE_HIPDNN_FOR='*'

What changes when a call is forwarded
=====================================

* **No automatic fallback.** If hipDNN can't run a problem, the call fails with
  ``miopenStatusUnsupportedOp`` or ``miopenStatusNotImplemented`` instead of running in MIOpen. To
  run that function in MIOpen, add it to ``MIOPEN_DISABLE_HIPDNN_FOR``.
* **hipDNN must load.** If hipDNN can't be loaded, every forwarded call fails with
  ``miopenStatusInternalError``. MIOpen prints the reason once, unless ``MIOPEN_LOG_LEVEL`` hides
  errors.
* **The algorithm argument is ignored.** hipDNN picks its own algorithm.
* **Workspace.** If the workspace you pass is too small for hipDNN, MIOpen allocates its own with
  ``hipMalloc`` and keeps it until ``miopenDestroy``. That memory doesn't come from the allocator
  set with ``miopenSetAllocator``, and it doesn't work with HIP stream capture.

Tell hipDNN failures apart from MIOpen failures
===============================================

A forwarded call that fails returns an ordinary ``miopenStatus_t``. To find out whether the failure
came from hipDNN:

* ``miopenGetErrorString`` adds ``[hipDNN-forwarded]`` to the start of its message when the status
  is the one the calling thread's last MIOpen call failed with in hipDNN.
* ``miopenGetLastForwardedError`` gives the status of the calling thread's last MIOpen call if that
  call was forwarded and failed, and ``miopenStatusSuccess`` otherwise.

Calling either function doesn't change the result. The next call to any other MIOpen function on
the same thread does.

``miopenGetLastForwardedError`` is a beta API declared in ``<miopen/hipdnn_forwarding.h>``, so
define ``MIOPEN_BETA_API`` before including the header. The header and function exist only in
builds with ``MIOPEN_ENABLE_HIPDNN_WRAPPER=On``. Code that must also build against other MIOpen
installations can check for the header:

.. code:: cpp

  #define MIOPEN_BETA_API
  #include <miopen/miopen.h>
  #if __has_include(<miopen/hipdnn_forwarding.h>)
  #include <miopen/hipdnn_forwarding.h>
  #define HAVE_MIOPEN_FORWARDED_ERROR 1
  #endif

  miopenStatus_t status = miopenConvolutionForward(/* ... */);
  #ifdef HAVE_MIOPEN_FORWARDED_ERROR
  miopenStatus_t forwarded = miopenStatusSuccess;
  miopenGetLastForwardedError(&forwarded);
  if(forwarded != miopenStatusSuccess)
  {
      // The call was forwarded to hipDNN and failed there.
  }
  #endif

If a program built this way against an MIOpen with forwarding support runs with a
``libMIOpen.so`` built without it, the program may fail to load, or fail on its first call to
``miopenGetLastForwardedError``. To work with both without rebuilding, look up
``miopenGetLastForwardedError`` at run time with ``dlsym`` instead.
