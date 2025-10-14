.. meta::
  :description: hipSOLVER logging documentation
  :keywords: hipSOLVER, ROCm, API, documentation, logging, tracing

.. _logging-label:

******************************
hipSOLVER multi-level logging
******************************

hipSOLVER exposes rocSOLVER's logging functionality. Upon handle creation, hipSOLVER checks environment variables
using ``std::getenv`` upon handle creation.

You can expose hipSOLVER logging using the following environment variables for the respective default, refactor, and sparse APIs::

    HIPSOLVER_ENABLE_ROCSOLVER_LOGGING=1
    HIPSOLVER_REFACTOR_ENABLE_ROCSOLVER_LOGGING=1
    HIPSOLVER_SPARSE_ENABLE_ROCSOLVER_LOGGING=1

for the respective default, refactor, and sparse APIs.

Refer to the rocSOLVER logging documentation for more details on how to configure logging output.
