.. list-table::
    :header-rows: 1
    :widths: 50,50

    * - **Environment variable**
      - **Value**

    * - | ``ROCBLAS_LAYER``
        | A bit mask to control the different types of logging.
      - | :code:`ROCBLAS_LAYER == 0`: Logging is disabled.
        | :code:`ROCBLAS_LAYER & 1 == 1`: Trace logging is enabled.
        | :code:`ROCBLAS_LAYER & 2 == 1`: Bench logging is enabled.
        | :code:`ROCBLAS_LAYER & 4 == 1`: Profile logging is enabled.
        | :code:`ROCBLAS_LAYER & 8 == 1`: Internal API logging is enabled.

    * - | ``ROCBLAS_LOG_PATH``
        | Sets the full path for logging.
      - Example: :code:`$PWD/logging.txt`

    * - | ``ROCBLAS_LOG_TRACE_PATH``
        | Specifies the full path for trace logging. If this environment variable is set, the ``ROCBLAS_LOG_PATH`` environment variable is ignored for trace logs.
      - Example: :code:`$PWD/trace_logging.txt`

    * - | ``ROCBLAS_LOG_BENCH_PATH``
        | Specifies the full path for bench logging. If this environment variable is set, the ``ROCBLAS_LOG_PATH`` environment variable is ignored for bench logs.
      - Example: :code:`$PWD/bench_logging.txt`

    * - | ``ROCBLAS_LOG_PROFILE_PATH``
        | Specifies the full path for profile logging. If this environment variable is set, the ``ROCBLAS_LOG_PATH`` environment variable is ignored for profile logs.
      - Example: :code:`$PWD/profile_logging.txt`
