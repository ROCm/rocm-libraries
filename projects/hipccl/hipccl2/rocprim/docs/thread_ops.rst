Thread-Level Operations (Utilities)
===================================

Scan
----

exclusive
.........

.. doxygenfunction:: thread_scan_exclusive(T (&input)[LENGTH], T (&output)[LENGTH], ScanOp scan_op, T prefix, bool apply_prefix = true)
.. doxygenfunction:: thread_scan_exclusive(T *input, T *output, ScanOp scan_op, T prefix, bool apply_prefix = true)
.. doxygenfunction:: thread_scan_exclusive(T inclusive, T exclusive, T *input, T *output, ScanOp scan_op, Int2Type<LENGTH>)

inclusive
.........

.. doxygenfunction:: thread_scan_inclusive (T inclusive, T *input, T *output, ScanOp scan_op, Int2Type< LENGTH >)
.. doxygenfunction:: thread_scan_inclusive (T *input, T *output, ScanOp scan_op)
.. doxygenfunction:: thread_scan_inclusive (T(&input)[LENGTH], T(&output)[LENGTH], ScanOp scan_op)
.. doxygenfunction:: thread_scan_inclusive (T *input, T *output, ScanOp scan_op, T prefix, bool apply_prefix=true)
.. doxygenfunction:: thread_scan_inclusive (T(&input)[LENGTH], T(&output)[LENGTH], ScanOp scan_op, T prefix, bool apply_prefix=true)
