The hipSPARSELt library is organized as follows:
<!-- spellcheck-disable -->

* @ref types_module
* @ref library_module
* @ref matrix_desc_module
* @ref matmul_module
* @ref matmul_desc_module
* @ref matmul_algo_module
* @ref helper_module
* @ref aux_module

<!-- spellcheck-enable -->
**Note:** Unless otherwise stated, all hipSPARSELt library functions are
non-blocking and run asynchronously with respect to the host. They might
return before the actual computation has finished. To force synchronization, use
`hipDeviceSynchronize` or `hipStreamSynchronize`.
