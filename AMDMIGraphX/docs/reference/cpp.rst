.. meta::
   :description: MIGraphX C++ API reference
   :keywords: MIGraphX, ROCm, C++, API, reference, development, developer

.. _cpp-api-reference:

C++ API reference
====================

shape
-----

.. doxygenenum:: migraphx_shape_datatype_t

.. doxygenstruct:: migraphx::shape
   :members:
   :undoc-members:

argument
--------

.. doxygenstruct:: migraphx::argument
   :members:
   :undoc-members:

target
------

.. doxygenstruct:: migraphx::target
   :members:
   :undoc-members:

program
-------

.. doxygenstruct:: migraphx::program_parameter_shapes
   :members:
   :undoc-members:

.. doxygenstruct:: migraphx::program_parameters
   :members:
   :undoc-members:

.. doxygenstruct:: migraphx_compile_options
   :members:
   :undoc-members:

.. doxygenstruct:: migraphx::program
   :members:
   :undoc-members:

quantize
--------

.. doxygenstruct:: migraphx::quantize_op_names
   :members:
   :undoc-members:

.. doxygenfunction:: migraphx::quantize_fp16(const program&)

.. doxygenfunction:: migraphx::quantize_fp16(const program&, const quantize_op_names&)

.. doxygenfunction:: migraphx::quantize_bf16(const program&)

.. doxygenstruct:: migraphx::quantize_int8_options
   :members:
   :undoc-members:

.. doxygenfunction::migraphx::quantize_int8

parse_onnx
----------

.. doxygenstruct:: migraphx::onnx_options
   :members:
   :undoc-members:

.. doxygenfunction:: migraphx::parse_onnx(const char *)

.. doxygenfunction:: migraphx::parse_onnx(const char *, const migraphx::onnx_options&)

.. doxygenfunction:: migraphx::parse_onnx_buffer(const std::string&)

.. doxygenfunction:: migraphx::parse_onnx_buffer(const std::string&, const migraphx::onnx_options&)

.. doxygenfunction:: migraphx::parse_onnx_buffer(const void *, size_t)

.. doxygenfunction:: migraphx::parse_onnx_buffer(const void *, size_t, const migraphx::onnx_options&)

load
----

.. doxygenstruct:: migraphx::file_options
   :members:
   :undoc-members:

.. doxygenfunction:: migraphx::load(const char *)

.. doxygenfunction:: migraphx::load(const char *, const file_options&)

save
----

.. doxygenfunction:: migraphx::save(const program&, const char *)

.. doxygenfunction:: migraphx::save(const program&, const char *, const file_options&)

