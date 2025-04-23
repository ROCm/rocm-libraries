.. meta::
  :description: MIGraphX internal data types
  :keywords: MIGraphX, code base, contribution, developing, data types

Data types
==========

shape
-----

.. doxygenstruct:: migraphx::internal::shape
   :members:
   :undoc-members:

literal
-------

.. doxygenstruct:: migraphx::internal::literal
   :members:
   :undoc-members:

argument
--------

.. doxygenstruct:: migraphx::internal::argument
   :members:
   :undoc-members:

raw_data
--------

.. doxygenstruct:: migraphx::internal::raw_data
   :members:
   :undoc-members:

.. doxygenfunction:: migraphx::internal::visit_all(T&& x, Ts&&... xs)
   
.. doxygenfunction:: migraphx::internal::visit_all(const std::vector<T>& x)

tensor_view
-----------

.. doxygenstruct:: migraphx::internal::tensor_view
   :members:
   :undoc-members:
