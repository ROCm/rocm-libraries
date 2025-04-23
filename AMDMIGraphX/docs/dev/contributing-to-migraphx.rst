.. meta::
   :description: MIGraphX provides an optimized execution engine for deep learning neural networks
   :keywords: MIGraphX, ROCm, library, API

.. _contributing-to-migraphx:

==========================
Developing for MIGraphX
==========================

This document is intended for anyone who wants to contribute to MIGraphX. This document covers some basic operations that can be used to develop for MIGraphX. The complete source code for the example shown here can be found at `ref_dev_examples.cpp <https://github.com/ROCm/AMDMIGraphX/blob/develop/test/ref_dev_examples.cpp>`_ on the MIGraphX repository.

More examples can be found on `the MIGraphX GitHub repository <https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/tree/develop/examples/migraphx>`_.


Adding two literals
----------------------------

A program is a collection of modules, which are collections of instructions to be executed when calling :cpp:any:`eval <migraphx::internal::program::eval>`.
Each instruction has an associated :cpp:any:`operation <migraphx::internal::operation>` which represents the computation to be performed by the instruction.

We start with a snippet of the simple ``add_two_literals()`` function::


    // create the program and get a pointer to the main module
    migraphx::program p;
    auto* mm = p.get_main_module();

    // add two literals to the program
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);

    // make the add operation between the two literals and add it to the program
    mm->add_instruction(migraphx::make_op("add"), one, two);

    // compile the program on the reference device
    p.compile(migraphx::ref::target{});

    // evaulate the program and retreive the result
    auto result = p.eval({}).back();
    std::cout << "add_two_literals: 1 + 2 = " << result << "\n";

In the above function, a simple :cpp:any:`program <migraphx::internal::program>` object is created along with a pointer to the main module of it.
The program is a collection of modules which starts execution from the main module, so instructions are added to the modules rather than the program object directly.
The :ref:`add_literal <migraphx-module>` function is used to add an instruction that stores the literal number ``1`` while returning an :cpp:any:`instruction_ref <migraphx::internal::instruction_ref>`.
The returned :cpp:any:`instruction_ref <migraphx::internal::instruction_ref>` can be used in another instruction as an input.
The same :ref:`add_literal <migraphx-module>` function is used to add the literal ``2`` to the program.
After the literals are created, the instruction is created to add the numbers. This is done by using the :ref:`add_instruction <migraphx-module>` function with the ``add`` :cpp:any:`operation <migraphx::internal::operation>` created by ``make_op`` and the previously created literals passed as the arguments for the instruction.
You can run this :cpp:any:`program <migraphx::internal::program>` by compiling it for the reference target (CPU) and then running it with :cpp:any:`eval <migraphx::internal::program::eval>`. This prints the result on the console.

To compile the program for the GPU, move the file to ``test/gpu/`` directory and include the given target::

    #include <migraphx/gpu/target.hpp>

Adding Parameters
----------------------------

While the ``add_two_literals()`` function above demonstrates add operation on constant values ``1`` and ``2``,
the following program demonstrates how to pass a parameter (``x``) to a module using ``add_parameter()`` function .

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {1}};

    // add parameter "x" with the shape s
    auto x   = mm->add_parameter("x", s);
    auto two = mm->add_literal(2);

    // add the "add" instruction between the "x" parameter and "two" to the module
    mm->add_instruction(migraphx::make_op("add"), x, two);
    p.compile(migraphx::ref::target{});

In the code snippet above, an add operation is performed on a parameter of type ``int32`` and literal ``2`` followed by compilation for the CPU.
To run the program, pass the parameter as a ``parameter_map`` while calling :cpp:any:`eval <migraphx::internal::program::eval>`.
To map the parameter ``x`` to an :cpp:any:`argument <migraphx::internal::argument>` object with an ``int`` data type, a ``parameter_map`` is created as shown below::

    // create a parameter_map object for passing a value to the "x" parameter
    std::vector<int> data = {4};
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(params).back();
    std::cout << "add_parameters: 4 + 2 = " << result << "\n";
    EXPECT(result.at<int>() == 6);

Handling Tensor Data
----------------------------

The above two examples demonstrate scalar operations. To describe multi-dimensional tensors, use the :cpp:any:`shape <migraphx::internal::shape>` class to compute a simple convolution as shown below::

    migraphx::program p;
    auto* mm = p.get_main_module();

    // create shape objects for the input tensor and weights
    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {3, 3, 3, 3}};

    // create the parameters and add the "convolution" operation to the module
    auto input   = mm->add_parameter("X", input_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}}), input, weights);

Most programs take data from allocated buffers that are usually on the GPU. To pass the buffer data as an argument, create :cpp:any:`argument <migraphx::internal::argument>` objects directly from the pointers to the buffers::

    // Compile the program
    p.compile(migraphx::ref::target{});

    // Allocated buffers by the user
    std::vector<float> a = ...;
    std::vector<float> c = ...;

    // Solution vector
    std::vector<float> sol = ...;

    // Create the arguments in a parameter_map
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_shape, a.data());
    params["W"] = migraphx::argument(weights_shape, c.data());

    // Evaluate and confirm the result
    auto result = p.eval(params).back();
    std::vector<float> results_vector(64);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(results_vector, sol));

An :cpp:any:`argument <migraphx::internal::argument>` can handle memory buffers from either the GPU or the CPU.
When running the :cpp:any:`program <migraphx::internal::program>`, buffers are allocated on the corresponding target by default.
By default, the buffers are allocated on the CPU when compiling for CPU and on the GPU when compiling for GPU.
To locate the buffers on the CPU even when compiling for GPU, set the option ``offload_copy=true``.

Importing From ONNX
----------------------------

To make it convenient to use neural networks directly from other frameworks, MIGraphX ONNX parser allows you to build a :cpp:any:`program <migraphx::internal::program>` directly from an ONNX file.
For usage, refer to the ``parse_onnx()`` function below::

    program p = migraphx::parse_onnx("model.onnx");
    p.compile(migraphx::gpu::target{});


Build this example
----------------------------

Build the `ref_dev_examples.cpp <https://github.com/ROCm/AMDMIGraphX/blob/develop/test/ref_dev_examples.cpp>`_ example with this command:

    make -j$(nproc) test_ref_dev_examples

This creates the ``test_ref_dev_examples`` under ``bin/`` in the build directory.

To verify the build, use:

    make -j$(nproc) check
