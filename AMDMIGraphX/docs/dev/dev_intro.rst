.. meta::
  :description: MIGraphX introduction to developing for the code base
  :keywords: MIGraphX, code base, contribution, developing, introduction, developers


Developer Introduction
======================

MIGraphX provides an optimized execution engine for deep learning neural networks.

We will cover some simple operations in the MIGraphX framework here.
For a quick start guide to using MIGraphX, look in the examples directory: ``https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/tree/develop/examples/migraphx``.


Location of the Examples
-------------------------

The ``ref_dev_examples.cpp`` can be found in the test directory (``/test``).
The executable file ``test_ref_dev_examples`` based on this file will be created in the ``bin/`` of the build directory after running ``make -j$(nproc) test_ref_dev_examples``.
The executable will also be created when running ``make -j$(nproc) check``, alongside with all the other tests.
Directions for building MIGraphX from source can be found in the main README file: ``https://github.com/ROCmSoftwarePlatform/AMDMIGraphX#readme``.


Adding Two Literals
--------------------

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

We start by creating a simple :cpp:any:`migraphx::program <migraphx::internal::program>` object and then getting a pointer to the main module of it.
The program is a collection of ``modules`` that start executing from the main module, so instructions are added to the modules rather than directly onto the program object.
We then use the :cpp:any:`add_literal <migraphx::internal::program::add_literal>` function to add an instruction that stores the literal number ``1`` while returning an :cpp:any:`instruction_ref <migraphx::internal::instruction_ref>`.
The returned :cpp:any:`instruction_ref <migraphx::internal::instruction_ref>` can be used in another instruction as an input.
We use the same :cpp:any:`add_literal <migraphx::internal::program::add_literal>` function to add a ``2`` to the program.
After creating the literals, we then create the instruction to add the numbers together.
This is done by using the :cpp:any:`add_instruction <migraphx::internal::program::add_instruction>` function with the ``"add"`` :cpp:any:`operation <migraphx::internal::program::operation>` created by :cpp:any:`make_op <migraphx::internal::program::make_op>` along with the previous `add_literal` :cpp:any:`instruction_ref <migraphx::internal::instruction_ref>` for the input arguments of the instruction.
Finally, we can run this :cpp:any:`program <migraphx::internal::program>` by compiling it for the reference target (CPU) and then running it with :cpp:any:`eval <migraphx::internal::program::eval>`
The result is then retreived and printed to the console.

We can compile the program for the GPU as well, but the file will have to be moved to the ``test/gpu/`` directory and the correct target must be included::

    #include <migraphx/gpu/target.hpp>


Using Parameters
-----------------

The previous program will always produce the same value of adding ``1`` and ``2``.
In the next program we want to pass an input to a program and compute a value based on the input.
We can modify the program to take an input parameter ``x``, as seen in the ``add_parameter()`` function::

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {1}};

    // add a "x" parameter with the shape s
    auto x   = mm->add_parameter("x", s);
    auto two = mm->add_literal(2);

    // add the "add" instruction between the "x" parameter and "two" to the module
    mm->add_instruction(migraphx::make_op("add"), x, two);
    p.compile(migraphx::ref::target{});

This adds a parameter of type ``int32``, and compiles it for the CPU.
To run the program, we need to pass the parameter as a ``parameter_map`` when we call :cpp:any:`eval <migraphx::internal::program::eval>`.
We create the ``parameter_map`` by setting the ``x`` key to an :cpp:any:`argument <migraphx::internal::argument>` object with an ``int`` data type::

    // create a parameter_map object for passing a value to the "x" parameter
    std::vector<int> data = {4};
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(params).back();
    std::cout << "add_parameters: 4 + 2 = " << result << "\n";
    EXPECT(result.at<int>() == 6);


Handling Tensor Data
---------------------

In the previous examples we have only been dealing with scalars, but the :cpp:any:`shape <migraphx::internal::shape>` class can describe multi-dimensional tensors.
For example, we can compute a simple convolution::

    migraphx::program p;
    auto* mm = p.get_main_module();

    // create shape objects for the input tensor and weights
    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {3, 3, 3, 3}};

    // create the parameters and add the "convolution" operation to the module
    auto input   = mm->add_parameter("X", input_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}}), input, weights);

Here we create two parameters for both the ``input`` and ``weights``.
In the previous examples, we created simple literals, however, most programs will take data from allocated buffers (usually on the GPU).
In this case, we can create :cpp:any:`argument <migraphx::internal::argument>` objects directly from the pointers to the buffers::

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
By default when running the :cpp:any:`program <migraphx::internal::program>`, buffers are allocated on the corresponding target.
When compiling for the CPU, the buffers by default will be allocated on the CPU.
When compiling for the GPU, the buffers by default will be allocated on the GPU.
With the option ``offload_copy=true`` set while compiling for the GPU, the buffers will be located on the CPU.


Importing From ONNX
--------------------

A :cpp:any:`program <migraphx::internal::program>` can be built directly from an onnx file using the MIGraphX ONNX parser.
This makes it easier to use neural networks directly from other frameworks.
In this case, there is an ``parse_onnx`` function::

    program p = migraphx::parse_onnx("model.onnx");
    p.compile(migraphx::gpu::target{});

