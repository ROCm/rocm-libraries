MIGraphX GPU-driver
===============

The MIGraphX gpu-driver is used to test the performance of GPU kernels produced in MIGraphX.
Example usage::
    
    gpu-driver kernel_test.json

The input json file describes the gpu kernel and the input data settings.
Random data is passed into the gpu kernel and the kernel is run a set number of iterations
(1000 in the example below) and timed for performance.

Format for the input json file
------------------------------

* settings:
    * iterations: the number of iterations to run the kernel
    * lens: the dimensions for the input data shape
    * strides: strides for the input dimension, optional for standard shapes
    * type: data type
* compile_op:
    * name: name of the gpu operation to compile
    * lambda: lambda function
    * inputs: input shapes into the kernel, need 1 more than lambda function input for output buffer

*TODO: many other possible settings*

Example hjson file that tests a pointwise GELU approximation (note this is hjson and needs
to be converted to json before usage)::

    # sigmoid GELU approximation
    {
        settings: {
            iterations: 1000
            lens: [10, 384, 3072]
            type: "float"
        },
        compile_op: {
            name: "pointwise"
            lambda: 
            '''
            [](auto x)
            {
                using x_type = decltype(x);
                x_type one = 1.;
                x_type fit_const = 2.;
                return x / (one + exp(-fit_const * x));
            }
            '''
            inputs: [{}, {}]
        }
    }

To convert the hjson file to a json file you can use ``hjson -j``. To install hjson: ``pip install hjson``
