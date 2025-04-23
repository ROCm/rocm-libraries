# Custom rocBLAS Kernel using MIGraphX API. 
 This is an example of a custom operator implementation using MIGraphX's C/C++ APIs. It also demonstrates how to use this custom op in conjunction with rest of MIGraphX operators to build  and run MIGraphX program on GPU. 

 Kernels can be written in either HIP, MIOpen, or by using RocBLAS library.  This particular example uses **rocBLAS** library calls.

 To build and run the example, ensure ROCm is installed at `/opt/rocm`. 
 1.  `export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH`
 2.  `cd $MIGRAPHX_SRC/examples/migraphx/custom_op_rocblas_kernel/`
 3.  `mkdir build && cd build`
 4.  `cmake ..  && make`
 5.  `./custom_op_rocblas_kernel`
