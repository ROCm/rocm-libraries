# rocRoller GEMM Selection

Running a GEMM using rocRoller follows this process:

User provides the problem, which consists of the `KernelType` and the input parameters.

Given the problem, Origami is used to help select `SolutionIndexParameters`. These are the parameters that
are unique between each kernel. Origami is used to select the tile size, but `SolutionIndexParameters`
can contain other parameters besides the tile size. A list of `SolutionIndexParameters` are returned
in sorted order, based on what Origami predicts is the best performing kernel.

If a kernel has already been generated for the specific `SolutionIndexParameters` instance that was selected,
the kernel can be found in the cache and returned.

Otherwise, the rest of the `SolutionParameters` need to be selected.
`SolutionParameters` contain all of the parameters required to generate a kernel. These parameters
are selected based on the `KernelType` and the `SolutionIndexParameters`.

Once all of the `SolutionParameters` have been selected, the kernel is generated using rocRoller. The kernel
is then saved for reuse in the cache and returned.

# Calling a rocRoller Kernel

Once a kernel has been selected, it is ready to be called. Most of the inputs to the kernel will come from
the user provided input parameters. However, there are some runtime parameters that need to be selected as
well (such as StreamK grid size). These runtime parameters need to be selected before calling the kernel.

Once all of the runtime parameters have been selected, the kernel can be called.
