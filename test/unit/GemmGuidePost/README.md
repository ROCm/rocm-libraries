# GEMM Guideposts

This project's performance can be compared to output from the [Tensile Library](https://github.com/ROCmSoftwarePlatform/Tensile).  This directory has examples of kernels that have been generated from Tensile and converted to RocRoller's C++ instruction format.

The following steps are needed to generate a kernel from Tensile that can be run as a unit test.

## Running Tensile

1. Create a [YAML benchmark config file](https://github.com/ROCmSoftwarePlatform/Tensile/wiki/Benchmark-Config-example). You can model off of one in this directory.

2. Clone the [Tensile Library](https://github.com/ROCmSoftwarePlatform/Tensile).

3. Make sure you have the necessary requirements for Tensile (dependencies are listed [here](https://github.com/ROCmSoftwarePlatform/Tensile/wiki/Dependencies)):

    ```bash
    sudo apt update && \
    sudo apt install \
        libmsgpack-dev \
        libboost-program-options-dev \
        libboost-filesystem-dev
    ```

4. Run the following:

    ```bash
    <PathToTensile>/Tensile/bin/Tensile <PathToYAML> <PathToWorkingDirectory>
    ```

    > Note:
    > To get the inputs that Tensile is passing the kernel, `export TENSILE_DB=0x40` can be used.

5. The assembly file can be found as:

    ```
    <PathToWorkingDirectory>/1_BenchmarkProblems/<Identifier>/00_Final/source/build_tmp/SOURCE/assembly/<KernelIdentifier>.s
    ```

## Generating A C++ Version

6. Run the [remove_unused_macros](../../../scripts/remove_unused_macros.py) script and [assembly_to_instructions](../../../scripts/assembly_to_instructions.py) script on the assembly file to generate C++ for testing.  Don't forget to add the C++ file to the [CMakeLists.txt](../../../CMakeLists.txt).

    ```bash
    ./scripts/remove_unused_macros.py ./input.s > tmp.s && ./scripts/assembly_to_instructions.py --leave_comments --instruction_list --ignore_registers --remove_name_label --function_name="kernel" tmp.s > out.cpp
    ```
