# Additional Sizes for Guideposts.

This project's performance can be compared to output from the [Tensile Library](https://github.com/ROCmSoftwarePlatform/Tensile).  This directory has examples of kernels that have been generated from Tensile and converted to RocRoller's C++ instruction format.

The following steps are needed to generate a kernel from Tensile that can be run as a unit test.

## Running Tensile

1. Use the yaml file from the 12544_1024_1_256 or the 7680_128_1_2560 subdirectories, or the Solution28.yaml file.

2. Clone the [Tensile Library](https://github.com/ROCmSoftwarePlatform/Tensile). Be sure to checkout the commit hash included as a comment in the yaml files contained in the directories mentioned in 1.

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

6. These yaml files will produce the kernels Cijk_Ailk_Bjlk_HHS_H_MT128x512x16_MI16x16x4x4_SE_K1_TT8_32 for the size 12544 x 1024 x 1 x 256,
Cijk_Ailk_Bjlk_HHS_H_MT128x128x16_MI32x32x8x1_SE_K1_TT4_32 for the size 7680 x 128 x 1 x 2560,
and Cijk_Ailk_Bjlk_HHS_BH_MT128x128x32_MI32x32x8x1_SE for the Solution28.yaml file.
