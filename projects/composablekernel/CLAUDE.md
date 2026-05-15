## Project description

We want to port the HIP conv kernels located in `projects/miopen/src/hipconv` to create CK Tile direct convoution library.

The CK Tile direct convolution library has README file located at `projects/composablekernel/include/ck_tile/ops/direct_convolution/README.md`. This is the main source for the project documentation and we should update it once we have concreate changes ready.

Note that the original HIP conv project uses notation 
- NHWC input
- KRSC weights
- NPQK output

## Development environment

You can assume that all required development tools are already installed.

## Project documentation

The entrypoint is is the project Readme file `projects/composablekernel/include/ck_tile/ops/direct_convolution/README.md`. It contains link to the detailed documentation of the different components. This detailed documentation is located at 
`projects/composablekernel/docs/direct_convolution` and it contains the mathematical and technical details of different components.
We should update this documentation as we go. 

## Supported device architectures

The original HIP conv target `gfx950` device architecture. One of the goals of this project is to extend the coverage to `gfx942`.

