# gfx950 FP6 storage

FP6 E2M3 (`fp6`) and E3M2 (`bf6`) use six-bit codes. Four consecutive logical
K values occupy three little-endian bytes, with the first code in the low six
bits. Each row occupies K*6/8 bytes. The shared packer writes codes; it does
not define floating-point quantization. The independent reference decodes bits
using the exponent/mantissa formula, including all 64 finite codes and signed
zeros. Focused GPU tests isolate each end of every K32 group and exercise all
64x64 code products for both homogeneous and mixed FP6/BF6 pairs.

The native atom consumes 32 values per lane in six live i32 words, padded with
two zero words for the LLVM ABI. One raw E8M0 byte belongs to each row/column
and K32 group. See [the common contract and commands](README.md).
