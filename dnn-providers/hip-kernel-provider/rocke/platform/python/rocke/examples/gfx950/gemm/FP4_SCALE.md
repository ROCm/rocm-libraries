# gfx950 FP4 and E8M0 storage

FP4 E2M1 (`fp4`) stores two consecutive logical K values per byte, first code
in the low nibble. Each row occupies K/2 bytes. The 16 codes cover sign, two
exponent bits and one mantissa bit; finite magnitudes extend through 6.
Packing accepts codes rather than imposing a quantization policy.

The native atom consumes 32 values per lane in four live i32 words, padded
with four zero words for the LLVM ABI. E8M0 scale tensors are separate byte
arrays, A_scale[M,K/32] and B_scale[K/32,N]. Every raw byte scales 32 logical
values; it is zero-extended into a separate A/B i32 VGPR operand. The examples
use 1/4, 1/2, 2 and 4 and an independent decoded oracle. See [the common
contract and commands](README.md).
