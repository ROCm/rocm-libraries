# Packaged gfx1250 forward convolution

`hipkernel:Gfx1250ConvFwdAot` runs rocKE's gfx1250 forward convolution kernels
(wave32 WMMA direct kernels, VALU depthwise kernels and implicit GEMM) through the
HIP Kernel Provider. It accepts one 2D cross-correlation with x, w and y in the same
FP16 or BF16 type, FP32 accumulation, packed channels-last storage (NHWC input, KYXC
filter, NHWK output), symmetric padding, every tensor within the kernels' i32 byte
sizes, and the padded input and dilated filter extents within signed 32 bits. x, w and
y must be 16-byte aligned (a multiple of 16) and not ragged. Grouped convolution,
depthwise included, is supported up to 65535 groups, as on gfx950: the group count is
inferred as X channels divided by W dimension 1. Every other device is declined
(architecture, warp size and compute type are checked).

It is the gfx1250 sibling of [`hipkernel:Gfx950ConvFwdAot`](../gfx950_conv_fwd/README.md)
and shares its code: the packaging adapter
(`rocke/library/builders/common/convolution_forward.py`, `TARGETS["gfx1250"]`), the
IngestorGenerator tools (`--arch gfx1250`) and the native pack
(`packs/ConvFwdAotNative.cpp`, instantiated per engine). What differs is the catalog
and its launch: each gfx1250 instance is one entry of rocKE's gfx1250 forward
dispatcher list, and the pack serves and launches it with a C++ port of that
dispatcher (`packs/Gfx1250ConvFwd*.hpp`).

See [AOT instances](aot_instances.md) for the instance fields, which graphs an instance
serves, the shipped catalog and how it is regenerated, ranking, and the tests.
