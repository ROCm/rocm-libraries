# TensileLite Components

## Signature

The signature for each KernArgsVersion is described here.

### Version 2

```
[0..3] gemm_count: 01 00 00 00 (1)
[4..7] internalArgs: 02 00 20 20 (538968066)
[8..11] internalArgs1: f8 ff ff ff (-8)
[12..15] numWorkGroups: 2c 01 00 00 (300)
# Above are universal args
# Below are problem type specific parameters
[16..19] size_0: 00 05 00 00 (1280)
[20..23] size_1: c0 03 00 00 (960)
[24..27] size_2: 01 00 00 00 (1)
[28..31] size_3: 00 20 00 00 (8192)
[32..39] ws_d: 00 00 80 b9 68 7f 00 00 (0x7f68b9800000)
[40..47] c: 00 be 98 c4 68 7f 00 00 (0x7f68c498be00)
[48..55] a: 00 00 60 c2 68 7f 00 00 (0x7f68c2600000)
[56..63] b: 00 ff a4 c3 68 7f 00 00 (0x7f68c3a4ff00)
[64..67] strideW_D1: 00 05 00 00 (1280)
[68..71] strideW_D2: 00 c0 12 00 (1228800)
[72..75] strideW_C1: 00 05 00 00 (1280)
[76..79] strideW_C2: 00 c0 12 00 (1228800)
[80..83] strideA1: 80 20 00 00 (8320)
[84..87] strideA2: 00 80 a2 00 (10649600)
[88..91] strideB1: 80 20 00 00 (8320)
[92..95] strideB2: 00 e0 79 00 (7987200)
[96..99] alpha: 00 00 80 3f (1)
[100..103] beta: 00 00 00 00 (0)
[104..111] dstD: 00 3e be c4 68 7f 00 00 (0x7f68c4be3e00)
[112..119] Synchronizer: 00 be e3 c4 68 7f 00 00 (0x7f68c4e3be00)
[120..123] GSUSync: 00 00 00 00 (0)
```
#### internalArgs

1. 2-bit input type
   1. 0 normal
   2. 1 hbm
   3. 2 user allocated input
2. 16-bit StaggerU
   1. 3-bit staggerUMapping
   2. 5-bit staggerUShift
   3. 8-bit staggerU
3. 16-bit GSU control info
   1. 1-bit GSUC
   2. 1-bit GSUWGMRR
   3. 14-bit GSU
4. 32-bit WGM control info
   1. 10-bit WGMXCCG
   2. 6-bit WGMXCC
   3. 16-bit WGM (with negative support)

### Version 1

```
[0..3] gemm_count: 01 00 00 00 (1)
[4..7] internalArgs: 02 00 20 20 (538968066)
[8..11] internalArgs1: f8 ff ff ff (-8)
[12..15] numWorkGroups: 2c 01 00 00 (300)
# Above are universal args
# Below are problem type specific parameters
[16..19] size_0: 00 05 00 00 (1280)
[20..23] size_1: c0 03 00 00 (960)
[24..27] size_2: 01 00 00 00 (1)
[28..31] size_3: 00 20 00 00 (8192)
[32..39] ws_d: 00 00 80 b9 68 7f 00 00 (0x7f68b9800000)
[40..47] c: 00 be 98 c4 68 7f 00 00 (0x7f68c498be00)
[48..55] a: 00 00 60 c2 68 7f 00 00 (0x7f68c2600000)
[56..63] b: 00 ff a4 c3 68 7f 00 00 (0x7f68c3a4ff00)
[64..67] strideW_D1: 00 05 00 00 (1280)
[68..71] strideW_D2: 00 c0 12 00 (1228800)
[72..75] strideW_C1: 00 05 00 00 (1280)
[76..79] strideW_C2: 00 c0 12 00 (1228800)
[80..83] strideA1: 80 20 00 00 (8320)
[84..87] strideA2: 00 80 a2 00 (10649600)
[88..91] strideB1: 80 20 00 00 (8320)
[92..95] strideB2: 00 e0 79 00 (7987200)
[96..99] alpha: 00 00 80 3f (1)
[100..103] beta: 00 00 00 00 (0)
[104..111] dstD: 00 3e be c4 68 7f 00 00 (0x7f68c4be3e00)
[112..119] Synchronizer: 00 be e3 c4 68 7f 00 00 (0x7f68c4e3be00)
[120..123] GSUSync: 00 00 00 00 (0)
```
#### internalArgs

1. 2-bit input type
   1. 0 normal
   2. 1 hbm
   3. 2 user allocated input
2. 16-bit StaggerU
   1. 3-bit staggerUMapping
   2. 5-bit staggerUShift
   3. 8-bit staggerU
3. 16-bit GSU
4. 32-bit WGM (with negative support)

Note1: Currently internalArgs1 is empty so we keep the WGM as 32-bit
Note2: The workgroup for gemm is changed to 1D for WGMXCC calculation. A total number of workgroups is also added (``numWorkGroups``) to the input.

### Pre-version (0)

1. Introducing universal arguments
2. Add GSU
3. Add WGM
4. Add SU related parameters

```
[0..3] gemm_count: 01 00 00 00 (1)
[4..7] internalArgs: 01 08 20 01 (18876417)
# Above are universal args
# Below are problem type specific parameters
[8..11] size_0: ff 00 00 00 (255)
[12..15] size_1: ff 00 00 00 (255)
[16..19] size_2: 01 00 00 00 (1)
[20..23] size_3: ff 00 00 00 (255)
[24..31] d: 00 00 a4 9c 46 7f 00 00 (0x7f469ca40000)
[32..39] c: 00 20 fc 9c 46 7f 00 00 (0x7f469cfc2000)
[40..47] a: 00 00 f0 9c 46 7f 00 00 (0x7f469cf00000)
[48..55] b: 00 10 f6 9c 46 7f 00 00 (0x7f469cf61000)
[56..59] strideD1: ff 00 00 00 (255)
[60..63] strideD2: 01 fe 00 00 (65025)
[64..67] strideC1: ff 00 00 00 (255)
[68..71] strideC2: 01 fe 00 00 (65025)
[72..75] strideA1: ff 00 00 00 (255)
[76..79] strideA2: 01 fe 00 00 (65025)
[80..83] strideB1: ff 00 00 00 (255)
[84..87] strideB2: 01 fe 00 00 (65025)
[88..91] alpha: 00 00 80 3f (1)
[92..95] beta: 00 00 00 00 (0)
[96..103] scaleAlphaVec: 00 80 fe 9c 46 7f 00 00 (0x7f469cfe8000)
[104..111] bias: 00 40 fe 9c 46 7f 00 00 (0x7f469cfe4000)
[112..115] bias_type: 00 00 00 00 (0)
[116..119] strideBias: 00 00 00 00 (0)
[120..123] activation_0: 00 00 00 40 (2)
[124..127] activation_1: 00 00 00 40 (2)
[128..131] activationType: 05 00 00 00 (5)
```

#### internalArgs

1. 2 bit input type
   1. 0 normal
   2. 1 hbm
   3. 2 user allocated input
2. 16-bit StaggerU
   1. 3-bit staggerUMapping
   2. 5-bit staggerUShift
   3. 8-bit staggerU
3. 8-bit WGM
4. 8-bit GSU (Positive only)
