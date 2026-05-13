// Direct CK Tile convolution (streaming) kernels — Fprop configs
// V3 instances (native CK tile distribution implementation)
// Group 0: no swizzle, direct DRAM epilogue — Fprop indices 5-9
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<5, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<6, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<7, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<8, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<9, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Fprop indices 15-19
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<15, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<16, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<17, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<18, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<19, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Fprop indices 25-29
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<25, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<26, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<27, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<28, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<29, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Fprop indices 35-39
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<35, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<36, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<37, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<38, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<39, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
// Cyclic shift swizzle
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<40, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<41, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
// Small vector load size for padded cases
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<46, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<47, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<50, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<51, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
// Enable these configurations when fixed.
//run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<54, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
//run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<55, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
// Direct CK Tile convolution (streaming) 16c v2 kernels — Fprop configs 9-17
// Group 0: no swizzle, direct DRAM epilogue — Fprop indices 9-17
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<9, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<10, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<11, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<12, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<13, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<14, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<15, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<16, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<17, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Fprop indices 27-35
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<27, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<28, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<29, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<30, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<31, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<32, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<33, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<34, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<35, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Fprop indices 45-53
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<45, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<46, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<47, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<48, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<49, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<50, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<51, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<52, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<53, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Fprop indices 63-71
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<63, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<64, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<65, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<66, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<67, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<68, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<69, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<70, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<71, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 4: Cyclic-shift swizzle - Fprop indices 72-73
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<72, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<73, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Padded CyclicShift configs — Fprop indices 80-83 (vector_size = 8, 4, 2, 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<80, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<81, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<82, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<83, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Padded no-swizzle fallback — Fprop index 85 (vector_size = 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<85, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Direct CK Tile convolution (streaming) 8c v2 kernels — Fprop configs
// Group 0: no swizzle, direct DRAM epilogue — Fprop indices 9-17
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<9, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<10, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<11, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<12, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<13, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<14, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<15, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<16, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<17, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Fprop indices 27-35
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<27, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<28, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<29, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<30, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<31, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<32, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<33, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<34, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<35, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Fprop indices 45-53
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<45, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<46, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<47, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<48, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<49, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<50, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<51, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<52, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<53, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Fprop indices 63-71
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<63, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<64, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<65, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<66, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<67, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<68, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<69, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<70, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<71, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 4: Cyclic-shift swizzle — Fprop indices 72-73
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<72, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<73, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Padded CyclicShift configs — Fprop indices 79-81 (vector_size = 4, 2, 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<79, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<80, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<81, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Padded no-swizzle fallback — Fprop index 83 (vector_size = 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<83, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Direct CK Tile convolution (streaming) 32c v2 kernels — Fprop configs
// Group 0: no swizzle, direct DRAM epilogue — Fprop indices 8-15
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<8, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<9, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<10, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<11, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<12, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<13, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<14, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<15, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Fprop indices 24-31
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<24, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<25, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<26, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<27, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<28, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<29, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<30, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<31, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Fprop indices 40-47
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<40, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<41, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<42, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<43, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<44, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<45, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<46, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<47, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Fprop indices 56-63
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<56, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<57, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<58, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<59, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<60, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<61, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<62, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<63, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 4: Cyclic-shift swizzle — Fprop indices 64-65
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<64, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<65, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Padded CyclicShift configs — Fprop indices 73-77 (vector_size = 16, 8, 4, 2, 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<73, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<74, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<75, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<76, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<77, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Padded no-swizzle fallback — Fprop index 79 (vector_size = 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<79, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Additional cyclic shift instances for 8-channel kernel and 32-channel kernels.
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<84, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<85, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<80, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<81, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
