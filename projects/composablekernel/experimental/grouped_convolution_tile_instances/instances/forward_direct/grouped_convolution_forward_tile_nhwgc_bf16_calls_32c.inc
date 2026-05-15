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

// Additional cyclic shift instances for 32-channel kernels.
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<80, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<81, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Non-grouped (standard) convolution — 32c MFMA with C-reduction
// G=1, C > 32, K > 32. Fprop configs: 3 (direct DRAM) + 3 (LDS epilogue)
////////////////////////////////////////////////////////////////////////
// Fprop, direct DRAM epilogue — configs 3-5 (waves 8, 4, 2)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<3, direct_conv::Version::v1, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<4, direct_conv::Version::v1, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<5, direct_conv::Version::v1, direct_conv::DataType::bf16>, SIGNATURE>);
// Fprop, LDS-staged epilogue — configs 9-11 (waves 8, 4, 2)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<9, direct_conv::Version::v1, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<10, direct_conv::Version::v1, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<11, direct_conv::Version::v1, direct_conv::DataType::bf16>, SIGNATURE>);
