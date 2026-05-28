// Direct CK Tile convolution (streaming) 8c v2 kernels — Dgrad configs
// Group 0: no swizzle, direct DRAM epilogue — Dgrad indices 0-8
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<0, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<1, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<2, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<3, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<4, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<5, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<6, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<7, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<8, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Dgrad indices 18-26
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<18, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<19, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<20, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<21, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<22, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<23, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<24, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<25, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<26, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Dgrad indices 36-44
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<36, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<37, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<38, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<39, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<40, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<41, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<42, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<43, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<44, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Dgrad indices 54-62
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<54, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<55, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<56, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<57, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<58, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<59, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<60, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<61, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<62, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 4: Cyclic-shift swizzle — Dgrad indices 74-75
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<74, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<75, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Padded CyclicShift configs — Dgrad indices 76-78 (vector_size = 4, 2, 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<76, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<77, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<78, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
// Padded no-swizzle fallback — Dgrad index 82 (vector_size = 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<82, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);

// Additional cyclic shift instances for 8-channel kernel.
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<86, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<87, direct_conv::Version::v2, direct_conv::DataType::bf16>, SIGNATURE>);
