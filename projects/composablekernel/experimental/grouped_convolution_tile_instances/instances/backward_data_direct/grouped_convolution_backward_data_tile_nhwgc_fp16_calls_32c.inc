////////////////////////////////////////////////////////////////////////
// Direct HIP convolution (streaming) 32c kernels — configs 0-1 are Dgrad
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData32CFp16Kernel<0>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData32CFp16Kernel<1>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) 32c v2 kernels — Dgrad configs
////////////////////////////////////////////////////////////////////////
// Group 0: no swizzle, direct DRAM epilogue — Dgrad indices 0-7
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<0, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<1, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<2, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<3, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<4, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<5, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<6, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<7, direct_conv::Version::v2>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Dgrad indices 16-23
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<16, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<17, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<18, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<19, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<20, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<21, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<22, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<23, direct_conv::Version::v2>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Dgrad indices 32-39
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<32, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<33, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<34, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<35, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<36, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<37, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<38, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<39, direct_conv::Version::v2>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Dgrad indices 48-55
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<48, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<49, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<50, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<51, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<52, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<53, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<54, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<55, direct_conv::Version::v2>, SIGNATURE>);
// Group 4: Cyclic-shift swizzle — Dgrad indices 66-67
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<66, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<67, direct_conv::Version::v2>, SIGNATURE>);
// Padded CyclicShift configs — Dgrad indices 68-72 (vector_size = 16, 8, 4, 2, 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<68, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<69, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<70, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<71, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<72, direct_conv::Version::v2>, SIGNATURE>);
// Padded no-swizzle fallback — Dgrad index 78 (vector_size = 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<78, direct_conv::Version::v2>, SIGNATURE>);

// Additional cyclic shift instances for 32-channel kernels.
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<82, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<83, direct_conv::Version::v2>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Non-grouped (standard) convolution — 32c MFMA with C-reduction
// G=1, C > 32, K > 32. Dgrad configs: 3 (direct DRAM) + 3 (LDS epilogue)
////////////////////////////////////////////////////////////////////////
// Dgrad, direct DRAM epilogue — configs 0-2 (waves 8, 4, 2)
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<0, direct_conv::Version::v1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<1, direct_conv::Version::v1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<2, direct_conv::Version::v1>, SIGNATURE>);
// Dgrad, LDS-staged epilogue — configs 6-8 (waves 8, 4, 2)
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<6, direct_conv::Version::v1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<7, direct_conv::Version::v1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<8, direct_conv::Version::v1>, SIGNATURE>);
