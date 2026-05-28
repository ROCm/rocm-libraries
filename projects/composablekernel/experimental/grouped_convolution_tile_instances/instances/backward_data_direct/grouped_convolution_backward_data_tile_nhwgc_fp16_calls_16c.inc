////////////////////////////////////////////////////////////////////////
// Direct HIP convolution (streaming) 16c kernels — configs 0-8 are Dgrad
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData16CFp16Kernel<0>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData16CFp16Kernel<1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData16CFp16Kernel<2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData16CFp16Kernel<3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData16CFp16Kernel<4>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData16CFp16Kernel<5>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData16CFp16Kernel<6>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData16CFp16Kernel<7>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData16CFp16Kernel<8>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) 16c v2 kernels — Dgrad configs 0-8
////////////////////////////////////////////////////////////////////////
// Group 0: no swizzle, direct DRAM epilogue — Dgrad indices 0-8
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<0, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<1, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<2, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<3, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<4, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<5, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<6, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<7, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<8, direct_conv::Version::v2>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Dgrad indices 18-26
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<18, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<19, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<20, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<21, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<22, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<23, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<24, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<25, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<26, direct_conv::Version::v2>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Dgrad indices 36-44
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<36, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<37, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<38, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<39, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<40, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<41, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<42, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<43, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<44, direct_conv::Version::v2>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Dgrad indices 54-62
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<54, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<55, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<56, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<57, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<58, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<59, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<60, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<61, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<62, direct_conv::Version::v2>, SIGNATURE>);

// Group 4: Cyclic-shift swizzle - Dgrad indices 74-75
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<74, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<75, direct_conv::Version::v2>, SIGNATURE>);
// Padded CyclicShift configs — Dgrad indices 76-79 (vector_size = 8, 4, 2, 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<76, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<77, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<78, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<79, direct_conv::Version::v2>, SIGNATURE>);
// Padded no-swizzle fallback — Dgrad index 84 (vector_size = 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CKernel<84, direct_conv::Version::v2>, SIGNATURE>);
