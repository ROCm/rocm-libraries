////////////////////////////////////////////////////////////////////////
// Direct HIP convolution (streaming) kernels — configs 5-9 are Fprop
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<5>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<6>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<7>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<8>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<9>, SIGNATURE>);

/////////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) kernels — Fprop configs
/////////////////////////////////////////////////////////////////////////////

// V3 instances (native CK tile distribution implementation)
// Group 0: no swizzle, direct DRAM epilogue — Fprop indices 5-9
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<5, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<6, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<7, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<8, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<9, direct_conv::Version::v3>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Fprop indices 15-19
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<15, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<16, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<17, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<18, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<19, direct_conv::Version::v3>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Fprop indices 25-29
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<25, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<26, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<27, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<28, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<29, direct_conv::Version::v3>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Fprop indices 35-39
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<35, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<36, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<37, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<38, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<39, direct_conv::Version::v3>, SIGNATURE>);
// Cyclic shift swizzle
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<40, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<41, direct_conv::Version::v3>, SIGNATURE>);
// Small vector load size for padded cases
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<46, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<47, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<50, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<51, direct_conv::Version::v3>, SIGNATURE>);
// Enable these configurations when fixed.
//run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<54, direct_conv::Version::v3>, SIGNATURE>);
//run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<55, direct_conv::Version::v3>, SIGNATURE>);
