// Direct HIP convolution (streaming) kernels — configs 0-4 are Dgrad
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData4CFp16Kernel<0>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData4CFp16Kernel<1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData4CFp16Kernel<2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData4CFp16Kernel<3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData4CFp16Kernel<4>, SIGNATURE>);

// Direct CK Tile convolution (streaming) kernels — configs 0-4 are Dgrad
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<0>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<4>, SIGNATURE>);

/////////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) kernels — Dgrad v3 configs
/////////////////////////////////////////////////////////////////////////////

// V3 instances (native CK tile distribution implementation)
// Group 0: no swizzle, direct DRAM epilogue — Dgrad indices 0-4
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<0, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<1, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<2, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<3, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<4, direct_conv::Version::v3>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Dgrad indices 10-14
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<10, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<11, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<12, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<13, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<14, direct_conv::Version::v3>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Dgrad indices 20-24
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<20, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<21, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<22, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<23, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<24, direct_conv::Version::v3>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Dgrad indices 30-34
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<30, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<31, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<32, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<33, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<34, direct_conv::Version::v3>, SIGNATURE>);

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
