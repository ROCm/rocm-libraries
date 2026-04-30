////////////////////////////////////////////////////////////////////////
// Direct HIP convolution (streaming) kernels — configs 0-4 are Dgrad
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData4CFp16Kernel<0>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData4CFp16Kernel<1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData4CFp16Kernel<2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData4CFp16Kernel<3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData4CFp16Kernel<4>, SIGNATURE>);

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
// Cyclic shift swizzle
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<42, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<43, direct_conv::Version::v3>, SIGNATURE>);


////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) 16c v2 kernels — Dgrad configs 0-8
////////////////////////////////////////////////////////////////////////
// Group 0: no swizzle, direct DRAM epilogue — Dgrad indices 0-8
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<0, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<1, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<2, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<3, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<4, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<5, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<6, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<7, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<8, direct_conv::Version::v2>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Dgrad indices 18-26
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<18, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<19, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<20, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<21, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<22, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<23, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<24, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<25, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<26, direct_conv::Version::v2>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Dgrad indices 36-44
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<36, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<37, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<38, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<39, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<40, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<41, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<42, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<43, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<44, direct_conv::Version::v2>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Dgrad indices 54-62
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<54, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<55, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<56, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<57, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<58, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<59, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<60, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<61, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<62, direct_conv::Version::v2>, SIGNATURE>);

// Group 4: Cyclic-shift swizzle - Dgrad indices 74-75
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<74, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData16CFp16Kernel<75, direct_conv::Version::v2>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Direct HIP convolution (streaming) 8c kernels — configs 0-8 are Dgrad
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<0>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<4>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<5>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<6>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<7>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<8>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) 8c v1 kernels — Dgrad configs 0-8
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<0, direct_conv::Version::v1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<1, direct_conv::Version::v1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<2, direct_conv::Version::v1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<3, direct_conv::Version::v1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<4, direct_conv::Version::v1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<5, direct_conv::Version::v1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<6, direct_conv::Version::v1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<7, direct_conv::Version::v1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<8, direct_conv::Version::v1>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) 8c v2 kernels — Dgrad configs
////////////////////////////////////////////////////////////////////////
// Group 0: no swizzle, direct DRAM epilogue — Dgrad indices 0-8
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<0, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<1, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<2, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<3, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<4, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<5, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<6, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<7, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<8, direct_conv::Version::v2>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Dgrad indices 18-26
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<18, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<19, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<20, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<21, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<22, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<23, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<24, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<25, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<26, direct_conv::Version::v2>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Dgrad indices 36-44
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<36, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<37, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<38, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<39, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<40, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<41, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<42, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<43, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<44, direct_conv::Version::v2>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Dgrad indices 54-62
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<54, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<55, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<56, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<57, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<58, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<59, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<60, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<61, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<62, direct_conv::Version::v2>, SIGNATURE>);
// Group 4: Cyclic-shift swizzle — Dgrad indices 74-75
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<74, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CFp16Kernel<75, direct_conv::Version::v2>, SIGNATURE>);
