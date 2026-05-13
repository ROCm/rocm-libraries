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
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<0, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<1, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<2, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<3, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<4, direct_conv::Version::v3>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Dgrad indices 10-14
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<10, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<11, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<12, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<13, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<14, direct_conv::Version::v3>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Dgrad indices 20-24
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<20, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<21, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<22, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<23, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<24, direct_conv::Version::v3>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Dgrad indices 30-34
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<30, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<31, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<32, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<33, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<34, direct_conv::Version::v3>, SIGNATURE>);
// Cyclic shift swizzle
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<42, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<43, direct_conv::Version::v3>, SIGNATURE>);
// Small vector load size for padded cases (Dgrad configs 44-46)
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<44, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<45, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<48, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<49, direct_conv::Version::v3>, SIGNATURE>);
// Enable thse configurations when fixed.
//run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<52, direct_conv::Version::v3>, SIGNATURE>);
//run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<53, direct_conv::Version::v3>, SIGNATURE>);


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
// Direct CK Tile convolution (streaming) 8c v2 kernels — Dgrad configs
////////////////////////////////////////////////////////////////////////
// Group 0: no swizzle, direct DRAM epilogue — Dgrad indices 0-8
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<0, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<1, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<2, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<3, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<4, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<5, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<6, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<7, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<8, direct_conv::Version::v2>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Dgrad indices 18-26
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<18, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<19, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<20, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<21, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<22, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<23, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<24, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<25, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<26, direct_conv::Version::v2>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Dgrad indices 36-44
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<36, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<37, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<38, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<39, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<40, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<41, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<42, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<43, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<44, direct_conv::Version::v2>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Dgrad indices 54-62
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<54, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<55, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<56, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<57, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<58, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<59, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<60, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<61, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<62, direct_conv::Version::v2>, SIGNATURE>);
// Group 4: Cyclic-shift swizzle — Dgrad indices 74-75
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<74, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<75, direct_conv::Version::v2>, SIGNATURE>);
// Padded CyclicShift configs — Dgrad indices 76-78 (vector_size = 4, 2, 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<76, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<77, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<78, direct_conv::Version::v2>, SIGNATURE>);
// Padded no-swizzle fallback — Dgrad index 82 (vector_size = 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<82, direct_conv::Version::v2>, SIGNATURE>);

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


// Additional cyclic shift instances for 8-channel kernel and 32-channel kernels.
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<86, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<87, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<82, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<83, direct_conv::Version::v2>, SIGNATURE>);
