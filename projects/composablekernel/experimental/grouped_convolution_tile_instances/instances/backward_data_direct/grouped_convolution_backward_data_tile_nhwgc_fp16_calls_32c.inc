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
// Group 2: XOR swizzle, direct DRAM epilogue — Dgrad indices 34-41
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

// Additional cyclic shift instances for 34-channel kernels.
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<82, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CKernel<83, direct_conv::Version::v2>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Non-grouped convolution v3 — CyclicShift swizzle
// Dgrad configs 8-9 (16x16x32), 12-13 (32x32x16)
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<8, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<9, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<12, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<13, direct_conv::Version::v3>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Non-grouped convolution v3 — XOR swizzle
// Dgrad configs 16-17 (16x16x32), 20-21 (32x32x16), 24 (32x32x16 8-wave)
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<16, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<17, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<20, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<21, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<24, direct_conv::Version::v3>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Non-grouped convolution v3 — CyclicShift swizzle + LDS-staged epilogue
// Dgrad configs 26-27 (16x16x32), 30-31 (32x32x16)
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<26, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<27, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<30, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<31, direct_conv::Version::v3>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Non-grouped convolution v3 — XOR swizzle + LDS-staged epilogue
// Dgrad configs 34-35 (16x16x32), 38-39 (32x32x16), 42 (32x32x16 8-wave)
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<34, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<35, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<38, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<39, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<42, direct_conv::Version::v3>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Non-grouped convolution v3 — CyclicShift 8-wave 3x3
// Dgrad configs 44 (DRAM), 46 (LDS)
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<44, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<46, direct_conv::Version::v3>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Non-grouped convolution v3 — c_slices_per_wave > 1 (16x16x32 only)
// Each wave streams N C-chunks of cpg=32 channels through fixed-size LDS.
// Dgrad indices: 50,52 (waves=2 N=2/4), 54,56 (waves=2 N=2 Cyclic/XOR),
//                58,60 (waves=4 N=2/4), 62,64 (waves=4 N=2 Cyclic/XOR)
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<50, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<52, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<54, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<56, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<58, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<60, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<62, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData32CDenseKernel<64, direct_conv::Version::v3>, SIGNATURE>);

