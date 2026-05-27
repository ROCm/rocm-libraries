////////////////////////////////////////////////////////////////////////
// Direct HIP convolution (streaming) 32c kernels — configs 2-3 are Fprop
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectHipConvForward32CFp16Kernel<2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward32CFp16Kernel<3>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) 32c v2 kernels — Fprop configs
////////////////////////////////////////////////////////////////////////
// Group 0: no swizzle, direct DRAM epilogue — Fprop indices 8-15
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<8, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<9, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<10, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<11, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<12, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<13, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<14, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<15, direct_conv::Version::v2>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Fprop indices 24-33
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<24, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<25, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<26, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<27, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<28, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<29, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<30, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<31, direct_conv::Version::v2>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Fprop indices 42-47
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<40, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<41, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<42, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<43, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<44, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<45, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<46, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<47, direct_conv::Version::v2>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Fprop indices 56-63
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<56, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<57, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<58, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<59, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<60, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<61, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<62, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<63, direct_conv::Version::v2>, SIGNATURE>);
// Group 4: Cyclic-shift swizzle — Fprop indices 64-65
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<64, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<65, direct_conv::Version::v2>, SIGNATURE>);
// Padded CyclicShift configs — Fprop indices 73-77 (vector_size = 16, 8, 4, 2, 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<73, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<74, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<75, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<76, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<77, direct_conv::Version::v2>, SIGNATURE>);
// Padded no-swizzle fallback — Fprop index 79 (vector_size = 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<79, direct_conv::Version::v2>, SIGNATURE>);

// Additional cyclic shift instances for 34-channel kernels.
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<80, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<81, direct_conv::Version::v2>, SIGNATURE>);


////////////////////////////////////////////////////////////////////////
// Non-grouped convolution v3 (mfma_f32_16x16x32, M16N16K32)
// Valid Fprop config indices (from KernelConfigurations::configs[]):
//   2,3   no swizzle          (waves=4,2)
//   6,7   CyclicShift         (waves=4,2)
//   10,11 XOR                 (waves=4,2)
//   14,15 CyclicShift+LDS     (waves=4,2)
//   18,19 XOR+LDS             (waves=4,2)
//   21    CyclicShift 8-wave  (DRAM epilogue)
//   23    CyclicShift 8-wave  (LDS epilogue)
////////////////////////////////////////////////////////////////////////
// No swizzle, direct DRAM epilogue
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<2, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<3, direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift, direct DRAM epilogue
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<6, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<7, direct_conv::Version::v3>, SIGNATURE>);
// XOR, direct DRAM epilogue
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<10, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<11, direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift + LDS-staged epilogue
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<14, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<15, direct_conv::Version::v3>, SIGNATURE>);
// XOR + LDS-staged epilogue
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<18, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<19, direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift 8-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<21, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<23, direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift DRAM, 3-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<25, direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift+LDS, 3-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<27, direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift DRAM, 5-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<29, direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift+LDS, 5-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<31, direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift DRAM, 6-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<33, direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift+LDS, 6-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<35, direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift DRAM, 7-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<37, direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift+LDS, 7-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<39, direct_conv::Version::v3>, SIGNATURE>);
// XOR DRAM, 8-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<41, direct_conv::Version::v3>, SIGNATURE>);
// XOR+LDS, 8-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<43, direct_conv::Version::v3>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Non-grouped convolution v3 — c_slices_per_wave > 1 (16x16x32 only)
// Each wave streams N C-chunks of cpg=32 channels through fixed-size LDS.
// Fprop indices: 49,51 (waves=2 N=2/4), 53,55 (waves=2 N=2 Cyclic/XOR),
//                57,59 (waves=4 N=2/4), 61,63 (waves=4 N=2 Cyclic/XOR)
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<49, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<51, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<53, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<55, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<57, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<59, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<61, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<63, direct_conv::Version::v3>, SIGNATURE>);
