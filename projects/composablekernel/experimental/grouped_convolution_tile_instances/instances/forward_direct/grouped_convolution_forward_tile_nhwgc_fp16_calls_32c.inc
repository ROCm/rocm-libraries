////////////////////////////////////////////////////////////////////////
// Direct HIP convolution (streaming) 32c kernels — configs 2-3 are Fprop
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectHipConvForward32CFp16Kernel<ck_tile::direct_hip_conv::grouped_32c::configs_map.get(2)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward32CFp16Kernel<ck_tile::direct_hip_conv::grouped_32c::configs_map.get(3)>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) 32c v2 kernels — Fprop configs
////////////////////////////////////////////////////////////////////////
// Group 0: no swizzle, direct DRAM epilogue — Fprop indices 8-15
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(8), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(9), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(10), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(11), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(12), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(13), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(14), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(15), direct_conv::Version::v2>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Fprop indices 24-33
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(24), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(25), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(26), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(27), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(28), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(29), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(30), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(31), direct_conv::Version::v2>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Fprop indices 42-47
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(40), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(41), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(42), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(43), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(44), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(45), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(46), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(47), direct_conv::Version::v2>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Fprop indices 56-63
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(56), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(57), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(58), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(59), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(60), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(61), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(62), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(63), direct_conv::Version::v2>, SIGNATURE>);
// Group 4: Cyclic-shift swizzle — Fprop indices 64-65
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(64), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(65), direct_conv::Version::v2>, SIGNATURE>);
// Padded CyclicShift configs — Fprop indices 73-77 (vector_size = 16, 8, 4, 2, 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(73), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(74), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(75), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(76), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(77), direct_conv::Version::v2>, SIGNATURE>);
// Padded no-swizzle fallback — Fprop index 79 (vector_size = 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(79), direct_conv::Version::v2>, SIGNATURE>);

// Additional cyclic shift instances for 34-channel kernels.
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(80), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CKernel<ck_tile::direct_conv::grouped_32c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(81), direct_conv::Version::v2>, SIGNATURE>);


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
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(2), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(3), direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift, direct DRAM epilogue
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(6), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(7), direct_conv::Version::v3>, SIGNATURE>);
// XOR, direct DRAM epilogue
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(10), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(11), direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift + LDS-staged epilogue
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(14), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(15), direct_conv::Version::v3>, SIGNATURE>);
// XOR + LDS-staged epilogue
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(18), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(19), direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift 8-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(21), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(23), direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift DRAM, 3-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(25), direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift+LDS, 3-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(27), direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift DRAM, 5-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(29), direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift+LDS, 5-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(31), direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift DRAM, 6-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(33), direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift+LDS, 6-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(35), direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift DRAM, 7-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(37), direct_conv::Version::v3>, SIGNATURE>);
// CyclicShift+LDS, 7-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(39), direct_conv::Version::v3>, SIGNATURE>);
// XOR DRAM, 8-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(41), direct_conv::Version::v3>, SIGNATURE>);
// XOR+LDS, 8-wave
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(43), direct_conv::Version::v3>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Non-grouped convolution v3 — c_slices_per_wave > 1 (16x16x32 only)
// Each wave streams N C-chunks of cpg=32 channels through fixed-size LDS.
// Fprop indices: 49,51 (waves=2 N=2/4), 53,55 (waves=2 N=2 Cyclic/XOR),
//                57,59 (waves=4 N=2/4), 61,63 (waves=4 N=2 Cyclic/XOR)
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(44), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(46), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(48), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(50), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(52), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(54), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(56), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward32CDenseKernel<ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(58), direct_conv::Version::v3>, SIGNATURE>);
