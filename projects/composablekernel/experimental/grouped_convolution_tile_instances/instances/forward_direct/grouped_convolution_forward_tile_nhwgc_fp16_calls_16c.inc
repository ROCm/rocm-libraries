////////////////////////////////////////////////////////////////////////
// Direct HIP convolution (streaming) 16c kernels — configs 9-17 are Fprop
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<ck_tile::direct_hip_conv::grouped_16c::configs_map.get(9)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<ck_tile::direct_hip_conv::grouped_16c::configs_map.get(10)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<ck_tile::direct_hip_conv::grouped_16c::configs_map.get(11)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<ck_tile::direct_hip_conv::grouped_16c::configs_map.get(12)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<ck_tile::direct_hip_conv::grouped_16c::configs_map.get(13)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<ck_tile::direct_hip_conv::grouped_16c::configs_map.get(14)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<ck_tile::direct_hip_conv::grouped_16c::configs_map.get(15)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<ck_tile::direct_hip_conv::grouped_16c::configs_map.get(16)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<ck_tile::direct_hip_conv::grouped_16c::configs_map.get(17)>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) 16c v2 kernels — Fprop configs 9-17
////////////////////////////////////////////////////////////////////////
// Group 0: no swizzle, direct DRAM epilogue — Fprop indices 9-17
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(9), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(10), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(11), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(12), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(13), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(14), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(15), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(16), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(17), direct_conv::Version::v2>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Fprop indices 27-35
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(27), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(28), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(29), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(30), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(31), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(32), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(33), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(34), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(35), direct_conv::Version::v2>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Fprop indices 45-53
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(45), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(46), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(47), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(48), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(49), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(50), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(51), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(52), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(53), direct_conv::Version::v2>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Fprop indices 63-71
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(63), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(64), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(65), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(66), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(67), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(68), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(69), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(70), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(71), direct_conv::Version::v2>, SIGNATURE>);

// Group 4: Cyclic-shift swizzle - Fprop indices 72-73
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(72), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(73), direct_conv::Version::v2>, SIGNATURE>);
// Padded CyclicShift configs — Fprop indices 80-83 (vector_size = 8, 4, 2, 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(80), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(81), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(82), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(83), direct_conv::Version::v2>, SIGNATURE>);
// Padded no-swizzle fallback — Fprop index 85 (vector_size = 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CKernel<ck_tile::direct_conv::grouped_16c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(85), direct_conv::Version::v2>, SIGNATURE>);
