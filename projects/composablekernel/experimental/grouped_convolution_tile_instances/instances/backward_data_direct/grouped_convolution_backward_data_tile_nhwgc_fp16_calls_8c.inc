////////////////////////////////////////////////////////////////////////
// Direct HIP convolution (streaming) 8c kernels — configs 0-8 are Dgrad
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<ck_tile::direct_hip_conv::grouped_8c::configs_map.get(0)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<ck_tile::direct_hip_conv::grouped_8c::configs_map.get(1)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<ck_tile::direct_hip_conv::grouped_8c::configs_map.get(2)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<ck_tile::direct_hip_conv::grouped_8c::configs_map.get(3)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<ck_tile::direct_hip_conv::grouped_8c::configs_map.get(4)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<ck_tile::direct_hip_conv::grouped_8c::configs_map.get(5)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<ck_tile::direct_hip_conv::grouped_8c::configs_map.get(6)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<ck_tile::direct_hip_conv::grouped_8c::configs_map.get(7)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData8CFp16Kernel<ck_tile::direct_hip_conv::grouped_8c::configs_map.get(8)>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) 8c v2 kernels — Dgrad configs
////////////////////////////////////////////////////////////////////////
// Group 0: no swizzle, direct DRAM epilogue — Dgrad indices 0-8
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(0), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(1), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(2), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(3), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(4), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(5), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(6), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(7), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(8), direct_conv::Version::v2>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Dgrad indices 18-26
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(18), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(19), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(20), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(21), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(22), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(23), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(24), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(25), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(26), direct_conv::Version::v2>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Dgrad indices 36-44
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(36), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(37), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(38), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(39), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(40), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(41), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(42), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(43), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(44), direct_conv::Version::v2>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Dgrad indices 54-62
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(54), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(55), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(56), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(57), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(58), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(59), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(60), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(61), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(62), direct_conv::Version::v2>, SIGNATURE>);
// Group 4: Cyclic-shift swizzle — Dgrad indices 74-75
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(74), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(75), direct_conv::Version::v2>, SIGNATURE>);
// Padded CyclicShift configs — Dgrad indices 76-78 (vector_size = 4, 2, 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(76), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(77), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(78), direct_conv::Version::v2>, SIGNATURE>);
// Padded no-swizzle fallback — Dgrad index 82 (vector_size = 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(82), direct_conv::Version::v2>, SIGNATURE>);

// Additional cyclic shift instances for 8-channel kernel.
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(86), direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData8CKernel<ck_tile::direct_conv::grouped_8c_tile::v2::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(87), direct_conv::Version::v2>, SIGNATURE>);
