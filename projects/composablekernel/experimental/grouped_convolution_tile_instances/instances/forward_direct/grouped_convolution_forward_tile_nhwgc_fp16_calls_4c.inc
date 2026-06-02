////////////////////////////////////////////////////////////////////////
// Direct HIP convolution (streaming) kernels — configs 5-9 are Fprop
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<ck_tile::direct_hip_conv::grouped_4c::configs_map.get(5)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<ck_tile::direct_hip_conv::grouped_4c::configs_map.get(6)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<ck_tile::direct_hip_conv::grouped_4c::configs_map.get(7)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<ck_tile::direct_hip_conv::grouped_4c::configs_map.get(8)>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<ck_tile::direct_hip_conv::grouped_4c::configs_map.get(9)>, SIGNATURE>);

/////////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) kernels — Fprop configs
/////////////////////////////////////////////////////////////////////////////

// V3 instances (native CK tile distribution implementation)
// Group 0: no swizzle, direct DRAM epilogue — Fprop indices 5-9
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(5), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(6), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(7), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(8), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(9), direct_conv::Version::v3>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Fprop indices 15-19
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(15), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(16), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(17), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(18), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(19), direct_conv::Version::v3>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Fprop indices 25-29
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(25), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(26), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(27), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(28), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(29), direct_conv::Version::v3>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Fprop indices 35-39
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(35), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(36), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(37), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(38), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(39), direct_conv::Version::v3>, SIGNATURE>);
// Cyclic shift swizzle
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(40), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(41), direct_conv::Version::v3>, SIGNATURE>);
// Small vector load size for padded cases
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(46), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(47), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(50), direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(51), direct_conv::Version::v3>, SIGNATURE>);
// Enable these configurations when fixed.
//run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(54), direct_conv::Version::v3>, SIGNATURE>);
//run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CKernel<ck_tile::direct_conv::grouped_4c_tile::v3::KernelConfigurations<ck_tile::direct_conv::DataType::fp16>::configs_map.get(55), direct_conv::Version::v3>, SIGNATURE>);
