// Direct CK Tile convolution (streaming) kernels — Dgrad v3 configs
// V3 instances (native CK tile distribution implementation)
// Group 0: no swizzle, direct DRAM epilogue — Dgrad indices 0-4
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<0, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<1, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<2, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<3, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<4, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Dgrad indices 10-14
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<10, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<11, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<12, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<13, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<14, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Dgrad indices 20-24
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<20, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<21, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<22, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<23, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<24, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Dgrad indices 30-34
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<30, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<31, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<32, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<33, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<34, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
// Cyclic shift swizzle
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<42, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<43, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
// Small vector load size for padded cases (Dgrad configs 44-46)
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<44, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<45, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<48, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<49, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
// Enable thse configurations when fixed.
//run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<52, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
//run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CKernel<53, direct_conv::Version::v3, direct_conv::DataType::bf16>, SIGNATURE>);
