////////////////////////////////////////////////////////////////////////
// Direct HIP convolution (streaming) 8c kernels — configs 9-17 are Fprop
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectHipConvForward8CFp16Kernel<9>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward8CFp16Kernel<10>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward8CFp16Kernel<11>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward8CFp16Kernel<12>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward8CFp16Kernel<13>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward8CFp16Kernel<14>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward8CFp16Kernel<15>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward8CFp16Kernel<16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward8CFp16Kernel<17>, SIGNATURE>);


////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) 8c v2 kernels — Fprop configs
////////////////////////////////////////////////////////////////////////
// Group 0: no swizzle, direct DRAM epilogue — Fprop indices 9-17
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<9, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<10, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<11, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<12, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<13, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<14, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<15, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<16, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<17, direct_conv::Version::v2>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Fprop indices 27-35
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<27, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<28, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<29, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<30, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<31, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<32, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<33, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<34, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<35, direct_conv::Version::v2>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Fprop indices 45-53
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<45, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<46, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<47, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<48, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<49, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<50, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<51, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<52, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<53, direct_conv::Version::v2>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Fprop indices 63-71
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<63, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<64, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<65, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<66, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<67, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<68, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<69, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<70, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<71, direct_conv::Version::v2>, SIGNATURE>);
// Group 4: Cyclic-shift swizzle — Fprop indices 72-73
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<72, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<73, direct_conv::Version::v2>, SIGNATURE>);
// Padded CyclicShift configs — Fprop indices 79-81 (vector_size = 4, 2, 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<79, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<80, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<81, direct_conv::Version::v2>, SIGNATURE>);
// Padded no-swizzle fallback — Fprop index 83 (vector_size = 1)
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<83, direct_conv::Version::v2>, SIGNATURE>);

// Additional cyclic shift instances for 8-channel kernel.
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<84, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward8CKernel<85, direct_conv::Version::v2>, SIGNATURE>);
