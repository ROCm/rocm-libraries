////////////////////////////////////////////////////////////////////////
// Direct HIP convolution (streaming) kernels — configs 5-9 are Fprop
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<5>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<6>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<7>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<8>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<9>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Direct HIP convolution (streaming) 16c kernels — configs 9-17 are Fprop
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<9>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<10>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<11>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<12>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<13>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<14>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<15>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<17>, SIGNATURE>);

/////////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) kernels — Fprop configs
/////////////////////////////////////////////////////////////////////////////

// V3 instances (native CK tile distribution implementation)
// Group 0: no swizzle, direct DRAM epilogue — Fprop indices 5-9
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<5, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<6, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<7, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<8, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<9, direct_conv::Version::v3>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Fprop indices 15-19
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<15, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<16, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<17, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<18, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<19, direct_conv::Version::v3>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Fprop indices 25-29
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<25, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<26, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<27, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<28, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<29, direct_conv::Version::v3>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Fprop indices 35-39
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<35, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<36, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<37, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<38, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<39, direct_conv::Version::v3>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) 16c v2 kernels — Fprop configs 9-17
////////////////////////////////////////////////////////////////////////
// Group 0: no swizzle, direct DRAM epilogue — Fprop indices 9-17
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<9, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<10, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<11, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<12, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<13, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<14, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<15, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<16, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<17, direct_conv::Version::v2>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Fprop indices 27-35
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<27, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<28, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<29, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<30, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<31, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<32, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<33, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<34, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<35, direct_conv::Version::v2>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Fprop indices 45-53
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<45, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<46, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<47, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<48, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<49, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<50, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<51, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<52, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<53, direct_conv::Version::v2>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Fprop indices 63-71
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<63, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<64, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<65, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<66, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<67, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<68, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<69, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<70, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<71, direct_conv::Version::v2>, SIGNATURE>);

// Group 4: Cyclic-shift swizzle - Fprop indices 72-73
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<72, direct_conv::Version::v2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<73, direct_conv::Version::v2>, SIGNATURE>);
