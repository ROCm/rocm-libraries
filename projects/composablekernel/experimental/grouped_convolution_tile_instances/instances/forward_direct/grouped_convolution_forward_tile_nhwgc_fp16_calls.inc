// Direct HIP convolution (streaming) kernels — configs 5-9 are Fprop
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<5>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<6>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<7>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<8>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<9>, SIGNATURE>);

// Direct CK Tile convolution (streaming) kernels — configs 5-9 are Fprop
// V1 SwizzleT instances
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<5>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<6>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<7>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<8>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<9>, SIGNATURE>);
// V1 SwizzleXOR instances
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<10>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<11>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<12>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<13>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<14>, SIGNATURE>);
// V2 instance - config 1 is Fprop
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<1, direct_conv::Version::v2>, SIGNATURE>);
