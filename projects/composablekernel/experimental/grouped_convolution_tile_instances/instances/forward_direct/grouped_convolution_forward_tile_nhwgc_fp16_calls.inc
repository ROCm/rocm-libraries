// Direct convolution (streaming) kernels — configs 5-9 are Fprop
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<5>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<6>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<7>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<8>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<9>, SIGNATURE>);
