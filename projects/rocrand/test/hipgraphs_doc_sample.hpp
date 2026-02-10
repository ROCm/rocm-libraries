size_t        size = 1000;
float*        data_0;
unsigned int* data_1;

HIP_CHECK(hipMalloc(&data_0, sizeof(*data_0) * size));
HIP_CHECK(hipMalloc(&data_1, sizeof(*data_1) * size));

hipGraph_t graph;
HIP_CHECK(hipGraphCreate(&graph, 0));

hipStream_t stream;
HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

rocrand_generator generator;
rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT);
rocrand_set_stream(generator, stream);
rocrand_initialize_generator(generator);

HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

rocrand_generate_normal(generator, data_0, size, 10.0F, 2.0F);
rocrand_generate_poisson(generator, data_1, size, 3);

HIP_CHECK(hipStreamEndCapture(stream, &graph));

hipGraphExec_t instance;
HIP_CHECK(hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

HIP_CHECK(hipGraphLaunch(instance, stream));
HIP_CHECK(hipStreamSynchronize(stream));

HIP_CHECK(hipGraphExecDestroy(instance));
rocrand_destroy_generator(generator);
HIP_CHECK(hipStreamDestroy(stream));
HIP_CHECK(hipGraphDestroy(graph));
HIP_CHECK(hipFree(data_1));
HIP_CHECK(hipFree(data_0));
