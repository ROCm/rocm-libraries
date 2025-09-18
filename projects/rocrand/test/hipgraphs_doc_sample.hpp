size_t        size = 1000;
float*        data_0;
unsigned int* data_1;

hipMalloc(&data_0, sizeof(*data_0) * size);
hipMalloc(&data_1, sizeof(*data_1) * size);

hipGraph_t graph;
hipGraphCreate(&graph, 0);

hipStream_t stream;
hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);

rocrand_generator generator;
rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT);
rocrand_set_stream(generator, stream);
rocrand_initialize_generator(generator);

hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);

rocrand_generate_normal(generator, data_0, size, 10.0F, 2.0F);
rocrand_generate_poisson(generator, data_1, size, 3);

hipStreamEndCapture(stream, &graph);

hipGraphExec_t instance;
hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

hipGraphLaunch(instance, stream);
hipStreamSynchronize(stream);

hipGraphExecDestroy(instance);
rocrand_destroy_generator(generator);
hipStreamDestroy(stream);
hipGraphDestroy(graph);
hipFree(data_1);
hipFree(data_0);
