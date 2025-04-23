# Zen DNN - Tests

## Convolution Layer Test

### Steps:
1. create engin and stream
2. create memory (source, weights, bias, & destination)
3. create memory descriptor
4. create convolution descriptor
5. create convolution primitive descriptor
6. create convolution primitive
7. execute the convlution

### Build & Run
```
mkdir build-conv && cd build-conv
cmake ../conv && make
ZENDNN_LOG_OPTS=ALL:5 ZENDNN_VERBOSE=1 ./zendnn_conv cpu
```

**NOTE:** use `export ZENDNN_LOG_OPTS=ALL:5 ZENDNN_VERBOSE=1` for developer information

## MNIST FP32 Sample
This C++ API example demonstrates how to build the MNIST neural network topology for forward-pass inference.

### Some key take-aways include:

* Tensor implementation and submission to primitives.
* Primitives creation.
* Dependency between the primitive input and output data.
* 'Inference-only' configurations.

**NOTE:** The example implements the MNIST layers as numbered primitives (for example, conv1, pool1, conv2).

### Build & Run
```
mkdir build-mnist && cd build-mnist
cmake ../mnist && make
ZENDNN_LOG_OPTS=ALL:5 ZENDNN_VERBOSE=1 ./zendnn_mnist_f32 cpu
```

## MNIST FP32 Inference Application
This C++ API example demonstrates how to build the MNIST neural network topology for forward-pass inference and test with images.

### Some key take-aways include:

* How to load weights & bias tensors.
* How to load user input
* How to get network output for post processing

**NOTE:** The example implements the MNIST layers as numbered primitives (for example, conv1, pool1, conv2).

### Build & Run
```
mkdir build-mnist && cd build-mnist
cmake ../mnist_app && make
ZENDNN_LOG_OPTS=ALL:5 ZENDNN_VERBOSE=1 ./zendnn_mnist_app data/weights.bin images/input_data_3.bin 
```
