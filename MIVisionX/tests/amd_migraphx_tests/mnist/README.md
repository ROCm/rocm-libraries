* This shows an example of how to run mnist inference model using MIVisionX's extension for AMD MIGraphx.

* To run the test, clone MIVisionX and build and install
```
cd <path-to-MIVisionX>/tests/amd_migraphx_test/mnist/
wget https://github.com/onnx/models/blob/main/validated/vision/classification/mnist/model/mnist-8.onnx?raw=true
mkdir build
cd build
./migraphx_node_test <path-to-model>
```

* Example
```
./migraphx_node_test ../mnist-8.onnx 
```
