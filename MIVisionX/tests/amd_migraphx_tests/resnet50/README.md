* This shows an example of how to run resnet50 inference model using MIVisionX's extension for AMD MIGraphx.

* To run the test, clone MIVisionX and build and install
```
cd <path-to-MIVisionX>/tests/amd_migraphx_test/resnet50/
wget https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx?raw=true
mkdir build
cd build
./migraphx_node_test <paht-to-model> <path-to-image>
```

* Example
```
./migraphx_node_test ../resnet50-v2-7.onnx https://github.com/ROCm/MIVisionX/tree/master/data/images/AMD-tinyDataSet/AMD-tinyDataSet_0000.JPEG
```
