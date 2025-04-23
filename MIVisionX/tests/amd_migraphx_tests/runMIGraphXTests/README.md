* This script is used for testing multiple onnx models using ```amdMIGraphXnode``` node extension for inference.
* The script takes one image for some of the models which use an imagenet dataset and one image for the mnist model.
* Models tested in this script (needed as input). All models can be found [here](https://github.com/onnx/models). 
  1. resnet50
  2. googlenet
  3. alexnet
  4. densenet
  5. squeezenet
  6. vgg19
* Test one or more of the models by specifying the paths to the models which need testing.
* Tests batch sizes from 1-N (N specified by user)
* Currently only mode 1 -- runs ONNX models
* Currently only Batch size = 1 works (TO test bigger batch sizes, build and install [this branch](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/tree/dyn_reshape_parsing))
* Usage:
```
Usage: ./runMIGraphXTests
            --tensor                    <path to tensor directory>
            --profiler_mode  <range:0-2; default:1> [optional]
              Mode 0 - Run all tests\n"
              Mode 1 - Run all ONNX tests\n"
              Mode 2 - Run all JSON tests\n"
            --profiler_level <range:0-N; default:1> [N = batch size][optional]
            --resnet50       <resnet50-model>
            --googlenet      <googlenet-model>
            --squeezenet     <resnet101-model>
            --alexnet        <resnet152-model>
            --vgg19          <vgg19-model>
            --densenet       <densenet-model>
```
* Example: (to run all models for batch sizes 1 to 128)
```
mkdir build
cd build
cmake ../
make -j
./runMIGraphXTests --tensor <path to Mike tensor folder> --googlenet <path to googlenet model> --alexnet <path to alexnet model> --squeezenet <path to squeezenet model> --resnet50 <path to resnet50 model> --vgg19 <path to vgg19 model> --densenet <path to densenet model> --profiler_level 1
```
