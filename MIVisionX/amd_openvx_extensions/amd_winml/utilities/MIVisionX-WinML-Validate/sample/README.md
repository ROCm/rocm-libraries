# Sample

Get ONNX models from [ONNX Model Zoo](https://github.com/onnx/models)

## Sample - SqeezeNet

* Download the [SqueezeNet](https://s3.amazonaws.com/download.onnx/models/opset_8/squeezenet.tar.gz) ONNX Model
* Use [Netron](https://lutzroeder.github.io/netron/) to open the model.onnx
	* Look at Model Properties to find Input & Output Tensor Name (data_0 - input; softmaxout_1 - output)
	* Look at output tensor dimensions (n,c,h,w  - [1,1000,1,1] for softmaxout_1)
* Use the label file - Labels.txt and sample image - car.JPEG to run the MIVisionX WinML Validation
* Use --d 0 if only CPU available, else use --d 3 for GPU inference

````
        MIVisionX-WinML-Validate.exe [options]  --m \full-path-to-model\model.onnx
                                                --i data_0
                                                --o softmaxout_1
                                                --s 1,1000,1,1
                                                --l \full-path-to-labels\Labels.txt 
                                                --f \full-path-to-labels\car.JPEG
                                                --d 3 [optional]
````
