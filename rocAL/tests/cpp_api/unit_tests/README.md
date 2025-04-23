# rocAL Unit Tests

This application can be used to verify the functionality of the API offered by rocAL.

## Pre-requisites

* Ubuntu Linux, version `22.04` or later
* rocAL library
* [OpenCV 3.4+](https://github.com/opencv/opencv/releases/tag/3.4.0)
* ROCm Performance Primitives (RPP)
* Python3


## Build Instructions

````bash
mkdir build
cd build
cmake ../
make
````

## Running the application

```bash
./unit_tests

Usage: ./unit_tests reader-type pipeline-type=1(classification)2(detection)3(keypoints) <image-dataset-folder> output_image_name <width> <height> test_case gpu=1/cpu=0 rgb=1/grayscale=0 one_hot_labels=num_of_classes/0  display_all=0(display_last_only)1(display_all)
```

> [!NOTE]
* Use the appropriate image dataset as per the reader being tested.
* For example: the coco reader needs images from the coco dataset.

## Output Verification

* Install Pillow library using `python3 -m pip install Pillow`

### Prepare dataset
* If the user has a dataset with golden outputs, please use the `testAllScripts.sh` script to provide the input and golden ouputs. 
* This test also runs the `image_comparison.py` script at the end to compare the results with the golden outputs.
* The data needs to be organized in separate folders under `rocal_data` directory for each reader and a golden output directory for the verification.

```
image_path (TurboJPEG image decoding)  : ${ROCAL_DATA_PATH}/rocal_data/images/
coco_detection_path (COCO Dataset)     : ${ROCAL_DATA_PATH}/rocal_data/coco/images/
tf_classification_path (TF records Dataset for classification) : ${ROCAL_DATA_PATH}/rocal_data/tf/classification/
tf_detection_path (TF records Dataset for detection) : ${ROCAL_DATA_PATH}/rocal_data/tf/detection/
caffe_classification_path (caffe .mdb for classification) : ${ROCAL_DATA_PATH}/rocal_data/caffe/classification/
caffe_detection_path  (caffe .mdb for detetction) : ${ROCAL_DATA_PATH}/rocal_data/caffe/detection/
caffe2_classification_path (caffe2 .mdb for classification) : ${ROCAL_DATA_PATH}/rocal_data/caffe2/classification/
caffe2_detection_path (caffe2 .mdb for classification) : ${ROCAL_DATA_PATH}/rocal_data/caffe2/detection/
mxnet_path (mxnet .idx, .lst and .rec files) : ${ROCAL_DATA_PATH}/rocal_data/mxnet/
webdataset_tar_path (web dataset .tar file) : ${ROCAL_DATA_PATH}/rocal_data/web_dataset/tar_file
```

* Golden output:

```
mkdir GoldenOutputsTensor/

golden_output_path (contains augmented images to cross verify correctness of each reader) : ${ROCAL_DATA_PATH}/rocal_data/GoldenOutputsTensor/
```

* To run the test:

`export ROCAL_DATA_PATH=<absolute_path_to_data_directory>`

```bash
./testAllScripts.sh <device_type 0/1/2> <color_format 0/1/2>
```

Device Type

* Option 0 - For only HOST backend
* Option 1 - For only HIP backend
* Option 2 - For both HOST and HIP backend

Color Format

* Option 0 - For only Greyscale inputs
* Option 1 - For only RGB inputs
* Option 2 - For both Greyscale and RGB inputs
