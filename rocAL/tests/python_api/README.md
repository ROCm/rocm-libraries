## To test pybind with GPU backend, `dlpack` is required.

### Install dlpack

* Ubuntu:

```
sudo apt install libdlpack-dev
```

* SLES:

```
sudo zypper install dlpack-devel
```

* Redhat:

```
sudo yum install https://rpmfind.net/linux/opensuse/tumbleweed/repo/oss/x86_64/dlpack-devel-0.8-1.5.x86_64.rpm
```

## Prepare dataset and Set environmental variables

* The data needs to be organized in separate folders under `rocal_data` directory for each reader and a golden output directory for the verification.

```
mkdir rocal_data/
cd rocal_data

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
numpy_data_path (numpy .npy file) : ${ROCAL_DATA_PATH}/rocal_data/numpy/
```

* Golden output:

```
golden_output_path (contains augmented images to cross verify correctness of each reader) : ${ROCAL_DATA_PATH}/rocal_data/GoldenOutputsTensor/
```

`export ROCAL_DATA_PATH=<absolute_path_to_data_directory>`

### NOTE: Refer to parse_config.py for more info on other args. This script is used with the `readers_test_file.sh` and `unit_tests.sh`

## Reader Pipeline Tests

* To test all the reader pipelines with a single script. The `readers_test_file.sh` tests the following cases:
  * unit test
  * coco reader
  * caffe reader
  * caffe2 reader
  * tf classification reader
  * tf detection reader
  * video reader
  * webdataset reader
  * numpy reader
* The default value of number of gpu's is "1" & display is "ON" by default
`./readers_test_file.sh`

### Options

* To test a single reader / multiple reader pipelines: Use the same script `readers_test_file.sh` as above and make the respective " Pipeline " to test equal to "1"

```shell
unit_test = 1
coco_reader = 0
```

* Example : To run COCO Pipeline

```shell
unit_test=0
coco_reader=1
caffe_reader=0
caffe2_reader=0
tf_classification_reader=0
tf_detection_reader=0
```

* To set options:
  * Number of GPUs `-n`
  * display `-d`
  * backend `-b`

```shell
./readers_test_file.sh -n <number_of_gpus> -d <true/false>
./readers_test_file.sh -n "1" -d "false" -b "cpu"
```

## Augmentation + Reader Tests

* This runs all augmentations and readers and compares them with golden outputs for both backends (HIP/HOST)

`./unit_tests.sh`

* This test also runs the `image_comparison.py` script at the end to compare the results with the golden outputs.
### Command line for individual files

* Test a single reader pipeline
* Example: COCO Pipeline

```shell
    # Mention the number of gpus
    gpus_per_node=4

    # Mention Batch Size
    batch_size=10

    # python version
    ver=$(python -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\.\2/')

    # Mention dataset_path
    data_dir=$ROCAL_DATA_PATH/coco/coco_10_img/val_10images_2017/

    # Mention json path
    json_path=$ROCAL_DATA_PATH/coco/coco_10_img/annotations/instances_val2017.json

    # coco_reader.py
    # By default : cpu backend, NCHW format , fp32
    # Annotation must be a json file

    # For printing all the arguments that can be passed from user
    python$ver coco_reader.py -h

    python$ver coco_reader.py --image-dataset-path $data_dir --json-path $json_path --batch-size $batch_size --display --rocal-gpu --NHWC \
        --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 1 2>&1 | tee -a run.log.coco_reader.txt
```

## Decoder Test

This test runs a simple decoder pipeline making use of an image file reader and resizing the given input images to 300x300.

It uses the [AMD-TinyDataSet](../../data/images/AMD-tinyDataSet/) by default, unless otherwise specified by user.

### Options

```bash
backend: <cpu/gpu>
Input image folder: <Path to image folder>
```

* Usage:

```shell
python3 decoder.py
python3 decoder.py gpu <path to image folder>
```

## External source reader test

This test runs a pipeline making use of the external source reader in 3 different modes. It uses coco2017 images by default.

* Mode 0 - Filename
* Mode 1 - Raw compressed images
* Mode 2 - JPEG reader with OpenCV

* Usage:

```bash
python3 external_source_reader.py
```

## Audio Unit Test

To run the Audio unit test with all test cases. Follow the steps below:

### Prepare dataset

The data needs to be organized in a separate `audio` folder under `rocal_data` directory.

```
mkdir rocal_data/
cd rocal_data

audio_path (.wav files and corresponding .wav_file_list.txt file) : ${ROCAL_DATA_PATH}/rocal_data/audio/
```

* Golden output:

```
mkdir GoldenOutputsTensor/reference_outputs_audio/

golden_output_path (contains augmented .bin to cross verify correctness of each augmentation) : ${ROCAL_DATA_PATH}/rocal_data/GoldenOutputsTensor/reference_outputs_audio/
```


```bash
export ROCAL_DATA_PATH=<absolute_path_to_data_directory>
```

To run the audio unit test and verify the correctness of the outputs

```bash
python3 audio_unit_test.py
```
To pass the audio data path, batch size, and run a particular test case use the following command

```bash
python3 audio_unit_test.py --audio_path=<path_to_data> --test_case <case(0-11)> --batch-size <batch_size>
```

**Available Test Cases**
* Case 0 - Audio Decoder
* Case 1 - PreEmphasis Filter
* Case 2 - Spectrogram
* Case 3 - Downmix
* Case 4 - ToDecibels
* Case 5 - Resample
* Case 6 - TensorAddTensor
* Case 7 - TensorMulScalar
* Case 8 - NonSilentRegionDetection
* Case 9 - Slice
* Case 10 - MelFilterBank
* Case 11 - Normalize
