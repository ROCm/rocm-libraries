#!/bin/bash
cwd=$(pwd)
if [ -d build ];then 
    sudo rm -rf ./build/*
else
    mkdir build
fi
cd build || exit
cmake ..
make -j"$(nproc)"

if [[ $ROCAL_DATA_PATH == "" ]]
then 
    echo "Need to export ROCAL_DATA_PATH"
    exit
fi

# Path to inputs and outputs
image_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img/images/
coco_detection_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img/images/
coco_keypoints_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img_keypoints/person_keypoints_10images_val2017/
tf_classification_path=${ROCAL_DATA_PATH}/rocal_data/tf/classification/
tf_detection_path=${ROCAL_DATA_PATH}/rocal_data/tf/detection/
caffe_classification_path=${ROCAL_DATA_PATH}/rocal_data/caffe/classification/
caffe_detection_path=${ROCAL_DATA_PATH}/rocal_data/caffe/detection/
caffe2_classification_path=${ROCAL_DATA_PATH}/rocal_data/caffe2/classification/
caffe2_detection_path=${ROCAL_DATA_PATH}/rocal_data/caffe2/detection/
mxnet_path=${ROCAL_DATA_PATH}/rocal_data/mxnet/
output_path=../rocal_unittest_output_folder_$(date +%Y-%m-%d_%H-%M-%S)/
golden_output_path=${ROCAL_DATA_PATH}/rocal_data/GoldenOutputsTensor/
webdataset_tar_path=${ROCAL_DATA_PATH}/rocal_data/web_dataset/tar_file/
numpy_data_path=${ROCAL_DATA_PATH}/rocal_data/numpy/

display=0
device=0
width=416 
height=416
device_name="host"
rgb_name=("gray" "rgb")
rgb=1
dev_start=0
dev_end=1
rgb_start=0
rgb_end=1

if [ "$#" -gt 0 ]; then 
    if [ "$1" -eq 0 ]; then # For only HOST backend
        dev_start=0
        dev_end=0
    elif [ "$1" -eq 1 ]; then # For only HIP backend
        dev_start=1
        dev_end=1
    elif [ "$1" -eq 2 ]; then # For both HOST and HIP backend
        dev_start=0
        dev_end=1
    fi
fi

if [ "$#" -gt 1 ]; then
    if [ "$2" -eq 0 ]; then # For only Greyscale inputs
        rgb_start=0
        rgb_end=0
    elif [ "$2" -eq 1 ]; then # For only RGB inputs
        rgb_start=1
        rgb_end=1
    elif [ "$2" -eq 2 ]; then # For both RGB and Greyscale inputs
        rgb_start=0
        rgb_end=1
    fi
fi

mkdir "$output_path"

for ((device=dev_start;device<=dev_end;device++))
do 
    if [ $device -eq 1 ]
    then 
        device_name="hip"
        echo "Running HIP Backend..."
    else
        echo "Running HOST Backend..."
    fi
    for ((rgb=rgb_start;rgb<=rgb_end;rgb++))
    do 
        # FileSource Reader
        ./unit_tests 0 "$image_path" "${output_path}LensCorrection_${rgb_name[$rgb]}_${device_name}" $width $height 45 $device $rgb 0 $display
        ./unit_tests 0 "$image_path" "${output_path}Exposure_${rgb_name[$rgb]}_${device_name}" $width $height 46 $device $rgb 0 $display
        ./unit_tests 0 "$image_path" "${output_path}Flip_${rgb_name[$rgb]}_${device_name}" $width $height 47 $device $rgb 0 $display

        # FileSource Reader + partial decoder
        ./unit_tests 1 "$image_path" "${output_path}Snow_${rgb_name[$rgb]}_${device_name}_FileReader_partial" $width $height 41 $device $rgb 0 $display
        ./unit_tests 1 "$image_path" "${output_path}Rain_${rgb_name[$rgb]}_${device_name}_FileReader_partial" $width $height 42 $device $rgb 0 $display
        ./unit_tests 1 "$image_path" "${output_path}SNPNoise_${rgb_name[$rgb]}_${device_name}_FileReader_partial" $width $height 40 $device $rgb 0 $display

        # coco detection
        ./unit_tests 2 "$coco_detection_path" "${output_path}Gamma_${rgb_name[$rgb]}_${device_name}" $width $height 33 $device $rgb 0 $display
        ./unit_tests 2 "$coco_detection_path" "${output_path}Contrast_${rgb_name[$rgb]}_${device_name}" $width $height 34 $device $rgb 0 $display
        # ./unit_tests 2 "$coco_detection_path" "${output_path}Vignette_${rgb_name[$rgb]}_${device_name}" $width $height 38 $device $rgb 0 $display

        # coco detection + partial decoder
        ./unit_tests 3 "$coco_detection_path" "${output_path}Snow_${rgb_name[$rgb]}_${device_name}_coco_partial" $width $height 41 $device $rgb 0 $display
        ./unit_tests 3 "$coco_detection_path" "${output_path}Rain_${rgb_name[$rgb]}_${device_name}_coco_partial" $width $height 42 $device $rgb 0 $display
        ./unit_tests 3 "$coco_detection_path" "${output_path}SNPNoise_${rgb_name[$rgb]}_${device_name}_coco_partial" $width $height 40 $device $rgb 0 $display

        # tf classification
        ./unit_tests 4 "$tf_classification_path" "${output_path}Hue_${rgb_name[$rgb]}_${device_name}" $width $height 48 $device $rgb 0 $display
        ./unit_tests 4 "$tf_classification_path" "${output_path}Saturation_${rgb_name[$rgb]}_${device_name}" $width $height 49 $device $rgb 0 $display
        ./unit_tests 4 "$tf_classification_path" "${output_path}ColorTwist_${rgb_name[$rgb]}_${device_name}" $width $height 50 $device $rgb 0 $display

        # tf detection
        ./unit_tests 5 "$tf_detection_path" "${output_path}SNPNoise_${rgb_name[$rgb]}_${device_name}" $width $height 40 $device $rgb 0 $display
        # ./unit_tests 5 "$tf_detection_path" "${output_path}ColorTemp_${rgb_name[$rgb]}_${device_name}" $width $height 43 $device $rgb 0 $display
        ./unit_tests 5 "$tf_detection_path" "${output_path}Fog_${rgb_name[$rgb]}_${device_name}" $width $height 44 $device $rgb 0 $display

        # caffe classification
        ./unit_tests 6 "$caffe_classification_path" "${output_path}Rotate_${rgb_name[$rgb]}_${device_name}" $width $height 31 $device $rgb 0 $display
        ./unit_tests 6 "$caffe_classification_path" "${output_path}Brightness_${rgb_name[$rgb]}_${device_name}" $width $height 32 $device $rgb 0 $display
        ./unit_tests 6 "$caffe_classification_path" "${output_path}Blend_${rgb_name[$rgb]}_${device_name}" $width $height 36 $device $rgb 0 $display

        # caffe detection
        ./unit_tests 7 "$caffe_detection_path" "${output_path}WarpAffine_${rgb_name[$rgb]}_${device_name}" $width $height 37 $device $rgb 0 $display
        ./unit_tests 7 "$caffe_detection_path" "${output_path}Blur_${rgb_name[$rgb]}_${device_name}" $width $height 35 $device $rgb 0 $display
        ./unit_tests 7 "$caffe_detection_path" "${output_path}Rain_${rgb_name[$rgb]}_${device_name}" $width $height 42 $device $rgb 0 $display

        # caffe2 classification
        ./unit_tests 8 "$caffe2_classification_path" "${output_path}CropCenter_${rgb_name[$rgb]}_${device_name}" $width $height 52 $device $rgb 0 $display
        ./unit_tests 8 "$caffe2_classification_path" "${output_path}ResizeCropMirror_${rgb_name[$rgb]}_${device_name}" $width $height 53 $device $rgb 0 $display
        ./unit_tests 8 "$caffe2_classification_path" "${output_path}Snow_${rgb_name[$rgb]}_${device_name}" $width $height 41 $device $rgb 0 $display

        # caffe2 detection
        ./unit_tests 9 "$caffe2_detection_path" "${output_path}FishEye_${rgb_name[$rgb]}_${device_name}" $width $height 10 $device $rgb 0 $display
        # ./unit_tests 9 "$caffe2_detection_path" "${output_path}Pixelate_${rgb_name[$rgb]}_${device_name}" $width $height 19 $device $rgb 0 $display
        ./unit_tests 9 "$caffe2_detection_path" "${output_path}CropCenterCMN_${rgb_name[$rgb]}_${device_name}" $width $height 55 $device $rgb 0 $display

        # COCO Keypoints
        ./unit_tests 10 "$coco_keypoints_path" "${output_path}SNP_${rgb_name[$rgb]}_${device_name}" 640 480 14 $device $rgb 0 $display
        ./unit_tests 10 "$coco_keypoints_path" "${output_path}Snow_${rgb_name[$rgb]}_${device_name}" 640 480 15 $device $rgb 0 $display
        ./unit_tests 10 "$coco_keypoints_path" "${output_path}Rain_${rgb_name[$rgb]}_${device_name}" 640 480 17 $device $rgb 0 $display

        # mxnet 
        # ./unit_tests 11 "$mxnet_path" "${output_path}Jitter_${rgb_name[$rgb]}_${device_name}" $width $height 39 $device $rgb 0 $display
        # ./unit_tests 11 "$mxnet_path" "${output_path}Pixelate_${rgb_name[$rgb]}_${device_name}" $width $height 19 $device $rgb 0 $display
        ./unit_tests 11 "$mxnet_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_mxnet" $width $height 25 $device $rgb 0 $display

        # Webdataset
        ./unit_tests 12 "$webdataset_tar_path" "${output_path}Normalize_${rgb_name[$rgb]}_${device_name}" $width $height 57 $device $rgb 0 $display
        ./unit_tests 12 "$webdataset_tar_path" "${output_path}Transpose_${rgb_name[$rgb]}_${device_name}" $width $height 58 $device $rgb 0 $display

        # CMN 
        ./unit_tests 0 "$image_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_FileReader" $width $height 25 $device $rgb 0 $display
        ./unit_tests 2 "$coco_detection_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_coco" $width $height 25 $device $rgb 0 $display
        ./unit_tests 4 "$tf_classification_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_tfClassification" $width $height 25 $device $rgb 0 $display
        ./unit_tests 5 "$tf_detection_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_tfDetection" $width $height 25 $device $rgb 0 $display
        ./unit_tests 6 "$caffe_classification_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffeClassification" $width $height 25 $device $rgb 0 $display
        ./unit_tests 7 "$caffe_detection_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffeDetection" $width $height 25 $device $rgb 0 $display
        ./unit_tests 8 "$caffe2_classification_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffe2Classification" $width $height 25 $device $rgb 0 $display
        ./unit_tests 9 "$caffe2_detection_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffe2Detection" $width $height 25 $device $rgb 0 $display
        ./unit_tests 11 "$mxnet_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_mxnet" $width $height 25 $device $rgb 0 $display
        ./unit_tests 12 "$webdataset_tar_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_web_dataset" $width $height 25 $device $rgb 0 $display
        ./unit_tests 13 "$numpy_data_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_numpy" $width $height 25 $device $rgb 0 $display

        # crop
        ./unit_tests 0 "$image_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_FileReader" $width $height 51 $device $rgb 0 $display
        ./unit_tests 2 "$coco_detection_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_coco" $width $height 51 $device $rgb 0 $display
        ./unit_tests 4 "$tf_classification_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_tfClassification" $width $height 51 $device $rgb 0 $display
        ./unit_tests 5 "$tf_detection_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_tfDetection" $width $height 51 $device $rgb 0 $display
        ./unit_tests 6 "$caffe_classification_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffeClassification" $width $height 51 $device $rgb 0 $display
        ./unit_tests 7 "$caffe_detection_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffeDetection" $width $height 51 $device $rgb 0 $display
        ./unit_tests 8 "$caffe2_classification_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffe2Classification" $width $height 51 $device $rgb 0 $display
        ./unit_tests 9 "$caffe2_detection_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffe2Detection" $width $height 51 $device $rgb 0 $display
        ./unit_tests 11 "$mxnet_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_mxnet" $width $height 51 $device $rgb 0 $display
        ./unit_tests 12 "$webdataset_tar_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_web_dataset" $width $height 51 $device $rgb 0 $display
        ./unit_tests 13 "$numpy_data_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_numpy" $width $height 51 $device $rgb 0 $display

        # resize
        # Last two parameters are interpolation type and scaling mode
        ./unit_tests 0 "$image_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_default_FileReader" $width $height 0 $device $rgb 0 $display 1 0 
        ./unit_tests 2 "$coco_detection_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_stretch_coco" $width $height 0 $device $rgb 0 $display 1 1
        ./unit_tests 4 "$tf_classification_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_notsmaller_tfClassification" $width $height 0 $device $rgb 0 $display 1 2
        ./unit_tests 5 "$tf_detection_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_notlarger_tfDetection" $width $height 0 $device $rgb 0 $display 1 3
        ./unit_tests 6 "$caffe_classification_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bicubic_default_caffeClassification" $width $height 0 $device $rgb 0 $display 2 0
        ./unit_tests 7 "$caffe_detection_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_nearestneighbor_default_caffeDetection" $width $height 0 $device $rgb 0 $display 0 0
        ./unit_tests 8 "$caffe2_classification_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_lanczos_default_caffe2Classification" $width $height 0 $device $rgb 0 $display 3 0
        ./unit_tests 9 "$caffe2_detection_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_triangular_default_caffe2Detection" $width $height 0 $device $rgb 0 $display 5 0
        ./unit_tests 11 "$mxnet_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_gaussian_default_mxnet" $width $height 0 $device $rgb 0 $display 4 0
        ./unit_tests 12 "$webdataset_tar_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_triangular_notsmaller_web_dataset" $width $height 0 $device $rgb 0 $display 5 2

        # code coverage tests
        ./unit_tests 2 "$coco_detection_path" "${output_path}CropResizeRandom_${rgb_name[$rgb]}_${device_name}" $width $height 1 $device $rgb 1 $display
        ./unit_tests 2 "$coco_detection_path" "${output_path}RotateRandom_${rgb_name[$rgb]}_${device_name}" $width $height 2 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}BrightnessRandom_${rgb_name[$rgb]}_${device_name}" $width $height 3 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}GammaRandom_${rgb_name[$rgb]}_${device_name}" $width $height 4 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}ContrastRandom_${rgb_name[$rgb]}_${device_name}" $width $height 5 $device $rgb 1 $display
        ./unit_tests 2 "$coco_detection_path" "${output_path}FlipRandom_${rgb_name[$rgb]}_${device_name}" $width $height 6 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}BlurRandom_${rgb_name[$rgb]}_${device_name}" $width $height 7 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}BlendRandom_${rgb_name[$rgb]}_${device_name}" $width $height 8 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}WarpAffineRandom_${rgb_name[$rgb]}_${device_name}" $width $height 9 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}SNPNoise_${rgb_name[$rgb]}_${device_name}" $width $height 13 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}Snow_${rgb_name[$rgb]}_${device_name}" $width $height 14 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}Rain_${rgb_name[$rgb]}_${device_name}" $width $height 15 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}FogRandom_${rgb_name[$rgb]}_${device_name}" $width $height 17 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}LensCorrectionRandom_${rgb_name[$rgb]}_${device_name}" $width $height 18 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}ExposureRandom_${rgb_name[$rgb]}_${device_name}" $width $height 20 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}HueRandom_${rgb_name[$rgb]}_${device_name}" $width $height 21 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}SaturationRandom_${rgb_name[$rgb]}_${device_name}" $width $height 22 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}Copy_${rgb_name[$rgb]}_${device_name}" $width $height 23 $device $rgb 1 $display
        ./unit_tests 0 "$image_path" "${output_path}ColorTwistRandom_${rgb_name[$rgb]}_${device_name}" $width $height 24 $device $rgb 1 $display
        ./unit_tests 2 "$coco_detection_path" "${output_path}CropRandom_${rgb_name[$rgb]}_${device_name}" $width $height 26 $device $rgb 1 $display
        ./unit_tests 2 "$coco_detection_path" "${output_path}ResizeCropMirrorRandom_${rgb_name[$rgb]}_${device_name}" $width $height 27 $device $rgb 1 $display
        ./unit_tests 2 "$coco_detection_path" "${output_path}CropResize_${rgb_name[$rgb]}_${device_name}" $width $height 30 $device $rgb 1 $display
        ./unit_tests 2 "$coco_detection_path" "${output_path}SSDRandomCrop_${rgb_name[$rgb]}_${device_name}" $width $height 54 $device $rgb 1 $display
        ./unit_tests 2 "$coco_detection_path" "${output_path}ResizeMirrorNormalize_${rgb_name[$rgb]}_${device_name}" $width $height 56 $device $rgb 1 $display
        ./unit_tests 2 "$coco_detection_path" "${output_path}RandomCrop_${rgb_name[$rgb]}_${device_name}" $width $height 59 $device $rgb 1 $display

    done
done

pwd

# Run python script to compare rocAL outputs with golden ouptuts
python3 "$cwd"/pixel_comparison/image_comparison.py "$golden_output_path" "$output_path"
