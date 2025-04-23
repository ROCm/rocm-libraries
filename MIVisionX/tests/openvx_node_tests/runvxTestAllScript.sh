#!/bin/bash

############# Help and Syntax #############

# Help

# The runvxTestAllScript.sh bash script runs runvx for AMD OpenVX functionalities in HOST/OCL/HIP backends.
# - It can optionally run tests for all kernels or a single kernel separately.
# - It can optionally generate .bin dumps for the image inputs/outputs.
# - It can optionally run HOST/OCL/HIP backends and also run a OCLvsHIP test to compare OCLvsHIP outputs and report inconsistencies (for all kernels or a single kernel separately).
# - It requires a user given width, height to run tests.
# - It requires a primary RunVX path while running HOST/OCL/HIP backends and optionally a secondary RunVX path while running OCLvsHIP comparison.

# Syntax

# Syntax: `./runvxTestAllScript.sh <W> <H> <D> <N> <B> <P> <S> <O>` where:
# ```
# - W     WIDTH of image in pixels
# - H     HEIGHT of image pixels
# - D     Output bin DUMP toggle (1 = True, 0 = False)
# - N     NAME of kernel to run ('ALL' = run all available kernels, '<kernel name>' = run specific kernel name from the list of kernels in this script)
# - B     BACKEND (HOST = On CPU / OCL = On GPU with OpenCL backend / HIP = On GPU with HIP backend / OCLvsHIP = On GPU with OCLvsHIP output comparison)
# - P     PRIMARY RunVX path (for MIVisionX built with HOST/OCL/HIP backend)
# - S     SECONDARY RunVX path (Required only for OCLvsHIP comparison. Primary path used for OCL backend, Secondary path used for HIP backend)
# - O     RunVX path OVERRIDE - Optional parameter (0 = Use primary/secondary RunVX paths (DEFAULT), 1 = make-install required backend and then run tests, 2 = Clean build-make-install required backend and then run tests)
# ```

############# Help and Syntax #############





############# Edit GDF path and kernel names here #############

GDF_PATH="kernelGDFs/"

GDF_ARITHMETIC_LIST="AbsDiff_U8_U8U8
AbsDiff_S16_S16S16_Sat
Add_U8_U8U8_Wrap
Add_U8_U8U8_Sat
Add_S16_U8U8_Wrap
Add_S16_S16U8_Wrap
Add_S16_S16U8_Sat
Add_S16_S16S16_Wrap
Add_S16_S16S16_Sat
Sub_U8_U8U8_Wrap
Sub_U8_U8U8_Sat
Sub_S16_U8U8_Wrap
Sub_S16_S16U8_Wrap
Sub_S16_S16U8_Sat
Sub_S16_U8S16_Wrap
Sub_S16_U8S16_Sat
Sub_S16_S16S16_Wrap
Sub_S16_S16S16_Sat
Mul_U8_U8U8_Wrap_Trunc
Mul_U8_U8U8_Wrap_Round
Mul_U8_U8U8_Sat_Trunc
Mul_U8_U8U8_Sat_Round
Mul_S16_U8U8_Wrap_Trunc
Mul_S16_U8U8_Wrap_Round
Mul_S16_U8U8_Sat_Trunc
Mul_S16_U8U8_Sat_Round
Mul_S16_S16U8_Wrap_Trunc
Mul_S16_S16U8_Wrap_Round
Mul_S16_S16U8_Sat_Trunc
Mul_S16_S16U8_Sat_Round
Mul_S16_S16S16_Wrap_Trunc
Mul_S16_S16S16_Wrap_Round
Mul_S16_S16S16_Sat_Trunc
Mul_S16_S16S16_Sat_Round
Magnitude_S16_S16S16
Phase_U8_S16S16
WeightedAverage_U8_U8U8"

GDF_LOGICAL_LIST="And_U8_U8U8
And_U8_U8U1
And_U8_U1U8
And_U8_U1U1
And_U1_U8U8
And_U1_U8U1
And_U1_U1U8
And_U1_U1U1
Or_U8_U8U8
Or_U8_U8U1
Or_U8_U1U8
Or_U8_U1U1
Or_U1_U8U8
Or_U1_U8U1
Or_U1_U1U8
Or_U1_U1U1
Xor_U8_U8U8
Xor_U8_U8U1
Xor_U8_U1U8
Xor_U8_U1U1
Xor_U1_U8U8
Xor_U1_U8U1
Xor_U1_U1U8
Xor_U1_U1U1
Not_U8_U8
Not_U1_U8
Not_U8_U1
Not_U1_U1"

GDF_COLOR_LIST="Lut_U8_U8
ColorDepth_U8_S16_Wrap
ColorDepth_U8_S16_Sat
ColorDepth_S16_U8
ChannelExtract_U8_U16_Pos0
ChannelExtract_U8_U16_Pos1
ChannelExtract_U8_U24_Pos0
ChannelExtract_U8_U24_Pos1
ChannelExtract_U8_U24_Pos2
ChannelExtract_U8_U32_Pos0
ChannelExtract_U8_U32_Pos1
ChannelExtract_U8_U32_Pos2
ChannelExtract_U8_U32_Pos3
ChannelExtract_U8U8U8_U24
ChannelExtract_U8U8U8_U32
ChannelExtract_U8U8U8U8_U32
ChannelCombine_U16_U8U8
ChannelCombine_U32_U8U8U8_UYVY
ChannelCombine_U32_U8U8U8_YUYV
ChannelCombine_U24_U8U8U8_RGB
ChannelCombine_U32_U8U8U8U8_RGBX
ColorConvert_RGB_RGBX
ColorConvert_RGB_UYVY
ColorConvert_RGB_YUYV
ColorConvert_RGB_IYUV
ColorConvert_RGB_NV12
ColorConvert_RGB_NV21
ColorConvert_RGBX_RGB
ColorConvert_RGBX_UYVY
ColorConvert_RGBX_YUYV
ColorConvert_RGBX_IYUV
ColorConvert_RGBX_NV12
ColorConvert_RGBX_NV21
ColorConvert_IYUV_RGB
ColorConvert_IYUV_RGBX
FormatConvert_IYUV_UYVY
FormatConvert_IYUV_YUYV
ColorConvert_NV12_RGB
ColorConvert_NV12_RGBX
FormatConvert_NV12_UYVY
FormatConvert_NV12_YUYV
ColorConvert_YUV4_RGB
ColorConvert_YUV4_RGBX
FormatConvert_IUV_UV12
FormatConvert_UV12_IUV
FormatConvert_UV_UV12
ScaleUp2x2_U8_U8
"

GDF_FILTER_LIST="Box_U8_U8_3x3
Dilate_U8_U8_3x3
Dilate_U1_U8_3x3
Dilate_U8_U1_3x3
Dilate_U1_U1_3x3
Erode_U8_U8_3x3
Erode_U1_U8_3x3
Erode_U8_U1_3x3
Erode_U1_U1_3x3
Median_U8_U8_3x3
Gaussian_U8_U8_3x3
ScaleGaussianHalf_U8_U8_3x3
ScaleGaussianHalf_U8_U8_5x5
Convolve_U8_U8_3x3
Convolve_S16_U8_3x3
Sobel_S16S16_U8_3x3_GXY
Sobel_S16_U8_3x3_GX
Sobel_S16_U8_3x3_GY
"

GDF_STATISTICAL_LIST="Threshold_U8_U8_Binary
Threshold_U8_U8_Range
Threshold_U1_U8_Binary
Threshold_U1_U8_Range
Threshold_U8_S16_Binary
Threshold_U8_S16_Range
"

GDF_GEOMETRIC_LIST="ScaleImage_U8_U8_Nearest
ScaleImage_U8_U8_Bilinear
ScaleImage_U8_U8_Bilinear_Replicate
ScaleImage_U8_U8_Bilinear_Constant
ScaleImage_U8_U8_Area
WarpAffine_U8_U8_Nearest
WarpAffine_U8_U8_Nearest_Constant
WarpAffine_U8_U8_Bilinear
WarpAffine_U8_U8_Bilinear_Constant
WarpPerspective_U8_U8_Nearest
WarpPerspective_U8_U8_Nearest_Constant
WarpPerspective_U8_U8_Bilinear
WarpPerspective_U8_U8_Bilinear_Constant
Remap_U8_U8_Nearest
Remap_U8_U8_Nearest_Constant
Remap_U8_U8_Bilinear
Remap_U8_U8_Bilinear_Constant
"

GDF_VISION_LIST="
Canny_3x3_L1NORM
Canny_3x3_L2NORM
FastCorners_XY_U8_Supression
FastCorners_XY_U8_NoSupression
Harris_3x3
"

############# Edit GDF path and kernel names here #############










############# Need not edit - Utility functions #############

# generator function to auto-generate gdfs for different image sizes, with/without binary dump

generator() {

    CATEGORY=$1

    if [ "$DUMP" -eq 0 ]; then
        cp "$GDF_PATH/$CATEGORY/$GDF.gdf" "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
    else
        while IFS= read -r LINE || [[ -n "$LINE" ]]; do

            if [[ $LINE == $STRING_I1* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"input_1".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
            elif [[ $LINE == $STRING_I2* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"input_2".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
            elif [[ $LINE == $STRING_I3* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"input_3".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
            elif [[ $LINE == $STRING_I4* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"input_4".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"

            elif [[ $LINE == $STRING_O1* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"output_1".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
            elif [[ $LINE == $STRING_O2* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"output_2".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
            elif [[ $LINE == $STRING_O3* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"output_3".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
            elif [[ $LINE == $STRING_O4* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"output_4".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"

            else
                echo "$LINE" >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
            fi
        done < "$GDF_PATH/$CATEGORY/$GDF.gdf"
    fi

    sed -i "s/1920,1080/$WIDTH,$HEIGHT/" "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
    sed -i "s/960,1080/$HALF_WIDTH,$HEIGHT/" "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
    sed -i "s/960,540/$HALF_WIDTH,$HALF_HEIGHT/" "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
    sed -i "s/3840,1080/$DOUBLE_WIDTH,$HEIGHT/" "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
}





# case_tester function to test each case in each functionality group

case_tester() {

    AFFINITY=$1

    printf "\n\n---------------------------------------------"
    printf "\nRunning ARITHMETIC GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF in $GDF_ARITHMETIC_LIST;
    do
        printf "\nRunning $GDF...\n"
        generator "arithmetic"
        "$RUNVX_PATH"runvx -frames:1 -affinity:"$AFFINITY" -dump-profile "$GENERATED_GDF_PATH"/arithmetic/"$GDF".gdf
    done

    printf "\n\n---------------------------------------------"
    printf "\nRunning LOGICAL GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF in $GDF_LOGICAL_LIST;
    do
        printf "\nRunning $GDF...\n"
        generator "logical"
        "$RUNVX_PATH"runvx -frames:1 -affinity:"$AFFINITY" -dump-profile "$GENERATED_GDF_PATH"/logical/"$GDF".gdf
    done

    printf "\n\n---------------------------------------------"
    printf "\nRunning COLOR GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF in $GDF_COLOR_LIST;
    do
        printf "\nRunning $GDF...\n"
        generator "color"
        "$RUNVX_PATH"runvx -frames:1 -affinity:"$AFFINITY" -dump-profile "$GENERATED_GDF_PATH"/color/"$GDF".gdf
    done

    printf "\n\n---------------------------------------------"
    printf "\nRunning FILTER GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF in $GDF_FILTER_LIST;
    do
        printf "\nRunning $GDF...\n"
        generator "filter"
        "$RUNVX_PATH"runvx -frames:1 -affinity:"$AFFINITY" -dump-profile "$GENERATED_GDF_PATH"/filter/"$GDF".gdf
    done

    printf "\n\n---------------------------------------------"
    printf "\nRunning STATISTICAL GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF in $GDF_STATISTICAL_LIST;
    do
        printf "\nRunning $GDF...\n"
        generator "statistical"
        "$RUNVX_PATH"runvx -frames:1 -affinity:"$AFFINITY" -dump-profile "$GENERATED_GDF_PATH"/statistical/"$GDF".gdf
    done

    printf "\n\n---------------------------------------------"
    printf "\nRunning GEOMETRIC GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF in $GDF_GEOMETRIC_LIST;
    do
        printf "\nRunning $GDF...\n"
        generator "geometric"
        "$RUNVX_PATH"runvx -frames:1 -affinity:"$AFFINITY" -dump-profile "$GENERATED_GDF_PATH"/geometric/"$GDF".gdf
    done

    printf "\n\n---------------------------------------------"
    printf "\nRunning VISION GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF in $GDF_VISION_LIST;
    do
        printf "\nRunning $GDF...\n"
        generator "vision"
        "$RUNVX_PATH"runvx -frames:1 -affinity:"$AFFINITY" -dump-profile "$GENERATED_GDF_PATH"/vision/"$GDF".gdf
    done
}





# case_tester_single function to test a single case

case_tester_single() {

    AFFINITY=$1

    GDF=$KERNEL_NAME
    printf "\nRunning $GDF...\n"
    generator "$GROUP"
    "$RUNVX_PATH"runvx -frames:1 -affinity:"$AFFINITY" -dump-profile "$GENERATED_GDF_PATH"/"$GROUP"/"$GDF".gdf
}

############# Need not edit - Utility functions #############










############# Need not edit - Main script #############

# Input parameters

if (( "$#" < 6 )); then
    echo
    echo "The runvxTestAllScript.sh bash script runs runvx for AMD OpenVX functionalities in HOST/OCL/HIP backends."
    echo "    - It can optionally run tests for all kernels or a single kernel separately."
    echo "    - It can optionally generate .bin dumps for the image inputs/outputs."
    echo "    - It can optionally run HOST/OCL/HIP backends and also run a OCLvsHIP test to compare OCLvsHIP outputs and report inconsistencies (for all kernels or a single kernel separately)."
    echo "    - It requires a user given width, height to run tests."
    echo "    - It requires a primary RunVX path while running HOST/OCL/HIP backends and optionally a secondary RunVX path while running OCLvsHIP comparison."
    echo
    echo "Syntax: ./runvxTestAllScript.sh <W> <H> <D> <N> <B> <P> <S> <O>"
    echo "W     WIDTH of image in pixels"
    echo "H     HEIGHT of image pixels"
    echo "D     Output bin DUMP toggle (1 = True, 0 = False)"
    echo "N     NAME of kernel to run ('ALL' = run all available kernels, '<kernel name>' = run specific kernel name from the list of kernels in this script)"
    echo "B     BACKEND (HOST = On CPU / OCL = On GPU with OpenCL backend / HIP = On GPU with HIP backend / OCLvsHIP = On GPU with OCLvsHIP output comparison)"
    echo "P     PRIMARY RunVX path (for MIVisionX built with HOST/OCL/HIP backend)"
    echo "S     SECONDARY RunVX path (Required only for OCLvsHIP comparison. Primary path used for OCL backend, Secondary path used for HIP backend)"
    echo "O     RunVX path OVERRIDE - Optional parameter (0 = Use primary/secondary RunVX paths (DEFAULT), 1 = make-install required backend and then run tests, 2 = Clean build-make-install required backend and then run tests)"
    exit 1
fi



if [ "$3" -ne 0 ]; then
    if [ "$3" -ne 1 ]; then
        echo "The bin dump toggle must be 0/1!"
        exit 1
    fi
fi

if [ "$5" != "HOST" ] && [ "$5" != "OCL" ] && [ "$5" != "HIP" ]  && [ "$5" != "OCLvsHIP" ]; then
    echo "The backend must be either HOST/OCL/HIP/OCLvsHIP!"
    exit 1
fi

if [[ ! -f "$6/runvx" ]]; then
    printf "\n$6/runvx does not exist!\n"
    exit 1
else
    printf "\n$6/runvx FOUND!\n"
fi

if [ "$7" != "" ]; then
    if [ "$5" != "OCLvsHIP" ]; then
        echo "SECONDARY RunVX path is taken only if backend is OCLvsHIP!"
        exit 1
    else
        if [[ ! -f "$7/runvx" ]]; then
            printf "\n$7/runvx does not exist!\n"
            exit 1
        else
            printf "\n$7/runvx FOUND!\n"
        fi
    fi
else
    if [ "$5" = "OCLvsHIP" ]; then
        echo "Invalid SECONDARY RunVX path!"
        exit 1
    fi
fi

if [ "$8" != "0" ] && [ "$8" != "1" ] && [ "$8" != "2" ] && [ "$8" != "" ]; then
    echo "The optional override must be 0/1/2!"
    exit 1
fi


WIDTH="$1"
HEIGHT="$2"
DUMP="$3"
KERNEL_NAME="$4"
BACKEND_TYPE="$5"
PRIMARY_RUNVX_PATH="$6"
SECONDARY_RUNVX_PATH="$7"

printf "\nBACKEND/S TO RUN ----->>>> $BACKEND_TYPE\n"

OVERRIDE=""
if [ "$8" = "" ]; then
    OVERRIDE="0"
else
    OVERRIDE="$8"
fi

FLAG=""
GROUP=""
if [ "$KERNEL_NAME" != "ALL" ]; then
    if [ "$KERNEL_NAME" = "" ]; then
        echo "The kernel name should either be 'all' or a valid name from the list in the script!"
        exit 1
    fi
    [[ $GDF_ARITHMETIC_LIST =~ (^|[[:space:]])$KERNEL_NAME($|[[:space:]]) ]] && FLAG="1" || FLAG="0"
    if [ "$FLAG" -eq 1 ]; then
        GROUP="arithmetic"
    else
        [[ $GDF_LOGICAL_LIST =~ (^|[[:space:]])$KERNEL_NAME($|[[:space:]]) ]] && FLAG="1" || FLAG="0"
        if [ "$FLAG" -eq 1 ]; then
            GROUP="logical"
        else
            [[ $GDF_COLOR_LIST =~ (^|[[:space:]])$KERNEL_NAME($|[[:space:]]) ]] && FLAG="1" || FLAG="0"
            if [ "$FLAG" -eq 1 ]; then
                GROUP="color"
            else
                [[ $GDF_FILTER_LIST =~ (^|[[:space:]])$KERNEL_NAME($|[[:space:]]) ]] && FLAG="1" || FLAG="0"
                if [ "$FLAG" -eq 1 ]; then
                    GROUP="filter"
                else
                    [[ $GDF_GEOMETRIC_LIST =~ (^|[[:space:]])$KERNEL_NAME($|[[:space:]]) ]] && FLAG="1" || FLAG="0"
                    if [ "$FLAG" -eq 1 ]; then
                        GROUP="geometric"
                    else
                        [[ $GDF_STATISTICAL_LIST =~ (^|[[:space:]])$KERNEL_NAME($|[[:space:]]) ]] && FLAG="1" || FLAG="0"
                        if [ "$FLAG" -eq 1 ]; then
                            GROUP="statistical"
                        else
                            [[ $GDF_VISION_LIST =~ (^|[[:space:]])$KERNEL_NAME($|[[:space:]]) ]] && FLAG="1" || FLAG="0"
                            if [ "$FLAG" -eq 1 ]; then
                                GROUP="vision"
                            else
                                echo "The kernel name $KERNEL_NAME is not a valid name from the list in the script!"
                                exit 1
                            fi
                        fi
                    fi
                fi
            fi
        fi
    fi
fi

HALF_WIDTH=$(( "$WIDTH" / 2))
HALF_HEIGHT=$(( "$HEIGHT" / 2))
DOUBLE_WIDTH=$(( "$WIDTH" \* 2))
STRING_I1="data input_1"
STRING_I2="data input_2"
STRING_I3="data input_3"
STRING_I4="data input_4"
STRING_O1="data output_1"
STRING_O2="data output_2"
STRING_O3="data output_3"
STRING_O4="data output_4"
RUNVX_PATH=""





# Running kernels for HOST backend

rm -rvf dumpsHOST
rm -rvf generatedGDFsHOST
if [ "$BACKEND_TYPE" = "HOST" ]; then
    mkdir generatedGDFsHOST
    GENERATED_GDF_PATH="generatedGDFsHOST"

    if [ "$OVERRIDE" = "2" ]; then
        cd ../../
        rm -rvf build_host
        mkdir build_host
        cd build_host
        cmake ..
        sudo make -j20 install
        export LD_LIBRARY_PATH="/opt/rocm/lib"
        cd ../tests/openvx_node_tests
    elif [ "$OVERRIDE" = "1" ]; then
        cd ../../build_host
        sudo make -j20 install
        export LD_LIBRARY_PATH="/opt/rocm/lib"
        cd ../tests/openvx_node_tests
    else
        RUNVX_PATH="$PRIMARY_RUNVX_PATH/"
        export LD_LIBRARY_PATH="$PRIMARY_RUNVX_PATH/../lib"
    fi
    echo "$LD_LIBRARY_PATH"

    if [ "$KERNEL_NAME" = "ALL" ]; then
        mkdir generatedGDFsHOST/arithmetic generatedGDFsHOST/logical generatedGDFsHOST/color generatedGDFsHOST/filter generatedGDFsHOST/geometric generatedGDFsHOST/statistical generatedGDFsHOST/vision
        case_tester "CPU"
        mkdir dumpsHOST
        mkdir dumpsHOST/arithmetic dumpsHOST/logical dumpsHOST/color dumpsHOST/filter dumpsHOST/geometric dumpsHOST/statistical dumpsHOST/vision
        mv "$GENERATED_GDF_PATH"/arithmetic/agoKernel_* dumpsHOST/arithmetic
        mv "$GENERATED_GDF_PATH"/logical/agoKernel_* dumpsHOST/logical
        mv "$GENERATED_GDF_PATH"/color/agoKernel_* dumpsHOST/color
        mv "$GENERATED_GDF_PATH"/filter/agoKernel_* dumpsHOST/filter
        mv "$GENERATED_GDF_PATH"/geometric/agoKernel_* dumpsHOST/geometric
        mv "$GENERATED_GDF_PATH"/statistical/agoKernel_* dumpsHOST/statistical
        mv "$GENERATED_GDF_PATH"/vision/agoKernel_* dumpsHOST/vision
    else
        mkdir generatedGDFsHOST/$GROUP
        case_tester_single "CPU"
        mkdir dumpsHOST
        mkdir dumpsHOST/$GROUP
        mv "$GENERATED_GDF_PATH"/"$GROUP"/agoKernel_* dumpsHOST/"$GROUP"
    fi
fi





# Running kernels for OCL backend

rm -rvf dumpsOCL
rm -rvf generatedGDFsOCL
if [ "$BACKEND_TYPE" = "OCL" ] || [ "$BACKEND_TYPE" = "OCLvsHIP" ]; then
    mkdir generatedGDFsOCL
    GENERATED_GDF_PATH="generatedGDFsOCL"

    if [ "$OVERRIDE" = "2" ]; then
        cd ../../
        rm -rvf build_ocl
        mkdir build_ocl
        cd build_ocl
        cmake ..
        sudo make -j20 install
        export LD_LIBRARY_PATH="/opt/rocm/lib"
        cd ../tests/openvx_node_tests
    elif [ "$OVERRIDE" = "1" ]; then
        cd ../../build_ocl
        sudo make -j20 install
        export LD_LIBRARY_PATH="/opt/rocm/lib"
        cd ../tests/openvx_node_tests
    else
        RUNVX_PATH="$PRIMARY_RUNVX_PATH/"
        export LD_LIBRARY_PATH="$PRIMARY_RUNVX_PATH/../lib"
    fi
    echo "$LD_LIBRARY_PATH"

    if [ "$KERNEL_NAME" = "ALL" ]; then
        mkdir generatedGDFsOCL/arithmetic generatedGDFsOCL/logical generatedGDFsOCL/color generatedGDFsOCL/filter generatedGDFsOCL/geometric generatedGDFsOCL/statistical generatedGDFsOCL/vision
        case_tester "GPU"
        mkdir dumpsOCL
        mkdir dumpsOCL/arithmetic dumpsOCL/logical dumpsOCL/color dumpsOCL/filter dumpsOCL/geometric dumpsOCL/statistical dumpsOCL/vision
        mv "$GENERATED_GDF_PATH"/arithmetic/agoKernel_* dumpsOCL/arithmetic
        mv "$GENERATED_GDF_PATH"/logical/agoKernel_* dumpsOCL/logical
        mv "$GENERATED_GDF_PATH"/color/agoKernel_* dumpsOCL/color
        mv "$GENERATED_GDF_PATH"/filter/agoKernel_* dumpsOCL/filter
        mv "$GENERATED_GDF_PATH"/geometric/agoKernel_* dumpsOCL/geometric
        mv "$GENERATED_GDF_PATH"/statistical/agoKernel_* dumpsOCL/statistical
        mv "$GENERATED_GDF_PATH"/vision/agoKernel_* dumpsOCL/vision
    else
        mkdir generatedGDFsOCL/$GROUP
        case_tester_single "GPU"
        mkdir dumpsOCL
        mkdir dumpsOCL/$GROUP
        mv "$GENERATED_GDF_PATH"/"$GROUP"/agoKernel_* dumpsOCL/"$GROUP"
    fi
fi





# Running kernels for HIP backend

rm -rvf dumpsHIP
rm -rvf generatedGDFsHIP
if [ "$BACKEND_TYPE" = "HIP" ] || [ "$BACKEND_TYPE" = "OCLvsHIP" ]; then
    mkdir generatedGDFsHIP
    GENERATED_GDF_PATH="generatedGDFsHIP"

    if [ "$OVERRIDE" = "2" ]; then
        cd ../../
        rm -rvf build_hip
        mkdir build_hip
        cd build_hip
        cmake -DBACKEND=HIP ..
        sudo make -j20 install
        export LD_LIBRARY_PATH="/opt/rocm/lib"
        cd ../tests/openvx_node_tests
    elif [ "$OVERRIDE" = "1" ]; then
        cd ../../build_hip
        sudo make -j20 install
        export LD_LIBRARY_PATH="/opt/rocm/lib"
        cd ../tests/openvx_node_tests
    else
        if [ "$BACKEND_TYPE" = "HIP" ]; then
            RUNVX_PATH="$PRIMARY_RUNVX_PATH/"
            export LD_LIBRARY_PATH="$PRIMARY_RUNVX_PATH/../lib"
        else
            RUNVX_PATH="$SECONDARY_RUNVX_PATH/"
            export LD_LIBRARY_PATH="$SECONDARY_RUNVX_PATH/../lib"
        fi
    fi
    echo "$LD_LIBRARY_PATH"

    if [ "$KERNEL_NAME" = "ALL" ]; then
        mkdir generatedGDFsHIP/arithmetic generatedGDFsHIP/logical generatedGDFsHIP/color generatedGDFsHIP/filter generatedGDFsHIP/geometric generatedGDFsHIP/statistical generatedGDFsHIP/vision
        case_tester "GPU"
        mkdir dumpsHIP
        mkdir dumpsHIP/arithmetic dumpsHIP/logical dumpsHIP/color dumpsHIP/filter dumpsHIP/geometric dumpsHIP/statistical dumpsHIP/vision
        mv "$GENERATED_GDF_PATH"/arithmetic/agoKernel_* dumpsHIP/arithmetic
        mv "$GENERATED_GDF_PATH"/logical/agoKernel_* dumpsHIP/logical
        mv "$GENERATED_GDF_PATH"/color/agoKernel_* dumpsHIP/color
        mv "$GENERATED_GDF_PATH"/filter/agoKernel_* dumpsHIP/filter
        mv "$GENERATED_GDF_PATH"/geometric/agoKernel_* dumpsHIP/geometric
        mv "$GENERATED_GDF_PATH"/statistical/agoKernel_* dumpsHIP/statistical
        mv "$GENERATED_GDF_PATH"/vision/agoKernel_* dumpsHIP/vision
    else
        mkdir generatedGDFsHIP/$GROUP
        case_tester_single "GPU"
        mkdir dumpsHIP
        mkdir dumpsHIP/$GROUP
        mv "$GENERATED_GDF_PATH"/"$GROUP"/agoKernel_* dumpsHIP/"$GROUP"
    fi
fi





# OCLvsHIP output match check - by running a diff check on all OCLvsHIP bin dumps and flag any inconsistencies in outputs

if [ "$BACKEND_TYPE" = "OCLvsHIP" ]; then

    if [ "$DUMP" -eq 1 ]; then

        OUTPUT_BIN_LIST="arithmetic/agoKernel_AbsDiff_U8_U8U8_output_1.bin
        arithmetic/agoKernel_AbsDiff_S16_S16S16_Sat_output_1.bin
        arithmetic/agoKernel_Add_U8_U8U8_Wrap_output_1.bin
        arithmetic/agoKernel_Add_U8_U8U8_Sat_output_1.bin
        arithmetic/agoKernel_Add_S16_U8U8_Wrap_output_1.bin
        arithmetic/agoKernel_Add_S16_S16U8_Wrap_output_1.bin
        arithmetic/agoKernel_Add_S16_S16U8_Sat_output_1.bin
        arithmetic/agoKernel_Add_S16_S16S16_Wrap_output_1.bin
        arithmetic/agoKernel_Add_S16_S16S16_Sat_output_1.bin
        arithmetic/agoKernel_Sub_U8_U8U8_Wrap_output_1.bin
        arithmetic/agoKernel_Sub_U8_U8U8_Sat_output_1.bin
        arithmetic/agoKernel_Sub_S16_U8U8_Wrap_output_1.bin
        arithmetic/agoKernel_Sub_S16_S16U8_Wrap_output_1.bin
        arithmetic/agoKernel_Sub_S16_S16U8_Sat_output_1.bin
        arithmetic/agoKernel_Sub_S16_U8S16_Wrap_output_1.bin
        arithmetic/agoKernel_Sub_S16_U8S16_Sat_output_1.bin
        arithmetic/agoKernel_Sub_S16_S16S16_Wrap_output_1.bin
        arithmetic/agoKernel_Sub_S16_S16S16_Sat_output_1.bin
        arithmetic/agoKernel_Mul_U8_U8U8_Wrap_Trunc_output_1.bin
        arithmetic/agoKernel_Mul_U8_U8U8_Wrap_Round_output_1.bin
        arithmetic/agoKernel_Mul_U8_U8U8_Sat_Trunc_output_1.bin
        arithmetic/agoKernel_Mul_U8_U8U8_Sat_Round_output_1.bin
        arithmetic/agoKernel_Mul_S16_U8U8_Wrap_Trunc_output_1.bin
        arithmetic/agoKernel_Mul_S16_U8U8_Wrap_Round_output_1.bin
        arithmetic/agoKernel_Mul_S16_U8U8_Sat_Trunc_output_1.bin
        arithmetic/agoKernel_Mul_S16_U8U8_Sat_Round_output_1.bin
        arithmetic/agoKernel_Mul_S16_S16U8_Wrap_Trunc_output_1.bin
        arithmetic/agoKernel_Mul_S16_S16U8_Wrap_Round_output_1.bin
        arithmetic/agoKernel_Mul_S16_S16U8_Sat_Trunc_output_1.bin
        arithmetic/agoKernel_Mul_S16_S16U8_Sat_Round_output_1.bin
        arithmetic/agoKernel_Mul_S16_S16S16_Wrap_Trunc_output_1.bin
        arithmetic/agoKernel_Mul_S16_S16S16_Wrap_Round_output_1.bin
        arithmetic/agoKernel_Mul_S16_S16S16_Sat_Trunc_output_1.bin
        arithmetic/agoKernel_Mul_S16_S16S16_Sat_Round_output_1.bin
        arithmetic/agoKernel_Magnitude_S16_S16S16_output_1.bin
        arithmetic/agoKernel_Phase_U8_S16S16_output_1.bin
        arithmetic/agoKernel_WeightedAverage_U8_U8U8_output_1.bin
        logical/agoKernel_And_U8_U8U8_output_1.bin
        logical/agoKernel_And_U8_U8U1_output_1.bin
        logical/agoKernel_And_U8_U1U8_output_1.bin
        logical/agoKernel_And_U8_U1U1_output_1.bin
        logical/agoKernel_And_U1_U8U8_output_1.bin
        logical/agoKernel_And_U1_U8U1_output_1.bin
        logical/agoKernel_And_U1_U1U8_output_1.bin
        logical/agoKernel_And_U1_U1U1_output_1.bin
        logical/agoKernel_Or_U8_U8U8_output_1.bin
        logical/agoKernel_Or_U8_U8U1_output_1.bin
        logical/agoKernel_Or_U8_U1U8_output_1.bin
        logical/agoKernel_Or_U8_U1U1_output_1.bin
        logical/agoKernel_Or_U1_U8U8_output_1.bin
        logical/agoKernel_Or_U1_U8U1_output_1.bin
        logical/agoKernel_Or_U1_U1U8_output_1.bin
        logical/agoKernel_Or_U1_U1U1_output_1.bin
        logical/agoKernel_Xor_U8_U8U8_output_1.bin
        logical/agoKernel_Xor_U8_U8U1_output_1.bin
        logical/agoKernel_Xor_U8_U1U8_output_1.bin
        logical/agoKernel_Xor_U8_U1U1_output_1.bin
        logical/agoKernel_Xor_U1_U8U8_output_1.bin
        logical/agoKernel_Xor_U1_U8U1_output_1.bin
        logical/agoKernel_Xor_U1_U1U8_output_1.bin
        logical/agoKernel_Xor_U1_U1U1_output_1.bin
        logical/agoKernel_Not_U8_U8_output_1.bin
        logical/agoKernel_Not_U1_U8_output_1.bin
        logical/agoKernel_Not_U8_U1_output_1.bin
        logical/agoKernel_Not_U1_U1_output_1.bin
        color/agoKernel_Lut_U8_U8_output_1.bin
        color/agoKernel_ColorDepth_U8_S16_Wrap_output_1.bin
        color/agoKernel_ColorDepth_U8_S16_Sat_output_1.bin
        color/agoKernel_ColorDepth_S16_U8_output_1.bin
        color/agoKernel_ChannelExtract_U8_U16_Pos0_output_1.bin
        color/agoKernel_ChannelExtract_U8_U16_Pos1_output_1.bin
        color/agoKernel_ChannelExtract_U8_U24_Pos0_output_1.bin
        color/agoKernel_ChannelExtract_U8_U24_Pos1_output_1.bin
        color/agoKernel_ChannelExtract_U8_U24_Pos2_output_1.bin
        color/agoKernel_ChannelExtract_U8_U32_Pos0_output_1.bin
        color/agoKernel_ChannelExtract_U8_U32_Pos0_output_2.bin
        color/agoKernel_ChannelExtract_U8_U32_Pos1_output_1.bin
        color/agoKernel_ChannelExtract_U8_U32_Pos1_output_2.bin
        color/agoKernel_ChannelExtract_U8_U32_Pos2_output_1.bin
        color/agoKernel_ChannelExtract_U8_U32_Pos2_output_2.bin
        color/agoKernel_ChannelExtract_U8_U32_Pos3_output_1.bin
        color/agoKernel_ChannelExtract_U8_U32_Pos3_output_2.bin
        color/agoKernel_ChannelExtract_U8U8U8_U24_output_1.bin
        color/agoKernel_ChannelExtract_U8U8U8_U24_output_2.bin
        color/agoKernel_ChannelExtract_U8U8U8_U24_output_3.bin
        color/agoKernel_ChannelExtract_U8U8U8_U32_output_1.bin
        color/agoKernel_ChannelExtract_U8U8U8_U32_output_2.bin
        color/agoKernel_ChannelExtract_U8U8U8_U32_output_3.bin
        color/agoKernel_ChannelExtract_U8U8U8U8_U32_output_1.bin
        color/agoKernel_ChannelExtract_U8U8U8U8_U32_output_2.bin
        color/agoKernel_ChannelExtract_U8U8U8U8_U32_output_3.bin
        color/agoKernel_ChannelExtract_U8U8U8U8_U32_output_4.bin
        color/agoKernel_ChannelCombine_U16_U8U8_output_1.bin
        color/agoKernel_ChannelCombine_U32_U8U8U8_UYVY_output_1.bin
        color/agoKernel_ChannelCombine_U32_U8U8U8_YUYV_output_1.bin
        color/agoKernel_ChannelCombine_U24_U8U8U8_RGB_output_1.bin
        color/agoKernel_ChannelCombine_U32_U8U8U8U8_RGBX_output_1.bin
        color/agoKernel_ColorConvert_RGB_RGBX_output_1.bin
        color/agoKernel_ColorConvert_RGB_UYVY_output_1.bin
        color/agoKernel_ColorConvert_RGB_YUYV_output_1.bin
        color/agoKernel_ColorConvert_RGB_IYUV_output_1.bin
        color/agoKernel_ColorConvert_RGB_NV12_output_1.bin
        color/agoKernel_ColorConvert_RGB_NV21_output_1.bin
        color/agoKernel_ColorConvert_RGBX_RGB_output_1.bin
        color/agoKernel_ColorConvert_RGBX_UYVY_output_1.bin
        color/agoKernel_ColorConvert_RGBX_YUYV_output_1.bin
        color/agoKernel_ColorConvert_RGBX_IYUV_output_1.bin
        color/agoKernel_ColorConvert_RGBX_NV12_output_1.bin
        color/agoKernel_ColorConvert_RGBX_NV21_output_1.bin
        color/agoKernel_ColorConvert_IYUV_RGB_output_1.bin
        color/agoKernel_ColorConvert_IYUV_RGBX_output_1.bin
        color/agoKernel_FormatConvert_IYUV_UYVY_output_1.bin
        color/agoKernel_FormatConvert_IYUV_YUYV_output_1.bin
        color/agoKernel_ColorConvert_NV12_RGB_output_1.bin
        color/agoKernel_ColorConvert_NV12_RGBX_output_1.bin
        color/agoKernel_FormatConvert_NV12_UYVY_output_1.bin
        color/agoKernel_FormatConvert_NV12_YUYV_output_1.bin
        color/agoKernel_ColorConvert_YUV4_RGB_output_1.bin
        color/agoKernel_ColorConvert_YUV4_RGBX_output_1.bin
        color/agoKernel_FormatConvert_IUV_UV12_output_1.bin
        color/agoKernel_FormatConvert_UV12_IUV_output_1.bin
        color/agoKernel_FormatConvert_UV_UV12_output_1.bin
        color/agoKernel_ScaleUp2x2_U8_U8_output_1.bin
        filter/agoKernel_Box_U8_U8_3x3_output_1.bin
        filter/agoKernel_Dilate_U8_U8_3x3_output_1.bin
        filter/agoKernel_Dilate_U1_U8_3x3_output_1.bin
        filter/agoKernel_Dilate_U8_U1_3x3_output_1.bin
        filter/agoKernel_Dilate_U1_U1_3x3_output_1.bin
        filter/agoKernel_Erode_U8_U8_3x3_output_1.bin
        filter/agoKernel_Erode_U1_U8_3x3_output_1.bin
        filter/agoKernel_Erode_U8_U1_3x3_output_1.bin
        filter/agoKernel_Erode_U1_U1_3x3_output_1.bin
        filter/agoKernel_Median_U8_U8_3x3_output_1.bin
        filter/agoKernel_Gaussian_U8_U8_3x3_output_1.bin
        filter/agoKernel_ScaleGaussianHalf_U8_U8_3x3_output_1.bin
        filter/agoKernel_ScaleGaussianHalf_U8_U8_5x5_output_1.bin
        filter/agoKernel_Convolve_U8_U8_3x3_output_1.bin
        filter/agoKernel_Convolve_S16_U8_3x3_output_1.bin
        filter/agoKernel_Sobel_S16S16_U8_3x3_GXY_output_1.bin
        filter/agoKernel_Sobel_S16S16_U8_3x3_GXY_output_2.bin
        filter/agoKernel_Sobel_S16_U8_3x3_GX_output_1.bin
        filter/agoKernel_Sobel_S16_U8_3x3_GY_output_1.bin
        statistical/agoKernel_Threshold_U8_U8_Binary_output_1.bin
        statistical/agoKernel_Threshold_U8_U8_Range_output_1.bin
        statistical/agoKernel_Threshold_U1_U8_Binary_output_1.bin
        statistical/agoKernel_Threshold_U1_U8_Range_output_1.bin
        statistical/agoKernel_Threshold_U8_S16_Binary_output_1.bin
        statistical/agoKernel_Threshold_U8_S16_Range_output_1.bin
        geometric/agoKernel_ScaleImage_U8_U8_Nearest_output_1.bin
        geometric/agoKernel_ScaleImage_U8_U8_Bilinear_output_1.bin
        geometric/agoKernel_ScaleImage_U8_U8_Bilinear_Replicate_output_1.bin
        geometric/agoKernel_ScaleImage_U8_U8_Bilinear_Constant_output_1.bin
        geometric/agoKernel_ScaleImage_U8_U8_Area_output_1.bin
        geometric/agoKernel_WarpAffine_U8_U8_Nearest_output_1.bin
        geometric/agoKernel_WarpAffine_U8_U8_Nearest_Constant_output_1.bin
        geometric/agoKernel_WarpAffine_U8_U8_Bilinear_output_1.bin
        geometric/agoKernel_WarpAffine_U8_U8_Bilinear_Constant_output_1.bin
        geometric/agoKernel_WarpPerspective_U8_U8_Nearest_output_1.bin
        geometric/agoKernel_WarpPerspective_U8_U8_Nearest_Constant_output_1.bin
        geometric/agoKernel_WarpPerspective_U8_U8_Bilinear_output_1.bin
        geometric/agoKernel_WarpPerspective_U8_U8_Bilinear_Constant_output_1.bin
        geometric/agoKernel_Remap_U8_U8_Nearest_output_1.bin
        geometric/agoKernel_Remap_U8_U8_Nearest_Constant_output_1.bin
        geometric/agoKernel_Remap_U8_U8_Bilinear_output_1.bin
        geometric/agoKernel_Remap_U8_U8_Bilinear_Constant_output_1.bin
        vision/agoKernel_Canny_3x3_L1NORM_output_1.bin
        vision/agoKernel_Canny_3x3_L2NORM_output_1.bin
        vision/agoKernel_FastCorners_XY_U8_Supression_output_1.bin
        vision/agoKernel_FastCorners_XY_U8_Supression_output_2.bin
        vision/agoKernel_FastCorners_XY_U8_NoSupression_output_1.bin
        vision/agoKernel_FastCorners_XY_U8_NoSupression_output_2.bin
        vision/agoKernel_Harris_3x3_output_1.bin
        "

        if [ "$KERNEL_NAME" != "ALL" ]; then
            printf "\nPicking output binary dumps...\n"
            OUTPUT_BIN_LIST_SINGLE=""
            STARTSWITHSTRING="$GROUP/agoKernel_$KERNEL_NAME"
            for OUTPUT_BIN in $OUTPUT_BIN_LIST;
            do
                if [[ $OUTPUT_BIN == $STARTSWITHSTRING* ]]; then
                    OUTPUT_BIN_LIST_SINGLE="$OUTPUT_BIN_LIST_SINGLE $OUTPUT_BIN"
                fi
            done
            OUTPUT_BIN_LIST="$OUTPUT_BIN_LIST_SINGLE"
        fi

        UNMATCHED_OUTPUT_LIST=""

        for OUTPUT_BIN in $OUTPUT_BIN_LIST;
        do
            printf "\n"
            echo "Checking $OUTPUT_BIN..."
            DIFF=""

            if [[ $OUTPUT_BIN == filter* ]]; then
                if [[ $OUTPUT_BIN == filter/agoKernel_ScaleGaussianHalf* ]]; then
                    WIDTH=$HALF_WIDTH
                    HEIGHT=$HALF_HEIGHT
                fi
                NUM_OF_OCTETS=$(("$WIDTH" * ("$HEIGHT" - 2)))
                START=13
                END=$((10 + "$WIDTH" * 2 + "$WIDTH" - 4))
                xxd -g 1 -c "$WIDTH" -s "$WIDTH" -l "$NUM_OF_OCTETS" dumpsOCL/"$OUTPUT_BIN" | cut -c "$START"-"$END" > "dumpsOCL/$OUTPUT_BIN.roi.txt"
                xxd -g 1 -c "$WIDTH" -s "$WIDTH" -l "$NUM_OF_OCTETS" dumpsHIP/"$OUTPUT_BIN" | cut -c "$START"-"$END" > "dumpsHIP/$OUTPUT_BIN.roi.txt"
                DIFF=$(diff "dumpsHIP/$OUTPUT_BIN.roi.txt" "dumpsOCL/$OUTPUT_BIN.roi.txt")
            else
                DIFF=$(diff "dumpsHIP/$OUTPUT_BIN" "dumpsOCL/$OUTPUT_BIN")
            fi

            if [ "$DIFF" != "" ]
            then
                echo "Outputs don't match!"
                UNMATCHED_OUTPUT_LIST="$UNMATCHED_OUTPUT_LIST $OUTPUT_BIN"
            else
                echo "MATCH!"
            fi
        done

        printf "\n\n"
        echo "Kernels for which OCL and HIP don't match:"
        if [ "$UNMATCHED_OUTPUT_LIST" = "" ]; then
            echo "NONE!"
        else
            for UNMATCHED_OUTPUT in $UNMATCHED_OUTPUT_LIST;
            do
                echo "$UNMATCHED_OUTPUT"
            done
        fi

        printf "\nFinished running funcitonalities. Finished the OCLvsHIP output matching test.\n"
    else
        printf "\nFinished running funcitonalities. Bin dump generation is required for the OCLvsHIP output matching test.\n"
    fi
else
    printf "\nFinished running funcitonalities.\n"
fi

############# Need not edit #############