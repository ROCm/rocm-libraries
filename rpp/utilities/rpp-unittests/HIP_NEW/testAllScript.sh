#!/bin/bash





# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>

cwd=$(pwd)

# Input Images - Single image (224 x 224)
# DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/single_image_224x224_src1"
# DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/single_image_224x224_src2"

# Input Images - Two images (224 x 224)
# DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/two_images_224x224_src1"
# DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/two_images_224x224_src2"

# Input Images - Three images (224 x 224)
DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/three_images_224x224_src1"
DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/three_images_224x224_src2"

# Input Images - Three images (256 x 256)
# DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/three_images_256x256_src1"
# DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/three_images_256x256_src1"

# Input Images - Two images (mixed size)
# DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/two_images_mixed_src1"
# DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/two_images_mixed_src2"

# Input Images - Six images (mixed size)
# DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/six_images_mixed_src1"
# DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/six_images_mixed_src1"

# Output Images
mkdir "$cwd/../OUTPUT_IMAGES_HIP_NEW"
DEFAULT_DST_FOLDER="$cwd/../OUTPUT_IMAGES_HIP_NEW"

# Images for unique functionalities
DEFAULT_FAST_CORNER_DETECTOR_IMAGES="$cwd/../TEST_IMAGES/fast_corner_detector"
DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES="$cwd/../TEST_IMAGES/harris_corner_detector"
DEFAULT_HOUGH_LINES_IMAGES="$cwd/../TEST_IMAGES/hough_lines"
DEFAULT_HOG_IMAGES="$cwd/../TEST_IMAGES/hog"

# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>





# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="$DEFAULT_SRC_FOLDER_1"
SRC_FOLDER_2="$DEFAULT_SRC_FOLDER_2"
DST_FOLDER="$DEFAULT_DST_FOLDER"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>





# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>

directory_name_generator() {

    AFFINITY=$1
    TYPE=$2
    CASE=$3

    if [[ "$case" -lt 15 ]]
    then
        FUNCTIONALITY_GROUP="image_augmentations"
    elif [[ "$case" -lt 20 ]]
    then
        FUNCTIONALITY_GROUP="statistical_functions"
    elif [[ "$case" -lt 29 ]]
    then
        FUNCTIONALITY_GROUP="geometry_transforms"
    elif [[ "$case" -lt 36 ]]
    then
        FUNCTIONALITY_GROUP="advanced_augmentations"
    elif [[ "$case" -lt 40 ]]
    then
        FUNCTIONALITY_GROUP="fused_functions"
    elif [[ "$case" -lt 42 ]]
    then
        FUNCTIONALITY_GROUP="morphological_transforms"
    elif [[ "$case" -lt 49 ]]
    then
        FUNCTIONALITY_GROUP="color_model_conversions"
    elif [[ "$case" -lt 56 ]]
    then
        FUNCTIONALITY_GROUP="filter_operations"
    elif [[ "$case" -lt 65 ]]
    then
        FUNCTIONALITY_GROUP="arithmetic_operations"
    elif [[ "$case" -lt 69 ]]
    then
        FUNCTIONALITY_GROUP="logical_operations"
    elif [[ "$case" -lt 79 ]]
    then
        FUNCTIONALITY_GROUP="computer_vision"
    else
        FUNCTIONALITY_GROUP="miscellaneous"
    fi

    DST_FOLDER_TEMP="$DST_FOLDER""/rpp_""$AFFINITY""_""$TYPE""_""$FUNCTIONALITY_GROUP"
}

if [[ "$1" -lt 0 ]] | [[ "$1" -gt 86 ]]; then
    echo "The starting case# must be in the 0:86 range!"
    echo
    echo "The rawLogsGenScript.sh bash script runs the RPP performance testsuite for AMDRPP functionalities in HOST/OCL/HIP backends."
    echo
    echo "Syntax: ./rawLogsGenScript.sh <S> <E> <U>"
    echo "S     CASE_START (Starting case# (0:86))"
    echo "E     CASE_END (Ending case# (0:86))"
    echo "U     UNIQUE_FUNC (0 = Skip / 1 = Run)"
    exit 1
fi

if [[ "$2" -lt 0 ]] | [[ "$2" -gt 86 ]]; then
    echo "The ending case# must be in the 0:86 range!"
    echo
    echo "The rawLogsGenScript.sh bash script runs the RPP performance testsuite for AMDRPP functionalities in HOST/OCL/HIP backends."
    echo
    echo "Syntax: ./rawLogsGenScript.sh <S> <E> <U>"
    echo "S     CASE_START (Starting case# (0:86))"
    echo "E     CASE_END (Ending case# (0:86))"
    echo "U     UNIQUE_FUNC (0 = Skip / 1 = Run)"
    exit 1
fi

if [ "$3" -ne 0 ]; then
    if [ "$3" -ne 1 ]; then
        echo "The unique functionalities option must be 0/1!"
        echo
        echo "The rawLogsGenScript.sh bash script runs the RPP performance testsuite for AMDRPP functionalities in HOST/OCL/HIP backends."
        echo
        echo "Syntax: ./rawLogsGenScript.sh <S> <E> <U>"
        echo "S     CASE_START (Starting case# (0:86))"
        echo "E     CASE_END (Ending case# (0:86))"
        echo "U     UNIQUE_FUNC (0 = Skip / 1 = Run)"
        exit 1
    fi
fi

if (( "$#" < 3 )); then
    CASE_START="0"
    CASE_END="86"
    UNIQUE_FUNC="0"
else
    CASE_START="$1"
    CASE_END="$2"
    UNIQUE_FUNC="$3"
fi

rm -rvf "$DST_FOLDER"/*
shopt -s extglob
mkdir build
cd build
rm -rvf ./*
cmake ..
make -j16

printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all PKD3 Inputs..."
echo "##########################################################################################"

printf "\n\nUsage: ./BatchPD_hip_pkd3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:86> <verbosity = 0/1>"

for ((case=$CASE_START;case<=$CASE_END;case++))
do
    directory_name_generator "hip" "pkd3" "$case"
    mkdir $DST_FOLDER_TEMP

    printf "\n\n\n\n"
    echo "--------------------------------"
    printf "Running a New Functionality...\n"
    echo "--------------------------------"
    for ((bitDepth=0;bitDepth<7;bitDepth++))
    do
        printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
        for ((outputFormatToggle=0;outputFormatToggle<2;outputFormatToggle++))
        do

            if [[ "$case" -eq 74 ]]
            then
                SRC_FOLDER_1_TEMP="$DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES"
                SRC_FOLDER_2_TEMP="$DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES"
            elif [[ "$case" -eq 75 ]]
            then
                SRC_FOLDER_1_TEMP="$DEFAULT_FAST_CORNER_DETECTOR_IMAGES"
                SRC_FOLDER_2_TEMP="$DEFAULT_FAST_CORNER_DETECTOR_IMAGES"
            elif [[ "$case" -eq 77 ]]
            then
                SRC_FOLDER_1_TEMP="$DEFAULT_HOUGH_LINES_IMAGES"
                SRC_FOLDER_2_TEMP="$DEFAULT_HOUGH_LINES_IMAGES"
            elif [[ "$case" -eq 78 ]]
            then
                SRC_FOLDER_1_TEMP="$DEFAULT_HOG_IMAGES"
                SRC_FOLDER_2_TEMP="$DEFAULT_HOG_IMAGES"
            else
                SRC_FOLDER_1_TEMP="$SRC_FOLDER_1"
                SRC_FOLDER_2_TEMP="$SRC_FOLDER_2"
            fi

            printf "\n./BatchPD_hip_pkd3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case 0"
            ./BatchPD_hip_pkd3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0"

            if [ "$case" -eq 40 ] || [ "$case" -eq 41 ] || [ "$case" -eq 49 ]
            then
                for ((kernelSize=3;kernelSize<=9;kernelSize+=2))
                do
                    printf "\n./Tensor_hip_pkd3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $kernelSize 0"
                    ./Tensor_hip_pkd3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$kernelSize" "0"
                done
            elif [ "$case" -eq 21 ] || [ "$case" -eq 23 ] || [ "$case" -eq 24 ]
            then
                for ((interpolationType=0;interpolationType<6;interpolationType++))
                do
                    printf "\n./Tensor_hip_pkd3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $interpolationType 0"
                    ./Tensor_hip_pkd3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "0"
                done
            elif [[ "$case" -eq 8 ]]
            then
                for ((noiseType=0;noiseType<3;noiseType++))
                do
                    printf "\n./Tensor_hip_pkd3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $noiseType 0"
                    ./Tensor_hip_pkd3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$noiseType" "0"
                done
            else
                printf "\n./Tensor_hip_pkd3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case 0"
                ./Tensor_hip_pkd3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0"
            fi

            echo "------------------------------------------------------------------------------------------"
        done
    done
done

mkdir "$DST_FOLDER/PKD3"
mv "$DST_FOLDER/"!(PKD3) "$DST_FOLDER/PKD3"




printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all PLN1 Inputs..."
echo "##########################################################################################"

printf "\n\nUsage: ./BatchPD_hip_pln1 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:86> <verbosity = 0/1>"

for ((case=$CASE_START;case<=$CASE_END;case++))
do
    directory_name_generator "hip" "pln1" "$case"
    mkdir $DST_FOLDER_TEMP

    printf "\n\n\n\n"
    echo "--------------------------------"
    printf "Running a New Functionality...\n"
    echo "--------------------------------"
    for ((bitDepth=0;bitDepth<7;bitDepth++))
    do
        printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
        for ((outputFormatToggle=0;outputFormatToggle<1;outputFormatToggle++))
        do

            if [[ "$case" -eq 74 ]]
            then
                SRC_FOLDER_1_TEMP="$DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES"
                SRC_FOLDER_2_TEMP="$DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES"
            elif [[ "$case" -eq 75 ]]
            then
                SRC_FOLDER_1_TEMP="$DEFAULT_FAST_CORNER_DETECTOR_IMAGES"
                SRC_FOLDER_2_TEMP="$DEFAULT_FAST_CORNER_DETECTOR_IMAGES"
            elif [[ "$case" -eq 77 ]]
            then
                SRC_FOLDER_1_TEMP="$DEFAULT_HOUGH_LINES_IMAGES"
                SRC_FOLDER_2_TEMP="$DEFAULT_HOUGH_LINES_IMAGES"
            elif [[ "$case" -eq 78 ]]
            then
                SRC_FOLDER_1_TEMP="$DEFAULT_HOG_IMAGES"
                SRC_FOLDER_2_TEMP="$DEFAULT_HOG_IMAGES"
            else
                SRC_FOLDER_1_TEMP="$SRC_FOLDER_1"
                SRC_FOLDER_2_TEMP="$SRC_FOLDER_2"
            fi

            printf "\n./BatchPD_hip_pln1 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case 0"
            ./BatchPD_hip_pln1 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0"

            if [ "$case" -eq 40 ] || [ "$case" -eq 41 ] || [ "$case" -eq 49 ]
            then
                for ((kernelSize=3;kernelSize<=9;kernelSize+=2))
                do
                    printf "\n./Tensor_hip_pln1 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $kernelSize 0"
                    ./Tensor_hip_pln1 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$kernelSize" "0"
                done
            elif [ "$case" -eq 21 ] || [ "$case" -eq 23 ] || [ "$case" -eq 24 ]
            then
                for ((interpolationType=0;interpolationType<6;interpolationType++))
                do
                    printf "\n./Tensor_hip_pln1 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $interpolationType 0"
                    ./Tensor_hip_pln1 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "0"
                done
            elif [[ "$case" -eq 8 ]]
            then
                for ((noiseType=0;noiseType<3;noiseType++))
                do
                    printf "\n./Tensor_hip_pln1 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $noiseType 0"
                    ./Tensor_hip_pln1 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$noiseType" "0"
                done
            else
                printf "\n./Tensor_hip_pln1 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case 0"
                ./Tensor_hip_pln1 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0"
            fi

            echo "------------------------------------------------------------------------------------------"
        done
    done
done

mkdir "$DST_FOLDER/PLN1"
mv "$DST_FOLDER/"!(PKD3|PLN1) "$DST_FOLDER/PLN1"




printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all PLN3 Inputs..."
echo "##########################################################################################"

printf "\n\nUsage: ./BatchPD_hip_pln3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:86> <verbosity = 0/1>"

for ((case=$CASE_START;case<=$CASE_END;case++))
do
    directory_name_generator "hip" "pln3" "$case"
    mkdir $DST_FOLDER_TEMP

    printf "\n\n\n\n"
    echo "--------------------------------"
    printf "Running a New Functionality...\n"
    echo "--------------------------------"
    for ((bitDepth=0;bitDepth<7;bitDepth++))
    do
        printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
        for ((outputFormatToggle=0;outputFormatToggle<2;outputFormatToggle++))
        do

            if [[ "$case" -eq 74 ]]
            then
                SRC_FOLDER_1_TEMP="$DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES"
                SRC_FOLDER_2_TEMP="$DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES"
            elif [[ "$case" -eq 75 ]]
            then
                SRC_FOLDER_1_TEMP="$DEFAULT_FAST_CORNER_DETECTOR_IMAGES"
                SRC_FOLDER_2_TEMP="$DEFAULT_FAST_CORNER_DETECTOR_IMAGES"
            elif [[ "$case" -eq 77 ]]
            then
                SRC_FOLDER_1_TEMP="$DEFAULT_HOUGH_LINES_IMAGES"
                SRC_FOLDER_2_TEMP="$DEFAULT_HOUGH_LINES_IMAGES"
            elif [[ "$case" -eq 78 ]]
            then
                SRC_FOLDER_1_TEMP="$DEFAULT_HOG_IMAGES"
                SRC_FOLDER_2_TEMP="$DEFAULT_HOG_IMAGES"
            else
                SRC_FOLDER_1_TEMP="$SRC_FOLDER_1"
                SRC_FOLDER_2_TEMP="$SRC_FOLDER_2"
            fi

            printf "\n./BatchPD_hip_pln3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case 0"
            ./BatchPD_hip_pln3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0"

            if [ "$case" -eq 40 ] || [ "$case" -eq 41 ] || [ "$case" -eq 49 ]
            then
                for ((kernelSize=3;kernelSize<=9;kernelSize+=2))
                do
                    printf "\n./Tensor_hip_pln3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $kernelSize 0"
                    ./Tensor_hip_pln3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$kernelSize" "0"
                done
            elif [ "$case" -eq 21 ] || [ "$case" -eq 23 ] || [ "$case" -eq 24 ]
            then
                for ((interpolationType=0;interpolationType<6;interpolationType++))
                do
                    printf "\n./Tensor_hip_pln3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $interpolationType 0"
                    ./Tensor_hip_pln3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "0"
                done
            elif [[ "$case" -eq 8 ]]
            then
                for ((noiseType=0;noiseType<3;noiseType++))
                do
                    printf "\n./Tensor_hip_pln3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $noiseType 0"
                    ./Tensor_hip_pln3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$noiseType" "0"
                done
            else
                printf "\n./Tensor_hip_pln3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case 0"
                ./Tensor_hip_pln3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0"
            fi

            echo "------------------------------------------------------------------------------------------"
        done
    done
done

mkdir "$DST_FOLDER/PLN3"
mv "$DST_FOLDER/"!(PKD3|PLN1|PLN3) "$DST_FOLDER/PLN3"




if [[ "$UNIQUE_FUNC" -eq 1 ]]
then
    printf "\n\n\n\n\n"
    echo "##########################################################################################"
    echo "Running all Unique functionalities..."
    echo "##########################################################################################"

    printf "\n\nUsage: ./uniqueFunctionalities_hip <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <case number = 0:12>"

    for ((case=0;case<13;case++))
    do
        printf "\n\n\n\n" | tee -a "$DST_FOLDER/uniqueFunctionalities_hip_log.txt"
        echo "--------------------------------" | tee -a "$DST_FOLDER/uniqueFunctionalities_hip_log.txt"
        printf "Running a New Functionality...\n" | tee -a "$DST_FOLDER/uniqueFunctionalities_hip_log.txt"
        echo "--------------------------------" | tee -a "$DST_FOLDER/uniqueFunctionalities_hip_log.txt"
        for ((bitDepth=0;bitDepth<7;bitDepth++))
        do
            printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n" | tee -a "$DST_FOLDER/uniqueFunctionalities_hip_log.txt"
            echo "./uniqueFunctionalities_hip $bitDepth $case" | tee -a "$DST_FOLDER/uniqueFunctionalities_hip_log.txt"
            ./uniqueFunctionalities_hip "$bitDepth" "$case" | tee -a "$DST_FOLDER/uniqueFunctionalities_hip_log.txt"
            echo "------------------------------------------------------------------------------------------"
        done
    done
fi

# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>
