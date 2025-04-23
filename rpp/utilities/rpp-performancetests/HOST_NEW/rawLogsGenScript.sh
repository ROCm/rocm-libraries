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
# DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/three_images_224x224_src1"
# DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/three_images_224x224_src2"

# Input Images - Eight images (224 x 224)
DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/eight_images_224x224_src1"
DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/eight_images_224x224_src2"

# Input Images - Two images (mixed size)
# DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/two_images_mixed_src1"
# DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/two_images_mixed_src2"

# Output Images
mkdir "$cwd/../OUTPUT_PERFORMANCE_LOGS_HOST_NEW"
DEFAULT_DST_FOLDER="$cwd/../OUTPUT_PERFORMANCE_LOGS_HOST_NEW"

# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>





# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="$DEFAULT_SRC_FOLDER_1"
SRC_FOLDER_2="$DEFAULT_SRC_FOLDER_2"
DST_FOLDER="$DEFAULT_DST_FOLDER"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>





# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>

group_name_generator() {

    CASE=$1

    FUNCTIONALITY_GROUP=""

    if [[ "$case" -eq 0 ]]
    then
        FUNCTIONALITY_GROUP="image_augmentations"
    elif [[ "$case" -eq 15 ]]
    then
        FUNCTIONALITY_GROUP="statistical_functions"
    elif [[ "$case" -eq 20 ]]
    then
        FUNCTIONALITY_GROUP="geometry_transforms"
    elif [[ "$case" -eq 29 ]]
    then
        FUNCTIONALITY_GROUP="advanced_augmentations"
    elif [[ "$case" -eq 36 ]]
    then
        FUNCTIONALITY_GROUP="fused_functions"
    elif [[ "$case" -eq 40 ]]
    then
        FUNCTIONALITY_GROUP="morphological_transforms"
    elif [[ "$case" -eq 42 ]]
    then
        FUNCTIONALITY_GROUP="color_model_conversions"
    elif [[ "$case" -eq 49 ]]
    then
        FUNCTIONALITY_GROUP="filter_operations"
    elif [[ "$case" -eq 56 ]]
    then
        FUNCTIONALITY_GROUP="arithmetic_operations"
    elif [[ "$case" -eq 65 ]]
    then
        FUNCTIONALITY_GROUP="logical_operations"
    elif [[ "$case" -eq 69 ]]
    then
        FUNCTIONALITY_GROUP="computer_vision"
    elif [[ "$case" -eq 79 ]]
    then
        FUNCTIONALITY_GROUP="miscellaneous"
    fi
}

if [[ "$1" -lt 0 ]] | [[ "$1" -gt 86 ]]; then
    echo "The starting case# must be in the 0:86 range!"
    echo
    echo "The rawLogsGenScript.sh bash script runs the RPP performance testsuite for AMDRPP functionalities in HOST/OCL/HIP backends."
    echo
    echo "Syntax: ./rawLogsGenScript.sh <S> <E>"
    echo "S     CASE_START (Starting case# (0:86))"
    echo "E     CASE_END (Ending case# (0:86))"
    exit 1
fi

if [[ "$2" -lt 0 ]] | [[ "$2" -gt 86 ]]; then
    echo "The ending case# must be in the 0:86 range!"
    echo
    echo "The rawLogsGenScript.sh bash script runs the RPP performance testsuite for AMDRPP functionalities in HOST/OCL/HIP backends."
    echo
    echo "Syntax: ./rawLogsGenScript.sh <S> <E>"
    echo "S     CASE_START (Starting case# (0:86))"
    echo "E     CASE_END (Ending case# (0:86))"
    exit 1
fi

CASE_START="$1"
CASE_END="$2"

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

printf "\n\nUsage: ./BatchPD_host_pkd3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:86> <verbosity = 0/1>"

for ((case=$CASE_START;case<=$CASE_END;case++))
do
    group_name_generator "$case"
    printf "\n\n$FUNCTIONALITY_GROUP\n\n" | tee -a "$DST_FOLDER/BatchPD_host_pkd3_host_raw_performance_log.txt"

    printf "\n\n\n\n"
    echo "--------------------------------"
    printf "Running a New Functionality...\n"
    echo "--------------------------------"
    for ((bitDepth=0;bitDepth<7;bitDepth++))
    do
        printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
        for ((outputFormatToggle=0;outputFormatToggle<2;outputFormatToggle++))
        do

            SRC_FOLDER_1_TEMP="$SRC_FOLDER_1"
            SRC_FOLDER_2_TEMP="$SRC_FOLDER_2"

            printf "\n./BatchPD_host_pkd3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case 0"
            ./BatchPD_host_pkd3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" | tee -a "$DST_FOLDER/BatchPD_host_pkd3_host_raw_performance_log.txt"

            if [ "$case" -eq 8 ]
            then
                for ((noiseType=0;noiseType<3;noiseType++))
                do
                    printf "\n./Tensor_host_pkd3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $noiseType 0"
                    ./Tensor_host_pkd3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$noiseType" "0" | tee -a "$DST_FOLDER/Tensor_host_pkd3_host_raw_performance_log.txt"
                done
            elif [ "$case" -eq 21 ] || [ "$case" -eq 23 ] || [ "$case" -eq 24 ]
            then
                for ((interpolationType=0;interpolationType<6;interpolationType++))
                do
                    printf "\n./Tensor_host_pkd3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $interpolationType 0"
                    ./Tensor_host_pkd3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "0" | tee -a "$DST_FOLDER/Tensor_host_pkd3_host_raw_performance_log.txt"
                done
            else
                printf "\n./Tensor_host_pkd3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case 0"
                ./Tensor_host_pkd3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" | tee -a "$DST_FOLDER/Tensor_host_pkd3_host_raw_performance_log.txt"
            fi

            echo "------------------------------------------------------------------------------------------"
        done
    done
done





printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all PLN1 Inputs..."
echo "##########################################################################################"

printf "\n\nUsage: ./BatchPD_host_pln1 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:86> <verbosity = 0/1>"

for ((case=$CASE_START;case<=$CASE_END;case++))
do
    group_name_generator "$case"
    printf "\n\n$FUNCTIONALITY_GROUP\n\n" | tee -a "$DST_FOLDER/BatchPD_host_pln1_host_raw_performance_log.txt"

    printf "\n\n\n\n"
    echo "--------------------------------"
    printf "Running a New Functionality...\n"
    echo "--------------------------------"
    for ((bitDepth=0;bitDepth<7;bitDepth++))
    do
        printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
        for ((outputFormatToggle=0;outputFormatToggle<1;outputFormatToggle++))
        do

            SRC_FOLDER_1_TEMP="$SRC_FOLDER_1"
            SRC_FOLDER_2_TEMP="$SRC_FOLDER_2"

            printf "\n./BatchPD_host_pln1 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case 0"
            ./BatchPD_host_pln1 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" | tee -a "$DST_FOLDER/BatchPD_host_pln1_host_raw_performance_log.txt"

            if [ "$case" -eq 8 ]
            then
                for ((noiseType=0;noiseType<3;noiseType++))
                do
                    printf "\n./Tensor_host_pln1 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $noiseType 0"
                    ./Tensor_host_pln1 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$noiseType" "0" | tee -a "$DST_FOLDER/Tensor_host_pln1_host_raw_performance_log.txt"
                done
            elif [ "$case" -eq 21 ] || [ "$case" -eq 23 ] || [ "$case" -eq 24 ]
            then
                for ((interpolationType=0;interpolationType<6;interpolationType++))
                do
                    printf "\n./Tensor_host_pln1 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $interpolationType 0"
                    ./Tensor_host_pln1 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "0" | tee -a "$DST_FOLDER/Tensor_host_pln1_host_raw_performance_log.txt"
                done
            else
                printf "\n./Tensor_host_pln1 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case 0"
                ./Tensor_host_pln1 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" | tee -a "$DST_FOLDER/Tensor_host_pln1_host_raw_performance_log.txt"
            fi

            echo "------------------------------------------------------------------------------------------"
        done
    done
done





printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all PLN3 Inputs..."
echo "##########################################################################################"

printf "\n\nUsage: ./BatchPD_host_pln3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:86> <verbosity = 0/1>"

for ((case=$CASE_START;case<=$CASE_END;case++))
do
    group_name_generator "$case"
    printf "\n\n$FUNCTIONALITY_GROUP\n\n" | tee -a "$DST_FOLDER/BatchPD_host_pln3_host_raw_performance_log.txt"

    printf "\n\n\n\n"
    echo "--------------------------------"
    printf "Running a New Functionality...\n"
    echo "--------------------------------"
    for ((bitDepth=0;bitDepth<7;bitDepth++))
    do
        printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
        for ((outputFormatToggle=0;outputFormatToggle<2;outputFormatToggle++))
        do

            SRC_FOLDER_1_TEMP="$SRC_FOLDER_1"
            SRC_FOLDER_2_TEMP="$SRC_FOLDER_2"

            printf "\n./BatchPD_host_pln3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case 0"
            ./BatchPD_host_pln3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" | tee -a "$DST_FOLDER/BatchPD_host_pln3_host_raw_performance_log.txt"

            if [ "$case" -eq 8 ]
            then
                for ((noiseType=0;noiseType<6;noiseType++))
                do
                    printf "\n./Tensor_host_pln3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $noiseType 0"
                    ./Tensor_host_pln3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$noiseType" "0" | tee -a "$DST_FOLDER/Tensor_host_pln3_host_raw_performance_log.txt"
                done
            elif [ "$case" -eq 21 ] || [ "$case" -eq 23 ] || [ "$case" -eq 24 ]
            then
                for ((interpolationType=0;interpolationType<6;interpolationType++))
                do
                    printf "\n./Tensor_host_pln3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $interpolationType 0"
                    ./Tensor_host_pln3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "0" | tee -a "$DST_FOLDER/Tensor_host_pln3_host_raw_performance_log.txt"
                done
            else
                printf "\n./Tensor_host_pln3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case 0"
                ./Tensor_host_pln3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" | tee -a "$DST_FOLDER/Tensor_host_pln3_host_raw_performance_log.txt"
            fi

            echo "------------------------------------------------------------------------------------------"
        done
    done
done

# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>