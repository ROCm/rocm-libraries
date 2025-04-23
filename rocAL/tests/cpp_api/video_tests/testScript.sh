#!/bin/bash

################ AVAILABLE READERS ######################
##   DEFAULT        : VideoReader                      ##
##   READER_CASE 2  : VideoReaderResize                ##
##   READER_CASE 3  : SequenceReader                   ##
#########################################################
INPUT_PATH=$1
READER_CASE=$2

if [ -z "$INPUT_PATH" ]
  then
    echo "No input argument supplied"
    exit
fi

# Handles relative input path
if [[ ! -d "$INPUT_PATH" ]] & [[ "$INPUT_PATH" != /* ]]
then
  CWD=$(pwd)
  INPUT_PATH="$CWD/$INPUT_PATH"
fi

if [ -z "$READER_CASE" ]
  then
    READER_CASE=1
fi

# Building video unit test
sudo rm -rvf build*
mkdir build
cd build || exit
cmake ..
make

# Arguments used in video unit test
SAVE_FRAMES=1   # (save_frames:on/off)
RGB=1           # (rgb:1/gray:0)
DEVICE=0        # (cpu:0/gpu:1)
DECODER_TYPE=0  # (ffmpeg_sw:0/ffmpeg_hw=1/rocDecode=2)
SHUFFLE=0       # (shuffle:on/off)

BATCH_SIZE=1         # Number of sequences per batch
SEQUENCE_LENGTH=3    # Number of frames per sequence
STEP=3               # Frame interval from one sequence to another sequence
STRIDE=1             # Frame interval within frames in a sequences
RESIZE_WIDTH=1280    # width with which frames should be resized (applicable only for READER_CASE 2)
RESIZE_HEIGHT=720    # height with which frames should be resized (applicable only for READER_CASE 2)

FILELIST_FRAMENUM=1          # enables file number or timestamps parsing for text file input
ENABLE_METADATA=0            # outputs labels and names of the associated frames
ENABLE_FRAME_NUMBER=0        # outputs the starting frame numbers of the sequences in the batch
ENABLE_TIMESTAMPS=0          # outputs timestamps of the frames in the batch
ENABLE_SEQUENCE_REARRANGE=0  # rearranges the frames in the sequence NOTE: The order needs to be set in the video_tests.cpp

echo ./video_tests "$INPUT_PATH" $READER_CASE $DEVICE $DECODER_TYPE $BATCH_SIZE $SEQUENCE_LENGTH $STEP $STRIDE \
$RGB $SAVE_FRAMES $SHUFFLE $RESIZE_WIDTH $RESIZE_HEIGHT $FILELIST_FRAMENUM \
$ENABLE_METADATA $ENABLE_FRAME_NUMBER $ENABLE_TIMESTAMPS $ENABLE_SEQUENCE_REARRANGE

./video_tests "$INPUT_PATH" $READER_CASE $DEVICE $DECODER_TYPE $BATCH_SIZE $SEQUENCE_LENGTH $STEP $STRIDE \
$RGB $SAVE_FRAMES $SHUFFLE $RESIZE_WIDTH $RESIZE_HEIGHT $FILELIST_FRAMENUM \
$ENABLE_METADATA $ENABLE_FRAME_NUMBER $ENABLE_TIMESTAMPS $ENABLE_SEQUENCE_REARRANGE
