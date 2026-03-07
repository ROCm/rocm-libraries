#! /bin/bash

# usage /.rocfft_to_octave.sh $arg1 #arg2 $file 
# arg1=1 (input) arg1=0 (output)
# arg=2 buffer id 

if [ $1 -eq 1 ]; then
    filename="rocfft_input_data.m"
elif [ $1 -eq 0 ]; then
    filename="rocfft_output_data.m"
else
    echo "error"
fi

# put input file in variable filename
sed '' $3 | sponge $filename

# Get buffer description lines in filename and append 
# line number to them (the lines starting with 
# '--- --- or final output')
cat -n $filename | sed -n '/--- ---\|final output/p' | sponge $filename

# remove lines with buffer hash
sed '/hash/d' $filename | sponge $filename

# store result in temp variable
tmp_var=`cat $filename`

# get line of buffer passed as argument
tmp_var=$(sed -n "/kernel $2/p; /kernel $2/q" <<< "$tmp_var")

# if no lines found, use line number of 'final output' buffer
if [[ -z "${tmp_var// }" ]] ; then
    sed -n "/final output/p; /final output/q" $filename | sponge $filename
else
    sed -n "/kernel $2/p; /kernel $2/q" $filename | sponge $filename
fi

# get line number from this line
sed 's/	.*//' $filename | sponge $filename

# store line number in variable tmp_var
tmp_var=`cat $filename`

# put input file in variable filename
sed '' $3 | sponge $filename

# get buffer from line1 line number to the next '--- ---' line
sed -n "1,$tmp_var b;/--- ---\|final output/ q;p" $3 | sponge $filename

#
sed '1i data=[' $filename | sponge $filename

# Remove character '(' from complex number
sed 's/(//g' $filename | sponge $filename

# Replace character ',' with '+' in complex number
sed 's/,/+/g' $filename | sponge $filename

# Remove new lines
tr '\n' ' ' < $filename | sponge $filename

# Replace character ')' with 'i;' 
sed 's/)/i;\n/g' $filename | sponge $filename

# Append '];' to the end of the file
sed '$a];' $filename | sponge $filename

#
if [ $1 -eq 1 ]; then
    sed -i "1s/^/function data = rocfft_input_data()\n/" $filename
elif [ $1 -eq 0 ]; then
    sed -i "1s/^/function data = rocfft_output_data()\n/" $filename
fi
