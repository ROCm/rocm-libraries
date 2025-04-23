[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# Classification Label Generator Toolset

CLG-Toolset creates a classification label validation list for a given image database and labels set. The images in the dataset are assumed to have the correct labels in their metadata. The tools help to rename the dataset, resize the images, pad the images with a value(0-255) if they are resized to a square image to keep the aspect ratio, extract the labels from the metadata, and generate logs to indicate errors and mismatches in the dataset.

## Prerequisites for running the Toolset

### Linux

* python
* pil
* exiftool

``` 
sudo apt install libimage-exiftool-perl
sudo apt-get install python-pip
sudo apt-get install python-dev libjpeg-dev libfreetype6-dev zlib1g-dev
sudo pip install pil
```

### Windows

* python
* pil
* exiftool
* qawk

## Running the Toolset

### Optional - rename the dataset to cleanup invalid filenames

Fix all file names in the input image folder by running the following command inside the image folder

``` 
ls | cat -n | while read n f; do mv "$f" "image-$n.jpg"; done
```

### Image DataBase Creator

Run **imageDataBaseCreator.py** to create the image database with the required width, height, padding, and image name

``` 
python imageDataBaseCreator.py 	-d [input image directory] 	--- required 
 								-o [output image directory] --- required
 								-f [new image file name] 	--- required
 								-w [resize width] 			--- optional
 								-h [resize height] 			--- optional
 								-p [padding value] 			--- optional
 								-c [image start count] 		--- optional
```

## Outputs

1. Output Image Directory - this folder contains all the images resized and renamed

2. `fileName-val.txt` - this is the classification label validation text file (fileName -- -f option )

``` 
imagename_1.JPEG 122
imagename_2.JPEG 928
```

3. `fileName-scriptOutput` - this folder contains all the logs and error files

 + `fileName-fileNameTanslation.csv`
 + `fileName-fileNameWithErrors.csv`
 + `fileName-fileNameWithLabels.csv`
 + `fileName-invalidLabelsFile.csv`
 + `fileName-multipleLabelsFile.csv`

## Scripts

This scripts folder has the following python scripts.

### step-1.py

Run **step-1.py** to resize and rename your image to the required width and height, also allows the padding to keep the image resolution

``` 
python step-1.py 	-d [input image directory] 	--- required 
 					-o [output image directory] --- required
 					-f [new image file name] 	--- required 
 					-w [resize width] 		    --- optional
 					-h [resize height] 			--- optional
 					-p [padding value] 			--- optional
```

this script will resize and rename all your images and put them in the output folder you created.

### step-2.py

Run **step-2.py** to extract all the tags and output a text file with the image name and all the tags associated with the image

``` 
python step-2.py 	-d [input image directory] 	--- required 
 					-f [tag_file_name.txt] 		--- required 
```

this script will output a CSV format image name & tags. The output file will be `CSV_tag_file_name.txt`

 **output example:**
 ```
 imagename.JPEG, tench, Tinca tinca (fileName.JPEG, tags)
 ```

### step-3.py

Run **step-3.py** to create a usable image validation .txt with the image name and class number

``` 
python step-3.py 	-l [label.txt with 1000 labels without synset numbers] 	--- required (script-labels.txt from this project)
 					-t [CSV_tag_file_name.txt] 								--- required (output from step 2)
```

this script will generate an annie inference app usable data on the cmd/terminal use >> to `val.txt` for output

**output example:**
```
imagename.JPEG 0 (fileName.JPEG Label)
```

### imageDataBaseCreator.py

Run **imageDataBaseCreator.py** to create the image database with the required width, height, padding, and image name

``` 
python imageDataBaseCreator.py 	-d [input image directory] 	--- required 
 								-o [output image directory] --- required
 								-f [new image file name] 	--- required 
 								-w [resize width] 			--- optional
 								-h [resize height] 			--- optional
 								-p [padding value] 			--- optional
 								-c [image start count] 		--- optional
```
