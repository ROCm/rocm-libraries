import os
from PIL import Image

os.system('(mkdir data; cd data; wget https://www.ptgui.com/images/vigntutorial/vigntutorial.zip)')
os.system('(cd data; unzip vigntutorial.zip)')
os.system('(cp data/step3.pts calibration.pts)')
Image.open("data/IMG_5265.JPG").save('cam00.bmp')
Image.open("data/IMG_5266.JPG").save('cam01.bmp')
Image.open("data/IMG_5267.JPG").save('cam02.bmp')
Image.open("data/IMG_5268.JPG").save('cam03.bmp')
