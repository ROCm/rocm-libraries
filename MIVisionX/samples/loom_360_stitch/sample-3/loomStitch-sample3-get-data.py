import os
from PIL import Image

os.system('(mkdir data; cd data; wget https://www.ptgui.com/images/masktut/louvre.zip)')
os.system('(cd data; unzip louvre.zip)')
os.system('(cp data/Louvre/step5.pts calibration.pts)')
Image.open("data/Louvre/IMG_1536.JPG").save('cam00.bmp')
Image.open("data/Louvre/IMG_1537.JPG").save('cam01.bmp')
Image.open("data/Louvre/IMG_1538.JPG").save('cam02.bmp')
Image.open("data/Louvre/IMG_1539.JPG").save('cam03.bmp')
Image.open("data/Louvre/IMG_1540.JPG").save('cam04.bmp')
