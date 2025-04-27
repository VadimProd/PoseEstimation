import os

for i in range(1, 15):  
    command = f'wget https://github.com/opencv/opencv/raw/master/samples/data/left0{str(i)}.jpg'
    os.system(command)

for i in range(1, 15):  
    command = f'wget https://github.com/opencv/opencv/raw/master/samples/data/right0{str(i)}.jpg'
    os.system(command)