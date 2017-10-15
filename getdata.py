import numpy as np
from PIL import ImageGrab
import cv2
import time
from getkeys import key_check

file = open("G://Works//Chrome T-rex//counter.txt", 'r')
items = file.readlines()
file.close()
counter = []
counter = list(map(int, items))
print(counter)

path = 'G://Works//Chrome T-rex//data//'

up = 38
down = 40

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)
last_time = time.time()
while True:
    screen =  np.array(ImageGrab.grab(bbox=(360,100,700,440)))
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    keys = key_check()

    if up in keys:
        cv2.imwrite(path + 'up/' + str(counter[0]) + ".jpg",screen)
        counter[0]+=1
        time.sleep(0.5)
    elif down in keys:
        cv2.imwrite(path + 'down/' + str(counter[1]) + ".jpg",screen)
        counter[1]+=1
        time.sleep(0.5)
    else:
        cv2.imwrite(path + 'null/' + str(counter[2]) + ".jpg",screen)
        counter[2]+=1

    if ord('E') in keys:
        break
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

file = open("G://Works//Chrome T-rex//counter.txt", 'w')
for ch in counter:
    print(ch)
    file.write(str(ch) + "\n")
file.close()
