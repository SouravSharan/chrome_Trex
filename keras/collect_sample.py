import cv2
import numpy as np
import math
from keras.models import load_model
import numpy as np
import pyautogui
import time
from PIL import ImageGrab

model = load_model('whole_model.h5')
while True:
    screen =  np.array(ImageGrab.grab(bbox=(360,100,700,440)))
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

    cv2.imwrite('./screen.jpg', screen)
    screen = cv2.imread('./screen.jpg')
    screen = np.expand_dims(screen, axis=0)
    key = model.predict(screen, batch_size = 1, verbose = 0)
    k = key[0]
    print(k[1])
    if k[1] == 1:
        pyautogui.press('space')
