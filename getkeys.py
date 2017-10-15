# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi
import time

keyList = []

for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(ord(char))

keyList.append(13) #"0x0D") #enter
keyList.append(37 ) #left_arrow
keyList.append(38) #up_arrow
keyList.append(39) #right_arrow
keyList.append(40) #down_arrow

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(int(key)):
            keys.append(key)

    return keys
