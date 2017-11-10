# reference : https://github.com/sjchoi86/tensorflow-101/blob/master/notebooks/basic_gendataset.ipynb
import numpy as np
import os
from scipy.misc import imread, imresize

cwd = os.getcwd()
print ("Current folder is %s" % (cwd) )

# Training set folder
paths = {"G:/Works/Chrome T-rex/data/null", "G:/Works/Chrome T-rex/data/up"}

# The reshape size
imgsize = [340, 340]

# Grayscale
use_gray = 1
# Save name
data_name = "custom_data"

def rgb2gray(rgb):
    if len(rgb.shape) is 3:
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    else:
        return rgb

nclass     = len(paths)
valid_exts = [".jpg",".gif",".png",".tga", ".jpeg"]
imgcnt     = 0
for i, relpath in zip(range(nclass), paths):
    path = relpath
    flist = os.listdir(path)
    for f in flist:
        if os.path.splitext(f)[1].lower() not in valid_exts:
            continue
        fullpath = os.path.join(path, f)
        print(fullpath)
        currimg  = imread(fullpath)
        # Convert to grayscale
        if use_gray:
            grayimg  = rgb2gray(currimg)
        else:
            grayimg  = currimg
        # Reshape
        graysmall = imresize(grayimg, [imgsize[0], imgsize[1]])/255.
        grayvec   = np.reshape(graysmall, (1, -1))
        # Save
        curr_label = np.eye(nclass, nclass)[i:i+1, :]
        if imgcnt is 0:
            totalimg   = grayvec
            totallabel = curr_label
        else:
            totalimg   = np.concatenate((totalimg, grayvec), axis=0)
            totallabel = np.concatenate((totallabel, curr_label), axis=0)
        imgcnt    = imgcnt + 1

print ("Total %d images loaded." % (imgcnt))

savepath = "G:/Works/Chrome T-rex/tensorflow/" + data_name + ".npz"

np.savez(savepath, trainimg=totalimg, trainlabel=totallabel , imgsize=imgsize, use_gray=use_gray)
