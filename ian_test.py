# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:22:23 2020

@author: Ian_b
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as numpy
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import time
from skimage import io
from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import random_label_cmap
from stardist.models import StarDist3D
demo_model = True

np.random.seed(6)
lbl_cmap = random_label_cmap()



X = sorted(glob('/Users/Ian_b/Documents/Github/stardist/data/c1_test_resized.tif'))
X = list(map(imread,X))

n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]
axis_norm = (0,1,2)   # normalize channels independently
# axis_norm = (0,1,2,3) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    
# show all test images
if True:
    fig, ax = plt.subplots(1,3, figsize=(16,16))
    for i,(a,x) in enumerate(zip(ax.flat, X)):
        a.imshow(x[x.shape[0]//2],cmap='gray')
        a.set_title(i)
    [a.axis('off') for a in ax.flat]
    plt.tight_layout()
None;    


if demo_model:
    print (
        "NOTE: This is loading a previously trained demo model!\n"
        "      Please set the variable 'demo_model = False' to load your own trained model.",
        file=sys.stderr, flush=True
    )
    model = StarDist3D(None, name='3D_demo', basedir='../../models/examples')
else:
    model = StarDist3D(None, name='stardist', basedir='models')
None;


img = X[0]/255
labels, details = model.predict_instances(img)

plt.figure(figsize=(13,10))
z = max(0, img.shape[0] // 2 - 5)
plt.subplot(121)
plt.imshow((img if img.ndim==3 else img[...,:3])[z], clim=(0,1), cmap='gray')
plt.title('Raw image (XY slice)')
plt.axis('off')
plt.subplot(122)
plt.imshow((img if img.ndim==3 else img[...,:3])[z], clim=(0,1), cmap='gray')
plt.imshow(labels[z], cmap=lbl_cmap, alpha=0.5)
plt.title('Image and predicted labels (XY slice)')
plt.axis('off');

io.imsave('/Users/Ian_b/Documents/Github/stardist/data/c1_test_resized_labels_2.tif', labels)