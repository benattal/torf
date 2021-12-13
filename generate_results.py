import numpy as np
import sys, os
from glob import glob
import matplotlib.pyplot as plt
import cv2
from losses import *

dir1 = os.path.join(sys.argv[1], 'eval_depth')
dir2 = os.path.join(sys.argv[1], 'eval_target_depth')
num_frames = int(sys.argv[2])

pred_fnames = sorted(glob(os.path.join(dir1, '*.npy')))[:num_frames]
target_fnames = sorted(glob(os.path.join(dir2, '*.npy')))[:num_frames]
preds = []
targets = []

for pf, tf in zip(pred_fnames, target_fnames):
    pred = np.load(pf)
    target = cv2.resize(np.load(tf), (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
    preds.append(pred)
    targets.append(target)
    #preds.append(pred[:200])
    #targets.append(target[:200])

    plt.subplot(1, 2, 1)
    plt.imshow(pred)
    plt.subplot(1, 2, 2)
    plt.imshow(target)
    plt.show()

    plt.imshow(np.abs(target - pred))
    plt.show()

preds = np.concatenate(preds, axis=0)
targets = np.concatenate(targets, axis=0)
diffs = np.square(pred - target)
mse = img2mse(preds, targets)
psnr = mse2psnr(mse)
print(mse, psnr)
