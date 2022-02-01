import json
import numpy as np
import PIL.Image as pil
#import tensorflow as tf
#import elpips.elpips as elpips
import matplotlib.pyplot as plt
import math
import sys

import cv2
from glob import glob

def img2mse(x, y):
    return np.mean(np.square(x - y))

def mse2psnr(x):
    return -10.*np.log(x)/np.log(10.)

def main(dir, num_images):
    # Color
    metric = elpips.Metric(elpips.elpips_vgg(batch_size=1),back_prop=False)
    out = []

    for i in range(0, num_images):
        print(f'{dir}/eval/{i:04d}.npy')
        print(f'{dir}/eval_target/{i:04d}.npy')

        # Read images (of size 255 x 255) from file.
        im1 = np.load(f'{dir}/eval/{i:04d}.npy').astype(np.float32)
        im2 = np.load(f'{dir}/eval_target/{i:04d}.npy').astype(np.float32)

        if len(im1.shape) < 3:
            im1 = im1[..., None]
            im2 = im2[..., None]

        # Add an outer batch for each image.
        im1 = tf.expand_dims(im1, axis=0)
        im2 = tf.expand_dims(im2, axis=0)

        ssim = np.array(tf.image.ssim(im1, im2, max_val=1.0))
        psnr = np.array(tf.image.psnr(im1, im2, max_val=1.0))

        if im1.shape[-1] == 3:
            elpips_score = np.array(metric.forward(im1, im2))
        else:
            im1 = tf.tile(im1, [1, 1, 1, 3])
            im2 = tf.tile(im2, [1, 1, 1, 3])
            elpips_score = np.array(metric.forward(im1, im2))

        out.append((ssim, psnr, elpips_score))
        print(out[-1])

    # Depth
    pred_fnames = sorted(glob(f'{dir}/eval_depth/*.npy'))[:num_images]
    target_fnames = sorted(glob(f'{dir}/eval_target_depth/*.npy'))[:num_images]

    preds = []
    targets = []

    for pf, tf in zip(pred_fnames, target_fnames):
        pred = np.load(pf)
        target = cv2.resize(np.load(tf), (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        preds.append(pred)
        targets.append(target)
        mse = img2mse(pred, target)
        psnr = mse2psnr(mse)
        print(mse, psnr)
        plt.imshow(np.abs(pred - target))
        plt.show()

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    mse = img2mse(preds, targets)
    psnr = mse2psnr(mse)

    print(mse, psnr)

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
