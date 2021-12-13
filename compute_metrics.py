import json
import numpy as np
import PIL.Image as pil
import tensorflow as tf
import elpips.elpips as elpips
import matplotlib.pyplot as plt
import math
import sys

def main(num_images):
    metric = elpips.Metric(elpips.elpips_vgg(batch_size=1),back_prop=False)
    out = []

    for i in range(0, num_images):
        print('eval/%04d.npy' % i)
        print('eval_target/%04d.npy' % i)

        # Read images (of size 255 x 255) from file.
        im1 = np.load('eval/%04d.npy' % i).astype(np.float32)
        im2 = np.load('eval_target/%04d.npy' % i).astype(np.float32)

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

if __name__ == '__main__':
    main(int(sys.argv[1]))
