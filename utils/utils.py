import numpy as np
import cv2
import tensorflow as tf

def has_member(key, ns):
    return key in ns.__dict__

def is_true_safe(key, ns):
    return key in ns.__dict__ and ns.__dict__[key]

def to8b(x):
    return (255*np.clip(x, 0, 1)).astype(np.uint8)

def remove_nans(im):
    im[np.isnan(im)] = 0.0

def normalize_im_max(im):
    im = im / np.max(im)
    im[np.isnan(im)] = 0.
    return im

def normalize_im(im):
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    im[np.isnan(im)] = 0.
    return np.clip(im, 0, 1)

def normalize_im_gt(im, im_gt):
    im = (im - np.min(im_gt)) / (np.max(im_gt) - np.min(im_gt))
    im[np.isnan(im)] = 0.
    return np.clip(im, 0, 1)

def depth_to_disparity(depth, near, far):
    depth = np.where(depth < near, near * np.ones_like(depth), depth)
    depth = np.where(depth > far, far * np.ones_like(depth), depth)
    return 1.0 / depth

def depth_from_tof(tof, depth_range, phase_offset=0.0):
    tof_phase = np.arctan2(tof[..., 1:2], tof[..., 0:1])
    tof_phase -= phase_offset
    tof_phase[tof_phase < 0] = tof_phase[tof_phase < 0] + 2 * np.pi
    return tof_phase * depth_range / (4 * np.pi)

def tof_from_depth(depth, amp, depth_range):
    tof_phase = depth * 4 * np.pi / depth_range
    amp *= 1. / np.maximum(depth * depth, (depth_range * 0.1) * (depth_range * 0.1))

    return np.stack(
        [np.cos(tof_phase) * amp, np.sin(tof_phase) * amp, amp],
        axis=-1
        )

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def resize_all_images(images, width, height, method=cv2.INTER_AREA):
    resized_images = []

    for i in range(images.shape[0]):
        resized_images.append(cv2.resize(images[i], (width, height), interpolation=method))
    
    return np.stack(resized_images, axis=0)