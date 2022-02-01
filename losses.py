import tensorflow as tf
import numpy as np

def img2mse(x, y):
    return tf.reduce_mean(tf.square(x - y))

def mse2psnr(x):
    return -10.*tf.math.log(x)/tf.math.log(10.)

def variance_weighted_loss(tof, gt, c=1.):
    tof = outputs['tof_map']

    tof_std = tof[..., -1:]
    tof = tof[..., :2]
    gt = gt[..., :2]

    mse = tf.reduce_mean(tf.square(tof - gt) / (2 * tf.square(tof_std)))
    return (mse + c * tf.reduce_mean(tf.math.log(tof_std)))

def tof_loss_variance(target_tof, outputs, tof_weight):
    img_loss = variance_weighted_loss(outputs['tof_map'], target_tof) * tof_weight
    img_loss0 = 0.0

    if 'tof_map0' in outputs:
        img_loss0 = variance_weighted_loss(outputs['tof_map0'], target_tof) * tof_weight

    return img_loss, img_loss0

def tof_loss_default(target_tof, outputs, tof_weight):
    img_loss = img2mse(outputs['tof_map'][..., :2], target_tof[..., :2]) * tof_weight
    img_loss0 = 0.0

    if 'tof_map0' in outputs:
        img_loss0 = img2mse(outputs['tof_map0'][..., :2], target_tof[..., :2]) * tof_weight
    
    return img_loss, img_loss0

def color_loss_default(target_color, outputs, color_weight):
    img_loss = img2mse(outputs['color_map'], target_color) * color_weight
    img_loss0 = 0.0

    if 'color_map0' in outputs:
        img_loss0 = img2mse(outputs['color_map0'], target_color) * color_weight
    
    return img_loss, img_loss0

def disparity_loss_default(target_depth, outputs, disp_weight, near, far):
    target_disp = 1. / np.clip(target_depth, near, far)
    target

    img_loss = img2mse(outputs['disp_map'], target_disp) * disp_weight
    img_loss0 = 0.0

    if 'disp_map0' in outputs:
        img_loss0 = img2mse(outputs['disp_map0'], target_disp) * disp_weight
    
    return img_loss, img_loss0

def depth_loss_default(target_depth, outputs, depth_weight):
    img_loss = img2mse(outputs['depth_map'], target_depth) * depth_weight
    img_loss0 = 0.0

    if 'depth_map0' in outputs:
        img_loss0 = img2mse(outputs['depth_map0'], target_depth) * depth_weight
    
    return img_loss, img_loss0

def empty_space_loss(outputs):
    loss = tf.reduce_mean(tf.abs(outputs['acc_map']))

    if 'acc_map0' in outputs:
        loss += tf.reduce_mean(tf.abs(outputs['acc_map0']))

    return loss

def make_pose_loss(model, key):
    def loss_fn(_):
        return tf.reduce_mean(tf.square(
            tf.abs(model.poses[key][1:] - model.poses[key][:-1])
        ))
    
    return loss_fn
