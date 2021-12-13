import tensorflow as tf
import tensorflow_addons as tfa

from utils.sampling_utils import *

def saturate(val):
    return tf.math.maximum(val, 0)

def clip(val, mi=0.0, ma=1.0):
    return tf.clip_by_value(val, mi, ma)

def linear_visibility(diff, falloff=0.05):
    return 1.0 - clip(diff * falloff)

def exponential_visibility(diff, exponent=10.0):
    visibility = tf.exp(
        -exponent * saturate(diff)
        )
    
    return visibility

def hard_visibility(diff, near, far, threshold=0.05):
    visibility = tf.cast(
        diff < ((far[..., None, :] - near[..., None, :]) * threshold),
        tf.float32
        )
    
    return visibility

def query_shadow_map(shadow_map, shadow_query_px, method='nearest'):
    query_z = None

    if method == 'bilinear':
        query_z = tfa.image.interpolate_bilinear(
            tf.cast(shadow_map[None, ..., None], tf.float32),
            shadow_query_px, indexing='xy'
            )
    elif method == 'nearest':
        query_z = interpolate_image(
            tf.cast(shadow_map[None, ..., None], tf.float32),
            tf.cast(shadow_query_px, tf.float32)[None],
            shifts_x=[0],
            shifts_y=[0],
            weight_fn=(lambda x, y, z, w: tf.ones_like(x))
            )
    
    return query_z