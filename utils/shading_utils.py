import numpy as np
import tensorflow as tf

from utils.projection_utils import *

def dot(v1, v2):
    return tf.reduce_sum(v1 * v2, axis=-1, keepdims=True)

def reflect(v, n):
    return 2 * dot(v, n) * n - v

def lambert(n, wi, intensity):
    return tf.math.maximum(dot(wi, n), 0.0) * intensity

def phong(ambient, kd, ks, alpha, n, v, wi, intensity):
    r = reflect(v, n)
    diffuse = kd * tf.math.maximum(dot(wi, n), 0.0) * intensity
    specular = ks * tf.math.pow(tf.math.maximum(dot(wi, r), 0.0), alpha) * intensity

    return ambient + diffuse + specular

def get_normals(points):
    u = normalize(tf.roll(points, 1, -2) - points)
    v = normalize(tf.roll(points, 1, -3) - points)
    normals = -tf.linalg.cross(u, v)
    return normals

EPS = 1e-7

def saturate(x, EPS):
    return tf.clip_by_value(x, EPS, 1.0)

def isclose(x: tf.Tensor, val: float, threshold: float = EPS) -> tf.Tensor:
    return tf.less_equal(tf.abs(x - val), threshold)

def safe_sqrt(x: tf.Tensor) -> tf.Tensor:
    sqrt_in = tf.nn.relu(tf.where(isclose(x, 0.0), tf.ones_like(x) * EPS, x))
    return tf.sqrt(sqrt_in)

def cook_torrance(
    kd, metalness, roughness, n, v, wi, intensity
    ):
    h = normalize(wi + v)

    # Dot products
    ndotl = tf.math.maximum(dot(n, wi), EPS)
    ndoth = tf.math.maximum(dot(n, h), EPS)
    ndotv = tf.math.maximum(dot(n, v), EPS)
    vdoth = tf.math.maximum(dot(v, h), EPS)

    # Fresnel reflectance
    ior = 1.4
    F0 = tf.math.abs((1.0 - ior) / (1.0 + ior))
    F0 = F0 * F0
    F0 = (1 - metalness) * F0 + metalness * kd
    F = tf.math.pow(1.0 - vdoth, 5.0) * (1.0 - F0) + F0

    # Beckmann distribution
    #r1 = tf.math.divide_no_nan(1.0, (m_squared * tf.math.pow(ndoth, 4.0)))
    #r2 = tf.math.divide_no_nan(
    #    (ndoth * ndoth - 1.0), (m_squared * ndoth * ndoth)
    #    )
    #D = r1 * tf.math.exp(r2)

    m_squared = roughness * roughness
    denom = (ndoth * ndoth) * (m_squared - 1) + 1.0
    D = tf.math.divide_no_nan(m_squared, np.pi * denom * denom)

    # Geometric shadowing
    g1 = tf.math.divide_no_nan(2 * ndoth * ndotv, vdoth)
    g2 = tf.math.divide_no_nan(2 * ndoth * ndotl, vdoth)
    G = tf.math.minimum(1.0, tf.math.minimum(g1, g2))

    rs = tf.math.divide_no_nan(
        F * D * G,
        4.0 * ndotl * ndotv
        )

    return intensity * ndotl * (kd + rs)