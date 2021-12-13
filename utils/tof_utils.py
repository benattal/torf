import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def get_phasor(phase, amp=1.):
    return tf.math.cos(phase) * amp, tf.math.sin(phase) * amp

def mul_phasors(p1_real, p1_imag, p2_real, p2_imag):
    p_real = p1_real * p2_real - p1_imag * p2_imag
    p_imag = p1_real * p2_imag + p1_imag * p2_real

    return (p_real, p_imag)

def get_phase(phasor_real, phasor_imag):
    return tf.math.atan2(phasor_imag, phasor_real)

def get_amplitude(phasor_real, phasor_imag):
    return tf.math.sqrt(
        phasor_real * phasor_real + phasor_imag * phasor_imag
        )

def add_same_phase(phasor, amp):
    phasor_real, phasor_imag, phasor_amp = phasor
    inv_phasor_amp = 1. / (phasor_amp + 1e-1)

    return mul_phasors(phasor_real, phasor_imag, amp * inv_phasor_amp + 1, 0)

def same_phase(phasor, amp):
    phasor_real, phasor_imag, phasor_amp = phasor
    inv_phasor_amp = 1. / (phasor_amp + 1e-1)

    return mul_phasors(phasor_real, phasor_imag, amp * inv_phasor_amp, 0)

def convert_to_square_wave(
        tof_im
        ):
    R = tof_im[..., -1]
    phi = tf.math.atan2(tof_im[..., 1], tof_im[..., 0])

    t1 = phi / np.pi

    t2 = (phi - np.pi / 2) / np.pi
    t2 = tf.where( t2 < 0, t2 + 2, t2 )
    t2 = tf.where( t2 > 1, t2 - 2, t2 )

    m1 = (1 - 2 * tf.abs(t1)) * R
    m2 = (1 - 2 * tf.abs(t2)) * R

    return tf.stack([m1, m2, R], axis=-1)

def get_falloff(dists_to_light, args):
    # R-squared falloff
    if args.use_falloff:
        factor = (0.5 / (dists_to_light * dists_to_light)) \
            * (args.depth_range * args.depth_range) / (args.falloff_range * args.falloff_range)
    else:
        factor = tf.ones_like(dists_to_light)
    
    return factor

def compute_tof(
    start_idx,
    raw,
    dists_to_light, dists_total,
    weights, transmittance, visibility,
    tof_nl_fn,
    args,
    use_phasor=False,
    no_over=False,
    chunk_inputs=None
    ):
    # R-squared falloff
    factor = get_falloff(dists_to_light, args)

    # TOF phasor
    phasor_amp = tof_nl_fn(raw[..., start_idx:start_idx+1])
    phasor_phase = dists_total * (2 * np.pi / args.depth_range)

    if hasattr(args, 'phase_offset'):
        phasor_phase += args.phase_offset

    if chunk_inputs is not None and args.use_phase_calib:
        phasor_phase = args.tof_phase_model(
            phasor_phase,
            tf.broadcast_to(chunk_inputs['coords'][..., None, :], phasor_phase.shape[:-1] + (2,)),
            args.models
            )

    if use_phasor:
        bias_phase = tf.math.sigmoid(
            raw[..., start_idx+1:start_idx+2]
            ) * 2 * np.pi * (args.bias_range / args.depth_range)
        phasor_phase += bias_phase
    
    # Full TOF phasor
    phasor_real, phasor_imag = \
        get_phasor(phasor_phase, phasor_amp)

    tof = tf.concat(
        [phasor_real, phasor_imag, phasor_amp],
        axis=-1
        ) * factor[..., None]

    if not no_over:
        # Over-composited TOF
        tof_map = tf.reduce_sum(
            visibility * weights[..., None] * tof,
            axis=-2
            ) 
        
        return tof_map
    else:
        return tof

def extract_tof(tof):
    return (tof[..., 0:1], tof[..., 1:2], tof[..., 2:3])