import tensorflow as tf
import numpy as np
import platform
use_xla = 'Windows' in platform.system()
import time

from types import SimpleNamespace

from utils.nerf_utils import *
from utils.projection_utils import *
from utils.sampling_utils import *
from utils.temporal_utils import *
from utils.tof_utils import *
from utils.utils import *

from render import *

def convert_to_outputs_dynamic_one(
    raw, z_vals, rays_o, rays_d, pts, light_pos, near, far, visibility, args, chunk_inputs, dynamic=False
    ):
    ## Setup

    outputs = {}

    # Non-linearities for time-of-flight
    tof_nl_fn = (lambda x: tf.abs(x)) if args.use_falloff else tf.math.sigmoid

    # Distances
    dists, dists_to_cam, dists_to_light, dists_total = \
        get_dists(z_vals, rays_d, light_pos, pts)
    
    ## Geometry

    alpha, weights, transmittance = compute_geometry(
        -1, raw, z_vals, rays_d, dists_to_cam, dists, args
        )

    if visibility is None:
        visibility = transmittance[..., None]
    
    ## Color

    if 'color_map' in args.outputs:
        color_map = compute_color(0, raw, weights, args, no_over=True)
        outputs['color_map'] = color_map

    ## Time-of-flight

    if 'tof_map' in args.outputs:
        tof_map = compute_tof(
            3, raw, dists_to_cam, dists_total,
            weights, transmittance, visibility,
            tof_nl_fn, args,
            args.use_phasor and (not dynamic),
            no_over=True,
            chunk_inputs=chunk_inputs
            )
        outputs['tof_map'] = tof_map
    
    return outputs, alpha, weights

def linear_blend(a, b, blend):
    return a * (1 - blend) + b

def convert_to_outputs_dynamic(
    raw, z_vals, rays_o, rays_d, pts, light_pos, near, far, visibility, args, chunk_inputs
    ):

    outputs = {}

    ## Blend
    
    if args.static_blend_weight:
        raw_static = raw[..., :6]
        raw_dynamic = raw[..., 7:13]

        blend_weight = tf.math.sigmoid(raw[..., 6:7])
    else:
        raw_static = raw[..., :6]
        raw_dynamic = raw[..., 6:12]

        blend_weight = tf.math.sigmoid(raw[..., -1:])

    outputs_static, alpha_static, _ = convert_to_outputs_dynamic_one(
        raw_static, z_vals, rays_o, rays_d, pts, light_pos, near, far, visibility, args, chunk_inputs, dynamic=False
    )

    outputs_dynamic, alpha_dynamic, _ = convert_to_outputs_dynamic_one(
        raw_dynamic, z_vals, rays_o, rays_d, pts, light_pos, near, far, visibility, args, chunk_inputs, dynamic=True
    )

    alpha_static = alpha_static[..., None]
    alpha_dynamic = alpha_dynamic[..., None]
    alpha_blended = linear_blend(alpha_static, alpha_dynamic, blend_weight)

    transmittance_blended = tf.math.cumprod(
        1. - alpha_blended + 1e-10, axis=-2, exclusive=True
    )
    weights_blended = alpha_blended * transmittance_blended

    if 'color_map' in args.outputs:
        color_blended = linear_blend(
            outputs_static['color_map'] * alpha_static, outputs_dynamic['color_map'] * alpha_dynamic, blend_weight
            )

        color_map = tf.reduce_sum(
            transmittance_blended * color_blended,
            axis=-2
            ) 

        outputs['color_map'] = color_map

    if 'tof_map' in args.outputs:
        tof_blended = linear_blend(
            outputs_static['tof_map'] * alpha_static, outputs_dynamic['tof_map'] * alpha_dynamic, blend_weight
            )

        if args.square_transmittance:
            tof_map = tf.reduce_sum(
                transmittance_blended * transmittance_blended * tof_blended,
                axis=-2
                ) 
        else:
            tof_map = tf.reduce_sum(
                transmittance_blended * tof_blended,
                axis=-2
                ) 

        outputs['tof_map'] = tof_map

    if 'disp_map' in args.outputs or 'depth_map' in args.outputs:
        depth_map = tf.reduce_sum(weights_blended * z_vals[..., None], axis=[-1, -2])
        disp_map = 1. / tf.maximum(
            1e-10, depth_map / (tf.reduce_sum(weights_blended, axis=[-1, -2]) + 1e-5)
            )

        outputs['depth_map'] = depth_map
        outputs['disp_map'] = disp_map

    ## Dynamic outputs
    alpha_dynamic = alpha_dynamic * blend_weight
    transmittance_dynamic = tf.math.cumprod(
        1. - alpha_dynamic + 1e-10, axis=-2, exclusive=True
    )
    weights_dynamic = alpha_dynamic * transmittance_dynamic

    if 'color_map_dynamic' in args.outputs:
        color_map = tf.reduce_sum(
            weights_dynamic * outputs_dynamic['color_map'],
            axis=-2
            ) 
        outputs['color_map_dynamic'] = color_map

    if 'tof_map_dynamic' in args.outputs:
        if args.square_transmittance:
            tof_map = tf.reduce_sum(
                transmittance_dynamic * transmittance_dynamic * outputs_dynamic['tof_map'],
                axis=-2
                ) 
        else:
            tof_map = tf.reduce_sum(
                weights_dynamic * outputs_dynamic['tof_map'],
                axis=-2
                ) 

        outputs['tof_map_dynamic'] = tof_map

    if 'disp_map_dynamic' in args.outputs or 'depth_map_dynamic' in args.outputs:
        depth_map = tf.reduce_sum(weights_dynamic * z_vals[..., None], axis=[-1, -2])
        disp_map = 1. / tf.maximum(
            1e-10, depth_map / (tf.reduce_sum(weights_dynamic, axis=[-1, -2]) + 1e-5)
        )
        outputs['depth_map_dynamic'] = depth_map
        outputs['disp_map_dynamic'] = disp_map

    ## Other outputs

    if 'acc_map' in args.outputs:
        acc_map = tf.reduce_mean(
            -blend_weight * tf.math.log(blend_weight + 1e-8),
            axis=[-1, -2]
        )
        outputs['acc_map'] = acc_map
    
    ## Return

    return outputs, weights_blended[..., 0]

@tf.function(experimental_compile=use_xla)
def render_rays_dynamic(
    chunk_inputs,
    **kwargs
    ):
    # Outputs
    outputs = {}
    outputs_fine = {}

    # Extract inputs
    args = SimpleNamespace(**kwargs)

    rays_o = chunk_inputs['rays_o']
    rays_d = chunk_inputs['rays_d']
    light_pos = chunk_inputs['light_pos']
    viewdirs = chunk_inputs['viewdirs']

    near = chunk_inputs['near']
    far = chunk_inputs['far']
    N_rays = rays_o.shape[0]

    scale_fac = np.array(
        [args.scene_scale_x, args.scene_scale_y, args.scene_scale],
        )[None] / (args.far - args.near)
    
    ## Coarse model

    coarse_z_vals = coarse_samples(near, far, args)
    weights = None

    if not args.use_depth_sampling:
        # Perturb sample distances
        if args.perturb > 0.:
            coarse_z_vals = perturb_samples(coarse_z_vals)

        # Sample points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            coarse_z_vals[..., :, None]

        # Evaluate coarse model
        latent_code = temporal_input(args.image_index, args)

        raw = args.network_query_fn[0](
            pts, scale_fac, viewdirs, latent_code, args.models
            )

        # Raw to outputs
        outputs, weights = convert_to_outputs_dynamic(
            raw, coarse_z_vals, rays_o, rays_d, pts, light_pos, near, far, None, args, chunk_inputs
            )

    ## Fine model

    if args.N_importance > 0:
        # Fine sample distances
        fine_z_vals = fine_samples(coarse_z_vals, near, far, weights, chunk_inputs, args)

        if not args.use_depth_sampling:
            z_vals = tf.sort(tf.concat([coarse_z_vals, fine_z_vals], -1), -1)
        else:
            z_vals = tf.sort(fine_z_vals, -1)

        # Query points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]

        # Evaluate fine model
        latent_code = temporal_input(args.image_index, args)

        raw = args.network_query_fn[0](
            pts, scale_fac, viewdirs, latent_code, args.models, fine=True
            )

        # Raw to outputs
        outputs_fine, _ = convert_to_outputs_dynamic(
            raw, z_vals, rays_o, rays_d, pts, light_pos, near, far, None, args, chunk_inputs
            )

    # Return values from convert_to_outputs
    ret = {}

    if args.N_importance > 0:
        for key in outputs:
            ret[key + '0'] = outputs[key]

        for key in outputs_fine:
            ret[key] = outputs_fine[key]
    else:
        for key in outputs:
            ret[key] = outputs[key]

    # Other return values
    if args.N_importance > 0:
        if 'z_std' in args.outputs:
            ret['z_std'] = tf.math.reduce_std(z_vals, -1)

    # Check numerics
    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret