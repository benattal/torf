import tensorflow as tf
import numpy as np
import time

from types import SimpleNamespace

from utils.nerf_utils import *
from utils.projection_utils import *
from utils.sampling_utils import *
from utils.shading_utils import *
from utils.shadow_utils import *
from utils.temporal_utils import *
from utils.tof_utils import *
from utils.utils import *

from render import *

def compute_normals(
    start_idx,
    raw
    ):
    return normalize(raw[..., start_idx:start_idx+3])

def compute_tof_lambert(
    start_idx,
    raw,
    dists_to_light, dists_total,
    pts, normals, light_dirs, view_dirs,
    weights, transmittance, visibility,
    tof_nl_fn,
    args
    ):
    # Material parameters
    tof_kd = tof_nl_fn(raw[..., start_idx:start_idx+1])

    # Lambertian shading
    lambert_factor = lambert(normals, light_dirs, 1.0)
    falloff_factor = get_falloff(dists_to_light, args)
    direct = tof_kd * lambert_factor

    # Calculate TOF phasor
    phasor_phase = dists_total * (2 * np.pi / args.depth_range)
    if hasattr(args, 'phase_offset'):
        phasor_phase += args.phase_offset
    phasor_real, phasor_imag = \
        get_phasor(phasor_phase, direct)

    tof_direct = tf.concat(
        [phasor_real, phasor_imag, direct],
        axis=-1
        )

    # Over-composite
    tof_albedo_map = tf.reduce_sum(
        visibility * weights[..., None] * tof_kd,
        axis=-2
        ) 

    tof_direct_map = tf.reduce_sum(
        visibility * weights[..., None] * tof_direct \
           * falloff_factor[..., None],
        axis=-2
        ) 

    return tof_albedo_map, tof_direct_map, tf.ones_like(tof_albedo_map)

def compute_color_lambert(
    start_idx,
    raw,
    dists_to_light,
    pts, normals, light_dirs, view_dirs,
    weights, transmittance,
    color_nl_fn,
    args
    ):
    # Material parameters
    kd = color_nl_fn(raw[..., start_idx:start_idx+3])

    # Lambertian shading
    lambert_factor = lambert(normals, light_dirs, 1.0)
    falloff_factor = get_falloff(dists_to_light, args)
    direct = kd * lambert_factor

    # Over-composite
    albedo_map = tf.reduce_sum(
        weights[..., None] * kd,
        axis=-2
        ) 
    
    direct_map = tf.reduce_sum(
        weights[..., None] * direct \
            * falloff_factor[..., None],
        axis=-2
        ) 

    return albedo_map, direct_map, tf.ones_like(albedo_map)

def compute_tof_microfacet(
    start_idx,
    raw,
    dists_to_light, dists_total,
    pts, normals, light_dirs, view_dirs,
    weights, transmittance, visibility,
    tof_nl_fn,
    args
    ):

    # Cook-Torrance shading
    tof_kd = tof_nl_fn(raw[..., start_idx:start_idx+1])
    tof_metalness = tf.math.sigmoid(raw[..., start_idx+1:start_idx+2])
    tof_roughness = tf.math.sigmoid(raw[..., start_idx+2:start_idx+3])
    falloff_factor = get_falloff(dists_to_light, args)

    direct = cook_torrance(
        tof_kd, tof_metalness, tof_roughness,
        normals, view_dirs[..., None, :], light_dirs,
        1.0
        )

    # Calculate TOF phasor
    phasor_phase = dists_total * (2 * np.pi / args.depth_range)
    if hasattr(args, 'phase_offset'):
        phasor_phase += args.phase_offset
    phasor_real, phasor_imag = \
        get_phasor(phasor_phase, direct)

    tof_direct = tf.concat(
        [phasor_real, phasor_imag, direct],
        axis=-1
        )

    # Over-composite
    tof_albedo_map = tf.reduce_sum(
        weights[..., None] * tof_kd,
        axis=-2
        ) 

    tof_roughness_map = tf.reduce_sum(
        weights[..., None] * tof_roughness,
        axis=-2
        ) 

    tof_direct_map = tf.reduce_sum(
        visibility * weights[..., None] * tof_direct \
            * falloff_factor[..., None],
        axis=-2
        ) 

    return tof_albedo_map, tof_direct_map, tof_roughness_map

def compute_color_microfacet(
    start_idx,
    raw,
    dists_to_light,
    pts, normals, light_dirs, view_dirs,
    weights, transmittance,
    color_nl_fn,
    args
    ):

    # Cook-Torrance shading
    kd = color_nl_fn(raw[..., start_idx:start_idx+3])
    metalness = tf.math.sigmoid(raw[..., start_idx+3:start_idx+4])
    roughness = tf.math.sigmoid(raw[..., start_idx+4:start_idx+5])
    falloff_factor = get_falloff(dists_to_light, args)

    direct = cook_torrance(
        kd, metalness, roughness,
        normals, view_dirs[..., None, :], light_dirs,
        1.0
        )

    # Over-composite
    albedo_map = tf.reduce_sum(
        weights[..., None] * kd,
        axis=-2
        ) 

    roughness_map = tf.reduce_sum(
        weights[..., None] * roughness,
        axis=-2
        ) 

    direct_map = tf.reduce_sum(
        weights[..., None] * direct \
            * falloff_factor[..., None],
        axis=-2
        ) 

    return albedo_map, direct_map, roughness_map

def get_render_features(start_idx, raw, weights, args):
    features = raw[..., start_idx:args.renderfeatures+start_idx]
    feature_map = tf.reduce_sum(
        weights[..., None] * features, axis=-2
        )

    return feature_map

def convert_to_outputs_deferred(
    raw, z_vals, rays_o, rays_d, pts, light_pos, near, far,
    args,
    fine=False
    ):
    ## Setup
    outputs = {}

    dists, dists_to_cam, dists_to_light, dists_total = \
        get_dists(z_vals, rays_d, light_pos, pts)

    view_dirs = -normalize(rays_d)
    light_dirs = normalize(light_pos[..., None, :] - pts)
    
    ## Geometry

    alpha, weights, transmittance = compute_geometry(
        -1, raw, dists, args
        )

    ## Normals

    if (args.use_brdf and fine) and args.use_analytic_normals:
        scale_fac = args.scene_scale / (args.far - args.near)
        pts_grad = pts

        if args.dynamic:
            latent_code = temporal_input(args.image_index, args)

            grads = args.network_query_fn[-1](
                pts_grad, scale_fac, rays_d, latent_code, args.models, fine=True
                )
        else:
            grads = args.network_query_fn[-1](
                pts_grad, scale_fac, rays_d, args.models, fine=True
                )

        normals = normalize(-grads)
    else:
        normals = compute_normals(-4, raw)

    normal_map = tf.reduce_sum(
        weights[..., None] * normals, axis=-2
        )
    outputs['normal_map'] = normal_map
    
    ## Color

    if args.use_brdf and fine:
        #albedo_map, direct_map, roughness_map = compute_color_lambert(
        albedo_map, direct_map, roughness_map = compute_color_microfacet(
            4,
            raw,
            dists_to_light,
            pts, normals, light_dirs, view_dirs,
            weights, transmittance,
            tf.math.abs,
            args
            )

        outputs['albedo_map'] = albedo_map
        outputs['direct_map'] = direct_map
        outputs['roughness_map'] = roughness_map
    else:
        albedo_map = compute_color_falloff(
            0,
            raw,
            dists_to_light,
            weights, transmittance, transmittance[..., None],
            tf.math.abs,
            args
            )

        outputs['albedo_map'] = albedo_map
        outputs['direct_map'] = albedo_map
        outputs['roughness_map'] = albedo_map

    ## TOF
    if args.use_brdf and fine:
        #tof_albedo_map, tof_direct_map, tof_roughness_map = compute_tof_lambert(
        tof_albedo_map, tof_direct_map, tof_roughness_map = compute_tof_microfacet(
            9,
            raw,
            dists_to_cam, (2 * dists_to_cam)[..., None],
            pts, normals, view_dirs[..., None, :], view_dirs,
            weights, transmittance, transmittance[..., None],
            tf.math.abs,
            args
            )

        outputs['tof_albedo_map'] = tof_albedo_map
        outputs['tof_direct_map'] = tof_direct_map
        outputs['tof_roughness_map'] = tof_roughness_map
    else:
        tof_albedo_map = compute_tof(
            3,
            raw,
            dists_to_cam, (2 * dists_to_cam)[..., None],
            weights, transmittance, transmittance[..., None],
            tf.math.abs,
            args,
            args.use_phasor
            )
        outputs['tof_albedo_map'] = tof_albedo_map
        outputs['tof_direct_map'] = tof_albedo_map
        outputs['tof_roughness_map'] = tof_albedo_map

    ## Features

    if args.use_residual:
        color_feature_map = get_render_features(12, raw, weights, args)
        outputs['color_feature_map_global'] = color_feature_map

        tof_feature_map = get_render_features(12 + args.renderfeatures, raw, weights, args)
        outputs['tof_feature_map_global'] = tof_feature_map

    ## Other outputs

    if 'raw' in args.outputs:
        outputs['raw'] = raw

    if ('disp_map' in args.outputs) or ('depth_map' in args.outputs):
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1)
        disp_map = 1. / tf.maximum(
            1e-10, depth_map / (tf.reduce_sum(weights, axis=-1) + 1e-5)
            )

        outputs['depth_map'] = depth_map
        outputs['disp_map'] = disp_map

    if 'acc_map' in args.outputs:
        acc_map = tf.reduce_mean(
            tf.math.log(1 + tf.math.square(raw[..., -1]) / 0.5), axis=[-1]
            )
        outputs['acc_map'] = acc_map
    
    ## Return

    return outputs, weights

def next_event_intensity(
    rays_o, rays_d, light_pos,
    near, far,
    args
    ):
    viewdirs = normalize(rays_d)
    scale_fac = args.scene_scale / (args.far - args.near)
    
    ## Coarse model

    # Sampling distances
    stop = tf.linalg.norm(light_pos - rays_o, axis=-1)
    start = stop * 0.01

    ## Coarse samples
    coarse_z_vals = shadow_samples(start, stop, args)
    coarse_z_vals = tf.sort(coarse_z_vals, -1)
    coarse_z_vals = tf.stop_gradient(coarse_z_vals)

    # Sample points
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        coarse_z_vals[..., :, None]
    
    # Evaluate coarse model
    if args.dynamic:
        latent_code = temporal_input(args.image_index, args)

        raw = args.network_query_fn[0](
            pts, scale_fac, normalize(rays_d), latent_code, args.models
            )
    else:
        raw = args.network_query_fn[0](
            pts, scale_fac, normalize(rays_d), args.models
            )

    # Visibility
    dists = (coarse_z_vals[..., 1:] - coarse_z_vals[..., :-1])
    dists = tf.concat(
        [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
        axis=-1
        )
    dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

    alpha, weights, transmittance = compute_geometry(
        -1, raw, dists, args
        )
    
    return transmittance[..., -1:]

def shade_color_global(start_idx, raw_rendered, outputs, args):
    specular_map = raw_rendered[..., start_idx:start_idx+3]
    radiance_map = outputs['direct_map'] + specular_map
    outputs['radiance_map'] = radiance_map
    outputs['specular_map'] = specular_map

def shade_tof_global(start_idx, raw_rendered, outputs, args):
    tof_map = outputs['tof_direct_map']
    tof_real, tof_imag, tof_amp \
        = extract_tof(tof_map)

    # Phasor bias
    if args.use_phasor:
        bias_map = tf.math.sigmoid(raw_rendered[..., start_idx+1:start_idx+2]) \
            * 2 * np.pi * (args.bias_range / args.depth_range)
        bias_real, bias_imag = get_phasor(bias_map)
        tof_real, tof_imag \
            = mul_phasors(tof_real, tof_imag, bias_real, bias_imag)
        outputs['tof_bias_map'] = bias_map
    
    # Phasor amplitude
    specular_map = raw_rendered[..., start_idx:start_idx+1]
    tof_real, tof_imag = add_same_phase(
        (tof_real, tof_imag, tof_amp), specular_map
        )

    outputs['tof_specular_map'] = specular_map

    # Concatenate
    tof_amp = get_amplitude(tof_real, tof_imag)
    tof_map = tf.concat([tof_real, tof_imag, tof_amp], axis=-1)
    outputs['tof_map'] = tof_map

def deferred_shading(
    outputs,
    rays_o, rays_d, viewdirs, light_pos,
    near, far,
    args
    ):
    if args.use_brdf:
        # Visibility
        z_vals = outputs['depth_map'][..., None]
        pts = rays_o + z_vals * rays_d
        light_dirs = normalize(light_pos - pts)

        visibility = next_event_intensity(
            pts, light_dirs, light_pos,
            near, far,
            args
            )

        outputs['direct_map'] = outputs['direct_map'] * visibility

        # Add outputs
        outputs['radiance_map'] = outputs['direct_map']
        outputs['tof_map'] = outputs['tof_direct_map']
    else:
        outputs['radiance_map'] = outputs['albedo_map']
        outputs['tof_map'] = outputs['tof_albedo_map']

    if args.use_residual:
        z_vals = outputs['depth_map'][..., None]
        pts = rays_o + z_vals * rays_d
        light_dirs = normalize(light_pos - pts)

        raw_color = args.network_query_fn[1](
            outputs['color_feature_map_global'], outputs['direct_map'], viewdirs, light_dirs, args.models, 'color_render_model_global'
            )
        raw_tof = args.network_query_fn[1](
            outputs['tof_feature_map_global'], outputs['tof_direct_map'], viewdirs, -viewdirs, args.models, 'tof_render_model_global'
            )

        shade_color_global(0, raw_color, outputs, args)
        shade_tof_global(0, raw_tof, outputs, args)

def render_rays_deferred(
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

    scale_fac = args.scene_scale / (args.far - args.near)
    
    ## Coarse model

    # Sampling distances
    coarse_z_vals = coarse_samples(near, far, args)
    weights = None

    if not args.use_depth_sampling \
        or (chunk_inputs['sampling_volume'] is None):
        # Perturb sample distances
        if args.perturb > 0.:
            coarse_z_vals = perturb_samples(coarse_z_vals)

        # Sample points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            coarse_z_vals[..., :, None]
        
        # Evaluate coarse model
        if args.dynamic:
            latent_code = temporal_input(args.image_index, args)
        
            raw = args.network_query_fn[0](
                pts, scale_fac, viewdirs, latent_code, args.models
                )
        else:
            raw = args.network_query_fn[0](
                pts, scale_fac, viewdirs, args.models
                )

        # Raw to outputs
        outputs, weights = convert_to_outputs_deferred(
            raw, coarse_z_vals, rays_o, rays_d, pts, light_pos, near, far,
            args, fine=False
            )

    ## Fine model

    if args.N_importance > 0:
        # Fine sample distances
        fine_z_vals = fine_samples(coarse_z_vals, near, far, weights, chunk_inputs, args)

        if not args.use_depth_sampling \
            or (chunk_inputs['sampling_volume'] is None):
            z_vals = tf.sort(tf.concat([coarse_z_vals, fine_z_vals], -1), -1)
        else:
            z_vals = tf.sort(fine_z_vals, -1)

        # Query points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]

        # Evaluate fine model
        if args.dynamic:
            latent_code = temporal_input(args.image_index, args)

            raw = args.network_query_fn[0](
                pts, scale_fac, viewdirs, latent_code, args.models, fine=True
                )
        else:
            raw = args.network_query_fn[0](
                pts, scale_fac, viewdirs, args.models, fine=True
                )

        # Raw to outputs
        outputs_fine, _ = convert_to_outputs_deferred(
            raw, z_vals, rays_o, rays_d, pts, light_pos, near, far,
            args, fine=True
            )
        
        deferred_shading(
            outputs_fine,
            rays_o, rays_d, viewdirs, light_pos,
            near, far,
            args
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
