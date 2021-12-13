import tensorflow as tf
import numpy as np
import time

from types import SimpleNamespace

from utils.projection_utils import *
from utils.sampling_utils import *
from utils.temporal_utils import *
from utils.tof_utils import *
from utils.nerf_utils import *

def convert_to_outputs(raw, z_vals, rays_o, rays_d, pts, light_pos, near, far, visibility, args, chunk_inputs):
    ## Setup

    outputs = {}

    # Non-linearities for time-of-flight
    tof_nl_fn = (lambda x: tf.abs(x)) if args.use_falloff else tf.math.sigmoid
    tof_nl_fn_basis = (lambda x: x) if args.use_falloff else tf.math.tanh

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

    if 'radiance_map' in args.outputs:
        radiance_map = compute_color(0, raw, weights, args)
        outputs['radiance_map'] = radiance_map

    ## Time-of-flight

    if 'tof_map' in args.outputs:
        tof_map = compute_tof(
            3, raw, dists_to_cam, dists_total,
            weights, transmittance, visibility,
            tof_nl_fn, args,
            args.use_phasor,
            chunk_inputs=chunk_inputs
            )
        outputs['tof_map'] = tof_map

    ## Other outputs

    if 'disp_map' in args.outputs or 'depth_map' in args.outputs:
        depth_map = tf.reduce_sum(weights * dists_to_cam, axis=-1)
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

@tf.function(experimental_compile=True)
def render_rays(
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

    #scale_fac = args.scene_scale / (args.far - args.near)
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
        outputs, weights = convert_to_outputs(
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
        outputs_fine, _ = convert_to_outputs(
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

def batchify_rays(
    batch_inputs,
    chunk=1024*32,
    render_rays_fn=render_rays,
    **kwargs
    ):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}

    for i in range(0, batch_inputs['rays_o'].shape[0], chunk):
        # Create chunk
        chunk_inputs = {}

        for key in batch_inputs:
            if batch_inputs[key] is not None:
                chunk_inputs[key] = batch_inputs[key][i:i+chunk]
            else:
                chunk_inputs[key] = None
        
        # Run
        ret = render_rays_fn(chunk_inputs, **kwargs)

        # Get outputs
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []

            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret

def batchify_rays_reduce(
    batch_inputs,
    chunk=1024*32,
    render_rays_fn=render_rays,
    **kwargs
    ):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = 0.0

    for i in range(0, batch_inputs['rays_o'].shape[0], chunk):
        # Create chunk
        chunk_inputs = {}

        for key in batch_inputs:
            if batch_inputs[key] is not None:
                chunk_inputs[key] = batch_inputs[key][i:i+chunk]
            else:
                chunk_inputs[key] = None
        
        # Run
        all_ret += render_rays_fn(chunk_inputs, **kwargs)

    return all_ret

def render(
    H, W,
    chunk=1024*32,
    rays=None,
    ray_gen_fn=None,
    pose=None,
    light_pose=None,
    near=0.,
    far=1.,
    batch_inputs={},
    **kwargs
    ):

    if rays is None:
        rays_o, rays_d = ray_gen_fn(pose)
        light_pos = tf.broadcast_to(light_pose[..., :3, -1], rays_o.shape)
    else:
        rays_o, rays_d, light_pos = rays

    out_shape = rays_d.shape

    # Create ray batch
    batch_inputs['rays_o'] = rays_o
    batch_inputs['rays_d'] = rays_d
    batch_inputs['light_pos'] = light_pos
    batch_inputs['viewdirs'] = normalize(rays_d)
    batch_inputs['near'] = near * tf.ones_like(rays_d[..., :1])
    batch_inputs['far'] = far * tf.ones_like(rays_d[..., :1])

    for key in batch_inputs:
        inp = batch_inputs[key]

        if inp is not None:
            batch_inputs[key] = \
                tf.cast(
                    tf.reshape(inp, [-1, inp.shape[-1]]), dtype=tf.float32
                    )
        else:
            batch_inputs[key] = None

    # Render
    all_ret = batchify_rays(
        batch_inputs, chunk,
        near=near, far=far,
        **kwargs
        )
    
    # Reshape
    for k in all_ret:
        k_sh = list(out_shape[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)

    return all_ret

def render_reduce(
    H, W,
    chunk=1024*32,
    rays=None,
    ray_gen_fn=None,
    pose=None,
    light_pose=None,
    near=0.,
    far=1.,
    batch_inputs={},
    **kwargs
    ):

    if rays is None:
        rays_o, rays_d = ray_gen_fn(pose)
        light_pos = tf.broadcast_to(light_pose[..., :3, -1], rays_o.shape)
    else:
        rays_o, rays_d, light_pos = rays

    out_shape = rays_d.shape

    # Create ray batch
    batch_inputs['rays_o'] = rays_o
    batch_inputs['rays_d'] = rays_d
    batch_inputs['light_pos'] = light_pos
    batch_inputs['viewdirs'] = normalize(rays_d)
    batch_inputs['near'] = near * tf.ones_like(rays_d[..., :1])
    batch_inputs['far'] = far * tf.ones_like(rays_d[..., :1])

    for key in batch_inputs:
        inp = batch_inputs[key]

        if inp is not None:
            batch_inputs[key] = \
                tf.cast(
                    tf.reshape(inp, [-1, inp.shape[-1]]), dtype=tf.float32
                    )
        else:
            batch_inputs[key] = None

    all_ret = batchify_rays_reduce(
        batch_inputs, chunk,
        near=near, far=far,
        **kwargs
        )
    
    return tf.reduce_sum(all_ret)

def render_path(
    i_train,
    dataset,
    all_poses,
    render_poses,
    light_poses,
    ray_gen_fn_camera,
    ray_gen_fn_light,
    H, W,
    near,
    far,
    chunk,
    savedir=None,
    render_factor=0,
    render_kwargs={},
    all_output_names=['tof_map', 'radiance_map', 'disp_map', 'depth_map', 'tof_map_dynamic', 'radiance_map_dynamic', 'disp_map_dynamic', 'depth_map_dynamic'],
    render_freezeframe=False,
    args=None
    ):
    # Render downsampled for speed
    if render_factor != 0:
        H = H // render_factor
        W = W // render_factor

    # Render all outputs
    all_outputs = {
        m: [] for m in all_output_names
        }

    t = time.time()

    for i, (pose, light_pose) \
        in enumerate(zip(render_poses, light_poses)):

        print(i, time.time() - t)
        t = time.time()

        if render_freezeframe:
            image_index = args.num_frames // 2
        else:
            image_index = (i / float(len(render_poses))) * (args.num_frames - 1),

        outputs = render(
            H=H, W=W, chunk=chunk,
            pose=pose, light_pose=light_pose,
            ray_gen_fn=ray_gen_fn_camera,
            all_poses=all_poses,
            image_height=H,
            image_width=W,
            image_index=image_index,
            **render_kwargs
        )

        for m in all_outputs:
            if m in outputs:
                all_outputs[m].append(outputs[m].numpy())

    # Stack
    for m in all_outputs:
        if len(all_outputs[m]) > 0:
            all_outputs[m] = np.stack(all_outputs[m], axis=0)

    return all_outputs

## Geometry
def raw2alpha(raw, dists, act_fn=tf.nn.relu):
    return 1.0 - tf.exp(-act_fn(raw) * dists)

def compute_geometry(
    start_idx,
    raw,
    z_vals,
    rays_d,
    dists_to_cam,
    dists,
    args,
    ):
    # Noise
    noise = 0.

    if args.raw_noise_std > 0.:
        noise = tf.random.normal(raw[..., -1].shape) * args.raw_noise_std

    # Geometry
    alpha = raw2alpha(raw[..., start_idx] + noise, dists)
    transmittance = tf.math.cumprod(
        1. - alpha + 1e-10, axis=-1, exclusive=True
        )
    weights = alpha * transmittance

    # Return
    return alpha, weights, transmittance

## Color
def compute_color(
    start_idx,
    raw,
    weights,
    args,
    no_over=False
    ):
    radiance = tf.math.sigmoid(raw[..., start_idx:start_idx+3])

    # Over-composited radiance
    if not no_over:
        radiance_map = tf.reduce_sum(
            weights[..., None] * radiance, axis=-2
            )
        
        return radiance_map
    else:
        return radiance

def compute_color_falloff(
    start_idx,
    raw,
    dists_to_light,
    weights, transmittance, visibility,
    color_nl_fn,
    args
    ):
    # R-squared falloff
    factor = get_falloff(dists_to_light, args)
    radiance = color_nl_fn(raw[..., start_idx:start_idx+3])

    # Over-composited radiance
    radiance_map = tf.reduce_sum(
        factor[..., None] * visibility * weights[..., None] * radiance, axis=-2
        )
    
    return radiance_map