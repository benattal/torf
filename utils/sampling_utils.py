import numpy as np
import tensorflow as tf
from scipy import ndimage, misc

from utils.projection_utils import *

def depth_transform_inv(t_depth, near, far):
    return np.exp(t_depth * np.log(far + 1) - 1)

def depth_transform(depth, near, far):
    return np.log(depth + 1) / np.log(far + 1)

def importance_sample_error(error_maps, N_samples):
    # Sample frame
    frame_pmf = np.sum(error_maps, axis=(1, 2))
    image_index = sample_pdf(
        np.arange(0, error_maps.shape[0]).astype(np.float32), frame_pmf, 1
        )
    image_index = int(np.squeeze(np.round(image_index)))

    # Sample pixels
    pixel_pmf = np.reshape(error_maps[image_index], [-1])
    pixel_index = sample_pdf(
        np.arange(0, pixel_pmf.shape[0]).astype(np.float32), pixel_pmf, N_samples
        )
    pixel_index = np.squeeze(np.round(pixel_index))

    pixel_x = np.mod(pixel_index, error_maps.shape[-1])
    pixel_y = (pixel_index // error_maps.shape[-1])
    pixels = np.stack([pixel_y, pixel_x], axis=-1).astype(np.int32)

    return image_index, pixels

def sample_pdf(bins, weights, N_samples, det=False, base_uncertainty=1e-5):

    # Get pdf
    weights += base_uncertainty  # prevent nans
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def coarse_samples_np(near, far, args):
    min_depth = near
    max_depth = far

    # Generate samples
    t_vals = np.linspace(0., 1., args.N_samples)

    if not args.lindisp:
        z_vals = min_depth * (1. - t_vals) + max_depth * (t_vals)
    else:
        z_vals = 1. / (1. / min_depth * (1. - t_vals) + 1. / max_depth * (t_vals))
    
    return z_vals

def shadow_samples(near, far, args):
    t_vals = tf.cast(tf.linspace(0., 1., args.N_shadow_samples), near.dtype)[None]
    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * (t_vals)
    
    return z_vals

def coarse_samples_np(near, far, args):
    min_depth = near
    max_depth = far

    # Generate samples
    t_vals = np.linspace(0., 1., args.N_samples)

    if not args.lindisp:
        z_vals = min_depth * (1. - t_vals) + max_depth * (t_vals)
    else:
        z_vals = 1. / (1. / min_depth * (1. - t_vals) + 1. / max_depth * (t_vals))
    
    return z_vals

def coarse_samples(near, far, args):
    min_depth = near
    max_depth = far

    # Generate samples
    t_vals = tf.cast(tf.linspace(0., 1., args.N_samples), near.dtype)

    if not args.lindisp:
        z_vals = min_depth * (1. - t_vals) + max_depth * (t_vals)
    else:
        z_vals = 1. / (1. / min_depth * (1. - t_vals) + 1. / max_depth * (t_vals))
    
    return z_vals

def perturb_samples(z_vals):
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = tf.concat([mids, z_vals[..., -1:]], -1)
    lower = tf.concat([z_vals[..., :1], mids], -1)

    t_rand = tf.random.uniform(z_vals.shape)
    z_vals = lower + (upper - lower) * t_rand

    return z_vals

def fine_samples(coarse_z_vals, near, far, weights, chunk_inputs, args):
    if 'sampling_volume' in chunk_inputs \
        and (chunk_inputs['sampling_volume'] is not None) \
            and args.use_depth_sampling:
        weights = tf.reshape(chunk_inputs['sampling_volume'], [-1, args.N_samples])

    # PDF distances
    coarse_z_vals_mid = .5 * (coarse_z_vals[..., 1:] + coarse_z_vals[..., :-1])
    fine_z_vals = sample_pdf(
        coarse_z_vals_mid, weights[..., 1:-1], args.N_importance,
        det=(args.perturb == 0.), base_uncertainty=args.base_uncertainty
        )
    fine_z_vals = tf.stop_gradient(fine_z_vals)

    # Fine sample points
    return fine_z_vals

def shadow_fine_samples(coarse_z_vals, near, far, weights, args):
    # PDF distances
    coarse_z_vals_mid = .5 * (coarse_z_vals[..., 1:] + coarse_z_vals[..., :-1])
    fine_z_vals = sample_pdf(
        coarse_z_vals_mid, weights[..., 1:-1], args.N_shadow_samples,
        det=(args.perturb == 0.), base_uncertainty=args.base_uncertainty
        )
    fine_z_vals = tf.stop_gradient(fine_z_vals)

    # Fine sample points
    return fine_z_vals

def repeat_int(x, num_repeats):
    x = tf.tile(tf.expand_dims(x, axis=1), [1, num_repeats])
    return tf.reshape(x, [-1])

def constant_weight(d, s):
    return tf.ones_like(d)

def linear_weight(d, s):
    return tf.math.maximum(1 - tf.abs(d), 0.)

def euclidean_weight(k):
    def weight(diff_x, shift_x, diff_y, shift_y):
        dist = tf.math.sqrt(diff_x * diff_x + diff_y * diff_y)
        return tf.math.maximum(1.0 - dist / (np.sqrt(2) * k), 0.0)

    return weight

def interpolate_image(
        image,
        pixels,
        shifts_x=list(range(-3, 5)),
        shifts_y=list(range(-3, 5)),
        weight_fn=euclidean_weight(3)
        ):
    batch_size, pixels_height, pixels_width, _ = pixels.shape
    _, height, width, channels = image.shape
    image_shape = [batch_size, pixels_height, pixels_width, channels]

    shifts = [(sx, sy) for sx in shifts_x for sy in shifts_y]

    # Unstack and reshape
    pixels = tf.transpose(pixels, [0, 3, 1, 2])
    pixels = tf.reshape(pixels, [batch_size, 2, -1])

    x, y = tf.unstack(pixels, axis=1)
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])

    # Sample points
    x0 = tf.cast(tf.floor(x), tf.int32)
    y0 = tf.cast(tf.floor(y), tf.int32)

    # Interpolate
    b = repeat_int(tf.range(batch_size), pixels_height * pixels_width)
    res = 0.
    total_weight = 0.

    for (sx, sy) in shifts:
        px, py = (x0 + sx, y0 + sy)
        diff_x, diff_y = (tf.abs(x - tf.cast(px, image.dtype)), tf.abs(y - tf.cast(py, image.dtype)))
        ind = tf.stack([b, py, px], axis=1)
        weight = weight_fn(diff_x, sx, diff_y, sy)[:, None]

        res += tf.gather_nd(image, ind) * weight
        total_weight += weight
    
    res = tf.math.divide_no_nan(res, total_weight)

    return tf.reshape(res, image_shape)

def splat_image(
        image,
        out_shape,
        pixels,
        shifts_x=list(range(-3, 5)),
        shifts_y=list(range(-3, 5)),
        weight_fn=euclidean_weight(3)
        ):
    batch_size, pixels_height, pixels_width, _ = pixels.shape
    _, height, width, channels = image.shape
    image_shape = [batch_size, pixels_height, pixels_width, channels]

    shifts = [(sx, sy) for sx in shifts_x for sy in shifts_y]

    # Unstack and reshape
    pixels = tf.transpose(pixels, [0, 3, 1, 2])
    pixels = tf.reshape(pixels, [batch_size, 2, -1])

    x, y = tf.unstack(pixels, axis=1)
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])

    # Sample points
    x0 = tf.cast(tf.floor(x), tf.int32)
    y0 = tf.cast(tf.floor(y), tf.int32)

    # Scatter for all shifts
    image = tf.reshape(image, [-1, channels])
    b = repeat_int(tf.range(batch_size), pixels_height * pixels_width)
    res = 0.0

    for (sx, sy) in shifts:
        px, py = x0 + sx, y0 + sy
        diff_x, diff_y = tf.abs(x - tf.cast(px, image.dtype)), tf.abs(y - tf.cast(py, image.dtype))

        # Pixel values
        weight = weight_fn(diff_x, sx, diff_y, sy)[..., None]
        ind = tf.stack([b, py, px], axis=1)
        res += tf.scatter_nd(ind, image * weight, out_shape)

    # Return
    return tf.reshape(res, out_shape)

def project_points(points, P):
    projected_points = transform_points(points, P)

    projected_pixels = tf.math.divide_no_nan(
        projected_points[..., :2],
        projected_points[..., -1:]
        )
    
    projected_pixels = tf.where(
        projected_points[..., -1:] < 0,
        -tf.ones_like(projected_pixels),
        projected_pixels
        )
    
    return tf.reshape(projected_pixels, points.shape[:-1] + (2,))

def depth_to_index_np(depth, near, far, args):
    depth_inds = np.floor(((depth - near) / (far - near)) * args.N_samples)
    depth_inds = np.clip(depth_inds, 0, args.N_samples - 1)
    return int(depth_inds)

def depth_to_index(depth, near, far, args):
    depth_inds = tf.math.floor(((depth - near) / (far - near)) * args.N_samples)
    depth_inds = tf.clip_by_value(depth_inds, 0, args.N_samples - 1)
    return tf.cast(depth_inds, tf.int32)

def normalize_volume_np(volume):
    volume = volume / (np.sum(volume, axis=-1, keepdims=True))

    return np.where(
        np.math.isnan(volume),
        np.zeros_like(volume),
        volume
        )

def normalize_volume(volume):
    volume = volume / (tf.reduce_sum(volume, axis=-1, keepdims=True))

    return tf.where(
        tf.math.is_nan(volume),
        tf.zeros_like(volume),
        volume
        )

def interp_sampling_volume(
    volume_probs, volume_poses,
    pose, H, W, K_inv, K_volume, near, far, args
    ):
    rays_o, rays_d = get_rays_matrix(H, W, pose @ K_inv)
    z_vals = coarse_samples(near, far, None, args)

    pose = tf.cast(pose, tf.float32)
    K_inv = tf.cast(K_inv, tf.float32)
    K_volume = tf.cast(K_volume, tf.float32)

    near = tf.cast(near, tf.float32)
    far = tf.cast(far, tf.float32)
    z_vals = tf.cast(z_vals, tf.float32)
    rays_o = tf.cast(rays_o, tf.float32)
    rays_d = tf.cast(rays_d, tf.float32)

    # Ray-march volume
    probs = [0.0 for z in z_vals]

    for volume_prob, v_pose in zip(volume_probs, volume_poses):
        v_pose = tf.cast(v_pose, tf.float32)
        inv_v_pose = tf.linalg.inv(v_pose)
        proj_matrix = K_volume @ inv_v_pose

        for i, z in enumerate(z_vals):
            points = rays_o + rays_d * z
            pixels = project_points(
                points, proj_matrix
                )

            cur_volume_prob = interpolate_image(
                volume_prob[None],
                pixels[None],
                shifts_x=[0],
                shifts_y=[0]
                )
            
            # Accumulate probability from sampling volume
            t_points = transform_points(points, inv_v_pose)
            look_up_z = t_points[..., -1]
            depth_ind = tf.cast(tf.one_hot(depth_to_index(look_up_z, near, far, args), args.N_samples), tf.bool)

            probs[i] += tf.reshape(cur_volume_prob[depth_ind[None]], cur_volume_prob.shape[:-1])
    
    return tf.stack(probs, axis=-1)

def splat_sampling_volume(
    depths, depth_poses, pose,
    ray_gen_fn, project_fn,
    H, W, near, far, args
    ):
    pose = tf.cast(pose, tf.float32)
    inv_pose = tf.linalg.inv(pose)
    near = tf.cast(near, tf.float32)
    far = tf.cast(far, tf.float32)

    # Depth projection
    probs = 0.0

    for depth, d_pose in zip(depths, depth_poses):
        # Pixels to splat to
        rays_o, rays_d = ray_gen_fn(d_pose)
        rays_o = tf.cast(rays_o, tf.float32)
        rays_d = tf.cast(rays_d, tf.float32)

        points = rays_o + rays_d * depth
        pixels = project_fn(points, pose)

        # Depth values to splat
        t_points = transform_points(points, inv_pose)
        z_vals = t_points[..., -1]

        # Encode as probability volume
        depth_inds = depth_to_index(z_vals, near, far, args)
        depth_one_hot = tf.one_hot(depth_inds, args.N_samples)

        # Splat probabilities
        probs += splat_image(
            depth_one_hot[None],
            depth_one_hot[None].shape,
            pixels[None],
            shifts_x=[0],
            shifts_y=[0]
            )
    
    return probs

def depth_to_sampling_volume(depth, near, far, args):
    depth_inds = depth_to_index(depth, tf.cast(near, tf.float32), tf.cast(far, tf.float32), args)
    sampling_volume = tf.one_hot(depth_inds, args.N_samples)

    sampling_volume = ndimage.gaussian_filter(sampling_volume, sigma=[5.0, 5.0, 0.0])
    sampling_volume = ndimage.gaussian_filter1d(sampling_volume, sigma=5.0, axis=-1)
    sampling_volume = normalize_volume(sampling_volume)

    return sampling_volume

def splat_sampling_volume_from_views(
    train_sampling_volumes, train_sampling_depths,
    all_poses, test_pose,
    ray_gen_fn, project_fn,
    H, W, near, far, render_kwargs, args
    ):

    # Sampling volume
    sampling_volume = 0.0

    for img_i in range(train_sampling_volumes.shape[0]):
        train_pose = all_poses[img_i]
        train_depth = train_sampling_depths[img_i][..., None]

        sampling_volume += splat_sampling_volume(
            [train_depth], [train_pose], test_pose,
            ray_gen_fn, project_fn,
            H, W, near, far, args
            )[0]

    sampling_volume = ndimage.gaussian_filter(sampling_volume, sigma=[5.0, 5.0, 0.0])
    sampling_volume = ndimage.gaussian_filter1d(sampling_volume, sigma=5.0)
    sampling_volume = normalize_volume(sampling_volume)

    return sampling_volume

def sample_on_unit_sphere(N):
    shape = (N,3)
    points = tf.random.normal(shape)
    return tf.linalg.normalize(points, axis=-1)