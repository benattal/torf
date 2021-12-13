import numpy as np
import tensorflow as tf

def project_points(points, P):
    projected_points = transform_points(points, P)

    projected_pixels =  projected_points[..., :2] \
        / projected_points[..., 2:3]
    
    projected_pixels = tf.where(
        tf.math.is_nan(projected_pixels),
        tf.zeros_like(projected_pixels),
        projected_pixels
        )
    
    return tf.reshape(projected_pixels, points.shape[:-1] + (2,))

# Ray generation and projection funcitons
def create_ray_generation_coords_fn(intrinsics):
    def fn(x, y, pose):
        return get_rays_matrix_coords(
            tf.cast(x, tf.float32), tf.cast(y, tf.float32),
            pose @ np.linalg.inv(intrinsics)
            )
    
    return fn

def create_ray_generation_fn(H, W, intrinsics):
    def fn(pose):
        return get_rays_matrix(H, W, pose @ np.linalg.inv(intrinsics))
    
    return fn

def create_projection_fn(intrinsics):
    def fn(points, pose):
        return project_points(points, intrinsics @ tf.linalg.inv(pose))
    
    return fn

# Twists
def add_pose_noise(poses, t_scale=0.1):
    noise_poses = np.random.normal(0.0, 0.15, poses.shape)
    noise_poses[..., 3:] *= 0.1

    return poses + noise_poses

def se3_vee(mat):
    mat = tf.linalg.logm(tf.cast(mat, tf.complex64))
    twist = tf.stack(
        [
            mat[..., 2, 1],
            mat[..., 0, 2],
            mat[..., 1, 0],
            mat[..., 0, 3],
            mat[..., 1, 3],
            mat[..., 2, 3],
        ],
        axis=-1
        )
    
    return tf.cast(twist, tf.float32)

def se3_hat(twist):
    twist = tf.cast(twist, tf.complex64)
    null = tf.zeros_like(twist[..., 0])

    mat = tf.stack(
        [
            tf.stack(
                [
                null,
                twist[..., 2],
                -twist[..., 1],
                null
                ],
                axis=-1
            ),
            tf.stack(
                [
                -twist[..., 2],
                null,
                twist[..., 0],
                null
                ],
                axis=-1
            ),
            tf.stack(
                [
                twist[..., 1],
                -twist[..., 0],
                null,
                null
                ],
                axis=-1
            ),
            tf.stack(
                [
                twist[..., 3],
                twist[..., 4],
                twist[..., 5],
                null
                ],
                axis=-1
            ),
        ],
        axis=-1
        )
    
    return tf.cast(tf.linalg.expm(mat), tf.float32)

def transform_vectors(vectors, M):
    vectors = tf.reshape(vectors, (-1, 3))
    return tf.transpose(M[..., :3, :3] @ tf.transpose(vectors, [1, 0]), [1, 0])

def transform_vectors_np(vectors, M):
    vectors = np.reshape(vectors, (-1, 3))
    return np.transpose(M[..., :3, :3] @ np.transpose(vectors, [1, 0]), [1, 0])

def transform_points(points, M):
    in_shape = points.shape
    points = tf.reshape(points, (-1, 3))

    points = tf.concat(
        [points, tf.ones((points.shape[0], 1), dtype=points.dtype)], axis=-1
        )
    points = tf.transpose(M @ tf.transpose(points, [1, 0]), [1, 0])

    return tf.reshape(points[..., :3], in_shape[:-1] + (3,))

def transform_points_np(points, M):
    points = np.reshape(points, (-1, 3))

    points = np.concatenate(
        [points, np.ones((points.shape[0], 1))], axis=-1
        )
    points = np.transpose(M @ np.transpose(points, [1, 0]), [1, 0])

    return tf.reshape(points[..., :3], in_shape[:-1] + (3,))

def normalize_vector_np(v):
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    v[np.isnan(v)] = 0.

    return v

def normalize(v):
    return tf.math.l2_normalize(v, axis=-1, epsilon=1e-6)

def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d

def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = tf.stack([o0, o1, o2], -1)
    rays_d = tf.stack([d0, d1, d2], -1)

    return rays_o, rays_d

def get_coords(H, W):
    x, y = tf.meshgrid(
            tf.linspace(0., W - 1., W),
            tf.linspace(0., H - 1., H),
            )
    
    return tf.concat([x, y], axis=-1)

def get_rays_matrix_coords(
    x, y,
    P_inv
    ):
    z = tf.ones_like(x)
    points = tf.stack([x, y, z], axis=-1)
    rays_d = transform_vectors(points, P_inv)
    rays_o = tf.broadcast_to(P_inv[:3, -1], rays_d.shape)

    return rays_o, rays_d

def get_rays_matrix(
        height,
        width,
        P_inv
        ):
    x, y = tf.meshgrid(
            tf.linspace(0., width - 1., width),
            tf.linspace(0., height - 1., height),
            )
    z = tf.ones_like(x)

    points = tf.stack([x, y, z], axis=-1)
    rays_d = transform_vectors(points, P_inv)
    rays_d = tf.reshape(rays_d, [height, width, 3])
    rays_o = tf.broadcast_to(P_inv[:3, -1], rays_d.shape)

    return rays_o, rays_d

def get_rays_matrix_np(
        height,
        width,
        P_inv
        ):
    x, y = np.meshgrid(
            np.linspace(0., width - 1., width),
            np.linspace(0., height - 1., height),
            )
    z = np.ones_like(x)

    points = np.stack([x, y, z], axis=-1)
    rays_d = transform_vectors_np(points, P_inv)
    rays_d = np.reshape(rays_d, [height, width, 3])
    rays_o = np.broadcast_to(P_inv[:3, -1], rays_d.shape)

    return rays_o, rays_d

def get_dists(z_vals, rays_d, light_pos, pts):
    # Distances between samples
    dists = (z_vals[..., 1:] - z_vals[..., :-1])
    dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)
    dists = tf.concat(
        [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
        axis=-1
        )

    # Distances from camera, light
    dists_to_cam = z_vals * tf.linalg.norm(rays_d[..., None, :], axis=-1)
    #dists_to_light = tf.linalg.norm(pts - light_pos[..., None, :], axis=-1)
    dists_to_light = dists_to_cam
    dists_total = (dists_to_cam + dists_to_light)[..., None]

    return dists, dists_to_cam, dists_to_light, dists_total

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    hwf = c2w[:,4:5]
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.sin(-theta), np.cos(-theta), np.sin(-theta * zrate), 1.]) * rads)
        z = normalize(-c + np.dot(c2w[:3,:4], np.array([0, 0, focal, 1.])))
        pose = np.eye(4)
        pose[:3, :4] = viewmatrix(z, up, c)
        render_poses.append(pose)

    return render_poses

def get_render_poses_spiral(focal_length, bounds_data, intrinsics, poses, args, N_views=60, N_rots=2):
    intrinsics = np.array(intrinsics)
    poses = np.array(poses)

    ## Focus distance
    if focal_length < 0:
        close_depth, inf_depth = bounds_data.min() * .9, bounds_data.max() * 5.
        dt = .75
        mean_dz = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)
        focal_length = mean_dz

    # Get average pose
    c2w = poses_avg(poses)
    c2w_path = c2w
    up = normalize(poses[:, :3, 1].sum(0))

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = bounds_data.min() * .2
    tt = (poses[:, :3, 3] - c2w[:3, 3])

    rads = np.percentile(np.abs(tt), 90, 0) \
        * np.array([args.rad_multiplier_x, args.rad_multiplier_y, args.rad_multiplier_z])
    light_rads = np.percentile(np.abs(tt), 90, 0) \
        * np.array([args.rad_multiplier_x, args.rad_multiplier_y, args.rad_multiplier_z])

    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal_length, zdelta, zrate=.5, rots=N_rots, N=N_views)
    render_poses = np.array(render_poses).astype(np.float32)

    render_light_poses = render_path_spiral(c2w_path, up, light_rads, focal_length, zdelta, zrate=.5, rots=N_rots, N=N_views)
    render_light_poses = np.array(render_light_poses).astype(np.float32)

    return render_poses, render_light_poses