import numpy as np
import tensorflow as tf

def temporal_input_basic(image_index, args):
    if args.static_scene:
        image_index = tf.random.uniform((), minval=0, maxval=args.num_views - 1, dtype=tf.int32)

    return (float(image_index) / float(args.num_views)) * 2.0 - 1.0

def temporal_input(image_index, args):
    if args.static_scene:
        image_index = tf.random.uniform((), minval=0, maxval=args.num_views - 1, dtype=tf.int32)

    image_index = tf.cast(image_index, tf.float32)
    low = tf.clip_by_value(tf.math.floor(image_index), 0, args.temporal_codes.shape[0] - 1)

    temporal_codes_low = tf.nn.embedding_lookup(
        args.temporal_codes, tf.cast(low, tf.int32)
        )
    temporal_codes_low = tf.math.l2_normalize(
        temporal_codes_low, axis=-1, epsilon=1e-7
        )

    return temporal_codes_low

def temporal_input_pointwise(pts, image_index, args):
    image_index = tf.cast(image_index, tf.float32)
    image_index = image_index * tf.ones_like(pts[..., 0])

    if hasattr(args, 'find_visible') and args.find_visible:
        visible_frames = []
        found_visible = tf.zeros_like(image_index, dtype=tf.bool)

        for i in range(int(np.round(args.image_index)), 0, -1):
            pose = args.all_poses[i]
            pixels = args.project_fn(pts, pose)
            x, y = pixels[..., 0], pixels[..., 1]
            visible_mask = tf.logical_and(x >= 0, x < args.image_width)
            visible_mask &= tf.logical_and(y >= 0, y < args.image_height)
            
            image_index = tf.where(
                ~visible_mask & ~found_visible,
                tf.ones_like(image_index) * (i - 1),
                image_index
                )
            found_visible = found_visible | visible_mask

    low = tf.clip_by_value(tf.math.floor(image_index), 0, args.temporal_codes.shape[0] - 1)

    temporal_codes_low = tf.nn.embedding_lookup(
        args.temporal_codes, tf.cast(low, tf.int32)
        )
    temporal_codes_low = tf.math.l2_normalize(
        temporal_codes_low, axis=-1, epsilon=1e-7
        )

    return temporal_codes_low

def temporal_input_interp(image_index, args):
    if args.static_scene:
        image_index = tf.random.uniform((), minval=0, maxval=args.num_views - 1, dtype=tf.int32)

    image_index = tf.cast(image_index, tf.float32)
    low = tf.clip_by_value(tf.math.floor(image_index), 0, args.temporal_codes.shape[0] - 1)
    high = tf.clip_by_value(low + 1, 0, args.temporal_codes.shape[0] - 1)
    frac = image_index - low

    temporal_codes_low = tf.nn.embedding_lookup(
        args.temporal_codes, tf.cast(low, tf.int32)
        )
    temporal_codes_low = tf.math.l2_normalize(
        temporal_codes_low, axis=-1, epsilon=1e-7
        )

    temporal_codes_high = tf.nn.embedding_lookup(
        args.temporal_codes, tf.cast(high, tf.int32)
        )
    temporal_codes_high = tf.math.l2_normalize(
        temporal_codes_high, axis=-1, epsilon=1e-7
        )
    
    temporal_codes_interp = \
        (1 - frac) * temporal_codes_low + frac * temporal_codes_high
    temporal_codes_interp = tf.math.l2_normalize(
        temporal_codes_interp, axis=-1, epsilon=1e-7
        )
    
    return temporal_codes_interp

def temporal_input_pointwise_interp(pts, image_index, args):
    image_index = tf.cast(image_index, tf.float32)
    image_index = image_index * tf.ones_like(pts[..., 0])

    if hasattr(args, 'find_visible') and args.find_visible:
        visible_frames = []
        found_visible = tf.zeros_like(image_index, dtype=tf.bool)

        for i in range(int(np.round(args.image_index)), 0, -1):
            pose = args.all_poses[i]
            pixels = args.project_fn(pts, pose)
            x, y = pixels[..., 0], pixels[..., 1]
            visible_mask = tf.logical_and(x >= 0, x < args.image_width)
            visible_mask &= tf.logical_and(y >= 0, y < args.image_height)
            
            image_index = tf.where(
                ~visible_mask & ~found_visible,
                tf.ones_like(image_index) * (i - 1),
                image_index
                )
            found_visible = found_visible | visible_mask

    low = tf.clip_by_value(tf.math.floor(image_index), 0, args.temporal_codes.shape[0] - 1)
    high = tf.clip_by_value(low + 1, 0, args.temporal_codes.shape[0] - 1)
    frac = (image_index - low)[..., None]

    temporal_codes_low = tf.nn.embedding_lookup(
        args.temporal_codes, tf.cast(low, tf.int32)
        )
    temporal_codes_low = tf.math.l2_normalize(
        temporal_codes_low, axis=-1, epsilon=1e-7
        )

    temporal_codes_high = tf.nn.embedding_lookup(
        args.temporal_codes, tf.cast(high, tf.int32)
        )
    temporal_codes_high = tf.math.l2_normalize(
        temporal_codes_high, axis=-1, epsilon=1e-7
        )
    
    temporal_codes_interp = \
        (1 - frac) * temporal_codes_low + frac * temporal_codes_high
    temporal_codes_interp = tf.math.l2_normalize(
        temporal_codes_interp, axis=-1, epsilon=1e-7
        )
    
    return temporal_codes_interp