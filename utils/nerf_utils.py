import os
import sys
import tensorflow as tf
import numpy as np
import imageio
import json

from render import *
from dynamic import *

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0

        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**tf.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return tf.identity, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tf.math.sin, tf.math.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

def init_render_model(
    D=2, W=16, input_ch=8, output_ch=3
    ):
    #relu = tf.keras.layers.ReLU()
    relu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def dense(W, act=relu, name=''):
        return tf.keras.layers.Dense(W, activation=act, name=name)

    print(
        'Render MODEL', input_ch,
        )

    inputs = tf.keras.Input(shape=(input_ch))
    outputs = inputs

    for i in range(D):
        outputs = dense(W)(outputs)

    outputs = dense(output_ch, act=None)(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def init_nerf_model(
    D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4],
    use_viewdirs=False, view_depth=1, view_width=None, prefix=''
    ):
    relu = tf.keras.layers.ReLU()
    ctr = {'val': 0}

    def dense(W, act=relu, name=''):
        ctr['val'] += 1
        return tf.keras.layers.Dense(W, activation=act, name=prefix + "dense%d" % ctr['val'])

    print(
        'MODEL', input_ch, input_ch_views,
        type(input_ch), type(input_ch_views), use_viewdirs
        )
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)

    if view_width is None:
        view_width = W // 2

    if use_viewdirs:
        inputs = tf.keras.Input(shape=(input_ch + input_ch_views))
        inputs_pts, inputs_views = tf.split(inputs, [input_ch, input_ch_views], -1)
        inputs_pts.set_shape([None, input_ch])
        inputs_views.set_shape([None, input_ch_views])

        print(inputs.shape, inputs_pts.shape, inputs_views.shape)
    else:
        inputs = tf.keras.Input(shape=(input_ch))
        inputs_pts = inputs
        inputs_pts.set_shape([None, input_ch])

        print(inputs.shape, inputs_pts.shape)

    outputs = inputs_pts

    for i in range(D):
        outputs = dense(W)(outputs)

        if i in skips:
            outputs = tf.concat([inputs_pts, outputs], -1)

    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)  # concat viewdirs
        outputs = inputs_viewdirs

        for i in range(view_depth):
            outputs = dense(view_width)(outputs)

        outputs = dense(output_ch - 1, act=None)(outputs)
        outputs = tf.concat([outputs, alpha_out], -1)
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def init_dynamic_nerf_model(
    D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4],
    use_viewdirs=False, view_depth=1, view_width=None, args=None, prefix=''
    ):
    relu = tf.keras.layers.ReLU()
    ctr = {'val': 0}

    def dense(W, act=relu, name=''):
        ctr['val'] += 1
        return tf.keras.layers.Dense(W, activation=act, name=prefix + "dense%d" % ctr['val'])

    print(
        'MODEL', input_ch, input_ch_views,
        type(input_ch), type(input_ch_views), use_viewdirs
        )
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)

    if view_width is None:
        view_width = W // 2

    if use_viewdirs:
        inputs = tf.keras.Input(shape=(input_ch + input_ch_views + args.latent_code_size))
        inputs_pts, inputs_views, inputs_code = tf.split(inputs, [input_ch, input_ch_views, args.latent_code_size], -1)
        inputs_pts.set_shape([None, input_ch])
        inputs_views.set_shape([None, input_ch_views])
        inputs_code.set_shape([None, args.latent_code_size])

        print(inputs.shape, inputs_pts.shape, inputs_views.shape)
    else:
        inputs = tf.keras.Input(shape=(input_ch + args.latent_code_size))
        inputs_pts, inputs_code = tf.split(inputs, [input_ch, args.latent_code_size], -1)
        inputs_pts.set_shape([None, input_ch])
        inputs_code.set_shape([None, args.latent_code_size])

        print(inputs.shape, inputs_pts.shape)
    
    outputs = tf.concat([inputs_pts, inputs_code], axis=-1)

    for i in range(D):
        outputs = dense(W)(outputs)

        if i in skips:
            outputs = tf.concat([inputs_pts, inputs_code, outputs], -1)

    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)  # concat viewdirs
        outputs = inputs_viewdirs

        for i in range(view_depth):
            outputs = dense(view_width)(outputs)

        outputs = dense(output_ch - 1, act=None)(outputs)
        outputs = tf.concat([outputs, alpha_out], -1)
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_appearance_net(
    models, input_ch, input_ch_views, output_ch, skips, args, use_viewdirs=False, prefix=''
    ):
    grad_vars = []

    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=use_viewdirs, prefix=prefix
        )

    grad_vars += model.trainable_variables
    models['%smodel' % prefix] = model

    if args.N_importance > 0:
        model_fine = init_nerf_model(
            D=args.netdepth, W=args.netwidth,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, use_viewdirs=use_viewdirs, prefix=prefix
            )

        grad_vars += model_fine.trainable_variables
        models['%smodel_fine' % prefix] = model_fine

    return grad_vars

def create_dynamic_appearance_net(
    models, input_ch, input_ch_views, output_ch, skips, args, use_viewdirs=False, prefix=''
    ):
    grad_vars = []

    model = init_dynamic_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=use_viewdirs, args=args, prefix=prefix
        )

    grad_vars += model.trainable_variables
    models['%smodel' % prefix] = model

    if args.N_importance > 0:
        model_fine = init_dynamic_nerf_model(
            D=args.netdepth, W=args.netwidth,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, use_viewdirs=use_viewdirs, args=args, prefix=prefix
            )

        grad_vars += model_fine.trainable_variables
        models['%smodel_fine' % prefix] = model_fine

    return grad_vars

def create_model(models, grad_vars, args):
    # Embed position
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, 3)
    embeddirs_fn, input_ch_views = get_embedder(
        args.multires_views, args.i_embed
        )

    # Coarse
    skips = [4]
    grad_vars += create_appearance_net(
        models, input_ch, input_ch_views, 6, skips, args, True
        )

    def network_query_fn(inputs, scale_fac, viewdirs, models, fine=False):
        suffix = '_fine' if fine else ''
        inputs = (inputs * scale_fac - 0.5) * 2.0

        raw_app = run_network_chunked(
            inputs, viewdirs, models['model%s' % suffix],
            embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn,
            netchunk=args.netchunk
        )

        return raw_app

    return models, grad_vars, (network_query_fn,)

def create_dynamic_model(models, grad_vars, args):
    # Embed
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, 3)
    embeddirs_fn, input_ch_views = get_embedder(
        args.multires_views, args.i_embed
        )
    embedpix_fn, input_ch_pix = get_embedder(
        args.multires_pix, args.i_embed, 2
        )

    # Coarse
    if args.static_blend_weight:
        skips = [4]
        grad_vars += create_appearance_net(
            models, input_ch, input_ch_views, 7, skips, args, True, prefix='static_'
            )
        skips = [4]
        grad_vars += create_dynamic_appearance_net(
            models, input_ch, input_ch_views, 6, skips, args, True, prefix='dynamic_'
            )
    else:
        skips = [4]
        grad_vars += create_appearance_net(
            models, input_ch, input_ch_views, 6, skips, args, True, prefix='static_'
            )
        skips = [4]
        grad_vars += create_dynamic_appearance_net(
            models, input_ch, input_ch_views, 7, skips, args, True, prefix='dynamic_'
            )

    def network_query_fn(
        inputs, scale_fac, viewdirs, latent_code,
        models, fine=False
        ):
        suffix = '_fine' if fine else ''
        inputs = (inputs * scale_fac - 0.5) * 2.0

        raw_static = run_network_chunked(
            inputs, viewdirs, models['static_model%s' % suffix],
            embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn,
            netchunk=args.netchunk
        )

        raw_dynamic = run_network_chunked(
            inputs, viewdirs, models['dynamic_model%s' % suffix],
            other_inps=latent_code,
            embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn,
            netchunk=args.netchunk
        )

        return tf.concat([raw_static, raw_dynamic], axis=-1)

    def network_query_fn_static(
        inputs, scale_fac, viewdirs,
        models, fine=False
        ):
        suffix = '_fine' if fine else ''
        inputs = (inputs * scale_fac - 0.5) * 2.0

        raw_static = run_network_chunked(
            inputs, viewdirs, models['static_model%s' % suffix],
            embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn,
            netchunk=args.netchunk
        )

        return raw_static

    def network_query_fn_dynamic(
        inputs, scale_fac, viewdirs, latent_code,
        models, fine=False
        ):
        suffix = '_fine' if fine else ''
        inputs = (inputs * scale_fac - 0.5) * 2.0

        raw_dynamic = run_network_chunked(
            inputs, viewdirs, models['dynamic_model%s' % suffix],
            other_inps=latent_code,
            embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn,
            netchunk=args.netchunk
        )

        return raw_dynamic

    tof_phase_model = init_render_model(
        D=args.phasenetdepth, W=args.phasenetwidth,
        input_ch=input_ch_pix + 1,
        output_ch=1
        )
    grad_vars += tof_phase_model.trainable_variables
    models['tof_phase_model'] = tof_phase_model

    def render_query_fn(tof, coords, models, args):
        out_shape = tof.shape

        # Coordiantes
        coords_flat = tf.reshape(coords, [-1, coords.shape[-1]])
        coords_flat = embedpix_fn(coords_flat)

        # ToF phase
        tof_flat = tf.reshape(tof, [-1, tof.shape[-1]])
        phase_flat = get_phase(tof_flat[..., 0:1], tof_flat[..., 1:2])

        # Run
        inputs_flat = tf.concat([coords_flat, phase_flat], axis=-1)
        phase_residual = run_network_chunked(
            inputs_flat, None, models['tof_phase_model'],
            embed_fn=None,
            embeddirs_fn=None,
            netchunk=args.netchunk
        )

        # Output phase
        tof_phase = phase_residual + phase_flat
        tof_amp = get_amplitude(tof_flat)
        tof_real, tof_imag = get_phasor(tof_phase, tof_amp)

        return tf.reshape(tf.concat([tof_real, tof_imag, tof_amp], axis=-1), out_shape)

    # Temporal latent codes
    temporal_codes = tf.random.normal(
        [args.num_frames, args.latent_code_size],
        mean=0.0,
        stddev=(0.01 / np.sqrt(args.latent_code_size))
        )
    temporal_codes = tf.Variable(temporal_codes)
    grad_vars += [temporal_codes]

    return models, grad_vars, (network_query_fn, network_query_fn_static, network_query_fn_dynamic), temporal_codes, tof_phase_model


def create_nerf(args):
    """Instantiate NeRF's MLP model."""
    ## All models
    models = {}
    grad_vars = []
    temporal_codes = None
    tof_phase_model = None

    if args.dynamic:
        models, grad_vars, network_query_fn, temporal_codes, tof_phase_model = create_dynamic_model(models, grad_vars, args)
        render_rays_fn = render_rays_dynamic
    else:
        models, grad_vars, network_query_fn = create_model(models, grad_vars, args)
        render_rays_fn = render_rays

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'N_shadow_samples': args.N_shadow_samples,
        'models': models,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'bias_range': args.bias_range,
        'depth_range': args.depth_range,
        'falloff_range': args.falloff_range,
        'scene_scale': args.scene_scale,
        'scene_scale_x': args.scene_scale_x,
        'scene_scale_y': args.scene_scale_y,
        'use_phasor': args.use_phasor,
        'dynamic': args.dynamic,
        'use_tof_uncertainty': args.use_tof_uncertainty,
        'use_depth_sampling': args.use_depth_sampling,
        'static_blend_weight': args.static_blend_weight,
        'depth_sampling_range': args.depth_sampling_range,
        'base_uncertainty': args.base_uncertainty,
        'use_variance_weighting': args.use_variance_weighting,
        'use_falloff': args.use_falloff,
        'tof_phase_model': tof_phase_model,
        'square_transmittance': args.square_transmittance,
        'render_rays_fn': render_rays_fn,
        'outputs': [
            'tof_map', 'color_map', 'disp_map', 'acc_map', 'z_std'
            ]
    }

    if args.dynamic:
        render_kwargs_train['temporal_codes'] = temporal_codes

    # NDC only good for LLFF-style forward facing data
    render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    # Checkpoint loading setup
    start = get_start_iter(args)

    print('Resetting step to', start)
    
    for model_name in models:
        print(model_name, start)
        load_model(models, model_name, start - 1, args)
    
    if args.dynamic:
        temporal_codes.assign(load_codes(temporal_codes, start - 1, args))

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models, temporal_codes

def get_start_iter(args):
    start = 0
    expdir = os.path.join(args.basedir, args.expname)

    opt_ckpts = [
        os.path.join(expdir, f) \
            for f in sorted(os.listdir(expdir)) if 'optimizer' in f
            ]
    
    for ckpt in opt_ckpts:
        start = int(ckpt[-10:-4]) + 1
    
    return start

def load_poses(start, args):
    expdir = os.path.join(args.basedir, args.expname)
    poses_ckpt = os.path.join(expdir, 'poses_{:06d}.npy'.format(start))
    light_poses_ckpt = os.path.join(expdir, 'light_poses_{:06d}.npy'.format(start))
    phase_offset_ckpt = os.path.join(expdir, 'phase_offset_{:06d}.npy'.format(start))

    exists = os.path.exists(poses_ckpt) \
        and os.path.exists(light_poses_ckpt) \
            and os.path.exists(phase_offset_ckpt)

    return exists, \
        np.load(poses_ckpt) if exists else None, \
            np.load(light_poses_ckpt) if exists else None, \
                np.load(phase_offset_ckpt) if exists else None

def load_codes(codes, start, args):
    expdir = os.path.join(args.basedir, args.expname)
    ckpt = 'codes_{:06d}.npy'.format(start)
    ckpt = os.path.join(expdir, ckpt)

    if os.path.exists(ckpt):
        print('Reloading from', ckpt)
        return np.load(ckpt)
    else:
        return codes

def load_model(models, model_name, start, args):
    expdir = os.path.join(args.basedir, args.expname)
    ckpt = '{}_{:06d}.npy'.format(model_name, start)
    ckpt = os.path.join(expdir, ckpt)

    if os.path.exists(ckpt):
        print('Reloading from', ckpt)

        models[model_name].set_weights(
            np.load(ckpt, allow_pickle=True)
            )

# Efficient network evaluation
def batchify(fn, chunk):
    if chunk is None:
        return fn

    def ret(inputs, other_inps=None):
        outputs = []

        for i in range(0, inputs.shape[0], chunk):
            cur_inputs = inputs[i:i+chunk]

            # Get / brodcast additional inputs
            if other_inps is not None:
                if other_inps.shape[0] > 1:
                    cur_other_inps = other_inps[i:i+chunk]
                else:
                    cur_other_inps = tf.tile(other_inps, [cur_inputs.shape[0], 1])

                cur_inputs = tf.concat([cur_inputs, cur_other_inps], axis=-1)

            # Run forward
            outputs.append(fn(cur_inputs))
        
        return tf.concat(outputs, axis=0)

    return ret

def batchify_derivatives(fn, scale_fac, embed_fn, chunk):
    if chunk is None:
        return fn

    def ret(inputs, other_inps=None):
        gradients = []

        for i in range(0, inputs.shape[0], chunk):
            cur_inputs = inputs[i:i+chunk]

            # Get / broadcast additional inputs
            if other_inps is not None:
                if len(other_inps.shape) == 1:
                    other_inps = other_inps[None]

                if other_inps.shape[0] > 1:
                    cur_other_inps = other_inps[i:i+chunk]
                else:
                    cur_other_inps = tf.tile(other_inps, [cur_inputs.shape[0], 1])

            with tf.GradientTape() as tape:
                tape.watch(cur_inputs)

                net_inputs = tf.concat(
                    [embed_fn(cur_inputs * scale_fac), cur_other_inps], axis=-1
                    )
                output = fn(net_inputs)[..., -1:]

            gradients.append(tape.gradient(output, cur_inputs))
        
        return tf.concat(gradients, axis=0)

    return ret

def run_network_chunked(inputs, viewdirs, fn, embed_fn, embeddirs_fn, other_inps=None, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])

    if embed_fn is not None:
        embedded = embed_fn(inputs_flat)
    else:
        embedded = inputs_flat

    if viewdirs is not None:
        if len(viewdirs.shape) < len(inputs.shape):
            input_dirs = tf.broadcast_to(viewdirs[:, None], inputs.shape[:-1] + [viewdirs.shape[-1]])
        else:
            input_dirs = viewdirs

        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = tf.concat([embedded, embedded_dirs], -1)
    
    if other_inps is not None:
        other_inps = tf.reshape(other_inps, [-1, other_inps.shape[-1]])

    outputs_flat = batchify(fn, netchunk)(embedded, other_inps)
    outputs = tf.reshape(
        outputs_flat,
        list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
        )

    return outputs

def run_network_chunked_derivatives(inputs, scale_fac, fn, embed_fn, other_inps=None, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])

    outputs_flat = batchify_derivatives(fn, scale_fac, embed_fn, netchunk)(inputs_flat, other_inps)
    outputs = tf.reshape(
        outputs_flat,
        list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
        )

    return outputs