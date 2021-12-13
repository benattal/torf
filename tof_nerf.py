import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import tensorflow as tf
import numpy as np
import imageio
import json
import random
import time

from scipy import ndimage, misc
from matplotlib import cm

from utils.nerf_utils import *
from utils.utils import *
from utils.camera_utils import *

from datasets.tof_dataset import TOFDataset, flip_coords, unflip_coords
from datasets.ios_dataset import IOSDataset
from datasets.real_dataset import RealDataset
from datasets.mitsuba_dataset import MitsubaDataset

from config import config_parser
from render import *
from losses import *

class NeRFTrainer(object):
    def __init__(self, args):
        ## Args
        self.args = args

        ## Logging
        self.setup_loggers()

        ## Dataset
        self.setup_dataset()

        ## Projection, ray generation functions
        self.setup_ray_generation()

        ## Models
        self.setup_models()

        ## Losses
        self.setup_losses()

        ## Training stages
        self.model_reset_done = not self.args.reset_static_model
        self.calibration_pretraining_done = not self.args.calibration_pretraining

    def setup_ray_generation(self):
        ## Projection, ray generation functions
        self.image_sizes = {}
        self.generate_rays = {}

        self.image_sizes['tof'] = (
            self.dataset['tof_images'].shape[2],
            self.dataset['tof_images'].shape[1]
        )

        self.image_sizes['color'] = (
            self.dataset['color_images'].shape[2],
            self.dataset['color_images'].shape[1]
        )

        self.generate_rays['tof_coords'] = create_ray_generation_coords_fn(
            self.dataset['tof_intrinsics'][0]
            )
        self.generate_rays['color_coords'] = create_ray_generation_coords_fn(
            self.dataset['color_intrinsics'][0]
            )

        self.generate_rays['tof'] = create_ray_generation_fn(
            self.image_sizes['tof'][1],
            self.image_sizes['tof'][0],
            self.dataset['tof_intrinsics'][0]
        )
        self.generate_rays['color'] = create_ray_generation_fn(
            self.image_sizes['color'][1],
            self.image_sizes['color'][0],
            self.dataset['color_intrinsics'][0]
        )

    def set_trainable_pose(self, key):
        self.poses[key] = tf.Variable(
            self.poses[key],
            dtype=tf.float32
        )
        self.grad_calib_vars.append(self.poses[key])
    
    def set_saveable_pose(self, key):
        self.calib_vars[key] = self.poses[key]
        
    def setup_calibration(self):
        self.calib_vars = {}
        self.grad_calib_vars = []

        ## Poses
        self.poses = {}

        if self.args.optimize_poses and self.args.use_relative_poses:
            trainable_pose_names = ['tof_poses']

            if not self.args.colocated_pose:
                trainable_pose_names += ['relative_pose']
        else:
            trainable_pose_names = ['tof_poses', 'color_poses']

        if 'relative_pose' not in self.dataset:
            self.dataset['relative_pose'] = np.eye(4)[None].astype(np.float32)

        for k in self.dataset:
            if 'pose' not in k:
                continue

            self.poses[k] = se3_vee(self.dataset[k])

            if self.args.optimize_poses and (k in trainable_pose_names):
                if self.args.identity_pose_initialization and not k == 'relative_pose':
                    self.poses[k] = np.zeros_like(
                        self.poses[k]
                    )

                if self.args.noisy_pose_initialization and not k == 'relative_pose':
                    self.poses[k] = add_pose_noise(
                        self.poses[k]
                    )

                self.set_trainable_pose(k)

            self.set_saveable_pose(k)

        ## Phase offset
        if 'phase_offset' in self.dataset:
            phase_offset = tf.convert_to_tensor(
                np.array(self.dataset['phase_offset']).astype(np.float32)
            )
        else:
            phase_offset = tf.convert_to_tensor(
                np.array(self.args.phase_offset).astype(np.float32)
            )

        if self.args.optimize_phase_offset:
            phase_offset = tf.Variable(phase_offset, dtype=tf.float32)

            self.calib_vars['phase_offset'] = phase_offset
            self.grad_vars.append(phase_offset)

        self.calib_vars['phase_offset'] = phase_offset

        ## Depth range
        if 'depth_range' in self.dataset:
            self.args.depth_range = self.dataset['depth_range']
            self.render_kwargs_train['depth_range'] = self.dataset['depth_range']
            self.render_kwargs_test['depth_range'] = self.dataset['depth_range']

    @property
    def relative_pose(self):
        return se3_hat(self.poses['relative_pose'])

    @property
    def tof_poses(self):
        return se3_hat(self.poses['tof_poses'])

    @property
    def tof_light_poses(self):
        return se3_hat(self.poses['tof_poses'])

    @property
    def color_poses(self):
        if self.args.optimize_poses and self.args.use_relative_poses:
            return self.tof_poses @ self.relative_pose
        else:
            return se3_hat(self.poses['color_poses'])

    @property
    def color_light_poses(self):
        return se3_hat(self.poses['color_light_poses'])

    def save_calibration(self, i):
        # Poses
        for k in self.calib_vars:
            calib_path = os.path.join(
                self.basedir, self.expname, '{}_{:06d}.npy'.format(k, i)
            )
            calib_path_full = os.path.join(
                self.basedir, self.expname, '{}_{:06d}_full.npy'.format(k, i)
            )
            
            if not isinstance(self.calib_vars[k], np.ndarray):
                var_to_save = self.calib_vars[k].numpy()
            else:
                var_to_save = self.calib_vars[k]

            np.save(calib_path, var_to_save)

            if 'pose' in calib_path and len(var_to_save.shape) == 2:
                full_var = unflip_coords(np.array(se3_hat(var_to_save)))
                np.save(calib_path_full, full_var)

        print('Saved calibration variables')

    def load_calibration(self, i):
        # Poses
        for k in self.calib_vars:
            calib_path = os.path.join(
                self.basedir, self.expname, '{}_{:06d}.npy'.format(k, i)
            )
            
            if os.path.exists(calib_path):
                temp_var = np.load(calib_path)
                
                if not isinstance(self.calib_vars[k], tf.Variable):
                    self.calib_vars[k] = temp_var
                else:
                    self.calib_vars[k].assign(temp_var)

        print('Loaded calibration variables')

    def save_codes(self, i):
        path = os.path.join(
            self.basedir, self.expname, 'codes_{:06d}.npy'.format(i)
        )
        np.save(path, self.temporal_codes.numpy())
        print('saved codes at', path)

    def save_weights(self, i):
        for k in self.models:
            path = os.path.join(
                self.basedir, self.expname, '{}_{:06d}.npy'.format(k, i)
            )

            np.save(path, self.models[k].get_weights())

        print('Saved weights')
    
    def save_all(self, i):
        self.save_calibration(i)
        self.save_weights(i)

        if self.args.dynamic:
            self.save_codes(i)

    def get_training_args(self, i, render_kwargs, args):
        render_kwargs['num_views'] = args.num_views
        render_kwargs['static_scene'] = i < args.static_scene_iters
        empty_weight = args.empty_weight

        if render_kwargs['static_scene'] and args.dynamic:
            render_kwargs['network_query_fn'] = (self.all_query_fns[1],)
            render_kwargs['dynamic'] = False
            render_kwargs['render_rays_fn'] = render_rays
            empty_weight = 0.0
        elif args.dynamic:
            render_kwargs['network_query_fn'] = self.all_query_fns
            render_kwargs['dynamic'] = True
            render_kwargs['render_rays_fn'] = render_rays_dynamic

        render_kwargs['use_phase_calib'] = i > args.no_phase_calib_iters and args.use_phase_calib

        render_kwargs['use_phasor'] = i > args.no_phase_iters and args.use_phasor

        render_kwargs['use_variance_weighting'] = \
            args.use_variance_weighting and (i > self.args.no_variance_iters)

        render_kwargs['use_tof_uncertainty'] = \
            args.use_tof_uncertainty and (i > args.i_aux)
        
        images_to_log = [
            'radiance', 'tof_cos', 'tof_amp', 'disp'
        ]

        # Calculate empty weight
        if args.empty_weight_decay < 1.0 and args.empty_weight_decay_steps != 0 and not render_kwargs['static_scene']:
            decay_exp = float(i) / ((args.tof_weight_decay_steps * 1000) - args.static_scene_iters)
            empty_weight = np.power(args.empty_weight_decay, decay_exp) * empty_weight
        
        # Calculate tof weight
        if args.tof_weight_decay < 1.0 and args.tof_weight_decay_steps != 0:
            decay_exp = np.minimum(i // (args.tof_weight_decay_steps * 1000), 1.0)
            tof_weight = np.power(args.tof_weight_decay, decay_exp) * args.tof_weight
        else:
            tof_weight = args.tof_weight

        # Calculate depth weight
        if args.depth_weight_decay < 1.0 and args.depth_weight_decay_steps != 0:
            decay_exp = np.minimum(i // (args.depth_weight_decay_steps * 1000))
            depth_weight = np.power(args.depth_weight_decay, decay_exp) * args.depth_weight
        else:
            depth_weight = args.depth_weight
        
        return tof_weight, args.radiance_weight, depth_weight, empty_weight, images_to_log

    def get_test_render_args(self, render_kwargs_train, i, args):
        render_kwargs_test = {
            k: render_kwargs_train[k] for k in render_kwargs_train
            }
        render_kwargs_test['perturb'] = False
        render_kwargs_test['raw_noise_std'] = 0.
        
        return render_kwargs_test
    
    def apply_gradients(self, loss, tape, filter_grad_vars=None):
        # Filter
        if filter_grad_vars is not None:
            grad_vars = [g for g in self.grad_vars if filter_grad_vars in g.name]
        else:
            grad_vars = self.grad_vars

        # Optimize poses and other vars separately
        if self.args.optimize_poses or self.args.optimize_phase_offset:
            gradients = tape.gradient(loss, grad_vars + self.grad_calib_vars)

            gradients_vars = gradients[:len(grad_vars)]
            self.optimizer.apply_gradients([
                (grad, var) for (grad, var) in zip(gradients_vars, grad_vars) if grad is not None
                ]
                )

            gradients_calib_vars = gradients[len(grad_vars):]
            self.calib_optimizer.apply_gradients([
                (grad, var) for (grad, var) in zip(gradients_calib_vars, self.grad_calib_vars) if grad is not None
                ]
                )
        # Optimize all vars together
        else:
            gradients = tape.gradient(loss, grad_vars)

            self.optimizer.apply_gradients([
                (grad, var) for (grad, var) in zip(gradients, grad_vars) if grad is not None
                ]
                )

    def get_ray_batch(self, coords, pose, light_pose, key):
        rays_o, rays_d = self.generate_rays[f'{key}_coords'](
            coords[..., 1], coords[..., 0], pose
        )
        light_pos = tf.cast(
            tf.broadcast_to(light_pose[..., :3, -1], rays_o.shape),
            tf.float32
            )
        batch_rays = tf.stack([rays_o, rays_d, light_pos], 0)

        return batch_rays

    def _train_step(
        self,
        train_iter,
        img_i,
        coords,
        batch_images,
        render_kwargs_train,
        outputs,
        losses,
        key,
        ):
        # Setup args
        render_kwargs_train = {
            k: render_kwargs_train[k] for k in render_kwargs_train
        }
        
        render_kwargs_train['outputs'] = outputs

        with tf.GradientTape() as tape:
            # Get poses as matrices
            pose = getattr(self, f'{key}_poses')[img_i]
            light_pose = getattr(self, f'{key}_light_poses')[img_i]

            # Get rays
            batch_rays = self.get_ray_batch(
                coords, pose, light_pose, key
            )

            # Make predictions for color, disparity, accumulated opacity.
            outputs = render(
                H=self.image_sizes[key][1], W=self.image_sizes[key][0],
                chunk=self.args.chunk, rays=batch_rays,
                image_index=img_i,
                **render_kwargs_train
            )

            # Image losses
            img_loss, img_loss0 = 0.0, 0.0
            psnr = 0.0
            psnr0 = 0.0

            for loss_key in losses:
                if self.loss_weights[loss_key] > 0:
                    _img_loss, _img_loss0 = self.img_loss_fns[loss_key](
                        batch_images[f'{loss_key}_images'], outputs, self.loss_weights[loss_key]
                    )
                    img_loss += _img_loss
                    img_loss0 += _img_loss0

                    if loss_key == 'color':
                        psnr = mse2psnr(img_loss / self.loss_weights[loss_key]).numpy()
                        psnr0 = mse2psnr(img_loss0 / self.loss_weights[loss_key]).numpy()
            
            # Regularization losses
            loss = 0.0

            for loss_key in self.reg_loss_fns:
                if self.loss_weights[loss_key] > 0:
                    reg_loss = self.reg_loss_fns[loss_key](outputs) * self.loss_weights[loss_key]
                    loss += reg_loss

            # Total
            loss += img_loss + img_loss0

        # Gradients
        self.apply_gradients(loss, tape)
        
        # Return
        return loss, psnr, psnr0
    
    def train_step(
        self,
        i,
        render_kwargs_train,
        args
        ):

        loss = 0.0

        if not args.train_both:
            # TOF
            if (i < args.no_radiance_iters or (i % 2) == 1):
                key = 'tof'
                batch_outputs = ['tof_images'] + (['depth_images'] if args.use_depth_loss else [])
                outputs = ['tof_map', 'acc_map']
                losses = ['tof']
            # Color
            else:
                key = 'color'
                batch_outputs = ['color_images'] + (['depth_images'] if args.use_depth_loss else [])
                outputs = ['radiance_map', 'acc_map']
                losses = ['color']
        else:
            # Both
            key = 'tof'
            batch_outputs = ['tof_images', 'color_images'] + (['depth_images'] if args.use_depth_loss else [])
            outputs = ['tof_map', 'radiance_map', 'acc_map'] + (['depth_map'] if args.use_depth_loss else [])
            losses = ['tof', 'color'] + (['depth'] if args.use_depth_loss else [])

        # Get batch
        img_i, coords, batch_images = self.dataloader.get_batch(
            self.dataloader.i_train,
            args.N_rand,
            self.image_sizes[key][1], self.image_sizes[key][0],
            args,
            outputs=batch_outputs
            )

        # Train step
        loss, psnr, psnr0 = self._train_step(
            i,
            img_i,
            coords,
            batch_images,
            render_kwargs_train,
            outputs,
            losses,
            key,
        )
        
        return loss, psnr, psnr0
    
    def setup_dataset(self):
        ## Load data
        if self.args.dataset_type == 'real':
            dataset = RealDataset(self.args)
        elif self.args.dataset_type == 'ios':
            dataset = IOSDataset(self.args)
        elif self.args.dataset_type == 'mitsuba':
            dataset = MitsubaDataset(self.args)
        else:
            dataset = TOFDataset(self.args)

        self.dataset = dataset.dataset
        self.dataloader = dataset

        print(
            'Loaded dataset',
            self.args.dataset_type,
            self.dataset['tof_images'].shape,
            self.dataset['tof_intrinsics'][0],
            self.dataset['color_intrinsics'][0],
            self.args.datadir
        )

        ## Bounds
        self.near = tf.reduce_min(self.dataset['bounds']) * 0.9
        self.far = tf.reduce_max(self.dataset['bounds']) * 1.1
        self.min_vis_depth = self.dataset['bounds'].min()

        print('NEAR FAR', self.near, self.far)

        ## Show images
        if self.args.show_images:
            for i in range(self.args.num_views):
                plt.imshow(self.dataset['color_images'][i])
                plt.show()

                plt.imshow(self.dataset['tof_images'][i][..., 0])
                plt.show()

                plt.imshow(self.dataset['tof_images'][i][..., 1])
                plt.show()

                plt.subplot(1, 2, 1)
                plt.imshow(self.dataset['tof_depth_images'][i], vmin=0, vmax=9)

                plt.subplot(1, 2, 2)
                plt.imshow(self.dataset['depth_images'][i], vmin=0, vmax=9)
                plt.show()
    
    def setup_loggers(self):
        self.basedir = self.args.basedir
        self.expname = self.args.expname

        # Logging directories
        os.makedirs(os.path.join(self.basedir, self.expname), exist_ok=True)
        f = os.path.join(self.basedir, self.expname, 'args.txt')

        with open(f, 'w') as file:
            for arg in sorted(vars(self.args)):
                attr = getattr(self.args, arg)
                file.write('{} = {}\n'.format(arg, attr))

        if self.args.config is not None:
            f = os.path.join(self.basedir, self.expname, 'config.txt')

            with open(f, 'w') as file:
                file.write(open(self.args.config, 'r').read())

        # Summary writer
        self.writer = tf.summary.create_file_writer(
            os.path.join(self.basedir, 'summaries', self.expname)
            )
        self.writer.set_as_default()

    def setup_models(self):
        ## Create model
        self.render_kwargs_train, self.render_kwargs_test, \
            self.start, self.grad_vars, self.models, self.temporal_codes = \
            create_nerf(self.args)
        self.all_query_fns = self.render_kwargs_train['network_query_fn']

        print(self.models)
        
        ## Create bounds
        bds_dict = {
            'near': tf.cast(self.near, tf.float32),
            'far': tf.cast(self.far, tf.float32)
        }

        self.render_kwargs_train.update(bds_dict)
        self.render_kwargs_test.update(bds_dict)

        ## Calibration variables
        self.setup_calibration()
        self.load_calibration(self.start - 1)

        ## Optimizers
        self.setup_optimizers()

        ## Step
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.global_step.assign(self.start)
    
    def setup_optimizers(self):
        ## Optimizer
        if self.args.lrate_decay > 0:
            lrate = tf.keras.optimizers.schedules.ExponentialDecay(
                self.args.lrate,
                decay_steps=self.args.lrate_decay * 1000,
                decay_rate=0.1
                )
        else:
            lrate = self.args.lrate

        self.optimizer = tf.keras.optimizers.Adam(lrate)

        ## Calib optimizer
        if self.args.lrate_decay_calib > 0:
            lrate_calib = tf.keras.optimizers.schedules.ExponentialDecay(
                self.args.lrate_calib,
                decay_steps=self.args.lrate_decay_calib * 1000,
                decay_rate=0.1
                )
        else:
            lrate_calib = self.args.lrate_calib

        self.calib_optimizer = tf.keras.optimizers.Adam(lrate_calib)

        ## Add to models
        self.models['optimizer'] = self.optimizer
        self.models['calib_optimizer'] = self.calib_optimizer
    
    def setup_losses(self):
        self.img_loss_fns = {
            'color': radiance_loss_default,
            'tof': tof_loss_default,
            'depth': depth_loss_default
        }
        self.reg_loss_fns = {
            'empty': empty_space_loss,
            'tof_poses': make_pose_loss(self, 'tof_poses'),
            'color_poses': make_pose_loss(self, 'color_poses'),
        }
        
    def update_args(self, i):
        # Get training params for current iter
        tof_weight, radiance_weight, depth_weight, empty_weight, self.images_to_log = \
            self.get_training_args(i, self.render_kwargs_train, self.args)

        self.loss_weights = {
            'color': radiance_weight,
            'tof': tof_weight,
            'depth': depth_weight,
            'empty': empty_weight,
            'tof_poses': self.args.pose_reg_weight,
            'color_poses': self.args.pose_reg_weight,
        }

        self.render_kwargs_train['phase_offset'] = self.calib_vars['phase_offset']
        self.render_kwargs_test = self.get_test_render_args(
            self.render_kwargs_train, i, self.args
            )

    def model_reset(self):
        self.model_reset_done = True

        for model_name in self.models:
            if 'optimizer' not in model_name:
                load_model(self.models, model_name, 0, self.args)

        if self.args.dynamic:
            self.temporal_codes.assign(load_codes(self.temporal_codes, 0, self.args))
        
        self.setup_optimizers()
    
    def finish_calibration_pretraining(self):
        self.calibration_pretraining_done = True

        # Reset optimization parameters
        self.args.optimize_poses = True

        ## Relative pose
        if self.args.use_relative_poses \
            and not self.args.colocated_pose \
                and not self.args.optimize_relative_pose:

            self.args.optimize_relative_pose = True
            self.set_trainable_pose('relative_pose')
            self.set_saveable_pose('relative_pose')

        ## Optimizers
        self.args.lrate_calib *= self.args.lrate_calib_fac
        self.setup_optimizers()
    
    def eval(self):
        i_val = self.dataloader.i_val
        split_pose = np.array(self.tof_poses[i_val])

        if self.args.extrinsics_file != "":
            temp_pose = np.load(self.args.extrinsics_file)
            split_pose = np.tile(np.eye(4)[None], (temp_pose.shape[0], 1, 1))
            split_pose[:, :3, :] = temp_pose[:, :3, :4]

            split_pose = flip_coords(np.linalg.inv(split_pose))
            split_pose, _ = recenter_poses(split_pose)

            if self.args.reverse_extrinsics:
                split_pose = split_pose[::-1]

            i_val = [0 for i in range(split_pose.shape[0])]

        split_colors = self.dataset['color_images'][i_val]
        split_tofs = self.dataset['tof_images'][i_val]
        split_depths = self.dataset['depth_images'][i_val]
        split_tof_depths = self.dataset['tof_depth_images'][i_val]
        split_frame_numbers = list(range(len(i_val)))

        # Get outputs
        for k, image_idx in enumerate(split_frame_numbers):
            pose = split_pose[k]
            print(k, pose, i_val)

            outputs = render(
                H=self.image_sizes['color'][1], W=self.image_sizes['color'][0],
                chunk=self.args.chunk,
                pose=pose, light_pose=pose,
                ray_gen_fn=self.generate_rays['color'],
                image_index=image_idx,
                **self.render_kwargs_test
            )

            start_idx = 0
            eval_dir = self.args.expname + '_eval'

            os.makedirs(eval_dir, exist_ok=True)
            os.makedirs(eval_dir + '/eval_depth', exist_ok=True)
            os.makedirs(eval_dir + '/eval_target_depth', exist_ok=True)
            os.makedirs(eval_dir + '/eval_depth_png', exist_ok=True)
            os.makedirs(eval_dir + '/eval_target_depth_png', exist_ok=True)
            os.makedirs(eval_dir + '/eval_png', exist_ok=True)
            os.makedirs(eval_dir + '/eval_target_png', exist_ok=True)

            np.save(eval_dir + '/eval_depth/%04d.npy' % (k + start_idx), outputs['depth_map'])
            np.save(eval_dir + '/eval_target_depth/%04d.npy' % (k + start_idx), split_depths[k])

            disps = outputs['depth_map']
            disps = 1 - (disps - self.near) / (self.far - self.near)
            disps = cm.magma(disps)
            imageio.imwrite(
                eval_dir + '/eval_depth_png/%04d.png' % (k + start_idx), to8b(disps)
            )
            disps = split_depths[k]
            disps = 1 - (disps - self.near) / (self.far - self.near)
            disps = cm.magma(disps)
            imageio.imwrite(
                eval_dir + '/eval_target_depth_png/%04d.png' % (k + start_idx), to8b(disps)
            )

            np.save(eval_dir + '/eval/%04d.npy' % (k + start_idx), outputs['radiance_map'])
            np.save(eval_dir + '/eval_target/%04d.npy' % (k + start_idx), split_colors[k])

            imageio.imwrite(
                eval_dir + '/eval_png/%04d.png' % (k + start_idx), to8b(outputs['radiance_map'])
            )
            imageio.imwrite(
                eval_dir + '/eval_target_png/%04d.png' % (k + start_idx), to8b(split_colors[k])
            )

            if self.args.show_images:
                plt.subplot(1, 2, 1)
                plt.imshow(split_colors[k])
                plt.subplot(1, 2, 2)
                plt.imshow(outputs['radiance_map'])
                plt.show()

                plt.imshow(outputs['radiance_map'])
                plt.show()

                plt.imshow(np.abs(outputs['radiance_map'] - split_colors[k]))
                plt.show()
    
    def video_logging(self, i, render_freezeframe=False):
        if self.args.extrinsics_file != "":
            temp_pose = np.load(self.args.extrinsics_file)
            split_pose = np.tile(np.eye(4)[None], (temp_pose.shape[0], 1, 1))
            split_pose[:, :3, :] = temp_pose[:, :3, :4]

            split_pose = flip_coords(np.linalg.inv(split_pose))
            split_pose[:, :3, -1] *= self.args.extrinsics_scale
            split_pose, _ = recenter_poses(split_pose)

            if self.args.reverse_extrinsics:
                split_pose = split_pose[::-1]

            render_poses = split_pose
            render_light_poses = np.copy(split_pose)

        elif self.args.render_test:
            render_poses = np.array(
                self.color_poses[self.dataloader.i_test]
                )
            render_light_poses = np.copy(render_poses)

        else:
            render_poses, render_light_poses = \
                get_render_poses_spiral(
                    self.args.focus_distance,
                    self.dataset['bounds'], self.dataset['tof_intrinsics'],
                    self.tof_poses, self.args, 60, 2
                )

        if self.args.dynamic:
            self.render_kwargs_test['outputs'] = [
                'tof_map', 'radiance_map',
                'tof_map_dynamic', 'radiance_map_dynamic',
                'disp_map', 'depth_map',
                'depth_map_dynamic', 'disp_map_dynamic',
            ]

        all_videos = render_path(
            self.dataloader.i_train,
            dataset=self.dataset, all_poses=self.tof_poses, 
            render_poses=render_poses, light_poses=render_light_poses,
            ray_gen_fn_camera=self.generate_rays['color'], ray_gen_fn_light=self.generate_rays['color'],
            H=self.image_sizes['color'][1], W=self.image_sizes['color'][0],
            near=self.near, far=self.far, chunk=self.args.chunk, render_kwargs=self.render_kwargs_test, args=self.args,
            render_freezeframe=render_freezeframe
            )

        # Write images
        image_base = 'spiral' if not render_freezeframe else 'freezeframe'
        depth_base = os.path.join(self.basedir, self.expname, image_base, 'depth_raw')
        rgb_base = os.path.join(self.basedir, self.expname, image_base, 'rgb_raw')

        os.makedirs(depth_base, exist_ok=True)
        os.makedirs(rgb_base, exist_ok=True)

        for j in range(all_videos['depth_map'].shape[0]):
            depth = all_videos['depth_map'][j]
            rgb = all_videos['radiance_map'][j]

            np.save(f'{depth_base}/{j:04d}.npy', depth)

            imageio.imwrite(
                f'{rgb_base}/{j:04d}.png', to8b(rgb)
            )

        # Write videos
        print('Done rendering videos, saving')

        moviebase = os.path.join(
            self.basedir, self.expname, f'{self.args.expname}_{image_base}_{i:08d}_'
        )
        
        if 'disp' in self.images_to_log:
            disps = all_videos['depth_map']
            disps = 1 - (disps - self.near) / (self.far - self.near)
            disps = cm.magma(disps)

            imageio.mimwrite(
                moviebase + 'disp.mp4', to8b(disps),
                fps=25, quality=8
            )

            if self.args.dynamic:
                disps = all_videos['depth_map_dynamic']
                disps = 1 - (disps - self.near) / (self.far - self.near)
                disps = cm.magma(disps)

                imageio.mimwrite(
                    moviebase + 'disp_dynamic.mp4', to8b(disps),
                    fps=25, quality=8
                )
        
        if 'tof_cos' in self.images_to_log:
            tofs = all_videos['tof_map']
            tofs_cos = normalize_im(np.abs(tofs[..., 0]))

            imageio.mimwrite(
                moviebase + 'tof_cos.mp4', to8b(tofs_cos),
                fps=25, quality=8
            )

        if 'tof_sin' in self.images_to_log:
            tofs = all_videos['tof_map']
            tofs_sin = normalize_im(np.abs(tofs[..., 1]))

            imageio.mimwrite(
                moviebase + 'tof_sin.mp4', to8b(tofs_sin),
                fps=25, quality=8
            )

        if 'tof_amp' in self.images_to_log:
            tofs = all_videos['tof_map']
            tofs_amp = normalize_im(np.abs(tofs[..., 1]))

            imageio.mimwrite(
                moviebase + 'tof_amp.mp4', to8b(tofs_amp),
                fps=25, quality=8
            )

            if self.args.dynamic:
                tofs = all_videos['tof_map_dynamic']
                tofs_amp = normalize_im(np.abs(tofs[..., 1]))

                imageio.mimwrite(
                    moviebase + 'tof_amp_dynamic.mp4', to8b(tofs_amp),
                    fps=25, quality=8
                )
        
        if 'radiance' in self.images_to_log:
            radiances = all_videos['radiance_map']
            imageio.mimwrite(
                moviebase + 'radiance.mp4', to8b(radiances),
                fps=25, quality=8
            )
            
            if self.args.dynamic:
                radiances = all_videos['radiance_map_dynamic']
                imageio.mimwrite(
                    moviebase + 'radiance_dynamic.mp4', to8b(radiances),
                    fps=25, quality=8
                )

    def image_logging(self, i):
        if (i % (2 * self.args.i_img) == 0):
            i_choice = self.dataloader.i_train
            suffix = "_train"
        else:
            i_choice = self.dataloader.i_test
            suffix = ""

        # Log a rendered validation view to Tensorboard
        img_i = np.random.choice(i_choice)
        target_image = self.dataset['color_images'][img_i]
        target_tof = self.dataset['tof_images'][img_i]
        target_depth = self.dataset['depth_images'][img_i]

        pose = self.tof_poses[img_i]
        light_pose = self.tof_light_poses[img_i]

        if self.args.dynamic:
            self.render_kwargs_test['outputs'] = [
                'tof_map', 'radiance_map',
                'depth_map', 'disp_map',
                'tof_map_dynamic', 'radiance_map_dynamic',
                'depth_map_dynamic', 'disp_map_dynamic',
            ]

        outputs = render(
            H=self.image_sizes['tof'][1], W=self.image_sizes['tof'][0],
            chunk=self.args.chunk,
            pose=pose, light_pose=light_pose,
            ray_gen_fn=self.generate_rays['tof'],
            image_index=img_i,
            **self.render_kwargs_test
        )

        # Save out the validation image for Tensorboard-free monitoring
        testimgdir = os.path.join(self.basedir, self.expname, 'tboard_val_imgs')
        os.makedirs(testimgdir, exist_ok=True)

        # Write images
        if 'disp' in self.images_to_log:
            disp = 1 - (outputs['depth_map'] - self.near) / (self.far - self.near)
            disp = cm.magma(disp)

            disp_gt = 1 - (target_depth - self.near) / (self.far - self.near)
            disp_gt = cm.magma(disp_gt)

            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_disp.png'.format(i)), to8b(disp)
            )
            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_disp_gt.png'.format(i)), to8b(disp_gt)
            )

            if 'disp_map_dynamic' in outputs:
                disp_dynamic = 1 - (outputs['depth_map_dynamic'] - self.near) / (self.far - self.near)
                disp_dynamic = cm.magma(disp_dynamic)

                imageio.imwrite(
                    os.path.join(testimgdir, '{:08d}_disp_dynamic.png'.format(i)), to8b(disp_dynamic)
                )

        if 'radiance' in self.images_to_log:
            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_color.png'.format(i)), to8b(outputs['radiance_map'])
            )
            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_color_gt.png'.format(i)), to8b(target_image)
            )

            if 'radiance_map_dynamic' in outputs:
                imageio.imwrite(
                    os.path.join(testimgdir, '{:08d}_color_dynamic.png'.format(i)), to8b(outputs['radiance_map_dynamic'])
                    )

        tof = outputs['tof_map']

        if 'tof_cos' in self.images_to_log:
            tof_cos = normalize_im_gt(np.abs(tof[..., 0]), np.abs(target_tof[..., 0]))
            tof_cos_gt = normalize_im(np.abs(target_tof[..., 0]))

            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_tof_cos.png'.format(i)), to8b(tof_cos)
            )
            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_tof_cos_gt.png'.format(i)), to8b(tof_cos_gt)
            )

        if 'tof_sin' in self.images_to_log:
            tof_sin = normalize_im_gt(np.abs(tof[..., 1]), np.abs(target_tof[..., 1]))
            tof_sin_gt = normalize_im(np.abs(target_tof[..., 1]))

            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_tof_sin.png'.format(i)), to8b(tof_sin)
            )
            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_tof_sin_gt.png'.format(i)), to8b(tof_sin_gt)
            )

        if 'tof_amp' in self.images_to_log:
            tof_amp = normalize_im_gt(np.abs(tof[..., 2]), np.abs(target_tof[..., 2]))
            tof_amp_gt = normalize_im(np.abs(target_tof[..., 2]))

            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_tof_amp.png'.format(i)), to8b(tof_amp)
            )
            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_tof_amp_gt.png'.format(i)), to8b(tof_amp_gt)
            )

            if 'tof_map_dynamic' in outputs:
                tof_dynamic = outputs['tof_map_dynamic']
                tof_amp_dynamic = normalize_im_gt(np.abs(tof_dynamic[..., 2]), np.abs(target_tof[..., 2]))
                imageio.imwrite(
                    os.path.join(testimgdir, '{:08d}_tof_amp_dynamic.png'.format(i)), to8b(tof_amp_dynamic)
                )

    def train(self):
        ## Train

        print('Begin training')
        print('TRAIN views are', self.dataloader.i_train)
        print('TEST views are', self.dataloader.i_test)
        print('VAL views are', self.dataloader.i_val)

        for i in range(self.start, self.args.N_iters + 1):
            # Update args
            self.update_args(i)

            # Model reset
            if i == self.args.model_reset_iters \
                and not self.model_reset_done:

                self.model_reset()
            
            # Calibration pretraining
            if i >= self.args.static_scene_iters \
                and not self.calibration_pretraining_done:

                self.finish_calibration_pretraining()

            # Evaluation
            if self.args.eval_only:
                tf.config.run_functions_eagerly(True)
                self.eval()
                tf.config.run_functions_eagerly(False)

                return

            # Rendering
            if self.args.render_only:
                tf.config.run_functions_eagerly(True)
                self.video_logging(i, False)
                self.video_logging(i, True)
                tf.config.run_functions_eagerly(False)

                return

            # Train step
            start_time = time.time()
            loss = 0.0

            loss, psnr, psnr0 = self.train_step(
                i,
                self.render_kwargs_train,
                self.args
                )

            time_elapsed = time.time() - start_time

            # Weights
            if i % self.args.i_save == 0:
                self.save_all(i)

            # Logging
            if i % self.args.i_print == 0 or i < 10:
                print("Phase offset:", self.calib_vars['phase_offset'])
                print("Poses:", self.color_poses[0], self.color_poses[10])

                if self.args.use_relative_poses:
                    print("Relative pose:", self.relative_pose)

                print(self.expname, i, psnr, loss.numpy(), self.global_step.numpy())
                print('iter time {:.05f}'.format(time_elapsed))

                tf.summary.scalar('loss', loss, step=i)
                tf.summary.scalar('psnr', psnr, step=i)

                if self.args.N_importance > 0:
                    tf.summary.scalar('psnr0', psnr0, step=i)

            # Video Logging
            if (i % self.args.i_video == 0 and i > 0):
                tf.config.run_functions_eagerly(True)
                self.video_logging(i, False)
                self.video_logging(i, True)
                tf.config.run_functions_eagerly(False)

            # Image Logging
            if (i % self.args.i_img == 0 and i > 0):
                tf.config.run_functions_eagerly(True)
                self.image_logging(i)
                tf.config.run_functions_eagerly(False)

            # Increment train step
            self.global_step.assign_add(1)

def main():
    ## Parse Args
    parser = config_parser()
    args = parser.parse_args()

    ## Random seed
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)
    
    ## Trainer
    trainer = NeRFTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()