import configargparse

def config_parser():

    ## Base experiment config
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='/data/unrolled/datasets/hypersim', help='input data directory')
    parser.add_argument("--dataset_type", type=str, default='hypersim',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--scan", type=str, default='ai_001_001',
                        help='num views to use')

    ## Auxiliary
    parser.add_argument("--show_images", action='store_true',
                        help='Show images before training')
    parser.add_argument("--depth_splat", action='store_true',
                        help='Depth splat visualization before training')
    parser.add_argument("--eval_only", action='store_true',
                        help='Only eval')
    parser.add_argument("--render_freezeframe", action='store_true',
                        help='Render freezeframe outputs')
    parser.add_argument("--render_only", action='store_true',
                        help='Only perform render')
    parser.add_argument("--focus_distance", type=float, default=-1.0,
                        help='Focus distance for rendering')
    parser.add_argument("--rad_multiplier_x", type=float, default=1.0,
                        help='Radius multiplier for spiral')
    parser.add_argument("--rad_multiplier_y", type=float, default=1.0,
                        help='Radius multiplier for spiral')
    parser.add_argument("--rad_multiplier_z", type=float, default=1.0,
                        help='Radius multiplier for spiral')
    parser.add_argument("--tof_image_width", type=int, default=512,
                        help='Image width')
    parser.add_argument("--tof_image_height", type=int, default=512,
                        help='Image height')
    parser.add_argument("--color_image_width", type=int, default=512,
                        help='Image width')
    parser.add_argument("--color_image_height", type=int, default=512,
                        help='Image height')

    ## Training options

    parser.add_argument("--N_iters", type=int, default=100000,
                        help='Number of optimization iters')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_calib", type=float,
                        default=1e-3, help='learning rate')
    parser.add_argument("--lrate_calib_fac", type=float,
                        default=0.5, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--lrate_decay_calib", type=int, default=150,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')
    parser.add_argument("--num_views", type=int, default=16,
                        help='Num views to use')
    parser.add_argument("--total_num_views", type=int, default=200,
                        help='Total number of views in dataset')
    parser.add_argument("--view_step", type=int, default=1,
                        help='Training view step')
    parser.add_argument("--view_start", type=int, default=0,
                        help='Training view start')
    parser.add_argument("--val_start", type=int, default=61,
                        help='Validation view index start')
    parser.add_argument("--val_end", type=int, default=122,
                        help='Validation view index end')

    parser.add_argument("--optimize_poses", action='store_true',
                        help='Optimize poses')
    parser.add_argument("--optimize_relative_pose", action='store_true',
                        help='Optimize relative pose')
    parser.add_argument("--optimize_phase_offset", action='store_true',
                        help='Optimize phase offset')
    parser.add_argument("--noisy_pose_initialization", action='store_true',
                        help='Noisy initialization for poses')
    parser.add_argument("--identity_pose_initialization", action='store_true',
                        help='Identity initialization for poses')
    parser.add_argument("--use_relative_poses", action='store_true',
                        help='Use relative poses')
    parser.add_argument("--collocated_pose", action='store_true',
                        help='Colocated source and camera')

    parser.add_argument("--use_depth_loss", action='store_true',
                        help='Use direct depth loss')
    parser.add_argument("--depth_weight", type=float, default=0.0,
                        help='Direct depth loss weight')
    parser.add_argument("--depth_weight_decay", type=float, default=1.0,
                        help='Decay for depth weight')
    parser.add_argument("--depth_weight_decay_steps",   type=int, default=30,
                        help='Apply depth weight decay after this many steps')

    parser.add_argument("--sparsity_weight", type=float, default=0.0,
                        help='Weight for loss encouraging sparsity')
    parser.add_argument("--sparsity_weight_decay", type=float, default=1.0,
                        help='Decay for sparsity weight')
    parser.add_argument("--sparsity_weight_decay_steps",   type=int, default=30,
                        help='Apply sparsity weight decay after this many steps')

    parser.add_argument("--train_both", action='store_true',
                        help='Train color and ToF separately')

    parser.add_argument("--no_phase_calib_iters", type=int, default=5000,
                        help='Iters before adding phase look u p table optimization')
    parser.add_argument("--no_phase_iters", type=int, default=1000000,
                        help='Iters before adding phase bias prediction')
    parser.add_argument("--no_color_iters", type=int, default=0,
                        help='Iters before color loss optimization')

    parser.add_argument("--calibration_pretraining", action='store_true',
                        help='Use calibration (static scene) pretraining')

    parser.add_argument("--reset_static_model", action='store_true',
                        help='Reset static model after static pre-training')

    parser.add_argument("--color_weight", type=float, default=1.0,
                        help='Color loss weight')
    parser.add_argument("--pose_reg_weight", type=float, default=0.0,
                        help='Pose regularization loss weight')
    parser.add_argument("--tof_weight", type=float, default=1.0,
                        help='ToF loss weight')

    parser.add_argument("--tof_weight_decay", type=float, default=1.0,
                        help='ToF weight decay multiplier')
    parser.add_argument("--tof_weight_decay_steps",   type=int, default=30,
                        help='How many steps before ToF weight decay')
    
    ## Network architecture
    
    parser.add_argument("--dynamic", action='store_true',
                        help='Optimize a dynamic scene representation')
    parser.add_argument("--single_dynamic_model", action='store_true',
                        help='Single model for both static and dynamic content')
    parser.add_argument("--fix_view", action='store_true',
                        help='Assume a fixed view')
    parser.add_argument("--latent_code_size", type=int, default=256,
                        help='size of render feature vector')
    parser.add_argument("--num_frames", type=int, default=8,
                        help='Number of frames in sequence')
    parser.add_argument("--model_reset_iters", type=int, default=0,
                        help='How many iters before model reset')
    parser.add_argument("--static_scene_iters", type=int, default=0,
                        help='How many iters to optimize static scene')
    parser.add_argument("--use_static_loss", action='store_true',
                        help='Static scene loss')
    parser.add_argument("--find_visible", action='store_true',
                        help='Do not render content outside of view for a given time step (expensive)')

    parser.add_argument("--tofnetdepth", type=int, default=4,
                        help='layers in network')
    parser.add_argument("--tofnetwidth", type=int, default=64,
                        help='channels per layer')
    parser.add_argument("--tofnetdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--tofnetwidth_fine", type=int, default=64,
                        help='channels per layer in fine network')

    parser.add_argument("--phasenetdepth", type=int, default=4,
                        help='Net depth for look up table model')
    parser.add_argument("--phasenetwidth", type=int, default=128,
                        help='Net width for look up table model')

    parser.add_argument("--colornetdepth", type=int, default=4,
                        help='layers in network')
    parser.add_argument("--colornetwidth", type=int, default=64,
                        help='channels per layer')
    parser.add_argument("--colornetdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--colornetwidth_fine", type=int, default=64,
                        help='channels per layer in fine network')

    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')

    ## Rendering options

    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_shadow_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_depth_samples", type=int, default=16,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--multires_pix", type=int, default=2,
                        help='log2 of max freq for positional encoding (2D direction) for phase look up table model')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    
    # Importance sampling
    parser.add_argument("--use_importance_sampling", action='store_true',
                        help='Use error importance sampling for training')
    parser.add_argument("--importance_sampling_start",     type=int, default=20000,
                        help='When to start importance sampling')
    parser.add_argument("--importance_sampling_interval",     type=int, default=10000,
                        help='')
    parser.add_argument("--importance_sampling_freq",     type=int, default=2,
                        help='How often to importance sample')
    
    # Depth sampling
    parser.add_argument("--use_depth_sampling", action='store_true',
                        help='Sample about ground truth depth')
    parser.add_argument("--depth_sampling_exclude_interval", type=int, default=1,
                        help='No depth sampling every X iterations')
    parser.add_argument("--depth_sampling_range", type=float, default=0.1,
                        help='depth range for tof camera')
    parser.add_argument("--base_uncertainty", type=float, default=1e-5,
                        help='Base uncertainty for sampling')
    parser.add_argument("--depth_sampling_range_min", type=float,
                        default=0.1, help='minimum depth sampling range')

    # ToF rendering
    parser.add_argument("--square_transmittance", action='store_true',
                        help='Square transmittance for forward model')
    parser.add_argument("--use_falloff", action='store_true',
                        help='Use r^2 falloff')
    parser.add_argument("--use_phasor", action='store_true',
                        help='Phasor output for ToF model')
    parser.add_argument("--use_phase_calib", action='store_true',
                        help='Use phase look up table MLP')
    parser.add_argument("--use_tof_uncertainty", action='store_true',
                        help='Use tof albedo for coarse sampling')
    parser.add_argument("--use_variance_weighting", action='store_true',
                        help='Use variance weighted loss')
    parser.add_argument("--bias_range", type=float, default=50.,
                        help='When the model regresses phase bias, this sets the bias range')
    parser.add_argument("--phase_offset", type=float, default=0.,
                        help='Base phase offset for ToF phase')
    parser.add_argument("--depth_range", type=float, default=-1.,
                        help='depth range for tof camera')
    parser.add_argument("--min_depth_fac", type=float, default=0.01,
                        help='Used to calculate sampling range for ToF datasets')
    parser.add_argument("--max_depth_fac", type=float, default=1.0,
                        help='Used to calculate sampling range for ToF datasets')
    parser.add_argument("--scene_scale", type=float, default=1.,
                        help='Scene scale')
    parser.add_argument("--scene_scale_x", type=float, default=2.,
                        help='Scene scale multiplier x direction')
    parser.add_argument("--scene_scale_y", type=float, default=2.,
                        help='Scene scale multiplier y direction')
    parser.add_argument("--falloff_range", type=float, default=16.,
                        help='Falloff range')

    ## Dataset options

    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--tof_scale_factor", type=float, default=1.0,
                        help='downsample factor for images')
    parser.add_argument("--color_scale_factor", type=float, default=1.0,
                        help='downsample factor for images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--autoholdout", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--train_views", type=str, default="",
                        help='Views to use for training')
    parser.add_argument("--render_extrinsics_file", type=str, default="",
                        help='File to load render extrinsics from')
    parser.add_argument("--render_extrinsics_scale", type=float, default=1.1,
                        help='Scale render extrinsics (if loaded from file)')
    parser.add_argument("--reverse_render_extrinsics", action='store_true',
                        help='Reverse order of render extrinsics')
    parser.add_argument("--val_split_file", type=str, default="",
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # Blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--static_blend_weight", action='store_true',
                        help='use static blending weight')

    ## Logging / saving options

    parser.add_argument("--print_extras",   action='store_true',
                        help='Print extra variables')
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_testset", type=int, default=1000,
                        help='frequency of testset saving')
    parser.add_argument("--i_save", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser

